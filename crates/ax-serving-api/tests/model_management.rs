//! Integration tests for model management REST endpoints and auth middleware.
//!
//! Uses a `NullBackend` (no real model file required for validation path tests)
//! and exercises the Axum router built by `ax_serving_api::rest::router`.

use std::path::Path;
use std::sync::Arc;

use ax_serving_api::{ServingLayer, config::ServeConfig, rest};
use ax_serving_engine::{
    GenerateEvent, GenerateInput, GenerationParams, GenerationStats, InferenceBackend, LoadConfig,
    ModelHandle, ModelMetadata, ThermalState,
};
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt; // for `oneshot`

/// Read the full response body as a UTF-8 string.
async fn body_text(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ── NullBackend ───────────────────────────────────────────────────────────────

/// Minimal backend that satisfies `InferenceBackend` without real inference.
/// `load_model` succeeds for any path (path validity is checked by the registry
/// before calling the backend). `unload_model` is a no-op.
struct NullBackend;

impl InferenceBackend for NullBackend {
    fn load_model(
        &self,
        _path: &Path,
        _config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        Ok((
            ModelHandle(1),
            ModelMetadata {
                architecture: "null".into(),
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                embedding_dim: 0,
                vocab_size: 0,
                context_length: 2048,
                load_time_ms: 1,
                peak_rss_bytes: 0,
            },
        ))
    }

    fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _handle: ModelHandle,
        _input: GenerateInput,
        _params: GenerationParams,
        _tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn tokenize(
        &self,
        _handle: ModelHandle,
        _text: &str,
        _add_bos: bool,
    ) -> anyhow::Result<Vec<u32>> {
        Ok(vec![])
    }

    fn decode_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> anyhow::Result<String> {
        Ok(String::new())
    }

    fn eos_tokens(&self, _handle: ModelHandle) -> anyhow::Result<Vec<u32>> {
        Ok(vec![2])
    }

    fn thermal_state(&self) -> ThermalState {
        ThermalState::Nominal
    }

    fn recommended_concurrency(&self) -> usize {
        4
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_layer() -> Arc<ServingLayer> {
    let backend: Arc<dyn InferenceBackend> = Arc::new(NullBackend);
    let mut config = ServeConfig::default();
    config.cache.enabled = false;
    Arc::new(ServingLayer::new(backend, config))
}

/// Build the Axum router with no API key requirement (auth disabled).
fn make_app_no_auth() -> axum::Router {
    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    rest::router(layer, keys)
}

/// Build the Axum router with a single required API key.
fn make_app_with_key(key: &str) -> axum::Router {
    let layer = make_layer();
    let mut keys = std::collections::HashSet::new();
    keys.insert(key.to_string());
    rest::router(layer, Arc::new(keys))
}

// ── Auth middleware tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn auth_health_unauthenticated_200() {
    let app = make_app_with_key("secret");
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "/health should not require auth"
    );
}

#[tokio::test]
async fn auth_models_without_key_401() {
    let app = make_app_with_key("secret");
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // RFC 7235 §4.1 requires WWW-Authenticate in 401 responses.
    assert_eq!(
        resp.headers().get("www-authenticate").map(|v| v.as_bytes()),
        Some(b"Bearer".as_ref()),
    );
}

#[tokio::test]
async fn auth_models_wrong_key_401() {
    let app = make_app_with_key("secret");
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .header("Authorization", "Bearer wrongkey")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_models_correct_key_200() {
    let app = make_app_with_key("secret");
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .header("Authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_metrics_unauthenticated_200() {
    let app = make_app_with_key("secret");
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "/metrics should not require auth"
    );
}

// ── Model management REST tests ───────────────────────────────────────────────

#[tokio::test]
async fn unload_not_loaded_404() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/v1/models/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn reload_not_loaded_404() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models/nonexistent/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn load_path_not_found_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model_id": "test",
        "path": "/nonexistent/path/model.gguf",
    })
    .to_string();

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn load_non_gguf_422() {
    // Create a real temp file with wrong extension so "file not found" check passes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.bin");
    std::fs::write(&path, b"dummy").unwrap();

    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model_id": "test",
        "path": path.to_string_lossy(),
    })
    .to_string();

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn load_model_201_and_duplicate_409() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let app = rest::router(Arc::clone(&layer), keys);

    let body = serde_json::json!({
        "model_id": "mymodel",
        "path": path.to_string_lossy(),
    })
    .to_string();

    // First load — should succeed with 201.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Second load of same model_id — should return 409 Conflict.
    let resp2 = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn load_then_unload_200() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let load_body = serde_json::json!({
        "model_id": "unload-test",
        "path": path.to_string_lossy(),
    })
    .to_string();

    let app = rest::router(Arc::clone(&layer), Arc::clone(&keys));
    let load_resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(load_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(load_resp.status(), StatusCode::CREATED);

    // Now unload it.
    let app2 = rest::router(Arc::clone(&layer), Arc::clone(&keys));
    let unload_resp = app2
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/v1/models/unload-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unload_resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn load_then_reload_200() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let load_body = serde_json::json!({
        "model_id": "reload-test",
        "path": path.to_string_lossy(),
    })
    .to_string();

    let app = rest::router(Arc::clone(&layer), Arc::clone(&keys));
    let load_resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(load_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(load_resp.status(), StatusCode::CREATED);

    // Now reload it.
    let app2 = rest::router(Arc::clone(&layer), Arc::clone(&keys));
    let reload_resp = app2
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models/reload-test/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reload_resp.status(), StatusCode::OK);

    // Model should still be registered after reload.
    assert!(layer.registry.get("reload-test").is_some());
}

// ── Capacity test ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn load_at_capacity_503() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = dir.path().join("model1.gguf");
    let path2 = dir.path().join("model2.gguf");
    std::fs::write(&path1, b"dummy").unwrap();
    std::fs::write(&path2, b"dummy").unwrap();

    // Layer capped at 1 loaded model.
    let backend: Arc<dyn InferenceBackend> = Arc::new(NullBackend);
    let mut config = ServeConfig::default();
    config.cache.enabled = false;
    config.registry.max_loaded_models = 1;
    let layer = Arc::new(ServingLayer::new(backend, config));
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let body1 =
        serde_json::json!({"model_id": "first", "path": path1.to_string_lossy()}).to_string();
    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body1))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CREATED,
        "first load should succeed"
    );

    // Second load should hit the capacity cap and return 503.
    let body2 =
        serde_json::json!({"model_id": "second", "path": path2.to_string_lossy()}).to_string();
    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body2))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "second load beyond cap should 503"
    );
}

// ── Burn-rate tests ───────────────────────────────────────────────────────────

/// With no requests the burn-rate gauges must be 0 (not NaN/inf).
#[tokio::test]
async fn burn_rate_no_requests_is_zero() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    assert!(
        text.contains("axs_slo_burn_rate_1h 0"),
        "burn_rate_1h should be 0 with no requests; got:\n{text}"
    );
    assert!(
        text.contains("axs_slo_burn_rate_alert 0"),
        "burn_rate_alert should be 0 with no requests; got:\n{text}"
    );
}

/// With 100% error rate, burn_rate = 1.0/0.001 = 1000 >> 14.4 → alert fires.
#[test]
fn burn_rate_all_errors_exceeds_threshold() {
    use ax_serving_api::metrics::BurnRateWindow;
    let mut w = BurnRateWindow::new(3_600_000);
    for _ in 0..10 {
        w.record(true);
    }
    assert!(
        w.burn_rate(0.001) > 14.4,
        "100% errors must trigger fast burn alert (burn_rate={:.1})",
        w.burn_rate(0.001)
    );
}

// ── SLO gauge tests ───────────────────────────────────────────────────────────

/// With no traffic the SLO pass gauges must all be 0, not 1.
///
/// If they were 1, any fresh deployment would look "passing" even before
/// it has served a single request — a false-positive that would suppress alerts.
#[tokio::test]
async fn slo_gauges_no_data_are_zero() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = body_text(resp).await;
    assert!(
        text.contains("axs_slo_e2e_p99_pass 0"),
        "e2e_pass should be 0 with no data; got:\n{text}"
    );
    assert!(
        text.contains("axs_slo_queue_p99_pass 0"),
        "queue_pass should be 0 with no data; got:\n{text}"
    );
    assert!(
        text.contains("axs_slo_error_rate_pass 0"),
        "error_rate_pass should be 0 with no data; got:\n{text}"
    );
}

// ── EchoBackend ───────────────────────────────────────────────────────────────

/// Backend that sends one token then Done — used to exercise the full inference path.
struct EchoBackend;

impl InferenceBackend for EchoBackend {
    fn load_model(
        &self,
        _path: &Path,
        _config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        Ok((
            ModelHandle(2),
            ModelMetadata {
                architecture: "echo".into(),
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                embedding_dim: 0,
                vocab_size: 0,
                context_length: 2048,
                load_time_ms: 1,
                peak_rss_bytes: 0,
            },
        ))
    }

    fn unload_model(&self, _handle: ModelHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _handle: ModelHandle,
        _input: GenerateInput,
        _params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> anyhow::Result<()> {
        tokio::spawn(async move {
            let _ = tx.send(GenerateEvent::Token("hello".to_string())).await;
            let _ = tx
                .send(GenerateEvent::Done(GenerationStats {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    prefill_tok_per_sec: 0.0,
                    decode_tok_per_sec: 0.0,
                    stop_reason: "stop".to_string(),
                }))
                .await;
        });
        Ok(())
    }

    fn tokenize(
        &self,
        _handle: ModelHandle,
        _text: &str,
        _add_bos: bool,
    ) -> anyhow::Result<Vec<u32>> {
        Ok(vec![])
    }

    fn decode_tokens(&self, _handle: ModelHandle, _tokens: &[u32]) -> anyhow::Result<String> {
        Ok(String::new())
    }

    fn eos_tokens(&self, _handle: ModelHandle) -> anyhow::Result<Vec<u32>> {
        Ok(vec![2])
    }

    fn thermal_state(&self) -> ThermalState {
        ThermalState::Nominal
    }

    fn recommended_concurrency(&self) -> usize {
        4
    }
}

fn make_echo_layer() -> Arc<ServingLayer> {
    let backend: Arc<dyn InferenceBackend> = Arc::new(EchoBackend);
    let mut config = ServeConfig::default();
    config.cache.enabled = false;
    Arc::new(ServingLayer::new(backend, config))
}

// ── Chat completions input-validation tests ───────────────────────────────────

#[tokio::test]
async fn chat_completions_model_id_too_long_400() {
    let app = make_app_no_auth();
    let long_id = "a".repeat(257);
    let body = serde_json::json!({
        "model": long_id,
        "messages": [{"role": "user", "content": "hi"}]
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_empty_messages_400() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "any",
        "messages": []
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_too_many_messages_400() {
    let app = make_app_no_auth();
    // 101 messages exceeds the MAX_MESSAGES=100 limit.
    let messages: Vec<serde_json::Value> = (0..101)
        .map(|_| serde_json::json!({"role": "user", "content": "hi"}))
        .collect();
    let body = serde_json::json!({
        "model": "any",
        "messages": messages
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_content_too_large_400() {
    let app = make_app_no_auth();
    // 33 KB exceeds the MAX_CONTENT_BYTES=32 KB limit.
    let big_content = "x".repeat(33 * 1024);
    let body = serde_json::json!({
        "model": "any",
        "messages": [{"role": "user", "content": big_content}]
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_max_tokens_exceeded_400() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "any",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32769
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_invalid_temperature_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "any",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": -0.1
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn chat_completions_invalid_top_k_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "any",
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 0
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn chat_completions_top_logprobs_without_logprobs_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "not-loaded",
        "messages": [{"role": "user", "content": "hello"}],
        "top_logprobs": 1
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn chat_completions_invalid_response_format_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "not-loaded",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "xml"}
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn chat_completions_model_not_found_404() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "not-loaded",
        "messages": [{"role": "user", "content": "hello"}]
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── Text completions input-validation tests ───────────────────────────────────

#[tokio::test]
async fn text_completions_empty_prompt_400() {
    let app = make_app_no_auth();
    let body = serde_json::json!({"model": "any", "prompt": ""}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn text_completions_prompt_too_large_400() {
    let app = make_app_no_auth();
    let big_prompt = "x".repeat(33 * 1024);
    let body = serde_json::json!({"model": "any", "prompt": big_prompt}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn text_completions_max_tokens_exceeded_400() {
    let app = make_app_no_auth();
    let body =
        serde_json::json!({"model": "any", "prompt": "hi", "max_tokens": 32769}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn text_completions_top_logprobs_without_logprobs_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "not-loaded",
        "prompt": "hello",
        "top_logprobs": 1
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn text_completions_invalid_response_format_422() {
    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model": "not-loaded",
        "prompt": "hello",
        "response_format": {"type": "xml"}
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn text_completions_model_not_found_404() {
    let app = make_app_no_auth();
    let body = serde_json::json!({"model": "not-loaded", "prompt": "hello"}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── Embeddings input-validation tests ────────────────────────────────────────

#[tokio::test]
async fn embeddings_model_id_too_long_400() {
    let app = make_app_no_auth();
    let long_id = "a".repeat(257);
    let body =
        serde_json::json!({"model": long_id, "input": "hello"}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/embeddings")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn embeddings_model_not_found_404() {
    let app = make_app_no_auth();
    let body = serde_json::json!({"model": "not-loaded", "input": "hello"}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/embeddings")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn embeddings_not_implemented_501() {
    // NullBackend.embed() returns the default error "not supported" → 501.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body =
        serde_json::json!({"model_id": "embed-test", "path": path.to_string_lossy()}).to_string();
    rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(load_body))
                .unwrap(),
        )
        .await
        .unwrap();

    let embed_body =
        serde_json::json!({"model": "embed-test", "input": "hello world"}).to_string();
    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/embeddings")
                .header("Content-Type", "application/json")
                .body(Body::from(embed_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

// ── Basic endpoint format tests ───────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok_status() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["status"], "ok", "health status must be 'ok'; got: {json}");
    assert!(json["thermal"].is_string(), "health must have thermal field");
    assert!(json["loaded_models"].is_array(), "health must have loaded_models array");
}

#[tokio::test]
async fn list_models_returns_list_format() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());
    assert_eq!(json["data"].as_array().unwrap().len(), 0, "no models loaded initially");
}

#[tokio::test]
async fn list_models_shows_loaded_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body =
        serde_json::json!({"model_id": "mylist-model", "path": path.to_string_lossy()}).to_string();
    rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(load_body))
                .unwrap(),
        )
        .await
        .unwrap();

    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "mylist-model");
    assert_eq!(data[0]["object"], "model");
}

#[tokio::test]
async fn v1_metrics_returns_scheduler_key() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(json["scheduler"].is_object(), "v1/metrics must include 'scheduler' key");
    assert!(json["scheduler"]["max_inflight"].is_number());
    assert!(json["cache"].is_object(), "v1/metrics must include 'cache' key");
    assert!(json["uptime_secs"].is_number());
}

#[tokio::test]
async fn dashboard_returns_html_content() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/dashboard")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    assert!(text.contains("<!DOCTYPE html>") || text.contains("<html"), "dashboard must return HTML");
}

#[tokio::test]
async fn license_get_returns_json() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/license")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(json["edition"].is_string(), "license response must include 'edition'");
}

#[tokio::test]
async fn license_set_missing_key_400() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/license")
                .header("Content-Type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn license_set_empty_key_400() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/license")
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"key":""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── Security header tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn security_headers_present_on_responses() {
    let app = make_app_no_auth();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.headers()
            .get("x-content-type-options")
            .map(|v| v.as_bytes()),
        Some(b"nosniff".as_ref()),
        "X-Content-Type-Options: nosniff must be present"
    );
    assert_eq!(
        resp.headers()
            .get("x-frame-options")
            .map(|v| v.as_bytes()),
        Some(b"DENY".as_ref()),
        "X-Frame-Options: DENY must be present"
    );
    assert!(
        resp.headers().contains_key("x-request-id"),
        "X-Request-ID must be present"
    );
}

// ── Full inference path tests (EchoBackend) ───────────────────────────────────

#[tokio::test]
async fn chat_completions_inference_200() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_echo_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    // Load model first.
    let load_body =
        serde_json::json!({"model_id": "echo-chat", "path": path.to_string_lossy()}).to_string();
    rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(load_body))
                .unwrap(),
        )
        .await
        .unwrap();

    let infer_body = serde_json::json!({
        "model": "echo-chat",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16
    })
    .to_string();

    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(infer_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "echo-chat");
    let choices = json["choices"].as_array().unwrap();
    assert!(!choices.is_empty());
    assert_eq!(choices[0]["message"]["content"], "hello");
    assert!(json["usage"].is_object());
}

#[tokio::test]
async fn text_completions_inference_200() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_echo_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let load_body =
        serde_json::json!({"model_id": "echo-text", "path": path.to_string_lossy()}).to_string();
    rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(load_body))
                .unwrap(),
        )
        .await
        .unwrap();

    let infer_body =
        serde_json::json!({"model": "echo-text", "prompt": "hello", "max_tokens": 16}).to_string();

    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(infer_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["object"], "text_completion");
    let choices = json["choices"].as_array().unwrap();
    assert!(!choices.is_empty());
    assert_eq!(choices[0]["text"], "hello");
    assert!(json["usage"].is_object());
}

#[tokio::test]
async fn load_model_response_has_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let app = make_app_no_auth();
    let body =
        serde_json::json!({"model_id": "field-check", "path": path.to_string_lossy()}).to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["model_id"], "field-check");
    assert!(json["architecture"].is_string());
    assert!(json["context_length"].is_number());
    assert!(json["load_time_ms"].is_number());
}

#[tokio::test]
async fn invalid_pooling_type_422() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let app = make_app_no_auth();
    let body = serde_json::json!({
        "model_id": "pooling-test",
        "path": path.to_string_lossy(),
        "pooling_type": "invalid_value"
    })
    .to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn auth_bearer_token_with_whitespace_trimmed() {
    // Auth middleware must strip leading/trailing whitespace from the Bearer token.
    let app = make_app_with_key("mykey");
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .header("Authorization", "Bearer  mykey ")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "auth must accept token with surrounding whitespace"
    );
}
