//! Integration tests for model management REST endpoints and auth middleware.
//!
//! Uses a `NullBackend` (no real model file required for validation path tests)
//! and exercises the Axum router built by `ax_serving_api::rest::router`.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

use ax_serving_api::{
    ServingLayer,
    config::{ProjectPolicyConfig, ProjectRuleConfig, ServeConfig},
    rest,
};
use ax_serving_engine::{
    EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput, GenerationParams,
    GenerationStats, InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, ThermalState,
};
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tokio::sync::Notify;
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
                resolved_backend: ax_serving_engine::BackendType::Auto,
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

struct CriticalBackend;

impl InferenceBackend for CriticalBackend {
    fn load_model(
        &self,
        _path: &Path,
        _config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        Ok((
            ModelHandle(2),
            ModelMetadata {
                architecture: "critical".into(),
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                embedding_dim: 0,
                vocab_size: 0,
                context_length: 2048,
                load_time_ms: 1,
                peak_rss_bytes: 0,
                resolved_backend: ax_serving_engine::BackendType::Auto,
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
        ThermalState::Critical
    }

    fn recommended_concurrency(&self) -> usize {
        1
    }
}

struct EmbeddingBackend;

impl InferenceBackend for EmbeddingBackend {
    fn load_model(
        &self,
        _path: &Path,
        _config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        Ok((
            ModelHandle(3),
            ModelMetadata {
                architecture: "embed".into(),
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                embedding_dim: 3,
                vocab_size: 0,
                context_length: 2048,
                load_time_ms: 1,
                peak_rss_bytes: 0,
                resolved_backend: ax_serving_engine::BackendType::Auto,
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

    fn embed(
        &self,
        _handle: ModelHandle,
        inputs: &EmbedInput<'_>,
        _config: &EmbedConfig,
    ) -> anyhow::Result<EmbedResult> {
        let (count, prompt_tokens) = match inputs {
            EmbedInput::Strings(texts) => (
                texts.len(),
                texts.iter().map(|s| s.len() as u32).sum::<u32>().max(1),
            ),
            EmbedInput::Tokens(seqs) => (
                seqs.len(),
                seqs.iter().map(|s| s.len() as u32).sum::<u32>().max(1),
            ),
        };

        Ok(EmbedResult {
            embeddings: (0..count)
                .map(|i| vec![i as f32, i as f32 + 0.5, i as f32 + 1.0])
                .collect(),
            prompt_tokens,
        })
    }
}

struct EmbeddingFailureBackend;

impl InferenceBackend for EmbeddingFailureBackend {
    fn load_model(
        &self,
        _path: &Path,
        _config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        Ok((
            ModelHandle(4),
            ModelMetadata {
                architecture: "embed-fail".into(),
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                embedding_dim: 3,
                vocab_size: 0,
                context_length: 2048,
                load_time_ms: 1,
                peak_rss_bytes: 0,
                resolved_backend: ax_serving_engine::BackendType::Auto,
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

    fn embed(
        &self,
        _handle: ModelHandle,
        _inputs: &EmbedInput<'_>,
        _config: &EmbedConfig,
    ) -> anyhow::Result<EmbedResult> {
        Err(anyhow::anyhow!("embedding backend failure"))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_layer() -> Arc<ServingLayer> {
    let backend: Arc<dyn InferenceBackend> = Arc::new(NullBackend);
    let mut config = ServeConfig::default();
    config.cache.enabled = false;
    Arc::new(ServingLayer::new(backend, config))
}

fn make_layer_with_backend(backend: Arc<dyn InferenceBackend>) -> Arc<ServingLayer> {
    let mut config = ServeConfig::default();
    config.cache.enabled = false;
    Arc::new(ServingLayer::new(backend, config))
}

fn make_layer_with_backend_and_config(
    backend: Arc<dyn InferenceBackend>,
    config: ServeConfig,
) -> Arc<ServingLayer> {
    Arc::new(ServingLayer::new(backend, config))
}

/// Build the Axum router with no API key requirement (auth disabled).
fn make_app_no_auth() -> axum::Router {
    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    rest::router(layer, keys)
}

fn make_app_with_backend_no_auth(backend: Arc<dyn InferenceBackend>) -> axum::Router {
    let layer = make_layer_with_backend(backend);
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    rest::router(layer, keys)
}

fn make_embedding_layer() -> Arc<ServingLayer> {
    make_layer_with_backend(Arc::new(EmbeddingBackend))
}

fn make_embedding_failure_layer() -> Arc<ServingLayer> {
    make_layer_with_backend(Arc::new(EmbeddingFailureBackend))
}

fn sample_project_policy(default_project: Option<&str>) -> ProjectPolicyConfig {
    ProjectPolicyConfig {
        enabled: true,
        default_project: default_project.map(str::to_string),
        rules: vec![
            ProjectRuleConfig {
                project: "fabric".into(),
                allowed_models: vec!["echo-*".into(), "embed-*".into()],
                max_tokens_limit: Some(64),
                worker_pool: Some("fabric".into()),
            },
            ProjectRuleConfig {
                project: "ops".into(),
                allowed_models: vec!["*".into()],
                max_tokens_limit: None,
                worker_pool: None,
            },
        ],
    }
}

struct BlockingEchoBackend {
    started: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    release: Arc<Notify>,
}

impl InferenceBackend for BlockingEchoBackend {
    fn load_model(
        &self,
        _path: &Path,
        _config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        Ok((
            ModelHandle(5),
            ModelMetadata {
                architecture: "blocking-echo".into(),
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                embedding_dim: 0,
                vocab_size: 0,
                context_length: 2048,
                load_time_ms: 1,
                peak_rss_bytes: 0,
                resolved_backend: ax_serving_engine::BackendType::Auto,
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
        let started = Arc::clone(&self.started);
        let released = Arc::clone(&self.released);
        let release = Arc::clone(&self.release);
        tokio::spawn(async move {
            started.fetch_add(1, Ordering::Relaxed);
            while !released.load(Ordering::Relaxed) {
                release.notified().await;
            }
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

fn make_blocking_cache_layer(
    started: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    release: Arc<Notify>,
) -> Arc<ServingLayer> {
    let backend: Arc<dyn InferenceBackend> = Arc::new(BlockingEchoBackend {
        started,
        released,
        release,
    });
    let mut config = ServeConfig::default();
    config.cache.enabled = true;
    Arc::new(ServingLayer::new(backend, config))
}

/// Build the Axum router with a single required API key.
fn make_app_with_key(key: &str) -> axum::Router {
    let layer = make_layer();
    let mut keys = std::collections::HashSet::new();
    keys.insert(key.to_string());
    rest::router(layer, Arc::new(keys))
}

fn make_app_with_key_and_layer(key: &str) -> (Arc<ServingLayer>, axum::Router) {
    let layer = make_layer();
    layer.set_public_auth_required(true);
    let mut keys = std::collections::HashSet::new();
    keys.insert(key.to_string());
    let app = rest::router(Arc::clone(&layer), Arc::new(keys));
    (layer, app)
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
async fn admin_policy_requires_auth_and_returns_summary() {
    let backend: Arc<dyn InferenceBackend> = Arc::new(NullBackend);
    let config = ServeConfig {
        project_policy: sample_project_policy(Some("fabric")),
        ..ServeConfig::default()
    };
    let layer = make_layer_with_backend_and_config(backend, config);
    layer.set_public_auth_required(true);
    let mut keys = std::collections::HashSet::new();
    keys.insert("secret".to_string());
    let app = rest::router(Arc::clone(&layer), Arc::new(keys));

    let unauth = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/policy")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/policy")
                .header("Authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["enabled"], true);
    assert_eq!(json["header"], "x-ax-project");
    assert_eq!(json["default_project"], "fabric");
    assert_eq!(json["rules"][0]["project"], "fabric");
}

#[tokio::test]
async fn chat_completions_project_policy_requires_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("echo-chat.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let backend: Arc<dyn InferenceBackend> = Arc::new(EchoBackend);
    let config = ServeConfig {
        project_policy: sample_project_policy(None),
        ..ServeConfig::default()
    };
    let layer = make_layer_with_backend_and_config(backend, config);
    let keys = Arc::new(std::collections::HashSet::<String>::new());

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
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn text_completions_project_policy_enforces_max_tokens_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("echo-text.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let backend: Arc<dyn InferenceBackend> = Arc::new(EchoBackend);
    let config = ServeConfig {
        project_policy: sample_project_policy(None),
        ..ServeConfig::default()
    };
    let layer = make_layer_with_backend_and_config(backend, config);
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
        serde_json::json!({"model": "echo-text", "prompt": "hello", "max_tokens": 128}).to_string();

    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("Content-Type", "application/json")
                .header("x-ax-project", "fabric")
                .body(Body::from(infer_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn embeddings_project_policy_requires_header_before_model_lookup() {
    let backend: Arc<dyn InferenceBackend> = Arc::new(EchoBackend);
    let config = ServeConfig {
        project_policy: sample_project_policy(None),
        ..ServeConfig::default()
    };
    let layer = make_layer_with_backend_and_config(backend, config);
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let body = serde_json::json!({
        "model": "embed-missing",
        "input": "hello"
    })
    .to_string();

    let resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
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
    let text = body_text(unload_resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["model_id"], "unload-test");
    assert_eq!(json["state"], "unloaded");
    assert_eq!(json["ready"], true);
    assert_eq!(json["model_available"], false);
    assert_eq!(json["loaded_model_count"], 0);
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
    let text = body_text(reload_resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["model_id"], "reload-test");
    assert_eq!(json["state"], "loaded");
    assert_eq!(json["ready"], true);
    assert_eq!(json["model_available"], true);
    assert_eq!(json["loaded_model_count"], 1);
    assert!(json["architecture"].is_string());
    assert!(json["load_time_ms"].is_number());

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
    assert!(
        text.contains("axs_scheduler_prefill_tokens_active "),
        "prometheus metrics should expose prefill_tokens_active; got:\n{text}"
    );
    assert!(
        text.contains("axs_scheduler_decode_sequences_active "),
        "prometheus metrics should expose decode_sequences_active; got:\n{text}"
    );
    assert!(
        text.contains("axs_cache_follower_waiting "),
        "prometheus metrics should expose cache_follower_waiting; got:\n{text}"
    );
    assert!(
        text.contains("axs_ttft_p50_us "),
        "prometheus metrics should expose ttft_p50_us; got:\n{text}"
    );
    assert!(
        text.contains("axs_ttft_p95_us "),
        "prometheus metrics should expose ttft_p95_us; got:\n{text}"
    );
    assert!(
        text.contains("axs_ttft_p99_us "),
        "prometheus metrics should expose ttft_p99_us; got:\n{text}"
    );
    assert!(
        text.contains("axs_request_class_cold_requests_total "),
        "prometheus metrics should expose cold request classification; got:\n{text}"
    );
    assert!(
        text.contains("axs_request_class_exact_cache_hits_total "),
        "prometheus metrics should expose exact cache-hit classification; got:\n{text}"
    );
    assert!(
        text.contains("axs_request_class_cache_follower_hits_total "),
        "prometheus metrics should expose follower cache-hit classification; got:\n{text}"
    );
    assert!(
        text.contains("axs_request_class_cache_fills_total "),
        "prometheus metrics should expose cache fill classification; got:\n{text}"
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
                resolved_backend: ax_serving_engine::BackendType::Auto,
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
    let body = serde_json::json!({"model": "any", "prompt": "hi", "max_tokens": 32769}).to_string();
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
    let body = serde_json::json!({"model": long_id, "input": "hello"}).to_string();
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

    let embed_body = serde_json::json!({"model": "embed-test", "input": "hello world"}).to_string();
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

#[tokio::test]
async fn embeddings_success_200() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("embed-model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_embedding_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body =
        serde_json::json!({"model_id": "embed-ok", "path": path.to_string_lossy()}).to_string();
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

    let embed_body = serde_json::json!({
        "model": "embed-ok",
        "input": ["hello", "world"],
        "encoding_format": "float"
    })
    .to_string();
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

    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["object"], "list");
    assert_eq!(json["model"], "embed-ok");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["object"], "embedding");
    assert!(data[0]["embedding"].is_array());
    assert_eq!(json["usage"]["prompt_tokens"], 10);
    assert_eq!(json["usage"]["total_tokens"], 10);
}

#[tokio::test]
async fn embeddings_backend_failure_500() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("embed-fail.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_embedding_failure_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body = serde_json::json!({
        "model_id": "embed-fail",
        "path": path.to_string_lossy()
    })
    .to_string();
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

    let embed_body = serde_json::json!({"model": "embed-fail", "input": "hello world"}).to_string();
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

    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(
        json["error"]
            .as_str()
            .unwrap_or_default()
            .contains("embedding backend failure"),
        "backend failure should surface in the error envelope; got: {json}"
    );
}

// ── Basic endpoint format tests ───────────────────────────────────────────────

#[tokio::test]
async fn health_without_models_is_degraded_but_ready() {
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
    assert_eq!(
        json["status"], "degraded",
        "health status must be 'degraded' with no models; got: {json}"
    );
    assert_eq!(
        json["ready"], true,
        "runtime should still be ready without models"
    );
    assert_eq!(
        json["model_available"], false,
        "health must report no models available"
    );
    assert_eq!(
        json["reason"], "no_models_loaded",
        "health should explain degraded no-model state"
    );
    assert!(
        json["thermal"].is_string(),
        "health must have thermal field"
    );
    assert!(
        json["loaded_models"].is_array(),
        "health must have loaded_models array"
    );
    assert_eq!(
        json["loaded_model_count"], 0,
        "health must include loaded model count"
    );
}

#[tokio::test]
async fn health_with_loaded_model_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("health-model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body = serde_json::json!({
        "model_id": "health-model",
        "path": path,
        "backend": "llama_cpp"
    });

    let app = rest::router(layer.clone(), keys.clone());
    let load_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(load_resp.status(), StatusCode::CREATED);

    let health_resp = rest::router(layer, keys)
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health_resp.status(), StatusCode::OK);
    let text = body_text(health_resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        json["status"], "ok",
        "health should be ok with a loaded model; got: {json}"
    );
    assert_eq!(json["ready"], true);
    assert_eq!(json["model_available"], true);
    assert_eq!(json["loaded_model_count"], 1);
}

#[tokio::test]
async fn startup_sequence_moves_from_degraded_to_ok_after_model_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("startup-sequence.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let app = rest::router(layer.clone(), keys.clone());

    let initial_health = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(initial_health.status(), StatusCode::OK);
    let initial_text = body_text(initial_health).await;
    let initial_json: serde_json::Value = serde_json::from_str(&initial_text).unwrap();
    assert_eq!(initial_json["status"], "degraded");
    assert_eq!(initial_json["ready"], true);
    assert_eq!(initial_json["model_available"], false);
    assert_eq!(initial_json["reason"], "no_models_loaded");

    let load_body = serde_json::json!({
        "model_id": "startup-sequence",
        "path": path,
        "backend": "llama_cpp"
    });
    let load_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(load_resp.status(), StatusCode::CREATED);

    let final_health = rest::router(layer, keys)
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(final_health.status(), StatusCode::OK);
    let final_text = body_text(final_health).await;
    let final_json: serde_json::Value = serde_json::from_str(&final_text).unwrap();
    assert_eq!(final_json["status"], "ok");
    assert_eq!(final_json["ready"], true);
    assert_eq!(final_json["model_available"], true);
    assert_eq!(final_json["loaded_model_count"], 1);
}

#[tokio::test]
async fn failed_model_load_leaves_health_degraded_without_models() {
    let app = make_app_no_auth();
    let load_body = serde_json::json!({
        "model_id": "missing-model",
        "path": "/definitely/missing/model.gguf"
    })
    .to_string();

    let load_resp = app
        .clone()
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
    assert_eq!(load_resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let health_resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health_resp.status(), StatusCode::OK);
    let text = body_text(health_resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["status"], "degraded");
    assert_eq!(json["ready"], true);
    assert_eq!(json["model_available"], false);
    assert_eq!(json["loaded_model_count"], 0);
    assert_eq!(json["reason"], "no_models_loaded");
}

#[tokio::test]
async fn health_after_unload_returns_to_degraded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("health-unload.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body = serde_json::json!({
        "model_id": "health-unload",
        "path": path,
        "backend": "llama_cpp"
    });

    let app = rest::router(layer.clone(), keys.clone());
    let load_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/models")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(load_resp.status(), StatusCode::CREATED);

    let unload_resp = rest::router(layer.clone(), keys.clone())
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/v1/models/health-unload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unload_resp.status(), StatusCode::OK);

    let health_resp = rest::router(layer, keys)
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health_resp.status(), StatusCode::OK);
    let text = body_text(health_resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["status"], "degraded");
    assert_eq!(json["ready"], true);
    assert_eq!(json["model_available"], false);
    assert_eq!(json["reason"], "no_models_loaded");
    assert_eq!(json["loaded_model_count"], 0);
}

#[tokio::test]
async fn health_critical_thermal_is_not_ready() {
    let app = make_app_with_backend_no_auth(Arc::new(CriticalBackend));
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
    assert_eq!(json["status"], "degraded");
    assert_eq!(json["ready"], false);
    assert_eq!(json["model_available"], false);
    assert_eq!(json["reason"], "thermal_critical_no_models");
    assert_eq!(json["thermal"], "Critical");
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
    assert_eq!(
        json["data"].as_array().unwrap().len(),
        0,
        "no models loaded initially"
    );
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
    assert!(
        json["scheduler"].is_object(),
        "v1/metrics must include 'scheduler' key"
    );
    assert!(json["scheduler"]["max_inflight"].is_number());
    assert!(json["scheduler"]["cache_follower_waiting"].is_number());
    assert!(json["scheduler"]["prefill_tokens_active"].is_number());
    assert!(json["scheduler"]["decode_sequences_active"].is_number());
    assert!(json["scheduler"]["split_scheduler_enabled"].is_boolean());
    assert!(json["scheduler"]["ttft_p50_us"].is_number());
    assert!(json["scheduler"]["ttft_p95_us"].is_number());
    assert!(json["scheduler"]["ttft_p99_us"].is_number());
    assert!(
        json["cache"].is_object(),
        "v1/metrics must include 'cache' key"
    );
    assert!(
        json["request_classes"].is_object(),
        "v1/metrics must include 'request_classes' key"
    );
    assert!(json["request_classes"]["cold_requests_total"].is_number());
    assert!(json["request_classes"]["exact_cache_hits_total"].is_number());
    assert!(json["request_classes"]["cache_follower_hits_total"].is_number());
    assert!(json["request_classes"]["cache_fills_total"].is_number());
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
    assert!(
        text.contains("<!DOCTYPE html>") || text.contains("<html"),
        "dashboard must return HTML"
    );
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
    assert!(
        json["edition"].is_string(),
        "license response must include 'edition'"
    );
}

#[tokio::test]
async fn admin_startup_report_requires_auth_and_returns_runtime_summary() {
    let (_layer, app) = make_app_with_key_and_layer("secret");

    let unauth = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/startup-report")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/startup-report")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["service"], "serving");
    assert_eq!(json["auth_required"], true);
    assert!(json["runtime"]["rest_addr"].is_string());
    assert!(json["license"]["edition"].is_string());
    assert_eq!(json["scheduler"]["scheduler_managed_batching"], false);
    assert_eq!(json["scheduler"]["batch_hints_advisory_only"], true);
    assert!(json["scheduler"]["max_batch_size_hint"].is_u64());
    assert!(json["scheduler"]["batch_window_ms_hint"].is_u64());
    assert!(json["cache"]["enabled"].is_boolean());
    assert_eq!(json["cache"]["kv_prefix_cache"], false);
}

#[tokio::test]
async fn admin_status_requires_auth_and_returns_operational_summary() {
    let (_layer, app) = make_app_with_key_and_layer("secret");

    let unauth = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/status")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-request-id", "req-admin-status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["request_id"], "req-admin-status");
    assert_eq!(json["service"], "serving");
    assert_eq!(json["auth_required"], true);
    assert!(json["models"]["loaded_model_count"].is_u64());
    assert!(json["scheduler"]["queue_depth"].is_i64());
    assert!(json["system"]["uptime_secs"].is_u64());
}

#[tokio::test]
async fn admin_diagnostics_and_audit_capture_license_change() {
    let (_layer, app) = make_app_with_key_and_layer("secret");

    let set_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/license")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-request-id", "req-license-audit")
                .body(Body::from(r#"{"key":"business-key"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(set_resp.status(), StatusCode::OK);

    let diag_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-request-id", "req-diag-1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(diag_resp.status(), StatusCode::OK);
    let diag_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(diag_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(diag_json["request_id"], "req-diag-1");
    assert_eq!(diag_json["startup_report"]["service"], "serving");
    assert!(
        diag_json["audit_tail"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["action"] == "license_set" && e["outcome"] == "ok")
    );

    let audit_resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/audit?limit=10")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(audit_resp.status(), StatusCode::OK);
    let audit_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(audit_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        audit_json["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["action"] == "license_set" && e["actor"] == "request:req-license-audit")
    );
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

#[tokio::test]
async fn license_set_validation_errors_are_audited() {
    let (_layer, app) = make_app_with_key_and_layer("secret");

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/license")
                .header("Authorization", "Bearer secret")
                .header("Content-Type", "application/json")
                .header("X-Request-ID", "req-license-invalid")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let audit = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/admin/audit?limit=10")
                .header("Authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(audit.status(), StatusCode::OK);
    let audit_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(audit.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(audit_json["events"].as_array().unwrap().iter().any(|e| {
        e["action"] == "license_set"
            && e["actor"] == "request:req-license-invalid"
            && e["outcome"] == "error"
            && e["detail"]["error"] == "missing field: key"
    }));
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
        resp.headers().get("x-frame-options").map(|v| v.as_bytes()),
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
async fn cache_follower_wait_does_not_consume_extra_scheduler_permit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache-follow.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let started = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let release = Arc::new(Notify::new());
    let layer = make_blocking_cache_layer(
        Arc::clone(&started),
        Arc::clone(&released),
        Arc::clone(&release),
    );
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let load_body =
        serde_json::json!({"model_id": "cache-follow", "path": path.to_string_lossy()}).to_string();
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

    let request_body = serde_json::json!({
        "model": "cache-follow",
        "messages": [{"role": "user", "content": "same request"}],
        "cache": "enable",
        "stream": false,
        "max_tokens": 8
    })
    .to_string();

    let app = rest::router(Arc::clone(&layer), Arc::clone(&keys));
    let req1 = {
        let app = app.clone();
        let body = request_body.clone();
        tokio::spawn(async move {
            app.oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        })
    };

    while started.load(Ordering::Relaxed) < 1 {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    let req2 = {
        let app = app.clone();
        let body = request_body.clone();
        tokio::spawn(async move {
            app.oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        })
    };

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    assert_eq!(
        layer
            .scheduler
            .metrics
            .inflight_count
            .load(Ordering::Relaxed),
        1,
        "follower wait must not consume a second scheduler permit"
    );
    assert_eq!(
        layer
            .scheduler
            .metrics
            .cache_follower_waiting
            .load(Ordering::Relaxed),
        1,
        "follower should be accounted as waiting pre-permit"
    );

    released.store(true, Ordering::Relaxed);
    release.notify_waiters();
    let resp1 = req1.await.unwrap();
    let resp2 = req2.await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);
    assert_eq!(resp2.status(), StatusCode::OK);
}

#[tokio::test]
async fn cold_request_updates_request_class_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cold-request.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_echo_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());

    let load_body = serde_json::json!({
        "model_id": "cold-request",
        "path": path.to_string_lossy()
    })
    .to_string();
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

    let request_body = serde_json::json!({
        "model": "cold-request",
        "messages": [{"role": "user", "content": "repeat me"}],
        "cache": "enable",
        "stream": false,
        "max_tokens": 8
    })
    .to_string();

    let app = rest::router(Arc::clone(&layer), Arc::clone(&keys));
    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(request_body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);

    let metrics_resp = rest::router(Arc::clone(&layer), Arc::clone(&keys))
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metrics_resp.status(), StatusCode::OK);
    let metrics_text = body_text(metrics_resp).await;
    let metrics_json: serde_json::Value = serde_json::from_str(&metrics_text).unwrap();
    assert_eq!(metrics_json["request_classes"]["cold_requests_total"], 1);
    assert_eq!(metrics_json["request_classes"]["exact_cache_hits_total"], 0);
    assert_eq!(
        metrics_json["request_classes"]["cache_follower_hits_total"],
        0
    );
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
    assert_eq!(json["state"], "loaded");
    assert_eq!(json["ready"], true);
    assert_eq!(json["model_available"], true);
    assert_eq!(json["loaded_model_count"], 1);
    assert!(json["architecture"].is_string());
    assert!(json["context_length"].is_number());
    assert!(json["load_time_ms"].is_number());
}

#[tokio::test]
async fn unload_model_response_has_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body =
        serde_json::json!({"model_id": "unload-fields", "path": path.to_string_lossy()})
            .to_string();

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
                .method(Method::DELETE)
                .uri("/v1/models/unload-fields")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["model_id"], "unload-fields");
    assert_eq!(json["state"], "unloaded");
    assert!(json["ready"].is_boolean());
    assert!(json["model_available"].is_boolean());
    assert!(json["loaded_model_count"].is_number());
}

#[tokio::test]
async fn reload_model_response_has_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"dummy").unwrap();

    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let load_body =
        serde_json::json!({"model_id": "reload-fields", "path": path.to_string_lossy()})
            .to_string();

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
                .method(Method::POST)
                .uri("/v1/models/reload-fields/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_text(resp).await;
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["model_id"], "reload-fields");
    assert_eq!(json["state"], "loaded");
    assert!(json["ready"].is_boolean());
    assert!(json["model_available"].is_boolean());
    assert!(json["loaded_model_count"].is_number());
    assert!(json["architecture"].is_string());
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
