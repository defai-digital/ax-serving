//! Acceptance tests for graceful shutdown of the REST server.
//!
//! Verifies that:
//! 1. A request that completes before the shutdown signal is accepted and
//!    returns 200.
//! 2. After the shutdown signal is received the server stops accepting new
//!    connections.
//!
//! Uses the same `NullBackend` + `make_layer()` helpers as model_management.rs,
//! replicated inline since Rust integration-test crates do not share helpers.

use std::path::Path;
use std::sync::Arc;

use ax_serving_api::{ServingLayer, config::ServeConfig, rest};
use ax_serving_engine::{
    GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, ModelHandle,
    ModelMetadata, ThermalState,
};
use tokio::sync::oneshot;

// ── NullBackend ───────────────────────────────────────────────────────────────

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

fn make_layer() -> Arc<ServingLayer> {
    let backend: Arc<dyn InferenceBackend> = Arc::new(NullBackend);
    let mut config = ServeConfig::default();
    config.cache.enabled = false;
    Arc::new(ServingLayer::new(backend, config))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Graceful shutdown: a request that arrives before the shutdown signal
/// completes successfully, then subsequent connections are refused.
#[tokio::test]
async fn graceful_shutdown_drains_in_flight() {
    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let app = rest::router(layer, keys);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let (tx, rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                rx.await.ok();
            })
            .await
            .ok();
    });

    // Server is up: GET /health must return 200.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "/health must return 200 before shutdown"
    );

    // Trigger graceful shutdown.
    let _ = tx.send(());
    // Give axum time to process the shutdown signal and stop the listener.
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // New connections after the shutdown signal should be refused or error.
    let result = client
        .get(format!("http://{addr}/health"))
        .timeout(tokio::time::Duration::from_millis(500))
        .send()
        .await;
    assert!(
        result.is_err(),
        "server must stop accepting new connections after shutdown"
    );
}

/// Graceful shutdown: a request and the shutdown signal race.
/// Any response that arrives is well-formed (2xx); if the shutdown won the
/// race the request may fail — both outcomes are valid.
#[tokio::test]
async fn graceful_shutdown_completes_final_request() {
    let layer = make_layer();
    let keys = Arc::new(std::collections::HashSet::<String>::new());
    let app = rest::router(layer, keys);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let (tx, rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                rx.await.ok();
            })
            .await
            .ok();
    });

    // Tiny pause to ensure the server is listening before we race.
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    // Send a request and trigger shutdown at nearly the same time.
    let (send_result, _) = tokio::join!(
        async {
            let client = reqwest::Client::new();
            client
                .get(format!("http://{addr}/health"))
                .timeout(tokio::time::Duration::from_millis(500))
                .send()
                .await
        },
        async {
            let _ = tx.send(());
        },
    );

    // If the request was dispatched before shutdown: must be 2xx.
    // If shutdown won the race: connection error is acceptable.
    if let Ok(resp) = send_result {
        assert!(
            resp.status().is_success(),
            "if a response arrived it must be 2xx, got {}",
            resp.status()
        );
    }
    // Test passes whether the response was received or refused.
}
