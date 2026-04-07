//! REST server: Axum-based OpenAI-compatible HTTP API.
//!
//! Endpoints:
//!   POST   /v1/chat/completions      — streaming (SSE) + blocking
//!   POST   /v1/completions          — text completions (streaming + blocking)
//!   GET    /v1/models                — list loaded models
//!   POST   /v1/models                — load a model from a GGUF file
//!   DELETE /v1/models/:id            — unload a loaded model
//!   POST   /v1/models/:id/reload     — atomically reload from same path/config
//!   GET    /health                   — liveness + readiness

pub mod admin;
pub mod inference;
pub mod license;
pub mod models;
pub mod routes;
pub mod schema;
pub mod validation;

use std::collections::HashSet;
use std::future::pending;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::{
    Router,
    http::StatusCode,
    middleware,
    routing::{delete, get, post},
};
use tower_http::timeout::TimeoutLayer;

use crate::ServingLayer;
use crate::auth;

/// Start the Axum REST server with graceful shutdown on SIGINT/SIGTERM.
pub async fn serve(
    layer: Arc<ServingLayer>,
    addr: String,
    keys: Arc<HashSet<String>>,
) -> Result<()> {
    let app = router(layer, keys);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("REST server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Build the Axum router.
pub fn router(layer: Arc<ServingLayer>, keys: Arc<HashSet<String>>) -> Router {
    let timeout_secs = std::env::var("AXS_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120);

    Router::new()
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::text_completions))
        .route("/v1/embeddings", post(routes::embeddings))
        .route(
            "/v1/models",
            get(routes::list_models).post(routes::rest_load_model),
        )
        .route("/v1/models/{id}", delete(routes::rest_unload_model))
        .route("/v1/models/{id}/reload", post(routes::rest_reload_model))
        .route("/health", get(routes::health))
        .route("/v1/metrics", get(routes::metrics))
        .route("/v1/admin/status", get(routes::admin_status))
        .route(
            "/v1/admin/startup-report",
            get(routes::admin_startup_report),
        )
        .route("/v1/admin/diagnostics", get(routes::admin_diagnostics))
        .route("/v1/admin/audit", get(routes::admin_audit))
        .route("/v1/admin/policy", get(routes::admin_policy))
        .route("/metrics", get(routes::prometheus_metrics))
        .route("/dashboard", get(routes::dashboard))
        .route(
            "/v1/license",
            get(routes::get_license).post(routes::set_license),
        )
        .route_layer(middleware::from_fn_with_state(keys, auth::auth_middleware))
        .layer(middleware::from_fn(auth::request_id_and_headers_middleware))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(timeout_secs),
        ))
        .with_state(layer)
}

/// Resolves when SIGINT (Ctrl-C) or SIGTERM is received.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let ctrl_c = async { tokio::signal::ctrl_c().await.ok() };
    let sigterm = async {
        match signal(SignalKind::terminate()) {
            Ok(mut stream) => stream.recv().await,
            Err(err) => {
                tracing::warn!(
                    %err,
                    "failed to install SIGTERM handler; continuing with SIGINT only"
                );
                pending::<()>().await;
                None
            }
        }
    };

    tokio::select! {
        _ = ctrl_c => {}
        _ = sigterm => {}
    }

    tracing::info!("shutdown signal received — draining connections");
}
