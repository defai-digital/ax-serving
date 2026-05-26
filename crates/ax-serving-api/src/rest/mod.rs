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

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware,
    middleware::Next,
    response::Response,
    routing::{delete, get, post},
};
use tower_http::timeout::TimeoutLayer;

use crate::ServingLayer;
use crate::auth;
use crate::rest::schema::MAX_HTTP_REQUEST_BODY_BYTES;

/// Start the Axum REST server with graceful shutdown on SIGINT/SIGTERM.
pub async fn serve(
    layer: Arc<ServingLayer>,
    addr: String,
    keys: Arc<HashSet<String>>,
) -> Result<()> {
    let app = try_router(layer, keys)?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("REST server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Build the Axum router.
pub fn router(layer: Arc<ServingLayer>, keys: Arc<HashSet<String>>) -> Router {
    let timeout_secs = match request_timeout_secs_from_env() {
        Ok(timeout_secs) => timeout_secs,
        Err(err) => {
            tracing::warn!(
                %err,
                "invalid AXS_REQUEST_TIMEOUT_SECS ignored by infallible REST router constructor"
            );
            DEFAULT_REQUEST_TIMEOUT_SECS
        }
    };
    router_with_timeout(layer, keys, timeout_secs)
}

/// Build the Axum router, failing on invalid explicit environment overrides.
pub fn try_router(layer: Arc<ServingLayer>, keys: Arc<HashSet<String>>) -> Result<Router> {
    let timeout_secs = request_timeout_secs_from_env()?;
    Ok(router_with_timeout(layer, keys, timeout_secs))
}

const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

fn request_timeout_secs_from_env() -> Result<u64> {
    match std::env::var("AXS_REQUEST_TIMEOUT_SECS") {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Ok(DEFAULT_REQUEST_TIMEOUT_SECS)
            } else {
                let timeout_secs = trimmed
                    .parse::<u64>()
                    .context("invalid AXS_REQUEST_TIMEOUT_SECS")?;
                anyhow::ensure!(timeout_secs > 0, "AXS_REQUEST_TIMEOUT_SECS must be > 0");
                Ok(timeout_secs)
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_REQUEST_TIMEOUT_SECS),
        Err(err) => Err(err).context("invalid AXS_REQUEST_TIMEOUT_SECS"),
    }
}

fn router_with_timeout(
    layer: Arc<ServingLayer>,
    keys: Arc<HashSet<String>>,
    timeout_secs: u64,
) -> Router {
    let slo_layer = Arc::clone(&layer);

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
        .layer(DefaultBodyLimit::max(MAX_HTTP_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            slo_layer,
            inference_slo_middleware,
        ))
        .with_state(layer)
}

async fn inference_slo_middleware(
    State(layer): State<Arc<ServingLayer>>,
    request: Request,
    next: Next,
) -> Response {
    let track = matches!(
        request.uri().path(),
        "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings"
    );
    let response = next.run(request).await;
    if track {
        layer
            .metrics
            .record_slo_sample(response.status().is_server_error());
    }
    response
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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{DEFAULT_REQUEST_TIMEOUT_SECS, request_timeout_secs_from_env};

    struct EnvGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn request_timeout_defaults_when_env_unset() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::unset("AXS_REQUEST_TIMEOUT_SECS");

        assert_eq!(
            request_timeout_secs_from_env().unwrap(),
            DEFAULT_REQUEST_TIMEOUT_SECS
        );
    }

    #[test]
    fn request_timeout_parses_trimmed_env_value() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::set("AXS_REQUEST_TIMEOUT_SECS", " 15 ");

        assert_eq!(request_timeout_secs_from_env().unwrap(), 15);
    }

    #[test]
    fn request_timeout_rejects_malformed_env_value() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::set("AXS_REQUEST_TIMEOUT_SECS", "soon");

        let err = request_timeout_secs_from_env().unwrap_err();
        assert!(err.to_string().contains("AXS_REQUEST_TIMEOUT_SECS"));
    }

    #[test]
    fn request_timeout_rejects_zero_env_value() {
        let _lock = crate::test_env::lock();
        let _guard = EnvGuard::set("AXS_REQUEST_TIMEOUT_SECS", "0");

        let err = request_timeout_secs_from_env().unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }
}
