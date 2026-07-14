//! Orchestrator / proxy / internal-router spawn helpers for integration tests.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use ax_serving_api::orchestration::{
    OrchestratorConfig, OrchestratorLayer, ProjectPolicyConfig,
    internal_routes::{
        InternalAuthState, InternalState, internal_auth_middleware, router as internal_router,
    },
};
use axum::{Router, middleware};

pub fn proxy_router_with_key(layer: Arc<OrchestratorLayer>, key: &str) -> Router {
    layer.set_public_auth_required(true);
    let mut keys = HashSet::new();
    keys.insert(key.to_string());
    ax_serving_api::orchestration::proxy_router(layer)
        .route_layer(middleware::from_fn_with_state(
            Arc::new(keys),
            ax_serving_api::auth::auth_middleware,
        ))
        .layer(middleware::from_fn(
            ax_serving_api::auth::request_id_and_headers_middleware,
        ))
}

/// Spawn an `OrchestratorLayer`-backed proxy server on an ephemeral port.
///
/// Returns the bound address and an `Arc` to the layer so tests can
/// manipulate the queue (hold permits, register workers) directly.
pub async fn spawn_orchestrator_with_layer(
    cfg: OrchestratorConfig,
) -> Option<(SocketAddr, Arc<OrchestratorLayer>)> {
    use ax_serving_api::orchestration::proxy_router;
    let layer = Arc::new(
        OrchestratorLayer::new(
            cfg,
            ax_serving_api::config::LicenseConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .ok()?,
    );
    let router = proxy_router(Arc::clone(&layer));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .ok();
    });
    Some((addr, layer))
}

pub async fn spawn_internal_router_with_auth(
    state: InternalState,
    auth_state: Option<InternalAuthState>,
) -> Option<SocketAddr> {
    let app = if let Some(auth_state) = auth_state {
        internal_router(state).route_layer(middleware::from_fn_with_state(
            auth_state,
            internal_auth_middleware,
        ))
    } else {
        internal_router(state)
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .ok();
    });
    Some(addr)
}
