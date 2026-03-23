//! Internal REST API — `/internal/workers/*`.
//!
//! Bind this router to loopback by default. When exposed on a non-loopback
//! interface, it must be protected with worker token auth and source-IP filtering.
//!
//! # Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/internal/workers/register` | Register or re-register a worker |
//! | POST | `/internal/workers/{id}/heartbeat` | Update health + inflight |
//! | POST | `/internal/workers/{id}/drain` | Stop sending new requests to worker |
//! | POST | `/internal/workers/{id}/drain-complete` | Remove worker (restart ready) |
//! | DELETE | `/internal/workers/{id}` | Remove worker immediately (one-step) |
//! | GET  | `/internal/workers` | List all workers |
//! | GET  | `/internal/workers/{id}` | Get single worker |

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::ConnectInfo,
    extract::Request,
    extract::{Path, State},
    http::HeaderValue,
    http::StatusCode,
    middleware::Next,
    response::IntoResponse,
    response::Response,
    routing::{get, post},
};
use tracing::info;

use std::net::SocketAddr;

use super::OrchestratorConfig;
use super::registry::{HeartbeatRequest, RegisterRequest, WorkerId, WorkerRegistry};
use crate::license::LicenseState;
use ipnet::IpNet;

// ── State ─────────────────────────────────────────────────────────────────────

/// State passed to internal route handlers.
#[derive(Clone)]
pub struct InternalState {
    pub registry: WorkerRegistry,
    pub config: Arc<OrchestratorConfig>,
    pub license: Arc<LicenseState>,
}

#[derive(Clone)]
pub struct InternalAuthState {
    pub token: Option<Arc<String>>,
    pub allowed_sources: Arc<Vec<IpNet>>,
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Build the internal Axum router.
///
/// Bind the returned router to the configured internal listener address.
pub fn router(state: InternalState) -> Router {
    Router::new()
        .route("/internal/workers/register", post(handle_register))
        .route("/internal/workers/{id}/heartbeat", post(handle_heartbeat))
        .route("/internal/workers/{id}/drain", post(handle_drain))
        .route(
            "/internal/workers/{id}/drain-complete",
            post(handle_drain_complete),
        )
        .route("/internal/workers", get(handle_list))
        .route(
            "/internal/workers/{id}",
            get(handle_get).delete(handle_delete),
        )
        .with_state(state)
}

/// Optional middleware for internal worker-control API token auth.
///
/// Enable by setting `AXS_INTERNAL_API_TOKEN` in both orchestrator and workers.
/// Workers send the token in `X-Internal-Token`.
pub async fn internal_auth_middleware(
    State(state): State<InternalAuthState>,
    request: Request,
    next: Next,
) -> Response {
    if !state.allowed_sources.is_empty() {
        let peer_ip = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|v| v.0.ip());

        match peer_ip {
            Some(ip) if state.allowed_sources.iter().any(|net| net.contains(&ip)) => {}
            Some(_) => {
                return (StatusCode::FORBIDDEN, "source IP not allowed").into_response();
            }
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "missing peer address for internal API request",
                )
                    .into_response();
            }
        }
    }

    if let Some(expected_token) = &state.token {
        let provided = request
            .headers()
            .get("x-internal-token")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .unwrap_or("");

        if !crate::auth::constant_time_eq_str(provided, expected_token.as_str()) {
            return (
                StatusCode::UNAUTHORIZED,
                [(
                    axum::http::header::WWW_AUTHENTICATE,
                    HeaderValue::from_static("X-Internal-Token"),
                )],
                "missing or invalid internal API token",
            )
                .into_response();
        }
    }

    next.run(request).await
}

pub fn parse_allowed_node_cidrs(raw: &str) -> anyhow::Result<Vec<IpNet>> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            if let Ok(net) = s.parse::<IpNet>() {
                return Ok(net);
            }
            if let Ok(ip) = s.parse::<std::net::IpAddr>() {
                return Ok(IpNet::from(ip));
            }
            Err(anyhow::anyhow!(
                "invalid AXS_ALLOWED_NODE_CIDRS entry '{s}': expected IP or CIDR"
            ))
        })
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_worker_id(id_str: &str) -> Result<WorkerId, StatusCode> {
    WorkerId::parse(id_str).ok_or(StatusCode::BAD_REQUEST)
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /internal/workers/register`
async fn handle_register(
    State(s): State<InternalState>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    // Validate addr before registering — a malformed addr would silently route
    // to 127.0.0.1:1 in the registry, accepting the worker but never sending it traffic.
    let Ok(addr) = req.addr.parse::<SocketAddr>() else {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "invalid worker addr '{}': must be a valid host:port",
                req.addr
            ),
        )
            .into_response();
    };

    // Detect remote (non-loopback) workers for the license reminder.
    if !addr.ip().is_loopback() {
        s.license.mark_remote_worker_seen();
    }
    let resp = s.registry.register(req, s.config.worker_heartbeat_ms);
    info!(worker_id = %resp.worker_id, "worker registered");
    (StatusCode::OK, Json(resp)).into_response()
}

/// `POST /internal/workers/{id}/heartbeat`
async fn handle_heartbeat(
    State(s): State<InternalState>,
    Path(id_str): Path<String>,
    Json(req): Json<HeartbeatRequest>,
) -> impl IntoResponse {
    let id = match parse_worker_id(&id_str) {
        Ok(id) => id,
        Err(status) => return (status, "invalid worker id").into_response(),
    };
    if s.registry.heartbeat(id, req) {
        StatusCode::OK.into_response()
    } else {
        (StatusCode::NOT_FOUND, "worker not found").into_response()
    }
}

/// `POST /internal/workers/{id}/drain`
async fn handle_drain(
    State(s): State<InternalState>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let id = match parse_worker_id(&id_str) {
        Ok(id) => id,
        Err(status) => return (status, "invalid worker id").into_response(),
    };
    if s.registry.mark_drain(id) {
        info!(%id, "worker marked for drain");
        StatusCode::OK.into_response()
    } else {
        (StatusCode::NOT_FOUND, "worker not found").into_response()
    }
}

/// `POST /internal/workers/{id}/drain-complete`
async fn handle_drain_complete(
    State(s): State<InternalState>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let id = match parse_worker_id(&id_str) {
        Ok(id) => id,
        Err(status) => return (status, "invalid worker id").into_response(),
    };
    s.registry.evict(id);
    info!(%id, "worker drain complete, evicted");
    StatusCode::NO_CONTENT.into_response()
}

/// `GET /internal/workers`
async fn handle_list(State(s): State<InternalState>) -> impl IntoResponse {
    let workers = s.registry.list_all();
    Json(serde_json::json!({ "workers": workers }))
}

/// `GET /internal/workers/{id}`
async fn handle_get(
    State(s): State<InternalState>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let id = match parse_worker_id(&id_str) {
        Ok(id) => id,
        Err(status) => return (status, "invalid worker id").into_response(),
    };
    match s.registry.get_snapshot(id) {
        Some(snap) => Json(snap).into_response(),
        None => (StatusCode::NOT_FOUND, "worker not found").into_response(),
    }
}

/// `DELETE /internal/workers/{id}` — remove a worker immediately (drain + evict in one step).
///
/// Use this to undo an accidental registration or force-remove a stuck worker.
/// For graceful shutdown of a live worker use the two-step drain → drain-complete flow instead.
async fn handle_delete(
    State(s): State<InternalState>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let id = match parse_worker_id(&id_str) {
        Ok(id) => id,
        Err(status) => return (status, "invalid worker id").into_response(),
    };
    if !s.registry.mark_drain(id) {
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    s.registry.evict(id);
    info!(%id, "worker force-removed");
    StatusCode::NO_CONTENT.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, middleware, routing::get};
    use tower::ServiceExt;

    #[test]
    fn parse_allowed_node_cidrs_accepts_ip_and_cidr() {
        let parsed = parse_allowed_node_cidrs("127.0.0.1,10.0.0.0/8").unwrap();
        let loopback: std::net::IpAddr = "127.0.0.1".parse().unwrap();
        let lab_ip: std::net::IpAddr = "10.1.2.3".parse().unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].contains(&loopback));
        assert!(parsed[1].contains(&lab_ip));
    }

    #[test]
    fn parse_allowed_node_cidrs_rejects_invalid_entry() {
        let err = parse_allowed_node_cidrs("127.0.0.1,not-a-cidr").unwrap_err();
        assert!(err.to_string().contains("not-a-cidr"));
    }

    #[tokio::test]
    async fn internal_auth_middleware_rejects_disallowed_source_ip() {
        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .route_layer(middleware::from_fn_with_state(
                InternalAuthState {
                    token: Some(Arc::new("secret".to_string())),
                    allowed_sources: Arc::new(parse_allowed_node_cidrs("127.0.0.1/32").unwrap()),
                },
                internal_auth_middleware,
            ));

        let mut req = Request::builder()
            .uri("/ok")
            .header("x-internal-token", "secret")
            .body(axum::body::Body::empty())
            .unwrap();
        req.extensions_mut()
            .insert(ConnectInfo("10.0.0.2:12345".parse::<SocketAddr>().unwrap()));

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn internal_auth_middleware_accepts_allowed_source_ip() {
        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .route_layer(middleware::from_fn_with_state(
                InternalAuthState {
                    token: Some(Arc::new("secret".to_string())),
                    allowed_sources: Arc::new(parse_allowed_node_cidrs("127.0.0.1/32").unwrap()),
                },
                internal_auth_middleware,
            ));

        let mut req = Request::builder()
            .uri("/ok")
            .header("x-internal-token", "secret")
            .body(axum::body::Body::empty())
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(
            "127.0.0.1:12345".parse::<SocketAddr>().unwrap(),
        ));

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn internal_auth_middleware_allows_allowlist_only_mode() {
        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .route_layer(middleware::from_fn_with_state(
                InternalAuthState {
                    token: None,
                    allowed_sources: Arc::new(parse_allowed_node_cidrs("127.0.0.1/32").unwrap()),
                },
                internal_auth_middleware,
            ));

        let mut req = Request::builder()
            .uri("/ok")
            .body(axum::body::Body::empty())
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(
            "127.0.0.1:12345".parse::<SocketAddr>().unwrap(),
        ));

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
