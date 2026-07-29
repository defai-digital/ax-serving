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

use std::collections::BTreeSet;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::ConnectInfo,
    extract::DefaultBodyLimit,
    extract::Request,
    extract::{Path, State},
    http::StatusCode,
    http::{HeaderMap, HeaderValue},
    middleware::Next,
    response::IntoResponse,
    response::Response,
    routing::{get, post},
};
use tracing::info;

use std::net::SocketAddr;

use super::OrchestratorConfig;
use super::fleet_state::{FleetMutationResult, FleetStateStore, unix_time_millis};
use super::registry::{
    HeartbeatRequest, ProtocolRegistryError, RegisterRequest, WorkerId, WorkerRegistry,
};
use ax_serving_protocol::{
    CURRENT_PROTOCOL, ProtocolCapability, RegisterWorkerRequest, WorkerId as ProtocolWorkerId,
    negotiate_protocol,
};
use ipnet::IpNet;

const MAX_INTERNAL_BODY_BYTES: usize = 2 * 1024 * 1024;
const LEASE_TOKEN_HEADER: &str = "x-ax-lease-token";

// ── State ─────────────────────────────────────────────────────────────────────

/// State passed to internal route handlers.
#[derive(Clone)]
pub struct InternalState {
    pub registry: WorkerRegistry,
    pub fleet_store: Arc<dyn FleetStateStore>,
    pub config: Arc<OrchestratorConfig>,
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
        .layer(DefaultBodyLimit::max(MAX_INTERNAL_BODY_BYTES))
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

fn parse_worker_id(registry: &WorkerRegistry, id_str: &str) -> Result<WorkerId, StatusCode> {
    registry
        .resolve_worker_id(id_str)
        .ok_or(StatusCode::BAD_REQUEST)
}

fn gateway_protocol_capabilities() -> BTreeSet<ProtocolCapability> {
    [
        ProtocolCapability::CONTROL_DRAIN,
        ProtocolCapability::CONTROL_EXECUTION_DOMAIN,
        ProtocolCapability::CONTROL_MAC_CLUSTER,
        ProtocolCapability::CONTROL_INVENTORY_DELTA,
        ProtocolCapability::DISPATCH_CANCEL,
        ProtocolCapability::DISPATCH_TYPED_ADMISSION,
        ProtocolCapability::TELEMETRY_CAPACITY,
        ProtocolCapability::TELEMETRY_DOMAIN_CAPACITY,
        ProtocolCapability::TELEMETRY_KV_CACHE,
        ProtocolCapability::TELEMETRY_PREFIX_CACHE,
    ]
    .into_iter()
    .map(|capability| ProtocolCapability::new(capability).expect("static protocol capability"))
    .collect()
}

fn advertised_endpoint(raw: &str) -> Result<super::worker_endpoint::WorkerEndpoint, String> {
    super::worker_endpoint::WorkerEndpoint::parse(raw)
}

fn validate_observation_age(observed_at: time::OffsetDateTime) -> Result<(), String> {
    const MAX_CLOCK_SKEW_SECS: i64 = 300;
    let now = time::OffsetDateTime::now_utc();
    let age = (now - observed_at).whole_seconds();
    if age.unsigned_abs() > MAX_CLOCK_SKEW_SECS as u64 {
        return Err(format!(
            "runtime observation exceeds the allowed {MAX_CLOCK_SKEW_SECS}s clock skew"
        ));
    }
    Ok(())
}

fn protocol_registry_error_response(error: ProtocolRegistryError) -> Response {
    let (status, code) = match &error {
        ProtocolRegistryError::NotRegistered => (StatusCode::NOT_FOUND, "AXS_WORKER_NOT_FOUND"),
        ProtocolRegistryError::InvalidLeaseToken => {
            (StatusCode::UNAUTHORIZED, "AXS_INVALID_LEASE_TOKEN")
        }
        ProtocolRegistryError::InstanceMismatch | ProtocolRegistryError::RegistrationMismatch => {
            (StatusCode::CONFLICT, "AXS_STALE_REGISTRATION")
        }
        ProtocolRegistryError::ReplayedHeartbeat { .. } => {
            (StatusCode::CONFLICT, "AXS_REPLAYED_HEARTBEAT")
        }
        ProtocolRegistryError::InvalidObservation(_) => {
            (StatusCode::UNPROCESSABLE_ENTITY, "AXS_INVALID_OBSERVATION")
        }
        ProtocolRegistryError::InternalRegistration => {
            (StatusCode::INTERNAL_SERVER_ERROR, "AXS_REGISTRATION_FAILED")
        }
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "code": code,
                "message": error.to_string(),
            }
        })),
    )
        .into_response()
}

fn fleet_unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": {
                "code": "AXS_FLEET_STATE_UNAVAILABLE",
                "message": "shared fleet state is temporarily unavailable",
            }
        })),
    )
        .into_response()
}

fn fleet_fencing_response(result: FleetMutationResult) -> Response {
    let (code, message) = match result {
        FleetMutationResult::Missing => (
            "AXS_LEASE_EXPIRED",
            "worker lease is missing or expired; re-registration is required",
        ),
        FleetMutationResult::Fenced => (
            "AXS_STALE_REGISTRATION",
            "worker registration has been superseded",
        ),
        FleetMutationResult::StaleSequence => (
            "AXS_REPLAYED_HEARTBEAT",
            "a newer worker heartbeat has already been accepted",
        ),
        FleetMutationResult::Applied => (
            "AXS_INTERNAL_ERROR",
            "unexpected fleet-state mutation result",
        ),
    };
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        })),
    )
        .into_response()
}

async fn synchronize_protocol_worker(
    state: &InternalState,
    worker_id: &ProtocolWorkerId,
) -> Result<(), Response> {
    let record = state
        .fleet_store
        .get(worker_id)
        .await
        .map_err(|_| fleet_unavailable_response())?;
    let Some(record) = record else {
        state.registry.evict_protocol(worker_id);
        return Err(protocol_registry_error_response(
            ProtocolRegistryError::NotRegistered,
        ));
    };
    if !record.is_fresh(unix_time_millis()) {
        let _ = state
            .fleet_store
            .remove_if_registration(worker_id, record.registration_id)
            .await;
        state.registry.evict_protocol(worker_id);
        return Err(fleet_fencing_response(FleetMutationResult::Missing));
    }
    state
        .registry
        .restore_protocol_record_if_newer(record)
        .map_err(protocol_registry_error_response)?;
    Ok(())
}

async fn persist_protocol_record(
    state: &InternalState,
    worker_id: &ProtocolWorkerId,
) -> Result<(), Response> {
    let record = state
        .registry
        .export_protocol_record(worker_id)
        .ok_or_else(|| protocol_registry_error_response(ProtocolRegistryError::NotRegistered))?;
    let result = state
        .fleet_store
        .compare_and_put(&record)
        .await
        .map_err(|_| fleet_unavailable_response())?;
    if result == FleetMutationResult::Applied {
        return Ok(());
    }
    // Pull the winning lease into the local mirror before returning the fence.
    let _ = synchronize_protocol_worker(state, worker_id).await;
    Err(fleet_fencing_response(result))
}

fn worker_id_for_control_action(
    registry: &WorkerRegistry,
    raw: &str,
    headers: &HeaderMap,
) -> Result<WorkerId, Box<Response>> {
    if let Some(id) = WorkerId::parse(raw) {
        return Ok(id);
    }
    let protocol_id = raw
        .parse::<ProtocolWorkerId>()
        .map_err(|_| Box::new((StatusCode::BAD_REQUEST, "invalid worker id").into_response()))?;
    let token = headers
        .get(LEASE_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .unwrap_or("");
    registry
        .validate_protocol_lease(&protocol_id, token)
        .map_err(|error| Box::new(protocol_registry_error_response(error)))
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /internal/workers/register`
async fn handle_register(State(s): State<InternalState>, body: Bytes) -> impl IntoResponse {
    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid worker registration JSON: {error}"),
            )
                .into_response();
        }
    };

    if value.get("protocol").is_some() {
        let request: RegisterWorkerRequest = match serde_json::from_value(value) {
            Ok(request) => request,
            Err(error) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("invalid protocol-v1 registration: {error}"),
                )
                    .into_response();
            }
        };
        return handle_protocol_register(&s, request).await;
    }

    let req: RegisterRequest = match serde_json::from_value(value) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid legacy worker registration: {error}"),
            )
                .into_response();
        }
    };
    // Validate addr before registering — a malformed addr would silently route
    // to a sentinel endpoint in the registry, accepting the worker but never
    // sending it traffic.
    if let Err(error) = advertised_endpoint(&req.addr) {
        return (
            StatusCode::BAD_REQUEST,
            format!("invalid worker addr '{}': {error}", req.addr),
        )
            .into_response();
    }

    let resp = s.registry.register(req, s.config.worker_heartbeat_ms);
    info!(worker_id = %resp.worker_id, "worker registered");
    (StatusCode::OK, Json(resp)).into_response()
}

async fn handle_protocol_register(
    state: &InternalState,
    request: RegisterWorkerRequest,
) -> Response {
    let addr = match advertised_endpoint(&request.worker.advertise_url) {
        Ok(addr) => addr,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    if let Err(error) = validate_observation_age(request.observation.observed_at) {
        return (StatusCode::UNPROCESSABLE_ENTITY, error).into_response();
    }
    if let Some(observation) = &request.domain_observation
        && let Err(error) = validate_observation_age(observation.observed_at)
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, error).into_response();
    }
    let capabilities = gateway_protocol_capabilities();
    let negotiated = match negotiate_protocol(
        &request.protocol,
        CURRENT_PROTOCOL.major,
        0,
        CURRENT_PROTOCOL.minor,
        &capabilities,
    ) {
        Ok(negotiated) => negotiated,
        Err(error) => {
            return (
                StatusCode::UPGRADE_REQUIRED,
                Json(serde_json::json!({
                    "error": {
                        "code": "AXS_PROTOCOL_INCOMPATIBLE",
                        "message": error.to_string(),
                    }
                })),
            )
                .into_response();
        }
    };

    let stable_id = request.worker.id.clone();
    let response = match state.registry.register_protocol(
        request,
        addr,
        negotiated,
        state.config.worker_heartbeat_ms,
        state.config.worker_ttl_ms,
    ) {
        Ok(response) => response,
        Err(error) => return protocol_registry_error_response(error),
    };
    let Some(record) = state.registry.export_protocol_record(&stable_id) else {
        state.registry.evict_protocol(&stable_id);
        return protocol_registry_error_response(ProtocolRegistryError::InternalRegistration);
    };
    if state.fleet_store.put(&record).await.is_err() {
        state.registry.evict_protocol(&stable_id);
        return fleet_unavailable_response();
    }
    info!(worker_id = %stable_id, "protocol-v1 worker registered");
    (StatusCode::OK, Json(response)).into_response()
}

/// `POST /internal/workers/{id}/heartbeat`
async fn handle_heartbeat(
    State(s): State<InternalState>,
    Path(id_str): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid heartbeat JSON: {error}"),
            )
                .into_response();
        }
    };

    if value.get("registration_id").is_some() {
        let worker_id = match id_str.parse::<ProtocolWorkerId>() {
            Ok(worker_id) => worker_id,
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "invalid protocol worker id").into_response();
            }
        };
        let request: ax_serving_protocol::HeartbeatRequest = match serde_json::from_value(value) {
            Ok(request) => request,
            Err(error) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("invalid protocol-v1 heartbeat: {error}"),
                )
                    .into_response();
            }
        };
        if let Err(error) = validate_observation_age(request.observed_at) {
            return (StatusCode::UNPROCESSABLE_ENTITY, error).into_response();
        }
        if let Some(observation) = &request.domain_observation
            && let Err(error) = validate_observation_age(observation.observed_at)
        {
            return (StatusCode::UNPROCESSABLE_ENTITY, error).into_response();
        }
        let lease_token = headers
            .get(LEASE_TOKEN_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .unwrap_or("");
        if let Err(response) = synchronize_protocol_worker(&s, &worker_id).await {
            return response;
        }
        let response = match s
            .registry
            .heartbeat_protocol(&worker_id, lease_token, request)
        {
            Ok(response) => response,
            Err(error) => return protocol_registry_error_response(error),
        };
        if let Err(response) = persist_protocol_record(&s, &worker_id).await {
            return response;
        }
        return (StatusCode::OK, Json(response)).into_response();
    }

    let req: HeartbeatRequest = match serde_json::from_value(value) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid legacy heartbeat: {error}"),
            )
                .into_response();
        }
    };
    let id = match parse_worker_id(&s.registry, &id_str) {
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
    headers: HeaderMap,
) -> impl IntoResponse {
    let protocol_worker_id = WorkerId::parse(&id_str)
        .is_none()
        .then(|| id_str.parse::<ProtocolWorkerId>().ok())
        .flatten();
    if let Some(worker_id) = &protocol_worker_id
        && let Err(response) = synchronize_protocol_worker(&s, worker_id).await
    {
        return response;
    }
    let id = match worker_id_for_control_action(&s.registry, &id_str, &headers) {
        Ok(id) => id,
        Err(response) => return *response,
    };
    if s.registry.mark_drain(id) {
        if let Some(worker_id) = &protocol_worker_id
            && let Err(response) = persist_protocol_record(&s, worker_id).await
        {
            return response;
        }
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
    headers: HeaderMap,
) -> impl IntoResponse {
    let protocol_worker_id = WorkerId::parse(&id_str)
        .is_none()
        .then(|| id_str.parse::<ProtocolWorkerId>().ok())
        .flatten();
    if let Some(worker_id) = &protocol_worker_id
        && let Err(response) = synchronize_protocol_worker(&s, worker_id).await
    {
        return response;
    }
    let id = match worker_id_for_control_action(&s.registry, &id_str, &headers) {
        Ok(id) => id,
        Err(response) => return *response,
    };
    if s.registry.get_snapshot(id).is_none() {
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    if let Some(worker_id) = &protocol_worker_id {
        let Some((_, registration_id)) = s.registry.protocol_identity_for_internal(id) else {
            return protocol_registry_error_response(ProtocolRegistryError::NotRegistered);
        };
        let result = match s
            .fleet_store
            .remove_if_registration(worker_id, registration_id)
            .await
        {
            Ok(result) => result,
            Err(_) => return fleet_unavailable_response(),
        };
        if result != FleetMutationResult::Applied {
            let _ = synchronize_protocol_worker(&s, worker_id).await;
            return fleet_fencing_response(result);
        }
    }
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
    let id = match parse_worker_id(&s.registry, &id_str) {
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
    let id = match parse_worker_id(&s.registry, &id_str) {
        Ok(id) => id,
        Err(status) => return (status, "invalid worker id").into_response(),
    };
    if !s.registry.mark_drain(id) {
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    if let Some((worker_id, registration_id)) = s.registry.protocol_identity_for_internal(id) {
        let result = match s
            .fleet_store
            .remove_if_registration(&worker_id, registration_id)
            .await
        {
            Ok(result) => result,
            Err(_) => return fleet_unavailable_response(),
        };
        if result != FleetMutationResult::Applied {
            let _ = synchronize_protocol_worker(&s, &worker_id).await;
            return fleet_fencing_response(result);
        }
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

    fn test_state() -> InternalState {
        test_state_with_store(super::super::fleet_state::MemoryFleetStateStore::shared())
    }

    fn test_state_with_store(fleet_store: Arc<dyn FleetStateStore>) -> InternalState {
        InternalState {
            registry: WorkerRegistry::new(),
            fleet_store,
            config: Arc::new(OrchestratorConfig::default()),
        }
    }

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

    #[tokio::test]
    async fn drain_complete_returns_404_for_unknown_worker() {
        let id = WorkerId::new();
        let response =
            handle_drain_complete(State(test_state()), Path(id.to_string()), HeaderMap::new())
                .await
                .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn drain_complete_evicts_known_worker() {
        let state = test_state();
        let register = state.registry.register(
            RegisterRequest {
                addr: "127.0.0.1:18081".into(),
                capabilities: super::super::registry::RegisterCapabilities::Legacy(vec![
                    "m1".into(),
                ]),
                max_inflight: 1,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&register.worker_id).unwrap();

        let response = handle_drain_complete(
            State(state.clone()),
            Path(register.worker_id),
            HeaderMap::new(),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(state.registry.get_snapshot(id).is_none());
    }

    fn current_rfc3339() -> String {
        time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap()
    }

    fn protocol_registration() -> serde_json::Value {
        serde_json::json!({
            "protocol": {
                "version": {"major": 1, "minor": 0},
                "capabilities": [
                    "control.drain",
                    "dispatch.typed-admission",
                    "telemetry.capacity"
                ]
            },
            "agent": {"name": "ax-runtime-agent", "version": "3.0.0"},
            "worker": {
                "id": "mac-worker-1",
                "instance_id": "627f7e26-348f-4fe6-b9b4-cce6785d17ea",
                "advertise_url": "http://127.0.0.1:18081",
                "pool_id": "mac-qwen",
                "trust_domain": "private-prod",
                "labels": {"node_class": "m3-ultra"}
            },
            "runtime": {"kind": "ax_engine", "version": "6.8.2", "api": "openai-v1"},
            "hardware": {
                "platform": "macos",
                "accelerator": "apple-gpu",
                "device_count": 1,
                "hardware_class": "m3-ultra"
            },
            "observation": {
                "observed_at": current_rfc3339(),
                "runtime": {"ready": true, "state": "ready"},
                "inventory_generation": 1,
                "models": [{
                    "runtime_model_id": "qwen-main",
                    "identity": {"runtime_kind": "ax_engine", "runtime_version": "6.8.2"},
                    "operations": ["chat_completions"],
                    "capabilities": [],
                    "max_context_tokens": 32768,
                    "max_output_tokens": 4096
                }],
                "capacity": {"active_requests": 0, "max_concurrent_requests": 8}
            }
        })
    }

    fn protocol_domain_registration() -> serde_json::Value {
        let mut value = protocol_registration();
        value["protocol"]["version"]["minor"] = serde_json::json!(1);
        value["protocol"]["capabilities"] = serde_json::json!([
            "control.drain",
            "control.execution-domain.v1",
            "dispatch.typed-admission",
            "telemetry.capacity",
            "telemetry.domain-capacity.v1"
        ]);
        value["agent"]["name"] = serde_json::json!("ax-dynamo-adapter");
        value["worker"]["id"] = serde_json::json!("dynamo-pc-adapter-1");
        value["worker"]["pool_id"] = serde_json::json!("nvidia-pc");
        value["worker"]["trust_domain"] = serde_json::json!("private-dc");
        value["runtime"]["kind"] = serde_json::json!("dynamo");
        value["runtime"]["version"] = serde_json::json!("1.2.1");
        value["hardware"]["platform"] = serde_json::json!("linux");
        value["hardware"]["accelerator"] = serde_json::json!("nvidia-cuda");
        value["hardware"]["hardware_class"] = serde_json::json!("nvidia-pc-cuda");
        value["observation"]["models"][0]["identity"]["runtime_kind"] = serde_json::json!("dynamo");
        value["observation"]["models"][0]["identity"]["runtime_version"] =
            serde_json::json!("1.2.1");
        value["domain"] = serde_json::json!({
            "id": "nvidia-pc-main",
            "kind": "nvidia_dynamo_pc",
            "endpoint_scope": "domain",
            "execution_owner": "dynamo",
            "qualification": "certified",
            "pool_id": "nvidia-pc",
            "trust_domain": "private-dc",
            "hardware_class": "nvidia-pc-cuda",
            "architecture": "x86_64",
            "compatibility_manifest": format!("sha256:{}", "a".repeat(64)),
            "labels": {"zone": "dc-a"}
        });
        value["domain_observation"] = serde_json::json!({
            "observed_at": current_rfc3339(),
            "generation": 1,
            "ready": true,
            "state": "ready",
            "frontend_instances_ready": 2,
            "aggregate_capacity": {
                "active_requests": 0,
                "max_concurrent_requests": 8
            },
            "manifest_digest": format!("sha256:{}", "a".repeat(64)),
            "models": value["observation"]["models"].clone()
        });
        value
    }

    fn protocol_mac_cluster_registration() -> serde_json::Value {
        let mut value = protocol_domain_registration();
        value["protocol"]["version"]["minor"] = serde_json::json!(2);
        value["protocol"]["capabilities"] = serde_json::json!([
            "control.drain",
            "control.execution-domain.v1",
            "control.mac-cluster.v1",
            "dispatch.typed-admission",
            "telemetry.capacity",
            "telemetry.domain-capacity.v1"
        ]);
        value["agent"]["name"] = serde_json::json!("ax-mac-cluster-adapter");
        value["worker"]["id"] = serde_json::json!("mac-cluster-adapter-1");
        value["worker"]["pool_id"] = serde_json::json!("mac-cluster");
        value["worker"]["trust_domain"] = serde_json::json!("private-lab");
        value["runtime"]["kind"] = serde_json::json!("ax_engine");
        value["runtime"]["version"] = serde_json::json!("4.10.0");
        value["hardware"]["platform"] = serde_json::json!("macos");
        value["hardware"]["accelerator"] = serde_json::json!("apple-silicon-cluster");
        value["hardware"]["hardware_class"] = serde_json::json!("apple-silicon-cluster");
        value["observation"]["models"][0]["identity"]["runtime_kind"] =
            serde_json::json!("ax_engine");
        value["observation"]["models"][0]["identity"]["runtime_version"] =
            serde_json::json!("4.10.0");
        value["domain"]["id"] = serde_json::json!("mac-cluster-main");
        value["domain"]["kind"] = serde_json::json!("mac_ax_engine_cluster");
        value["domain"]["execution_owner"] = serde_json::json!("ax_engine");
        value["domain"]["pool_id"] = serde_json::json!("mac-cluster");
        value["domain"]["trust_domain"] = serde_json::json!("private-lab");
        value["domain"]["hardware_class"] = serde_json::json!("apple-silicon-cluster");
        value["domain"]["architecture"] = serde_json::json!("arm64");
        value
    }

    #[tokio::test]
    async fn protocol_v1_1_domain_registration_and_observation_are_visible() {
        let state = test_state();
        let app = router(state.clone());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        protocol_domain_registration().to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let registration: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let heartbeat = serde_json::json!({
            "registration_id": registration["registration_id"],
            "instance_id": "627f7e26-348f-4fe6-b9b4-cce6785d17ea",
            "sequence": 1,
            "observed_at": current_rfc3339(),
            "runtime": {"ready": true, "state": "ready"},
            "inventory_generation": 1,
            "domain_observation": {
                "observed_at": current_rfc3339(),
                "generation": 2,
                "ready": true,
                "state": "degraded",
                "reason_code": "capacity_reduced",
                "frontend_instances_ready": 1,
                "aggregate_capacity": {
                    "active_requests": 3,
                    "max_concurrent_requests": 8
                },
                "manifest_digest": format!("sha256:{}", "a".repeat(64)),
                "models": protocol_domain_registration()["observation"]["models"].clone()
            }
        });
        let heartbeat_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/dynamo-pc-adapter-1/heartbeat")
                    .header("content-type", "application/json")
                    .header(
                        LEASE_TOKEN_HEADER,
                        registration["lease_token"].as_str().unwrap(),
                    )
                    .body(axum::body::Body::from(heartbeat.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(heartbeat_response.status(), StatusCode::OK);

        let missing_domain_observation = serde_json::json!({
            "registration_id": registration["registration_id"],
            "instance_id": "627f7e26-348f-4fe6-b9b4-cce6785d17ea",
            "sequence": 2,
            "observed_at": current_rfc3339(),
            "runtime": {"ready": true, "state": "ready"},
            "inventory_generation": 1
        });
        let missing_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/dynamo-pc-adapter-1/heartbeat")
                    .header("content-type", "application/json")
                    .header(
                        LEASE_TOKEN_HEADER,
                        registration["lease_token"].as_str().unwrap(),
                    )
                    .body(axum::body::Body::from(
                        missing_domain_observation.to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_response.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let id = state
            .registry
            .resolve_worker_id("dynamo-pc-adapter-1")
            .unwrap();
        let snapshot = state.registry.get_snapshot(id).unwrap();
        assert_eq!(
            snapshot.domain.as_ref().map(|domain| domain.id.as_str()),
            Some("nvidia-pc-main")
        );
        assert_eq!(
            snapshot
                .domain_observation
                .as_ref()
                .map(|observation| observation.generation),
            Some(2)
        );
        assert_eq!(snapshot.active_sequences, 3);
    }

    #[tokio::test]
    async fn protocol_v1_2_mac_cluster_registration_is_domain_scoped() {
        let state = test_state();
        let app = router(state.clone());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        protocol_mac_cluster_registration().to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let id = state
            .registry
            .resolve_worker_id("mac-cluster-adapter-1")
            .unwrap();
        let snapshot = state.registry.get_snapshot(id).unwrap();
        let descriptor = snapshot.domain.unwrap();
        assert_eq!(
            descriptor.kind,
            ax_serving_protocol::ExecutionDomainKind::MacAxEngineCluster
        );
        assert_eq!(
            descriptor.endpoint_scope,
            ax_serving_protocol::EndpointScope::Domain
        );
    }

    #[tokio::test]
    async fn domain_descriptor_without_v1_1_capability_is_rejected() {
        let mut registration = protocol_domain_registration();
        registration["protocol"]["capabilities"] = serde_json::json!(["telemetry.capacity"]);
        let response = router(test_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(registration.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn protocol_registration_and_lease_heartbeat_are_fenced() {
        let state = test_state();
        let app = router(state.clone());
        let register_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(protocol_registration().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(register_response.status(), StatusCode::OK);
        let register_body = axum::body::to_bytes(register_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let registration: serde_json::Value = serde_json::from_slice(&register_body).unwrap();
        let lease_token = registration["lease_token"].as_str().unwrap();
        let registration_id = registration["registration_id"].as_str().unwrap();

        let heartbeat = serde_json::json!({
            "registration_id": registration_id,
            "instance_id": "627f7e26-348f-4fe6-b9b4-cce6785d17ea",
            "sequence": 1,
            "observed_at": current_rfc3339(),
            "runtime": {
                "ready": false,
                "state": "unavailable",
                "reason_code": "runtime_connect_failed"
            },
            "inventory_generation": 1,
            "capacity": {"active_requests": 0, "max_concurrent_requests": 8}
        });
        let rejected = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/mac-worker-1/heartbeat")
                    .header("content-type", "application/json")
                    .header(LEASE_TOKEN_HEADER, "wrong-lease-token")
                    .body(axum::body::Body::from(heartbeat.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);

        let accepted = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/mac-worker-1/heartbeat")
                    .header("content-type", "application/json")
                    .header(LEASE_TOKEN_HEADER, lease_token)
                    .body(axum::body::Body::from(heartbeat.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);

        let internal_id = state.registry.resolve_worker_id("mac-worker-1").unwrap();
        let snapshot = state.registry.get_snapshot(internal_id).unwrap();
        assert_eq!(snapshot.protocol_worker_id.as_deref(), Some("mac-worker-1"));
        assert_eq!(snapshot.runtime_ready, Some(false));
        assert!(state.registry.eligible_workers("qwen-main").is_empty());
    }

    #[tokio::test]
    async fn shared_fleet_state_restores_worker_on_another_gateway() {
        let store: Arc<dyn FleetStateStore> =
            super::super::fleet_state::MemoryFleetStateStore::shared();
        let gateway_a = test_state_with_store(Arc::clone(&store));
        let gateway_b = test_state_with_store(store);
        let register_response = router(gateway_a)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(protocol_registration().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(register_response.status(), StatusCode::OK);
        let register_body = axum::body::to_bytes(register_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let registration: serde_json::Value = serde_json::from_slice(&register_body).unwrap();

        let heartbeat = serde_json::json!({
            "registration_id": registration["registration_id"],
            "instance_id": "627f7e26-348f-4fe6-b9b4-cce6785d17ea",
            "sequence": 1,
            "observed_at": current_rfc3339(),
            "runtime": {"ready": true, "state": "ready"},
            "inventory_generation": 1,
            "capacity": {"active_requests": 1, "max_concurrent_requests": 8}
        });
        let heartbeat_response = router(gateway_b.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/mac-worker-1/heartbeat")
                    .header("content-type", "application/json")
                    .header(
                        LEASE_TOKEN_HEADER,
                        registration["lease_token"].as_str().unwrap(),
                    )
                    .body(axum::body::Body::from(heartbeat.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(heartbeat_response.status(), StatusCode::OK);
        let internal_id = gateway_b
            .registry
            .resolve_worker_id("mac-worker-1")
            .unwrap();
        let snapshot = gateway_b.registry.get_snapshot(internal_id).unwrap();
        assert_eq!(snapshot.protocol_worker_id.as_deref(), Some("mac-worker-1"));
        assert_eq!(snapshot.runtime_ready, Some(true));
        assert_eq!(snapshot.inflight, 1);
    }

    #[tokio::test]
    async fn newer_registration_fences_old_gateway_lease() {
        let store: Arc<dyn FleetStateStore> =
            super::super::fleet_state::MemoryFleetStateStore::shared();
        let gateway_a = test_state_with_store(Arc::clone(&store));
        let gateway_b = test_state_with_store(store);

        let first = router(gateway_a.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(protocol_registration().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let first_body = axum::body::to_bytes(first.into_body(), usize::MAX)
            .await
            .unwrap();
        let first_registration: serde_json::Value = serde_json::from_slice(&first_body).unwrap();

        let second = router(gateway_b)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/register")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(protocol_registration().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::OK);

        let old_heartbeat = serde_json::json!({
            "registration_id": first_registration["registration_id"],
            "instance_id": "627f7e26-348f-4fe6-b9b4-cce6785d17ea",
            "sequence": 1,
            "observed_at": current_rfc3339(),
            "runtime": {"ready": true, "state": "ready"},
            "inventory_generation": 1
        });
        let rejected = router(gateway_a)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/workers/mac-worker-1/heartbeat")
                    .header("content-type", "application/json")
                    .header(
                        LEASE_TOKEN_HEADER,
                        first_registration["lease_token"].as_str().unwrap(),
                    )
                    .body(axum::body::Body::from(old_heartbeat.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(rejected.status(), StatusCode::CONFLICT);
    }
}
