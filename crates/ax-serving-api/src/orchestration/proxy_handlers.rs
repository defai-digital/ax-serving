use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::{
    Json,
    body::{Body, BodyDataStream, Bytes},
    extract::{ConnectInfo, Extension, Path, Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{Html, IntoResponse},
};
use serde::Deserialize;
use tracing::warn;

use super::OrchestratorLayer;
use super::queue::AcquireResult;
use super::registry::RequestKind;
use crate::auth::RequestId;
use crate::project_policy;
use crate::rest::schema::InputMessage;
use crate::utils::request_meta::{
    audit_actor, default_audit_limit, estimate_chat_prompt_tokens_u32,
    estimate_text_prompt_tokens_u32,
};

// ── Shared inference proxy ────────────────────────────────────────────────────

async fn proxy_inference(
    layer: Arc<OrchestratorLayer>,
    peer_addr: Option<SocketAddr>,
    req_headers: HeaderMap,
    body: Bytes,
    worker_path: &'static str,
) -> axum::response::Response {
    #[derive(Deserialize)]
    struct ProxyRequestMeta {
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        backend: Option<String>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        max_tokens: Option<u32>,
        #[serde(default)]
        messages: Vec<InputMessage>,
        #[serde(default)]
        prompt: Option<String>,
    }

    let auth_header = req_headers.get(header::AUTHORIZATION);
    let requested_pool = req_headers
        .get("x-ax-worker-pool")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let (model_id, backend_hint, stream, max_tokens, min_context) =
        match serde_json::from_slice::<ProxyRequestMeta>(&body) {
            Ok(v) => (
                v.model.unwrap_or_else(|| "default".to_string()),
                v.backend,
                v.stream,
                v.max_tokens,
                if !v.messages.is_empty() {
                    Some(estimate_chat_prompt_tokens_u32(&v.messages))
                } else {
                    v.prompt.as_deref().map(estimate_text_prompt_tokens_u32)
                },
            ),
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "invalid JSON body").into_response();
            }
        };

    let resolved_policy =
        match project_policy::enforce(&req_headers, &model_id, max_tokens, &layer.project_policy) {
            Ok(v) => v,
            Err(resp) => return resp.into_response(),
        };
    let preferred_pool = resolved_policy
        .as_ref()
        .and_then(|v| v.worker_pool.as_deref())
        .or(requested_pool);

    // Admission control: acquire a queue slot before dispatching.
    let permit = match layer
        .queue
        .acquire(fairness_client_key(&req_headers, peer_addr))
        .await
    {
        AcquireResult::Permit(p) => p,

        AcquireResult::Rejected => {
            // 429 with X-Queue-Depth + Retry-After (PRD §FR-3.3)
            let queued = layer.queue.queued();
            return axum::response::Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header("content-type", "text/plain; charset=utf-8")
                .header("x-queue-depth", queued.to_string())
                .header("retry-after", layer.retry_after_secs.to_string())
                .body(Body::from("request rejected: concurrency limit exceeded"))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
        }

        AcquireResult::Shed => {
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                "request shed: queue overload",
            )
                .into_response();
            resp.headers_mut().insert(
                HeaderName::from_static("x-reason"),
                HeaderValue::from_static("request_shed"),
            );
            return resp;
        }

        AcquireResult::Timeout => {
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                "request timed out waiting for a queue slot",
            )
                .into_response();
            resp.headers_mut().insert(
                HeaderName::from_static("x-reason"),
                HeaderValue::from_static("queue_timeout"),
            );
            return resp;
        }
    };

    let mut resp = layer
        .dispatcher
        .forward_kind(
            &layer.registry,
            layer.policy.as_ref(),
            &model_id,
            match worker_path {
                "/v1/embeddings" => RequestKind::Embedding,
                _ => RequestKind::Llm,
            },
            backend_hint.as_deref(),
            min_context,
            stream,
            preferred_pool,
            worker_path,
            body,
            auth_header,
        )
        .await;

    // Add X-Reason header for dispatcher-level errors (PRD §FR-3.3).
    if !resp.headers().contains_key("x-reason") {
        let reason: Option<&'static str> = match resp.status() {
            StatusCode::SERVICE_UNAVAILABLE => Some("no_eligible_worker"),
            StatusCode::BAD_GATEWAY => Some("worker_crash"),
            _ => None,
        };
        if let Some(r) = reason {
            resp.headers_mut().insert(
                HeaderName::from_static("x-reason"),
                HeaderValue::from_static(r),
            );
        }
    }

    // For streaming responses the body is delivered lazily after this handler
    // returns.  Carry the permit inside the body stream so the global
    // concurrency slot is held until the stream ends or the client disconnects.
    // This matches the non-streaming path semantics: forward() buffers the full
    // body before returning, so the permit there is held for the entire
    // inference duration as well.
    if stream {
        let (parts, old_body) = resp.into_parts();
        let guarded = futures::stream::unfold(
            (old_body.into_data_stream(), Some(permit)),
            |(mut data_stream, permit): (BodyDataStream, Option<_>)| async move {
                use futures::StreamExt as _;
                match data_stream.next().await {
                    Some(chunk) => Some((chunk, (data_stream, permit))),
                    None => {
                        drop(permit);
                        None
                    }
                }
            },
        );
        axum::response::Response::from_parts(parts, Body::from_stream(guarded))
    } else {
        drop(permit);
        resp
    }
}

fn fairness_client_key(headers: &HeaderMap, peer_addr: Option<SocketAddr>) -> String {
    if let Some(auth) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        auth.hash(&mut hasher);
        return format!("auth:{:016x}", hasher.finish());
    }
    if let Some(peer_ip) = peer_addr.map(|addr| addr.ip()) {
        return format!("ip:{peer_ip}");
    }
    "anonymous".to_string()
}

#[derive(Deserialize)]
pub(super) struct AuditQuery {
    #[serde(default = "default_audit_limit")]
    limit: usize,
}

pub(super) fn orchestrator_startup_report_value(
    layer: &Arc<OrchestratorLayer>,
) -> serde_json::Value {
    serde_json::json!({
        "service": "orchestrator",
        "status": "ok",
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "license": layer.license.to_json(),
        "runtime": {
            "host": layer.config.host,
            "port": layer.config.port,
            "internal_bind_addr": layer.config.internal_bind_addr,
            "allowed_node_cidrs": layer.config.allowed_node_cidrs,
            "internal_port": layer.config.internal_port,
            "dispatch_policy": layer.config.dispatch_policy,
            "worker_heartbeat_ms": layer.config.worker_heartbeat_ms,
            "worker_ttl_ms": layer.config.worker_ttl_ms,
            "request_timeout_secs": layer.config.request_timeout_secs,
            "global_queue_max": layer.config.global_queue_max,
            "global_queue_depth": layer.config.global_queue_depth,
            "global_queue_wait_ms": layer.config.global_queue_wait_ms,
        },
        "dispatch_runtime": {},
        "project_policy": project_policy::summary_json(&layer.project_policy),
        "governance": {
            "project_policy_enabled": layer.project_policy.enabled,
        }
    })
}

// ── Route handlers ────────────────────────────────────────────────────────────

pub(super) async fn proxy_chat_completions(
    State(layer): State<Arc<OrchestratorLayer>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(
        layer,
        Some(peer_addr),
        headers,
        body,
        "/v1/chat/completions",
    )
    .await
}

pub(super) async fn proxy_completions(
    State(layer): State<Arc<OrchestratorLayer>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(layer, Some(peer_addr), headers, body, "/v1/completions").await
}

pub(super) async fn proxy_embeddings(
    State(layer): State<Arc<OrchestratorLayer>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(layer, Some(peer_addr), headers, body, "/v1/embeddings").await
}

pub(super) async fn proxy_models(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
    let workers = layer.registry.list_all();
    let mut models: Vec<String> = workers
        .iter()
        // Mirror dispatch eligibility: only healthy, non-draining workers.
        // Unhealthy workers may recover but are not currently routable, so
        // advertising their models here would produce 503s on inference.
        .filter(|w| !w.drain && w.health == "healthy")
        .flat_map(|w| w.capability_descriptor.models.iter().cloned())
        .collect();
    models.sort_unstable();
    models.dedup();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(serde_json::json!({
        "object": "list",
        "data": models.iter().map(|id| serde_json::json!({
            "id": id,
            "object": "model",
            "created": now,
            "owned_by": "ax-serving",
        })).collect::<Vec<_>>()
    }))
}

pub(super) async fn proxy_health(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    // "ok" only when at least one worker can actually accept requests:
    // healthy AND not draining. A fully-draining pool shows as "degraded" even
    // if all workers are technically Healthy, because the dispatcher would
    // return 503 for every request.
    let eligible = layer.registry.eligible_healthy_count();
    let status = if eligible > 0 { "ok" } else { "degraded" };
    let qm = &layer.queue.metrics;
    Json(serde_json::json!({
        "status": status,
        "workers": {
            "total": healthy + unhealthy,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "draining": draining,
            "eligible": eligible,
        },
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        }
    }))
}

pub(super) async fn proxy_metrics(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    let workers = layer.registry.list_all();
    let total_inflight: usize = workers.iter().map(|w| w.inflight).sum();
    let qm = &layer.queue.metrics;

    Json(serde_json::json!({
        "mode": "direct",
        "policy": layer.config.dispatch_policy,
        "workers": {
            "healthy": healthy,
            "unhealthy": unhealthy,
            "draining": draining,
        },
        "total_inflight": total_inflight,
        "reroute_total": layer.dispatcher.reroutes(),
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "permit_total": qm.permit_total.load(Ordering::Relaxed),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        },
        "worker_detail": workers,
    }))
}

pub(super) async fn proxy_admin_status(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<axum::extract::Extension<crate::auth::RequestId>>,
) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    let workers = layer.registry.list_all();
    let total_workers = workers.len();
    let total_inflight: usize = workers.iter().map(|w| w.inflight).sum();
    let total_active_sequences: usize = workers.iter().map(|w| w.active_sequences).sum();
    let eligible = layer.registry.eligible_healthy_count();
    let qm = &layer.queue.metrics;

    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "mode": "direct",
        "status": if eligible > 0 { "ok" } else { "degraded" },
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "dispatch_policy": layer.config.dispatch_policy,
        "license": layer.license.to_json(),
        "workers": {
            "total": total_workers,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "draining": draining,
            "eligible": eligible,
            "total_inflight": total_inflight,
            "total_active_sequences": total_active_sequences,
        },
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "permit_total": qm.permit_total.load(Ordering::Relaxed),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        },
        "dispatcher": {
            "reroute_total": layer.dispatcher.reroutes(),
            "request_timeout_secs": layer.config.request_timeout_secs,
            "retry_after_secs": layer.retry_after_secs,
        }
    }))
}

pub(super) async fn proxy_admin_startup_report(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(orchestrator_startup_report_value(&layer))
}

pub(super) async fn proxy_admin_diagnostics(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
) -> impl IntoResponse {
    let (healthy, unhealthy, draining) = layer.registry.counts();
    let workers = layer.registry.list_all();
    let total_inflight: usize = workers.iter().map(|w| w.inflight).sum();
    let total_active_sequences: usize = workers.iter().map(|w| w.active_sequences).sum();
    let eligible = layer.registry.eligible_healthy_count();
    let qm = &layer.queue.metrics;
    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "startup_report": orchestrator_startup_report_value(&layer),
        "health": {
            "status": if eligible > 0 { "ok" } else { "degraded" },
            "workers": {
                "total": healthy + unhealthy,
                "healthy": healthy,
                "unhealthy": unhealthy,
                "draining": draining,
                "eligible": eligible,
            },
            "queue": {
                "active": layer.queue.active(),
                "queued": layer.queue.queued(),
                "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
                "shed_total": qm.shed_total.load(Ordering::Relaxed),
                "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
            }
        },
        "metrics": {
            "mode": "direct",
            "policy": layer.config.dispatch_policy,
            "workers": {
                "healthy": healthy,
                "unhealthy": unhealthy,
                "draining": draining,
                "eligible": eligible,
                "total_inflight": total_inflight,
                "total_active_sequences": total_active_sequences,
            },
            "reroute_total": layer.dispatcher.reroutes(),
            "queue": {
                "active": layer.queue.active(),
                "queued": layer.queue.queued(),
                "permit_total": qm.permit_total.load(Ordering::Relaxed),
                "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
                "shed_total": qm.shed_total.load(Ordering::Relaxed),
                "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
            }
        },
        "workers": workers,
        "audit_tail": layer.audit.tail(50),
    }))
}

pub(super) async fn proxy_admin_policy(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(project_policy::summary_json(&layer.project_policy))
}

pub(super) async fn proxy_admin_audit(
    State(layer): State<Arc<OrchestratorLayer>>,
    Query(query): Query<AuditQuery>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "events": layer.audit.tail(query.limit.clamp(1, 200)),
    }))
}

pub(super) async fn proxy_admin_fleet(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    let workers = layer.registry.list_all();
    let mut pools = serde_json::Map::new();
    let mut node_classes = serde_json::Map::new();
    let mut backends = serde_json::Map::new();

    for worker in &workers {
        accumulate_fleet_bucket(
            &mut pools,
            worker.worker_pool.as_deref().unwrap_or("default"),
            worker,
        );
        accumulate_fleet_bucket(
            &mut node_classes,
            worker.node_class.as_deref().unwrap_or("unknown"),
            worker,
        );
        accumulate_fleet_bucket(&mut backends, &worker.backend, worker);
    }

    Json(serde_json::json!({
        "total_workers": workers.len(),
        "eligible_workers": layer.registry.eligible_healthy_count(),
        "pools": pools,
        "node_classes": node_classes,
        "backends": backends,
        "workers": workers,
    }))
}

fn accumulate_fleet_bucket(
    buckets: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    worker: &super::registry::WorkerSnapshot,
) {
    let entry = buckets.entry(key.to_string()).or_insert_with(|| {
        serde_json::json!({
            "workers": 0usize,
            "healthy": 0usize,
            "draining": 0usize,
            "eligible": 0usize,
            "total_inflight": 0usize,
            "total_active_sequences": 0usize,
            "total_queue_depth": 0usize,
            "max_error_rate": 0.0_f64,
        })
    });
    if let Some(obj) = entry.as_object_mut() {
        increment_bucket(obj, "workers", 1_u64);
        if worker.health == "healthy" {
            increment_bucket(obj, "healthy", 1_u64);
        }
        if worker.drain {
            increment_bucket(obj, "draining", 1_u64);
        }
        if worker.health == "healthy" && !worker.drain {
            increment_bucket(obj, "eligible", 1_u64);
        }
        increment_bucket(obj, "total_inflight", worker.inflight as u64);
        increment_bucket(
            obj,
            "total_active_sequences",
            worker.active_sequences as u64,
        );
        increment_bucket(obj, "total_queue_depth", worker.queue_depth as u64);
        let current_max = obj
            .get("max_error_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        if worker.error_rate > current_max {
            obj.insert(
                "max_error_rate".to_string(),
                serde_json::Value::from(worker.error_rate),
            );
        }
    } else {
        warn!(key = %key, "unexpected non-object fleet bucket encountered");
    }
}

fn increment_bucket(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    amount: impl Into<u64>,
) {
    let amount = amount.into();
    let current = obj.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    obj.insert(key.to_string(), serde_json::json!(current + amount));
}

pub(super) async fn proxy_list_workers(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "workers": layer.registry.list_all(),
    }))
}

pub(super) async fn proxy_get_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let Some(id) = WorkerId::parse(&id_str) else {
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    match layer.registry.get_snapshot(id) {
        Some(worker) => Json(worker).into_response(),
        None => (StatusCode::NOT_FOUND, "worker not found").into_response(),
    }
}

pub(super) async fn proxy_drain_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let actor = audit_actor(req_id);
    let Some(id) = WorkerId::parse(&id_str) else {
        layer.audit.record(
            actor,
            "worker_drain",
            "worker",
            Some(id_str),
            "error",
            Some(serde_json::json!({"error": "invalid worker id"})),
        );
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    if !layer.registry.mark_drain(id) {
        layer.audit.record(
            actor,
            "worker_drain",
            "worker",
            Some(id.to_string()),
            "error",
            Some(serde_json::json!({"error": "worker not found"})),
        );
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.audit.record(
        actor,
        "worker_drain",
        "worker",
        Some(id.to_string()),
        "ok",
        None,
    );
    tracing::info!(%id, "worker marked for drain via public API");
    StatusCode::OK.into_response()
}

pub(super) async fn proxy_drain_complete_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let actor = audit_actor(req_id);
    let Some(id) = WorkerId::parse(&id_str) else {
        layer.audit.record(
            actor,
            "worker_drain_complete",
            "worker",
            Some(id_str),
            "error",
            Some(serde_json::json!({"error": "invalid worker id"})),
        );
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    if layer.registry.get_snapshot(id).is_none() {
        layer.audit.record(
            actor,
            "worker_drain_complete",
            "worker",
            Some(id.to_string()),
            "error",
            Some(serde_json::json!({"error": "worker not found"})),
        );
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.registry.evict(id);
    layer.audit.record(
        actor,
        "worker_drain_complete",
        "worker",
        Some(id.to_string()),
        "ok",
        None,
    );
    tracing::info!(%id, "worker drain complete via public API");
    StatusCode::NO_CONTENT.into_response()
}

pub(super) async fn proxy_delete_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    use super::registry::WorkerId;
    let actor = audit_actor(req_id);
    let Some(id) = WorkerId::parse(&id_str) else {
        layer.audit.record(
            actor,
            "worker_delete",
            "worker",
            Some(id_str),
            "error",
            Some(serde_json::json!({"error": "invalid worker id"})),
        );
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    // mark_drain returns false when the worker does not exist.
    if !layer.registry.mark_drain(id) {
        layer.audit.record(
            actor,
            "worker_delete",
            "worker",
            Some(id.to_string()),
            "error",
            Some(serde_json::json!({"error": "worker not found"})),
        );
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.registry.evict(id);
    layer.audit.record(
        actor,
        "worker_delete",
        "worker",
        Some(id.to_string()),
        "ok",
        None,
    );
    tracing::info!(%id, "worker force-removed via public API");
    StatusCode::NO_CONTENT.into_response()
}

pub(super) async fn proxy_dashboard() -> impl IntoResponse {
    Html(include_str!("../dashboard.html"))
}

pub(super) async fn proxy_get_license(
    State(layer): State<Arc<OrchestratorLayer>>,
) -> impl IntoResponse {
    Json(layer.license.to_json())
}

pub(super) async fn proxy_set_license(
    State(layer): State<Arc<OrchestratorLayer>>,
    req_id: Option<Extension<RequestId>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let Some(key) = body.get("key").and_then(|v| v.as_str()) else {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "missing field: key"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "missing field: key"})),
        )
            .into_response();
    };
    let key = key.trim().to_string();
    if key.is_empty() {
        layer.audit.record(
            audit_actor(req_id),
            "license_set",
            "license",
            None,
            "error",
            Some(serde_json::json!({"error": "key must not be empty"})),
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "key must not be empty"})),
        )
            .into_response();
    }
    match layer.license.set_key(key) {
        Ok(()) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "ok",
                None,
            );
            Json(layer.license.to_json()).into_response()
        }
        Err(e) => {
            layer.audit.record(
                audit_actor(req_id),
                "license_set",
                "license",
                None,
                "error",
                Some(serde_json::json!({"error": e.to_string()})),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::fairness_client_key;

    #[test]
    fn fairness_client_key_hashes_authorization_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-test-secret"),
        );

        let key = fairness_client_key(&headers, Some("10.0.0.8:443".parse().unwrap()));
        assert!(key.starts_with("auth:"));
        assert!(!key.contains("sk-test-secret"));
    }

    #[test]
    fn fairness_client_key_uses_peer_addr_not_forwarded_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", HeaderValue::from_static("203.0.113.1"));
        headers.insert("x-real-ip", HeaderValue::from_static("203.0.113.2"));

        let key = fairness_client_key(&headers, Some("10.0.0.9:1234".parse().unwrap()));
        assert_eq!(key, "ip:10.0.0.9");
    }
}
