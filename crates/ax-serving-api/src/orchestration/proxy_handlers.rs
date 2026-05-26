use std::collections::{BTreeMap, BTreeSet};
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
use super::registry::{BackendKind, RequestKind, RuntimeKind};
use crate::auth::RequestId;
use crate::project_policy;
use crate::rest::schema::{EmbeddingsInput, InputMessage, MAX_MODEL_ID_BYTES};
use crate::utils::request_meta::{
    audit_actor, default_audit_limit, estimate_chat_prompt_tokens_u32,
    estimate_embedding_input_max_tokens_u32, estimate_text_prompt_tokens_u32,
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
        runtime: Option<String>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        max_tokens: Option<u32>,
        #[serde(default)]
        messages: Vec<InputMessage>,
        #[serde(default)]
        prompt: Option<String>,
        #[serde(default)]
        input: Option<serde_json::Value>,
    }

    let auth_header = req_headers.get(header::AUTHORIZATION);
    let requested_pool = req_headers
        .get("x-ax-worker-pool")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let meta = match serde_json::from_slice::<ProxyRequestMeta>(&body) {
        Ok(meta) => meta,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid JSON body").into_response();
        }
    };
    let model_id = match validate_proxy_model_id(meta.model) {
        Ok(model_id) => model_id,
        Err((status, error)) => {
            return (status, Json(serde_json::json!({ "error": error }))).into_response();
        }
    };
    let backend_hint = match validate_dispatch_hint(meta.runtime.or(meta.backend)) {
        Ok(hint) => hint,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };
    let stream = meta.stream;
    let max_tokens = meta.max_tokens;
    let min_context = if worker_path == "/v1/embeddings" {
        meta.input
            .as_ref()
            .and_then(|input| serde_json::from_value::<EmbeddingsInput>(input.clone()).ok())
            .map(|input| estimate_embedding_input_max_tokens_u32(&input))
    } else if !meta.messages.is_empty() {
        context_requirement_with_generation(
            Some(estimate_chat_prompt_tokens_u32(&meta.messages)),
            max_tokens,
        )
    } else {
        context_requirement_with_generation(
            meta.prompt.as_deref().map(estimate_text_prompt_tokens_u32),
            max_tokens,
        )
    };

    let resolved_policy =
        match project_policy::enforce(&req_headers, &model_id, max_tokens, &layer.project_policy) {
            Ok(v) => v,
            Err(resp) => return resp.into_response(),
        };
    let policy_pool = resolved_policy
        .as_ref()
        .and_then(|v| v.worker_pool.as_deref());
    let preferred_pool = policy_pool.or(requested_pool);
    let require_preferred_pool = policy_pool.is_some();

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
            require_preferred_pool,
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

fn validate_proxy_model_id(model: Option<String>) -> Result<String, (StatusCode, String)> {
    let Some(model) = model else {
        return Err((StatusCode::BAD_REQUEST, "missing field: model".to_string()));
    };
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "model must not be empty".to_string(),
        ));
    }
    if model != trimmed {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "model contains unsupported whitespace".to_string(),
        ));
    }
    if model.len() > MAX_MODEL_ID_BYTES {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("model exceeds max length of {MAX_MODEL_ID_BYTES}"),
        ));
    }
    if !model
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "model must be alphanumeric with '-', '_', or '.'".to_string(),
        ));
    }
    Ok(model)
}

fn context_requirement_with_generation(
    prompt_tokens: Option<u32>,
    max_tokens: Option<u32>,
) -> Option<u32> {
    match (prompt_tokens, max_tokens) {
        (Some(prompt_tokens), Some(max_tokens)) => Some(prompt_tokens.saturating_add(max_tokens)),
        (Some(prompt_tokens), None) => Some(prompt_tokens),
        (None, Some(max_tokens)) => Some(max_tokens),
        (None, None) => None,
    }
}

fn validate_dispatch_hint(hint: Option<String>) -> Result<Option<String>, String> {
    let Some(raw) = hint else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }

    if BackendKind::parse(trimmed) != BackendKind::Auto
        || RuntimeKind::parse(trimmed) != RuntimeKind::Unknown
    {
        return Ok(Some(trimmed.to_ascii_lowercase()));
    }

    Err(format!(
        "invalid backend/runtime hint; expected native, ax_engine, llama_cpp, sglang, vllm, or auto but got {trimmed}"
    ))
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
    let runtime_buckets = runtime_fleet_buckets(&workers);
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
            "runtimes": runtime_buckets,
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
    let runtime_buckets = runtime_fleet_buckets(&workers);
    let runtime_diagnostics = runtime_diagnostics(&workers);
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
                "runtimes": runtime_buckets,
            },
            "runtime_diagnostics": runtime_diagnostics,
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
        "runtime_diagnostics": runtime_diagnostics,
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
    let mut runtimes = serde_json::Map::new();

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
        accumulate_fleet_bucket(&mut runtimes, &worker.runtime, worker);
    }

    Json(serde_json::json!({
        "total_workers": workers.len(),
        "eligible_workers": layer.registry.eligible_healthy_count(),
        "pools": pools,
        "node_classes": node_classes,
        "backends": backends,
        "runtimes": runtimes,
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

fn runtime_fleet_buckets(
    workers: &[super::registry::WorkerSnapshot],
) -> serde_json::Map<String, serde_json::Value> {
    let mut runtimes = serde_json::Map::new();
    for worker in workers {
        accumulate_fleet_bucket(&mut runtimes, &worker.runtime, worker);
    }
    runtimes
}

const RUNTIME_ERROR_RATE_WARN_THRESHOLD: f64 = 0.05;
const RUNTIME_KV_PRESSURE_WARN_THRESHOLD: f64 = 0.90;
const RUNTIME_BATCH_PRESSURE_WARN_THRESHOLD: f64 = 0.90;

#[derive(Default)]
struct RuntimeDiagnostic {
    workers: usize,
    healthy: usize,
    unhealthy: usize,
    draining: usize,
    eligible: usize,
    total_inflight: usize,
    total_active_sequences: usize,
    total_queue_depth: usize,
    max_error_rate: f64,
    models: BTreeSet<String>,
    model_inventory: Vec<serde_json::Value>,
    hardware_classes: BTreeMap<String, usize>,
    node_classes: BTreeMap<String, usize>,
    worker_pools: BTreeMap<String, usize>,
    runtime_modes: BTreeMap<String, usize>,
    supported_operations: BTreeSet<String>,
    runtime_endpoints: BTreeSet<String>,
    missing_runtime_endpoint_workers: Vec<String>,
    unhealthy_workers: Vec<String>,
    draining_workers: Vec<String>,
    compatibility_workers: Vec<String>,
    unknown_runtime_workers: Vec<String>,
    empty_model_inventory_workers: Vec<String>,
    unexpected_hardware_class_workers: Vec<String>,
    high_error_rate_workers: Vec<String>,
    queue_backlog_workers: Vec<String>,
    high_kv_pressure_workers: Vec<String>,
    high_batch_pressure_workers: Vec<String>,
}

impl RuntimeDiagnostic {
    fn observe(&mut self, worker: &super::registry::WorkerSnapshot) {
        self.workers += 1;
        if worker.health == "healthy" {
            self.healthy += 1;
        } else {
            self.unhealthy += 1;
            self.unhealthy_workers.push(worker.id.to_string());
        }
        if worker.drain {
            self.draining += 1;
            self.draining_workers.push(worker.id.to_string());
        }
        if worker.health == "healthy" && !worker.drain {
            self.eligible += 1;
        }
        self.total_inflight += worker.inflight;
        self.total_active_sequences += worker.active_sequences;
        self.total_queue_depth += worker.queue_depth;
        self.max_error_rate = self.max_error_rate.max(worker.error_rate);
        if worker.error_rate >= RUNTIME_ERROR_RATE_WARN_THRESHOLD {
            self.high_error_rate_workers.push(worker.id.to_string());
        }
        if worker.queue_depth >= worker.max_inflight.max(1) {
            self.queue_backlog_workers.push(worker.id.to_string());
        }
        if worker
            .kv_utilization
            .is_some_and(|value| value >= RUNTIME_KV_PRESSURE_WARN_THRESHOLD)
        {
            self.high_kv_pressure_workers.push(worker.id.to_string());
        }
        if worker
            .batch_utilization
            .is_some_and(|value| value >= RUNTIME_BATCH_PRESSURE_WARN_THRESHOLD)
            || (worker.max_batch_size > 0 && worker.active_batch_size >= worker.max_batch_size)
        {
            self.high_batch_pressure_workers.push(worker.id.to_string());
        }

        if worker.capabilities.is_empty() {
            self.empty_model_inventory_workers
                .push(worker.id.to_string());
        }
        for model in &worker.capabilities {
            self.models.insert(model.clone());
        }
        for model in &worker.model_inventory {
            self.model_inventory.push(serde_json::json!({
                "worker_id": worker.id,
                "model_id": model.id.as_str(),
                "runtime": worker.runtime.as_str(),
                "node_class": worker.node_class.as_deref(),
                "hardware_class": worker.hardware_class.as_deref(),
                "max_context": model.max_context,
                "quantization": model.quantization.as_deref(),
                "artifact_format": model.artifact_format.as_deref(),
                "modalities": &model.modalities,
                "supported_operations": &model.supported_operations,
            }));
        }
        if let Some(hardware_class) = worker.hardware_class.as_deref() {
            increment_count(&mut self.hardware_classes, hardware_class);
            if let Some(expected) = expected_hardware_classes(worker.runtime.as_str())
                && !expected.contains(&hardware_class)
            {
                self.unexpected_hardware_class_workers
                    .push(worker.id.to_string());
            }
        } else if expected_hardware_classes(worker.runtime.as_str()).is_some() {
            self.unexpected_hardware_class_workers
                .push(worker.id.to_string());
        }
        if let Some(node_class) = worker.node_class.as_deref() {
            increment_count(&mut self.node_classes, node_class);
        }
        if let Some(worker_pool) = worker.worker_pool.as_deref() {
            increment_count(&mut self.worker_pools, worker_pool);
        }
        if let Some(runtime_mode) = worker.runtime_mode.as_deref() {
            increment_count(&mut self.runtime_modes, runtime_mode);
        }
        for operation in &worker.supported_operations {
            self.supported_operations.insert(operation.clone());
        }
        if let Some(endpoint) = worker.runtime_endpoint.as_deref() {
            self.runtime_endpoints.insert(endpoint.to_string());
        } else {
            self.missing_runtime_endpoint_workers
                .push(worker.id.to_string());
        }

        if worker.runtime == "unknown" {
            self.unknown_runtime_workers.push(worker.id.to_string());
        }
        if worker.runtime_mode.as_deref() == Some("embedded")
            || (worker.runtime == "ax_engine"
                && worker.runtime_mode.is_none()
                && matches!(
                    worker.backend.as_str(),
                    "native" | "auto" | "llama_cpp" | "mlx"
                ))
        {
            self.compatibility_workers.push(worker.id.to_string());
        }
    }

    fn issues(&self) -> Vec<serde_json::Value> {
        let mut issues = Vec::new();
        if self.eligible == 0 {
            issues.push(serde_json::json!({
                "code": "no_eligible_workers",
                "severity": "error",
                "message": "runtime has no healthy non-draining workers"
            }));
        }
        if !self.unhealthy_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "unhealthy_workers",
                "severity": "warning",
                "workers": self.unhealthy_workers
            }));
        }
        if !self.draining_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "draining_workers",
                "severity": "info",
                "workers": self.draining_workers
            }));
        }
        if !self.unknown_runtime_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "unknown_runtime",
                "severity": "warning",
                "workers": self.unknown_runtime_workers
            }));
        }
        if !self.missing_runtime_endpoint_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "missing_runtime_endpoint",
                "severity": "warning",
                "workers": self.missing_runtime_endpoint_workers
            }));
        }
        if !self.empty_model_inventory_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "empty_model_inventory",
                "severity": "warning",
                "workers": self.empty_model_inventory_workers
            }));
        }
        if !self.unexpected_hardware_class_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "unexpected_hardware_class",
                "severity": "warning",
                "workers": self.unexpected_hardware_class_workers
            }));
        }
        if !self.high_error_rate_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "high_runtime_error_rate",
                "severity": "warning",
                "workers": self.high_error_rate_workers,
                "threshold": RUNTIME_ERROR_RATE_WARN_THRESHOLD
            }));
        }
        if !self.queue_backlog_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "runtime_queue_backlog",
                "severity": "warning",
                "workers": self.queue_backlog_workers
            }));
        }
        if !self.high_kv_pressure_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "high_runtime_kv_pressure",
                "severity": "warning",
                "workers": self.high_kv_pressure_workers,
                "threshold": RUNTIME_KV_PRESSURE_WARN_THRESHOLD
            }));
        }
        if !self.high_batch_pressure_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "high_runtime_batch_pressure",
                "severity": "info",
                "workers": self.high_batch_pressure_workers,
                "threshold": RUNTIME_BATCH_PRESSURE_WARN_THRESHOLD
            }));
        }
        if !self.compatibility_workers.is_empty() {
            issues.push(serde_json::json!({
                "code": "embedded_compatibility_path",
                "severity": "info",
                "workers": self.compatibility_workers
            }));
        }
        issues
    }

    fn recommended_actions(&self, runtime: &str) -> Vec<serde_json::Value> {
        let mut actions = Vec::new();
        if self.eligible == 0 {
            actions.push(serde_json::json!({
                "action": "restore_runtime_capacity",
                "runtime": runtime,
                "priority": "high",
                "reason": "runtime has no eligible workers",
                "operator_hint": "Start or recover at least one healthy non-draining runtime node for this runtime."
            }));
        }
        if !self.unhealthy_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "replace_unhealthy_workers",
                "runtime": runtime,
                "priority": "high",
                "worker_ids": self.unhealthy_workers,
                "suggested_commands": worker_replacement_commands(&self.unhealthy_workers),
                "operator_hint": "Drain or remove unhealthy workers, restart the runtime node, then verify registration and heartbeat."
            }));
        }
        if !self.draining_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "complete_drain_when_idle",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.draining_workers,
                "suggested_commands": worker_drain_complete_commands(&self.draining_workers),
                "operator_hint": "Wait for inflight requests to reach zero, then call drain-complete before replacement."
            }));
        }
        if !self.missing_runtime_endpoint_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "fix_runtime_endpoint_registration",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.missing_runtime_endpoint_workers,
                "operator_hint": "Restart the adapter with AXS_NODE_RUNTIME_URL or AXS_WORKER_RUNTIME_ENDPOINT set."
            }));
        }
        if !self.empty_model_inventory_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "refresh_model_inventory",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.empty_model_inventory_workers,
                "operator_hint": "Check the runtime /v1/models endpoint and restart the adapter after the model is loaded."
            }));
        }
        if !self.unknown_runtime_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "fix_runtime_class",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.unknown_runtime_workers,
                "operator_hint": "Register the worker with runtime ax_engine or vllm."
            }));
        }
        if !self.unexpected_hardware_class_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "fix_hardware_class",
                "runtime": runtime,
                "priority": "medium",
                "worker_ids": self.unexpected_hardware_class_workers,
                "expected_hardware_classes": expected_hardware_classes(runtime).unwrap_or(&[]),
                "operator_hint": "Restart the adapter with the hardware class expected for this runtime."
            }));
        }
        if !self.high_error_rate_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "investigate_runtime_errors",
                "runtime": runtime,
                "priority": "high",
                "worker_ids": self.high_error_rate_workers,
                "suggested_commands": worker_inspection_commands(&self.high_error_rate_workers),
                "operator_hint": "Check runtime logs and recent failed requests before returning these workers to normal routing."
            }));
        }
        if !self.queue_backlog_workers.is_empty()
            || !self.high_kv_pressure_workers.is_empty()
            || !self.high_batch_pressure_workers.is_empty()
        {
            let pressure_workers = unique_worker_ids([
                &self.queue_backlog_workers,
                &self.high_kv_pressure_workers,
                &self.high_batch_pressure_workers,
            ]);
            actions.push(serde_json::json!({
                "action": "relieve_runtime_pressure",
                "runtime": runtime,
                "priority": "medium",
                "queue_backlog_worker_ids": self.queue_backlog_workers,
                "high_kv_pressure_worker_ids": self.high_kv_pressure_workers,
                "high_batch_pressure_worker_ids": self.high_batch_pressure_workers,
                "suggested_commands": worker_replacement_commands(&pressure_workers),
                "operator_hint": "Reduce admission pressure, add runtime capacity, or drain and replace overloaded nodes."
            }));
        }
        if !self.compatibility_workers.is_empty() {
            actions.push(serde_json::json!({
                "action": "migrate_embedded_compatibility_path",
                "runtime": runtime,
                "priority": "low",
                "worker_ids": self.compatibility_workers,
                "suggested_commands": [
                    "ax-serving status --diagnostics --url <gateway-url>",
                    "AXS_EMBEDDED_RUNTIME_POLICY=deny ax-serving-api"
                ],
                "operator_hint": "Move inference to ax-runtime-agent plus ax-engine or vLLM, then set AXS_EMBEDDED_RUNTIME_POLICY=deny in production."
            }));
        }
        actions
    }

    fn to_json(&self, runtime: &str) -> serde_json::Value {
        serde_json::json!({
            "workers": self.workers,
            "healthy": self.healthy,
            "unhealthy": self.unhealthy,
            "draining": self.draining,
            "eligible": self.eligible,
            "total_inflight": self.total_inflight,
            "total_active_sequences": self.total_active_sequences,
            "total_queue_depth": self.total_queue_depth,
            "max_error_rate": self.max_error_rate,
            "models": self.models,
            "model_inventory": self.model_inventory,
            "hardware_classes": self.hardware_classes,
            "node_classes": self.node_classes,
            "worker_pools": self.worker_pools,
            "runtime_modes": self.runtime_modes,
            "supported_operations": self.supported_operations,
            "runtime_endpoints": self.runtime_endpoints,
            "unhealthy_workers": self.unhealthy_workers,
            "draining_workers": self.draining_workers,
            "high_error_rate_workers": self.high_error_rate_workers,
            "queue_backlog_workers": self.queue_backlog_workers,
            "high_kv_pressure_workers": self.high_kv_pressure_workers,
            "high_batch_pressure_workers": self.high_batch_pressure_workers,
            "issues": self.issues(),
            "recommended_actions": self.recommended_actions(runtime),
            "runtime_guidance": runtime_guidance(runtime),
        })
    }
}

fn unique_worker_ids<'a>(groups: impl IntoIterator<Item = &'a Vec<String>>) -> Vec<String> {
    groups
        .into_iter()
        .flat_map(|group| group.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn worker_inspection_commands(worker_ids: &[String]) -> Vec<String> {
    worker_ids
        .iter()
        .map(|id| format!("ax-serving workers get {id} --url <gateway-url>"))
        .collect()
}

fn worker_drain_complete_commands(worker_ids: &[String]) -> Vec<String> {
    worker_ids
        .iter()
        .map(|id| format!("ax-serving workers drain {id} --complete-when-idle --url <gateway-url>"))
        .collect()
}

fn worker_replacement_commands(worker_ids: &[String]) -> Vec<String> {
    worker_ids
        .iter()
        .flat_map(|id| {
            [
                format!("ax-serving workers drain {id} --complete-when-idle --url <gateway-url>"),
                "start or restart the replacement ax-runtime-agent node".to_string(),
                "ax-serving status --diagnostics --url <gateway-url>".to_string(),
            ]
        })
        .collect()
}

fn expected_hardware_classes(runtime: &str) -> Option<&'static [&'static str]> {
    match runtime {
        "ax_engine" => Some(&["mac"]),
        "vllm" => Some(&["pc-cuda", "thor"]),
        _ => None,
    }
}

fn runtime_guidance(runtime: &str) -> serde_json::Value {
    match runtime {
        "ax_engine" => serde_json::json!({
            "runtime_owner": "ax-engine",
            "expected_hardware_classes": ["mac"],
            "adapter": "ax-runtime-agent",
            "required_registration": {
                "runtime": "ax_engine",
                "hardware_class": "mac"
            },
            "operator_checks": [
                "runtime endpoint exposes /health",
                "runtime endpoint exposes /v1/models",
                "adapter reports ax_runtime_* metrics when available",
                "embedded compatibility workers should be migrated before production"
            ]
        }),
        "vllm" => serde_json::json!({
            "runtime_owner": "vLLM",
            "expected_hardware_classes": ["pc-cuda", "thor"],
            "adapter": "ax-runtime-agent",
            "required_registration": {
                "runtime": "vllm",
                "hardware_class": "pc-cuda or thor"
            },
            "operator_checks": [
                "vLLM OpenAI-compatible endpoint exposes /health",
                "vLLM OpenAI-compatible endpoint exposes /v1/models",
                "adapter reports runtime endpoint and supported operations",
                "PC CUDA and Thor placement should be represented by hardware_class and worker_pool"
            ]
        }),
        _ => serde_json::json!({
            "runtime_owner": "unknown",
            "expected_hardware_classes": [],
            "adapter": "unknown",
            "operator_checks": [
                "register the node with runtime ax_engine or vllm",
                "verify the adapter follows the AX Serving node contract"
            ]
        }),
    }
}

fn increment_count(counts: &mut BTreeMap<String, usize>, key: &str) {
    *counts.entry(key.to_string()).or_default() += 1;
}

fn runtime_diagnostics(workers: &[super::registry::WorkerSnapshot]) -> serde_json::Value {
    let mut diagnostics = BTreeMap::<String, RuntimeDiagnostic>::new();
    for worker in workers {
        diagnostics
            .entry(worker.runtime.clone())
            .or_default()
            .observe(worker);
    }

    let mut runtimes = serde_json::Map::new();
    let mut issues = Vec::new();
    let mut recommended_actions = Vec::new();
    if workers.is_empty() {
        issues.push(serde_json::json!({
            "code": "no_workers_registered",
            "severity": "error",
            "message": "no runtime nodes are registered"
        }));
        recommended_actions.push(serde_json::json!({
            "action": "register_runtime_nodes",
            "priority": "high",
            "reason": "no runtime nodes are registered",
            "operator_hint": "Start ax-serving-api and register ax-runtime-agent nodes for ax_engine or vllm."
        }));
    }
    for (runtime, diagnostic) in diagnostics {
        let runtime_issues = diagnostic.issues();
        for issue in &runtime_issues {
            issues.push(serde_json::json!({
                "runtime": runtime,
                "code": issue["code"],
                "severity": issue["severity"],
            }));
        }
        recommended_actions.extend(diagnostic.recommended_actions(&runtime));
        runtimes.insert(runtime.clone(), diagnostic.to_json(&runtime));
    }

    serde_json::json!({
        "runtimes": runtimes,
        "issues": issues,
        "recommended_actions": recommended_actions,
    })
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
