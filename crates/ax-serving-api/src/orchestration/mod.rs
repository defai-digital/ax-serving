//! Multi-worker orchestration layer (ADR-012).
//!
//! # Modes
//!
//! - **direct** (default): orchestrator proxies requests directly to workers
//!   over loopback HTTP.  No external dependencies.
//! - **nats** (M4): orchestrator publishes to JetStream; workers subscribe.
//!
//! # Architecture
//!
//! ```text
//! OrchestratorLayer
//!   ├── WorkerRegistry   — worker identity, health, TTL
//!   ├── DispatchPolicy   — worker selection algorithm
//!   ├── DirectDispatcher — HTTP reverse proxy to selected worker
//!   └── GlobalQueue      — admission control + concurrency cap
//!
//! start_orchestrator()
//!   ├── public Axum router  :{orchestrator_port}  →  proxy /v1/* to workers
//!   ├── internal Axum router:{internal_port}      →  /internal/workers/*
//!   └── HealthTicker (tokio task)
//! ```

pub mod direct;
pub mod health_ticker;
pub mod internal_routes;
#[cfg(feature = "nats-dispatch")]
pub mod nats;
#[cfg(feature = "nats-dispatch")]
pub mod nats_worker;
pub mod policy;
pub mod queue;
pub mod registry;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use axum::{
    Json, Router,
    body::{Body, BodyDataStream, Bytes},
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    middleware,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::Deserialize;
use tokio::sync::watch;
use tracing::info;

use self::direct::DirectDispatcher;
use self::health_ticker::HealthTicker;
use self::internal_routes::{InternalState, internal_auth_middleware, router as internal_router};
use self::policy::DispatchPolicy;
use self::queue::{AcquireResult, GlobalQueue, GlobalQueueConfig, OverloadPolicy};
use self::registry::WorkerRegistry;
use crate::license::LicenseState;

pub use crate::config::{LicenseConfig, OrchestratorConfig};

// ── OrchestratorLayer ─────────────────────────────────────────────────────────

/// Shared state for the orchestrator's public router.
pub struct OrchestratorLayer {
    pub registry: WorkerRegistry,
    pub policy: Arc<dyn DispatchPolicy>,
    pub dispatcher: DirectDispatcher,
    pub config: Arc<OrchestratorConfig>,
    pub queue: GlobalQueue,
    /// Value emitted in `Retry-After` header on 429 responses (from config).
    pub retry_after_secs: u64,
    /// Soft license reminder state.
    pub license: Arc<LicenseState>,
}

impl OrchestratorLayer {
    pub fn new(config: OrchestratorConfig, license_config: LicenseConfig) -> Result<Self> {
        let policy = policy::policy_from_str(&config.dispatch_policy)?;
        let retry_after_secs = config.retry_after_secs;
        let pool_max_idle = config.pool_max_idle_per_host;
        let timeout_secs = config.request_timeout_secs;
        let queue_policy = match config.global_queue_policy.to_lowercase().as_str() {
            "shed_oldest" | "shedoldest" => OverloadPolicy::ShedOldest,
            "reject" => OverloadPolicy::Reject,
            _ => OverloadPolicy::Queue,
        };
        let queue_config = GlobalQueueConfig {
            max_concurrent: config.global_queue_max,
            max_queue_depth: config.global_queue_depth,
            wait_ms: config.global_queue_wait_ms,
            overload_policy: queue_policy,
        };
        Ok(Self {
            registry: WorkerRegistry::new(),
            policy: Arc::from(policy),
            dispatcher: DirectDispatcher::new(pool_max_idle, timeout_secs),
            config: Arc::new(config),
            queue: GlobalQueue::new(queue_config),
            retry_after_secs,
            license: LicenseState::new(&license_config),
        })
    }
}

// ── Public proxy router ───────────────────────────────────────────────────────

pub fn proxy_router(layer: Arc<OrchestratorLayer>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(proxy_chat_completions))
        .route("/v1/completions", post(proxy_completions))
        .route("/v1/models", get(proxy_models))
        .route("/health", get(proxy_health))
        .route("/v1/metrics", get(proxy_metrics))
        .route("/dashboard", get(proxy_dashboard))
        .route(
            "/v1/license",
            get(proxy_get_license).post(proxy_set_license),
        )
        .route(
            "/v1/workers/{id}",
            axum::routing::delete(proxy_delete_worker),
        )
        .with_state(layer)
}

// ── Shared inference proxy ────────────────────────────────────────────────────

/// Shared admission-control + dispatch logic for `/v1/chat/completions` and
/// `/v1/completions`.  `worker_path` is the path forwarded to the worker.
///
/// # Response headers (PRD §FR-3.3)
///
/// | Condition                  | Status | Header                      |
/// |----------------------------|--------|-----------------------------|
/// | Queue full (Reject policy) | 429    | X-Queue-Depth, Retry-After  |
/// | ShedOldest eviction        | 503    | X-Reason: request_shed      |
/// | Queue wait timeout         | 503    | X-Reason: queue_timeout     |
/// | No eligible worker / cap   | 503    | X-Reason: no_eligible_worker|
/// | Worker network / 5xx error | 502    | X-Reason: worker_crash      |
async fn proxy_inference(
    layer: Arc<OrchestratorLayer>,
    req_headers: HeaderMap,
    body: Bytes,
    worker_path: &'static str,
) -> axum::response::Response {
    #[derive(Deserialize)]
    struct ProxyRequestMeta {
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        stream: bool,
    }

    let auth_header = req_headers.get(header::AUTHORIZATION);

    let (model_id, stream) = match serde_json::from_slice::<ProxyRequestMeta>(&body) {
        Ok(v) => (v.model.unwrap_or_else(|| "default".to_string()), v.stream),
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid JSON body").into_response();
        }
    };

    // Admission control: acquire a queue slot before dispatching.
    let permit = match layer.queue.acquire().await {
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
        .forward(
            &layer.registry,
            layer.policy.as_ref(),
            &model_id,
            stream,
            worker_path,
            body,
            auth_header,
        )
        .await;

    // Add X-Reason header for dispatcher-level errors (PRD §FR-3.3).
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

// ── Route handlers ────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions` — admission control, extract model, dispatch.
async fn proxy_chat_completions(
    State(layer): State<Arc<OrchestratorLayer>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(layer, headers, body, "/v1/chat/completions").await
}

/// `POST /v1/completions` — same admission control, forward to worker's
/// `/v1/completions` endpoint.
async fn proxy_completions(
    State(layer): State<Arc<OrchestratorLayer>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    proxy_inference(layer, headers, body, "/v1/completions").await
}

/// `GET /v1/models` — aggregate model list across all healthy workers.
async fn proxy_models(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
    let workers = layer.registry.list_all();
    let mut models: Vec<String> = workers
        .iter()
        // Mirror dispatch eligibility: only healthy, non-draining workers.
        // Unhealthy workers may recover but are not currently routable, so
        // advertising their models here would produce 503s on inference.
        .filter(|w| !w.drain && w.health == "healthy")
        .flat_map(|w| w.capabilities.iter().cloned())
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

/// `GET /health` — orchestrator health with worker pool and queue summary.
async fn proxy_health(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
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

/// `GET /v1/metrics` — orchestrator-level metrics including reroutes + queue.
async fn proxy_metrics(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
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

/// `DELETE /v1/workers/{id}` — force-remove a worker (auth-protected).
///
/// One-step remove: marks the worker as draining then evicts it immediately.
/// Intended for accidental registrations or stuck workers reachable from the
/// public proxy (the internal `/internal/workers/{id}` DELETE is loopback-only
/// and not reachable from a browser dashboard).
async fn proxy_delete_worker(
    State(layer): State<Arc<OrchestratorLayer>>,
    axum::extract::Path(id_str): axum::extract::Path<String>,
) -> impl IntoResponse {
    use self::registry::WorkerId;
    let Some(id) = WorkerId::parse(&id_str) else {
        return (StatusCode::BAD_REQUEST, "invalid worker id").into_response();
    };
    // mark_drain returns false when the worker does not exist.
    if !layer.registry.mark_drain(id) {
        return (StatusCode::NOT_FOUND, "worker not found").into_response();
    }
    layer.registry.evict(id);
    tracing::info!(%id, "worker force-removed via public API");
    StatusCode::NO_CONTENT.into_response()
}

/// `GET /dashboard` — embedded monitoring dashboard (no auth required).
async fn proxy_dashboard() -> impl IntoResponse {
    Html(include_str!("../dashboard.html"))
}

/// `GET /v1/license` — current license state (no auth required).
async fn proxy_get_license(State(layer): State<Arc<OrchestratorLayer>>) -> impl IntoResponse {
    Json(layer.license.to_json())
}

/// `POST /v1/license` — activate a license key (no auth required).
///
/// Body: `{"key": "<license-key>"}`
async fn proxy_set_license(
    State(layer): State<Arc<OrchestratorLayer>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let Some(key) = body.get("key").and_then(|v| v.as_str()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "missing field: key"})),
        )
            .into_response();
    };
    let key = key.trim().to_string();
    if key.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "key must not be empty"})),
        )
            .into_response();
    }
    match layer.license.set_key(key) {
        Ok(()) => Json(layer.license.to_json()).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Start the orchestrator: public proxy + internal API + health ticker.
///
/// Runs until SIGINT / SIGTERM.
pub async fn start_orchestrator(
    config: OrchestratorConfig,
    license_config: LicenseConfig,
) -> Result<()> {
    let layer = Arc::new(OrchestratorLayer::new(config.clone(), license_config)?);

    let public_addr = format!("{}:{}", config.host, config.port);
    let internal_addr = format!("127.0.0.1:{}", config.internal_port);

    info!(%public_addr, "orchestrator public proxy starting");
    info!(%internal_addr, "orchestrator internal API starting (loopback only)");
    info!(
        policy = %config.dispatch_policy,
        heartbeat_ms = config.worker_heartbeat_ms,
        ttl_ms = config.worker_ttl_ms,
        "orchestrator config"
    );

    // Shutdown channel — send `true` to trigger graceful shutdown.
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Health ticker background task.
    let ticker = HealthTicker::new(
        layer.registry.clone(),
        config.worker_heartbeat_ms,
        config.worker_ttl_ms,
    );
    let ticker_shutdown = shutdown_rx.clone();
    tokio::spawn(async move {
        ticker.run(ticker_shutdown).await;
    });

    // Internal router (loopback).
    let internal_state = InternalState {
        registry: layer.registry.clone(),
        config: Arc::clone(&layer.config),
        license: Arc::clone(&layer.license),
    };
    let internal_app = {
        let app = internal_router(internal_state);
        let maybe_token = std::env::var("AXS_INTERNAL_API_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        if let Some(token) = maybe_token {
            info!("orchestrator internal API token auth enabled");
            app.route_layer(middleware::from_fn_with_state(
                Arc::new(token),
                internal_auth_middleware,
            ))
        } else {
            app
        }
    };
    let internal_listener = tokio::net::TcpListener::bind(&internal_addr).await?;

    let public_listener = tokio::net::TcpListener::bind(&public_addr).await?;

    // Auth: load API keys and apply to the public proxy.
    // The internal router (loopback-only) is not authenticated — it is
    // network-isolated and intended for worker-to-orchestrator communication.
    let api_keys = crate::auth::load_api_keys();
    if api_keys.is_empty() {
        let allow_no_auth = std::env::var("AXS_ALLOW_NO_AUTH")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);
        if allow_no_auth {
            info!(
                "orchestrator auth disabled — AXS_ALLOW_NO_AUTH=true (development / testing only)"
            );
        } else {
            anyhow::bail!(
                "AXS_API_KEY is not set — the orchestrator public proxy is exposed on \
                 {}:{} without authentication, which is unsafe in production. \
                 Set AXS_API_KEY to a comma-separated list of bearer tokens, or set \
                 AXS_ALLOW_NO_AUTH=true to explicitly allow unauthenticated access \
                 (development and testing only).",
                config.host,
                config.port
            );
        }
    } else {
        info!(
            "orchestrator API key authentication enabled ({} key(s))",
            api_keys.len()
        );
    }

    let public_app = proxy_router(Arc::clone(&layer))
        .route_layer(middleware::from_fn_with_state(
            api_keys,
            crate::auth::auth_middleware,
        ))
        .layer(middleware::from_fn(
            crate::auth::request_id_and_headers_middleware,
        ));

    // Shutdown signal handler — handle both SIGINT (Ctrl-C) and SIGTERM
    // (sent by Docker, systemd, Kubernetes, and other process supervisors).
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};

        let ctrl_c = async { tokio::signal::ctrl_c().await.ok() };
        let sigterm = async {
            signal(SignalKind::terminate())
                .expect("failed to install SIGTERM handler")
                .recv()
                .await
        };
        tokio::select! {
            _ = ctrl_c => {}
            _ = sigterm => {}
        }
        info!("shutdown signal received — draining connections");
        let _ = shutdown_tx_clone.send(true);
    });

    // Wire the shutdown watch into both listeners so they drain open connections
    // instead of dropping them abruptly.
    let internal_shutdown = shutdown_rx.clone();
    let public_shutdown = shutdown_rx;
    tokio::try_join!(
        async {
            axum::serve(internal_listener, internal_app)
                .with_graceful_shutdown(async move {
                    let mut rx = internal_shutdown;
                    while !*rx.borrow() {
                        rx.changed().await.ok();
                    }
                })
                .await
                .map_err(anyhow::Error::from)
        },
        async {
            axum::serve(public_listener, public_app)
                .with_graceful_shutdown(async move {
                    let mut rx = public_shutdown;
                    while !*rx.borrow() {
                        rx.changed().await.ok();
                    }
                })
                .await
                .map_err(anyhow::Error::from)
        },
    )?;

    Ok(())
}
