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

pub mod deployment;
pub mod deployment_lifecycle;
pub mod direct;
pub mod error;
pub mod fleet_state;
pub mod gateway_ops;
pub mod health_ticker;
pub mod internal_routes;
pub mod jobs;
#[cfg(feature = "nats-dispatch")]
pub mod nats;
#[cfg(feature = "nats-dispatch")]
pub mod nats_worker;
pub mod policy;
mod proxy_handlers;
pub mod queue;
pub mod registry;
pub mod request_profile;
pub mod worker_endpoint;

use std::future::pending;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{ConnectInfo, DefaultBodyLimit, Request},
    middleware,
    response::Response,
    routing::{get, post},
};
use opentelemetry::propagation::Extractor;
use tokio::sync::watch;
use tracing::Instrument as _;
use tracing::{info, warn};
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use self::deployment::{DeploymentCatalog, DeploymentCatalogStore};
use self::direct::DirectDispatcher;
use self::fleet_state::{FleetStateStore, store_from_config, unix_time_millis};
use self::gateway_ops::{GatewayOperationalState, ReadyzMode, ShutdownDeadlines};
use self::health_ticker::HealthTicker;
use self::internal_routes::{
    InternalAuthState, InternalState, internal_auth_middleware, parse_allowed_node_cidrs,
    router as internal_router,
};
use self::policy::DispatchPolicy;
use self::queue::{GlobalQueue, GlobalQueueConfig, OverloadPolicy, TenantLimiter};
use self::registry::WorkerRegistry;
use crate::audit::AuditLog;
use crate::license::LicenseState;
use crate::rest::schema::MAX_HTTP_REQUEST_BODY_BYTES;

pub use crate::config::{LicenseConfig, OrchestratorConfig, ProjectPolicyConfig};

fn is_loopback_bind_host(host: &str) -> bool {
    matches!(host, "localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

fn parse_global_queue_policy(raw: &str) -> Result<OverloadPolicy> {
    match raw.trim().to_lowercase().as_str() {
        "queue" => Ok(OverloadPolicy::Queue),
        "reject" => Ok(OverloadPolicy::Reject),
        "shed_oldest" | "shed-oldest" | "shedoldest" => Ok(OverloadPolicy::ShedOldest),
        other => anyhow::bail!(
            "unknown global_queue_policy '{}'; valid: queue, reject, shed_oldest, shed-oldest",
            other
        ),
    }
}

// ── OrchestratorLayer ─────────────────────────────────────────────────────────

/// Shared state for the orchestrator's public router.
pub struct OrchestratorLayer {
    pub registry: WorkerRegistry,
    pub policy: Arc<dyn DispatchPolicy>,
    pub dispatcher: DirectDispatcher,
    /// Validated desired-state catalog for logical models and runtime pools.
    pub deployment_catalog: Arc<DeploymentCatalogStore>,
    /// Lease-scoped fleet state shared across gateway replicas in HA mode.
    pub fleet_store: Arc<dyn FleetStateStore>,
    pub config: Arc<OrchestratorConfig>,
    pub queue: GlobalQueue,
    pub tenant_limiter: Arc<TenantLimiter>,
    /// Value emitted in `Retry-After` header on 429 responses (from config).
    pub retry_after_secs: u64,
    /// Soft license reminder state.
    pub license: Arc<LicenseState>,
    /// Project-scoped admission policy shared with the public serving API.
    pub project_policy: Arc<ProjectPolicyConfig>,
    /// Whether the public proxy requires bearer authentication.
    pub public_auth_required: AtomicBool,
    /// In-process audit log for admin and worker lifecycle actions.
    pub audit: Arc<AuditLog>,
    /// Process readiness, routability, drain, and inflight accounting.
    pub ops: Arc<GatewayOperationalState>,
}

impl OrchestratorLayer {
    pub fn new(
        config: OrchestratorConfig,
        license_config: LicenseConfig,
        project_policy: ProjectPolicyConfig,
    ) -> Result<Self> {
        let fleet_store = store_from_config(&config)?;
        Self::new_with_fleet_store(config, license_config, project_policy, fleet_store)
    }

    pub fn new_with_fleet_store(
        config: OrchestratorConfig,
        license_config: LicenseConfig,
        project_policy: ProjectPolicyConfig,
        fleet_store: Arc<dyn FleetStateStore>,
    ) -> Result<Self> {
        let policy = policy::policy_from_str(&config.dispatch_policy)?;
        let deployment_catalog = Arc::new(DeploymentCatalogStore::new(
            DeploymentCatalog::from_config(&config)?,
        ));
        let retry_after_secs = config.retry_after_secs;
        let pool_max_idle = config.pool_max_idle_per_host;
        let timeout_secs = config.request_timeout_secs;
        let queue_policy = parse_global_queue_policy(&config.global_queue_policy)?;
        let queue_config = GlobalQueueConfig {
            max_concurrent: config.global_queue_max,
            max_queue_depth: config.global_queue_depth,
            wait_ms: config.global_queue_wait_ms,
            overload_policy: queue_policy,
        };
        let readyz_mode = ReadyzMode::parse(&config.readyz_mode).map_err(anyhow::Error::msg)?;
        let shutdown = ShutdownDeadlines::new(
            config.shutdown_propagation_ms,
            config.shutdown_drain_secs,
            config.shutdown_hard_secs,
        )
        .map_err(anyhow::Error::msg)?;
        let ops = Arc::new(GatewayOperationalState::new(
            &config.fleet_store,
            readyz_mode,
            config.fleet_store_ready_max_stale_ms,
            shutdown,
        ));
        ops.mark_config_validated();
        let layer = Self {
            registry: WorkerRegistry::new(),
            policy: Arc::from(policy),
            dispatcher: DirectDispatcher::try_new_with_timeouts(
                pool_max_idle,
                timeout_secs,
                config.first_byte_timeout_ms,
                config.stream_idle_timeout_ms,
                config.dispatch_token.as_ref().map(|token| token.expose()),
            )?
            .with_fleet_state(Arc::clone(&fleet_store), config.worker_ttl_ms),
            deployment_catalog,
            fleet_store,
            config: Arc::new(config),
            queue: GlobalQueue::new(queue_config),
            tenant_limiter: TenantLimiter::shared(),
            retry_after_secs,
            license: LicenseState::new(&license_config),
            project_policy: Arc::new(project_policy),
            public_auth_required: AtomicBool::new(false),
            audit: AuditLog::default_shared(),
            ops,
        };
        layer.audit.record(
            "system",
            "startup",
            "orchestrator_layer",
            None,
            "ok",
            Some(serde_json::json!({
                "dispatch_policy": layer.config.dispatch_policy,
                "deployment_mode": layer.config.deployment_mode,
                "dispatch_auth_configured": layer.config.dispatch_token.is_some(),
                "public_port": layer.config.port,
                "internal_bind_addr": layer.config.internal_bind_addr,
                "allowed_node_cidrs": layer.config.allowed_node_cidrs,
                "internal_port": layer.config.internal_port,
            })),
        );
        Ok(layer)
    }

    pub fn set_public_auth_required(&self, required: bool) {
        self.public_auth_required.store(required, Ordering::Relaxed);
    }

    pub async fn reconcile_fleet_state(&self) -> Result<usize> {
        let now = unix_time_millis();
        let records = match self.fleet_store.list().await {
            Ok(records) => {
                self.ops.fleet_store_health.record_success(now);
                records
            }
            Err(error) => {
                self.ops.fleet_store_health.record_failure();
                return Err(error);
            }
        };
        let mut active_worker_ids = std::collections::BTreeSet::new();
        let mut restored = 0usize;
        for record in records {
            if !record.is_fresh(now) {
                let _ = self
                    .fleet_store
                    .remove_if_registration(&record.worker_id, record.registration_id)
                    .await?;
                continue;
            }
            let worker_id = record.worker_id.clone();
            let registration_id = record.registration_id;
            match self.registry.restore_protocol_record_if_newer(record) {
                Ok(was_restored) => {
                    active_worker_ids.insert(worker_id);
                    restored += usize::from(was_restored);
                }
                Err(error) => {
                    warn!(%worker_id, %error, "discarding invalid shared worker record");
                    let _ = self
                        .fleet_store
                        .remove_if_registration(&worker_id, registration_id)
                        .await?;
                }
            }
        }
        for worker_id in self.registry.protocol_worker_ids() {
            if !active_worker_ids.contains(&worker_id) {
                self.registry.evict_protocol(&worker_id);
            }
        }
        Ok(restored)
    }

    pub async fn reconcile_deployment_state(&self) -> Result<usize> {
        use ax_serving_protocol::{DeploymentControlRecord, DeploymentDesiredState};

        let mut records = self.fleet_store.list_deployments().await?;
        if records.is_empty()
            && self.deployment_catalog.snapshot().mode() == deployment::DeploymentMode::Explicit
        {
            for deployment in &self.config.deployments {
                let record = DeploymentControlRecord {
                    deployment: deployment.clone(),
                    generation: 1,
                    desired_state: if deployment.enabled {
                        DeploymentDesiredState::Enabled
                    } else {
                        DeploymentDesiredState::Disabled
                    },
                    updated_at: time::OffsetDateTime::now_utc(),
                };
                let _ = self
                    .fleet_store
                    .put_deployment_if_generation(&record, None)
                    .await?;
            }
            records = self.fleet_store.list_deployments().await?;
        }
        if self.deployment_catalog.snapshot().mode() == deployment::DeploymentMode::Explicit {
            self.deployment_catalog.apply_control_records(&records)?;
        }
        Ok(records.len())
    }
}

// ── Public proxy router ───────────────────────────────────────────────────────

pub fn proxy_router(layer: Arc<OrchestratorLayer>) -> Router {
    use deployment_lifecycle::*;
    use proxy_handlers::*;

    Router::new()
        .route("/v1/chat/completions", post(proxy_chat_completions))
        .route("/v1/completions", post(proxy_completions))
        .route("/v1/embeddings", post(proxy_embeddings))
        .route("/v1/models", get(proxy_models))
        .route("/health", get(proxy_health))
        .route("/livez", get(proxy_liveness))
        .route("/readyz", get(proxy_readiness))
        .route("/routablez", get(proxy_routability))
        .route("/v1/metrics", get(proxy_metrics))
        .route("/metrics", get(proxy_prometheus_metrics))
        .route("/v1/admin/status", get(proxy_admin_status))
        .route("/v1/admin/startup-report", get(proxy_admin_startup_report))
        .route("/v1/admin/diagnostics", get(proxy_admin_diagnostics))
        .route("/v1/admin/audit", get(proxy_admin_audit))
        .route("/v1/admin/decisions", get(proxy_admin_decisions))
        .route("/v1/admin/policy", get(proxy_admin_policy))
        .route("/v1/admin/fleet", get(proxy_admin_fleet))
        .route("/v1/admin/deployments", get(proxy_admin_deployments))
        .route(
            "/admin/v1/deployments",
            get(list_deployments).post(create_deployment),
        )
        .route(
            "/admin/v1/deployments/{id}",
            get(get_deployment)
                .patch(patch_deployment)
                .delete(delete_deployment),
        )
        .route("/admin/v1/jobs", get(list_jobs))
        .route("/admin/v1/jobs/{id}", get(get_job))
        .route("/dashboard", get(proxy_dashboard))
        .route(
            "/v1/license",
            get(proxy_get_license).post(proxy_set_license),
        )
        .route("/v1/workers", get(proxy_list_workers))
        .route(
            "/v1/workers/{id}",
            get(proxy_get_worker).delete(proxy_delete_worker),
        )
        .route("/v1/workers/{id}/drain", post(proxy_drain_worker))
        .route(
            "/v1/workers/{id}/drain-complete",
            post(proxy_drain_complete_worker),
        )
        .layer(DefaultBodyLimit::max(MAX_HTTP_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn(trace_request_middleware))
        .layer(middleware::from_fn(ensure_public_connect_info))
        .with_state(layer)
}

struct HeaderExtractor<'a>(&'a axum::http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(axum::http::HeaderName::as_str).collect()
    }
}

async fn trace_request_middleware(request: Request, next: middleware::Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let request_id = request
        .extensions()
        .get::<crate::auth::AxRequestId>()
        .map(|value| value.0.to_string())
        .unwrap_or_else(|| "unknown".into());
    let parent = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(request.headers()))
    });
    let span = tracing::info_span!(
        "axs.request",
        otel.kind = "server",
        http.request.method = %method,
        url.path = %path,
        axs.request.id = %request_id,
        http.response.status_code = tracing::field::Empty,
    );
    let _ = span.set_parent(parent);
    let response = next.run(request).instrument(span.clone()).await;
    span.record("http.response.status_code", response.status().as_u16());
    response
}

async fn ensure_public_connect_info(mut request: Request, next: middleware::Next) -> Response {
    if request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .is_none()
    {
        request
            .extensions_mut()
            .insert(ConnectInfo(std::net::SocketAddr::from(([127, 0, 0, 1], 0))));
    }
    next.run(request).await
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Start the orchestrator: public proxy + internal API + health ticker.
///
/// Runs until SIGINT / SIGTERM.
pub async fn start_orchestrator(
    config: OrchestratorConfig,
    license_config: LicenseConfig,
    project_policy: ProjectPolicyConfig,
) -> Result<()> {
    let layer = Arc::new(OrchestratorLayer::new(
        config.clone(),
        license_config,
        project_policy,
    )?);
    let restored_workers = layer
        .reconcile_fleet_state()
        .await
        .context("failed to reconcile shared fleet state during startup")?;
    let desired_deployments = layer
        .reconcile_deployment_state()
        .await
        .context("failed to reconcile shared deployment state during startup")?;

    let public_addr = format!("{}:{}", config.host, config.port);
    let internal_addr = format!("{}:{}", config.internal_bind_addr, config.internal_port);
    let internal_is_loopback = is_loopback_bind_host(&config.internal_bind_addr);

    info!(%public_addr, "orchestrator public proxy starting");
    if internal_is_loopback {
        info!(%internal_addr, "orchestrator internal API starting (loopback)");
    } else {
        info!(%internal_addr, "orchestrator internal API starting (remote-capable)");
    }
    info!(
        policy = %config.dispatch_policy,
        gateway_id = %config.gateway_id,
        fleet_store = layer.fleet_store.kind(),
        restored_workers,
        desired_deployments,
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
    )
    .with_probe_ownership(
        Arc::clone(&layer.fleet_store),
        config.gateway_id.clone(),
        config.worker_heartbeat_ms.max(1_000),
    );
    let ticker_shutdown = shutdown_rx.clone();
    tokio::spawn(async move {
        ticker.run(ticker_shutdown).await;
    });

    // Keep protocol worker leases synchronized across active gateway replicas.
    let fleet_layer = Arc::clone(&layer);
    let mut fleet_shutdown = shutdown_rx.clone();
    let fleet_interval_ms = config.worker_heartbeat_ms.max(250);
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_millis(fleet_interval_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(error) = fleet_layer.reconcile_fleet_state().await {
                        warn!(%error, "shared fleet-state reconciliation failed");
                    }
                    if let Err(error) = fleet_layer.reconcile_deployment_state().await {
                        warn!(%error, "shared deployment-state reconciliation failed");
                    }
                }
                changed = fleet_shutdown.changed() => {
                    if changed.is_err() || *fleet_shutdown.borrow() {
                        break;
                    }
                }
            }
        }
    });

    // Internal router (loopback).
    let internal_state = InternalState {
        registry: layer.registry.clone(),
        fleet_store: Arc::clone(&layer.fleet_store),
        config: Arc::clone(&layer.config),
        license: Arc::clone(&layer.license),
    };
    let internal_app = {
        let app = internal_router(internal_state);
        let allowed_sources = parse_allowed_node_cidrs(&config.allowed_node_cidrs)?;
        let maybe_token = std::env::var("AXS_INTERNAL_API_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        if !internal_is_loopback && maybe_token.is_none() {
            anyhow::bail!(
                "AXS_INTERNAL_API_TOKEN is required when the orchestrator internal API is bound \
                 to a non-loopback address ({}).",
                config.internal_bind_addr
            );
        }
        if maybe_token.is_some() || !allowed_sources.is_empty() {
            if maybe_token.is_some() {
                info!("orchestrator internal API token auth enabled");
            }
            if !allowed_sources.is_empty() {
                info!(
                    entries = allowed_sources.len(),
                    "orchestrator internal API source allowlist enabled"
                );
            }
            app.route_layer(middleware::from_fn_with_state(
                InternalAuthState {
                    token: maybe_token.map(Arc::new),
                    allowed_sources: Arc::new(allowed_sources),
                },
                internal_auth_middleware,
            ))
        } else {
            app
        }
    };
    let public_is_loopback = is_loopback_bind_host(&config.host);
    let tls_profile = config.tls_profile.trim().to_ascii_lowercase();
    if !matches!(
        tls_profile.as_str(),
        "loopback_dev" | "loopback-dev" | "trusted_mesh" | "trusted-mesh"
    ) {
        anyhow::bail!(
            "unsupported AXS_TLS_PROFILE '{}'; expected loopback_dev or trusted_mesh",
            config.tls_profile
        );
    }
    if matches!(tls_profile.as_str(), "loopback_dev" | "loopback-dev")
        && (!public_is_loopback || !internal_is_loopback)
    {
        anyhow::bail!(
            "AXS_TLS_PROFILE=loopback_dev cannot expose AX Serving on a non-loopback address; use a trusted mTLS service-mesh profile"
        );
    }
    if !internal_is_loopback && config.dispatch_token.is_none() {
        anyhow::bail!(
            "AXS_DISPATCH_TOKEN is required when the worker control plane accepts remote agents"
        );
    }
    let internal_listener = tokio::net::TcpListener::bind(&internal_addr).await?;

    let public_listener = tokio::net::TcpListener::bind(&public_addr).await?;

    // Load independent public and admin keys. The internal router above uses
    // its own worker-control token whenever it is remotely reachable.
    let api_keys = crate::auth::load_api_keys();
    let admin_api_keys = crate::auth::load_admin_api_keys();
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
        if admin_api_keys.is_empty() {
            anyhow::bail!(
                "AXS_ADMIN_API_KEY is required when AXS_API_KEY is configured; public client credentials are not accepted by admin routes"
            );
        }
        info!(
            "orchestrator admin authentication enabled ({} key(s))",
            admin_api_keys.len()
        );
    }
    layer.set_public_auth_required(!api_keys.is_empty());

    let public_app = proxy_router(Arc::clone(&layer))
        .route_layer(middleware::from_fn_with_state(
            crate::auth::GatewayAuthState {
                public_keys: api_keys,
                admin_keys: admin_api_keys,
            },
            crate::auth::gateway_auth_middleware,
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
            match signal(SignalKind::terminate()) {
                Ok(mut stream) => stream.recv().await,
                Err(err) => {
                    warn!(
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
        info!("shutdown signal received — draining connections");
        let _ = shutdown_tx_clone.send(true);
    });

    layer.ops.mark_listeners_ready();
    // Memory fleet store is always ready; seed success so redis readiness has a baseline after first reconcile.
    if layer.fleet_store.kind() == "memory" {
        layer
            .ops
            .fleet_store_health
            .record_success(unix_time_millis());
    }

    // Wire the shutdown watch into both listeners so they drain open connections
    // instead of dropping them abruptly.
    //
    // IMPORTANT: the hard process deadline is measured from when the shutdown
    // signal arrives (drain start), never from process start. Wrapping the whole
    // serve future in a timeout from t=0 would kill long-lived gateways.
    let internal_shutdown = shutdown_rx.clone();
    let public_shutdown = shutdown_rx.clone();
    let hard_shutdown_rx = shutdown_rx;
    let drain_ops = Arc::clone(&layer.ops);
    let hard_deadlines = layer.ops.shutdown;

    let serve = async {
        tokio::try_join!(
            async {
                axum::serve(
                    internal_listener,
                    internal_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                )
                .with_graceful_shutdown({
                    let mut rx = internal_shutdown;
                    let ops = Arc::clone(&drain_ops);
                    async move {
                        while !*rx.borrow() {
                            rx.changed().await.ok();
                        }
                        let shutdown_started = std::time::Instant::now();
                        // Keep control plane up through drain so agents can finish cleanup.
                        // Exit this future before the hard deadline so the hard-exit
                        // select arm remains the last-resort force-stop.
                        let keep_alive = ops
                            .shutdown
                            .remaining_until_hard(shutdown_started, std::time::Instant::now())
                            .saturating_sub(std::time::Duration::from_secs(1));
                        if !keep_alive.is_zero() {
                            tokio::time::sleep(keep_alive).await;
                        }
                    }
                })
                .await
                .map_err(anyhow::Error::from)
            },
            async {
                axum::serve(
                    public_listener,
                    public_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                )
                .with_graceful_shutdown({
                    let mut rx = public_shutdown;
                    let ops = Arc::clone(&drain_ops);
                    async move {
                        while !*rx.borrow() {
                            rx.changed().await.ok();
                        }
                        let shutdown_started = std::time::Instant::now();
                        ops.begin_drain();
                        tokio::time::sleep(std::time::Duration::from_millis(
                            ops.shutdown.propagation_ms,
                        ))
                        .await;
                        let drain_budget = std::time::Duration::from_secs(ops.shutdown.drain_secs);
                        let hard_remaining = ops
                            .shutdown
                            .remaining_until_hard(shutdown_started, std::time::Instant::now());
                        let drain_cap = drain_budget.min(hard_remaining);
                        let drain_deadline = std::time::Instant::now() + drain_cap;
                        while ops.inflight() > 0 && std::time::Instant::now() < drain_deadline {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                        if ops.inflight() > 0 {
                            warn!(
                                inflight = ops.inflight(),
                                "gateway drain deadline reached with accepted work remaining"
                            );
                        }
                    }
                })
                .await
                .map_err(anyhow::Error::from)
            },
        )?;
        Ok::<(), anyhow::Error>(())
    };

    // Hard exit only after the shutdown signal; duration is hard_secs from that Instant.
    let hard_exit = async move {
        let mut rx = hard_shutdown_rx;
        while !*rx.borrow() {
            if rx.changed().await.is_err() {
                return;
            }
        }
        let shutdown_started = std::time::Instant::now();
        let remaining =
            hard_deadlines.remaining_until_hard(shutdown_started, std::time::Instant::now());
        if !remaining.is_zero() {
            tokio::time::sleep(remaining).await;
        }
    };

    tokio::select! {
        result = serve => result?,
        _ = hard_exit => {
            warn!("gateway hard shutdown deadline reached; exiting");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestrator_layer_rejects_unknown_global_queue_policy() {
        let config = OrchestratorConfig {
            global_queue_policy: "drop_newest".into(),
            ..Default::default()
        };

        let err = match OrchestratorLayer::new(
            config,
            LicenseConfig::default(),
            ProjectPolicyConfig::default(),
        ) {
            Ok(_) => panic!("invalid global queue policy should be rejected"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("global_queue_policy"), "got: {err}");
    }

    #[test]
    fn orchestrator_layer_accepts_global_queue_policy_aliases() {
        let config = OrchestratorConfig {
            global_queue_policy: " Shed-Oldest ".into(),
            ..Default::default()
        };

        assert!(
            OrchestratorLayer::new(
                config,
                LicenseConfig::default(),
                ProjectPolicyConfig::default(),
            )
            .is_ok()
        );
    }
}
