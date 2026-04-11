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
mod proxy_handlers;
pub mod queue;
pub mod registry;

use std::future::pending;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use axum::{
    Router,
    extract::{ConnectInfo, Request},
    middleware,
    response::Response,
    routing::{get, post},
};
use tokio::sync::watch;
use tracing::{info, warn};

use self::direct::DirectDispatcher;
use self::health_ticker::HealthTicker;
use self::internal_routes::{
    InternalAuthState, InternalState, internal_auth_middleware, parse_allowed_node_cidrs,
    router as internal_router,
};
use self::policy::DispatchPolicy;
use self::queue::{GlobalQueue, GlobalQueueConfig, OverloadPolicy};
use self::registry::WorkerRegistry;
use crate::audit::AuditLog;
use crate::license::LicenseState;

pub use crate::config::{LicenseConfig, OrchestratorConfig, ProjectPolicyConfig};

fn is_loopback_bind_host(host: &str) -> bool {
    matches!(host, "localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

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
    /// Project-scoped admission policy shared with the public serving API.
    pub project_policy: Arc<ProjectPolicyConfig>,
    /// Whether the public proxy requires bearer authentication.
    pub public_auth_required: AtomicBool,
    /// In-process audit log for admin and worker lifecycle actions.
    pub audit: Arc<AuditLog>,
}

impl OrchestratorLayer {
    pub fn new(
        config: OrchestratorConfig,
        license_config: LicenseConfig,
        project_policy: ProjectPolicyConfig,
    ) -> Result<Self> {
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
        let layer = Self {
            registry: WorkerRegistry::new(),
            policy: Arc::from(policy),
            dispatcher: DirectDispatcher::new(pool_max_idle, timeout_secs),
            config: Arc::new(config),
            queue: GlobalQueue::new(queue_config),
            retry_after_secs,
            license: LicenseState::new(&license_config),
            project_policy: Arc::new(project_policy),
            public_auth_required: AtomicBool::new(false),
            audit: AuditLog::default_shared(),
        };
        layer.audit.record(
            "system",
            "startup",
            "orchestrator_layer",
            None,
            "ok",
            Some(serde_json::json!({
                "dispatch_policy": layer.config.dispatch_policy,
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
}

// ── Public proxy router ───────────────────────────────────────────────────────

pub fn proxy_router(layer: Arc<OrchestratorLayer>) -> Router {
    use proxy_handlers::*;

    Router::new()
        .route("/v1/chat/completions", post(proxy_chat_completions))
        .route("/v1/completions", post(proxy_completions))
        .route("/v1/embeddings", post(proxy_embeddings))
        .route("/v1/models", get(proxy_models))
        .route("/health", get(proxy_health))
        .route("/v1/metrics", get(proxy_metrics))
        .route("/v1/admin/status", get(proxy_admin_status))
        .route("/v1/admin/startup-report", get(proxy_admin_startup_report))
        .route("/v1/admin/diagnostics", get(proxy_admin_diagnostics))
        .route("/v1/admin/audit", get(proxy_admin_audit))
        .route("/v1/admin/policy", get(proxy_admin_policy))
        .route("/v1/admin/fleet", get(proxy_admin_fleet))
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
        .layer(middleware::from_fn(ensure_public_connect_info))
        .with_state(layer)
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
    layer.set_public_auth_required(!api_keys.is_empty());

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

    // Wire the shutdown watch into both listeners so they drain open connections
    // instead of dropping them abruptly.
    let internal_shutdown = shutdown_rx.clone();
    let public_shutdown = shutdown_rx;
    tokio::try_join!(
        async {
            axum::serve(
                internal_listener,
                internal_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
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
            axum::serve(
                public_listener,
                public_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
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
