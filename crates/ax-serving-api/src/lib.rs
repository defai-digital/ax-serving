//! ax-serving-api: Serving layer — OpenAI REST + gRPC + model registry + metrics.
//!
//! # Architecture
//!
//! ```text
//! ServingLayer (shared state)
//!   ├── ModelRegistry  — thread-safe model lifecycle
//!   ├── MetricsStore   — inference counters + latency histograms
//!   ├── Scheduler      — admission queue + concurrency control (PRD M1)
//!   └── Arc<dyn InferenceBackend>  — inference delegation
//!
//! start_servers(layer)
//!   ├── Axum REST  :18080  →  /v1/chat/completions, /v1/models, /health, /v1/metrics, /metrics
//!   └── tonic gRPC :sock  →  LoadModel, Infer, Health, GetMetrics
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let backend = Arc::new(RouterBackend::from_env());
//! let config = ServeConfig::from_env();
//! let layer = Arc::new(ServingLayer::new(backend, config));
//! start_servers(layer, &config).await?;
//! ```

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-api only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub mod auth;
pub mod audit;
pub mod cache;
pub mod config;
pub mod grpc;
pub mod license;
pub mod metrics;
pub mod orchestration;
pub mod project_policy;
pub mod registry;
pub mod rest;
pub mod scheduler;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use ax_serving_engine::{InferenceBackend, ThermalMonitor};
use tracing::{info, warn};

use crate::cache::{CacheInflight, CacheMetrics, ResponseCache};
use crate::config::ServeConfig;
use crate::license::LicenseState;
use crate::metrics::MetricsStore;
use crate::registry::ModelRegistry;
use crate::scheduler::{PerModelScheduler, Scheduler};
use crate::audit::AuditLog;

/// Shared serving state — held by both the gRPC and REST servers.
pub struct ServingLayer {
    pub registry: ModelRegistry,
    pub metrics: Arc<MetricsStore>,
    pub backend: Arc<dyn InferenceBackend>,
    pub config: Arc<ServeConfig>,
    /// Global admission queue + concurrency control (PRD M1).
    pub scheduler: Scheduler,
    /// Per-model concurrency limiter for multi-model concurrent serving (Phase 3).
    ///
    /// Requests to different models proceed in parallel; requests to the same
    /// model are serialized up to `AXS_PER_MODEL_MAX_INFLIGHT` (default 2).
    pub per_model_scheduler: PerModelScheduler,
    pub cache: Option<ResponseCache>,
    pub cache_metrics: Arc<CacheMetrics>,
    pub cache_inflight: Arc<CacheInflight>,
    /// Max follow-up attempts on cache-inflight collisions (from `DispatcherConfig`).
    pub cache_inflight_max_retries: usize,
    /// Default `max_tokens` applied when the client omits the field (0 = disabled).
    /// Prevents unbounded generation from monopolizing the inference slot.
    /// Controlled via `AXS_DEFAULT_MAX_TOKENS` (default 2048).
    pub default_max_tokens: u32,
    /// Soft license reminder state.
    pub license: Arc<LicenseState>,
    /// Whether the public REST surface requires bearer authentication.
    pub public_auth_required: AtomicBool,
    /// In-process audit log for admin and model lifecycle actions.
    pub audit: Arc<AuditLog>,
}

impl ServingLayer {
    pub fn new(backend: Arc<dyn InferenceBackend>, config: ServeConfig) -> Self {
        let cache = if config.cache.enabled {
            match ResponseCache::new(&config.cache) {
                Ok(c) => Some(c),
                Err(e) => {
                    warn!("cache disabled: failed to init valkey backend: {e}");
                    None
                }
            }
        } else {
            None
        };
        let cache_metrics = cache
            .as_ref()
            .map(|c| c.metrics())
            .unwrap_or_else(|| Arc::new(CacheMetrics::default()));
        // ThermalMonitor poll interval from config (or env fallback via new()).
        let thermal = Arc::new(ThermalMonitor::with_poll(config.thermal_poll_secs));
        let cache_inflight_max_retries = config.dispatcher.cache_inflight_max_retries;
        let layer = Self {
            registry: ModelRegistry::new(config.registry.max_loaded_models),
            metrics: Arc::new(MetricsStore::new()),
            config: Arc::new(config.clone()),
            scheduler: Scheduler::from_serve_config(
                config.sched_max_inflight,
                config.sched_max_queue,
                config.sched_max_wait_ms,
                &config.sched_overload_policy,
                config.sched_max_batch_size,
                config.sched_batch_window_ms,
                thermal,
            ),
            per_model_scheduler: PerModelScheduler::new(config.sched_per_model_max_inflight),
            backend,
            cache,
            cache_metrics,
            cache_inflight: Arc::new(CacheInflight::new()),
            cache_inflight_max_retries,
            default_max_tokens: config.default_max_tokens,
            license: LicenseState::new(&config.license),
            public_auth_required: AtomicBool::new(false),
            audit: AuditLog::default_shared(),
        };
        layer.audit.record(
            "system",
            "startup",
            "serving_layer",
            None,
            "ok",
            Some(serde_json::json!({
                "rest_addr": layer.config.rest_addr,
                "split_scheduler": layer.config.split_scheduler,
                "cache_enabled": layer.config.cache.enabled,
            })),
        );
        layer
    }

    pub fn set_public_auth_required(&self, required: bool) {
        self.public_auth_required.store(required, Ordering::Relaxed);
    }
}

/// Start both REST and gRPC servers, running until shutdown signal.
///
/// If `config.idle_timeout_secs` > 0, also spawns a background task that
/// evicts models idle longer than that threshold (checked every `registry.idle_check_interval_secs`).
pub async fn start_servers(layer: Arc<ServingLayer>, config: &ServeConfig) -> Result<()> {
    info!("starting REST server on {}", config.rest_addr);
    info!("starting gRPC server on {}", config.grpc_socket);

    let idle_secs = config.idle_timeout_secs;

    if idle_secs > 0 {
        info!("idle eviction enabled: models idle > {idle_secs}s will be unloaded");
        let registry = layer.registry.clone();
        let backend = Arc::clone(&layer.backend);
        let check_interval_secs = config.registry.idle_check_interval_secs;
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(check_interval_secs));
            let idle_ms = idle_secs * 1_000;
            loop {
                interval.tick().await;
                // Run the blocking eviction pass off the async executor.
                let registry2 = registry.clone();
                let backend2 = Arc::clone(&backend);
                let evicted = match tokio::task::spawn_blocking(move || {
                    registry2.idle_evict_pass(&*backend2, idle_ms)
                })
                .await
                {
                    Ok(ids) => ids,
                    Err(e) => {
                        warn!("idle eviction task panicked: {e}");
                        vec![]
                    }
                };
                for id in &evicted {
                    info!("idle eviction: unloaded '{id}' (idle > {idle_secs}s)");
                }
            }
        });
    }

    let api_keys = auth::load_api_keys();
    if api_keys.is_empty() {
        let allow_no_auth = std::env::var("AXS_ALLOW_NO_AUTH")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);
        if allow_no_auth {
            info!("auth disabled — AXS_ALLOW_NO_AUTH=true (development / testing only)");
        } else {
            anyhow::bail!(
                "AXS_API_KEY is not set — serving without authentication is unsafe in \
                 production. Set AXS_API_KEY to a comma-separated list of bearer tokens, \
                 or set AXS_ALLOW_NO_AUTH=true to explicitly enable unauthenticated \
                 access (development and testing only)."
            );
        }
    } else {
        info!("API key authentication enabled ({} key(s))", api_keys.len());
    }
    layer.set_public_auth_required(!api_keys.is_empty());

    tokio::try_join!(
        rest::serve(layer.clone(), config.rest_addr.clone(), api_keys.clone()),
        grpc::serve(
            layer.clone(),
            config.grpc_socket.clone(),
            config.grpc_host.clone(),
            config.grpc_port,
            api_keys
        ),
    )?;

    Ok(())
}
