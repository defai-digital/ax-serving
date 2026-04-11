//! Configuration for the serving layer.
//!
//! Loaded from (in priority order): CLI flags → env vars → config file (`.yaml`/`.toml`) → defaults.

use std::path::Path;

use anyhow::{Context, Result};
use ax_serving_engine::{LlamaCppConfig, MlxConfig};
use serde::Deserialize;

/// Top-level serving configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServeConfig {
    pub rest_addr: String,
    pub grpc_socket: String,
    /// TCP gRPC bind host (used only when `grpc_port` is set).
    /// env: `AXS_GRPC_HOST`
    pub grpc_host: String,
    pub grpc_port: Option<u16>,

    // ── Admission scheduler ────────────────────────────────────────────────────
    /// env: `AXS_SCHED_MAX_INFLIGHT`
    pub sched_max_inflight: usize,
    /// env: `AXS_SCHED_MAX_QUEUE`
    pub sched_max_queue: usize,
    /// env: `AXS_SCHED_MAX_WAIT_MS`
    pub sched_max_wait_ms: u64,
    /// Overload policy: `"reject"` (default) or `"shed_oldest"`.
    /// env: `AXS_OVERLOAD_POLICY`
    pub sched_overload_policy: String,
    /// Per-model concurrency cap (each model ID gets its own semaphore).
    /// env: `AXS_PER_MODEL_MAX_INFLIGHT`
    pub sched_per_model_max_inflight: usize,
    /// Default `max_tokens` applied when the client omits the field.
    /// Prevents unbounded generation from monopolizing the inference slot.
    /// Set to 0 to disable the default (pass `None` to the backend).
    /// env: `AXS_DEFAULT_MAX_TOKENS`
    pub default_max_tokens: u32,
    /// Enable prefill/decode split scheduler tracking (WS2).
    /// env: `AXS_SPLIT_SCHEDULER`
    pub split_scheduler: bool,
    /// Max concurrent prompt tokens in the prefill phase.
    /// env: `AXS_SCHED_MAX_PREFILL_TOKENS` (default 4096)
    pub sched_max_prefill_tokens: u64,
    /// Max concurrent decode sequences.
    /// env: `AXS_SCHED_MAX_DECODE_SEQUENCES` (default = sched_max_inflight)
    pub sched_max_decode_sequences: usize,

    // ── Idle eviction & thermal ────────────────────────────────────────────────
    /// Unload models idle longer than this many seconds (0 = disabled).
    /// env: `AXS_IDLE_TIMEOUT_SECS`
    pub idle_timeout_secs: u64,
    /// How often the thermal state is polled (seconds).
    /// env: `AXS_THERMAL_POLL_SECS`
    pub thermal_poll_secs: u64,

    // ── Sub-sections ──────────────────────────────────────────────────────────
    pub cache: CacheConfig,
    pub registry: RegistryConfig,
    pub dispatcher: DispatcherConfig,
    /// llama.cpp subprocess backend settings.
    pub llamacpp: LlamaCppConfig,
    /// mlx-lm subprocess backend settings.
    pub mlx: MlxConfig,
    /// Multi-worker orchestrator settings (`ax-llama orchestrate`).
    pub orchestrator: OrchestratorConfig,
    /// License reminder and dashboard settings.
    pub license: LicenseConfig,
    /// Project-scoped runtime admission policy.
    pub project_policy: ProjectPolicyConfig,
}

// ── Default addresses (overridable via AXS_* env vars) ────────────────────────

/// Default loopback host used across REST, gRPC and orchestrator bindings.
const DEFAULT_BIND_HOST: &str = "127.0.0.1";
const DEFAULT_REST_ADDR: &str = "127.0.0.1:18080";
const DEFAULT_GRPC_SOCKET: &str = "/tmp/ax-serving.sock";
const DEFAULT_REDIS_URL: &str = "redis://127.0.0.1:6379";

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            rest_addr: DEFAULT_REST_ADDR.into(),
            grpc_socket: DEFAULT_GRPC_SOCKET.into(),
            grpc_host: DEFAULT_BIND_HOST.into(),
            grpc_port: None,
            sched_max_inflight: 16,
            sched_max_queue: 128,
            sched_max_wait_ms: 120_000,
            sched_overload_policy: "queue".into(),
            sched_per_model_max_inflight: 4,
            default_max_tokens: 2048,
            split_scheduler: false,
            sched_max_prefill_tokens: 4096,
            sched_max_decode_sequences: 0,
            idle_timeout_secs: 0,
            thermal_poll_secs: 5,
            cache: CacheConfig::default(),
            registry: RegistryConfig::default(),
            dispatcher: DispatcherConfig::default(),
            llamacpp: LlamaCppConfig::default(),
            mlx: MlxConfig::default(),
            orchestrator: OrchestratorConfig::default(),
            license: LicenseConfig::default(),
            project_policy: ProjectPolicyConfig::default(),
        }
    }
}

// ── RegistryConfig ────────────────────────────────────────────────────────────

/// Model registry settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RegistryConfig {
    /// Hard cap on simultaneously loaded models (env: `AXS_MAX_LOADED_MODELS`).
    pub max_loaded_models: usize,
    /// How often the idle-eviction background task runs (env: `AXS_IDLE_CHECK_INTERVAL_SECS`).
    pub idle_check_interval_secs: u64,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_loaded_models: 16,
            idle_check_interval_secs: 60,
        }
    }
}

// ── DispatcherConfig ──────────────────────────────────────────────────────────

/// Dispatcher / request-routing settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DispatcherConfig {
    /// Max idle connections per worker in the reqwest connection pool
    /// (env: `AXS_DISPATCHER_POOL_MAX_IDLE`).
    pub pool_max_idle_per_host: usize,
    /// Value of the `Retry-After` header sent on 429 Rejected responses, in seconds
    /// (env: `AXS_RETRY_AFTER_SECS`).
    pub retry_after_secs: u64,
    /// Max follow-up attempts on cache-inflight collisions before bypassing the cache
    /// (env: `AXS_CACHE_INFLIGHT_MAX_RETRIES`).
    pub cache_inflight_max_retries: usize,
    /// Timeout for requests from orchestrator to workers, in seconds
    /// (env: `AXS_DISPATCHER_TIMEOUT_SECS`).
    pub request_timeout_secs: u64,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            pool_max_idle_per_host: 8,
            retry_after_secs: 5,
            cache_inflight_max_retries: 3,
            request_timeout_secs: 300,
        }
    }
}

// ── CacheConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub enabled: bool,
    pub url: String,
    pub key_prefix: String,
    pub default_ttl: String,
    pub max_ttl: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: DEFAULT_REDIS_URL.into(),
            key_prefix: "axs:chat:v1".into(),
            default_ttl: "1h".into(),
            max_ttl: "30d".into(),
        }
    }
}

// ── OrchestratorConfig ────────────────────────────────────────────────────────

/// Multi-worker orchestrator settings (used by `ax-llama orchestrate`).
///
/// These values configure the public proxy gateway and its internal worker
/// registry.  Loaded from the `orchestrator:` section of `serving.yaml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OrchestratorConfig {
    /// Public proxy host (clients send inference requests here).
    /// env: `AXS_ORCHESTRATOR_HOST`
    pub host: String,
    /// Public proxy port (clients send inference requests here).
    /// env: `AXS_ORCHESTRATOR_PORT`
    pub port: u16,
    /// Internal API port (workers register / heartbeat here).
    /// env: `AXS_INTERNAL_PORT`
    pub internal_port: u16,
    /// Internal API bind host.
    /// env: `AXS_INTERNAL_BIND_ADDR`
    pub internal_bind_addr: String,
    /// Comma-separated worker source IP/CIDR allowlist for the internal API.
    /// env: `AXS_ALLOWED_NODE_CIDRS`
    pub allowed_node_cidrs: String,
    /// How often workers should send heartbeats (ms).
    /// env: `AXS_WORKER_HEARTBEAT_MS`
    pub worker_heartbeat_ms: u64,
    /// Age at which a worker is considered dead and evicted (ms).
    /// env: `AXS_WORKER_TTL_MS`
    pub worker_ttl_ms: u64,
    /// Dispatch policy name: `"least_inflight"` (default), `"weighted_round_robin"`,
    /// `"model_affinity"`, `"token_cost"`, or `"cache_affinity"`.
    /// env: `AXS_DISPATCH_POLICY`
    pub dispatch_policy: String,
    /// `Retry-After` header value on 429 responses (secs). env: `AXS_RETRY_AFTER_SECS`
    pub retry_after_secs: u64,
    /// Max idle connections per worker in the reqwest pool. env: `AXS_DISPATCHER_POOL_MAX_IDLE`
    pub pool_max_idle_per_host: usize,
    /// Timeout for proxy requests to workers (secs). env: `AXS_DISPATCHER_TIMEOUT_SECS`
    pub request_timeout_secs: u64,

    // ── Global queue (admission control for the proxy) ─────────────────────
    /// Max concurrent active requests the proxy will forward simultaneously.
    /// env: `AXS_GLOBAL_QUEUE_MAX`
    pub global_queue_max: usize,
    /// Max requests waiting in the queue before the overload policy kicks in.
    /// env: `AXS_GLOBAL_QUEUE_DEPTH`
    pub global_queue_depth: usize,
    /// How long a queued request waits before timing out (ms).
    /// env: `AXS_GLOBAL_QUEUE_WAIT_MS`
    pub global_queue_wait_ms: u64,
    /// Queue overload policy: `"reject"` (HTTP 429, default) or `"shed_oldest"` (HTTP 503).
    /// env: `AXS_GLOBAL_QUEUE_POLICY`
    pub global_queue_policy: String,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_BIND_HOST.into(),
            port: 18080,
            internal_port: 19090,
            internal_bind_addr: DEFAULT_BIND_HOST.into(),
            allowed_node_cidrs: String::new(),
            worker_heartbeat_ms: 5_000,
            worker_ttl_ms: 15_000,
            dispatch_policy: "least_inflight".into(),
            retry_after_secs: 5,
            pool_max_idle_per_host: 8,
            request_timeout_secs: 300,
            global_queue_max: 128,
            global_queue_depth: 256,
            global_queue_wait_ms: 10_000,
            global_queue_policy: "queue".into(),
        }
    }
}

// ── LicenseConfig ─────────────────────────────────────────────────────────────

/// License reminder and dashboard settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LicenseConfig {
    /// URL shown in the dashboard commercial licensing link.
    /// env: `AXS_LICENSE_BUY_LINK`
    pub buy_link: String,
    /// Directory under `~/.config/` (or `$XDG_CONFIG_HOME/`) where the license key file lives.
    /// env: `AXS_LICENSE_CONFIG_DIR`
    pub config_dir: String,
    /// License key filename within `config_dir`.
    /// env: `AXS_LICENSE_KEY_FILE`
    pub key_file: String,
    /// Dashboard polling interval in milliseconds.
    /// env: `AXS_DASHBOARD_POLL_MS`
    pub dashboard_poll_ms: u64,
}

impl Default for LicenseConfig {
    fn default() -> Self {
        Self {
            buy_link: "https://license.automatosx.com".into(),
            config_dir: "ax-serving".into(),
            key_file: "license.key".into(),
            dashboard_poll_ms: 2000,
        }
    }
}

// ── ProjectPolicyConfig ───────────────────────────────────────────────────────

/// Project-scoped runtime admission policy.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ProjectPolicyConfig {
    /// Enable project-scoped admission policy.
    pub enabled: bool,
    /// Default project name when the request omits `X-AX-Project`.
    pub default_project: Option<String>,
    /// Project rules evaluated by exact project-name match.
    pub rules: Vec<ProjectRuleConfig>,
}

/// Per-project runtime admission rule.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ProjectRuleConfig {
    /// Project name matched against `X-AX-Project`.
    pub project: String,
    /// Allowed model IDs or prefixes (`prefix*`). Empty = deny all.
    pub allowed_models: Vec<String>,
    /// Maximum allowed requested max_tokens for this project (optional).
    pub max_tokens_limit: Option<u32>,
    /// Optional worker pool enforced by policy when routing through the orchestrator.
    pub worker_pool: Option<String>,
}

// ── Env-var parsing helpers ──────────────────────────────────────────────────

/// Read an env var as a string, returning None if unset.
fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Read an env var and parse it as type T.
fn env_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse().ok()
}

/// Read an env var as a boolean (true/1/yes -> true, anything else -> false).
fn env_bool(name: &str) -> Option<bool> {
    Some(matches!(
        std::env::var(name).ok()?.to_lowercase().as_str(),
        "true" | "1" | "yes"
    ))
}

// ── ServeConfig methods ───────────────────────────────────────────────────────

impl ServeConfig {
    /// Validate that the resolved configuration is internally consistent.
    ///
    /// Call after loading + applying env overrides, before starting servers.
    pub fn validate(&self) -> Result<()> {
        // ── Scheduler ─────────────────────────────────────────────────────────
        if self.sched_max_inflight == 0 {
            anyhow::bail!("sched_max_inflight must be > 0");
        }
        if self.sched_max_wait_ms == 0 {
            anyhow::bail!("sched_max_wait_ms must be > 0");
        }
        if self.sched_per_model_max_inflight == 0 {
            anyhow::bail!("sched_per_model_max_inflight must be >= 1");
        }
        if self.sched_max_queue < self.sched_max_inflight {
            anyhow::bail!(
                "sched_max_queue ({}) must be >= sched_max_inflight ({})",
                self.sched_max_queue,
                self.sched_max_inflight
            );
        }
        const VALID_OVERLOAD: &[&str] = &["queue", "reject", "shed_oldest"];
        if !VALID_OVERLOAD.contains(&self.sched_overload_policy.as_str()) {
            anyhow::bail!(
                "unknown sched_overload_policy '{}'; valid: queue, reject, shed_oldest",
                self.sched_overload_policy
            );
        }

        // ── Thermal ───────────────────────────────────────────────────────────
        if self.thermal_poll_secs == 0 {
            anyhow::bail!("thermal_poll_secs must be >= 1");
        }

        // ── Dispatcher ────────────────────────────────────────────────────────
        if self.dispatcher.request_timeout_secs == 0 {
            anyhow::bail!("dispatcher.request_timeout_secs must be > 0");
        }

        // ── Cache ─────────────────────────────────────────────────────────────
        if self.cache.enabled {
            let def = crate::cache::parse_ttl(&self.cache.default_ttl)
                .context("invalid cache.default_ttl")?;
            let max =
                crate::cache::parse_ttl(&self.cache.max_ttl).context("invalid cache.max_ttl")?;
            if def > max {
                anyhow::bail!("cache.default_ttl must not exceed cache.max_ttl");
            }
        }

        // ── Orchestrator ──────────────────────────────────────────────────────
        if self.orchestrator.port == self.orchestrator.internal_port {
            anyhow::bail!(
                "orchestrator.port and internal_port must differ (both set to {})",
                self.orchestrator.port
            );
        }
        if self.orchestrator.worker_ttl_ms <= self.orchestrator.worker_heartbeat_ms {
            anyhow::bail!(
                "worker_ttl_ms ({}) must be > worker_heartbeat_ms ({})",
                self.orchestrator.worker_ttl_ms,
                self.orchestrator.worker_heartbeat_ms
            );
        }
        if self.orchestrator.global_queue_max == 0 {
            anyhow::bail!("global_queue_max must be > 0");
        }
        const VALID_DISPATCH: &[&str] = &[
            "least_inflight",
            "weighted_round_robin",
            "model_affinity",
            "token_cost",
            "cache_affinity",
        ];
        if !VALID_DISPATCH.contains(&self.orchestrator.dispatch_policy.as_str()) {
            anyhow::bail!(
                "unknown dispatch_policy '{}'; valid: least_inflight, weighted_round_robin, model_affinity, token_cost, cache_affinity",
                self.orchestrator.dispatch_policy
            );
        }
        if self.orchestrator.request_timeout_secs == 0 {
            anyhow::bail!("orchestrator.request_timeout_secs must be > 0");
        }
        if self.orchestrator.global_queue_max > self.orchestrator.global_queue_depth {
            anyhow::bail!(
                "global_queue_depth ({}) must be >= global_queue_max ({})",
                self.orchestrator.global_queue_depth,
                self.orchestrator.global_queue_max
            );
        }

        // ── Project policy ───────────────────────────────────────────────────
        if self.project_policy.enabled {
            let mut seen = std::collections::HashSet::new();
            for rule in &self.project_policy.rules {
                let project = rule.project.trim();
                if project.is_empty() {
                    anyhow::bail!("project_policy.rules[].project must not be empty");
                }
                if !seen.insert(project.to_string()) {
                    anyhow::bail!("duplicate project_policy rule for project '{project}'");
                }
                if rule.allowed_models.is_empty() {
                    anyhow::bail!(
                        "project_policy rule '{}' must declare at least one allowed model",
                        project
                    );
                }
                if let Some(limit) = rule.max_tokens_limit
                    && limit == 0
                {
                    anyhow::bail!(
                        "project_policy rule '{}' max_tokens_limit must be > 0",
                        project
                    );
                }
            }
            if let Some(default_project) = self.project_policy.default_project.as_deref()
                && !seen.contains(default_project)
            {
                anyhow::bail!(
                    "project_policy.default_project '{}' must match a declared rule",
                    default_project
                );
            }
        }

        Ok(())
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading serve config {}", path.display()))?;
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let mut cfg = match ext.as_str() {
            "yaml" | "yml" => serde_yaml::from_str::<Self>(&text)
                .with_context(|| format!("parsing yaml config {}", path.display()))?,
            _ => toml::from_str::<Self>(&text)
                .with_context(|| format!("parsing toml config {}", path.display()))?,
        };
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.apply_env_overrides();
        cfg
    }

    pub fn apply_env_overrides(&mut self) {
        // ── REST / gRPC ───────────────────────────────────────────────────────
        if let Some(port) = env_parse::<u16>("AXS_REST_PORT") {
            self.rest_addr = format!("{DEFAULT_BIND_HOST}:{port}");
        }
        if let Ok(v) = std::env::var("AXS_REST_HOST") {
            let port = self.rest_addr.rsplit(':').next().unwrap_or("18080");
            self.rest_addr = format!("{v}:{port}");
        }
        if let Some(v) = env_str("AXS_GRPC_SOCKET") {
            self.grpc_socket = v;
        }
        if let Some(v) = env_str("AXS_GRPC_HOST") {
            self.grpc_host = v;
        }
        if let Some(port) = env_parse::<u16>("AXS_GRPC_PORT") {
            self.grpc_port = Some(port);
        }

        // ── Scheduler ─────────────────────────────────────────────────────────
        if let Some(n) = env_parse::<usize>("AXS_SCHED_MAX_INFLIGHT") {
            self.sched_max_inflight = n.max(1);
        }
        if let Some(n) = env_parse::<usize>("AXS_SCHED_MAX_QUEUE") {
            self.sched_max_queue = n.max(1);
        }
        if let Some(ms) = env_parse::<u64>("AXS_SCHED_MAX_WAIT_MS") {
            self.sched_max_wait_ms = ms;
        }
        if let Some(v) = env_str("AXS_OVERLOAD_POLICY") {
            self.sched_overload_policy = v;
        }
        if let Some(n) = env_parse::<u32>("AXS_DEFAULT_MAX_TOKENS") {
            self.default_max_tokens = n;
        }
        if let Some(n) = env_parse::<usize>("AXS_PER_MODEL_MAX_INFLIGHT") {
            self.sched_per_model_max_inflight = n.max(1);
        }
        if let Some(b) = env_bool("AXS_SPLIT_SCHEDULER") {
            self.split_scheduler = b;
        }
        if let Some(n) = env_parse::<u64>("AXS_SCHED_MAX_PREFILL_TOKENS") {
            self.sched_max_prefill_tokens = n;
        }
        if let Some(n) = env_parse::<usize>("AXS_SCHED_MAX_DECODE_SEQUENCES") {
            self.sched_max_decode_sequences = n;
        }

        // ── Idle eviction & thermal ───────────────────────────────────────────
        if let Some(n) = env_parse::<u64>("AXS_IDLE_TIMEOUT_SECS") {
            self.idle_timeout_secs = n;
        }
        if let Some(n) = env_parse::<u64>("AXS_THERMAL_POLL_SECS") {
            self.thermal_poll_secs = n.max(1);
        }

        // ── Cache ─────────────────────────────────────────────────────────────
        if let Some(enabled) = env_parse::<bool>("AXS_CACHE_ENABLED") {
            self.cache.enabled = enabled;
        }
        if let Some(v) = env_str("AXS_CACHE_URL") {
            self.cache.url = v;
        }
        if let Some(v) = env_str("AXS_CACHE_KEY_PREFIX") {
            self.cache.key_prefix = v;
        }
        if let Some(v) = env_str("AXS_CACHE_DEFAULT_TTL") {
            self.cache.default_ttl = v;
        }
        if let Some(v) = env_str("AXS_CACHE_MAX_TTL") {
            self.cache.max_ttl = v;
        }

        // ── Registry ─────────────────────────────────────────────────────────
        if let Some(n) = env_parse::<usize>("AXS_MAX_LOADED_MODELS") {
            self.registry.max_loaded_models = n.max(1);
        }
        if let Some(n) = env_parse::<u64>("AXS_IDLE_CHECK_INTERVAL_SECS") {
            self.registry.idle_check_interval_secs = n.max(1);
        }

        // ── Dispatcher ───────────────────────────────────────────────────────
        if let Some(n) = env_parse::<usize>("AXS_DISPATCHER_POOL_MAX_IDLE") {
            self.dispatcher.pool_max_idle_per_host = n.max(1);
        }
        if let Some(n) = env_parse::<u64>("AXS_DISPATCHER_TIMEOUT_SECS") {
            self.dispatcher.request_timeout_secs = n.max(1);
        }
        if let Some(n) = env_parse::<u64>("AXS_RETRY_AFTER_SECS") {
            self.dispatcher.retry_after_secs = n;
        }
        if let Some(n) = env_parse::<usize>("AXS_CACHE_INFLIGHT_MAX_RETRIES") {
            self.dispatcher.cache_inflight_max_retries = n.max(1);
        }

        // ── LlamaCpp ─────────────────────────────────────────────────────────
        self.llamacpp.apply_env_overrides();

        // ── MLX ───────────────────────────────────────────────────────────────
        self.mlx.apply_env_overrides();

        // ── Orchestrator ──────────────────────────────────────────────────────
        if let Some(v) = env_str("AXS_ORCHESTRATOR_HOST") {
            self.orchestrator.host = v;
        }
        if let Some(p) = env_parse::<u16>("AXS_ORCHESTRATOR_PORT") {
            self.orchestrator.port = p;
        }
        if let Some(p) = env_parse::<u16>("AXS_INTERNAL_PORT") {
            self.orchestrator.internal_port = p;
        }
        if let Some(v) = env_str("AXS_INTERNAL_BIND_ADDR") {
            self.orchestrator.internal_bind_addr = v;
        }
        if let Some(v) = env_str("AXS_ALLOWED_NODE_CIDRS") {
            self.orchestrator.allowed_node_cidrs = v;
        }
        if let Some(ms) = env_parse::<u64>("AXS_WORKER_HEARTBEAT_MS") {
            self.orchestrator.worker_heartbeat_ms = ms.max(100);
        }
        if let Some(ms) = env_parse::<u64>("AXS_WORKER_TTL_MS") {
            self.orchestrator.worker_ttl_ms = ms.max(500);
        }
        if let Some(v) = env_str("AXS_DISPATCH_POLICY") {
            self.orchestrator.dispatch_policy = v;
        }
        if let Some(n) = env_parse::<usize>("AXS_GLOBAL_QUEUE_MAX") {
            self.orchestrator.global_queue_max = n.max(1);
        }
        if let Some(n) = env_parse::<usize>("AXS_GLOBAL_QUEUE_DEPTH") {
            self.orchestrator.global_queue_depth = n.max(1);
        }
        if let Some(ms) = env_parse::<u64>("AXS_GLOBAL_QUEUE_WAIT_MS") {
            self.orchestrator.global_queue_wait_ms = ms;
        }
        if let Some(v) = env_str("AXS_GLOBAL_QUEUE_POLICY") {
            self.orchestrator.global_queue_policy = v;
        }
        if let Some(n) = env_parse::<u64>("AXS_RETRY_AFTER_SECS") {
            self.orchestrator.retry_after_secs = n;
        }
        if let Some(n) = env_parse::<usize>("AXS_DISPATCHER_POOL_MAX_IDLE") {
            self.orchestrator.pool_max_idle_per_host = n.max(1);
        }
        if let Some(n) = env_parse::<u64>("AXS_DISPATCHER_TIMEOUT_SECS") {
            self.orchestrator.request_timeout_secs = n.max(1);
        }

        // ── License / dashboard ───────────────────────────────────────────────
        if let Some(v) = env_str("AXS_LICENSE_BUY_LINK") {
            self.license.buy_link = v;
        }
        if let Some(v) = env_str("AXS_LICENSE_CONFIG_DIR") {
            self.license.config_dir = v;
        }
        if let Some(v) = env_str("AXS_LICENSE_KEY_FILE") {
            self.license.key_file = v;
        }
        if let Some(ms) = env_parse::<u64>("AXS_DASHBOARD_POLL_MS") {
            self.license.dashboard_poll_ms = ms.max(500);
        }
    }

    /// Load from the default config file path, falling back to env-only defaults.
    ///
    /// Search order: `AXS_CONFIG` env var → `config/serving.yaml` → `serving.yaml` → `~/.config/ax-serving/serving.yaml`.
    pub fn load_default() -> Self {
        let candidates: Vec<std::path::PathBuf> = {
            let mut v = Vec::new();
            if let Ok(p) = std::env::var("AXS_CONFIG") {
                v.push(std::path::PathBuf::from(p));
            }
            v.push(std::path::PathBuf::from("config/serving.yaml"));
            v.push(std::path::PathBuf::from("serving.yaml"));
            if let Ok(home) = std::env::var("HOME") {
                v.push(std::path::PathBuf::from(home).join(".config/ax-serving/serving.yaml"));
            }
            v
        };

        for path in &candidates {
            if path.exists() {
                match Self::from_file(path) {
                    Ok(cfg) => {
                        tracing::info!("serve config loaded from {}", path.display());
                        return cfg;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "failed to load serve config {}: {e} — using defaults",
                            path.display()
                        );
                    }
                }
            }
        }

        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env var mutations are process-global. Serialize all tests that call
    // set_var / remove_var to prevent data races between parallel test threads.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn valid_cfg() -> ServeConfig {
        ServeConfig::default()
    }

    #[test]
    fn default_config_passes_validation() {
        assert!(valid_cfg().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_inflight() {
        let mut cfg = valid_cfg();
        cfg.sched_max_inflight = 0;
        // sched_max_queue (128) < sched_max_inflight (0) check comes after, so inflight=0 fires first.
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_queue_smaller_than_inflight() {
        let mut cfg = valid_cfg();
        cfg.sched_max_inflight = 64;
        cfg.sched_max_queue = 32;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("sched_max_queue"), "got: {err}");
    }

    #[test]
    fn validate_rejects_unknown_overload_policy() {
        let mut cfg = valid_cfg();
        cfg.sched_overload_policy = "drop_newest".into();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("sched_overload_policy"), "got: {err}");
    }

    #[test]
    fn validate_rejects_unknown_dispatch_policy() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.dispatch_policy = "random".into();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("dispatch_policy"), "got: {err}");
    }

    #[test]
    fn validate_accepts_cache_affinity_dispatch_policy() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.dispatch_policy = "cache_affinity".into();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_port_conflict() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.port = 18080;
        cfg.orchestrator.internal_port = 18080;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("internal_port"), "got: {err}");
    }

    #[test]
    fn validate_rejects_ttl_not_greater_than_heartbeat() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.worker_ttl_ms = 5_000;
        cfg.orchestrator.worker_heartbeat_ms = 5_000;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("worker_ttl_ms"), "got: {err}");
    }

    #[test]
    fn validate_rejects_zero_global_queue_max() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.global_queue_max = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_thermal_poll() {
        let mut cfg = valid_cfg();
        cfg.thermal_poll_secs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_request_timeout() {
        let mut cfg = valid_cfg();
        cfg.dispatcher.request_timeout_secs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_queue_depth_less_than_queue_max() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.global_queue_max = 200;
        cfg.orchestrator.global_queue_depth = 100; // less than max
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("global_queue_depth"), "got: {err}");
    }

    #[test]
    fn validate_accepts_queue_depth_equal_to_queue_max() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.global_queue_max = 128;
        cfg.orchestrator.global_queue_depth = 128;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_duplicate_project_policy_rules() {
        let mut cfg = valid_cfg();
        cfg.project_policy.enabled = true;
        cfg.project_policy.rules = vec![
            ProjectRuleConfig {
                project: "fabric".into(),
                allowed_models: vec!["embed-*".into()],
                max_tokens_limit: None,
                worker_pool: None,
            },
            ProjectRuleConfig {
                project: "fabric".into(),
                allowed_models: vec!["chat-*".into()],
                max_tokens_limit: None,
                worker_pool: None,
            },
        ];
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("duplicate project_policy"), "got: {err}");
    }

    #[test]
    fn validate_rejects_unknown_default_project() {
        let mut cfg = valid_cfg();
        cfg.project_policy.enabled = true;
        cfg.project_policy.default_project = Some("missing".into());
        cfg.project_policy.rules = vec![ProjectRuleConfig {
            project: "fabric".into(),
            allowed_models: vec!["embed-*".into()],
            max_tokens_limit: None,
            worker_pool: None,
        }];
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("default_project"), "got: {err}");
    }

    // ── apply_env_overrides ────────────────────────────────────────────────────
    // Env var mutations are process-global and `unsafe` in edition 2024.
    // Serialize all env-mutating tests through ENV_LOCK to avoid test-thread UB.

    #[test]
    fn env_override_grpc_host() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_GRPC_HOST", "0.0.0.0") };
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_GRPC_HOST") };
        assert_eq!(cfg.grpc_host, "0.0.0.0");
    }

    #[test]
    fn env_override_orchestrator_host() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_ORCHESTRATOR_HOST", "192.168.1.5") };
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_ORCHESTRATOR_HOST") };
        assert_eq!(cfg.orchestrator.host, "192.168.1.5");
    }

    #[test]
    fn env_override_sched_max_inflight_enforces_minimum_one() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_SCHED_MAX_INFLIGHT", "0") };
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_SCHED_MAX_INFLIGHT") };
        assert_eq!(cfg.sched_max_inflight, 1, "0 should be clamped to 1");
    }

    #[test]
    fn env_override_default_max_tokens() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_DEFAULT_MAX_TOKENS", "4096") };
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_DEFAULT_MAX_TOKENS") };
        assert_eq!(cfg.default_max_tokens, 4096);
    }

    #[test]
    fn env_override_dispatch_policy() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_DISPATCH_POLICY", "weighted_round_robin") };
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_DISPATCH_POLICY") };
        assert_eq!(cfg.orchestrator.dispatch_policy, "weighted_round_robin");
    }

    #[test]
    fn env_override_orchestrator_ports() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("AXS_ORCHESTRATOR_PORT", "9000");
            std::env::set_var("AXS_INTERNAL_PORT", "9001");
            std::env::set_var("AXS_INTERNAL_BIND_ADDR", "0.0.0.0");
            std::env::set_var("AXS_ALLOWED_NODE_CIDRS", "127.0.0.1/32,10.0.0.0/8");
        }
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe {
            std::env::remove_var("AXS_ORCHESTRATOR_PORT");
            std::env::remove_var("AXS_INTERNAL_PORT");
            std::env::remove_var("AXS_INTERNAL_BIND_ADDR");
            std::env::remove_var("AXS_ALLOWED_NODE_CIDRS");
        }
        assert_eq!(cfg.orchestrator.port, 9000);
        assert_eq!(cfg.orchestrator.internal_port, 9001);
        assert_eq!(cfg.orchestrator.internal_bind_addr, "0.0.0.0");
        assert_eq!(
            cfg.orchestrator.allowed_node_cidrs,
            "127.0.0.1/32,10.0.0.0/8"
        );
    }

    #[test]
    fn env_override_worker_heartbeat_ms_enforces_minimum() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_WORKER_HEARTBEAT_MS", "10") }; // below 100 ms floor
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_WORKER_HEARTBEAT_MS") };
        assert_eq!(cfg.orchestrator.worker_heartbeat_ms, 100, "floor is 100ms");
    }

    #[test]
    fn env_override_dashboard_poll_ms_enforces_minimum() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_DASHBOARD_POLL_MS", "100") }; // below 500 ms floor
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_DASHBOARD_POLL_MS") };
        assert_eq!(cfg.license.dashboard_poll_ms, 500, "floor is 500ms");
    }

    #[test]
    fn env_override_invalid_value_ignored() {
        let _g = ENV_LOCK.lock().unwrap();
        // Non-numeric value for a numeric field should leave the default intact.
        unsafe { std::env::set_var("AXS_SCHED_MAX_QUEUE", "not_a_number") };
        let mut cfg = ServeConfig::default();
        cfg.apply_env_overrides();
        unsafe { std::env::remove_var("AXS_SCHED_MAX_QUEUE") };
        assert_eq!(cfg.sched_max_queue, ServeConfig::default().sched_max_queue);
    }

    // ── ServeConfig::load_default ──────────────────────────────────────────────

    #[test]
    fn load_default_reads_axs_config_env_pointing_to_yaml_file() {
        let _g = ENV_LOCK.lock().unwrap();
        // Write a minimal valid YAML config to a temp file.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test_serving.yaml");
        std::fs::write(&path, "sched_max_inflight: 7\nsched_max_queue: 128\n").unwrap();

        unsafe { std::env::set_var("AXS_CONFIG", path.to_str().unwrap()) };
        let cfg = ServeConfig::load_default();
        unsafe { std::env::remove_var("AXS_CONFIG") };

        assert_eq!(
            cfg.sched_max_inflight, 7,
            "sched_max_inflight must be read from the YAML file"
        );
    }

    #[test]
    fn load_default_falls_back_to_env_when_no_file_exists() {
        let _g = ENV_LOCK.lock().unwrap();
        // Point AXS_CONFIG to a nonexistent path → load_default falls back to
        // env-only defaults (from_env). We verify a known default survives.
        unsafe { std::env::set_var("AXS_CONFIG", "/nonexistent/path/serving.yaml") };
        // Ensure the two CWD candidates don't accidentally exist.
        let cfg = ServeConfig::load_default();
        unsafe { std::env::remove_var("AXS_CONFIG") };

        // Default sched_max_inflight is 16; nothing else overrides it here.
        assert_eq!(
            cfg.sched_max_inflight,
            ServeConfig::default().sched_max_inflight,
            "must fall back to defaults when no config file is found"
        );
    }

    #[test]
    fn from_env_applies_env_vars_to_defaults() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AXS_SCHED_MAX_INFLIGHT", "5") };
        let cfg = ServeConfig::from_env();
        unsafe { std::env::remove_var("AXS_SCHED_MAX_INFLIGHT") };
        assert_eq!(cfg.sched_max_inflight, 5);
    }

    #[test]
    fn from_file_parses_yaml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("s.yaml");
        std::fs::write(&path, "default_max_tokens: 512\n").unwrap();
        let cfg = ServeConfig::from_file(&path).expect("parse yaml");
        assert_eq!(cfg.default_max_tokens, 512);
    }

    #[test]
    fn from_file_parses_toml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("s.toml");
        std::fs::write(&path, "default_max_tokens = 1024\n").unwrap();
        let cfg = ServeConfig::from_file(&path).expect("parse toml");
        assert_eq!(cfg.default_max_tokens, 1024);
    }

    #[test]
    fn from_file_returns_error_for_missing_file() {
        let result = ServeConfig::from_file(std::path::Path::new("/nonexistent/config.yaml"));
        assert!(result.is_err(), "must error when file is missing");
    }

    #[test]
    fn from_file_returns_error_for_invalid_yaml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad.yaml");
        // Unclosed brace is a definitive YAML parse error.
        std::fs::write(&path, "{unclosed: brace\n").unwrap();
        let result = ServeConfig::from_file(&path);
        assert!(result.is_err(), "must error for invalid YAML");
    }

    // ── validate: remaining constraint coverage ────────────────────────────────

    #[test]
    fn validate_rejects_zero_per_model_max_inflight() {
        let mut cfg = valid_cfg();
        cfg.sched_per_model_max_inflight = 0;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("sched_per_model_max_inflight"), "got: {err}");
    }

    #[test]
    fn validate_rejects_zero_max_wait_ms() {
        let mut cfg = valid_cfg();
        cfg.sched_max_wait_ms = 0;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("sched_max_wait_ms"), "got: {err}");
    }

    #[test]
    fn validate_rejects_zero_orchestrator_request_timeout() {
        let mut cfg = valid_cfg();
        cfg.orchestrator.request_timeout_secs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_cache_default_ttl_exceeds_max_ttl() {
        let mut cfg = valid_cfg();
        cfg.cache.enabled = true;
        cfg.cache.default_ttl = "30d".into(); // 30 days
        cfg.cache.max_ttl = "1h".into(); // 1 hour — less than default
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("default_ttl") || err.contains("max_ttl"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_malformed_cache_default_ttl() {
        let mut cfg = valid_cfg();
        cfg.cache.enabled = true;
        cfg.cache.default_ttl = "not-a-duration".into();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("default_ttl") || err.contains("invalid"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_project_policy_empty_project_name() {
        let mut cfg = valid_cfg();
        cfg.project_policy.enabled = true;
        cfg.project_policy.rules = vec![ProjectRuleConfig {
            project: "   ".into(), // whitespace-only → trimmed to ""
            allowed_models: vec!["*".into()],
            max_tokens_limit: None,
            worker_pool: None,
        }];
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("project") && err.contains("empty"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_project_policy_empty_allowed_models() {
        let mut cfg = valid_cfg();
        cfg.project_policy.enabled = true;
        cfg.project_policy.rules = vec![ProjectRuleConfig {
            project: "fabric".into(),
            allowed_models: vec![], // no models declared
            max_tokens_limit: None,
            worker_pool: None,
        }];
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("allowed model"), "got: {err}");
    }

    #[test]
    fn validate_rejects_project_policy_zero_max_tokens_limit() {
        let mut cfg = valid_cfg();
        cfg.project_policy.enabled = true;
        cfg.project_policy.rules = vec![ProjectRuleConfig {
            project: "fabric".into(),
            allowed_models: vec!["*".into()],
            max_tokens_limit: Some(0),
            worker_pool: None,
        }];
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("max_tokens_limit"), "got: {err}");
    }
}
