//! Worker registry: identity, health state machine, eligible-worker query.
//!
//! # Health state machine
//!
//! Worker health is derived from heartbeat age relative to `ttl_ms`:
//!
//! ```text
//! age ≤ ttl/3           → Healthy
//! ttl/3 < age ≤ 2*ttl/3 → Unhealthy { missed: 1 }
//! 2*ttl/3 < age ≤ ttl   → Unhealthy { missed: 2 }
//! age > ttl              → Dead  (evicted from registry)
//! ```
//!
//! With the defaults `heartbeat_ms = 5000`, `ttl_ms = 15000`, a worker
//! must heartbeat at least once every 5 s or it transitions through
//! Unhealthy within 10 s and is evicted at 15 s.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use dashmap::DashMap;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

const MAX_WORKER_INFLIGHT: usize = 1_000_000;

// ── WorkerId ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub Uuid);

impl WorkerId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn parse(s: &str) -> Option<Self> {
        Uuid::parse_str(s).ok().map(Self)
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// ── BackendKind ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Native,
    LlamaCpp,
    SgLang,
    Vllm,
    Auto,
}

impl BackendKind {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama_cpp" | "llamacpp" | "llama-cpp" => Self::LlamaCpp,
            "sglang" | "sg_lang" | "sg-lang" => Self::SgLang,
            "vllm" | "v_llm" | "v-llm" => Self::Vllm,
            "native" => Self::Native,
            _ => Self::Auto,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LlamaCpp => "llama_cpp",
            Self::SgLang => "sglang",
            Self::Vllm => "vllm",
            Self::Auto => "auto",
        }
    }
}

fn backend_filter_from_hint(hint: Option<&str>) -> Option<BackendKind> {
    let raw = hint?.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    match BackendKind::parse(raw) {
        BackendKind::Auto => None,
        kind => Some(kind),
    }
}

fn runtime_filter_from_hint(hint: Option<&str>) -> Option<RuntimeKind> {
    let raw = hint?.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    match RuntimeKind::parse(raw) {
        RuntimeKind::Unknown => None,
        kind => Some(kind),
    }
}

// ── RuntimeKind ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    AxEngine,
    LlamaCpp,
    SgLang,
    Vllm,
    Unknown,
}

impl RuntimeKind {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "ax_engine" | "ax-engine" | "axengine" | "native" => Self::AxEngine,
            "llama_cpp" | "llamacpp" | "llama-cpp" => Self::LlamaCpp,
            "sglang" | "sg_lang" | "sg-lang" => Self::SgLang,
            "vllm" | "v_llm" | "v-llm" => Self::Vllm,
            _ => Self::Unknown,
        }
    }

    pub fn from_backend(backend: &BackendKind) -> Self {
        match backend {
            BackendKind::Native => Self::AxEngine,
            BackendKind::LlamaCpp => Self::LlamaCpp,
            BackendKind::SgLang => Self::SgLang,
            BackendKind::Vllm => Self::Vllm,
            BackendKind::Auto => Self::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AxEngine => "ax_engine",
            Self::LlamaCpp => "llama_cpp",
            Self::SgLang => "sglang",
            Self::Vllm => "vllm",
            Self::Unknown => "unknown",
        }
    }
}

// ── WorkerCapabilities ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    #[serde(default)]
    pub llm: bool,
    #[serde(default)]
    pub embedding: bool,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub max_context: Option<u32>,
}

impl WorkerCapabilities {
    fn from_legacy_models(models: Vec<String>) -> Self {
        Self {
            llm: true,
            embedding: false,
            vision: false,
            models,
            max_context: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RegisterCapabilities {
    Legacy(Vec<String>),
    Structured(WorkerCapabilities),
}

impl Default for RegisterCapabilities {
    fn default() -> Self {
        Self::Legacy(Vec::new())
    }
}

impl RegisterCapabilities {
    fn into_parts(self) -> (WorkerCapabilities, CapabilitySource) {
        match self {
            Self::Legacy(models) => (
                WorkerCapabilities::from_legacy_models(models),
                CapabilitySource::Legacy,
            ),
            Self::Structured(capabilities) => (capabilities, CapabilitySource::Structured),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapabilitySource {
    Legacy,
    Structured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Llm,
    Embedding,
}

// ── WorkerHealth ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerHealth {
    Healthy,
    Unhealthy { missed: u8 },
    Dead,
}

impl WorkerHealth {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Unhealthy { .. } => "unhealthy",
            Self::Dead => "dead",
        }
    }
}

// ── WorkerEntry ───────────────────────────────────────────────────────────────

/// Internal mutable entry held under the registry lock.
pub struct WorkerEntry {
    pub id: WorkerId,
    pub addr: SocketAddr,
    pub capabilities: WorkerCapabilities,
    /// Optional runtime-reported per-model metadata.
    pub model_inventory: Vec<ModelInventoryEntry>,
    capability_source: CapabilitySource,
    pub backend: BackendKind,
    pub runtime: RuntimeKind,
    /// Runtime integration mode reported by the worker, e.g. `adapter` or `embedded`.
    pub runtime_mode: Option<String>,
    /// Runtime version reported by the worker adapter, if known.
    pub runtime_version: Option<String>,
    /// Hardware class used for placement and fleet summaries.
    pub hardware_class: Option<String>,
    /// Runtime-compatible endpoint or proxy target reported by the worker.
    pub runtime_endpoint: Option<String>,
    /// Operations the worker supports, e.g. `llm`, `embedding`, `vision`.
    pub supported_operations: Vec<String>,
    pub max_inflight: usize,
    /// Dispatcher-owned in-flight count, updated without taking the registry lock.
    pub inflight: Arc<AtomicUsize>,
    /// Last in-flight count reported by the worker heartbeat.
    pub reported_inflight: usize,
    pub health: WorkerHealth,
    pub last_heartbeat: Instant,
    pub drain: bool,
    /// Thermal state string from the last heartbeat (e.g. "nominal", "serious").
    pub thermal_state: String,
    /// RSS memory in bytes from the last heartbeat.
    pub rss_bytes: u64,
    /// Human-readable machine name (e.g. "Aki's MacBook Pro"), set at registration.
    pub friendly_name: Option<String>,
    /// Apple Silicon chip identifier (e.g. "Apple M3 Pro"), set at registration.
    pub chip_model: Option<String>,
    /// Optional worker pool label for placement and maintenance grouping.
    pub worker_pool: Option<String>,
    /// Optional node class label for fleet summaries and placement hints.
    pub node_class: Option<String>,
    /// Active inference sequences from the last heartbeat (for token-cost dispatch).
    pub active_sequences: usize,
    /// Recent decode throughput in tokens/second (0 = unknown).
    pub decode_tok_per_sec: f64,
    /// P95 TTFT in milliseconds from the worker's scheduler metrics (0 = unknown).
    pub ttft_p95_ms: u64,
    /// Current pending queue depth reported by the worker.
    pub queue_depth: usize,
    /// Recent error rate fraction from the worker (0.0 = unknown / no errors).
    pub error_rate: f64,
    /// KV cache pages currently allocated (0 = unknown).
    pub kv_pages_used: u64,
    /// KV cache page budget (0 = unknown).
    pub kv_pages_total: u64,
    /// KV/cache utilization ratio (0.0-1.0). Used when page counters are unavailable.
    pub kv_utilization: Option<f64>,
    /// Tokens in reusable prefix cache (0 = unsupported).
    pub prefix_reusable_tokens: u64,
    /// Current internal batch occupancy (0 = unknown).
    pub active_batch_size: u32,
    /// Backend's max batch capacity (0 = unknown).
    pub max_batch_size: u32,
    /// Batch utilization ratio (0.0-1.0). Used when batch counters are unavailable.
    pub batch_utilization: Option<f64>,
}

// ── Payloads (serialised over the internal REST API) ─────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInventoryEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_format: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_operations: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    /// Omit for first registration; include to re-register with same identity.
    pub worker_id: Option<String>,
    /// `"127.0.0.1:8081"` — loopback address the orchestrator can reach.
    pub addr: String,
    /// Either a legacy model-id list or a structured capability descriptor.
    #[serde(default)]
    pub capabilities: RegisterCapabilities,
    /// Optional structured model inventory. If absent, AX Serving derives
    /// id-only entries from `capabilities.models`.
    #[serde(default)]
    pub model_inventory: Vec<ModelInventoryEntry>,
    /// `"native"` | `"llama_cpp"` | `"sglang"` | `"vllm"` | `"auto"`
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Runtime type owned by the node, e.g. `"ax_engine"` or `"vllm"`.
    #[serde(default)]
    pub runtime: Option<String>,
    /// Runtime integration mode, e.g. `"adapter"` or `"embedded"`.
    #[serde(default)]
    pub runtime_mode: Option<String>,
    /// Runtime version, if the node adapter can report it.
    #[serde(default)]
    pub runtime_version: Option<String>,
    /// Hardware placement class, e.g. `"mac"`, `"pc-cuda"`, or `"thor"`.
    #[serde(default)]
    pub hardware_class: Option<String>,
    /// Runtime-compatible endpoint or proxy target, if different from `addr`.
    #[serde(default)]
    pub runtime_endpoint: Option<String>,
    /// Explicit supported operations. If absent, AX Serving derives them from structured
    /// capabilities while legacy model-id registrations keep compatibility routing.
    #[serde(default)]
    pub supported_operations: Vec<String>,
    pub max_inflight: usize,
    /// Human-readable machine name from `scutil --get ComputerName` (optional).
    #[serde(default)]
    pub friendly_name: Option<String>,
    /// Apple Silicon chip (e.g. "Apple M3 Pro") from `system_profiler` (optional).
    #[serde(default)]
    pub chip_model: Option<String>,
    /// Operator-defined worker pool label (e.g. "blue", "canary", "studio-a").
    #[serde(default)]
    pub worker_pool: Option<String>,
    /// Operator-defined node class label (e.g. "m3-max-128g").
    #[serde(default)]
    pub node_class: Option<String>,
}

impl Default for RegisterRequest {
    fn default() -> Self {
        Self {
            worker_id: None,
            addr: String::new(),
            capabilities: RegisterCapabilities::default(),
            model_inventory: Vec::new(),
            backend: default_backend(),
            runtime: None,
            runtime_mode: None,
            runtime_version: None,
            hardware_class: None,
            runtime_endpoint: None,
            supported_operations: Vec::new(),
            max_inflight: 1,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: None,
        }
    }
}

fn default_backend() -> String {
    "auto".into()
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub worker_id: String,
    pub heartbeat_interval_ms: u64,
}

#[derive(Debug, Default, Deserialize)]
pub struct HeartbeatRequest {
    pub inflight: usize,
    #[serde(default)]
    pub thermal_state: String,
    /// Current loaded model IDs.
    ///
    /// This is treated as an authoritative snapshot from the worker and
    /// replaces the registry capabilities on each heartbeat (including empty).
    #[serde(default)]
    pub model_ids: Vec<String>,
    /// Optional structured model inventory snapshot. If absent, model_ids
    /// remain authoritative and existing per-model metadata is retained where
    /// ids still match.
    #[serde(default)]
    pub model_inventory: Vec<ModelInventoryEntry>,
    /// RSS memory in bytes from the worker process.
    #[serde(default)]
    pub rss_bytes: u64,
    /// Active inference sequences (synonym for inflight; used for token-cost dispatch).
    /// Workers that do not send this field default to 0 (dispatcher falls back to
    /// `inflight / max_inflight` scoring).
    #[serde(default)]
    pub active_sequences: usize,
    /// Recent decode throughput in tokens/second.  0 = unknown / no recent requests.
    #[serde(default)]
    pub decode_tok_per_sec: f64,
    /// P95 time-to-first-token in milliseconds from the worker's own histogram.
    /// 0 = unknown / no streaming requests yet.
    #[serde(default)]
    pub ttft_p95_ms: u64,
    /// Current pending queue depth at the worker.
    #[serde(default)]
    pub queue_depth: usize,
    /// Recent worker-side error rate fraction (0.0-1.0).
    #[serde(default)]
    pub error_rate: f64,
    /// KV cache pages currently allocated (0 = unknown).
    #[serde(default)]
    pub kv_pages_used: u64,
    /// KV cache page budget (0 = unknown).
    #[serde(default)]
    pub kv_pages_total: u64,
    /// KV/cache utilization ratio (0.0-1.0), for runtimes that expose a ratio
    /// instead of page counters.
    #[serde(default)]
    pub kv_utilization: Option<f64>,
    /// Tokens in reusable prefix cache (0 = unsupported).
    #[serde(default)]
    pub prefix_reusable_tokens: u64,
    /// Current internal batch occupancy (0 = unknown).
    #[serde(default)]
    pub active_batch_size: u32,
    /// Backend's max batch capacity (0 = unknown).
    #[serde(default)]
    pub max_batch_size: u32,
    /// Batch utilization ratio (0.0-1.0), for runtimes that expose a ratio
    /// instead of batch counters.
    #[serde(default)]
    pub batch_utilization: Option<f64>,
}

// ── Read-only snapshot for dispatch policies ──────────────────────────────────

/// A point-in-time snapshot of a worker's state, passed to [`DispatchPolicy`].
///
/// `inflight_counter` is the live atomic — the dispatcher increments it
/// *before* forwarding the request so that concurrent policy calls see
/// up-to-date load.
///
/// [`DispatchPolicy`]: super::policy::DispatchPolicy
#[derive(Clone)]
pub struct WorkerStatus {
    pub id: WorkerId,
    pub addr: SocketAddr,
    pub inflight: usize,
    pub max_inflight: usize,
    /// Active inference sequences (token-cost dispatch).  0 = unknown (legacy worker).
    pub active_sequences: usize,
    /// P95 TTFT in milliseconds (0 = unknown / no streaming requests yet).
    pub ttft_p95_ms: u64,
    /// KV cache utilization (0.0-1.0). `None` = worker does not report KV telemetry.
    pub kv_utilization: Option<f64>,
    /// Batch headroom ratio (0.0-1.0). `None` = worker does not report batch telemetry.
    pub batch_headroom: Option<f64>,
}

// ── JSON snapshot for the listing endpoints ───────────────────────────────────

#[derive(Debug, Serialize)]
pub struct WorkerSnapshot {
    pub id: WorkerId,
    pub addr: String,
    pub capabilities: Vec<String>,
    pub model_inventory: Vec<ModelInventoryEntry>,
    pub capability_descriptor: WorkerCapabilities,
    pub backend: String,
    pub runtime: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_endpoint: Option<String>,
    pub supported_operations: Vec<String>,
    pub max_inflight: usize,
    pub inflight: usize,
    /// `inflight / max_inflight` — 0.0 when idle, ≥ 1.0 when at or above capacity.
    pub saturation: f64,
    pub health: String,
    pub drain: bool,
    pub last_heartbeat_age_ms: u64,
    /// Thermal state reported by the worker's last heartbeat.
    pub thermal_state: String,
    /// RSS memory in bytes from the worker's last heartbeat.
    pub rss_bytes: u64,
    /// Human-readable machine name (e.g. "Aki's MacBook Pro").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub friendly_name: Option<String>,
    /// Apple Silicon chip identifier (e.g. "Apple M3 Pro").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chip_model: Option<String>,
    /// Optional worker pool label for placement and maintenance grouping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_pool: Option<String>,
    /// Optional node class label for fleet summaries and placement hints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_class: Option<String>,
    /// Active inference sequences (token-cost dispatch telemetry).
    pub active_sequences: usize,
    /// Recent decode throughput in tokens/second (0 = unknown).
    pub decode_tok_per_sec: f64,
    /// P95 TTFT in milliseconds (0 = unknown).
    pub ttft_p95_ms: u64,
    /// Current pending queue depth at the worker.
    pub queue_depth: usize,
    /// Recent worker-side error rate fraction.
    pub error_rate: f64,
    /// KV cache pages currently allocated (0 = unknown).
    pub kv_pages_used: u64,
    /// KV cache page budget (0 = unknown).
    pub kv_pages_total: u64,
    /// KV/cache utilization ratio (0.0-1.0), when reported by the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_utilization: Option<f64>,
    /// Tokens in reusable prefix cache (0 = unsupported).
    pub prefix_reusable_tokens: u64,
    /// Current internal batch occupancy (0 = unknown).
    pub active_batch_size: u32,
    /// Backend's max batch capacity (0 = unknown).
    pub max_batch_size: u32,
    /// Batch utilization ratio (0.0-1.0), when reported by the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_utilization: Option<f64>,
}

// ── WorkerRegistry ────────────────────────────────────────────────────────────

/// Thread-safe worker registry.  All orchestration components share one
/// instance via `Clone` (backed by an `Arc`).
///
/// # Concurrency model
///
/// The registry uses a [`DashMap`] (sharded `RwLock<HashMap>`) instead of a
/// single `RwLock<HashMap>`.  Heartbeats from N concurrent workers each lock
/// only one shard, so they proceed in parallel rather than serialising on a
/// global write lock.  Read operations (`eligible_workers`, `list_all`, etc.)
/// are also sharded and do not block each other or mutation on other shards.
///
/// The only operation that must touch all entries is `tick()` (health eviction);
/// it iterates every shard with `iter_mut()`, collecting dead IDs, then removes
/// them in a second pass.
#[derive(Clone)]
pub struct WorkerRegistry {
    inner: Arc<DashMap<WorkerId, WorkerEntry>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Register (or re-register) a worker.  Returns the assigned `WorkerId`.
    pub fn register(&self, req: RegisterRequest, heartbeat_interval_ms: u64) -> RegisterResponse {
        let RegisterRequest {
            worker_id,
            addr: raw_addr,
            capabilities,
            model_inventory,
            backend,
            runtime,
            runtime_mode,
            runtime_version,
            hardware_class,
            runtime_endpoint,
            supported_operations,
            max_inflight,
            friendly_name,
            chip_model,
            worker_pool,
            node_class,
        } = req;
        let id = worker_id
            .as_deref()
            .and_then(WorkerId::parse)
            .unwrap_or_default();

        // Sentinel: loopback on the reserved port 1 so the registry isn't
        // poisoned but the worker will never receive real traffic.
        let addr: SocketAddr = match raw_addr.parse() {
            Ok(addr) => addr,
            Err(err) => {
                warn!(
                    raw_addr = %raw_addr,
                    err = %err,
                    "worker registered with unparseable address; it will never receive traffic"
                );
                SocketAddr::from(([127, 0, 0, 1], 1))
            }
        };
        let max_inflight = max_inflight.clamp(1, MAX_WORKER_INFLIGHT);
        let backend = BackendKind::parse(&backend);
        let runtime = runtime
            .as_deref()
            .map(RuntimeKind::parse)
            .filter(|runtime| *runtime != RuntimeKind::Unknown)
            .unwrap_or_else(|| RuntimeKind::from_backend(&backend));
        let runtime_mode = normalize_runtime_mode(runtime_mode);
        let (mut capabilities, capability_source) = capabilities.into_parts();
        let incoming_model_inventory = model_inventory;
        let incoming_model_inventory_empty = incoming_model_inventory.is_empty();
        let model_inventory =
            normalize_model_inventory(&capabilities.models, incoming_model_inventory);
        let inventory_supported_operations = if incoming_model_inventory_empty {
            Vec::new()
        } else {
            let operations = supported_operations_from_model_inventory(&model_inventory);
            refresh_capabilities_from_inventory_summary(
                &mut capabilities,
                &operations,
                max_context_from_model_inventory(&model_inventory),
                false,
            );
            operations
        };
        if !incoming_model_inventory_empty {
            capabilities.models = model_inventory
                .iter()
                .map(|model| model.id.clone())
                .collect();
        }
        let supported_operations = if supported_operations.is_empty() {
            if !inventory_supported_operations.is_empty() {
                inventory_supported_operations
            } else {
                match capability_source {
                    CapabilitySource::Legacy => Vec::new(),
                    CapabilitySource::Structured => {
                        supported_operations_from_capabilities(&capabilities)
                    }
                }
            }
        } else {
            normalize_supported_operations(supported_operations)
        };

        self.inner
            .entry(id)
            .and_modify(|existing| {
                // Idempotent re-registration: update mutable fields, reset health.
                existing.addr = addr;
                existing.capabilities = capabilities.clone();
                existing.model_inventory = if incoming_model_inventory_empty {
                    retain_model_inventory_for_ids(&existing.model_inventory, &capabilities.models)
                } else {
                    model_inventory.clone()
                };
                existing.capability_source = capability_source;
                existing.backend = backend.clone();
                existing.runtime = runtime.clone();
                existing.runtime_mode = runtime_mode
                    .clone()
                    .or_else(|| existing.runtime_mode.clone());
                existing.max_inflight = max_inflight;
                existing.health = WorkerHealth::Healthy;
                existing.last_heartbeat = Instant::now();
                existing.drain = false;
                existing.supported_operations = supported_operations.clone();
                existing.runtime_version = runtime_version.clone();
                existing.hardware_class = hardware_class.clone();
                existing.runtime_endpoint = runtime_endpoint.clone();
                existing.friendly_name = friendly_name.clone();
                existing.chip_model = chip_model.clone();
                existing.worker_pool = worker_pool.clone();
                existing.node_class = node_class.clone();
            })
            .or_insert_with(|| WorkerEntry {
                id,
                addr,
                capabilities,
                model_inventory,
                capability_source,
                backend,
                runtime,
                runtime_mode,
                runtime_version,
                hardware_class,
                runtime_endpoint,
                supported_operations,
                max_inflight,
                inflight: Arc::new(AtomicUsize::new(0)),
                reported_inflight: 0,
                health: WorkerHealth::Healthy,
                last_heartbeat: Instant::now(),
                drain: false,
                thermal_state: String::new(),
                rss_bytes: 0,
                friendly_name,
                chip_model,
                worker_pool,
                node_class,
                active_sequences: 0,
                decode_tok_per_sec: 0.0,
                ttft_p95_ms: 0,
                queue_depth: 0,
                error_rate: 0.0,
                kv_pages_used: 0,
                kv_pages_total: 0,
                kv_utilization: None,
                prefix_reusable_tokens: 0,
                active_batch_size: 0,
                max_batch_size: 0,
                batch_utilization: None,
            });

        RegisterResponse {
            worker_id: id.to_string(),
            heartbeat_interval_ms,
        }
    }

    /// Record a heartbeat.  Returns `false` if the worker is not registered.
    pub fn heartbeat(&self, id: WorkerId, req: HeartbeatRequest) -> bool {
        match self.inner.get_mut(&id) {
            Some(mut e) => {
                e.last_heartbeat = Instant::now();
                e.health = WorkerHealth::Healthy;
                e.reported_inflight = req.inflight.min(MAX_WORKER_INFLIGHT);
                e.thermal_state = req.thermal_state;
                e.rss_bytes = req.rss_bytes;
                // Authoritative capability snapshot from worker heartbeat.
                // Empty model_ids means the worker currently has no models.
                let mut model_ids = req.model_ids;
                let heartbeat_has_inventory = !req.model_inventory.is_empty();
                if !req.model_inventory.is_empty() {
                    model_ids.extend(req.model_inventory.iter().map(|model| model.id.clone()));
                }
                e.model_inventory = if req.model_inventory.is_empty() {
                    retain_model_inventory_for_ids(&e.model_inventory, &model_ids)
                } else {
                    normalize_model_inventory(&model_ids, req.model_inventory)
                };
                e.capabilities.models = e
                    .model_inventory
                    .iter()
                    .map(|model| model.id.clone())
                    .collect();
                if heartbeat_has_inventory {
                    let operations = supported_operations_from_model_inventory(&e.model_inventory);
                    let max_context = max_context_from_model_inventory(&e.model_inventory);
                    refresh_capabilities_from_inventory_summary(
                        &mut e.capabilities,
                        &operations,
                        max_context,
                        true,
                    );
                    e.supported_operations = operations;
                }
                // Token-cost dispatch telemetry — graceful defaults for legacy workers.
                // active_sequences == 0 and inflight != 0 means the worker doesn't send
                // the extended field; TokenCostPolicy falls back to inflight ratio.
                e.active_sequences = req.active_sequences.min(MAX_WORKER_INFLIGHT);
                e.decode_tok_per_sec = non_negative_finite(req.decode_tok_per_sec);
                e.ttft_p95_ms = req.ttft_p95_ms;
                e.queue_depth = req.queue_depth.min(MAX_WORKER_INFLIGHT);
                e.error_rate = ratio_or_zero(req.error_rate);
                e.kv_pages_used = req.kv_pages_used;
                e.kv_pages_total = req.kv_pages_total;
                e.kv_utilization = req.kv_utilization.map(ratio_or_zero);
                e.prefix_reusable_tokens = req.prefix_reusable_tokens;
                e.active_batch_size = req.active_batch_size;
                e.max_batch_size = req.max_batch_size;
                e.batch_utilization = req.batch_utilization.map(ratio_or_zero);
                true
            }
            None => false,
        }
    }

    /// Start graceful drain.  Returns `false` if worker not found.
    pub fn mark_drain(&self, id: WorkerId) -> bool {
        match self.inner.get_mut(&id) {
            Some(mut e) => {
                e.drain = true;
                true
            }
            None => false,
        }
    }

    /// Mark a worker as unhealthy after a failed dispatch.
    ///
    /// No-op if the worker is already unhealthy, dead, or not found.
    /// The health ticker will re-evaluate on the next tick.
    pub fn mark_unhealthy(&self, id: WorkerId) {
        if let Some(mut entry) = self.inner.get_mut(&id)
            && matches!(entry.health, WorkerHealth::Healthy)
        {
            entry.health = WorkerHealth::Unhealthy { missed: 1 };
        }
    }

    /// Remove a worker entirely (drain-complete or explicit eviction).
    pub fn evict(&self, id: WorkerId) {
        self.inner.remove(&id);
    }

    /// Remove a worker only if it still matches a stale unhealthy probe snapshot.
    ///
    /// Active TCP probes are launched from a point-in-time list of unhealthy
    /// workers. A heartbeat or re-registration can make that snapshot stale
    /// before the probe result returns, so failed probes must not evict a worker
    /// that has already recovered or moved to a different address.
    pub fn evict_if_unhealthy_at_addr(&self, id: WorkerId, addr: SocketAddr) -> bool {
        self.inner
            .remove_if(&id, |_, entry| {
                entry.addr == addr && matches!(entry.health, WorkerHealth::Unhealthy { .. })
            })
            .is_some()
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns workers eligible to receive a request for `model_id`:
    /// healthy, not draining, and has `model_id` in capabilities.
    pub fn eligible_workers(&self, model_id: &str) -> Vec<WorkerStatus> {
        self.eligible_workers_filtered(model_id, RequestKind::Llm, None, None)
    }

    /// Returns workers eligible to receive a request for `model_id` and request kind:
    /// healthy, not draining, advertises the model, and supports the request kind.
    pub fn eligible_workers_for(
        &self,
        model_id: &str,
        request_kind: RequestKind,
    ) -> Vec<WorkerStatus> {
        self.eligible_workers_filtered(model_id, request_kind, None, None)
    }

    pub fn eligible_workers_filtered(
        &self,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
    ) -> Vec<WorkerStatus> {
        self.dispatch_workers_filtered(
            model_id,
            request_kind,
            backend_hint,
            min_context,
            None,
            None,
        )
    }

    pub fn dispatch_workers_filtered(
        &self,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        preferred_pool: Option<&str>,
        excluded_id: Option<WorkerId>,
    ) -> Vec<WorkerStatus> {
        self.dispatch_workers_filtered_with_pool_mode(
            model_id,
            request_kind,
            backend_hint,
            min_context,
            preferred_pool,
            false,
            excluded_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_workers_filtered_with_pool_mode(
        &self,
        model_id: &str,
        request_kind: RequestKind,
        backend_hint: Option<&str>,
        min_context: Option<u32>,
        preferred_pool: Option<&str>,
        require_preferred_pool: bool,
        excluded_id: Option<WorkerId>,
    ) -> Vec<WorkerStatus> {
        let backend_filter = backend_filter_from_hint(backend_hint);
        let runtime_filter = runtime_filter_from_hint(backend_hint);
        let preferred_pool = preferred_pool
            .map(str::trim)
            .filter(|pool| !pool.is_empty());
        let Some(preferred_pool) = preferred_pool else {
            return self
                .inner
                .iter()
                .filter_map(|r| {
                    let e = r.value();
                    dispatch_filter_matches(
                        e,
                        model_id,
                        request_kind,
                        backend_filter.as_ref(),
                        runtime_filter.as_ref(),
                        min_context,
                        excluded_id,
                    )
                    .then(|| worker_status_of(e))
                })
                .collect();
        };
        let mut preferred_pool_exists = false;
        let mut preferred_workers = Vec::new();
        let mut fallback_workers = Vec::new();

        for r in self.inner.iter() {
            let e = r.value();
            let in_preferred_pool = e.worker_pool.as_deref() == Some(preferred_pool);
            let matches_without_exclusion = dispatch_filter_matches(
                e,
                model_id,
                request_kind,
                backend_filter.as_ref(),
                runtime_filter.as_ref(),
                min_context,
                None,
            );

            if !matches_without_exclusion {
                continue;
            }
            if in_preferred_pool {
                preferred_pool_exists = true;
            }
            if excluded_id == Some(e.id) {
                continue;
            }

            let worker = worker_status_of(e);
            if in_preferred_pool {
                preferred_workers.push(worker);
            } else {
                fallback_workers.push(worker);
            }
        }

        if require_preferred_pool || preferred_pool_exists {
            preferred_workers
        } else {
            fallback_workers
        }
    }

    /// Shared inflight counter for a specific worker.
    ///
    /// This is used only after dispatch policy selection so the hot-path
    /// candidate list does not clone the counter `Arc` for every worker.
    pub fn inflight_counter(&self, id: WorkerId) -> Option<Arc<AtomicUsize>> {
        self.inner
            .get(&id)
            .map(|entry| Arc::clone(&entry.value().inflight))
    }

    /// All workers — for the `/internal/workers` listing endpoint.
    pub fn list_all(&self) -> Vec<WorkerSnapshot> {
        self.inner.iter().map(|r| snapshot_of(r.value())).collect()
    }

    /// Single worker — for the `/internal/workers/{id}` endpoint.
    pub fn get_snapshot(&self, id: WorkerId) -> Option<WorkerSnapshot> {
        self.inner.get(&id).map(|r| snapshot_of(r.value()))
    }

    /// Workers currently in `Unhealthy` state — used by the health ticker for
    /// active TCP probing.  Returns `(WorkerId, SocketAddr)` pairs.
    pub fn list_unhealthy_addrs(&self) -> Vec<(WorkerId, std::net::SocketAddr)> {
        self.inner
            .iter()
            .filter(|r| matches!(r.value().health, WorkerHealth::Unhealthy { .. }))
            .map(|r| (r.value().id, r.value().addr))
            .collect()
    }

    // ── Health ticker ─────────────────────────────────────────────────────────

    /// Derive health state from heartbeat age and evict Dead workers.
    ///
    /// Called by [`HealthTicker`] on each tick.  Returns the IDs of any
    /// workers that were evicted in this call.
    ///
    /// [`HealthTicker`]: super::health_ticker::HealthTicker
    pub fn tick(&self, ttl_ms: u64) -> Vec<WorkerId> {
        // First pass: update health states and collect dead IDs.
        // DashMap's `iter_mut` locks one shard at a time — concurrent heartbeats
        // on other shards proceed in parallel.
        let mut evicted = Vec::new();

        for mut r in self.inner.iter_mut() {
            let entry = r.value_mut();
            let age_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
            if entry.drain {
                // Draining workers are normally removed via drain-complete, but
                // if the worker crashes before calling it we must still evict.
                if age_ms > ttl_ms {
                    evicted.push(entry.id);
                }
                continue;
            }

            entry.health = if age_ms <= ttl_ms / 3 {
                if matches!(entry.health, WorkerHealth::Unhealthy { .. }) {
                    entry.health.clone()
                } else {
                    WorkerHealth::Healthy
                }
            } else if age_ms <= (2 * ttl_ms) / 3 {
                WorkerHealth::Unhealthy { missed: 1 }
            } else if age_ms <= ttl_ms {
                WorkerHealth::Unhealthy { missed: 2 }
            } else {
                evicted.push(entry.id);
                WorkerHealth::Dead // removed below
            };
        }

        // Second pass: remove only entries that are still stale. This closes the
        // race where a heartbeat or re-registration refreshes the worker after
        // the first pass but before removal.
        for id in &evicted {
            self.inner.remove_if(id, |_, entry| {
                let age_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
                if entry.drain {
                    age_ms > ttl_ms
                } else {
                    age_ms > ttl_ms && matches!(entry.health, WorkerHealth::Dead)
                }
            });
        }

        evicted
    }

    /// Count workers that are healthy AND not draining.
    ///
    /// This mirrors [`eligible_workers`] — only these workers can actually
    /// receive dispatched requests.  Use this for the orchestrator health
    /// `status` field so `"ok"` means "at least one worker can serve traffic",
    /// not "at least one worker exists but may be draining".
    ///
    /// [`eligible_workers`]: Self::eligible_workers
    pub fn eligible_healthy_count(&self) -> usize {
        self.inner
            .iter()
            .filter(|r| {
                let e = r.value();
                !e.drain && matches!(e.health, WorkerHealth::Healthy)
            })
            .count()
    }

    /// Count workers by health state and drain flag (for /health endpoint).
    ///
    /// Returns `(healthy, unhealthy, draining)`.  The `draining` count is
    /// orthogonal to health state — a draining worker may be healthy or unhealthy.
    ///
    /// Note: there is no `dead` count because `tick()` removes Dead workers in a
    /// second pass after the `iter_mut` sweep.  Between the two passes a Dead
    /// worker is briefly visible but excluded from `eligible_workers` and
    /// `eligible_healthy_count` because its health is `Dead`.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut healthy = 0usize;
        let mut unhealthy = 0usize;
        let mut draining = 0usize;
        for r in self.inner.iter() {
            let e = r.value();
            if e.drain {
                draining += 1;
            }
            match e.health {
                WorkerHealth::Healthy => healthy += 1,
                WorkerHealth::Unhealthy { .. } => unhealthy += 1,
                WorkerHealth::Dead => {} // briefly visible between tick() passes; ignore
            }
        }
        (healthy, unhealthy, draining)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn snapshot_of(e: &WorkerEntry) -> WorkerSnapshot {
    let inflight = effective_inflight(e);
    let kv_utilization = worker_kv_utilization(e);
    let batch_utilization = worker_batch_utilization(e);
    WorkerSnapshot {
        id: e.id,
        addr: e.addr.to_string(),
        capabilities: e.capabilities.models.clone(),
        model_inventory: e.model_inventory.clone(),
        capability_descriptor: e.capabilities.clone(),
        backend: e.backend.as_str().to_string(),
        runtime: e.runtime.as_str().to_string(),
        runtime_mode: e.runtime_mode.clone(),
        runtime_version: e.runtime_version.clone(),
        hardware_class: e.hardware_class.clone(),
        runtime_endpoint: e.runtime_endpoint.clone(),
        supported_operations: e.supported_operations.clone(),
        max_inflight: e.max_inflight,
        inflight,
        saturation: inflight as f64 / e.max_inflight.max(1) as f64,
        health: e.health.as_str().to_string(),
        drain: e.drain,
        last_heartbeat_age_ms: e.last_heartbeat.elapsed().as_millis() as u64,
        thermal_state: e.thermal_state.clone(),
        rss_bytes: e.rss_bytes,
        friendly_name: e.friendly_name.clone(),
        chip_model: e.chip_model.clone(),
        worker_pool: e.worker_pool.clone(),
        node_class: e.node_class.clone(),
        active_sequences: e.active_sequences,
        decode_tok_per_sec: e.decode_tok_per_sec,
        ttft_p95_ms: e.ttft_p95_ms,
        queue_depth: e.queue_depth,
        error_rate: e.error_rate,
        kv_pages_used: e.kv_pages_used,
        kv_pages_total: e.kv_pages_total,
        kv_utilization,
        prefix_reusable_tokens: e.prefix_reusable_tokens,
        active_batch_size: e.active_batch_size,
        max_batch_size: e.max_batch_size,
        batch_utilization,
    }
}

fn supported_operations_from_capabilities(capabilities: &WorkerCapabilities) -> Vec<String> {
    let mut operations = Vec::new();
    if capabilities.llm {
        operations.push("llm".to_string());
    }
    if capabilities.embedding {
        operations.push("embedding".to_string());
    }
    if capabilities.vision {
        operations.push("vision".to_string());
    }
    operations
}

fn supported_operations_from_model_inventory(inventory: &[ModelInventoryEntry]) -> Vec<String> {
    let mut seen = FxHashSet::default();
    let mut operations = Vec::new();
    for operation in inventory
        .iter()
        .flat_map(|model| model.supported_operations.iter())
    {
        if seen.insert(operation.clone()) {
            operations.push(operation.clone());
        }
    }
    operations
}

fn max_context_from_model_inventory(inventory: &[ModelInventoryEntry]) -> Option<u32> {
    inventory.iter().filter_map(|model| model.max_context).max()
}

fn refresh_capabilities_from_inventory_summary(
    capabilities: &mut WorkerCapabilities,
    operations: &[String],
    max_context: Option<u32>,
    clear_missing_max_context: bool,
) {
    if !operations.is_empty() {
        capabilities.llm = operations.iter().any(|op| op == "llm");
        capabilities.embedding = operations.iter().any(|op| op == "embedding");
        capabilities.vision = operations.iter().any(|op| op == "vision");
    }
    if max_context.is_some() || clear_missing_max_context {
        capabilities.max_context = max_context;
    }
}

fn normalize_supported_operations(operations: Vec<String>) -> Vec<String> {
    let mut seen = FxHashSet::default();
    operations
        .into_iter()
        .map(|op| op.trim().to_ascii_lowercase().replace('-', "_"))
        .filter(|op| !op.is_empty())
        .filter(|op| seen.insert(op.clone()))
        .collect()
}

fn normalize_model_inventory(
    model_ids: &[String],
    inventory: Vec<ModelInventoryEntry>,
) -> Vec<ModelInventoryEntry> {
    let mut by_id = std::collections::BTreeMap::<String, ModelInventoryEntry>::new();
    for mut item in inventory {
        item.id = item.id.trim().to_string();
        if item.id.is_empty() {
            continue;
        }
        item.modalities.sort();
        item.modalities.dedup();
        item.supported_operations = normalize_supported_operations(item.supported_operations);
        by_id.insert(item.id.clone(), item);
    }
    for id in model_ids {
        if !id.trim().is_empty() {
            by_id
                .entry(id.clone())
                .or_insert_with(|| ModelInventoryEntry {
                    id: id.clone(),
                    ..Default::default()
                });
        }
    }
    by_id.into_values().collect()
}

fn retain_model_inventory_for_ids(
    previous: &[ModelInventoryEntry],
    model_ids: &[String],
) -> Vec<ModelInventoryEntry> {
    let retained = previous
        .iter()
        .filter(|entry| model_ids.iter().any(|id| id == &entry.id))
        .cloned()
        .collect();
    normalize_model_inventory(model_ids, retained)
}

fn normalize_runtime_mode(mode: Option<String>) -> Option<String> {
    mode.and_then(|value| {
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    })
}

fn ratio_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn non_negative_finite(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn worker_status_of(e: &WorkerEntry) -> WorkerStatus {
    let kv_utilization = worker_kv_utilization(e);
    let batch_headroom = worker_batch_utilization(e).map(|value| 1.0 - value);
    WorkerStatus {
        id: e.id,
        addr: e.addr,
        inflight: effective_inflight(e),
        max_inflight: e.max_inflight,
        active_sequences: e.active_sequences,
        ttft_p95_ms: e.ttft_p95_ms,
        kv_utilization,
        batch_headroom,
    }
}

fn effective_inflight(e: &WorkerEntry) -> usize {
    e.inflight.load(Ordering::Relaxed).max(e.reported_inflight)
}

fn worker_kv_utilization(e: &WorkerEntry) -> Option<f64> {
    if e.kv_pages_total > 0 {
        Some((e.kv_pages_used as f64 / e.kv_pages_total as f64).clamp(0.0, 1.0))
    } else {
        e.kv_utilization
    }
}

fn worker_batch_utilization(e: &WorkerEntry) -> Option<f64> {
    if e.max_batch_size > 0 {
        Some((e.active_batch_size as f64 / e.max_batch_size as f64).clamp(0.0, 1.0))
    } else {
        e.batch_utilization
    }
}

fn dispatch_filter_matches(
    entry: &WorkerEntry,
    model_id: &str,
    request_kind: RequestKind,
    backend_filter: Option<&BackendKind>,
    runtime_filter: Option<&RuntimeKind>,
    min_context: Option<u32>,
    excluded_id: Option<WorkerId>,
) -> bool {
    excluded_id != Some(entry.id)
        && !entry.drain
        && matches!(entry.health, WorkerHealth::Healthy)
        && entry.capabilities.models.iter().any(|c| c == model_id)
        && supports_request_kind(entry, request_kind)
        && model_inventory_supports_request_kind(entry, model_id, request_kind)
        && backend_filter.is_none_or(|kind| &entry.backend == kind)
        && runtime_filter.is_none_or(|kind| &entry.runtime == kind)
        && model_context_supports_request(entry, model_id, min_context)
}

fn model_context_supports_request(
    entry: &WorkerEntry,
    model_id: &str,
    min_context: Option<u32>,
) -> bool {
    let Some(required) = min_context else {
        return true;
    };

    let worker_context_ok = entry
        .capabilities
        .max_context
        .is_none_or(|worker_max| worker_max >= required);
    if !worker_context_ok {
        return false;
    }

    entry
        .model_inventory
        .iter()
        .find(|model| model.id == model_id)
        .and_then(|model| model.max_context)
        .is_none_or(|model_max| model_max >= required)
}

fn model_inventory_supports_request_kind(
    entry: &WorkerEntry,
    model_id: &str,
    request_kind: RequestKind,
) -> bool {
    entry
        .model_inventory
        .iter()
        .find(|model| model.id == model_id)
        .is_none_or(|model| {
            model.supported_operations.is_empty()
                || model
                    .supported_operations
                    .iter()
                    .any(|operation| operation == request_kind.as_operation())
        })
}

fn supports_request_kind(entry: &WorkerEntry, request_kind: RequestKind) -> bool {
    if !entry.supported_operations.is_empty()
        && !entry
            .supported_operations
            .iter()
            .any(|operation| operation == request_kind.as_operation())
    {
        return false;
    }

    match entry.capability_source {
        // Compatibility path: legacy workers historically routed by model-id only.
        CapabilitySource::Legacy => true,
        CapabilitySource::Structured => match request_kind {
            RequestKind::Llm => entry.capabilities.llm,
            RequestKind::Embedding => entry.capabilities.embedding,
        },
    }
}

impl RequestKind {
    fn as_operation(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Embedding => "embedding",
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn reg_req(addr: &str, caps: &[&str], max: usize) -> RegisterRequest {
        RegisterRequest {
            worker_id: None,
            addr: addr.into(),
            capabilities: RegisterCapabilities::Legacy(
                caps.iter().map(|s| s.to_string()).collect(),
            ),
            backend: "native".into(),
            max_inflight: max,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: None,
            ..Default::default()
        }
    }

    #[test]
    fn register_and_eligible() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["llama3-8b"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let workers = r.eligible_workers("llama3-8b");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, id);

        // Unknown model → empty
        assert!(r.eligible_workers("unknown-model").is_empty());
    }

    #[test]
    fn register_caps_max_inflight() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], usize::MAX), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.max_inflight, MAX_WORKER_INFLIGHT);
    }

    #[test]
    fn dispatch_workers_prefer_matching_pool_when_available() {
        let r = WorkerRegistry::new();
        let blue = r.register(
            RegisterRequest {
                worker_pool: Some("blue".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_pool: Some("green".into()),
                ..reg_req("127.0.0.1:8082", &["m1"], 4)
            },
            5000,
        );

        let workers =
            r.dispatch_workers_filtered("m1", RequestKind::Llm, None, None, Some("blue"), None);

        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, WorkerId::parse(&blue.worker_id).unwrap());
    }

    #[test]
    fn dispatch_workers_fall_back_when_preferred_pool_missing() {
        let r = WorkerRegistry::new();
        let blue = r.register(
            RegisterRequest {
                worker_pool: Some("blue".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        let green = r.register(
            RegisterRequest {
                worker_pool: Some("green".into()),
                ..reg_req("127.0.0.1:8082", &["m1"], 4)
            },
            5000,
        );

        let workers =
            r.dispatch_workers_filtered("m1", RequestKind::Llm, None, None, Some("red"), None);

        assert_eq!(workers.len(), 2);
        assert!(
            workers
                .iter()
                .any(|worker| worker.id == WorkerId::parse(&blue.worker_id).unwrap())
        );
        assert!(
            workers
                .iter()
                .any(|worker| worker.id == WorkerId::parse(&green.worker_id).unwrap())
        );
    }

    #[test]
    fn dispatch_workers_apply_exclusion_after_pool_preference() {
        let r = WorkerRegistry::new();
        let blue = r.register(
            RegisterRequest {
                worker_pool: Some("blue".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_pool: Some("green".into()),
                ..reg_req("127.0.0.1:8082", &["m1"], 4)
            },
            5000,
        );

        let workers = r.dispatch_workers_filtered(
            "m1",
            RequestKind::Llm,
            None,
            None,
            Some("blue"),
            Some(WorkerId::parse(&blue.worker_id).unwrap()),
        );

        assert!(
            workers.is_empty(),
            "preferred-pool filtering must still win before exclusion"
        );
    }

    #[test]
    fn parse_sglang_backend() {
        assert_eq!(BackendKind::parse("sglang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg_lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg-lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::SgLang.as_str(), "sglang");
    }

    #[test]
    fn parse_vllm_backend() {
        assert_eq!(BackendKind::parse("vllm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v_llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v-llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::Vllm.as_str(), "vllm");
        assert_eq!(
            RuntimeKind::from_backend(&BackendKind::Vllm),
            RuntimeKind::Vllm
        );
    }

    #[test]
    fn structured_capabilities_registration_is_preserved() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["embed-1".into()],
                    max_context: Some(8192),
                }),
                backend: "sglang".into(),
                max_inflight: 8,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("thor".into()),
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.backend, "sglang");
        assert_eq!(snapshot.runtime, "sglang");
        assert_eq!(snapshot.capabilities, vec!["embed-1".to_string()]);
        assert_eq!(
            snapshot.supported_operations,
            vec!["llm".to_string(), "embedding".to_string()]
        );
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(snapshot.capability_descriptor.max_context, Some(8192));
    }

    #[test]
    fn structured_embedding_only_worker_is_not_llm_eligible() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: true,
                    vision: false,
                    models: vec!["embed-1".into()],
                    max_context: None,
                }),
                backend: "sglang".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("thor".into()),
                ..Default::default()
            },
            5000,
        );

        assert!(r.eligible_workers("embed-1").is_empty());
        assert_eq!(
            r.eligible_workers_for("embed-1", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn model_inventory_operations_constrain_mixed_runtime_models() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["chat-model".into(), "embed-model".into()],
                    max_context: None,
                }),
                model_inventory: vec![
                    ModelInventoryEntry {
                        id: "chat-model".into(),
                        supported_operations: vec!["llm".into()],
                        ..Default::default()
                    },
                    ModelInventoryEntry {
                        id: "embed-model".into(),
                        supported_operations: vec!["embedding".into()],
                        ..Default::default()
                    },
                ],
                supported_operations: vec!["llm".into(), "embedding".into()],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(r.eligible_workers("chat-model").len(), 1);
        assert!(
            r.eligible_workers_for("chat-model", RequestKind::Embedding)
                .is_empty(),
            "chat-only model inventory must reject embedding requests"
        );
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "embedding-only model inventory must reject llm requests"
        );
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn registration_routes_models_from_inventory_when_capability_models_absent() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "inventory-model".into(),
                    max_context: Some(32768),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["inventory-model".to_string()]);
        assert_eq!(snapshot.model_inventory[0].id, "inventory-model");
        assert_eq!(r.eligible_workers("inventory-model").len(), 1);
    }

    #[test]
    fn registration_refreshes_structured_operations_from_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: None,
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    max_context: Some(8192),
                    supported_operations: vec!["embedding".into()],
                    ..Default::default()
                }],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["embed-model".to_string()]);
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(snapshot.capability_descriptor.max_context, Some(8192));
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "inventory-only embedding registration must not be llm eligible"
        );
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn registration_preserves_explicit_max_context_when_inventory_omits_it() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: Some(4096),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "chat-model".into(),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capability_descriptor.max_context, Some(4096));
        assert_eq!(
            r.eligible_workers_filtered("chat-model", RequestKind::Llm, None, Some(4096))
                .len(),
            1
        );
        assert!(
            r.eligible_workers_filtered("chat-model", RequestKind::Llm, None, Some(4097))
                .is_empty()
        );
    }

    #[test]
    fn registration_treats_model_inventory_as_additive() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["capability-model".into()],
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "inventory-model".into(),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(r.eligible_workers("capability-model").len(), 1);
        assert_eq!(r.eligible_workers("inventory-model").len(), 1);
    }

    #[test]
    fn legacy_worker_explicit_llm_only_operations_are_not_embedding_eligible() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                supported_operations: vec!["llm".into()],
                ..reg_req("127.0.0.1:8081", &["shared-model"], 4)
            },
            5000,
        );

        assert_eq!(r.eligible_workers("shared-model").len(), 1);
        assert!(
            r.eligible_workers_for("shared-model", RequestKind::Embedding)
                .is_empty(),
            "explicit supported_operations must constrain legacy worker routing"
        );
    }

    #[test]
    fn legacy_worker_without_explicit_operations_keeps_model_id_compatibility() {
        let r = WorkerRegistry::new();
        r.register(reg_req("127.0.0.1:8081", &["shared-model"], 4), 5000);

        assert_eq!(
            r.eligible_workers_for("shared-model", RequestKind::Embedding)
                .len(),
            1,
            "legacy model-id-only registrations remain backward compatible"
        );
    }

    #[test]
    fn explicit_operations_are_normalized_and_deduplicated() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["shared-model".into()],
                    max_context: None,
                }),
                supported_operations: vec![
                    " LLM ".into(),
                    "embedding".into(),
                    "llm".into(),
                    "text-generation".into(),
                    "text_generation".into(),
                ],
                backend: "native".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(
            snapshot.supported_operations,
            vec![
                "llm".to_string(),
                "embedding".to_string(),
                "text_generation".to_string(),
            ]
        );
    }

    #[test]
    fn backend_hint_filters_workers() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["shared-model".into()],
                    max_context: Some(4096),
                }),
                backend: "native".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("mac".into()),
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["shared-model".into()],
                    max_context: Some(16384),
                }),
                backend: "sglang".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("thor".into()),
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("shared-model", RequestKind::Llm, Some("sglang"), None)
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("shared-model", RequestKind::Llm, Some("native"), None)
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("shared-model", RequestKind::Llm, Some("unknown"), None)
                .len(),
            2
        );
    }

    #[test]
    fn runtime_hint_filters_workers() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["runtime-model".into()],
                    max_context: Some(4096),
                }),
                backend: "auto".into(),
                runtime: Some("ax_engine".into()),
                max_inflight: 4,
                node_class: Some("mac".into()),
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["runtime-model".into()],
                    max_context: Some(16384),
                }),
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                max_inflight: 4,
                node_class: Some("pc-cuda".into()),
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("runtime-model", RequestKind::Llm, Some("ax_engine"), None)
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("runtime-model", RequestKind::Llm, Some("vllm"), None)
                .len(),
            1
        );
    }

    #[test]
    fn vllm_worker_exposes_runtime_metadata() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: true,
                    models: vec!["qwen3-32b".into()],
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "qwen3-32b".into(),
                    max_context: Some(32768),
                    quantization: Some("awq".into()),
                    artifact_format: Some("safetensors".into()),
                    modalities: vec!["text".into()],
                    supported_operations: vec!["llm".into(), "vision".into()],
                }],
                backend: "vllm".into(),
                runtime_mode: Some("adapter".into()),
                max_inflight: 16,
                friendly_name: None,
                chip_model: None,
                worker_pool: Some("cuda".into()),
                node_class: Some("pc-cuda".into()),
                runtime: Some("vllm".into()),
                runtime_version: Some("0.13.0".into()),
                hardware_class: Some("pc-cuda".into()),
                runtime_endpoint: Some("http://127.0.0.1:8000".into()),
                supported_operations: vec!["llm".into(), "vision".into()],
            },
            5000,
        );

        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.backend, "vllm");
        assert_eq!(snapshot.runtime, "vllm");
        assert_eq!(snapshot.runtime_mode.as_deref(), Some("adapter"));
        assert_eq!(snapshot.runtime_version.as_deref(), Some("0.13.0"));
        assert_eq!(snapshot.hardware_class.as_deref(), Some("pc-cuda"));
        assert_eq!(
            snapshot.runtime_endpoint.as_deref(),
            Some("http://127.0.0.1:8000")
        );
        assert_eq!(
            snapshot.supported_operations,
            vec!["llm".to_string(), "vision".to_string()]
        );
        assert_eq!(snapshot.model_inventory.len(), 1);
        assert_eq!(
            snapshot.model_inventory[0].quantization.as_deref(),
            Some("awq")
        );
        assert_eq!(
            snapshot.model_inventory[0].artifact_format.as_deref(),
            Some("safetensors")
        );
        assert_eq!(
            r.eligible_workers_filtered("qwen3-32b", RequestKind::Llm, Some("vllm"), None)
                .len(),
            1
        );
    }

    #[test]
    fn min_context_filters_workers() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["ctx-model".into()],
                    max_context: Some(4096),
                }),
                backend: "native".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["ctx-model".into()],
                    max_context: Some(16384),
                }),
                backend: "sglang".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("ctx-model", RequestKind::Llm, None, Some(8000))
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("ctx-model", RequestKind::Llm, None, Some(20000))
                .len(),
            0
        );
    }

    #[test]
    fn min_context_respects_model_inventory_context_limit() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["short-model".into()],
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "short-model".into(),
                    max_context: Some(4096),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("short-model", RequestKind::Llm, None, Some(4096))
                .len(),
            1
        );
        assert!(
            r.eligible_workers_filtered("short-model", RequestKind::Llm, None, Some(8000))
                .is_empty(),
            "per-model context limit must override broader worker context"
        );
    }

    #[test]
    fn reregister_is_idempotent() {
        let r = WorkerRegistry::new();
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = resp1.worker_id.clone();

        // Re-register with same id — updates capabilities
        let mut req2 = reg_req("127.0.0.1:8081", &["m1", "m2"], 8);
        req2.worker_id = Some(id1.clone());
        let resp2 = r.register(req2, 5000);

        assert_eq!(resp2.worker_id, id1);
        assert_eq!(r.eligible_workers("m2").len(), 1);
        assert_eq!(r.list_all().len(), 1); // still one entry
    }

    #[test]
    fn reregister_without_inventory_preserves_matching_model_metadata() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    models: vec!["m1".into()],
                    ..Default::default()
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    quantization: Some("Q4_K_M".into()),
                    artifact_format: Some("gguf".into()),
                    ..Default::default()
                }],
                backend: "native".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let mut req2 = reg_req("127.0.0.1:8081", &["m1"], 8);
        req2.worker_id = Some(resp.worker_id);
        r.register(req2, 5000);

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.model_inventory.len(), 1);
        assert_eq!(snapshot.model_inventory[0].id, "m1");
        assert_eq!(
            snapshot.model_inventory[0].quantization.as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            snapshot.model_inventory[0].artifact_format.as_deref(),
            Some("gguf")
        );
    }

    #[test]
    fn reregister_without_models_clears_stale_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    models: vec!["m1".into()],
                    ..Default::default()
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    quantization: Some("Q4_K_M".into()),
                    ..Default::default()
                }],
                backend: "native".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let mut req2 = reg_req("127.0.0.1:8081", &[], 8);
        req2.worker_id = Some(resp.worker_id);
        r.register(req2, 5000);

        let snapshot = r.get_snapshot(id).unwrap();
        assert!(snapshot.model_inventory.is_empty());
        assert!(snapshot.capability_descriptor.models.is_empty());
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn drain_removes_from_eligible() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert_eq!(r.eligible_workers("m1").len(), 1);
        r.mark_drain(id);
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn eligible_healthy_count_excludes_draining() {
        let r = WorkerRegistry::new();

        // Register two workers.
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = WorkerId::parse(&resp1.worker_id).unwrap();
        let resp2 = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);
        let id2 = WorkerId::parse(&resp2.worker_id).unwrap();

        // Both healthy, neither draining → eligible = 2.
        assert_eq!(r.eligible_healthy_count(), 2);

        // Drain worker 1 — still healthy but not eligible.
        r.mark_drain(id1);
        assert_eq!(r.eligible_healthy_count(), 1);

        // Mark worker 2 unhealthy — now eligible = 0.
        r.mark_unhealthy(id2);
        assert_eq!(r.eligible_healthy_count(), 0);

        // counts() returns healthy=1 (worker 1 is Healthy but draining).
        // eligible_healthy_count() returns 0 because draining workers are
        // excluded from dispatch even if their health state is Healthy.
        let (healthy, _unhealthy, _draining) = r.counts();
        assert_eq!(healthy, 1); // worker 1 is still Healthy (just draining)
        // eligible_healthy_count correctly reports 0 despite healthy=1.
        assert_eq!(r.eligible_healthy_count(), 0);
    }

    #[test]
    fn unhealthy_removed_from_eligible_until_heartbeat() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert_eq!(r.eligible_workers("m1").len(), 1);
        r.mark_unhealthy(id);
        assert!(
            r.eligible_workers("m1").is_empty(),
            "unhealthy workers must be excluded from dispatch eligibility"
        );

        // A fresh heartbeat restores the worker to healthy/eligible.
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));
        assert_eq!(r.eligible_workers("m1").len(), 1);
    }

    #[test]
    fn tick_preserves_recent_unhealthy_until_heartbeat() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.mark_unhealthy(id);
        r.tick(9_000);

        assert_eq!(
            r.inner.get(&id).unwrap().health,
            WorkerHealth::Unhealthy { missed: 1 },
            "tick must not erase a dispatch failure before a heartbeat restores health"
        );
        assert!(r.eligible_workers("m1").is_empty());
        assert!(
            r.list_unhealthy_addrs()
                .iter()
                .any(|(candidate_id, _)| *candidate_id == id),
            "health ticker must still see the failed worker as a probe candidate"
        );

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        assert_eq!(
            r.inner.get(&id).unwrap().health,
            WorkerHealth::Healthy,
            "heartbeat is the signal that restores a dispatch-failed worker"
        );
        assert_eq!(r.eligible_workers("m1").len(), 1);
    }

    #[test]
    fn evict_removes_entry() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.evict(id);
        assert!(r.get_snapshot(id).is_none());
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn tick_evicts_stale_workers() {
        let r = WorkerRegistry::new();

        // Register with a past last_heartbeat by manipulating via a fake entry.
        // We can't set last_heartbeat directly from outside, so we test tick
        // with a very small ttl_ms (1 ms) so any entry looks stale immediately.
        r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);

        // With ttl=1ms, any worker will appear stale after the first tick.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let evicted = r.tick(1);
        assert!(!evicted.is_empty());
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn heartbeat_updates_capabilities_from_model_ids() {
        let r = WorkerRegistry::new();
        // Register with no initial capabilities.
        let resp = r.register(reg_req("127.0.0.1:8081", &[], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        // Worker not eligible for any model yet.
        assert!(r.eligible_workers("m1").is_empty());

        // Heartbeat carries model_ids — orchestrator now knows worker has m1 loaded.
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 256 * 1024 * 1024,
                ..Default::default()
            }
        ));
        assert_eq!(r.eligible_workers("m1").len(), 1);

        // Subsequent heartbeat with empty model_ids clears capabilities.
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec![],
                rss_bytes: 0,
                ..Default::default()
            }
        ));
        assert!(
            r.eligible_workers("m1").is_empty(),
            "empty model_ids heartbeat must clear stale capabilities"
        );
    }

    #[test]
    fn heartbeat_treats_model_inventory_as_additive() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1", "m2"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                model_ids: vec!["m1".into(), "m2".into()],
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    quantization: Some("q4".into()),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }
        ));

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(
            snapshot.capabilities,
            vec!["m1".to_string(), "m2".to_string()]
        );
        assert_eq!(snapshot.model_inventory.len(), 2);
        assert_eq!(r.eligible_workers("m1").len(), 1);
        assert_eq!(r.eligible_workers("m2").len(), 1);
    }

    #[test]
    fn heartbeat_refreshes_structured_operations_from_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["chat-model".into()],
                    max_context: Some(2048),
                }),
                supported_operations: vec!["llm".into()],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                model_ids: vec!["embed-model".into()],
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    max_context: Some(8192),
                    supported_operations: vec!["embedding".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }
        ));

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["embed-model".to_string()]);
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert_eq!(snapshot.capability_descriptor.max_context, Some(8192));
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "heartbeat inventory must remove stale llm eligibility"
        );
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn heartbeat_stores_thermal_and_rss() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 3,
                thermal_state: "serious".into(),
                model_ids: vec![],
                rss_bytes: 1_073_741_824, // 1 GiB
                ..Default::default()
            },
        );

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.thermal_state, "serious");
        assert_eq!(snap.rss_bytes, 1_073_741_824);
        assert_eq!(snap.inflight, 3);
    }

    #[test]
    fn heartbeat_does_not_overwrite_dispatcher_inflight_counter() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let counter = r.inflight_counter(id).expect("registered worker counter");

        counter.fetch_add(1, Ordering::Relaxed);
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(
            snap.inflight, 1,
            "heartbeat must not erase dispatcher-owned in-flight accounting"
        );
        assert_eq!(r.eligible_workers("m1")[0].inflight, 1);

        counter.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(r.get_snapshot(id).unwrap().inflight, 0);
    }

    #[test]
    fn heartbeat_reported_inflight_is_used_when_dispatch_counter_is_lower() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 2,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        assert_eq!(r.get_snapshot(id).unwrap().inflight, 2);
        assert_eq!(r.eligible_workers("m1")[0].inflight, 2);
    }

    #[test]
    fn heartbeat_clamps_runtime_load_telemetry() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: usize::MAX,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                active_sequences: usize::MAX,
                decode_tok_per_sec: f64::INFINITY,
                queue_depth: usize::MAX,
                error_rate: 2.5,
                kv_utilization: Some(f64::NAN),
                batch_utilization: Some(f64::INFINITY),
                ..Default::default()
            }
        ));

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.inflight, MAX_WORKER_INFLIGHT);
        assert_eq!(snap.active_sequences, MAX_WORKER_INFLIGHT);
        assert_eq!(snap.decode_tok_per_sec, 0.0);
        assert_eq!(snap.queue_depth, MAX_WORKER_INFLIGHT);
        assert_eq!(snap.error_rate, 1.0);
        assert_eq!(snap.kv_utilization, Some(0.0));
        assert_eq!(snap.batch_utilization, Some(0.0));

        let worker = r.eligible_workers("m1").remove(0);
        assert_eq!(worker.active_sequences, MAX_WORKER_INFLIGHT);
        assert_eq!(worker.kv_utilization, Some(0.0));
    }

    #[test]
    fn register_stores_identity_fields() {
        let r = WorkerRegistry::new();
        let req = RegisterRequest {
            worker_id: None,
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec![]),
            backend: "auto".into(),
            max_inflight: 4,
            friendly_name: Some("Aki's MacBook Pro".to_string()),
            chip_model: Some("Apple M3 Pro".to_string()),
            worker_pool: Some("blue".to_string()),
            node_class: Some("m3-pro".to_string()),
            ..Default::default()
        };
        let resp = r.register(req, 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.friendly_name.as_deref(), Some("Aki's MacBook Pro"));
        assert_eq!(snap.chip_model.as_deref(), Some("Apple M3 Pro"));
        assert_eq!(snap.worker_pool.as_deref(), Some("blue"));
        assert_eq!(snap.node_class.as_deref(), Some("m3-pro"));
    }

    #[test]
    fn reregister_clears_stale_identity_and_routing_fields() {
        let r = WorkerRegistry::new();
        let req = RegisterRequest {
            worker_id: None,
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
            backend: "vllm".into(),
            runtime: Some("vllm".into()),
            runtime_mode: Some("adapter".into()),
            runtime_version: Some("0.13.0".into()),
            hardware_class: Some("pc-cuda".into()),
            runtime_endpoint: Some("http://127.0.0.1:8000".into()),
            max_inflight: 4,
            friendly_name: Some("node-a".to_string()),
            chip_model: Some("NVIDIA L40S".to_string()),
            worker_pool: Some("blue".to_string()),
            node_class: Some("pc-cuda".to_string()),
            ..Default::default()
        };
        let resp = r.register(req, 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let req = RegisterRequest {
            worker_id: Some(resp.worker_id),
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
            backend: "vllm".into(),
            runtime: Some("vllm".into()),
            runtime_mode: Some("adapter".into()),
            max_inflight: 4,
            ..Default::default()
        };
        r.register(req, 5000);

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.runtime_version, None);
        assert_eq!(snap.hardware_class, None);
        assert_eq!(snap.runtime_endpoint, None);
        assert_eq!(snap.friendly_name, None);
        assert_eq!(snap.chip_model, None);
        assert_eq!(snap.worker_pool, None);
        assert_eq!(snap.node_class, None);
        assert!(
            r.dispatch_workers_filtered_with_pool_mode(
                "m1",
                RequestKind::Llm,
                None,
                None,
                Some("blue"),
                true,
                None,
            )
            .is_empty()
        );
    }

    #[test]
    fn counts_basic_breakdown() {
        let r = WorkerRegistry::new();

        // Two healthy, one to be drained, one to be marked unhealthy.
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = WorkerId::parse(&resp1.worker_id).unwrap();
        let _resp2 = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);
        let resp3 = r.register(reg_req("127.0.0.1:8083", &["m1"], 4), 5000);
        let id3 = WorkerId::parse(&resp3.worker_id).unwrap();
        let resp4 = r.register(reg_req("127.0.0.1:8084", &["m1"], 4), 5000);
        let id4 = WorkerId::parse(&resp4.worker_id).unwrap();

        // Initially all healthy, none draining.
        let (h, u, d) = r.counts();
        assert_eq!(h, 4);
        assert_eq!(u, 0);
        assert_eq!(d, 0);

        r.mark_drain(id1);
        r.mark_unhealthy(id3);

        // id1: Healthy+drain; id3: Unhealthy; id4: Healthy (for eviction below)
        let _ = id4;

        let (h, u, d) = r.counts();
        assert_eq!(
            h, 3,
            "id1 is still Healthy (just draining), id2+id4 healthy"
        );
        assert_eq!(u, 1, "id3 is unhealthy");
        assert_eq!(d, 1, "id1 is draining");
    }

    #[test]
    fn list_unhealthy_addrs_returns_unhealthy() {
        let r = WorkerRegistry::new();
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = WorkerId::parse(&resp1.worker_id).unwrap();
        let _resp2 = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);

        // Initially no unhealthy workers.
        assert!(r.list_unhealthy_addrs().is_empty());

        r.mark_unhealthy(id1);
        let unhealthy = r.list_unhealthy_addrs();
        assert_eq!(unhealthy.len(), 1);
        assert_eq!(unhealthy[0].0, id1);
        assert_eq!(unhealthy[0].1.to_string(), "127.0.0.1:8081");
    }

    #[test]
    fn backend_kind_parse_variants() {
        assert_eq!(BackendKind::parse("llama_cpp"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("llamacpp"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("llama-cpp"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("LLAMA_CPP"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("sglang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg_lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg-lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("vllm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v_llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v-llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("native"), BackendKind::Native);
        assert_eq!(BackendKind::parse("NATIVE"), BackendKind::Native);
        assert_eq!(BackendKind::parse("auto"), BackendKind::Auto);
        assert_eq!(BackendKind::parse("unknown"), BackendKind::Auto);
        assert_eq!(BackendKind::parse(""), BackendKind::Auto);
    }

    #[test]
    fn worker_id_display_and_parse_roundtrip() {
        let id = WorkerId::new();
        let s = id.to_string();
        let parsed = WorkerId::parse(&s).expect("must parse valid UUID string");
        assert_eq!(id, parsed);
        assert!(WorkerId::parse("not-a-uuid").is_none());
    }

    // ── register: invalid address falls back gracefully ───────────────────────

    #[test]
    fn register_invalid_addr_falls_back_to_loopback_sentinel() {
        let r = WorkerRegistry::new();
        // A bad address must not poison the registry — the worker is registered
        // with the "127.0.0.1:1" sentinel so it never receives real traffic.
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "not-a-valid:addr:at:all".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".to_string()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snap = r.get_snapshot(id).unwrap();
        // The sentinel address is "127.0.0.1:1".
        assert_eq!(snap.addr, "127.0.0.1:1");
        // Other fields should still be set correctly.
        assert_eq!(snap.max_inflight, 4);
        // The registry should still contain this entry (not poisoned/absent).
        assert_eq!(r.list_all().len(), 1);
    }

    // ── tick: full health state-machine transition matrix ─────────────────────

    #[test]
    fn tick_health_state_transitions_all_four_stages() {
        let r = WorkerRegistry::new();
        // ttl = 9000 ms → boundaries at ttl/3 = 3000 ms, 2*ttl/3 = 6000 ms.
        let ttl_ms = 9_000u64;

        // Helper: register a worker then backdates its last_heartbeat.
        let make_aged = |age_ms: u64| -> WorkerId {
            let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
            let id = WorkerId::parse(&resp.worker_id).unwrap();
            let past = std::time::Instant::now()
                .checked_sub(std::time::Duration::from_millis(age_ms))
                .expect("test machine must have been running for at least 10 s");
            r.inner.get_mut(&id).unwrap().last_heartbeat = past;
            id
        };

        let id_healthy = make_aged(1_000); // 1 s → Healthy   (≤ 3 s)
        let id_miss1 = make_aged(4_000); // 4 s → Unhealthy{1} (3 s < age ≤ 6 s)
        let id_miss2 = make_aged(7_000); // 7 s → Unhealthy{2} (6 s < age ≤ 9 s)
        let id_dead = make_aged(10_000); // 10 s → Dead         (> 9 s)

        let evicted = r.tick(ttl_ms);

        assert_eq!(evicted.len(), 1, "only the dead worker should be evicted");
        assert!(evicted.contains(&id_dead));
        assert!(
            r.inner.get(&id_dead).is_none(),
            "dead worker must be removed"
        );

        assert_eq!(
            r.inner.get(&id_healthy).unwrap().health,
            WorkerHealth::Healthy
        );
        assert_eq!(
            r.inner.get(&id_miss1).unwrap().health,
            WorkerHealth::Unhealthy { missed: 1 }
        );
        assert_eq!(
            r.inner.get(&id_miss2).unwrap().health,
            WorkerHealth::Unhealthy { missed: 2 }
        );
    }

    // ── tick: draining workers ─────────────────────────────────────────────────

    #[test]
    fn tick_draining_worker_evicted_only_after_ttl() {
        let r = WorkerRegistry::new();
        let ttl_ms = 9_000u64;

        // Register two draining workers: one fresh, one stale.
        let resp_fresh = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id_fresh = WorkerId::parse(&resp_fresh.worker_id).unwrap();
        r.mark_drain(id_fresh);

        let resp_stale = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);
        let id_stale = WorkerId::parse(&resp_stale.worker_id).unwrap();
        r.mark_drain(id_stale);
        // Backdate the stale worker past the TTL.
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(ttl_ms + 1_000))
            .unwrap();
        r.inner.get_mut(&id_stale).unwrap().last_heartbeat = past;

        let evicted = r.tick(ttl_ms);

        assert!(
            evicted.contains(&id_stale),
            "stale draining worker must be evicted"
        );
        assert!(
            !evicted.contains(&id_fresh),
            "fresh draining worker must not be evicted yet"
        );
        assert!(r.inner.get(&id_fresh).is_some());
    }

    // ── mark_unhealthy: idempotent — already-unhealthy stays at missed:1 ───────

    #[test]
    fn mark_unhealthy_is_idempotent_does_not_escalate() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.mark_unhealthy(id);
        // Second call on an already-unhealthy worker must not escalate missed count.
        r.mark_unhealthy(id);

        assert_eq!(
            r.inner.get(&id).unwrap().health,
            WorkerHealth::Unhealthy { missed: 1 },
            "mark_unhealthy must not escalate beyond missed:1"
        );
    }

    // ── eligible_workers: exact model match (no substring) ───────────────────

    #[test]
    fn eligible_workers_requires_exact_model_id_match() {
        let r = WorkerRegistry::new();
        // Worker has "llama3-8b" — must NOT be returned for "llama3" or "llama3-8b-v2".
        r.register(reg_req("127.0.0.1:8081", &["llama3-8b"], 4), 5000);

        assert_eq!(
            r.eligible_workers("llama3-8b").len(),
            1,
            "exact match must work"
        );
        assert!(
            r.eligible_workers("llama3").is_empty(),
            "substring 'llama3' must not match 'llama3-8b'"
        );
        assert!(
            r.eligible_workers("llama3-8b-v2").is_empty(),
            "extended name must not match shorter capability"
        );
    }

    // ── mark_drain / mark_unhealthy on unknown worker ────────────────────────

    #[test]
    fn mark_drain_returns_false_for_unknown_worker() {
        let r = WorkerRegistry::new();
        assert!(
            !r.mark_drain(WorkerId::new()),
            "mark_drain must return false for an unregistered worker"
        );
    }

    #[test]
    fn mark_unhealthy_noop_for_unknown_worker() {
        // Should not panic — no entry exists, so the call is silently ignored.
        let r = WorkerRegistry::new();
        r.mark_unhealthy(WorkerId::new()); // must not panic
        assert!(r.list_all().is_empty());
    }

    #[test]
    fn heartbeat_resets_health() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        // Heartbeat returns true for a known worker and resets last_heartbeat.
        let hb = HeartbeatRequest {
            inflight: 2,
            thermal_state: "nominal".into(),
            model_ids: vec!["m1".to_string()],
            rss_bytes: 1024 * 1024 * 512,
            ..Default::default()
        };
        assert!(r.heartbeat(id, hb));

        // Heartbeat returns false for an unknown worker.
        assert!(!r.heartbeat(
            WorkerId::new(),
            HeartbeatRequest {
                inflight: 0,
                thermal_state: String::new(),
                model_ids: vec![],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        // After a fresh heartbeat, a tick with a large TTL must not evict the worker.
        let evicted = r.tick(60_000);
        assert!(evicted.is_empty());
        assert_eq!(r.eligible_workers("m1").len(), 1);
    }

    #[test]
    fn heartbeat_stores_cache_telemetry() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::default(),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 2,
                kv_pages_used: 100,
                kv_pages_total: 256,
                active_batch_size: 3,
                max_batch_size: 8,
                ..Default::default()
            },
        );
        let workers = reg.list_all();
        let snap = workers.iter().find(|w| w.id == id).unwrap();
        assert_eq!(snap.kv_pages_used, 100);
        assert_eq!(snap.kv_pages_total, 256);
        assert_eq!(snap.active_batch_size, 3);
        assert_eq!(snap.max_batch_size, 8);
    }

    #[test]
    fn worker_status_computes_kv_utilization_and_batch_headroom() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 1,
                kv_pages_used: 200,
                kv_pages_total: 400,
                active_batch_size: 2,
                max_batch_size: 8,
                model_ids: vec!["m1".into()],
                ..Default::default()
            },
        );
        let eligible = reg.eligible_workers("m1");
        assert_eq!(eligible.len(), 1);
        let ws = &eligible[0];
        assert!((ws.kv_utilization.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!((ws.batch_headroom.unwrap() - 0.75).abs() < f64::EPSILON);

        let all = reg.list_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].kv_pages_used, 200);
        assert_eq!(all[0].kv_pages_total, 400);
        assert_eq!(all[0].kv_utilization, Some(0.5));
        assert_eq!(all[0].batch_utilization, Some(0.25));
    }

    #[test]
    fn worker_status_uses_ratio_telemetry_when_counters_are_absent() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8084".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 1,
                kv_utilization: Some(0.6),
                batch_utilization: Some(0.25),
                model_ids: vec!["m1".into()],
                ..Default::default()
            },
        );

        let eligible = reg.eligible_workers("m1");
        assert_eq!(eligible[0].kv_utilization, Some(0.6));
        assert_eq!(eligible[0].batch_headroom, Some(0.75));

        let all = reg.list_all();
        assert_eq!(all[0].kv_utilization, Some(0.6));
        assert_eq!(all[0].batch_utilization, Some(0.25));
    }

    #[test]
    fn worker_status_clamps_kv_utilization_to_one() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 1,
                kv_pages_used: 500,
                kv_pages_total: 400,
                model_ids: vec!["m1".into()],
                ..Default::default()
            },
        );
        let eligible = reg.eligible_workers("m1");
        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].kv_utilization, Some(1.0));
    }
}
