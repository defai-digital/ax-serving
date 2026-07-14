//! Registry types, DTOs, and identity enums.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use ax_serving_protocol::{
    AgentDescriptor, ProtocolDescriptor, ProtocolVersion,
    RegisterWorkerRequest as ProtocolRegisterRequest, RegistrationId, WorkerInstanceId,
};

use super::super::worker_endpoint::WorkerEndpoint;

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
        match s.trim().to_lowercase().as_str() {
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
        match s.trim().to_lowercase().as_str() {
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
    pub(super) fn from_legacy_models(models: Vec<String>) -> Self {
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
    pub(super) fn into_parts(self) -> (WorkerCapabilities, CapabilitySource) {
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
pub(super) enum CapabilitySource {
    Legacy,
    Structured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Llm,
    Embedding,
    Vision,
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
    pub addr: WorkerEndpoint,
    pub capabilities: WorkerCapabilities,
    /// Optional runtime-reported per-model metadata.
    pub model_inventory: Vec<ModelInventoryEntry>,
    pub(super) capability_source: CapabilitySource,
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
    pub protocol_worker_id: Option<String>,
    pub worker_instance_id: Option<String>,
    pub registration_id: Option<String>,
    pub trust_domain: Option<String>,
    pub agent_name: Option<String>,
    /// Operations the worker supports, e.g. `llm`, `embedding`, `vision`.
    pub supported_operations: Vec<String>,
    pub(super) supported_operations_explicit: bool,
    pub max_inflight: usize,
    /// Dispatcher-owned in-flight count, updated without taking the registry lock.
    pub inflight: Arc<AtomicUsize>,
    /// Last in-flight count reported by the worker heartbeat.
    pub reported_inflight: usize,
    pub health: WorkerHealth,
    pub last_heartbeat: Instant,
    /// Authoritative upstream runtime readiness when reported by a v1-capable agent.
    pub runtime_ready: Option<bool>,
    pub runtime_state: Option<String>,
    pub runtime_status_reason: Option<String>,
    pub observed_at_unix_ms: Option<u64>,
    pub protocol_version: Option<ProtocolVersion>,
    pub agent_version: Option<String>,
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
    /// Runtime-neutral per-model protocol capabilities.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protocol_capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
}

#[derive(Clone)]
pub(super) struct ProtocolSession {
    pub(super) internal_id: WorkerId,
    pub(super) instance_id: WorkerInstanceId,
    pub(super) registration_id: RegistrationId,
    pub(super) lease_token_digest: [u8; 32],
    pub(super) negotiated: ProtocolDescriptor,
    pub(super) agent: AgentDescriptor,
    pub(super) last_sequence: u64,
    pub(super) inventory_generation: u64,
    pub(super) heartbeat_interval_ms: u64,
    pub(super) lease_ttl_ms: u64,
    pub(super) registration: ProtocolRegisterRequest,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolRegistryError {
    #[error("protocol worker is not registered")]
    NotRegistered,
    #[error("worker instance does not match the active registration")]
    InstanceMismatch,
    #[error("registration id does not match the active lease")]
    RegistrationMismatch,
    #[error("missing or invalid worker lease token")]
    InvalidLeaseToken,
    #[error("heartbeat sequence {received} is older than accepted sequence {accepted}")]
    ReplayedHeartbeat { received: u64, accepted: u64 },
    #[error("runtime observation is invalid: {0}")]
    InvalidObservation(String),
    #[error("internal worker registration failed")]
    InternalRegistration,
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
    /// Authoritative readiness of the upstream runtime, not merely the agent process.
    #[serde(default)]
    pub runtime_ready: Option<bool>,
    #[serde(default)]
    pub runtime_state: Option<String>,
    #[serde(default)]
    pub runtime_status_reason: Option<String>,
    #[serde(default)]
    pub observed_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub protocol_version: Option<ProtocolVersion>,
    #[serde(default)]
    pub agent_version: Option<String>,
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
    pub addr: WorkerEndpoint,
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
    /// Runtime-reported pending request count. `None` means the signal is unavailable.
    pub queue_depth: Option<usize>,
    /// Recent runtime error rate. `None` means the signal is unavailable.
    pub error_rate: Option<f64>,
    /// Recent runtime decode throughput. `None` means the signal is unavailable.
    pub decode_tok_per_sec: Option<f64>,
    /// Age of the authoritative runtime observation, when protocol v1 telemetry is available.
    pub telemetry_age_ms: Option<u64>,
}

/// Model-scoped endpoint snapshot used by explicit deployment routing.
#[derive(Clone)]
pub struct WorkerModelEndpoint {
    pub worker: WorkerStatus,
    pub worker_pool: Option<String>,
    pub node_class: Option<String>,
    pub hardware_class: Option<String>,
    pub runtime_kind: String,
    pub runtime_version: Option<String>,
    pub trust_domain: Option<String>,
    pub protocol_worker_id: Option<String>,
    pub worker_instance_id: Option<String>,
    pub registration_id: Option<String>,
    pub model: ModelInventoryEntry,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol_worker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_instance_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registration_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust_domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    pub supported_operations: Vec<String>,
    pub max_inflight: usize,
    pub inflight: usize,
    /// `inflight / max_inflight` — 0.0 when idle, ≥ 1.0 when at or above capacity.
    pub saturation: f64,
    pub health: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_ready: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_status_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<ProtocolVersion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_version: Option<String>,
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

impl RequestKind {
    pub(super) fn as_operation(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Embedding => "embedding",
            Self::Vision => "vision",
        }
    }
}
