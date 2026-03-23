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

use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

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
    Auto,
}

impl BackendKind {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama_cpp" | "llamacpp" | "llama-cpp" => Self::LlamaCpp,
            "sglang" | "sg_lang" | "sg-lang" => Self::SgLang,
            "native" => Self::Native,
            _ => Self::Auto,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LlamaCpp => "llama_cpp",
            Self::SgLang => "sglang",
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
    capability_source: CapabilitySource,
    pub backend: BackendKind,
    pub max_inflight: usize,
    /// Atomically updated by the dispatcher without taking the registry lock.
    pub inflight: Arc<AtomicUsize>,
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
}

// ── Payloads (serialised over the internal REST API) ─────────────────────────

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    /// Omit for first registration; include to re-register with same identity.
    pub worker_id: Option<String>,
    /// `"127.0.0.1:8081"` — loopback address the orchestrator can reach.
    pub addr: String,
    /// Either a legacy model-id list or a structured capability descriptor.
    #[serde(default)]
    pub capabilities: RegisterCapabilities,
    /// `"native"` | `"llama_cpp"` | `"auto"`
    #[serde(default = "default_backend")]
    pub backend: String,
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
    /// Shared with `WorkerEntry` — increment before dispatch, decrement after.
    pub inflight_counter: Arc<AtomicUsize>,
    /// Active inference sequences (token-cost dispatch).  0 = unknown (legacy worker).
    pub active_sequences: usize,
    /// Recent decode throughput in tokens/second (0 = unknown).
    pub decode_tok_per_sec: f64,
    /// P95 TTFT in milliseconds (0 = unknown / no streaming requests yet).
    pub ttft_p95_ms: u64,
    /// Optional pool label for placement hints.
    pub worker_pool: Option<String>,
    /// Optional node class for topology and fleet inventory.
    pub node_class: Option<String>,
    /// Structured worker capabilities used by capability-aware filtering.
    pub capabilities: WorkerCapabilities,
    /// Current pending queue depth reported by the worker.
    pub queue_depth: usize,
    /// Recent worker-side error rate fraction.
    pub error_rate: f64,
}

// ── JSON snapshot for the listing endpoints ───────────────────────────────────

#[derive(Debug, Serialize)]
pub struct WorkerSnapshot {
    pub id: WorkerId,
    pub addr: String,
    pub capabilities: Vec<String>,
    pub capability_descriptor: WorkerCapabilities,
    pub backend: String,
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
            backend,
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
        const SENTINEL_ADDR: &str = "127.0.0.1:1";
        let addr: SocketAddr = raw_addr.parse().unwrap_or_else(|e| {
            warn!(raw_addr = %raw_addr, err = %e, "worker registered with unparseable address; it will never receive traffic");
            SENTINEL_ADDR.parse().unwrap()
        });
        let backend = BackendKind::parse(&backend);
        let (capabilities, capability_source) = capabilities.into_parts();

        self.inner
            .entry(id)
            .and_modify(|existing| {
                // Idempotent re-registration: update mutable fields, reset health.
                existing.addr = addr;
                existing.capabilities = capabilities.clone();
                existing.capability_source = capability_source;
                existing.backend = backend.clone();
                existing.max_inflight = max_inflight;
                existing.health = WorkerHealth::Healthy;
                existing.last_heartbeat = Instant::now();
                existing.drain = false;
                // Preserve richer identity fields if re-registering with them.
                if friendly_name.is_some() {
                    existing.friendly_name = friendly_name.clone();
                }
                if chip_model.is_some() {
                    existing.chip_model = chip_model.clone();
                }
                if worker_pool.is_some() {
                    existing.worker_pool = worker_pool.clone();
                }
                if node_class.is_some() {
                    existing.node_class = node_class.clone();
                }
            })
            .or_insert_with(|| WorkerEntry {
                id,
                addr,
                capabilities,
                capability_source,
                backend,
                max_inflight,
                inflight: Arc::new(AtomicUsize::new(0)),
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
                e.inflight.store(req.inflight, Ordering::Relaxed);
                e.thermal_state = req.thermal_state;
                e.rss_bytes = req.rss_bytes;
                // Authoritative capability snapshot from worker heartbeat.
                // Empty model_ids means the worker currently has no models.
                e.capabilities.models = req.model_ids;
                // Token-cost dispatch telemetry — graceful defaults for legacy workers.
                // active_sequences == 0 and inflight != 0 means the worker doesn't send
                // the extended field; TokenCostPolicy falls back to inflight ratio.
                e.active_sequences = req.active_sequences;
                e.decode_tok_per_sec = req.decode_tok_per_sec;
                e.ttft_p95_ms = req.ttft_p95_ms;
                e.queue_depth = req.queue_depth;
                e.error_rate = req.error_rate;
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
        let backend_filter = backend_filter_from_hint(backend_hint);
        self.inner
            .iter()
            .filter(|r| {
                let e = r.value();
                !e.drain
                    && matches!(e.health, WorkerHealth::Healthy)
                    && e.capabilities.models.iter().any(|c| c == model_id)
                    && supports_request_kind(e, request_kind)
                    && backend_filter
                        .as_ref()
                        .is_none_or(|kind| &e.backend == kind)
                    && min_context.is_none_or(|required| {
                        e.capabilities
                            .max_context
                            .is_none_or(|worker_max| worker_max >= required)
                    })
            })
            .map(|r| {
                let e = r.value();
                WorkerStatus {
                    id: e.id,
                    addr: e.addr,
                    inflight: e.inflight.load(Ordering::Relaxed),
                    max_inflight: e.max_inflight,
                    inflight_counter: Arc::clone(&e.inflight),
                    active_sequences: e.active_sequences,
                    decode_tok_per_sec: e.decode_tok_per_sec,
                    ttft_p95_ms: e.ttft_p95_ms,
                    worker_pool: e.worker_pool.clone(),
                    node_class: e.node_class.clone(),
                    capabilities: e.capabilities.clone(),
                    queue_depth: e.queue_depth,
                    error_rate: e.error_rate,
                }
            })
            .collect()
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
                WorkerHealth::Healthy
            } else if age_ms <= (2 * ttl_ms) / 3 {
                WorkerHealth::Unhealthy { missed: 1 }
            } else if age_ms <= ttl_ms {
                WorkerHealth::Unhealthy { missed: 2 }
            } else {
                evicted.push(entry.id);
                WorkerHealth::Dead // removed below
            };
        }

        // Second pass: remove dead entries.
        for id in &evicted {
            self.inner.remove(id);
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
    let inflight = e.inflight.load(Ordering::Relaxed);
    WorkerSnapshot {
        id: e.id,
        addr: e.addr.to_string(),
        capabilities: e.capabilities.models.clone(),
        capability_descriptor: e.capabilities.clone(),
        backend: e.backend.as_str().to_string(),
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
    }
}

fn supports_request_kind(entry: &WorkerEntry, request_kind: RequestKind) -> bool {
    match entry.capability_source {
        // Compatibility path: legacy workers historically routed by model-id only.
        CapabilitySource::Legacy => true,
        CapabilitySource::Structured => match request_kind {
            RequestKind::Llm => entry.capabilities.llm,
            RequestKind::Embedding => entry.capabilities.embedding,
        },
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
    fn parse_sglang_backend() {
        assert_eq!(BackendKind::parse("sglang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg_lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg-lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::SgLang.as_str(), "sglang");
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
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.backend, "sglang");
        assert_eq!(snapshot.capabilities, vec!["embed-1".to_string()]);
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
}
