//! Admission queue + concurrency scheduler (PRD M1).
//!
//! # What this does
//!
//! Every inference request must acquire a [`SchedulerPermit`] before calling
//! `generate`.  The permit is RAII — dropping it releases the slot and records
//! the end-to-end latency sample.
//!
//! Two limits are enforced:
//!
//! 1. **Admission queue** (`AXS_SCHED_MAX_QUEUE`, default 64) — maximum requests
//!    allowed to wait for an inflight slot.  If the queue is full the request is
//!    rejected with 503.
//!
//! 2. **Inflight semaphore** (`AXS_SCHED_MAX_INFLIGHT`, default 8) — concurrent
//!    in-flight inference calls.  Excess requests wait in the queue up to
//!    `AXS_SCHED_MAX_WAIT_MS` (default 250 ms).
//!
//! # Latency percentiles
//!
//! [`SchedulerMetrics`] maintains HDR histograms (O(1) record, ~0.1% precision)
//! for two signals:
//!
//! - **queue_wait**: time from request arrival to semaphore acquisition
//!   (slow-path requests only; fast-path requests report 0 µs).
//! - **e2e**: time from request arrival to permit drop (= queue_wait + inference).
//!   Reflects the full latency a caller observes.
//!
//! Access via `metrics.queue_wait_p99_us()` / `metrics.e2e_p99_us()`.
//!
//! # Adaptive concurrency (AIMD)
//!
//! Set `AXS_TARGET_P99_MS` to a non-zero value to enable the adaptive
//! controller.  The effective inflight limit is adjusted every
//! `AXS_ADAPTIVE_PROBE_INTERVAL` completions (default 50):
//!
//! - **p99 > target** → multiplicative decrease (×7/8, floor 2)
//! - **p99 < target × 0.8** → additive increase (+1, ceil = `max_inflight`)
//! - **otherwise** → hold
//!
//! The controller never exceeds `config.max_inflight` and never drops below 2.
//!
//! # Batching hints (`AXS_BATCH_WINDOW_MS`, `AXS_MAX_BATCH_SIZE`)
//!
//! These values are stored in [`SchedulerConfig`] and exposed to callers via
//! [`Scheduler::config`].  They are advisory and have no effect on the current
//! llama.cpp path.
//!
//! # Overload policy (`AXS_OVERLOAD_POLICY`)
//!
//! - `queue` *(default)* — allow queueing up to `max_queue`; reject with 503 when the queue is full.
//! - `reject` — alias for `queue`; reject with 503 when the queue is full.
//! - `shed_oldest` — when the queue is full, drop the oldest waiter and admit the new request.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;
use ax_serving_engine::ThermalMonitor;
use dashmap::DashMap;
use hdrhistogram::Histogram;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::warn;

const HISTOGRAM_SHARDS: usize = 8;

// ── Config ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverloadPolicy {
    /// Queue requests until `max_queue`; return 503 when queue overflows.
    Queue,
    /// Alias for `Queue` — same behavior, kept for backward-compat.
    Reject,
    /// When queue is full, drop the oldest waiter and admit the incoming request.
    ShedOldest,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_queue: usize,
    pub max_inflight: usize,
    pub max_wait_ms: u64,
    pub overload_policy: OverloadPolicy,
    /// Maximum number of requests to group into one continuous batch.
    /// `AXS_MAX_BATCH_SIZE` (default 8). Currently advisory.
    pub max_batch_size: usize,
    /// How long to wait for a batch to fill before dispatching it (ms).
    /// `AXS_BATCH_WINDOW_MS` (default 5). Currently advisory.
    pub batch_window_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue: 128,
            max_inflight: 16,
            max_wait_ms: 120_000,
            overload_policy: OverloadPolicy::Queue,
            max_batch_size: 8,
            batch_window_ms: 5,
        }
    }
}

impl SchedulerConfig {
    pub fn from_serve_config(
        max_inflight: usize,
        max_queue: usize,
        max_wait_ms: u64,
        overload_policy: &str,
        max_batch_size: usize,
        batch_window_ms: u64,
    ) -> Self {
        Self {
            max_inflight: max_inflight.max(1),
            max_queue,
            max_wait_ms,
            overload_policy: match overload_policy.to_lowercase().as_str() {
                "shed_oldest" | "shed-oldest" => OverloadPolicy::ShedOldest,
                "reject" => OverloadPolicy::Reject,
                _ => OverloadPolicy::Queue,
            },
            max_batch_size: max_batch_size.max(1),
            batch_window_ms,
        }
    }
}

// ── Metrics ────────────────────────────────────────────────────────────────────

pub struct SchedulerMetrics {
    /// Requests currently waiting for a slot.
    pub queue_depth: AtomicI64,
    /// Requests currently executing in the backend.
    pub inflight_count: AtomicI64,
    /// Total requests received (lifetime).
    pub total_requests: AtomicU64,
    /// Total requests rejected (lifetime).
    pub rejected_requests: AtomicU64,
    /// Requests that actually entered the slow-path wait queue (lifetime).
    pub queued_requests: AtomicU64,
    /// Cumulative queue wait in microseconds (slow-path only).
    pub queue_wait_us_total: AtomicU64,
    /// Cache followers currently waiting pre-permit (WS3/WS5).
    pub cache_follower_waiting: AtomicI64,
    /// Prompt tokens currently in the prefill phase (WS2 split scheduler).
    pub prefill_tokens_active: AtomicI64,
    /// Sequences currently in the decode phase (WS2 split scheduler).
    pub decode_sequences_active: AtomicI64,

    // Sharded HDR histograms reduce writer contention on the hot path while
    // keeping percentile reads cheap enough for metrics scrapes.
    queue_wait_histogram: HistogramShards,
    e2e_histogram: HistogramShards,
    /// Time-to-first-token histogram (streaming requests only).
    ttft_histogram: HistogramShards,
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self {
            queue_depth: AtomicI64::new(0),
            inflight_count: AtomicI64::new(0),
            total_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            queued_requests: AtomicU64::new(0),
            queue_wait_us_total: AtomicU64::new(0),
            cache_follower_waiting: AtomicI64::new(0),
            prefill_tokens_active: AtomicI64::new(0),
            decode_sequences_active: AtomicI64::new(0),
            queue_wait_histogram: HistogramShards::new(),
            e2e_histogram: HistogramShards::new(),
            ttft_histogram: HistogramShards::new(),
        }
    }
}

impl SchedulerMetrics {
    pub fn avg_queue_wait_us(&self) -> u64 {
        let queued = self.queued_requests.load(Ordering::Relaxed);
        let wait = self.queue_wait_us_total.load(Ordering::Relaxed);
        if queued == 0 { 0 } else { wait / queued }
    }

    // ── Queue wait percentiles (slow-path requests only) ───────────────────

    // Percentile reads use try_lock() for the same reason the recording path
    // does: blocking here during a Prometheus scrape would hold the mutex
    // across 6 sequential reads, causing concurrent request completions to
    // drop their samples (try_lock() → silent miss). A stale 0 on a scrape
    // is acceptable; the next scrape will observe the current value.

    pub fn queue_wait_p50_us(&self) -> u64 {
        self.queue_wait_histogram.p50_us()
    }

    pub fn queue_wait_p95_us(&self) -> u64 {
        self.queue_wait_histogram.p95_us()
    }

    pub fn queue_wait_p99_us(&self) -> u64 {
        self.queue_wait_histogram.p99_us()
    }

    // ── End-to-end percentiles (all admitted requests) ─────────────────────

    pub fn e2e_p50_us(&self) -> u64 {
        self.e2e_histogram.p50_us()
    }

    pub fn e2e_p95_us(&self) -> u64 {
        self.e2e_histogram.p95_us()
    }

    pub fn e2e_p99_us(&self) -> u64 {
        self.e2e_histogram.p99_us()
    }

    // ── TTFT percentiles (streaming requests only) ─────────────────────────

    pub fn ttft_p50_us(&self) -> u64 {
        self.ttft_histogram.p50_us()
    }

    pub fn ttft_p95_us(&self) -> u64 {
        self.ttft_histogram.p95_us()
    }

    pub fn ttft_p99_us(&self) -> u64 {
        self.ttft_histogram.p99_us()
    }

    // ── Internal recording helpers ─────────────────────────────────────────

    fn record_queue_wait(&self, us: u64) {
        self.queue_wait_histogram.record_us(us);
    }

    fn record_e2e(&self, us: u64) {
        self.e2e_histogram.record_us(us);
    }

    pub fn record_ttft(&self, us: u64) {
        self.ttft_histogram.record_us(us);
    }
}

struct HistogramShards {
    next_shard: AtomicUsize,
    shards: [std::sync::Mutex<LatencyWindow>; HISTOGRAM_SHARDS],
}

impl HistogramShards {
    fn new() -> Self {
        Self {
            next_shard: AtomicUsize::new(0),
            shards: std::array::from_fn(|_| std::sync::Mutex::new(LatencyWindow::new())),
        }
    }

    fn record_us(&self, us: u64) {
        let idx = self.next_shard.fetch_add(1, Ordering::Relaxed) % HISTOGRAM_SHARDS;
        if let Ok(mut shard) = self.shards[idx].lock() {
            shard.record_us(us);
        }
    }

    fn snapshot(&self) -> Option<LatencyWindow> {
        let mut merged = LatencyWindow::new();
        let mut saw_data = false;
        for shard in &self.shards {
            if let Ok(guard) = shard.try_lock()
                && !guard.hist.is_empty()
            {
                let _ = merged.hist.add(&guard.hist);
                saw_data = true;
            }
        }
        if saw_data { Some(merged) } else { None }
    }

    fn p50_us(&self) -> u64 {
        self.snapshot()
            .map(|h| h.p50().as_micros() as u64)
            .unwrap_or(0)
    }

    fn p95_us(&self) -> u64 {
        self.snapshot()
            .map(|h| h.p95().as_micros() as u64)
            .unwrap_or(0)
    }

    fn p99_us(&self) -> u64 {
        self.snapshot()
            .map(|h| h.p99().as_micros() as u64)
            .unwrap_or(0)
    }
}

// ── Adaptive concurrency controller (AIMD) ─────────────────────────────────────

/// AIMD adaptive concurrency controller.
///
/// Adjusts the effective inflight limit up or down based on observed p99
/// end-to-end latency relative to a configured target.  All atomic ops use
/// relaxed ordering — occasional stale reads are harmless.
pub struct AdaptiveController {
    effective_limit: AtomicUsize,
    /// Target p99 latency in microseconds.
    target_p99_us: u64,
    /// Re-evaluate every N completions.
    probe_interval: u64,
    request_counter: AtomicU64,
    max_limit: usize,
}

impl AdaptiveController {
    fn new(initial_limit: usize, target_p99_ms: u64, probe_interval: u64) -> Self {
        Self {
            effective_limit: AtomicUsize::new(initial_limit),
            target_p99_us: target_p99_ms.saturating_mul(1_000),
            probe_interval: probe_interval.max(1),
            request_counter: AtomicU64::new(0),
            max_limit: initial_limit,
        }
    }

    pub fn effective_limit(&self) -> usize {
        self.effective_limit.load(Ordering::Relaxed)
    }

    pub fn target_p99_ms(&self) -> u64 {
        self.target_p99_us / 1_000
    }

    fn maybe_adapt(&self, metrics: &SchedulerMetrics) {
        // fetch_add returns the OLD value; add 1 to get the new count so the
        // probe fires at completions probe_interval, 2*probe_interval, …
        // (not at completion #1 due to 0 being a multiple of everything).
        let n = self.request_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if !n.is_multiple_of(self.probe_interval) {
            return;
        }
        let p99_us = metrics.e2e_p99_us();
        if p99_us == 0 {
            return; // no data yet
        }
        let current = self.effective_limit.load(Ordering::Relaxed);
        if p99_us > self.target_p99_us {
            // Multiplicative decrease: ×7/8, floor 2.
            let new_limit = ((current * 7) / 8).max(2);
            self.effective_limit.store(new_limit, Ordering::Relaxed);
            tracing::debug!(
                p99_us,
                target_us = self.target_p99_us,
                old = current,
                new = new_limit,
                "adaptive: decreased inflight limit (p99 > target)"
            );
        } else if p99_us < (self.target_p99_us * 4 / 5) {
            // Additive increase: +1, ceil = max_limit.
            let new_limit = (current + 1).min(self.max_limit);
            self.effective_limit.store(new_limit, Ordering::Relaxed);
            tracing::debug!(
                p99_us,
                target_us = self.target_p99_us,
                old = current,
                new = new_limit,
                "adaptive: increased inflight limit (p99 well below target)"
            );
        }
    }
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

pub struct Scheduler {
    config: SchedulerConfig,
    semaphore: Arc<Semaphore>,
    pub metrics: Arc<SchedulerMetrics>,
    /// Thermal monitor for dynamic concurrency capping.
    thermal: Arc<ThermalMonitor>,
    /// Optional AIMD adaptive controller (enabled by `AXS_TARGET_P99_MS > 0`).
    adaptive: Option<Arc<AdaptiveController>>,
    /// WS2: split prefill/decode tracking enabled (AXS_SPLIT_SCHEDULER).
    pub split_enabled: bool,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, thermal: Arc<ThermalMonitor>) -> Self {
        let adaptive = std::env::var("AXS_TARGET_P99_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(|target_ms| {
                let probe_interval = std::env::var("AXS_ADAPTIVE_PROBE_INTERVAL")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(50);
                Arc::new(AdaptiveController::new(
                    config.max_inflight,
                    target_ms,
                    probe_interval,
                ))
            });
        if let Some(ref a) = adaptive {
            tracing::info!(
                target_p99_ms = a.target_p99_ms(),
                probe_interval = a.probe_interval,
                "adaptive concurrency controller enabled"
            );
        }
        // ShedOldest requires a VecDeque of oneshot channels (as in GlobalQueue) and is
        // not implemented in the semaphore-based per-worker Scheduler. Warn at startup so
        // the misconfiguration is visible; behavior will be identical to Queue (reject
        // the incoming request when the admission queue is full).
        if config.overload_policy == OverloadPolicy::ShedOldest {
            warn!(
                "AXS_OVERLOAD_POLICY=shed_oldest is not supported in the per-worker scheduler \
                 (only in the orchestrator GlobalQueue); requests will be rejected when the \
                 admission queue is full, not shed. Use the orchestrator-level \
                 AXS_GLOBAL_QUEUE_OVERLOAD_POLICY for shed_oldest behavior."
            );
        }
        let split_enabled = std::env::var("AXS_SPLIT_SCHEDULER")
            .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(false);
        let semaphore = Arc::new(Semaphore::new(config.max_inflight));
        Self {
            config,
            semaphore,
            metrics: Arc::new(SchedulerMetrics::default()),
            thermal,
            adaptive,
            split_enabled,
        }
    }

    pub fn from_serve_config(
        max_inflight: usize,
        max_queue: usize,
        max_wait_ms: u64,
        overload_policy: &str,
        max_batch_size: usize,
        batch_window_ms: u64,
        thermal: Arc<ThermalMonitor>,
    ) -> Self {
        Self::new(
            SchedulerConfig::from_serve_config(
                max_inflight,
                max_queue,
                max_wait_ms,
                overload_policy,
                max_batch_size,
                batch_window_ms,
            ),
            thermal,
        )
    }

    /// Current effective inflight limit (adaptive or static).
    pub fn effective_inflight_limit(&self) -> usize {
        if let Some(ref a) = self.adaptive {
            a.effective_limit().min(self.config.max_inflight)
        } else {
            self.config.max_inflight
        }
    }

    /// Target p99 ms if adaptive is enabled, else None.
    pub fn adaptive_target_p99_ms(&self) -> Option<u64> {
        self.adaptive.as_ref().map(|a| a.target_p99_ms())
    }

    /// Acquire an inflight permit, waiting in the admission queue if necessary.
    ///
    /// Returns `Err` with 503-appropriate message if the queue is full or
    /// the wait times out.
    ///
    /// The effective inflight limit is:
    ///   `min(adaptive_limit, config.max_inflight, thermal.recommended_concurrency())`
    /// Thermal and adaptive caps are soft — requests that exceed them are
    /// queued or rejected per overload policy.
    pub async fn acquire(&self) -> Result<SchedulerPermit> {
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        let arrived_at = Instant::now();

        // Compute effective limit: thermal cap × adaptive cap × static cap.
        let effective_limit = self
            .thermal
            .recommended_concurrency()
            .min(self.effective_inflight_limit());

        // Fast path: thermal/adaptive cap not exceeded and a semaphore slot is free.
        let current_inflight = self.metrics.inflight_count.load(Ordering::Relaxed);
        if current_inflight < effective_limit as i64
            && let Ok(permit) = Arc::clone(&self.semaphore).try_acquire_owned()
        {
            self.metrics.inflight_count.fetch_add(1, Ordering::Relaxed);
            return Ok(SchedulerPermit {
                _permit: permit,
                metrics: Arc::clone(&self.metrics),
                adaptive: self.adaptive.clone(),
                arrived_at,
                queue_wait_us: 0,
                estimated_prompt_tokens: 0,
                ttft_fired: AtomicBool::new(false),
            });
        }

        // Slow path: all slots busy — check queue capacity.
        let queue_len = self.metrics.queue_depth.fetch_add(1, Ordering::SeqCst);
        if queue_len >= self.config.max_queue as i64 {
            self.metrics.queue_depth.fetch_sub(1, Ordering::SeqCst);
            self.metrics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "admission queue full: {} requests waiting (max {}); try again later",
                queue_len,
                self.config.max_queue
            );
        }

        // Count requests that actually enter the wait queue (used as denominator
        // for avg_queue_wait_us — fast-path requests must not pollute this count).
        self.metrics.queued_requests.fetch_add(1, Ordering::Relaxed);

        // Wait for a slot with bounded timeout.
        let result = tokio::time::timeout(
            Duration::from_millis(self.config.max_wait_ms),
            Arc::clone(&self.semaphore).acquire_owned(),
        )
        .await;

        self.metrics.queue_depth.fetch_sub(1, Ordering::SeqCst);

        let permit = match result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => {
                // Semaphore closed (server shutting down). Record actual wait so
                // percentiles reflect the full slow-path experience, not just successes.
                let wait_us = arrived_at.elapsed().as_micros() as u64;
                self.metrics
                    .queue_wait_us_total
                    .fetch_add(wait_us, Ordering::Relaxed);
                self.metrics.record_queue_wait(wait_us);
                self.metrics
                    .rejected_requests
                    .fetch_add(1, Ordering::Relaxed);
                anyhow::bail!("scheduler semaphore closed (server shutting down)");
            }
            Err(_) => {
                // Timeout: request waited the full max_wait_ms. Record that wait.
                let wait_us = arrived_at.elapsed().as_micros() as u64;
                self.metrics
                    .queue_wait_us_total
                    .fetch_add(wait_us, Ordering::Relaxed);
                self.metrics.record_queue_wait(wait_us);
                self.metrics
                    .rejected_requests
                    .fetch_add(1, Ordering::Relaxed);
                anyhow::bail!(
                    "request timed out after {}ms in admission queue",
                    self.config.max_wait_ms
                );
            }
        };

        // Re-check thermal + adaptive soft cap after acquiring the semaphore.
        let effective_limit_now = self
            .thermal
            .recommended_concurrency()
            .min(self.effective_inflight_limit()) as i64;
        if self.metrics.inflight_count.load(Ordering::Relaxed) >= effective_limit_now {
            drop(permit);
            // Record actual wait (request did spend time in the queue before being throttled).
            let wait_us = arrived_at.elapsed().as_micros() as u64;
            self.metrics
                .queue_wait_us_total
                .fetch_add(wait_us, Ordering::Relaxed);
            self.metrics.record_queue_wait(wait_us);
            self.metrics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "request throttled: concurrency cap ({effective_limit_now}) reached; try again later",
            );
        }

        let queue_wait_us = arrived_at.elapsed().as_micros() as u64;
        self.metrics
            .queue_wait_us_total
            .fetch_add(queue_wait_us, Ordering::Relaxed);
        self.metrics.record_queue_wait(queue_wait_us);
        self.metrics.inflight_count.fetch_add(1, Ordering::Relaxed);

        Ok(SchedulerPermit {
            _permit: permit,
            metrics: Arc::clone(&self.metrics),
            adaptive: self.adaptive.clone(),
            arrived_at,
            queue_wait_us,
            estimated_prompt_tokens: 0,
            ttft_fired: AtomicBool::new(false),
        })
    }

    /// Acquire a permit and set the estimated prompt token count for split-scheduler tracking.
    ///
    /// When `AXS_SPLIT_SCHEDULER=true`, this enables prefill/decode phase tracking in metrics:
    /// - `prefill_tokens_active` increments by `tokens` immediately.
    /// - On first token (`record_ttft_now()`), transitions to `decode_sequences_active`.
    /// - On drop, releases whichever phase is still held.
    ///
    /// If `tokens == 0` or split scheduler is disabled, behaves identically to `acquire()`.
    pub async fn acquire_with_tokens(&self, tokens: u64) -> Result<SchedulerPermit> {
        let mut permit = self.acquire().await?;
        if tokens > 0 && self.split_enabled {
            permit.estimated_prompt_tokens = tokens;
            self.metrics
                .prefill_tokens_active
                .fetch_add(tokens as i64, Ordering::Relaxed);
        }
        Ok(permit)
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

// ── SchedulerPermit (RAII) ─────────────────────────────────────────────────────

/// RAII guard: holds one inflight slot.
///
/// Dropping releases the slot, records the end-to-end latency sample, and
/// (if the adaptive controller is enabled) triggers a concurrency probe.
pub struct SchedulerPermit {
    _permit: OwnedSemaphorePermit,
    metrics: Arc<SchedulerMetrics>,
    adaptive: Option<Arc<AdaptiveController>>,
    /// Time the request arrived at the scheduler.
    arrived_at: Instant,
    /// Queue wait in µs (0 for fast-path requests that skipped the queue).
    queue_wait_us: u64,
    /// Estimated prompt token count for split-scheduler tracking (0 = disabled).
    estimated_prompt_tokens: u64,
    /// True after record_ttft_now() fires the prefill→decode transition.
    ttft_fired: AtomicBool,
}

impl SchedulerPermit {
    /// Queue wait in microseconds (0 for fast-path requests).
    pub fn queue_wait_us(&self) -> u64 {
        self.queue_wait_us
    }

    /// Record time-to-first-token using the arrival timestamp stored in this permit.
    ///
    /// Call exactly once, when the first `Token` event is ready to send to the client.
    /// No-op if the histogram lock is contended (stale read on next scrape is acceptable).
    pub fn record_ttft_now(&self) {
        if self.ttft_fired.swap(true, Ordering::Relaxed) {
            return;
        }

        self.metrics
            .record_ttft(self.arrived_at.elapsed().as_micros() as u64);
        // WS2: on first token, transition from prefill to decode phase.
        if self.estimated_prompt_tokens > 0 {
            self.metrics
                .prefill_tokens_active
                .fetch_sub(self.estimated_prompt_tokens as i64, Ordering::Relaxed);
            self.metrics
                .decode_sequences_active
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.metrics.inflight_count.fetch_sub(1, Ordering::Relaxed);
        self.metrics
            .record_e2e(self.arrived_at.elapsed().as_micros() as u64);
        if let Some(ref adaptive) = self.adaptive {
            adaptive.maybe_adapt(&self.metrics);
        }
        // WS2: release whichever phase budget is still held.
        if self.estimated_prompt_tokens > 0 {
            if self.ttft_fired.load(Ordering::Relaxed) {
                // Streaming: was in decode phase at completion.
                self.metrics
                    .decode_sequences_active
                    .fetch_sub(1, Ordering::Relaxed);
            } else {
                // Non-streaming or error: never transitioned, still in prefill.
                self.metrics
                    .prefill_tokens_active
                    .fetch_sub(self.estimated_prompt_tokens as i64, Ordering::Relaxed);
            }
        }
    }
}

// ── PerModelScheduler ─────────────────────────────────────────────────────────

/// Per-model concurrency limiter for multi-model concurrent serving.
///
/// Each distinct model ID gets an independent semaphore capped at
/// `AXS_PER_MODEL_MAX_INFLIGHT` (default 2) concurrent slots. Requests to
/// *different* models proceed in parallel; requests to the *same* model are
/// serialized/queued up to `max_wait_ms`.
///
/// This is applied **in addition to** the global [`Scheduler`], so the total
/// inflight count is bounded by both limits simultaneously.
pub struct PerModelScheduler {
    max_per_model: usize,
    slots: DashMap<String, Arc<Semaphore>>,
}

impl PerModelScheduler {
    pub fn new(max_per_model: usize) -> Self {
        Self {
            max_per_model: max_per_model.max(1),
            slots: DashMap::new(),
        }
    }

    /// Acquire a per-model slot. Returns a permit that releases the slot on drop.
    ///
    /// Waits up to `max_wait_ms` ms. Returns `Err` on timeout or semaphore close.
    pub async fn acquire(&self, model_id: &str, max_wait_ms: u64) -> Result<OwnedSemaphorePermit> {
        let sem = Arc::clone(
            &*self
                .slots
                .entry(model_id.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(self.max_per_model))),
        );

        tokio::time::timeout(Duration::from_millis(max_wait_ms), sem.acquire_owned())
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "per-model slot timeout for '{}' after {}ms",
                    model_id,
                    max_wait_ms
                )
            })?
            .map_err(|_| anyhow::anyhow!("per-model semaphore closed for '{}'", model_id))
    }
}

// ── Rolling latency window ─────────────────────────────────────────────────────

/// HDR histogram-backed latency tracker for request percentiles.
///
/// Replaces the previous `VecDeque` implementation that sorted all samples on
/// every percentile query (O(n log n) + Vec alloc).  `hdrhistogram` records
/// in O(1) and answers percentile queries in O(log n) with ~0.1% precision
/// (3 significant figures).
///
/// Range: 1 µs – 1 hour.  Values outside this range are clamped.
pub struct LatencyWindow {
    hist: Histogram<u64>,
}

impl Default for LatencyWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyWindow {
    pub fn new() -> Self {
        // 1 µs min, 3 600 s max (1 hour), 3 significant figures (~0.1% error).
        let hist =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("hdrhistogram bounds are valid");
        Self { hist }
    }

    pub fn record_us(&mut self, us: u64) {
        // Clamp to [1 µs, 1 hour] to stay within histogram bounds.
        let _ = self.hist.record(us.clamp(1, 3_600_000_000));
    }

    pub fn p50(&self) -> Duration {
        if self.hist.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_micros(self.hist.value_at_quantile(0.50))
    }

    pub fn p95(&self) -> Duration {
        if self.hist.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_micros(self.hist.value_at_quantile(0.95))
    }

    pub fn p99(&self) -> Duration {
        if self.hist.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_micros(self.hist.value_at_quantile(0.99))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn permit_raii_decrements_inflight() {
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 2,
                max_queue: 10,
                max_wait_ms: 100,
                overload_policy: OverloadPolicy::Reject,
                max_batch_size: 8,
                batch_window_ms: 5,
            },
            Arc::new(ThermalMonitor::new()),
        );
        let p1 = s.acquire().await.unwrap();
        let p2 = s.acquire().await.unwrap();
        assert_eq!(s.metrics.inflight_count.load(Ordering::Relaxed), 2);
        drop(p1);
        assert_eq!(s.metrics.inflight_count.load(Ordering::Relaxed), 1);
        drop(p2);
        assert_eq!(s.metrics.inflight_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn rejects_when_queue_full() {
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 0,
                max_wait_ms: 50,
                overload_policy: OverloadPolicy::Reject,
                max_batch_size: 8,
                batch_window_ms: 5,
            },
            Arc::new(ThermalMonitor::new()),
        );
        // Occupy the one slot.
        let _p = Arc::clone(&s.semaphore).try_acquire_owned().unwrap();
        // Next should be rejected immediately (queue capacity = 0).
        let res = s.acquire().await;
        assert!(res.is_err());
        assert_eq!(s.metrics.rejected_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn fast_path_records_zero_queue_wait() {
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 4,
                max_queue: 10,
                max_wait_ms: 100,
                overload_policy: OverloadPolicy::Reject,
                max_batch_size: 8,
                batch_window_ms: 5,
            },
            Arc::new(ThermalMonitor::new()),
        );
        let p = s.acquire().await.unwrap();
        assert_eq!(p.queue_wait_us(), 0);
        drop(p);
        // Fast-path requests increment total_requests but NOT queued_requests.
        assert_eq!(s.metrics.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(s.metrics.queued_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn latency_window_percentiles() {
        let mut w = LatencyWindow::new();
        for ms in 1u64..=100 {
            w.record_us(ms * 1_000);
        }
        // hdrhistogram has ~0.1% precision (3 significant figures), which is
        // well under 1 ms at these scales — as_millis() truncation absorbs it.
        assert_eq!(w.p50().as_millis(), 50);
        assert_eq!(w.p95().as_millis(), 95);
        assert_eq!(w.p99().as_millis(), 99);
    }

    #[test]
    fn adaptive_controller_decreases_on_high_p99() {
        let ctrl = AdaptiveController::new(8, 100 /* ms */, 1 /* probe every request */);
        // Simulate metrics with p99 = 500ms (above 100ms target).
        let metrics = SchedulerMetrics::default();
        // Feed 100 samples of 500ms into e2e histogram to get p99 = 500ms.
        for _ in 0..100 {
            metrics.e2e_histogram.record_us(500_000);
        }
        ctrl.maybe_adapt(&metrics);
        // Effective limit should have decreased from 8.
        assert!(ctrl.effective_limit() < 8);
    }

    #[test]
    fn adaptive_controller_increases_on_low_p99() {
        let ctrl = AdaptiveController::new(8, 100 /* ms */, 1);
        // Start limit at 4 (simulate previous decrease).
        ctrl.effective_limit.store(4, Ordering::Relaxed);
        let metrics = SchedulerMetrics::default();
        // Feed 100 samples of 20ms into e2e histogram (well below 80ms = 0.8×100ms).
        for _ in 0..100 {
            metrics.e2e_histogram.record_us(20_000);
        }
        ctrl.maybe_adapt(&metrics);
        // Effective limit should have increased from 4.
        assert!(ctrl.effective_limit() > 4);
    }

    #[test]
    fn adaptive_controller_holds_steady_in_range() {
        // p99 between target×0.8 and target → no change.
        let ctrl = AdaptiveController::new(8, 100 /* ms */, 1);
        ctrl.effective_limit.store(6, Ordering::Relaxed);
        let metrics = SchedulerMetrics::default();
        // 90ms is above 80ms (hold threshold) and below 100ms (decrease threshold).
        for _ in 0..100 {
            metrics.e2e_histogram.record_us(90_000);
        }
        ctrl.maybe_adapt(&metrics);
        assert_eq!(
            ctrl.effective_limit(),
            6,
            "limit must not change in hold-steady band"
        );
    }

    #[test]
    fn adaptive_controller_probe_interval_skips() {
        // With probe_interval=5, the controller should only act every 5th completion.
        let ctrl = AdaptiveController::new(8, 100 /* ms */, 5);
        ctrl.effective_limit.store(8, Ordering::Relaxed);
        let metrics = SchedulerMetrics::default();
        for _ in 0..100 {
            metrics.e2e_histogram.record_us(500_000); // high p99 → would decrease
        }
        // Completions 1-4 should be no-ops.
        for _ in 0..4 {
            ctrl.maybe_adapt(&metrics);
        }
        assert_eq!(
            ctrl.effective_limit(),
            8,
            "should not change before probe interval"
        );
        // Completion 5 should trigger a decrease.
        ctrl.maybe_adapt(&metrics);
        assert!(
            ctrl.effective_limit() < 8,
            "should decrease at probe interval"
        );
    }

    #[test]
    fn scheduler_config_from_serve_config_policies() {
        let shed = SchedulerConfig::from_serve_config(4, 32, 200, "shed_oldest", 8, 5);
        assert_eq!(shed.overload_policy, OverloadPolicy::ShedOldest);
        let shed2 = SchedulerConfig::from_serve_config(4, 32, 200, "shed-oldest", 8, 5);
        assert_eq!(shed2.overload_policy, OverloadPolicy::ShedOldest);
        let rej = SchedulerConfig::from_serve_config(4, 32, 200, "reject", 8, 5);
        assert_eq!(rej.overload_policy, OverloadPolicy::Reject);
        let queue = SchedulerConfig::from_serve_config(4, 32, 200, "queue", 8, 5);
        assert_eq!(queue.overload_policy, OverloadPolicy::Queue);
        // Unknown falls back to Queue.
        let unk = SchedulerConfig::from_serve_config(4, 32, 200, "DROP_ALL", 8, 5);
        assert_eq!(unk.overload_policy, OverloadPolicy::Queue);
    }

    #[test]
    fn scheduler_config_enforces_min_inflight() {
        let cfg = SchedulerConfig::from_serve_config(0, 32, 200, "queue", 8, 5);
        assert_eq!(cfg.max_inflight, 1, "min inflight should be 1");
    }

    #[test]
    fn scheduler_metrics_avg_queue_wait() {
        let m = SchedulerMetrics::default();
        // No queued requests → avg = 0.
        assert_eq!(m.avg_queue_wait_us(), 0);
        m.queued_requests.store(4, Ordering::Relaxed);
        m.queue_wait_us_total.store(2_000_000, Ordering::Relaxed);
        assert_eq!(m.avg_queue_wait_us(), 500_000);
    }

    #[tokio::test]
    async fn per_model_scheduler_acquires_up_to_limit() {
        let sched = PerModelScheduler::new(2);
        let p1 = sched.acquire("model-a", 1_000).await;
        assert!(p1.is_ok());
        let p2 = sched.acquire("model-a", 1_000).await;
        assert!(p2.is_ok());
        // Third acquire should time out immediately (max_per_model=2, both slots held).
        let p3 = sched.acquire("model-a", 1).await;
        assert!(p3.is_err(), "expected timeout with 2 slots occupied");
    }

    #[tokio::test]
    async fn per_model_scheduler_different_models_are_independent() {
        let sched = PerModelScheduler::new(1);
        let _p_a = sched.acquire("model-a", 1_000).await.unwrap();
        // model-b has its own semaphore and should not be blocked.
        let p_b = sched.acquire("model-b", 100).await;
        assert!(p_b.is_ok());
    }

    #[tokio::test]
    async fn per_model_scheduler_slot_released_on_drop() {
        let sched = PerModelScheduler::new(1);
        {
            let _p = sched.acquire("model-a", 1_000).await.unwrap();
            // slot held here
        }
        // After drop the slot should be free again.
        let p2 = sched.acquire("model-a", 100).await;
        assert!(p2.is_ok());
    }

    #[tokio::test]
    async fn acquire_with_tokens_tracks_prefill_then_decode() {
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 1,
                max_wait_ms: 100,
                overload_policy: OverloadPolicy::Reject,
                max_batch_size: 8,
                batch_window_ms: 5,
            },
            Arc::new(ThermalMonitor::new()),
        );

        let permit = s.acquire_with_tokens(16).await.unwrap();
        let prefill = s.metrics.prefill_tokens_active.load(Ordering::Relaxed);
        if s.split_enabled {
            assert_eq!(prefill, 16, "split tracking should charge prompt tokens");
            permit.record_ttft_now();
            assert_eq!(s.metrics.prefill_tokens_active.load(Ordering::Relaxed), 0);
            assert_eq!(s.metrics.decode_sequences_active.load(Ordering::Relaxed), 1);
        } else {
            assert_eq!(
                prefill, 0,
                "without split scheduling, token-aware acquire should behave like acquire()"
            );
        }

        drop(permit);
        assert_eq!(s.metrics.decode_sequences_active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn record_ttft_only_counts_once_and_transitions_once() {
        let metrics = Arc::new(SchedulerMetrics::default());
        let permit_raw = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("semaphore permit");
        let permit = SchedulerPermit {
            _permit: permit_raw,
            metrics: Arc::clone(&metrics),
            adaptive: None,
            arrived_at: Instant::now(),
            queue_wait_us: 0,
            estimated_prompt_tokens: 32,
            ttft_fired: AtomicBool::new(false),
        };

        metrics.prefill_tokens_active.store(32, Ordering::Relaxed);
        permit.record_ttft_now();
        permit.record_ttft_now();

        assert_eq!(metrics.prefill_tokens_active.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.decode_sequences_active.load(Ordering::Relaxed), 1);

        let ttft_p50 = metrics.ttft_p50_us();
        assert!(ttft_p50 > 0, "ttft histogram should have one sample");
    }
}
