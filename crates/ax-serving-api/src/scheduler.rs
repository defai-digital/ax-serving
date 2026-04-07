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
//!    rejected with 429 (queue-full) or 503 (timeout / thermal throttle).
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
//! # Overload policy (`AXS_OVERLOAD_POLICY`)
//!
//! - `queue` *(default)* — allow queueing up to `max_queue`; reject with 503 when the queue is full.
//! - `reject` — alias for `queue`; reject with 503 when the queue is full.
//! - `shed_oldest` — when the queue is full, drop the oldest waiter and admit the new request.

use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;
use ax_serving_engine::ThermalMonitor;
use dashmap::DashMap;
use hdrhistogram::Histogram;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

const HISTOGRAM_SHARDS: usize = 8;

// ── Typed scheduler errors ─────────────────────────────────────────────────────

/// Typed errors produced by the admission scheduler.
///
/// Route handlers downcast via `anyhow::Error::downcast_ref::<SchedulerError>()`
/// to map each variant to the correct HTTP status code — the same pattern used
/// by [`crate::registry::RegistryError`].
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// Admission queue is full; the caller should retry with back-off (HTTP 429).
    #[error("admission queue full: {waiting} requests waiting (max {max}); try again later")]
    QueueFull { waiting: i64, max: usize },
    /// Request waited longer than `max_wait_ms` for an inflight slot (HTTP 503).
    #[error("request timed out after {wait_ms}ms in admission queue")]
    Timeout { wait_ms: u64 },
    /// Thermal or adaptive soft cap rejected the request after it acquired the semaphore (HTTP 503).
    #[error("request throttled: concurrency cap ({cap}) reached; try again later")]
    Throttled { cap: i64 },
    /// Request was shed by the `shed_oldest` overload policy (HTTP 503).
    #[error("request shed: a newer request evicted this one from the admission queue")]
    Shed,
    /// Server is shutting down (HTTP 503).
    #[error("scheduler semaphore closed (server shutting down)")]
    ShuttingDown,
}

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
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue: 128,
            max_inflight: 16,
            max_wait_ms: 120_000,
            overload_policy: OverloadPolicy::Queue,
        }
    }
}

impl SchedulerConfig {
    pub fn from_serve_config(
        max_inflight: usize,
        max_queue: usize,
        max_wait_ms: u64,
        overload_policy: &str,
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
    /// Requests shed by the `shed_oldest` overload policy (lifetime).
    pub shed_requests: AtomicU64,
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
            shed_requests: AtomicU64::new(0),
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
    //
    // IMPORTANT: always use queue_wait_percentiles_us() when you need all
    // three values together.  Calling p50/p95/p99 separately takes three
    // independent snapshots from different shard subsets under contention,
    // which can produce p50 > p99 in the output.

    /// Returns (p50_us, p95_us, p99_us) from a single consistent snapshot.
    pub fn queue_wait_percentiles_us(&self) -> (u64, u64, u64) {
        self.queue_wait_histogram.percentiles_us()
    }

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

    /// Returns (p50_us, p95_us, p99_us) from a single consistent snapshot.
    pub fn e2e_percentiles_us(&self) -> (u64, u64, u64) {
        self.e2e_histogram.percentiles_us()
    }

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

    /// Returns (p50_us, p95_us, p99_us) from a single consistent snapshot.
    pub fn ttft_percentiles_us(&self) -> (u64, u64, u64) {
        self.ttft_histogram.percentiles_us()
    }

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
        let merged_hist = merged.hist.as_mut()?;
        let mut saw_data = false;
        for shard in &self.shards {
            if let Ok(guard) = shard.try_lock()
                && let Some(hist) = guard.hist.as_ref()
                && !hist.is_empty()
            {
                let _ = merged_hist.add(hist);
                saw_data = true;
            }
        }
        if saw_data { Some(merged) } else { None }
    }

    /// Returns (p50_us, p95_us, p99_us) from a single consistent snapshot.
    ///
    /// All three percentiles are derived from the same merged histogram so the
    /// invariant p50 ≤ p95 ≤ p99 is always maintained.  Callers that need all
    /// three values MUST use this method rather than calling p50_us/p95_us/p99_us
    /// individually — three separate snapshots can sample different shard subsets
    /// under high write contention (try_lock failing), producing p50 > p99.
    fn percentiles_us(&self) -> (u64, u64, u64) {
        match self.snapshot() {
            Some(h) => (
                h.p50().as_micros() as u64,
                h.p95().as_micros() as u64,
                h.p99().as_micros() as u64,
            ),
            None => (0, 0, 0),
        }
    }

    fn p50_us(&self) -> u64 {
        self.percentiles_us().0
    }

    fn p95_us(&self) -> u64 {
        self.percentiles_us().1
    }

    fn p99_us(&self) -> u64 {
        self.percentiles_us().2
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
    /// Cancellation senders for requests in the slow-path wait queue.
    /// Used by `shed_oldest` policy to evict the oldest waiter when the queue
    /// is full, making room for the incoming request.
    shed_waiters: Mutex<VecDeque<oneshot::Sender<()>>>,
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
        if config.overload_policy == OverloadPolicy::ShedOldest {
            tracing::info!(
                "AXS_OVERLOAD_POLICY=shed_oldest enabled: when the admission queue is full, \
                 the oldest waiting request will be evicted (503) to make room for the new one."
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
            shed_waiters: Mutex::new(VecDeque::new()),
        }
    }

    pub fn from_serve_config(
        max_inflight: usize,
        max_queue: usize,
        max_wait_ms: u64,
        overload_policy: &str,
        thermal: Arc<ThermalMonitor>,
    ) -> Self {
        Self::new(
            SchedulerConfig::from_serve_config(
                max_inflight,
                max_queue,
                max_wait_ms,
                overload_policy,
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
            if self.config.overload_policy == OverloadPolicy::ShedOldest {
                // Evict the oldest waiter to make room for the incoming request.
                let oldest = self.shed_waiters.lock().unwrap().pop_front();
                if let Some(tx) = oldest {
                    // Signal the victim — its select! branch will fire and return Shed.
                    let _ = tx.send(());
                    self.metrics.shed_requests.fetch_add(1, Ordering::Relaxed);
                    // queue_depth stays the same: oldest leaves, new request enters.
                } else {
                    // Queue was empty despite being "full" (max_queue=0) — nothing to shed.
                    self.metrics.queue_depth.fetch_sub(1, Ordering::SeqCst);
                    self.metrics
                        .rejected_requests
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(SchedulerError::QueueFull {
                        waiting: queue_len,
                        max: self.config.max_queue,
                    }
                    .into());
                }
            } else {
                self.metrics.queue_depth.fetch_sub(1, Ordering::SeqCst);
                self.metrics
                    .rejected_requests
                    .fetch_add(1, Ordering::Relaxed);
                return Err(SchedulerError::QueueFull {
                    waiting: queue_len,
                    max: self.config.max_queue,
                }
                .into());
            }
        }

        // Count requests that actually enter the wait queue (used as denominator
        // for avg_queue_wait_us — fast-path requests must not pollute this count).
        self.metrics.queued_requests.fetch_add(1, Ordering::Relaxed);

        // Register a cancellation channel for shed_oldest eviction.
        let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
        self.shed_waiters.lock().unwrap().push_back(cancel_tx);

        // Wait for a slot with bounded timeout, racing against shed cancellation.
        let sem = Arc::clone(&self.semaphore);
        let result = tokio::time::timeout(
            Duration::from_millis(self.config.max_wait_ms),
            async {
                tokio::select! {
                    biased;
                    _ = cancel_rx => Err(SchedulerError::Shed),
                    permit = sem.acquire_owned() => match permit {
                        Ok(p) => Ok(p),
                        Err(_) => Err(SchedulerError::ShuttingDown),
                    },
                }
            },
        )
        .await;

        self.metrics.queue_depth.fetch_sub(1, Ordering::SeqCst);

        let permit = match result {
            Ok(Ok(p)) => {
                // Remove our cancel_tx from the deque (it was not used).
                // It's already been consumed by the select, so the sender was dropped.
                // Clean up any closed senders left behind.
                self.cleanup_shed_waiters();
                p
            }
            Ok(Err(SchedulerError::Shed)) => {
                let wait_us = arrived_at.elapsed().as_micros() as u64;
                self.metrics
                    .queue_wait_us_total
                    .fetch_add(wait_us, Ordering::Relaxed);
                self.metrics.record_queue_wait(wait_us);
                return Err(SchedulerError::Shed.into());
            }
            Ok(Err(e)) => {
                // Semaphore closed (server shutting down). Record actual wait so
                // percentiles reflect the full slow-path experience, not just successes.
                self.cleanup_shed_waiters();
                let wait_us = arrived_at.elapsed().as_micros() as u64;
                self.metrics
                    .queue_wait_us_total
                    .fetch_add(wait_us, Ordering::Relaxed);
                self.metrics.record_queue_wait(wait_us);
                self.metrics
                    .rejected_requests
                    .fetch_add(1, Ordering::Relaxed);
                return Err(e.into());
            }
            Err(_) => {
                // Timeout: request waited the full max_wait_ms. Record that wait.
                self.cleanup_shed_waiters();
                let wait_us = arrived_at.elapsed().as_micros() as u64;
                self.metrics
                    .queue_wait_us_total
                    .fetch_add(wait_us, Ordering::Relaxed);
                self.metrics.record_queue_wait(wait_us);
                self.metrics
                    .rejected_requests
                    .fetch_add(1, Ordering::Relaxed);
                return Err(SchedulerError::Timeout {
                    wait_ms: self.config.max_wait_ms,
                }
                .into());
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
            return Err(SchedulerError::Throttled {
                cap: effective_limit_now,
            }
            .into());
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

    /// Remove closed/consumed senders from the shed waiter deque.
    fn cleanup_shed_waiters(&self) {
        let mut waiters = self.shed_waiters.lock().unwrap();
        waiters.retain(|tx| !tx.is_closed());
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

    /// Remove the semaphore entry for a model that has been unloaded.
    pub fn remove(&self, model_id: &str) {
        self.slots.remove(model_id);
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
    hist: Option<Histogram<u64>>,
}

impl Default for LatencyWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyWindow {
    pub fn new() -> Self {
        // 1 µs min, 3 600 s max (1 hour), 3 significant figures (~0.1% error).
        let hist = match Histogram::new_with_bounds(1, 3_600_000_000, 3) {
            Ok(hist) => Some(hist),
            Err(err) => {
                tracing::warn!(
                    %err,
                    "failed to initialize latency histogram; disabling latency tracking"
                );
                None
            }
        };
        Self { hist }
    }

    pub fn record_us(&mut self, us: u64) {
        // Clamp to [1 µs, 1 hour] to stay within histogram bounds.
        if let Some(hist) = self.hist.as_mut() {
            let _ = hist.record(us.clamp(1, 3_600_000_000));
        }
    }

    pub fn p50(&self) -> Duration {
        let Some(hist) = self.hist.as_ref() else {
            return Duration::ZERO;
        };

        if hist.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_micros(hist.value_at_quantile(0.50))
    }

    pub fn p95(&self) -> Duration {
        let Some(hist) = self.hist.as_ref() else {
            return Duration::ZERO;
        };

        if hist.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_micros(hist.value_at_quantile(0.95))
    }

    pub fn p99(&self) -> Duration {
        let Some(hist) = self.hist.as_ref() else {
            return Duration::ZERO;
        };

        if hist.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_micros(hist.value_at_quantile(0.99))
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
        let shed = SchedulerConfig::from_serve_config(4, 32, 200, "shed_oldest");
        assert_eq!(shed.overload_policy, OverloadPolicy::ShedOldest);
        let shed2 = SchedulerConfig::from_serve_config(4, 32, 200, "shed-oldest");
        assert_eq!(shed2.overload_policy, OverloadPolicy::ShedOldest);
        let rej = SchedulerConfig::from_serve_config(4, 32, 200, "reject");
        assert_eq!(rej.overload_policy, OverloadPolicy::Reject);
        let queue = SchedulerConfig::from_serve_config(4, 32, 200, "queue");
        assert_eq!(queue.overload_policy, OverloadPolicy::Queue);
        // Unknown falls back to Queue.
        let unk = SchedulerConfig::from_serve_config(4, 32, 200, "DROP_ALL");
        assert_eq!(unk.overload_policy, OverloadPolicy::Queue);
    }

    #[test]
    fn scheduler_config_enforces_min_inflight() {
        let cfg = SchedulerConfig::from_serve_config(0, 32, 200, "queue");
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

    // ── slow-path queue wait recording ────────────────────────────────────────

    #[tokio::test]
    async fn slow_path_permit_records_nonzero_queue_wait() {
        let s = Arc::new(Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 4,
                max_wait_ms: 1000,
                overload_policy: OverloadPolicy::Reject,
            },
            Arc::new(ThermalMonitor::new()),
        ));

        // Occupy the only inflight slot.
        let p1 = s.acquire().await.unwrap();

        // Spawn a waiter that will enter the slow-path queue.
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move { s2.acquire().await });

        // Give the waiter time to block on the semaphore.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Release the slot so the waiter can proceed.
        drop(p1);

        let p2 = waiter
            .await
            .expect("waiter task panicked")
            .expect("second acquire failed");
        assert!(
            p2.queue_wait_us() > 0,
            "slow-path permit must record non-zero queue_wait_us"
        );
    }

    // ── timeout in admission queue ─────────────────────────────────────────────

    #[tokio::test]
    async fn acquire_times_out_when_inflight_slot_never_freed() {
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 4,
                max_wait_ms: 5, // extremely short timeout
                overload_policy: OverloadPolicy::Reject,
            },
            Arc::new(ThermalMonitor::new()),
        );

        // Hold the only slot for the duration of the test.
        let _p1 = s.acquire().await.unwrap();

        let result = s.acquire().await;
        assert!(result.is_err(), "should return Err on queue timeout");
        let msg = match result {
            Ok(_) => panic!("should return Err on queue timeout"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("timed out"),
            "error message should mention timeout: {msg}"
        );
        assert_eq!(s.metrics.rejected_requests.load(Ordering::Relaxed), 1);
    }

    // ── split-scheduler drop-without-ttft clears prefill ──────────────────────

    #[test]
    fn split_scheduler_drop_without_ttft_clears_prefill_budget() {
        // Verify that dropping a permit that was charged prefill tokens but never
        // called record_ttft_now() correctly releases the prefill budget rather
        // than leaking it.
        let metrics = Arc::new(SchedulerMetrics::default());
        let sem = Arc::new(Semaphore::new(1));
        let raw = sem.try_acquire_owned().expect("semaphore");

        let permit = SchedulerPermit {
            _permit: raw,
            metrics: Arc::clone(&metrics),
            adaptive: None,
            arrived_at: Instant::now(),
            queue_wait_us: 0,
            estimated_prompt_tokens: 48,
            ttft_fired: AtomicBool::new(false),
        };

        // Simulate what acquire_with_tokens does: charge prefill at acquisition.
        metrics.prefill_tokens_active.store(48, Ordering::Relaxed);

        // Drop without calling record_ttft_now() (error / non-streaming path).
        drop(permit);

        assert_eq!(
            metrics.prefill_tokens_active.load(Ordering::Relaxed),
            0,
            "prefill budget must be released on drop even if ttft never fired"
        );
        assert_eq!(
            metrics.decode_sequences_active.load(Ordering::Relaxed),
            0,
            "decode_sequences must not be incremented when ttft never fired"
        );
    }

    // ── shed_oldest overload policy ──────────────────────────────────────────

    #[tokio::test]
    async fn shed_oldest_evicts_oldest_waiter_when_queue_full() {
        let s = Arc::new(Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 1,
                max_wait_ms: 5_000,
                overload_policy: OverloadPolicy::ShedOldest,
            },
            Arc::new(ThermalMonitor::new()),
        ));

        // Occupy the only inflight slot.
        let p1 = s.acquire().await.unwrap();

        // Spawn waiter A — will enter the queue (queue_len = 0, capacity = 1).
        let s2 = Arc::clone(&s);
        let waiter_a = tokio::spawn(async move { s2.acquire().await });

        // Give waiter A time to block.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Spawn waiter B — queue is now full (len=1, cap=1), should shed waiter A.
        let s3 = Arc::clone(&s);
        let waiter_b = tokio::spawn(async move { s3.acquire().await });

        // Give waiter B time to trigger the shed.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Waiter A should have been shed.
        let a_result = waiter_a.await.expect("waiter_a task panicked");
        assert!(a_result.is_err(), "oldest waiter should be shed");
        let err_msg = match a_result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected shed error"),
        };
        assert!(
            err_msg.contains("shed"),
            "error should mention shed: {err_msg}"
        );
        assert_eq!(s.metrics.shed_requests.load(Ordering::Relaxed), 1);

        // Release inflight slot so waiter B can proceed.
        drop(p1);
        let b_result = waiter_b.await.expect("waiter_b task panicked");
        assert!(b_result.is_ok(), "newer waiter should succeed");
    }

    #[tokio::test]
    async fn shed_oldest_with_zero_queue_rejects() {
        // max_queue=0: nothing to shed, should reject.
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 0,
                max_wait_ms: 50,
                overload_policy: OverloadPolicy::ShedOldest,
            },
            Arc::new(ThermalMonitor::new()),
        );
        let _p = Arc::clone(&s.semaphore).try_acquire_owned().unwrap();
        let res = s.acquire().await;
        assert!(res.is_err());
        assert_eq!(s.metrics.rejected_requests.load(Ordering::Relaxed), 1);
        assert_eq!(s.metrics.shed_requests.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn queue_policy_still_rejects_when_full() {
        // Verify Queue policy is unchanged — always rejects, never sheds.
        let s = Scheduler::new(
            SchedulerConfig {
                max_inflight: 1,
                max_queue: 0,
                max_wait_ms: 50,
                overload_policy: OverloadPolicy::Queue,
            },
            Arc::new(ThermalMonitor::new()),
        );
        let _p = Arc::clone(&s.semaphore).try_acquire_owned().unwrap();
        let res = s.acquire().await;
        assert!(res.is_err());
        assert_eq!(s.metrics.rejected_requests.load(Ordering::Relaxed), 1);
        assert_eq!(s.metrics.shed_requests.load(Ordering::Relaxed), 0);
    }
}
