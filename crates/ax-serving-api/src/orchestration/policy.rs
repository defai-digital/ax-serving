//! Dispatch policy trait and built-in implementations.
//!
//! # Policies
//!
//! | Name | Description |
//! |---|---|
//! | `least_inflight` | Lowest `inflight / max_inflight` ratio; tie-break by WorkerId |
//! | `weighted_round_robin` | Round-robin weighted by available capacity |
//! | `model_affinity` | Prefer workers that have previously served the model; fall back to least-inflight |
//!
//! # Adding a new policy
//!
//! 1. Implement [`DispatchPolicy`] for your type.
//! 2. Return it from [`policy_from_str`] under a new name.
//! 3. Add to the `AXS_DISPATCH_POLICY` documentation comment below.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::registry::{WorkerId, WorkerStatus};

// ── Types ─────────────────────────────────────────────────────────────────────

/// Request context passed to the dispatch policy alongside the worker slice.
pub struct DispatchContext<'a> {
    pub model_id: &'a str,
    pub stream: bool,
    pub preferred_pool: Option<&'a str>,
}

/// Pluggable worker selection algorithm.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// async tasks behind an `Arc`.
///
/// # Extension hooks
///
/// `record_dispatch` and `worker_evicted` are no-op defaults.  Override them
/// in policies that maintain internal state (e.g. `ModelAffinityPolicy`).
pub trait DispatchPolicy: Send + Sync {
    /// Select one worker from the eligible slice.
    ///
    /// Returns `None` if no suitable worker is available (e.g. all are at
    /// `max_inflight` capacity).  The caller is responsible for returning
    /// 503 to the client in that case.
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus>;

    /// Called after a request is successfully dispatched to `worker_id`.
    ///
    /// Policies that track per-worker statistics (e.g. `ModelAffinityPolicy`)
    /// override this.  Default implementation is a no-op.
    fn record_dispatch(&self, _worker_id: WorkerId, _model_id: &str) {}
}

// ── Shared helper ─────────────────────────────────────────────────────────────

/// Apply least-inflight selection over an iterator of workers.
///
/// Returns the worker with the lowest `inflight / max_inflight` load ratio,
/// excluding workers at full capacity.  Ties are broken by `WorkerId` bytes.
///
/// Accepts any iterator of `&'a WorkerStatus` so callers can avoid intermediate
/// `Vec` allocations — pass `slice.iter()` or a chained filter directly.
fn least_inflight_from<'a>(
    candidates: impl Iterator<Item = &'a WorkerStatus>,
) -> Option<&'a WorkerStatus> {
    candidates
        .filter(|w| w.inflight < w.max_inflight)
        .min_by(|a, b| {
            let la = a.inflight * b.max_inflight;
            let lb = b.inflight * a.max_inflight;
            la.cmp(&lb)
                .then_with(|| a.id.0.as_bytes().cmp(b.id.0.as_bytes()))
        })
}

fn effective_active_sequences(worker: &WorkerStatus) -> usize {
    if worker.active_sequences > 0 || worker.inflight == 0 {
        worker.active_sequences
    } else {
        worker.inflight
    }
}

// ── LeastInflightPolicy ───────────────────────────────────────────────────────

/// Select the worker with the lowest `inflight / max_inflight` load ratio.
///
/// Workers at full capacity (`inflight >= max_inflight`) are excluded.
/// Ties are broken by `WorkerId` bytes (deterministic, avoids hot-spots).
pub struct LeastInflightPolicy;

impl DispatchPolicy for LeastInflightPolicy {
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        _ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus> {
        // Pass slice iterator directly — no intermediate Vec allocation.
        least_inflight_from(workers.iter())
    }
}

// ── WeightedRoundRobinPolicy ──────────────────────────────────────────────────

/// Round-robin weighted by available capacity (`max_inflight - inflight`).
///
/// Workers with zero available capacity are skipped.  The distribution is
/// proportional to each worker's remaining capacity: a worker with capacity 8
/// receives twice as many requests as one with capacity 4.
///
/// An `AtomicU64` position counter advances on every `select` call, making
/// the policy thread-safe without a mutex.
pub struct WeightedRoundRobinPolicy {
    position: AtomicU64,
}

impl WeightedRoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            position: AtomicU64::new(0),
        }
    }
}

impl Default for WeightedRoundRobinPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchPolicy for WeightedRoundRobinPolicy {
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        _ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus> {
        // First pass: sum available capacity across all non-full workers.
        // No Vec allocation — two linear scans over the (typically small) slice.
        // Use saturating addition to guard against overflow in debug builds where
        // .sum() (checked add) would panic if the total ever exceeded usize::MAX.
        let total_weight: usize = workers
            .iter()
            .map(|w| w.max_inflight.saturating_sub(w.inflight))
            .fold(0usize, |acc, cap| acc.saturating_add(cap));

        if total_weight == 0 {
            return None;
        }

        // Map the advancing position counter to a weighted slot.
        let pos = self.position.fetch_add(1, Ordering::Relaxed) as usize;
        let slot = pos % total_weight;

        // Second pass: find the worker that owns this slot.
        let mut cumulative = 0usize;
        for w in workers {
            let cap = w.max_inflight.saturating_sub(w.inflight);
            if cap > 0 {
                cumulative += cap;
                if slot < cumulative {
                    return Some(w);
                }
            }
        }

        // Unreachable in practice (slot < total_weight) but safe fallback.
        workers.iter().find(|w| w.max_inflight > w.inflight)
    }
}

// ── ModelAffinityPolicy ───────────────────────────────────────────────────────

/// Prefer workers that have previously served a given `model_id`.
///
/// # Selection logic
///
/// 1. Filter to eligible workers (not at capacity).
/// 2. Among eligible workers, partition into those with positive affinity
///    for `model_id` (have served it before) and those without.
/// 3. If any affinity workers exist, select the least-loaded among them.
/// 4. Otherwise fall back to least-inflight across all eligible workers.
///
/// # Affinity tracking
///
/// The dispatcher calls `record_dispatch(worker_id, model_id)` after each
/// successful dispatch. Each worker keeps a bounded LRU of recently-served
/// model IDs, and entries for workers no longer in the eligible set are pruned
/// lazily on each `select` call.
const MODEL_AFFINITY_LRU_CAPACITY: usize = 64;

pub struct ModelAffinityPolicy {
    /// `worker_id -> recently served model IDs`, newest at the back.
    affinity: Mutex<HashMap<WorkerId, VecDeque<String>>>,
}

impl ModelAffinityPolicy {
    pub fn new() -> Self {
        Self {
            affinity: Mutex::new(HashMap::new()),
        }
    }

    fn affinity_lock(&self) -> std::sync::MutexGuard<'_, HashMap<WorkerId, VecDeque<String>>> {
        match self.affinity.lock() {
            Ok(guard) => guard,
            Err(err) => {
                tracing::warn!(
                    "model affinity mutex poisoned; continuing with inner value after recovery"
                );
                err.into_inner()
            }
        }
    }
}

impl Default for ModelAffinityPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchPolicy for ModelAffinityPolicy {
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus> {
        // Fast short-circuit: if every worker is at full capacity, bail without
        // touching the affinity mutex.
        if !workers.iter().any(|w| w.inflight < w.max_inflight) {
            return None;
        }

        let mut map = self.affinity_lock();

        // Prune affinity entries for workers that have left the pool entirely
        // (evicted, drained, or lost the model capability).  Do NOT prune
        // workers that are merely at full capacity — they may gain a free slot
        // on the next request and should retain their affinity score.
        //
        // Use a direct O(n·m) membership check to avoid allocating a HashSet.
        // In practice n (workers) and m (affinity entries) are both small.
        map.retain(|wid, _| workers.iter().any(|w| w.id == *wid));

        let has_affinity_match = workers.iter().any(|w| {
            w.inflight < w.max_inflight
                && map
                    .get(&w.id)
                    .is_some_and(|models| models.iter().any(|model| model == ctx.model_id))
        });

        if !has_affinity_match {
            // No affinity data — fall back to least-inflight over all eligible
            // workers without allocating an `eligible` Vec.
            least_inflight_from(workers.iter().filter(|w| w.inflight < w.max_inflight))
        } else {
            least_inflight_from(workers.iter().filter(|w| {
                w.inflight < w.max_inflight
                    && map
                        .get(&w.id)
                        .is_some_and(|models| models.iter().any(|model| model == ctx.model_id))
            }))
        }
    }

    fn record_dispatch(&self, worker_id: WorkerId, model_id: &str) {
        let mut map = self.affinity_lock();
        let models = map.entry(worker_id).or_default();
        if let Some(pos) = models.iter().position(|existing| existing == model_id) {
            models.remove(pos);
        }
        models.push_back(model_id.to_string());
        if models.len() > MODEL_AFFINITY_LRU_CAPACITY {
            models.pop_front();
        }
    }
}

// ── TokenCostPolicy ───────────────────────────────────────────────────────────

/// Select the worker with the lowest predicted latency cost, combining TTFT
/// and active sequence load into a composite score.
///
/// # Scoring formula
///
/// ```text
/// score(w) = ttft_weight  * (w.ttft_p95_ms  / ref_ttft_ms)
///          + seq_weight   * (w.active_sequences / w.max_inflight)
/// ```
///
/// where `ref_ttft_ms = max(1, max(ttft_p95_ms across all candidates))` so
/// scores are always in `[0, 2]` for a two-worker pool.
///
/// # Graceful degradation
///
/// If **all** workers report `ttft_p95_ms == 0` (legacy workers or no streaming
/// requests yet), the TTFT term vanishes and the score reduces to the pure
/// sequence load ratio — identical behaviour to `LeastInflightPolicy`.
///
/// Workers at full capacity (`active_sequences >= max_inflight`) are excluded
/// before scoring regardless of score value.  Tie-breaking is by `WorkerId`
/// bytes for determinism.
/// TTFT is weighted more heavily than sequence load because tail latency is
/// the dominant user-visible signal (60/40 split from empirical tuning).
const TOKEN_COST_TTFT_WEIGHT: f64 = 0.6;
const TOKEN_COST_SEQ_WEIGHT: f64 = 0.4;

pub struct TokenCostPolicy {
    /// Weight for the TTFT component.
    ttft_weight: f64,
    /// Weight for the active-sequence component.
    seq_weight: f64,
}

impl TokenCostPolicy {
    pub fn new() -> Self {
        Self {
            ttft_weight: TOKEN_COST_TTFT_WEIGHT,
            seq_weight: TOKEN_COST_SEQ_WEIGHT,
        }
    }
}

impl Default for TokenCostPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchPolicy for TokenCostPolicy {
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        _ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus> {
        // Eligible = workers that still have capacity.
        // Use `active_sequences` as the primary capacity metric; fall back to
        // `inflight` if the worker hasn't sent the extended telemetry yet
        // (active_sequences == 0 && inflight != 0 means legacy worker).
        //
        // Keep this allocation-free because it runs on every dispatch.
        let mut ref_ttft_ms = 0u64;
        let mut saw_eligible = false;
        for worker in workers {
            if effective_active_sequences(worker) < worker.max_inflight {
                saw_eligible = true;
                ref_ttft_ms = ref_ttft_ms.max(worker.ttft_p95_ms);
            }
        }

        if !saw_eligible {
            return None;
        }

        // Compute the reference TTFT across all eligible workers so scores are
        // normalised relative to the busiest worker in the pool.
        let ref_ttft_ms = ref_ttft_ms.max(1) as f64;

        // Select the worker with the minimum composite score.  Ties broken by
        // WorkerId bytes so dispatch is deterministic under equal load.
        workers
            .iter()
            .filter(|w| effective_active_sequences(w) < w.max_inflight)
            .min_by(|a, b| {
                let score = |w: &&WorkerStatus| {
                    let seqs = effective_active_sequences(w) as f64;
                    let ttft_norm = w.ttft_p95_ms as f64 / ref_ttft_ms;
                    let seq_norm = seqs / w.max_inflight.max(1) as f64;
                    self.ttft_weight * ttft_norm + self.seq_weight * seq_norm
                };
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.0.as_bytes().cmp(b.id.0.as_bytes()))
            })
    }
}

// ── CacheAffinityPolicy ──────────────────────────────────────────────────────

/// Select the worker with the best composite score combining cache locality,
/// KV headroom, batch headroom, and load ratio.
///
/// # Scoring formula
///
/// ```text
/// score(w) = prefix_weight  * prefix_hit(w)
///          + kv_weight      * (1.0 - w.kv_utilization)
///          + batch_weight   * w.batch_headroom
///          + load_weight    * (1.0 - active_sequences / max_inflight)
/// ```
///
/// Higher score is better (more headroom, warmer cache).
///
/// # Graceful degradation
///
/// When workers report no cache telemetry (`kv_utilization == 0.0`,
/// `batch_headroom == 0.0`) and no prefix hits are recorded, the score
/// reduces to the load ratio term — equivalent to `LeastInflightPolicy`.
///
/// # Prefix tracking
///
/// On `record_dispatch`, a hash of the model_id is stored in a bounded LRU
/// per worker (max 256 entries). On `select`, workers with a matching prefix
/// hash receive a bonus.
const CACHE_PREFIX_WEIGHT: f64 = 0.3;
const CACHE_KV_WEIGHT: f64 = 0.25;
const CACHE_BATCH_WEIGHT: f64 = 0.15;
const CACHE_LOAD_WEIGHT: f64 = 0.3;
const PREFIX_LRU_CAPACITY: usize = 256;

pub struct CacheAffinityPolicy {
    prefix_cache: Mutex<HashMap<WorkerId, VecDeque<u64>>>,
}

impl CacheAffinityPolicy {
    pub fn new() -> Self {
        Self {
            prefix_cache: Mutex::new(HashMap::new()),
        }
    }

    fn prefix_lock(&self) -> std::sync::MutexGuard<'_, HashMap<WorkerId, VecDeque<u64>>> {
        match self.prefix_cache.lock() {
            Ok(guard) => guard,
            Err(err) => {
                tracing::warn!("cache affinity prefix mutex poisoned; continuing with inner value");
                err.into_inner()
            }
        }
    }

    fn has_prefix(
        cache: &HashMap<WorkerId, VecDeque<u64>>,
        worker_id: &WorkerId,
        model_id: &str,
    ) -> bool {
        let hash = Self::model_prefix_hash(model_id);
        cache
            .get(worker_id)
            .map(|hashes| hashes.contains(&hash))
            .unwrap_or(false)
    }

    fn model_prefix_hash(model_id: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model_id.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for CacheAffinityPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchPolicy for CacheAffinityPolicy {
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus> {
        if !workers.iter().any(|w| w.inflight < w.max_inflight) {
            return None;
        }

        let cache = self.prefix_lock();

        workers
            .iter()
            .filter(|w| w.inflight < w.max_inflight)
            .max_by(|a, b| {
                let score = |w: &&WorkerStatus| {
                    let seqs = effective_active_sequences(w) as f64;
                    let load_headroom = 1.0 - seqs / w.max_inflight.max(1) as f64;
                    let prefix_hit = if Self::has_prefix(&cache, &w.id, ctx.model_id) {
                        1.0
                    } else {
                        0.0
                    };
                    let kv_headroom = w.kv_utilization.map(|u| 1.0 - u).unwrap_or(0.5);
                    let batch = w.batch_headroom.unwrap_or(0.5);

                    CACHE_PREFIX_WEIGHT * prefix_hit
                        + CACHE_KV_WEIGHT * kv_headroom
                        + CACHE_BATCH_WEIGHT * batch
                        + CACHE_LOAD_WEIGHT * load_headroom
                };
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.0.as_bytes().cmp(b.id.0.as_bytes()))
            })
    }

    fn record_dispatch(&self, worker_id: WorkerId, model_id: &str) {
        let hash = Self::model_prefix_hash(model_id);
        let mut cache = self.prefix_lock();
        let hashes = cache.entry(worker_id).or_default();
        if let Some(pos) = hashes.iter().position(|&h| h == hash) {
            hashes.remove(pos);
        }
        hashes.push_back(hash);
        if hashes.len() > PREFIX_LRU_CAPACITY {
            hashes.pop_front();
        }
    }
}

// ── Factory ───────────────────────────────────────────────────────────────────

/// Construct a [`DispatchPolicy`] from the `AXS_DISPATCH_POLICY` value.
///
/// Supported values:
/// - `"least_inflight"` (default) — lowest load ratio
/// - `"weighted_round_robin"` — proportional to available capacity
/// - `"model_affinity"` — prefers cache-warm workers; falls back to least-inflight
/// - `"token_cost"` — composite TTFT + sequence-load score; degrades to least-inflight
///   for legacy workers that don't send extended heartbeat telemetry
///
/// Returns an error for unknown policy names.
pub fn policy_from_str(name: &str) -> anyhow::Result<Box<dyn DispatchPolicy>> {
    match name {
        "least_inflight" => Ok(Box::new(LeastInflightPolicy)),
        "weighted_round_robin" => Ok(Box::new(WeightedRoundRobinPolicy::new())),
        "model_affinity" => Ok(Box::new(ModelAffinityPolicy::new())),
        "token_cost" => Ok(Box::new(TokenCostPolicy::new())),
        "cache_affinity" => Ok(Box::new(CacheAffinityPolicy::new())),
        other => anyhow::bail!(
            "unknown dispatch policy: {other:?} \
             (supported: least_inflight, weighted_round_robin, model_affinity, \
             token_cost, cache_affinity)"
        ),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::orchestration::registry::WorkerId;

    fn make_worker(inflight: usize, max_inflight: usize) -> WorkerStatus {
        WorkerStatus {
            id: WorkerId(Uuid::new_v4()),
            addr: "127.0.0.1:8081".parse().unwrap(),
            inflight,
            max_inflight,
            active_sequences: inflight,
            ttft_p95_ms: 0,
            kv_utilization: None,
            batch_headroom: None,
        }
    }

    fn ctx() -> DispatchContext<'static> {
        DispatchContext {
            model_id: "m1",
            stream: false,
            preferred_pool: None,
        }
    }

    // ── LeastInflightPolicy ───────────────────────────────────────────────────

    #[test]
    fn empty_returns_none() {
        assert!(LeastInflightPolicy.select(&[], &ctx()).is_none());
    }

    #[test]
    fn full_workers_excluded() {
        let workers = vec![make_worker(4, 4), make_worker(4, 4)];
        assert!(LeastInflightPolicy.select(&workers, &ctx()).is_none());
    }

    #[test]
    fn selects_least_loaded() {
        let workers = vec![
            make_worker(3, 4), // 75%
            make_worker(1, 4), // 25%  ← expected
            make_worker(2, 4), // 50%
        ];
        let selected = LeastInflightPolicy.select(&workers, &ctx()).unwrap();
        assert_eq!(selected.inflight, 1);
    }

    #[test]
    fn tie_broken_by_id() {
        // Both at 0 load — tie should always resolve to the same worker
        // (lowest id bytes), not oscillate.
        let w1 = make_worker(0, 4);
        let w2 = make_worker(0, 4);
        let workers = vec![w1.clone(), w2.clone()];

        let result = LeastInflightPolicy.select(&workers, &ctx()).unwrap();
        let expected_id = [w1.id, w2.id]
            .iter()
            .min_by_key(|id| *id.0.as_bytes())
            .copied()
            .unwrap();
        assert_eq!(result.id, expected_id);
    }

    // ── WeightedRoundRobinPolicy ──────────────────────────────────────────────

    #[test]
    fn wrr_empty_returns_none() {
        let policy = WeightedRoundRobinPolicy::new();
        assert!(policy.select(&[], &ctx()).is_none());
    }

    #[test]
    fn wrr_full_workers_skipped() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers = vec![make_worker(4, 4), make_worker(4, 4)];
        assert!(policy.select(&workers, &ctx()).is_none());
    }

    #[test]
    fn wrr_proportional_distribution() {
        // w_high: capacity 4, w_low: capacity 1.
        // Over 5 calls we expect w_high 4 times and w_low 1 time.
        let policy = WeightedRoundRobinPolicy::new();
        let w_high = make_worker(0, 4); // capacity 4
        let w_low = make_worker(0, 1); // capacity 1
        let workers = vec![w_high.clone(), w_low.clone()];

        let mut high_count = 0usize;
        let mut low_count = 0usize;
        // 5 rounds covers one complete cycle of total_weight=5.
        for _ in 0..5 {
            let sel = policy.select(&workers, &ctx()).unwrap();
            if sel.id == w_high.id {
                high_count += 1;
            } else {
                low_count += 1;
            }
        }
        assert_eq!(
            high_count, 4,
            "high-capacity worker should get 4/5 requests"
        );
        assert_eq!(low_count, 1, "low-capacity worker should get 1/5 requests");
    }

    #[test]
    fn wrr_skips_zero_capacity() {
        // One worker at full capacity, one with capacity.
        let policy = WeightedRoundRobinPolicy::new();
        let full = make_worker(4, 4); // no capacity
        let avail = make_worker(2, 4); // has capacity
        let workers = vec![full, avail.clone()];

        let sel = policy.select(&workers, &ctx()).unwrap();
        assert_eq!(sel.id, avail.id);
    }

    // ── ModelAffinityPolicy ───────────────────────────────────────────────────

    #[test]
    fn affinity_no_data_falls_back_to_least_inflight() {
        let policy = ModelAffinityPolicy::new();
        let workers = vec![
            make_worker(3, 4), // 75%
            make_worker(1, 4), // 25% ← expected
        ];
        let sel = policy.select(&workers, &ctx()).unwrap();
        assert_eq!(sel.inflight, 1);
    }

    #[test]
    fn affinity_prefers_warm_worker() {
        let policy = ModelAffinityPolicy::new();
        let warm = make_worker(3, 4); // higher load but warm
        let cold = make_worker(1, 4); // lower load but no history

        // Record two dispatches to `warm` for model "m1".
        policy.record_dispatch(warm.id, "m1");
        policy.record_dispatch(warm.id, "m1");

        let workers = vec![warm.clone(), cold];
        let sel = policy
            .select(
                &workers,
                &DispatchContext {
                    model_id: "m1",
                    stream: false,
                    preferred_pool: None,
                },
            )
            .unwrap();
        // warm has affinity — should win despite higher inflight.
        assert_eq!(sel.id, warm.id);
    }

    #[test]
    fn affinity_cold_worker_wins_among_non_affinity() {
        // Neither worker has affinity → falls back to least-inflight.
        let policy = ModelAffinityPolicy::new();
        let w_busy = make_worker(3, 4);
        let w_idle = make_worker(0, 4);
        let workers = vec![w_busy, w_idle.clone()];
        let sel = policy
            .select(
                &workers,
                &DispatchContext {
                    model_id: "m1",
                    stream: false,
                    preferred_pool: None,
                },
            )
            .unwrap();
        assert_eq!(sel.id, w_idle.id);
    }

    #[test]
    fn affinity_stale_entries_pruned_on_select() {
        // Register affinity for a worker, then remove it from the candidates.
        // Verify select() cleans up the stale entry.
        let policy = ModelAffinityPolicy::new();
        let gone = make_worker(0, 4);
        policy.record_dispatch(gone.id, "m1");

        // Select without the gone worker → stale entry should be pruned.
        let still_here = make_worker(0, 4);
        let workers = vec![still_here.clone()];
        let sel = policy
            .select(
                &workers,
                &DispatchContext {
                    model_id: "m1",
                    stream: false,
                    preferred_pool: None,
                },
            )
            .unwrap();
        assert_eq!(sel.id, still_here.id);

        // Affinity map should no longer contain gone worker.
        let map = policy.affinity_lock();
        assert!(!map.contains_key(&gone.id));
    }

    #[test]
    fn affinity_lru_is_bounded_per_worker() {
        let policy = ModelAffinityPolicy::new();
        let worker = make_worker(0, 4);
        for i in 0..(MODEL_AFFINITY_LRU_CAPACITY + 1) {
            policy.record_dispatch(worker.id, &format!("m{i}"));
        }

        let map = policy.affinity_lock();
        let models = map.get(&worker.id).unwrap();
        assert_eq!(models.len(), MODEL_AFFINITY_LRU_CAPACITY);
        assert!(
            !models.iter().any(|model| model == "m0"),
            "oldest model should be evicted once the LRU reaches capacity"
        );
    }

    // ── policy_from_str ───────────────────────────────────────────────────────

    #[test]
    fn policy_from_str_all_valid() {
        assert!(policy_from_str("least_inflight").is_ok());
        assert!(policy_from_str("weighted_round_robin").is_ok());
        assert!(policy_from_str("model_affinity").is_ok());
        assert!(policy_from_str("token_cost").is_ok());
        assert!(policy_from_str("cache_affinity").is_ok());
    }

    #[test]
    fn policy_from_str_unknown_errors() {
        let result = policy_from_str("bananas");
        assert!(result.is_err(), "unknown policy name must return Err");
        // Use .err().unwrap() — unwrap_err() requires T: Debug which Box<dyn DispatchPolicy> lacks.
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("bananas"),
            "error should mention bad name: {msg}"
        );
        assert!(
            msg.contains("least_inflight") && msg.contains("model_affinity"),
            "error should list supported policies: {msg}"
        );
    }

    // ── TokenCostPolicy ───────────────────────────────────────────────────────

    fn make_worker_with_telemetry(
        inflight: usize,
        max_inflight: usize,
        ttft_p95_ms: u64,
    ) -> WorkerStatus {
        WorkerStatus {
            ttft_p95_ms,
            ..make_worker(inflight, max_inflight)
        }
    }

    /// Like `make_worker` but with `active_sequences: 0` to simulate a legacy
    /// worker that hasn't started reporting the new field yet.
    fn make_legacy_worker(inflight: usize, max_inflight: usize) -> WorkerStatus {
        WorkerStatus {
            active_sequences: 0,
            ..make_worker(inflight, max_inflight)
        }
    }

    #[test]
    fn token_cost_empty_returns_none() {
        assert!(TokenCostPolicy::new().select(&[], &ctx()).is_none());
    }

    #[test]
    fn token_cost_full_workers_excluded() {
        let workers = vec![
            make_worker_with_telemetry(4, 4, 200),
            make_worker_with_telemetry(4, 4, 100),
        ];
        assert!(TokenCostPolicy::new().select(&workers, &ctx()).is_none());
    }

    #[test]
    fn token_cost_selects_lower_ttft_when_equal_load() {
        // Both workers have the same active_sequences (1/4 load), but different TTFT.
        // The lower-TTFT worker should win.
        let fast = make_worker_with_telemetry(1, 4, 100);
        let slow = make_worker_with_telemetry(1, 4, 400);
        let workers = vec![slow.clone(), fast.clone()];
        let sel = TokenCostPolicy::new().select(&workers, &ctx()).unwrap();
        assert_eq!(sel.id, fast.id, "lower TTFT worker should be selected");
    }

    #[test]
    fn token_cost_selects_lower_load_when_equal_ttft() {
        // Both workers have the same TTFT, but different active_sequences.
        let idle = make_worker_with_telemetry(0, 4, 200);
        let busy = make_worker_with_telemetry(3, 4, 200);
        let workers = vec![busy.clone(), idle.clone()];
        let sel = TokenCostPolicy::new().select(&workers, &ctx()).unwrap();
        assert_eq!(sel.id, idle.id, "lower-load worker should be selected");
    }

    #[test]
    fn token_cost_degrades_to_load_ratio_when_ttft_unknown() {
        // All workers have ttft_p95_ms == 0 (legacy / no streaming requests yet).
        // Policy must fall back to sequence load ratio (= least-inflight behaviour).
        let idle = make_worker_with_telemetry(0, 4, 0);
        let busy = make_worker_with_telemetry(3, 4, 0);
        let workers = vec![busy.clone(), idle.clone()];
        let sel = TokenCostPolicy::new().select(&workers, &ctx()).unwrap();
        assert_eq!(
            sel.id, idle.id,
            "should select lower-load worker when TTFT is unknown"
        );
    }

    #[test]
    fn token_cost_legacy_worker_uses_inflight_as_seqs() {
        // Legacy worker: active_sequences == 0 && inflight != 0.
        // The policy must use `inflight` as the sequence count in that case.
        let legacy_busy = make_legacy_worker(3, 4); // active_sequences: 0 — not yet sending the new field
        let modern_idle = make_worker(1, 4);
        let workers = vec![legacy_busy.clone(), modern_idle.clone()];
        let sel = TokenCostPolicy::new().select(&workers, &ctx()).unwrap();
        // modern_idle has lower effective seqs (1) vs legacy (inflight=3) → should win.
        assert_eq!(sel.id, modern_idle.id);
    }

    #[test]
    fn token_cost_legacy_worker_at_capacity_excluded() {
        // Legacy worker where inflight == max_inflight must be treated as full.
        let legacy_full = make_legacy_worker(4, 4);
        assert!(
            TokenCostPolicy::new()
                .select(&[legacy_full], &ctx())
                .is_none(),
            "legacy worker at inflight==max_inflight must be excluded"
        );
    }

    // ── CacheAffinityPolicy ──────────────────────────────────────────────────

    fn make_worker_with_cache(
        inflight: usize,
        max_inflight: usize,
        kv_utilization: f64,
        batch_headroom: f64,
    ) -> WorkerStatus {
        WorkerStatus {
            kv_utilization: Some(kv_utilization),
            batch_headroom: Some(batch_headroom),
            ..make_worker(inflight, max_inflight)
        }
    }

    #[test]
    fn cache_affinity_empty_returns_none() {
        assert!(CacheAffinityPolicy::new().select(&[], &ctx()).is_none());
    }

    #[test]
    fn cache_affinity_full_workers_excluded() {
        let workers = vec![
            make_worker_with_cache(4, 4, 0.5, 0.5),
            make_worker_with_cache(4, 4, 0.1, 0.9),
        ];
        assert!(
            CacheAffinityPolicy::new()
                .select(&workers, &ctx())
                .is_none()
        );
    }

    #[test]
    fn cache_affinity_prefers_warm_worker() {
        let policy = CacheAffinityPolicy::new();
        let warm = make_worker_with_cache(2, 4, 0.5, 0.5);
        let cold = make_worker_with_cache(1, 4, 0.5, 0.5);

        // Record dispatch to warm worker for model "m1".
        policy.record_dispatch(warm.id, "m1");

        let workers = vec![warm.clone(), cold];
        let sel = policy.select(&workers, &ctx()).unwrap();
        assert_eq!(
            sel.id, warm.id,
            "warm worker should be preferred despite higher load"
        );
    }

    #[test]
    fn cache_affinity_degrades_to_load_without_telemetry() {
        let policy = CacheAffinityPolicy::new();
        let idle = make_worker(0, 4);
        let busy = make_worker(3, 4);
        let workers = vec![busy.clone(), idle.clone()];
        let sel = policy.select(&workers, &ctx()).unwrap();
        assert_eq!(
            sel.id, idle.id,
            "without telemetry, should select lower-load worker"
        );
    }

    #[test]
    fn cache_affinity_prefers_lower_kv_utilization() {
        let policy = CacheAffinityPolicy::new();
        let high_kv = make_worker_with_cache(1, 4, 0.9, 0.5);
        let low_kv = make_worker_with_cache(1, 4, 0.2, 0.5);
        let workers = vec![high_kv.clone(), low_kv.clone()];
        let sel = policy.select(&workers, &ctx()).unwrap();
        assert_eq!(
            sel.id, low_kv.id,
            "lower KV utilization should be preferred"
        );
    }

    #[test]
    fn cache_affinity_prefers_more_batch_headroom() {
        let policy = CacheAffinityPolicy::new();
        let low_batch = make_worker_with_cache(1, 4, 0.5, 0.1);
        let high_batch = make_worker_with_cache(1, 4, 0.5, 0.9);
        let workers = vec![low_batch.clone(), high_batch.clone()];
        let sel = policy.select(&workers, &ctx()).unwrap();
        assert_eq!(
            sel.id, high_batch.id,
            "higher batch headroom should be preferred"
        );
    }

    #[test]
    fn cache_affinity_prefix_lru_evicts_oldest() {
        let policy = CacheAffinityPolicy::new();
        let worker = make_worker(0, 4);

        // Fill LRU to capacity with distinct model IDs.
        for i in 0..PREFIX_LRU_CAPACITY {
            policy.record_dispatch(worker.id, &format!("model-{i}"));
        }
        // model-0 should still be present.
        {
            let cache = policy.prefix_lock();
            assert!(CacheAffinityPolicy::has_prefix(
                &cache, &worker.id, "model-0"
            ));
        }

        // One more entry should evict model-0.
        policy.record_dispatch(worker.id, "model-overflow");
        {
            let cache = policy.prefix_lock();
            assert!(
                !CacheAffinityPolicy::has_prefix(&cache, &worker.id, "model-0"),
                "oldest entry should be evicted"
            );
            assert!(CacheAffinityPolicy::has_prefix(
                &cache,
                &worker.id,
                "model-overflow"
            ));
        }
    }

    #[test]
    fn cache_affinity_duplicate_moves_to_back() {
        let policy = CacheAffinityPolicy::new();
        let worker = make_worker(0, 4);

        policy.record_dispatch(worker.id, "a");
        policy.record_dispatch(worker.id, "b");
        policy.record_dispatch(worker.id, "a"); // move "a" to back

        let cache = policy.prefix_lock();
        let hashes = cache.get(&worker.id).unwrap();
        assert_eq!(hashes.len(), 2, "no duplicates after re-dispatch");
    }
}
