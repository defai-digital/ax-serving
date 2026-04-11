//! Inference metrics: throughput counters + latency histograms + RSS.
//!
//! Ported from ax-engine's metrics module. No external dependencies.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ax_serving_engine::GenerationStats;

/// Per-request inference metrics.
#[derive(Debug, Default, Clone)]
pub struct InferenceMetrics {
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
    pub peak_rss_bytes: u64,
}

impl InferenceMetrics {
    pub fn prefill_tok_per_sec(&self) -> f64 {
        let secs = self.prefill_duration.as_secs_f64();
        if secs > 0.0 {
            self.prefill_tokens as f64 / secs
        } else {
            0.0
        }
    }

    pub fn decode_tok_per_sec(&self) -> f64 {
        let secs = self.decode_duration.as_secs_f64();
        if secs > 0.0 {
            self.decode_tokens as f64 / secs
        } else {
            0.0
        }
    }
}

/// Latency histogram for per-token decode timing.
#[derive(Debug)]
pub struct LatencyHistogram {
    samples: Vec<Duration>,
    sorted_samples: OnceLock<Vec<Duration>>,
}

impl LatencyHistogram {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(cap),
            sorted_samples: OnceLock::new(),
        }
    }

    pub fn record(&mut self, d: Duration) {
        self.samples.push(d);
        self.sorted_samples = OnceLock::new();
    }

    pub fn p50(&self) -> Duration {
        self.percentile(50)
    }
    pub fn p95(&self) -> Duration {
        self.percentile(95)
    }
    pub fn p99(&self) -> Duration {
        self.percentile(99)
    }

    /// Return (p50, p95, p99) in one sort pass instead of three.
    pub fn percentiles(&self) -> (Duration, Duration, Duration) {
        let sorted = self.sorted_samples();
        if sorted.is_empty() {
            return (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        }
        let n = sorted.len();
        let at = |p: usize| sorted[(n * p / 100).min(n - 1)];
        (at(50), at(95), at(99))
    }

    fn percentile(&self, p: usize) -> Duration {
        let sorted = self.sorted_samples();
        if sorted.is_empty() {
            return Duration::ZERO;
        }
        let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
        sorted[idx]
    }

    fn sorted_samples(&self) -> &[Duration] {
        self.sorted_samples
            .get_or_init(|| {
                let mut sorted = self.samples.clone();
                sorted.sort_unstable();
                sorted
            })
            .as_slice()
    }
}

/// Sliding-window error-rate tracker for SLO burn-rate alerting.
///
/// Stores `(timestamp_ms, is_error)` samples; prunes samples older than
/// `window_ms` on each `record()` call.
pub struct BurnRateWindow {
    window_ms: u64,
    samples: VecDeque<(u64, bool)>,
}

const BURN_RATE_MAX_SAMPLES: usize = 100_000;

impl BurnRateWindow {
    pub fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            samples: VecDeque::new(),
        }
    }

    pub fn record(&mut self, is_error: bool) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        while self
            .samples
            .front()
            .is_some_and(|s| now_ms.saturating_sub(s.0) > self.window_ms)
        {
            self.samples.pop_front();
        }
        while self.samples.len() >= BURN_RATE_MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back((now_ms, is_error));
    }

    /// Burn rate = (error rate in window) / error_budget.
    /// Returns 0.0 if no samples or error_budget ≤ 0.
    pub fn burn_rate(&self, error_budget: f64) -> f64 {
        if self.samples.is_empty() || error_budget <= 0.0 {
            return 0.0;
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut total = 0usize;
        let mut errors = 0usize;
        for &(timestamp_ms, is_error) in self.samples.iter().rev() {
            if now_ms.saturating_sub(timestamp_ms) > self.window_ms {
                break;
            }
            total += 1;
            if is_error {
                errors += 1;
            }
        }
        if total == 0 {
            return 0.0;
        }
        let rate = errors as f64 / total as f64;
        rate / error_budget
    }
}

/// Sliding window duration for the fast SLO burn-rate alert (1 hour).
const BURN_RATE_1H_MS: u64 = 60 * 60 * 1_000;
/// Sliding window duration for the slow SLO burn-rate alert (6 hours).
const BURN_RATE_6H_MS: u64 = 6 * 60 * 60 * 1_000;

/// Global metrics store.
pub struct MetricsStore {
    pub uptime_start: Instant,
    /// 1-hour sliding error-rate window for fast burn-rate alerting (> 14.4×).
    pub burn_1h: Mutex<BurnRateWindow>,
    /// 6-hour sliding error-rate window for slow burn-rate alerting (> 6.0×).
    pub burn_6h: Mutex<BurnRateWindow>,
    /// Most recent decode throughput reported by a completed request.
    recent_decode_tok_per_sec_bits: AtomicU64,
    /// Most recent prefill throughput reported by a completed request.
    recent_prefill_tok_per_sec_bits: AtomicU64,
    /// Non-streaming requests that executed inference because no cache result was available.
    cold_requests_total: AtomicU64,
    /// Requests served immediately from an exact response-cache hit.
    exact_cache_hits_total: AtomicU64,
    /// Requests served from a follower wait after another leader populated cache.
    cache_follower_hits_total: AtomicU64,
    /// Successful response-cache writes after a completed inference.
    cache_fills_total: AtomicU64,
}

impl MetricsStore {
    pub fn new() -> Self {
        Self {
            uptime_start: Instant::now(),
            burn_1h: Mutex::new(BurnRateWindow::new(BURN_RATE_1H_MS)),
            burn_6h: Mutex::new(BurnRateWindow::new(BURN_RATE_6H_MS)),
            recent_decode_tok_per_sec_bits: AtomicU64::new(0.0_f64.to_bits()),
            recent_prefill_tok_per_sec_bits: AtomicU64::new(0.0_f64.to_bits()),
            cold_requests_total: AtomicU64::new(0),
            exact_cache_hits_total: AtomicU64::new(0),
            cache_follower_hits_total: AtomicU64::new(0),
            cache_fills_total: AtomicU64::new(0),
        }
    }

    pub fn uptime_secs(&self) -> u64 {
        self.uptime_start.elapsed().as_secs()
    }

    pub fn record_generation_stats(&self, stats: &GenerationStats) {
        // Guard against NaN / infinity before storing.  Non-finite values arise
        // when the backend reports throughput for a request that completes in near-
        // zero time (decode_time ≈ 0 → tokens/time = +∞ or NaN).  Storing them
        // as-is is safe here, but `recent_decode_tok_per_sec()` is serialized by
        // the heartbeat loop via `serde_json::json!`, which panics on non-finite
        // f64 (serde_json rejects NaN/±∞ as invalid JSON).  Clamp to 0.0 so the
        // heartbeat never panics and the metric is simply stale rather than fatal.
        let decode = if stats.decode_tok_per_sec.is_finite() {
            stats.decode_tok_per_sec
        } else {
            0.0
        };
        let prefill = if stats.prefill_tok_per_sec.is_finite() {
            stats.prefill_tok_per_sec
        } else {
            0.0
        };
        self.recent_decode_tok_per_sec_bits
            .store(decode.to_bits(), Ordering::Relaxed);
        self.recent_prefill_tok_per_sec_bits
            .store(prefill.to_bits(), Ordering::Relaxed);
    }

    pub fn recent_decode_tok_per_sec(&self) -> f64 {
        f64::from_bits(self.recent_decode_tok_per_sec_bits.load(Ordering::Relaxed))
    }

    pub fn recent_prefill_tok_per_sec(&self) -> f64 {
        f64::from_bits(self.recent_prefill_tok_per_sec_bits.load(Ordering::Relaxed))
    }

    pub fn record_cold_request(&self) {
        self.cold_requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_exact_cache_hit(&self) {
        self.exact_cache_hits_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_follower_hit(&self) {
        self.cache_follower_hits_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_fill(&self) {
        self.cache_fills_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_slo_sample(&self, is_error: bool) {
        match self.burn_1h.lock() {
            Ok(mut metric) => metric.record(is_error),
            Err(err) => {
                tracing::warn!(%err, "burn-1h metric lock poisoned; continuing with poisoned state");
                err.into_inner().record(is_error);
            }
        }
        match self.burn_6h.lock() {
            Ok(mut metric) => metric.record(is_error),
            Err(err) => {
                tracing::warn!(%err, "burn-6h metric lock poisoned; continuing with poisoned state");
                err.into_inner().record(is_error);
            }
        }
    }

    pub fn cold_requests_total(&self) -> u64 {
        self.cold_requests_total.load(Ordering::Relaxed)
    }

    pub fn exact_cache_hits_total(&self) -> u64 {
        self.exact_cache_hits_total.load(Ordering::Relaxed)
    }

    pub fn cache_follower_hits_total(&self) -> u64 {
        self.cache_follower_hits_total.load(Ordering::Relaxed)
    }

    pub fn cache_fills_total(&self) -> u64 {
        self.cache_fills_total.load(Ordering::Relaxed)
    }
}

impl Default for MetricsStore {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII timer that records elapsed time on drop.
pub struct OpTimer {
    start: Instant,
}

impl OpTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Query current RSS from macOS task info.
pub fn current_rss_bytes() -> u64 {
    ax_serving_engine::current_rss_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── InferenceMetrics ───────────────────────────────────────────────────────

    #[test]
    fn inference_metrics_throughput_normal() {
        let m = InferenceMetrics {
            prefill_tokens: 512,
            prefill_duration: Duration::from_millis(500),
            decode_tokens: 100,
            decode_duration: Duration::from_millis(1_000),
            peak_rss_bytes: 0,
        };
        let prefill = m.prefill_tok_per_sec();
        let decode = m.decode_tok_per_sec();
        // 512 tokens / 0.5 s = 1024 tok/s
        assert!((prefill - 1_024.0).abs() < 1.0, "prefill={prefill}");
        // 100 tokens / 1.0 s = 100 tok/s
        assert!((decode - 100.0).abs() < 0.5, "decode={decode}");
    }

    #[test]
    fn inference_metrics_zero_duration_returns_zero() {
        let m = InferenceMetrics {
            prefill_tokens: 100,
            prefill_duration: Duration::ZERO,
            decode_tokens: 50,
            decode_duration: Duration::ZERO,
            peak_rss_bytes: 0,
        };
        assert_eq!(m.prefill_tok_per_sec(), 0.0);
        assert_eq!(m.decode_tok_per_sec(), 0.0);
    }

    // ── LatencyHistogram ───────────────────────────────────────────────────────

    #[test]
    fn latency_histogram_empty_returns_zero() {
        let h = LatencyHistogram::with_capacity(16);
        assert_eq!(h.p50(), Duration::ZERO);
        assert_eq!(h.p95(), Duration::ZERO);
        assert_eq!(h.p99(), Duration::ZERO);
        let (p50, p95, p99) = h.percentiles();
        assert_eq!(
            (p50, p95, p99),
            (Duration::ZERO, Duration::ZERO, Duration::ZERO)
        );
    }

    #[test]
    fn latency_histogram_single_sample() {
        let mut h = LatencyHistogram::with_capacity(4);
        h.record(Duration::from_millis(42));
        assert_eq!(h.p50(), Duration::from_millis(42));
        assert_eq!(h.p99(), Duration::from_millis(42));
    }

    #[test]
    fn latency_histogram_percentiles_known_values() {
        let mut h = LatencyHistogram::with_capacity(100);
        for i in 1u64..=100 {
            h.record(Duration::from_millis(i));
        }
        // Sorted: [1ms, 2ms, …, 100ms] (n=100)
        // percentile formula: idx = (n * p / 100).min(n-1)
        // p50: idx = 50, sorted[50] = 51ms
        // p95: idx = 95, sorted[95] = 96ms
        // p99: idx = 99, sorted[99] = 100ms
        let p50_ms = h.p50().as_millis();
        let p95_ms = h.p95().as_millis();
        let p99_ms = h.p99().as_millis();
        assert!((50..=52).contains(&p50_ms), "p50={p50_ms}");
        assert!((95..=97).contains(&p95_ms), "p95={p95_ms}");
        assert!((99..=100).contains(&p99_ms), "p99={p99_ms}");
    }

    #[test]
    fn latency_histogram_percentiles_triple_matches_individual() {
        let mut h = LatencyHistogram::with_capacity(20);
        for i in 1u64..=20 {
            h.record(Duration::from_millis(i * 10));
        }
        let (pp50, pp95, pp99) = h.percentiles();
        assert_eq!(pp50, h.p50());
        assert_eq!(pp95, h.p95());
        assert_eq!(pp99, h.p99());
    }

    #[test]
    fn latency_histogram_cache_is_invalidated_on_record() {
        let mut h = LatencyHistogram::with_capacity(4);
        h.record(Duration::from_millis(10));
        h.record(Duration::from_millis(20));
        assert_eq!(h.p99(), Duration::from_millis(20));

        h.record(Duration::from_millis(100));

        assert_eq!(h.p99(), Duration::from_millis(100));
        let (p50, p95, p99) = h.percentiles();
        assert_eq!(p50, Duration::from_millis(20));
        assert_eq!(p95, Duration::from_millis(100));
        assert_eq!(p99, Duration::from_millis(100));
    }

    // ── BurnRateWindow ─────────────────────────────────────────────────────────

    #[test]
    fn burn_rate_window_empty_returns_zero() {
        let w = BurnRateWindow::new(60_000);
        assert_eq!(w.burn_rate(0.05), 0.0);
    }

    #[test]
    fn burn_rate_window_zero_budget_returns_zero() {
        let mut w = BurnRateWindow::new(60_000);
        w.record(true);
        assert_eq!(w.burn_rate(0.0), 0.0);
        assert_eq!(w.burn_rate(-1.0), 0.0);
    }

    #[test]
    fn burn_rate_window_all_errors() {
        let mut w = BurnRateWindow::new(60_000);
        for _ in 0..10 {
            w.record(true);
        }
        // error_rate = 10/10 = 1.0; budget = 0.05 → burn_rate = 20.0
        let rate = w.burn_rate(0.05);
        assert!((rate - 20.0).abs() < 0.01, "burn_rate={rate}");
    }

    #[test]
    fn burn_rate_window_no_errors() {
        let mut w = BurnRateWindow::new(60_000);
        for _ in 0..10 {
            w.record(false);
        }
        assert_eq!(w.burn_rate(0.05), 0.0);
    }

    #[test]
    fn burn_rate_window_mixed() {
        let mut w = BurnRateWindow::new(60_000);
        for _ in 0..5 {
            w.record(true);
        }
        for _ in 0..5 {
            w.record(false);
        }
        // error_rate = 5/10 = 0.5; budget = 0.1 → burn_rate = 5.0
        let rate = w.burn_rate(0.1);
        assert!((rate - 5.0).abs() < 0.01, "burn_rate={rate}");
    }

    #[test]
    fn burn_rate_window_caps_samples() {
        let mut w = BurnRateWindow::new(60_000);
        for i in 0..(BURN_RATE_MAX_SAMPLES + 10) {
            w.record(i.is_multiple_of(2));
        }
        assert_eq!(w.samples.len(), BURN_RATE_MAX_SAMPLES);
    }

    // ── MetricsStore & OpTimer ─────────────────────────────────────────────────

    #[test]
    fn metrics_store_uptime_near_zero_on_construction() {
        let store = MetricsStore::new();
        let uptime = store.uptime_secs();
        assert!(
            uptime < 5,
            "uptime should be near 0 at construction, got {uptime}"
        );
    }

    #[test]
    fn metrics_store_default_equals_new() {
        // Both should construct without panic and have burn windows.
        let _s1 = MetricsStore::new();
        let _s2 = MetricsStore::default();
    }

    #[test]
    fn record_generation_stats_clamps_non_finite_to_zero() {
        // Non-finite throughput values (NaN, ±∞) must never reach the heartbeat
        // JSON serialization path where serde_json::json! would panic.
        let store = MetricsStore::new();
        let nan_stats = ax_serving_engine::GenerationStats {
            prompt_tokens: 0,
            completion_tokens: 0,
            prefill_tok_per_sec: f64::NAN,
            decode_tok_per_sec: f64::INFINITY,
            stop_reason: String::new(),
        };
        store.record_generation_stats(&nan_stats);
        assert_eq!(
            store.recent_decode_tok_per_sec(),
            0.0,
            "infinity must be clamped to 0.0"
        );
        assert_eq!(
            store.recent_prefill_tok_per_sec(),
            0.0,
            "NaN must be clamped to 0.0"
        );
        // Verify that finite values pass through unchanged.
        let finite_stats = ax_serving_engine::GenerationStats {
            prompt_tokens: 512,
            completion_tokens: 100,
            prefill_tok_per_sec: 1200.0,
            decode_tok_per_sec: 85.5,
            stop_reason: "stop".to_string(),
        };
        store.record_generation_stats(&finite_stats);
        assert!((store.recent_decode_tok_per_sec() - 85.5).abs() < 0.001);
        assert!((store.recent_prefill_tok_per_sec() - 1200.0).abs() < 0.001);
    }

    #[test]
    fn request_class_counters_increment() {
        let store = MetricsStore::new();
        store.record_cold_request();
        store.record_exact_cache_hit();
        store.record_cache_follower_hit();
        store.record_cache_fill();
        store.record_slo_sample(true);
        assert_eq!(store.cold_requests_total(), 1);
        assert_eq!(store.exact_cache_hits_total(), 1);
        assert_eq!(store.cache_follower_hits_total(), 1);
        assert_eq!(store.cache_fills_total(), 1);
        assert!(store.burn_1h.lock().unwrap().burn_rate(0.001) > 0.0);
    }

    #[test]
    fn op_timer_elapsed_positive_after_sleep() {
        let t = OpTimer::start();
        std::thread::sleep(Duration::from_millis(5));
        assert!(t.elapsed() >= Duration::from_millis(1));
    }

    // ── current_rss_bytes ─────────────────────────────────────────────────────

    #[test]
    fn current_rss_bytes_returns_nonzero_on_running_process() {
        // The function delegates to the engine's macOS task_info call.
        // Any live process must have at least some resident memory.
        let rss = current_rss_bytes();
        assert!(
            rss > 0,
            "expected non-zero RSS for the current process, got {rss}"
        );
    }
}
