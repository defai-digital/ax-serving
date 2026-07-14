//! Dispatch metrics: latency histograms, outcome counters, and reservation renew signals.
//!
//! Reservation renew gauges/counters are low-cardinality measurement hooks so a future
//! shared renewer (P2) can be justified by data rather than assumed load.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed, cumulative microsecond buckets keep the dispatch hot path lock-free
/// and prevent unbounded label/cardinality growth.
pub const GATEWAY_LATENCY_BUCKETS_US: [u64; 9] =
    [50, 100, 250, 500, 1_000, 2_000, 5_000, 15_000, u64::MAX];

#[derive(Debug, Default)]
pub(super) struct DispatchMetrics {
    pub(super) requests_total: AtomicU64,
    pub(super) attempts_total: AtomicU64,
    pub(super) completed_total: AtomicU64,
    pub(super) failed_total: AtomicU64,
    pub(super) cancelled_total: AtomicU64,
    pub(super) endpoint_selection: AtomicLatencyHistogram,
    pub(super) response_headers: AtomicLatencyHistogram,
    pub(super) attempt_duration: AtomicLatencyHistogram,
    pub(super) time_to_first_byte: AtomicLatencyHistogram,
    pub(super) stream_duration: AtomicLatencyHistogram,
    pub(super) selection_selected_total: AtomicU64,
    pub(super) selection_no_candidate_total: AtomicU64,
    pub(super) selection_at_capacity_total: AtomicU64,
    pub(super) selection_error_total: AtomicU64,
    /// Approximate count of active per-attempt reservation renew loops.
    pub(super) reservation_renew_tasks: AtomicU64,
    pub(super) reservation_renew_ok_total: AtomicU64,
    pub(super) reservation_renew_fenced_total: AtomicU64,
    pub(super) reservation_renew_error_total: AtomicU64,
}

#[derive(Debug)]
pub(super) struct AtomicLatencyHistogram {
    count: AtomicU64,
    sum_us: AtomicU64,
    cumulative_buckets: [AtomicU64; GATEWAY_LATENCY_BUCKETS_US.len()],
}

impl Default for AtomicLatencyHistogram {
    fn default() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            cumulative_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl AtomicLatencyHistogram {
    pub(super) fn record(&self, elapsed: std::time::Duration) {
        let elapsed_us = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .sum_us
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(elapsed_us))
            });
        for (index, upper_bound) in GATEWAY_LATENCY_BUCKETS_US.iter().enumerate() {
            if elapsed_us <= *upper_bound {
                self.cumulative_buckets[index].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub(super) fn snapshot(&self) -> LatencyHistogramSnapshot {
        LatencyHistogramSnapshot {
            count: self.count.load(Ordering::Relaxed),
            sum_us: self.sum_us.load(Ordering::Relaxed),
            cumulative_buckets: std::array::from_fn(|index| {
                self.cumulative_buckets[index].load(Ordering::Relaxed)
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyHistogramSnapshot {
    pub count: u64,
    pub sum_us: u64,
    pub cumulative_buckets: [u64; GATEWAY_LATENCY_BUCKETS_US.len()],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SelectionOutcome {
    Selected,
    NoCandidate,
    AtCapacity,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchMetricsSnapshot {
    pub requests_total: u64,
    pub attempts_total: u64,
    pub completed_total: u64,
    pub failed_total: u64,
    pub cancelled_total: u64,
    pub retries_total: u64,
    pub endpoint_selection: LatencyHistogramSnapshot,
    pub response_headers: LatencyHistogramSnapshot,
    pub attempt_duration: LatencyHistogramSnapshot,
    pub time_to_first_byte: LatencyHistogramSnapshot,
    pub stream_duration: LatencyHistogramSnapshot,
    pub selection_selected_total: u64,
    pub selection_no_candidate_total: u64,
    pub selection_at_capacity_total: u64,
    pub selection_error_total: u64,
    /// Active per-attempt reservation renew tasks (gauge).
    pub reservation_renew_tasks: u64,
    pub reservation_renew_ok_total: u64,
    pub reservation_renew_fenced_total: u64,
    pub reservation_renew_error_total: u64,
}

pub(super) struct DispatchOutcomeGuard {
    metrics: Arc<DispatchMetrics>,
    resolved: bool,
    successful_response: bool,
    attempt_started: std::time::Instant,
    first_byte_recorded: bool,
}

impl DispatchOutcomeGuard {
    pub(super) fn new(
        metrics: Arc<DispatchMetrics>,
        successful_response: bool,
        attempt_started: std::time::Instant,
    ) -> Self {
        Self {
            metrics,
            resolved: false,
            successful_response,
            attempt_started,
            first_byte_recorded: false,
        }
    }

    pub(super) fn first_byte(&mut self) {
        if !self.first_byte_recorded {
            self.metrics
                .time_to_first_byte
                .record(self.attempt_started.elapsed());
            self.first_byte_recorded = true;
        }
    }

    fn finish_timing(&self) {
        let elapsed = self.attempt_started.elapsed();
        self.metrics.attempt_duration.record(elapsed);
        self.metrics.stream_duration.record(elapsed);
    }

    pub(super) fn completed(&mut self) {
        if !self.resolved {
            self.finish_timing();
            if self.successful_response {
                self.metrics.completed_total.fetch_add(1, Ordering::Relaxed);
            }
            self.resolved = true;
        }
    }

    pub(super) fn failed(&mut self) {
        if !self.resolved {
            self.finish_timing();
            if self.successful_response {
                self.metrics.failed_total.fetch_add(1, Ordering::Relaxed);
            }
            self.resolved = true;
        }
    }
}

impl Drop for DispatchOutcomeGuard {
    fn drop(&mut self) {
        if !self.resolved {
            self.finish_timing();
            if self.successful_response {
                self.metrics.cancelled_total.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
