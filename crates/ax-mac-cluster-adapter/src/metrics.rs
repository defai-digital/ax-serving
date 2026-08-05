//! Bounded lifecycle metrics for Mac cluster operations.
//!
//! Labels never include prompts, outputs, user IDs, or rank-local cache indexes.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

/// Operator-visible stage counters and cumulative durations.
#[derive(Debug, Default)]
pub struct ClusterMetrics {
    pub placement_total: AtomicU64,
    pub placement_micros: AtomicU64,
    pub download_total: AtomicU64,
    pub download_micros: AtomicU64,
    pub connect_total: AtomicU64,
    pub connect_micros: AtomicU64,
    pub load_total: AtomicU64,
    pub load_micros: AtomicU64,
    pub warmup_total: AtomicU64,
    pub warmup_micros: AtomicU64,
    pub request_total: AtomicU64,
    pub request_micros: AtomicU64,
    pub transport_error_total: AtomicU64,
    pub rank_failure_total: AtomicU64,
    pub generation_fence_total: AtomicU64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClusterMetricsSnapshot {
    pub placement_total: u64,
    pub placement_micros: u64,
    pub download_total: u64,
    pub download_micros: u64,
    pub connect_total: u64,
    pub connect_micros: u64,
    pub load_total: u64,
    pub load_micros: u64,
    pub warmup_total: u64,
    pub warmup_micros: u64,
    pub request_total: u64,
    pub request_micros: u64,
    pub transport_error_total: u64,
    pub rank_failure_total: u64,
    pub generation_fence_total: u64,
}

impl ClusterMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record_stage(&self, stage: MetricStage, elapsed: std::time::Duration) {
        let micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        match stage {
            MetricStage::Placement => {
                self.placement_total.fetch_add(1, Ordering::Relaxed);
                self.placement_micros.fetch_add(micros, Ordering::Relaxed);
            }
            MetricStage::Download => {
                self.download_total.fetch_add(1, Ordering::Relaxed);
                self.download_micros.fetch_add(micros, Ordering::Relaxed);
            }
            MetricStage::Connect => {
                self.connect_total.fetch_add(1, Ordering::Relaxed);
                self.connect_micros.fetch_add(micros, Ordering::Relaxed);
            }
            MetricStage::Load => {
                self.load_total.fetch_add(1, Ordering::Relaxed);
                self.load_micros.fetch_add(micros, Ordering::Relaxed);
            }
            MetricStage::Warmup => {
                self.warmup_total.fetch_add(1, Ordering::Relaxed);
                self.warmup_micros.fetch_add(micros, Ordering::Relaxed);
            }
            MetricStage::Request => {
                self.request_total.fetch_add(1, Ordering::Relaxed);
                self.request_micros.fetch_add(micros, Ordering::Relaxed);
            }
        }
    }

    pub fn record_transport_error(&self) {
        self.transport_error_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rank_failure(&self) {
        self.rank_failure_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_generation_fence(&self) {
        self.generation_fence_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ClusterMetricsSnapshot {
        ClusterMetricsSnapshot {
            placement_total: self.placement_total.load(Ordering::Relaxed),
            placement_micros: self.placement_micros.load(Ordering::Relaxed),
            download_total: self.download_total.load(Ordering::Relaxed),
            download_micros: self.download_micros.load(Ordering::Relaxed),
            connect_total: self.connect_total.load(Ordering::Relaxed),
            connect_micros: self.connect_micros.load(Ordering::Relaxed),
            load_total: self.load_total.load(Ordering::Relaxed),
            load_micros: self.load_micros.load(Ordering::Relaxed),
            warmup_total: self.warmup_total.load(Ordering::Relaxed),
            warmup_micros: self.warmup_micros.load(Ordering::Relaxed),
            request_total: self.request_total.load(Ordering::Relaxed),
            request_micros: self.request_micros.load(Ordering::Relaxed),
            transport_error_total: self.transport_error_total.load(Ordering::Relaxed),
            rank_failure_total: self.rank_failure_total.load(Ordering::Relaxed),
            generation_fence_total: self.generation_fence_total.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricStage {
    Placement,
    Download,
    Connect,
    Load,
    Warmup,
    Request,
}

/// Helper that records stage duration when dropped or finished.
pub struct StageTimer {
    metrics: Arc<ClusterMetrics>,
    stage: MetricStage,
    started: Instant,
    finished: bool,
}

impl StageTimer {
    pub fn start(metrics: Arc<ClusterMetrics>, stage: MetricStage) -> Self {
        Self {
            metrics,
            stage,
            started: Instant::now(),
            finished: false,
        }
    }

    pub fn finish(mut self) {
        self.metrics
            .record_stage(self.stage, self.started.elapsed());
        self.finished = true;
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics
                .record_stage(self.stage, self.started.elapsed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn metrics_distinguish_lifecycle_stages() {
        let metrics = ClusterMetrics::new();
        metrics.record_stage(MetricStage::Download, Duration::from_millis(5));
        metrics.record_stage(MetricStage::Load, Duration::from_millis(9));
        metrics.record_rank_failure();
        metrics.record_generation_fence();
        let snap = metrics.snapshot();
        assert_eq!(snap.download_total, 1);
        assert_eq!(snap.load_total, 1);
        assert!(snap.download_micros > 0);
        assert!(snap.load_micros > 0);
        assert_eq!(snap.rank_failure_total, 1);
        assert_eq!(snap.generation_fence_total, 1);
        assert_eq!(snap.request_total, 0);
    }
}
