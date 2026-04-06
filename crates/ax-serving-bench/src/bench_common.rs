//! Shared utilities for benchmark subcommands.

use std::sync::Arc;
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::Client;

// ── Prompt classes ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum PromptClass {
    Short,
    Medium,
    Long,
}

impl PromptClass {
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Short => "What is 2+2?",
            Self::Medium => {
                "Explain the differences between supervised, unsupervised, and reinforcement learning in machine learning. Cover the key characteristics, when to use each approach, and give one concrete example for each."
            }
            Self::Long => {
                "You are an expert in distributed systems. Provide a comprehensive analysis of the CAP theorem and its implications for modern distributed database design. Include discussion of BASE vs ACID properties, common trade-offs made by popular databases like Cassandra, MongoDB, and PostgreSQL, and provide guidance on selecting the right consistency model for different use cases such as financial transactions, social media feeds, and real-time analytics."
            }
        }
    }
}

pub const CLASSES: [PromptClass; 3] = [PromptClass::Short, PromptClass::Medium, PromptClass::Long];

// ── Statistics ─────────────────────────────────────────────────────────────────

pub fn percentile(values: &[u64], p: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)] as f64
}

/// Same nearest-rank percentile formula as [`percentile`] but for `f64` slices.
/// Deduplicates the ad-hoc implementations in `service_perf.rs` (BUG-090).
pub fn percentile_f64(values: &[f64], p: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
}

pub fn mean(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
}

pub fn rps(total_success: usize, total_ms: u64) -> f64 {
    if total_ms > 0 {
        total_success as f64 / (total_ms as f64 / 1_000.0)
    } else {
        0.0
    }
}

// ── Concurrency helpers ────────────────────────────────────────────────────────

pub fn lock_or_recover<T>(mutex: &std::sync::Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(err) => {
            eprintln!("bench mutex poisoned; recovering inner state and continuing");
            err.into_inner()
        }
    }
}

pub fn create_bench_client() -> anyhow::Result<Arc<Client>> {
    Ok(Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?,
    ))
}

/// Thread-safe latency collector for three prompt classes + error counter.
pub struct LatencyCollector {
    pub short: Arc<std::sync::Mutex<Vec<u64>>>,
    pub medium: Arc<std::sync::Mutex<Vec<u64>>>,
    pub long: Arc<std::sync::Mutex<Vec<u64>>>,
    pub errors: Arc<AtomicU64>,
}

impl Default for LatencyCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyCollector {
    pub fn new() -> Self {
        Self {
            short: Arc::new(std::sync::Mutex::new(Vec::new())),
            medium: Arc::new(std::sync::Mutex::new(Vec::new())),
            long: Arc::new(std::sync::Mutex::new(Vec::new())),
            errors: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn record(&self, class: PromptClass, elapsed_ms: u64) {
        let bucket = match class {
            PromptClass::Short => &self.short,
            PromptClass::Medium => &self.medium,
            PromptClass::Long => &self.long,
        };
        lock_or_recover(bucket).push(elapsed_ms);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>, u64) {
        let s = lock_or_recover(&self.short).clone();
        let m = lock_or_recover(&self.medium).clone();
        let l = lock_or_recover(&self.long).clone();
        let e = self.errors.load(Ordering::Relaxed);
        (s, m, l, e)
    }
}
