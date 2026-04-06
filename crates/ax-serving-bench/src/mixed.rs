//! Mixed-workload benchmark: exercises the scheduler under realistic
//! heterogeneous load (short / medium / long prompts, concurrent clients).
//!
//! Three prompt classes are cycled in equal proportion:
//!   - **short**:  ~32 tokens — latency-sensitive interactive requests
//!   - **medium**: ~256 tokens — document summarisation
//!   - **long**:   ~512 tokens — context-heavy reasoning
//!
//! Metrics reported per class and overall: P50 / P95 / P99 latency, RPS,
//! error rate.  A pass/fail gate checks that the overall P99 < `target_p99_ms`.
//!
//! # Usage
//!
//! ```text
//! ax-serving-bench mixed \
//!     --url http://127.0.0.1:18080 \
//!     --model llama3-8b \
//!     --concurrency 4 \
//!     --requests 60 \
//!     --target-p99-ms 10000
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;
use tracing::info;

use crate::bench_common::{CLASSES, LatencyCollector, create_bench_client, mean, percentile, rps};

// ── Config ─────────────────────────────────────────────────────────────────────

pub struct MixedConfig {
    /// Base URL of the running ax-serving REST server.
    pub url: String,
    /// Maximum concurrent in-flight requests (used when concurrency_levels is empty).
    pub concurrency: usize,
    /// Total number of requests to issue (evenly split across classes).
    pub total_requests: usize,
    /// Model ID to use in each request.
    pub model_id: String,
    /// `max_tokens` for each request.
    pub max_tokens: usize,
    /// Pass/fail gate: overall P99 must be below this threshold (ms).
    pub target_p99_ms: u64,
    /// If set, write benchmark results (worst-case P99 across levels) to this path.
    pub json: Option<PathBuf>,
    /// When non-empty, run the full suite at each listed concurrency level.
    /// When empty, run once with `concurrency`.
    pub concurrency_levels: Vec<usize>,
}

// ── Per-class results ──────────────────────────────────────────────────────────

struct ClassStats {
    label: &'static str,
    latencies_ms: Vec<u64>,
}

impl ClassStats {
    fn p50(&self) -> f64 {
        percentile(&self.latencies_ms, 50)
    }
    fn p95(&self) -> f64 {
        percentile(&self.latencies_ms, 95)
    }
    fn p99(&self) -> f64 {
        percentile(&self.latencies_ms, 99)
    }
    fn mean(&self) -> f64 {
        mean(&self.latencies_ms)
    }
}

// ── JSON output format ─────────────────────────────────────────────────────────

/// Worst-case P99 values across all concurrency levels.  Used as the machine-
/// readable artifact for CI regression gates.  All fields are `Option<f64>` so
/// that a baseline file with `null` values can skip the corresponding checks.
#[derive(Serialize)]
pub struct MixedResults {
    pub short_p99: Option<f64>,
    pub medium_p99: Option<f64>,
    pub long_p99: Option<f64>,
    pub overall_p99: Option<f64>,
}

// ── Single-run result ──────────────────────────────────────────────────────────

struct RunResult {
    short_p99: f64,
    medium_p99: f64,
    long_p99: f64,
    overall_p99: f64,
}

// ── Entry point ────────────────────────────────────────────────────────────────

pub async fn run(cfg: MixedConfig) -> Result<()> {
    let levels: Vec<usize> = if cfg.concurrency_levels.is_empty() {
        vec![cfg.concurrency]
    } else {
        cfg.concurrency_levels.clone()
    };

    let client = create_bench_client()?;

    let mut all_results: Vec<RunResult> = Vec::with_capacity(levels.len());

    for &concurrency in &levels {
        let result = run_at_concurrency(&cfg, &client, concurrency).await?;
        all_results.push(result);
    }

    // Write JSON output if requested (worst-case P99 across all concurrency levels).
    if let Some(path) = &cfg.json {
        let worst = MixedResults {
            short_p99: all_results.iter().map(|r| r.short_p99).reduce(f64::max),
            medium_p99: all_results.iter().map(|r| r.medium_p99).reduce(f64::max),
            long_p99: all_results.iter().map(|r| r.long_p99).reduce(f64::max),
            overall_p99: all_results.iter().map(|r| r.overall_p99).reduce(f64::max),
        };
        let json_str = serde_json::to_string_pretty(&worst)?;
        std::fs::write(path, &json_str)?;
        eprintln!("Results written to {}", path.display());
    }

    Ok(())
}

// ── Single concurrency-level run ───────────────────────────────────────────────

async fn run_at_concurrency(
    cfg: &MixedConfig,
    client: &Arc<reqwest::Client>,
    concurrency: usize,
) -> Result<RunResult> {
    info!(
        url = %cfg.url,
        concurrency = concurrency,
        total = cfg.total_requests,
        model = %cfg.model_id,
        "mixed-workload bench starting"
    );

    anyhow::ensure!(concurrency > 0, "concurrency must be >= 1");
    let sem = Arc::new(Semaphore::new(concurrency));
    let url = format!("{}/v1/chat/completions", cfg.url);

    let collector = Arc::new(LatencyCollector::new());

    let bench_start = Instant::now();
    let mut handles = Vec::with_capacity(cfg.total_requests);

    for i in 0..cfg.total_requests {
        let class = CLASSES[i % 3];
        let permit = Arc::clone(&sem).acquire_owned().await?;

        let client = Arc::clone(client);
        let url = url.clone();
        let model_id = cfg.model_id.clone();
        let max_tokens = cfg.max_tokens;
        let collector = Arc::clone(&collector);

        handles.push(tokio::spawn(async move {
            let _permit = permit;

            let body = json!({
                "model": model_id,
                "messages": [{"role": "user", "content": class.prompt()}],
                "max_tokens": max_tokens,
                "stream": false,
            });

            let t = Instant::now();
            let result = client.post(&url).json(&body).send().await;
            let elapsed_ms = t.elapsed().as_millis() as u64;

            let ok = match result {
                Ok(r) => r.status().is_success(),
                Err(_) => false,
            };

            if ok {
                collector.record(class, elapsed_ms);
            } else {
                collector.record_error();
            }
        }));
    }

    for h in handles {
        if let Err(e) = h.await
            && e.is_panic()
        {
            tracing::warn!("bench task panicked: {e}");
        }
    }

    let total_ms = bench_start.elapsed().as_millis() as u64;

    let (short_v, medium_v, long_v, errors) = collector.snapshot();

    let all_lats: Vec<u64> = short_v
        .iter()
        .chain(medium_v.iter())
        .chain(long_v.iter())
        .copied()
        .collect();

    let classes = [
        ClassStats {
            label: "short",
            latencies_ms: short_v,
        },
        ClassStats {
            label: "medium",
            latencies_ms: medium_v,
        },
        ClassStats {
            label: "long",
            latencies_ms: long_v,
        },
    ];

    let overall_p99 = percentile(&all_lats, 99);

    print_report(cfg, concurrency, &classes, &all_lats, errors, total_ms)?;

    Ok(RunResult {
        short_p99: classes[0].p99(),
        medium_p99: classes[1].p99(),
        long_p99: classes[2].p99(),
        overall_p99,
    })
}

// ── Report ─────────────────────────────────────────────────────────────────────

fn print_report(
    cfg: &MixedConfig,
    concurrency: usize,
    classes: &[ClassStats],
    all_lats: &[u64],
    errors: u64,
    total_ms: u64,
) -> anyhow::Result<()> {
    let total_success: usize = classes.iter().map(|c| c.latencies_ms.len()).sum();
    let computed_rps = rps(total_success, total_ms);

    println!("\nMixed workload benchmark");
    println!("========================");
    println!("URL: {}", cfg.url);
    println!(
        "Concurrency: {concurrency}, Requests: {} ({} per class), max_tokens: {}",
        cfg.total_requests,
        cfg.total_requests / 3,
        cfg.max_tokens
    );
    println!("Duration: {total_ms} ms  |  RPS: {computed_rps:.1}  |  Errors: {errors}");
    println!();
    println!(
        "{:<8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Class", "Requests", "Mean (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)"
    );
    println!("{}", "-".repeat(64));

    for c in classes {
        println!(
            "{:<8}  {:>8}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}",
            c.label,
            c.latencies_ms.len(),
            c.mean(),
            c.p50(),
            c.p95(),
            c.p99(),
        );
    }

    println!("{}", "-".repeat(64));
    let overall_p50 = percentile(all_lats, 50);
    let overall_p95 = percentile(all_lats, 95);
    let overall_p99 = percentile(all_lats, 99);
    let overall_mean = mean(all_lats);
    println!(
        "{:<8}  {:>8}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}",
        "overall", total_success, overall_mean, overall_p50, overall_p95, overall_p99,
    );
    println!();

    let gate_pass = overall_p99 < cfg.target_p99_ms as f64;
    println!(
        "Gate: P99 < {}ms: {} (actual: {:.0}ms)",
        cfg.target_p99_ms,
        if gate_pass { "PASS" } else { "FAIL" },
        overall_p99
    );
    anyhow::ensure!(
        gate_pass,
        "P99 gate failed: {:.0}ms >= {}ms target",
        overall_p99,
        cfg.target_p99_ms
    );
    Ok(())
}
