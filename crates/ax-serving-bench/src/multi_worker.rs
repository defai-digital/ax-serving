//! TASK-MW-016: Multi-worker concurrent benchmark.
//!
//! Spawns `concurrency` async tasks, each sending inference requests to a
//! running orchestrator endpoint.  Measures per-request latency and reports
//! throughput, P50/P95/P99 percentiles, and error rate.
//!
//! # Usage
//!
//! ```text
//! ax-serving-bench multi-worker \
//!     --url http://127.0.0.1:18080 \
//!     --workers 8 \
//!     --requests 200 \
//!     --model llama3-8b
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use tokio::sync::Semaphore;
use tracing::{info, warn};

// ── Config ────────────────────────────────────────────────────────────────────

pub struct MultiWorkerConfig {
    /// Orchestrator URL, e.g. `http://127.0.0.1:18080`.
    pub url: String,
    /// Maximum concurrent in-flight requests.
    pub concurrency: usize,
    /// Total number of requests to issue (used when `duration_secs` is None).
    pub total_requests: usize,
    /// If set, run for this many seconds instead of `total_requests`.
    pub duration_secs: Option<u64>,
    /// Model ID to use in the request body.
    pub model_id: String,
    /// User prompt text to send.
    pub prompt: String,
    /// `max_tokens` for each request.
    pub decode_tokens: usize,
    /// Dispatcher mode label for reports ("direct" or "nats").
    pub mode: String,
    /// Optional path to write a JSON results file.
    pub json: Option<PathBuf>,
    /// If true, auto-write a markdown report to `target/bench-reports/`.
    pub write_report: bool,
}

// ── Results ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MultiWorkerResults {
    pub total_requests: usize,
    pub success: u64,
    pub errors: u64,
    pub total_duration_ms: u64,
    pub requests_per_sec: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    /// Whether the dispatch overhead gate (< 5 ms median) passed.
    pub overhead_gate_pass: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(cfg: MultiWorkerConfig) -> Result<()> {
    info!(
        url = %cfg.url,
        concurrency = cfg.concurrency,
        mode = %cfg.mode,
        model_id = %cfg.model_id,
        "multi-worker bench starting"
    );

    let results = bench(&cfg).await?;
    print_report(&cfg, &results);

    if let Some(path) = &cfg.json {
        write_json(path, &cfg, &results)?;
    }

    if cfg.write_report {
        write_markdown_report(&cfg, &results)?;
    }

    Ok(())
}

// ── Core benchmark ────────────────────────────────────────────────────────────

async fn bench(cfg: &MultiWorkerConfig) -> Result<MultiWorkerResults> {
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let url = format!("{}/v1/chat/completions", cfg.url.trim_end_matches('/'));
    let request_body = json!({
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": cfg.prompt}],
        "max_tokens": cfg.decode_tokens,
        "stream": false,
    });
    let body_str = serde_json::to_string(&request_body)?;

    let sem = Arc::new(Semaphore::new(cfg.concurrency));
    let success_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));
    let latencies: Arc<tokio::sync::Mutex<Vec<f64>>> = Arc::new(tokio::sync::Mutex::new(
        Vec::with_capacity(cfg.total_requests.max(1024)),
    ));

    let start = Instant::now();
    let deadline = cfg.duration_secs.map(|d| start + Duration::from_secs(d));

    // Duration-based mode: dispatch requests in a loop until the wall clock
    // deadline is reached.  Fixed-count mode: dispatch exactly total_requests.
    let mut handles = Vec::new();
    let mut req_index: usize = 0;

    loop {
        // Check termination condition.
        if deadline.is_some_and(|dl| Instant::now() >= dl) || req_index >= cfg.total_requests {
            break;
        }

        // Throttle: only spawn if we have semaphore capacity.
        let permit = sem.clone().acquire_owned().await;
        let Ok(permit) = permit else { break };

        // Re-check the deadline after blocking on the semaphore: in duration mode a
        // full-concurrency burst can block here past the wall-clock deadline.
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            break;
        }

        let client = client.clone();
        let url = url.clone();
        let body_str = body_str.clone();
        let success_count = Arc::clone(&success_count);
        let error_count = Arc::clone(&error_count);
        let latencies = Arc::clone(&latencies);
        let i = req_index;

        let h = tokio::spawn(async move {
            let _permit = permit; // held for the life of the request
            let req_start = Instant::now();
            let result = client
                .post(&url)
                .header("content-type", "application/json")
                .body(body_str)
                .send()
                .await;

            match result {
                Err(e) => {
                    warn!(req = i, %e, "request failed");
                    error_count.fetch_add(1, Ordering::Relaxed);
                }
                Ok(resp) => {
                    let status = resp.status();
                    let _ = resp.bytes().await;
                    let elapsed_ms = req_start.elapsed().as_secs_f64() * 1000.0;
                    if status.is_success() {
                        success_count.fetch_add(1, Ordering::Relaxed);
                        latencies.lock().await.push(elapsed_ms);
                    } else {
                        warn!(req = i, status = %status, "non-2xx response");
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
        handles.push(h);
        req_index += 1;
    }

    for h in handles {
        let _ = h.await;
    }

    let total_ms = start.elapsed().as_millis() as u64;
    let success = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);

    let mut lats = latencies.lock().await.clone();
    sanitize_and_sort_latencies(&mut lats);

    let (p50, p95, p99, mean) = percentiles(&lats);
    let rps = if total_ms > 0 {
        (success + errors) as f64 / (total_ms as f64 / 1000.0)
    } else {
        0.0
    };

    // Dispatch overhead gate: p50 latency < 5 ms means dispatch overhead is within budget.
    // (This is meaningful only when the model is very fast or stubbed; real LLM latency
    // will dominate. The gate is documented in TASK-MW-016/017 as < 5 ms *dispatch overhead*,
    // measurable in isolation without an actual inference model by pointing at a mock server.)
    // Guard: if no requests succeeded, lats is empty and p50 is 0.0 — which would
    // incorrectly pass the gate. Only pass when we have actual measurements.
    let overhead_gate_pass = !lats.is_empty() && p50 < 5.0;

    Ok(MultiWorkerResults {
        total_requests: req_index,
        success,
        errors,
        total_duration_ms: total_ms,
        requests_per_sec: rps,
        p50_ms: p50,
        p95_ms: p95,
        p99_ms: p99,
        mean_ms: mean,
        overhead_gate_pass,
    })
}

fn percentiles(sorted: &[f64]) -> (f64, f64, f64, f64) {
    if sorted.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let pct = |p: f64| {
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    (pct(50.0), pct(95.0), pct(99.0), mean)
}

fn sanitize_and_sort_latencies(latencies: &mut Vec<f64>) {
    latencies.retain(|v| v.is_finite() && *v >= 0.0);
    latencies.sort_by(|a, b| a.total_cmp(b));
}

// ── Reporting ─────────────────────────────────────────────────────────────────

fn print_report(cfg: &MultiWorkerConfig, r: &MultiWorkerResults) {
    println!();
    println!("# Multi-Worker Benchmark Report");
    println!();
    println!("| Parameter        | Value                       |");
    println!("|------------------|-----------------------------|");
    println!("| URL              | {:<27} |", cfg.url);
    println!("| Mode             | {:<27} |", cfg.mode);
    println!("| Model            | {:<27} |", cfg.model_id);
    println!("| Concurrency      | {:<27} |", cfg.concurrency);
    println!("| Total requests   | {:<27} |", r.total_requests);
    println!("| Successful       | {:<27} |", r.success);
    println!("| Errors           | {:<27} |", r.errors);
    println!(
        "| Total time       | {:<27} |",
        format!("{} ms", r.total_duration_ms)
    );
    println!(
        "| Throughput       | {:<27} |",
        format!("{:.1} req/s", r.requests_per_sec)
    );
    println!();
    println!("## Latency (successful requests)");
    println!();
    println!("| Percentile | Latency    |");
    println!("|------------|------------|");
    println!("| mean       | {:<10} |", format!("{:.1} ms", r.mean_ms));
    println!("| P50        | {:<10} |", format!("{:.1} ms", r.p50_ms));
    println!("| P95        | {:<10} |", format!("{:.1} ms", r.p95_ms));
    println!("| P99        | {:<10} |", format!("{:.1} ms", r.p99_ms));
    println!();
    println!("## Gates");
    println!();
    let gate_str = if r.overhead_gate_pass { "PASS" } else { "FAIL" };
    println!("| Gate                         | Result |");
    println!("|------------------------------|--------|");
    println!("| dispatch overhead < 5 ms P50 | {gate_str:<6} |");
    println!();
}

fn write_markdown_report(cfg: &MultiWorkerConfig, r: &MultiWorkerResults) -> Result<()> {
    let date = chrono_date_str();
    let dir = "target/bench-reports";
    std::fs::create_dir_all(dir)?;
    let path = format!("{dir}/bench-multi-worker-{date}.md");
    let gate_str = if r.overhead_gate_pass {
        "PASS ✓"
    } else {
        "FAIL ✗"
    };
    let error_rate = if r.total_requests > 0 {
        r.errors as f64 / r.total_requests as f64 * 100.0
    } else {
        0.0
    };

    let content = format!(
        "# Multi-Worker Benchmark — {date}\n\
         \n\
         **Date:** {date}  \n\
         **Mode:** {mode}  \n\
         **Policy:** least_inflight (default)  \n\
         \n\
         ## Configuration\n\
         \n\
         | Parameter | Value |\n\
         |-----------|-------|\n\
         | URL | {url} |\n\
         | Mode | {mode} |\n\
         | Model | {model} |\n\
         | Concurrency | {concurrency} |\n\
         | Total requests | {total_requests} |\n\
         | Decode tokens | {decode_tokens} |\n\
         \n\
         ## Results\n\
         \n\
         | Metric | Value |\n\
         |--------|-------|\n\
         | Successful | {success} |\n\
         | Errors | {errors} ({error_rate:.1}%) |\n\
         | Total duration | {total_ms} ms |\n\
         | Throughput | {rps:.1} req/s |\n\
         | Latency mean | {mean:.1} ms |\n\
         | Latency P50 | {p50:.1} ms |\n\
         | Latency P95 | {p95:.1} ms |\n\
         | Latency P99 | {p99:.1} ms |\n\
         \n\
         ## Gates\n\
         \n\
         | Gate | Threshold | Result |\n\
         |------|-----------|--------|\n\
         | Dispatch overhead P50 | < 5 ms | {gate_str} |\n\
         | Error rate | < 1% | {err_gate} |\n\
         ",
        date = date,
        mode = cfg.mode,
        url = cfg.url,
        model = cfg.model_id,
        concurrency = cfg.concurrency,
        total_requests = r.total_requests,
        decode_tokens = cfg.decode_tokens,
        success = r.success,
        errors = r.errors,
        error_rate = error_rate,
        total_ms = r.total_duration_ms,
        rps = r.requests_per_sec,
        mean = r.mean_ms,
        p50 = r.p50_ms,
        p95 = r.p95_ms,
        p99 = r.p99_ms,
        gate_str = gate_str,
        err_gate = if error_rate < 1.0 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        },
    );

    std::fs::write(&path, content)?;
    info!(path = %path, "markdown report written");
    println!("Report written to {path}");
    Ok(())
}

fn chrono_date_str() -> String {
    // Minimal date string without pulling in chrono: use std time.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Approximate: 2026-03-01 = unix 1772150400; good enough for filenames.
    let days_since_epoch = secs / 86400;
    // Zeller's congruence — accurate for years 2000+.
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn write_json(path: &PathBuf, cfg: &MultiWorkerConfig, r: &MultiWorkerResults) -> Result<()> {
    let obj = json!({
        "config": {
            "url": cfg.url,
            "mode": cfg.mode,
            "concurrency": cfg.concurrency,
            "total_requests": r.total_requests,
            "model_id": cfg.model_id,
            "decode_tokens": cfg.decode_tokens,
        },
        "results": {
            "success": r.success,
            "errors": r.errors,
            "total_duration_ms": r.total_duration_ms,
            "requests_per_sec": r.requests_per_sec,
            "latency_ms": {
                "mean": r.mean_ms,
                "p50": r.p50_ms,
                "p95": r.p95_ms,
                "p99": r.p99_ms,
            },
            "gates": {
                "overhead_p50_lt_5ms": r.overhead_gate_pass,
            }
        }
    });
    std::fs::write(path, serde_json::to_string_pretty(&obj)?)?;
    info!(path = %path.display(), "results written");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{percentiles, sanitize_and_sort_latencies};

    #[test]
    fn sanitize_and_sort_latencies_filters_invalid_values() {
        let mut lats = vec![10.0, f64::NAN, 5.0, f64::INFINITY, -1.0, 7.0];
        sanitize_and_sort_latencies(&mut lats);
        assert_eq!(lats, vec![5.0, 7.0, 10.0]);
    }

    #[test]
    fn percentiles_empty_input_returns_zeros() {
        assert_eq!(percentiles(&[]), (0.0, 0.0, 0.0, 0.0));
    }
}
