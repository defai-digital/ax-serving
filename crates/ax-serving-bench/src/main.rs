//! ax-serving-bench: Benchmarking for ax-serving.
//!
//! Subcommands:
//!   bench             — Throughput measurement (prefill + decode tok/s)
//!   soak              — Long-run stability (24h, RSS drift < 5%, P95 drift < 5%)
//!   profile           — Per-operation timing breakdown
//!   cache-bench       — Prefix-cache speedup validation (cold vs warm latency)
//!   mixed             — Mixed-workload (short/medium/long prompts) P50/P95/P99 benchmark
//!   compare           — Side-by-side A/B comparison of two endpoints
//!   regression-check  — Compare a results JSON against a baseline

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-bench only supports aarch64-apple-darwin (Apple Silicon M3+)");

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
pub mod bench_common;
mod cache_bench;
mod compare;
mod mixed;
mod multi_worker;
mod perf;
mod regression;
mod soak;

#[derive(Parser, Debug)]
#[command(name = "ax-serving-bench", about = "AX Serving benchmarks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Measure throughput: prefill tok/s and decode tok/s.
    Bench {
        #[arg(short = 'm', long)]
        model: PathBuf,
        /// Prompt lengths to test (tokens).
        #[arg(long, value_delimiter = ',', default_value = "39,209,509,1024")]
        prompt_lengths: Vec<usize>,
        /// Number of decode tokens per run.
        #[arg(long, default_value = "128")]
        decode_tokens: usize,
        /// Warmup iterations.
        #[arg(long, default_value = "2")]
        warmup: usize,
        /// Measurement iterations.
        #[arg(long, default_value = "5")]
        iters: usize,
        /// Output JSON to file.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Long-run stability test (default: 24h).
    Soak {
        #[arg(short = 'm', long)]
        model: PathBuf,
        /// Total duration in minutes.
        #[arg(long, default_value = "1440")]
        duration_min: u64,
        /// Interval between drift checks in minutes.
        #[arg(long, default_value = "10")]
        check_interval_min: u64,
        /// RSS drift threshold (fraction, e.g. 0.05 = 5%).
        #[arg(long, default_value = "0.05")]
        max_rss_drift: f64,
        /// P95 latency drift threshold.
        #[arg(long, default_value = "0.05")]
        max_p95_drift: f64,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Per-operation timing breakdown.
    Profile {
        #[arg(short = 'm', long)]
        model: PathBuf,
        /// Prompt length.
        #[arg(long, default_value = "128")]
        prompt_length: usize,
        /// Warmup tokens.
        #[arg(long, default_value = "16")]
        warmup: usize,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Prefix-cache speedup validation against a live ax-serving instance.
    ///
    /// Sends the same prompt N times (first = cold, rest = warm) and reports
    /// the speedup from the exact-match `ResponseCache` (Valkey backend).
    ///
    /// Requires `AXS_CACHE_ENABLED=true` and a running Valkey/Redis instance.
    CacheBench {
        /// Base URL of the running ax-serving REST server.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Model ID to include in each request body.
        #[arg(long, default_value = "default")]
        model: String,
        /// Prompt to send (same on every request to exercise the cache).
        #[arg(long)]
        prompt: Option<String>,
        /// Total number of requests (1 cold + N-1 warm). Minimum 2.
        #[arg(long, default_value = "5")]
        requests: usize,
        /// `max_tokens` for each request.
        #[arg(long, default_value = "128")]
        max_tokens: usize,
    },
    /// Mixed-workload latency benchmark against a live ax-serving instance.
    ///
    /// Issues short (32 tok), medium (256 tok), and long (512 tok) prompts in
    /// equal proportion under configurable concurrency.  Reports P50/P95/P99
    /// per class and overall, plus a pass/fail gate on the overall P99.
    Mixed {
        /// Base URL of the running ax-serving REST server.
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Maximum concurrent in-flight requests (ignored when --concurrency-levels is set).
        #[arg(long, default_value = "4")]
        concurrency: usize,
        /// Run the full suite at each listed concurrency level (e.g. `1,2,4,8`).
        #[arg(long, value_delimiter = ',')]
        concurrency_levels: Vec<usize>,
        /// Total number of requests (evenly split across short/medium/long).
        #[arg(long, default_value = "60")]
        requests: usize,
        /// Model ID to include in each request body.
        #[arg(long, default_value = "default")]
        model: String,
        /// `max_tokens` per request.
        #[arg(long, default_value = "128")]
        max_tokens: usize,
        /// Pass/fail gate: overall P99 must be below this threshold (ms).
        #[arg(long, default_value = "10000")]
        target_p99_ms: u64,
        /// Write worst-case P99 results JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Side-by-side A/B comparison of two serving endpoints.
    ///
    /// Runs the same mixed-workload request set against endpoint A and B
    /// sequentially, then prints a delta table.
    Compare {
        /// Base URL of endpoint A.
        #[arg(long)]
        url_a: String,
        /// Base URL of endpoint B.
        #[arg(long)]
        url_b: String,
        /// Label for endpoint A.
        #[arg(long, default_value = "A")]
        label_a: String,
        /// Label for endpoint B.
        #[arg(long, default_value = "B")]
        label_b: String,
        /// Maximum concurrent in-flight requests.
        #[arg(long, default_value = "4")]
        concurrency: usize,
        /// Total requests per endpoint.
        #[arg(long, default_value = "60")]
        requests: usize,
        /// Model ID for each request.
        #[arg(long, default_value = "default")]
        model: String,
        /// `max_tokens` per request.
        #[arg(long, default_value = "128")]
        max_tokens: usize,
        /// Write comparison results JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Compare a benchmark results JSON against a stored baseline.
    ///
    /// Exits non-zero if any metric exceeds `baseline * (1 + tolerance_pct/100)`.
    /// Null baseline values are silently skipped.
    RegressionCheck {
        /// Path to the results JSON produced by `mixed --json`.
        #[arg(long)]
        results: PathBuf,
        /// Path to the baseline JSON (same format as results).
        #[arg(long)]
        baseline: PathBuf,
        /// Allowed regression tolerance (percent).
        #[arg(long, default_value = "10.0")]
        tolerance_pct: f64,
    },
    /// Concurrent multi-worker load benchmark against a live orchestrator.
    ///
    /// Sends requests with up to `--workers` concurrent in-flight at a time
    /// and reports throughput and latency percentiles.
    ///
    /// Use `--requests` for a fixed count or `--duration-secs` for a time-bounded run.
    /// A markdown report is automatically written to `target/bench-reports/`.
    MultiWorker {
        /// Orchestrator base URL (e.g. `http://127.0.0.1:18080`).
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Number of concurrent workers (in-flight requests).
        #[arg(long, default_value = "16")]
        workers: usize,
        /// Total number of requests to issue (ignored when --duration-secs is set).
        #[arg(long, default_value = "100")]
        requests: usize,
        /// Run for this many seconds instead of a fixed request count.
        #[arg(long)]
        duration_secs: Option<u64>,
        /// Dispatcher mode label: direct (default) or nats.
        #[arg(long, default_value = "direct")]
        mode: String,
        /// Model ID to include in each request body.
        #[arg(long, default_value = "default")]
        model: String,
        /// Prompt text to send (defaults to a short synthetic prompt).
        #[arg(long)]
        prompt: Option<String>,
        /// `max_tokens` for each request.
        #[arg(long, default_value = "128")]
        decode_tokens: usize,
        /// Write results to a JSON file.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Skip writing the markdown report to `target/bench-reports/`.
        #[arg(long)]
        no_report: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("AXS_LOG")
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        // bench::run is sync — the selected backend owns its execution path, and drain uses a
        // small separate current_thread runtime. No #[tokio::main] needed.
        Command::Bench {
            model,
            prompt_lengths,
            decode_tokens,
            warmup,
            iters,
            json,
        } => bench::run(model, prompt_lengths, decode_tokens, warmup, iters, json),
        // Soak and Profile are inherently long-running async; give them their own runtime.
        Command::Soak {
            model,
            duration_min,
            check_interval_min,
            max_rss_drift,
            max_p95_drift,
            json,
        } => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(soak::run(
                model,
                duration_min,
                check_interval_min,
                max_rss_drift,
                max_p95_drift,
                json,
            )),
        Command::Profile {
            model,
            prompt_length,
            warmup,
            json,
        } => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(perf::run(model, prompt_length, warmup, json)),
        Command::CacheBench {
            url,
            model,
            prompt,
            requests,
            max_tokens,
        } => {
            let cfg = cache_bench::CacheBenchConfig {
                url,
                model,
                prompt: prompt.unwrap_or_else(|| {
                    "Explain the difference between supervised and unsupervised learning.".into()
                }),
                n_requests: requests,
                max_tokens,
            };
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(cache_bench::run(cfg))
        }
        Command::Mixed {
            url,
            concurrency,
            concurrency_levels,
            requests,
            model,
            max_tokens,
            target_p99_ms,
            json,
        } => {
            let cfg = mixed::MixedConfig {
                url,
                concurrency,
                concurrency_levels,
                total_requests: requests,
                model_id: model,
                max_tokens,
                target_p99_ms,
                json,
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(mixed::run(cfg))
        }
        Command::Compare {
            url_a,
            url_b,
            label_a,
            label_b,
            concurrency,
            requests,
            model,
            max_tokens,
            json,
        } => {
            let cfg = compare::CompareConfig {
                url_a,
                url_b,
                label_a,
                label_b,
                concurrency,
                total_requests: requests,
                model_id: model,
                max_tokens,
                json,
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(compare::run(cfg))
        }
        Command::RegressionCheck {
            results,
            baseline,
            tolerance_pct,
        } => regression::check(results, baseline, tolerance_pct),
        Command::MultiWorker {
            url,
            workers,
            requests,
            duration_secs,
            mode,
            model,
            prompt,
            decode_tokens,
            json,
            no_report,
        } => {
            let cfg = multi_worker::MultiWorkerConfig {
                url,
                concurrency: workers,
                // When duration_secs is set, disable the request-count cap so
                // the run terminates on the wall-clock deadline only (matching
                // the CLI help: "ignored when --duration-secs is set").
                total_requests: if duration_secs.is_some() {
                    usize::MAX
                } else {
                    requests
                },
                duration_secs,
                mode,
                model_id: model,
                prompt: prompt.unwrap_or_else(|| {
                    "Summarize the key differences between LLaMA 3 and Mistral 7B in one sentence."
                        .into()
                }),
                decode_tokens,
                json,
                write_report: !no_report,
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(multi_worker::run(cfg))
        }
    }
}
