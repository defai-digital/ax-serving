//! Side-by-side A/B comparator.
//!
//! Runs the same three-class mixed-workload request set against two endpoints
//! sequentially, then prints a delta table showing which is faster for each
//! latency percentile.
//!
//! # Usage
//!
//! ```text
//! ax-serving-bench compare \
//!     --url-a http://127.0.0.1:18080 \
//!     --url-b http://127.0.0.1:18081 \
//!     --label-a llamacpp \
//!     --label-b native \
//!     --model default
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use reqwest::Client;
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;

// ── Config ─────────────────────────────────────────────────────────────────────

pub struct CompareConfig {
    pub url_a: String,
    pub url_b: String,
    pub label_a: String,
    pub label_b: String,
    pub concurrency: usize,
    pub total_requests: usize,
    pub model_id: String,
    pub max_tokens: usize,
    pub json: Option<PathBuf>,
}

// ── Prompt classes ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum PromptClass {
    Short,
    Medium,
    Long,
}

impl PromptClass {
    fn prompt(self) -> &'static str {
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

const CLASSES: [PromptClass; 3] = [PromptClass::Short, PromptClass::Medium, PromptClass::Long];

// ── Per-class stats ────────────────────────────────────────────────────────────

struct ClassLatencies {
    latencies_ms: Vec<u64>,
}

impl ClassLatencies {
    fn p50(&self) -> f64 {
        percentile(&self.latencies_ms, 50)
    }
    fn p95(&self) -> f64 {
        percentile(&self.latencies_ms, 95)
    }
    fn p99(&self) -> f64 {
        percentile(&self.latencies_ms, 99)
    }
}

fn percentile(v: &[u64], p: usize) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    s[(s.len() * p / 100).min(s.len() - 1)] as f64
}

// ── Single-endpoint result ─────────────────────────────────────────────────────

struct EndpointResult {
    label: String,
    short_p50: f64,
    short_p95: f64,
    short_p99: f64,
    medium_p50: f64,
    medium_p95: f64,
    medium_p99: f64,
    long_p50: f64,
    long_p95: f64,
    long_p99: f64,
    overall_p50: f64,
    overall_p95: f64,
    overall_p99: f64,
    rps: f64,
    errors: u64,
}

/// Machine-readable comparison output.
#[derive(Serialize)]
pub struct CompareResults {
    pub label_a: String,
    pub label_b: String,
    pub a_short_p99: f64,
    pub b_short_p99: f64,
    pub a_medium_p99: f64,
    pub b_medium_p99: f64,
    pub a_long_p99: f64,
    pub b_long_p99: f64,
    pub a_overall_p99: f64,
    pub b_overall_p99: f64,
    pub winner: String,
}

// ── Entry point ────────────────────────────────────────────────────────────────

pub async fn run(cfg: CompareConfig) -> Result<()> {
    let client = Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?,
    );

    println!("\nComparing {} vs {}", cfg.label_a, cfg.label_b);
    println!("{}", "=".repeat(60));

    let result_a = run_endpoint(&cfg, &client, &cfg.url_a, &cfg.label_a).await?;
    let result_b = run_endpoint(&cfg, &client, &cfg.url_b, &cfg.label_b).await?;

    print_comparison(&result_a, &result_b);

    if let Some(path) = &cfg.json {
        let winner = if result_a.overall_p99 <= result_b.overall_p99 {
            cfg.label_a.clone()
        } else {
            cfg.label_b.clone()
        };
        let out = CompareResults {
            label_a: cfg.label_a.clone(),
            label_b: cfg.label_b.clone(),
            a_short_p99: result_a.short_p99,
            b_short_p99: result_b.short_p99,
            a_medium_p99: result_a.medium_p99,
            b_medium_p99: result_b.medium_p99,
            a_long_p99: result_a.long_p99,
            b_long_p99: result_b.long_p99,
            a_overall_p99: result_a.overall_p99,
            b_overall_p99: result_b.overall_p99,
            winner,
        };
        std::fs::write(path, serde_json::to_string_pretty(&out)?)?;
        eprintln!("Comparison results written to {}", path.display());
    }

    Ok(())
}

// ── Single-endpoint run ────────────────────────────────────────────────────────

async fn run_endpoint(
    cfg: &CompareConfig,
    client: &Arc<Client>,
    url: &str,
    label: &str,
) -> Result<EndpointResult> {
    println!("\nRunning against {label} ({url}) …");

    let sem = Arc::new(Semaphore::new(cfg.concurrency));
    let chat_url = format!("{url}/v1/chat/completions");

    let short_lats: Arc<std::sync::Mutex<Vec<u64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let medium_lats: Arc<std::sync::Mutex<Vec<u64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let long_lats: Arc<std::sync::Mutex<Vec<u64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let errors = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let mut handles = Vec::with_capacity(cfg.total_requests);

    for i in 0..cfg.total_requests {
        let class = CLASSES[i % 3];
        let permit = Arc::clone(&sem).acquire_owned().await?;

        let client = Arc::clone(client);
        let chat_url = chat_url.clone();
        let model_id = cfg.model_id.clone();
        let max_tokens = cfg.max_tokens;
        let short_lats = Arc::clone(&short_lats);
        let medium_lats = Arc::clone(&medium_lats);
        let long_lats = Arc::clone(&long_lats);
        let errors = Arc::clone(&errors);

        handles.push(tokio::spawn(async move {
            let _permit = permit;

            let body = json!({
                "model": model_id,
                "messages": [{"role": "user", "content": class.prompt()}],
                "max_tokens": max_tokens,
                "stream": false,
            });

            let t = Instant::now();
            let ok = match client.post(&chat_url).json(&body).send().await {
                Ok(r) => r.status().is_success(),
                Err(_) => false,
            };
            let elapsed = t.elapsed().as_millis() as u64;

            if ok {
                let bucket = match class {
                    PromptClass::Short => &short_lats,
                    PromptClass::Medium => &medium_lats,
                    PromptClass::Long => &long_lats,
                };
                bucket.lock().unwrap().push(elapsed);
            } else {
                errors.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    for h in handles {
        let _ = h.await;
    }

    let total_ms = start.elapsed().as_millis() as u64;
    let sv = short_lats.lock().unwrap().clone();
    let mv = medium_lats.lock().unwrap().clone();
    let lv = long_lats.lock().unwrap().clone();
    let err = errors.load(Ordering::Relaxed);

    let classes = [
        ClassLatencies { latencies_ms: sv },
        ClassLatencies { latencies_ms: mv },
        ClassLatencies { latencies_ms: lv },
    ];

    let all: Vec<u64> = classes
        .iter()
        .flat_map(|c| c.latencies_ms.iter().copied())
        .collect();
    let total_success: usize = classes.iter().map(|c| c.latencies_ms.len()).sum();
    let rps = if total_ms > 0 {
        total_success as f64 / (total_ms as f64 / 1_000.0)
    } else {
        0.0
    };

    Ok(EndpointResult {
        label: label.to_string(),
        short_p50: classes[0].p50(),
        short_p95: classes[0].p95(),
        short_p99: classes[0].p99(),
        medium_p50: classes[1].p50(),
        medium_p95: classes[1].p95(),
        medium_p99: classes[1].p99(),
        long_p50: classes[2].p50(),
        long_p95: classes[2].p95(),
        long_p99: classes[2].p99(),
        overall_p50: percentile(&all, 50),
        overall_p95: percentile(&all, 95),
        overall_p99: percentile(&all, 99),
        rps,
        errors: err,
    })
}

// ── Side-by-side delta table ───────────────────────────────────────────────────

fn delta_str(a: f64, b: f64) -> String {
    if a == 0.0 && b == 0.0 {
        return "  N/A".to_string();
    }
    let pct = (b - a) / a * 100.0;
    format!("{pct:+.0}%")
}

fn winner_label(a: f64, b: f64) -> &'static str {
    if a < b {
        "A"
    } else if b < a {
        "B"
    } else {
        "tie"
    }
}

fn print_comparison(a: &EndpointResult, b: &EndpointResult) {
    println!("\nSide-by-side delta (B vs A)");
    println!("{}", "=".repeat(70));
    println!(
        "{:<22}  {:>10}  {:>10}  {:>8}  Winner",
        "Metric",
        &format!("A ({})", a.label),
        &format!("B ({})", b.label),
        "Delta"
    );
    println!("{}", "-".repeat(70));

    let rows: &[(&str, f64, f64)] = &[
        ("short  P50 (ms)", a.short_p50, b.short_p50),
        ("short  P95 (ms)", a.short_p95, b.short_p95),
        ("short  P99 (ms)", a.short_p99, b.short_p99),
        ("medium P50 (ms)", a.medium_p50, b.medium_p50),
        ("medium P95 (ms)", a.medium_p95, b.medium_p95),
        ("medium P99 (ms)", a.medium_p99, b.medium_p99),
        ("long   P50 (ms)", a.long_p50, b.long_p50),
        ("long   P95 (ms)", a.long_p95, b.long_p95),
        ("long   P99 (ms)", a.long_p99, b.long_p99),
        ("overall P50 (ms)", a.overall_p50, b.overall_p50),
        ("overall P95 (ms)", a.overall_p95, b.overall_p95),
        ("overall P99 (ms)", a.overall_p99, b.overall_p99),
    ];

    for (name, va, vb) in rows {
        let w = winner_label(*va, *vb);
        println!(
            "{:<22}  {:>10.1}  {:>10.1}  {:>8}  {}",
            name,
            va,
            vb,
            delta_str(*va, *vb),
            w,
        );
    }

    println!("{}", "-".repeat(70));
    println!("{:<22}  {:>10.1}  {:>10.1}", "RPS", a.rps, b.rps);
    println!("{:<22}  {:>10}  {:>10}", "Errors", a.errors, b.errors);

    let overall_winner = if a.overall_p99 <= b.overall_p99 {
        &a.label
    } else {
        &b.label
    };
    println!("\nOverall winner (P99): {overall_winner}");
}
