use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;

use crate::bench_common::percentile_f64;

const PROMPT_LENGTHS: [usize; 4] = [39, 209, 509, 1024];
const SHARED_REPLAY_PREFIX: &str = "Shared enterprise support transcript prefix: customer reports intermittent request spikes after deployment, repeated retries, and degraded p95 latency. Summarize root-cause patterns and mitigation actions. ";

pub struct ServicePerfConfig {
    pub url: String,
    pub model: String,
    pub concurrency: usize,
    pub requests: usize,
    pub max_tokens: usize,
    pub replay_share_pct: usize,
    pub burst_window_ms: u64,
    pub json: Option<PathBuf>,
    pub require_cache_hits: bool,
    pub min_exact_cache_hit_rate: Option<f64>,
    pub max_queue_wait_p95_ms: Option<f64>,
    pub require_generation_batching: bool,
    pub min_generation_batch_request_share: Option<f64>,
    pub require_llamacpp_prompt_cache_hint: bool,
    pub require_warm_pool: bool,
    pub max_warm_pool_evictions: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ServicePerfResults {
    pub total_requests: usize,
    pub success: u64,
    pub errors: u64,
    pub replay_requests: usize,
    pub burst_window_ms: u64,
    pub throughput_rps: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub queue_wait_p50_ms: f64,
    pub queue_wait_p95_ms: f64,
    pub queue_wait_p99_ms: f64,
    pub cache_hits_delta: i64,
    pub cache_misses_delta: i64,
    pub exact_cache_hit_rate: Option<f64>,
    pub cache_mode: String,
    pub llamacpp_prompt_cache_enabled: bool,
    pub warm_pool_enabled: bool,
    pub warm_pool_max_models: Option<u64>,
    pub warm_pool_loaded_models: u64,
    pub warm_pool_evictions_delta: i64,
    pub backend_generate_batch_support: bool,
    pub generation_batching_enabled: bool,
    pub generation_batches_delta: i64,
    pub generation_batch_requests_delta: i64,
    pub generation_batch_request_share: Option<f64>,
    pub generation_batch_largest_requests: u64,
}

#[derive(Clone)]
struct RunSnapshot {
    cache_hits: u64,
    cache_misses: u64,
    cache_mode: String,
    llamacpp_prompt_cache_enabled: bool,
    warm_pool_enabled: bool,
    warm_pool_max_models: Option<u64>,
    warm_pool_loaded_models: u64,
    warm_pool_evictions_total: u64,
    backend_generate_batch_support: bool,
    generation_batching_enabled: bool,
    generation_batches: u64,
    generation_batch_requests: u64,
    generation_batch_largest_requests: u64,
}

pub async fn run(cfg: ServicePerfConfig) -> Result<()> {
    let client = Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("failed to build reqwest client")?,
    );

    let baseline = match fetch_snapshot(&client, &cfg.url).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("WARNING: baseline metrics fetch failed ({e}); results may be incomplete");
            RunSnapshot {
            cache_hits: 0,
            cache_misses: 0,
            cache_mode: "unknown".into(),
            llamacpp_prompt_cache_enabled: false,
            warm_pool_enabled: false,
            warm_pool_max_models: None,
            warm_pool_loaded_models: 0,
            warm_pool_evictions_total: 0,
            backend_generate_batch_support: false,
            generation_batching_enabled: false,
            generation_batches: 0,
            generation_batch_requests: 0,
            generation_batch_largest_requests: 0,
        }
        }
    };

    let sem = Arc::new(Semaphore::new(cfg.concurrency.max(1)));
    let latency_ms: Arc<tokio::sync::Mutex<Vec<f64>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let queue_wait_ms: Arc<tokio::sync::Mutex<Vec<f64>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let success = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    let endpoint = format!("{}/v1/chat/completions", cfg.url.trim_end_matches('/'));
    let replay_requests =
        ((cfg.requests as f64) * (cfg.replay_share_pct as f64 / 100.0)).round() as usize;
    let burst_window_ms = cfg.burst_window_ms;
    let mut handles = Vec::with_capacity(cfg.requests);

    for i in 0..cfg.requests {
        let permit = Arc::clone(&sem).acquire_owned().await?;
        // BUG-078: capture start AFTER acquiring the permit so the stagger
        // offset is computed relative to permit acquisition, not the loop start.
        // Without this, waiting for a permit pushes deadlines into the past
        // and the entire burst fires simultaneously.
        let start = Instant::now();
        let client = Arc::clone(&client);
        let endpoint = endpoint.clone();
        let model = cfg.model.clone();
        let latency_ms = Arc::clone(&latency_ms);
        let queue_wait_ms = Arc::clone(&queue_wait_ms);
        let success = Arc::clone(&success);
        let errors = Arc::clone(&errors);
        let max_tokens = cfg.max_tokens;
        let prompt = build_prompt(
            PROMPT_LENGTHS[i % PROMPT_LENGTHS.len()],
            i < replay_requests,
            i,
        );
        let launch_offset_ms = if cfg.requests > 1 {
            ((i as u64) * burst_window_ms) / ((cfg.requests - 1) as u64)
        } else {
            0
        };

        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let deadline = start + Duration::from_millis(launch_offset_ms);
            tokio::time::sleep_until(deadline.into()).await;

            let body = json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": false,
                "cache": "enable",
            });

            let req_start = Instant::now();
            match client.post(&endpoint).json(&body).send().await {
                Ok(resp) => {
                    let elapsed = req_start.elapsed().as_secs_f64() * 1000.0;
                    let queue_wait = parse_queue_wait_ms(resp.headers().get("x-ax-stage-timing"));
                    let ok = resp.status().is_success();
                    let _ = resp.bytes().await;
                    if ok {
                        success.fetch_add(1, Ordering::Relaxed);
                        latency_ms.lock().await.push(elapsed);
                        if let Some(wait_ms) = queue_wait {
                            queue_wait_ms.lock().await.push(wait_ms);
                        }
                    } else {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        if let Err(e) = handle.await
            && e.is_panic()
        {
            tracing::warn!("bench task panicked: {e}");
        }
    }

    let total_duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let after = fetch_snapshot(&client, &cfg.url)
        .await
        .unwrap_or_else(|_| baseline.clone());

    let mut latencies = latency_ms.lock().await.clone();
    let mut queue_waits = queue_wait_ms.lock().await.clone();
    sanitize(&mut latencies);
    sanitize(&mut queue_waits);

    let success_count = success.load(Ordering::Relaxed);
    let error_count = errors.load(Ordering::Relaxed);
    let cache_hits_delta = after.cache_hits as i64 - baseline.cache_hits as i64;
    let cache_misses_delta = after.cache_misses as i64 - baseline.cache_misses as i64;
    let generation_batches_delta = after.generation_batches as i64 - baseline.generation_batches as i64;
    let generation_batch_requests_delta =
        after.generation_batch_requests as i64 - baseline.generation_batch_requests as i64;
    let warm_pool_evictions_delta =
        after.warm_pool_evictions_total as i64 - baseline.warm_pool_evictions_total as i64;
    let total_cache = cache_hits_delta + cache_misses_delta;

    let results = ServicePerfResults {
        total_requests: cfg.requests,
        success: success_count,
        errors: error_count,
        replay_requests,
        burst_window_ms,
        throughput_rps: if total_duration_ms > 0.0 {
            (success_count + error_count) as f64 / (total_duration_ms / 1000.0)
        } else {
            0.0
        },
        latency_p50_ms: percentile_f64(&latencies, 50),
        latency_p95_ms: percentile_f64(&latencies, 95),
        latency_p99_ms: percentile_f64(&latencies, 99),
        queue_wait_p50_ms: percentile_f64(&queue_waits, 50),
        queue_wait_p95_ms: percentile_f64(&queue_waits, 95),
        queue_wait_p99_ms: percentile_f64(&queue_waits, 99),
        cache_hits_delta,
        cache_misses_delta,
        exact_cache_hit_rate: if total_cache > 0 {
            Some(cache_hits_delta as f64 / total_cache as f64)
        } else {
            None
        },
        cache_mode: after.cache_mode.clone(),
        llamacpp_prompt_cache_enabled: after.llamacpp_prompt_cache_enabled,
        warm_pool_enabled: after.warm_pool_enabled,
        warm_pool_max_models: after.warm_pool_max_models,
        warm_pool_loaded_models: after.warm_pool_loaded_models,
        warm_pool_evictions_delta,
        backend_generate_batch_support: after.backend_generate_batch_support,
        generation_batching_enabled: after.generation_batching_enabled,
        generation_batches_delta,
        generation_batch_requests_delta,
        generation_batch_request_share: if success_count > 0 && generation_batch_requests_delta > 0 {
            Some(generation_batch_requests_delta as f64 / success_count as f64)
        } else {
            None
        },
        generation_batch_largest_requests: after.generation_batch_largest_requests,
    };

    print_report(&cfg, &results);

    validate_expectations(&cfg, &results)?;

    if let Some(path) = &cfg.json {
        let body = serde_json::to_string_pretty(&results)?;
        std::fs::write(path, body)?;
        println!("json: {}", path.display());
    }

    Ok(())
}

fn build_prompt(target_tokens: usize, replay_shared: bool, request_index: usize) -> String {
    let mut prompt = if replay_shared {
        SHARED_REPLAY_PREFIX.to_string()
    } else {
        format!(
            "Unique request {}: analyze scheduler latency, queueing behavior, and recovery strategy. ",
            request_index
        )
    };

    let filler = format!(
        "token{:04} workload analysis concurrency cache thermal mitigation dispatch ",
        request_index % 10_000
    );
    while approx_tokens(&prompt) < target_tokens {
        prompt.push_str(&filler);
    }
    prompt
}

fn approx_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn parse_queue_wait_ms(value: Option<&reqwest::header::HeaderValue>) -> Option<f64> {
    let value = value?.to_str().ok()?;
    let wait_us = value
        .split(',')
        .find_map(|part| part.trim().strip_prefix("queue_wait_us="))?
        .parse::<f64>()
        .ok()?;
    Some(wait_us / 1000.0)
}

async fn fetch_snapshot(client: &Client, base_url: &str) -> Result<RunSnapshot> {
    let url = format!("{}/v1/metrics", base_url.trim_end_matches('/'));
    let value: serde_json::Value = client
        .get(&url)
        .send()
        .await
        .context("failed to fetch /v1/metrics")?
        .error_for_status()
        .context("metrics scrape returned non-success status")?
        .json()
        .await
        .context("failed to decode metrics json")?;
    Ok(snapshot_from_metrics_value(&value))
}

fn snapshot_from_metrics_value(value: &serde_json::Value) -> RunSnapshot {
    RunSnapshot {
        cache_hits: value["cache"]["hits"].as_u64().unwrap_or(0),
        cache_misses: value["cache"]["misses"].as_u64().unwrap_or(0),
        cache_mode: value["cache"]["mode"].as_str().unwrap_or("unknown").into(),
        llamacpp_prompt_cache_enabled: value["cache"]["kv_prefix_cache"]
            ["llama_cpp_cache_prompt_enabled"]
            .as_bool()
            .unwrap_or(false),
        warm_pool_enabled: value["cache"]["warm_pool"]["enabled"]
            .as_bool()
            .unwrap_or(false),
        warm_pool_max_models: value["cache"]["warm_pool"]["max_models"].as_u64(),
        warm_pool_loaded_models: value["cache"]["warm_pool"]["loaded_models"]
            .as_u64()
            .unwrap_or(0),
        warm_pool_evictions_total: value["cache"]["warm_pool"]["evictions_total"]
            .as_u64()
            .unwrap_or(0),
        backend_generate_batch_support: value["scheduler"]["backend_generate_batch_support"]
            .as_bool()
            .unwrap_or(false),
        generation_batching_enabled: value["scheduler"]["generation_batching"]["enabled"]
            .as_bool()
            .unwrap_or(false),
        generation_batches: value["scheduler"]["generation_batching"]["executed_batches"]
            .as_u64()
            .unwrap_or(0),
        generation_batch_requests: value["scheduler"]["generation_batching"]["executed_requests"]
            .as_u64()
            .unwrap_or(0),
        generation_batch_largest_requests: value["scheduler"]["generation_batching"]
            ["largest_batch_requests"]
            .as_u64()
            .unwrap_or(0),
    }
}

fn sanitize(values: &mut Vec<f64>) {
    values.retain(|v| v.is_finite() && *v >= 0.0);
    values.sort_by(|a, b| a.total_cmp(b));
}


fn print_report(cfg: &ServicePerfConfig, results: &ServicePerfResults) {
    println!();
    println!("# Service-Perf Benchmark");
    println!();
    println!("URL: {}", cfg.url);
    println!("Model: {}", cfg.model);
    println!(
        "Concurrency: {} | Requests: {} | Replay: {}% ({} req) | Burst window: {} ms",
        cfg.concurrency,
        cfg.requests,
        cfg.replay_share_pct,
        results.replay_requests,
        cfg.burst_window_ms
    );
    println!(
        "Success: {} | Errors: {} | Throughput: {:.1} req/s",
        results.success, results.errors, results.throughput_rps
    );
    println!(
        "Latency ms: p50={:.1} p95={:.1} p99={:.1}",
        results.latency_p50_ms, results.latency_p95_ms, results.latency_p99_ms
    );
    println!(
        "Queue wait ms: p50={:.1} p95={:.1} p99={:.1}",
        results.queue_wait_p50_ms, results.queue_wait_p95_ms, results.queue_wait_p99_ms
    );
    println!(
        "Cache delta: hits={} misses={} hit_rate={}",
        results.cache_hits_delta,
        results.cache_misses_delta,
        results
            .exact_cache_hit_rate
            .map(|rate| format!("{:.1}%", rate * 100.0))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "Cache runtime: mode={} llamacpp_prompt_cache={} warm_pool_enabled={} warm_pool_max_models={} warm_pool_loaded_models={} warm_pool_evictions={}",
        results.cache_mode,
        results.llamacpp_prompt_cache_enabled,
        results.warm_pool_enabled,
        results
            .warm_pool_max_models
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        results.warm_pool_loaded_models,
        results.warm_pool_evictions_delta
    );
    println!(
        "Generation batching: backend_support={} enabled={} batches={} batched_requests={} share={} largest_batch={}",
        results.backend_generate_batch_support,
        results.generation_batching_enabled,
        results.generation_batches_delta,
        results.generation_batch_requests_delta,
        results
            .generation_batch_request_share
            .map(|share| format!("{:.1}%", share * 100.0))
            .unwrap_or_else(|| "n/a".to_string()),
        results.generation_batch_largest_requests
    );
    println!();
}

fn validate_expectations(cfg: &ServicePerfConfig, results: &ServicePerfResults) -> Result<()> {
    if cfg.require_cache_hits {
        anyhow::ensure!(
            results.cache_hits_delta > 0,
            "cache gate failed: no exact cache hits observed during the run"
        );
    }

    if let Some(min_rate) = cfg.min_exact_cache_hit_rate {
        anyhow::ensure!(
            (0.0..=1.0).contains(&min_rate),
            "min_exact_cache_hit_rate must be in [0.0, 1.0]"
        );
        let observed_rate = results.exact_cache_hit_rate.unwrap_or(0.0);
        anyhow::ensure!(
            observed_rate >= min_rate,
            "cache hit-rate gate failed: observed {:.1}% < required {:.1}%",
            observed_rate * 100.0,
            min_rate * 100.0
        );
    }

    if let Some(max_p95) = cfg.max_queue_wait_p95_ms {
        anyhow::ensure!(
            max_p95 >= 0.0,
            "max_queue_wait_p95_ms must be >= 0.0"
        );
        anyhow::ensure!(
            results.queue_wait_p95_ms <= max_p95,
            "queue wait gate failed: observed p95 {:.1} ms > allowed {:.1} ms",
            results.queue_wait_p95_ms,
            max_p95
        );
    }

    if cfg.require_llamacpp_prompt_cache_hint {
        anyhow::ensure!(
            results.llamacpp_prompt_cache_enabled,
            "prompt-cache gate failed: llama.cpp prompt-cache hint is not enabled on the server"
        );
    }

    if cfg.require_warm_pool {
        anyhow::ensure!(
            results.warm_pool_enabled,
            "warm-pool gate failed: warm pool is not enabled on the server"
        );
    }

    if let Some(max_evictions) = cfg.max_warm_pool_evictions {
        anyhow::ensure!(
            results.warm_pool_evictions_delta >= 0,
            "warm-pool eviction gate failed: observed negative eviction delta {}",
            results.warm_pool_evictions_delta
        );
        anyhow::ensure!(
            results.warm_pool_evictions_delta as u64 <= max_evictions,
            "warm-pool eviction gate failed: observed {} evictions > allowed {}",
            results.warm_pool_evictions_delta,
            max_evictions
        );
    }

    if cfg.require_generation_batching {
        anyhow::ensure!(
            results.backend_generate_batch_support,
            "generation batching gate failed: backend does not advertise generate_batch support"
        );
        anyhow::ensure!(
            results.generation_batching_enabled,
            "generation batching gate failed: scheduler generation batching is not enabled on the server"
        );
        anyhow::ensure!(
            results.generation_batches_delta > 0 && results.generation_batch_requests_delta > 0,
            "generation batching gate failed: no generation batches executed during the run"
        );
    }

    if let Some(min_share) = cfg.min_generation_batch_request_share {
        anyhow::ensure!(
            (0.0..=1.0).contains(&min_share),
            "min_generation_batch_request_share must be in [0.0, 1.0]"
        );
        anyhow::ensure!(
            results.backend_generate_batch_support,
            "generation batching share gate failed: backend does not advertise generate_batch support"
        );
        anyhow::ensure!(
            results.generation_batching_enabled,
            "generation batching share gate failed: scheduler generation batching is not enabled on the server"
        );
        let observed_share = results.generation_batch_request_share.unwrap_or(0.0);
        anyhow::ensure!(
            observed_share >= min_share,
            "generation batching share gate failed: observed {:.1}% < required {:.1}%",
            observed_share * 100.0,
            min_share * 100.0
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ServicePerfConfig, ServicePerfResults, approx_tokens, build_prompt,
        parse_queue_wait_ms, snapshot_from_metrics_value, validate_expectations,
    };
    use std::path::PathBuf;

    #[test]
    fn prompt_builder_reaches_target_size() {
        let prompt = build_prompt(128, true, 7);
        assert!(approx_tokens(&prompt) >= 128);
        assert!(prompt.contains("Shared enterprise support transcript prefix"));
    }

    #[test]
    fn queue_wait_header_parser_handles_expected_format() {
        let value = reqwest::header::HeaderValue::from_static("queue_wait_us=12345");
        assert_eq!(parse_queue_wait_ms(Some(&value)), Some(12.345));
    }

    #[test]
    fn snapshot_parser_extracts_generation_batching_fields() {
        let value = serde_json::json!({
            "cache": {
                "hits": 4,
                "misses": 7,
                "mode": "exact_response",
                "kv_prefix_cache": {
                    "llama_cpp_cache_prompt_enabled": true
                },
                "warm_pool": {
                    "enabled": true,
                    "max_models": 3,
                    "loaded_models": 2,
                    "evictions_total": 9
                }
            },
            "scheduler": {
                "backend_generate_batch_support": true,
                "generation_batching": {
                    "enabled": true,
                    "executed_batches": 3,
                    "executed_requests": 5,
                    "largest_batch_requests": 2
                }
            }
        });

        let snapshot = snapshot_from_metrics_value(&value);
        assert_eq!(snapshot.cache_hits, 4);
        assert_eq!(snapshot.cache_misses, 7);
        assert_eq!(snapshot.cache_mode, "exact_response");
        assert!(snapshot.llamacpp_prompt_cache_enabled);
        assert!(snapshot.warm_pool_enabled);
        assert_eq!(snapshot.warm_pool_max_models, Some(3));
        assert_eq!(snapshot.warm_pool_loaded_models, 2);
        assert_eq!(snapshot.warm_pool_evictions_total, 9);
        assert!(snapshot.backend_generate_batch_support);
        assert!(snapshot.generation_batching_enabled);
        assert_eq!(snapshot.generation_batches, 3);
        assert_eq!(snapshot.generation_batch_requests, 5);
        assert_eq!(snapshot.generation_batch_largest_requests, 2);
    }

    fn base_config() -> ServicePerfConfig {
        ServicePerfConfig {
            url: "http://127.0.0.1:18080".into(),
            model: "default".into(),
            concurrency: 8,
            requests: 80,
            max_tokens: 128,
            replay_share_pct: 40,
            burst_window_ms: 5000,
            json: Option::<PathBuf>::None,
            require_cache_hits: false,
            min_exact_cache_hit_rate: None,
            max_queue_wait_p95_ms: None,
            require_generation_batching: false,
            min_generation_batch_request_share: None,
            require_llamacpp_prompt_cache_hint: false,
            require_warm_pool: false,
            max_warm_pool_evictions: None,
        }
    }

    fn base_results() -> ServicePerfResults {
        ServicePerfResults {
            total_requests: 80,
            success: 80,
            errors: 0,
            replay_requests: 32,
            burst_window_ms: 5000,
            throughput_rps: 10.0,
            latency_p50_ms: 100.0,
            latency_p95_ms: 200.0,
            latency_p99_ms: 300.0,
            queue_wait_p50_ms: 1.0,
            queue_wait_p95_ms: 2.0,
            queue_wait_p99_ms: 3.0,
            cache_hits_delta: 10,
            cache_misses_delta: 70,
            exact_cache_hit_rate: Some(0.125),
            cache_mode: "exact_response".into(),
            llamacpp_prompt_cache_enabled: true,
            warm_pool_enabled: true,
            warm_pool_max_models: Some(4),
            warm_pool_loaded_models: 1,
            warm_pool_evictions_delta: 0,
            backend_generate_batch_support: true,
            generation_batching_enabled: true,
            generation_batches_delta: 8,
            generation_batch_requests_delta: 32,
            generation_batch_request_share: Some(0.4),
            generation_batch_largest_requests: 4,
        }
    }

    #[test]
    fn generation_batching_gate_accepts_matching_results() {
        let mut cfg = base_config();
        cfg.require_generation_batching = true;
        cfg.min_generation_batch_request_share = Some(0.25);
        assert!(validate_expectations(&cfg, &base_results()).is_ok());
    }

    #[test]
    fn generation_batching_gate_rejects_missing_batch_execution() {
        let mut cfg = base_config();
        cfg.require_generation_batching = true;
        let mut results = base_results();
        results.generation_batches_delta = 0;
        results.generation_batch_requests_delta = 0;
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn generation_batching_share_gate_rejects_low_share() {
        let mut cfg = base_config();
        cfg.min_generation_batch_request_share = Some(0.5);
        let mut results = base_results();
        results.generation_batch_request_share = Some(0.25);
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn cache_gate_rejects_missing_hits() {
        let mut cfg = base_config();
        cfg.require_cache_hits = true;
        let mut results = base_results();
        results.cache_hits_delta = 0;
        results.exact_cache_hit_rate = Some(0.0);
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn cache_hit_rate_gate_rejects_low_rate() {
        let mut cfg = base_config();
        cfg.min_exact_cache_hit_rate = Some(0.3);
        let mut results = base_results();
        results.exact_cache_hit_rate = Some(0.1);
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn queue_wait_gate_rejects_high_p95() {
        let mut cfg = base_config();
        cfg.max_queue_wait_p95_ms = Some(50.0);
        let mut results = base_results();
        results.queue_wait_p95_ms = 75.0;
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn prompt_cache_gate_rejects_disabled_hint() {
        let mut cfg = base_config();
        cfg.require_llamacpp_prompt_cache_hint = true;
        let mut results = base_results();
        results.llamacpp_prompt_cache_enabled = false;
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn warm_pool_gate_rejects_disabled_pool() {
        let mut cfg = base_config();
        cfg.require_warm_pool = true;
        let mut results = base_results();
        results.warm_pool_enabled = false;
        assert!(validate_expectations(&cfg, &results).is_err());
    }

    #[test]
    fn warm_pool_eviction_gate_rejects_excess_evictions() {
        let mut cfg = base_config();
        cfg.max_warm_pool_evictions = Some(0);
        let mut results = base_results();
        results.warm_pool_evictions_delta = 2;
        assert!(validate_expectations(&cfg, &results).is_err());
    }
}
