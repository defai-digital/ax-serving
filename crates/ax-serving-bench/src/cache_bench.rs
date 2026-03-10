//! Cache-bench: prefix-cache hit validation against a live ax-serving instance.
//!
//! Sends the same prompt N times to a running server. The first request is
//! the "cold" path (cache miss + full inference); subsequent requests should
//! hit the exact-match `ResponseCache` and return near-instantly.
//!
//! Reports cold latency, warm (cached) latency, and speedup ratio.

use anyhow::Result;
use reqwest::Client;

pub struct CacheBenchConfig {
    /// Base URL of the running ax-serving REST server.
    pub url: String,
    /// Model ID to target.
    pub model: String,
    /// Prompt to send on every request.
    pub prompt: String,
    /// Total number of requests to send (first = cold, rest = warm).
    pub n_requests: usize,
    /// `max_tokens` for each request.
    pub max_tokens: usize,
}

pub async fn run(cfg: CacheBenchConfig) -> Result<()> {
    if cfg.n_requests < 2 {
        anyhow::bail!("--requests must be >= 2 (1 cold + at least 1 warm)");
    }

    let client = Client::new();
    let endpoint = format!("{}/v1/chat/completions", cfg.url.trim_end_matches('/'));

    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [{"role": "user", "content": cfg.prompt}],
        "max_tokens": cfg.max_tokens,
        "stream": false,
        "cache": "enable",
    });

    println!("cache-bench: {} requests → {endpoint}", cfg.n_requests,);
    println!("  model   : {}", cfg.model);
    println!("  prompt  : {:?}", &cfg.prompt[..cfg.prompt.len().min(72)]);
    println!("  tokens  : {}", cfg.max_tokens);
    println!();

    let mut latencies_ms: Vec<u128> = Vec::with_capacity(cfg.n_requests);

    for i in 0..cfg.n_requests {
        let start = std::time::Instant::now();
        let resp = client
            .post(&endpoint)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let elapsed = start.elapsed().as_millis();
        let _body: serde_json::Value = resp.json().await?;
        latencies_ms.push(elapsed);

        let tag = if i == 0 { "cold" } else { "warm" };
        println!("  request {:>3}: {:>6}ms  ({})", i + 1, elapsed, tag);
    }

    println!();

    let cold_ms = latencies_ms[0];
    let warm_slice = &latencies_ms[1..];
    let warm_avg_ms = warm_slice.iter().sum::<u128>() / warm_slice.len() as u128;
    let warm_min_ms = warm_slice.iter().copied().min().unwrap_or(0);

    let speedup = if warm_avg_ms > 0 {
        cold_ms as f64 / warm_avg_ms as f64
    } else {
        f64::INFINITY
    };

    println!("Results:");
    println!("  cold latency : {:>6}ms", cold_ms);
    println!("  warm avg     : {:>6}ms", warm_avg_ms);
    println!("  warm min     : {:>6}ms", warm_min_ms);
    println!("  speedup      : {:.1}x", speedup);
    println!();

    if speedup < 2.0 {
        println!(
            "NOTE: speedup {:.1}x < 2.0x — cache may not be enabled on this server.",
            speedup
        );
        println!(
            "      Start the server with a Valkey backend: AXS_CACHE_ENABLED=true AXS_REDIS_URL=redis://127.0.0.1:6379"
        );
    } else {
        println!(
            "Cache is active ({:.1}x speedup on warm requests).",
            speedup
        );
    }

    Ok(())
}
