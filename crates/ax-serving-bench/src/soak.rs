//! 24h stability soak test.
//!
//! Runs continuous inference and checks for RSS drift and P95 latency drift
//! at regular intervals. Fails if either drift exceeds the configured threshold.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use ax_serving_api::metrics::LatencyHistogram;
use ax_serving_engine::{
    GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, RouterBackend,
};
use tokio::sync::mpsc;

pub async fn run(
    model: PathBuf,
    duration_min: u64,
    check_interval_min: u64,
    max_rss_drift: f64,
    max_p95_drift: f64,
    json_out: Option<PathBuf>,
) -> Result<()> {
    // load_model may touch backend runtimes depending on routing. Since
    // soak::run() is async, run model load on a blocking thread.
    let (backend, handle, meta) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let backend = RouterBackend::from_env();
        let (handle, meta) = backend.load_model(&model, LoadConfig::default())?;
        Ok((backend, handle, meta))
    })
    .await??;

    let total = Duration::from_secs(duration_min * 60);
    let interval = Duration::from_secs(check_interval_min * 60);

    println!(
        "soak: {} (ctx={}), duration={}min, interval={}min",
        meta.architecture, meta.context_length, duration_min, check_interval_min
    );
    println!(
        "thresholds: rss_drift<{:.0}%, p95_drift<{:.0}%",
        max_rss_drift * 100.0,
        max_p95_drift * 100.0
    );

    let start = Instant::now();
    let baseline_rss = {
        let raw = ax_serving_api::metrics::current_rss_bytes() as f64;
        if raw > 0.0 { raw } else { 1.0 }
    };
    let mut baseline_p95: Option<Duration> = None;
    let mut measurements: Vec<serde_json::Value> = Vec::new();
    let mut last_check = start;

    let params = GenerationParams {
        temperature: None, // greedy
        top_p: None,
        top_k: Some(1),
        max_tokens: Some(64),
        stop_seqs: Vec::new(),
        seed: None,
        repeat_penalty: None,
        ..Default::default()
    };

    while start.elapsed() < total {
        // One generation burst: 39 prompt tokens, 64 decode tokens.
        let tokens: Vec<u32> = (1..=39).collect();
        let (tx, mut rx) = mpsc::channel::<GenerateEvent>(256);
        let t0 = Instant::now();

        backend.generate(handle, GenerateInput::Tokens(tokens), params.clone(), tx)?;

        let mut hist = LatencyHistogram::with_capacity(64);
        let mut last_tok_time = t0;
        let mut received_done = false;

        while let Some(event) = rx.recv().await {
            match event {
                GenerateEvent::Token(_) => {
                    let now = Instant::now();
                    hist.record(now.duration_since(last_tok_time));
                    last_tok_time = now;
                }
                GenerateEvent::Done(_) => {
                    received_done = true;
                    break;
                }
                GenerateEvent::Error(e) => anyhow::bail!("generation error: {e}"),
                GenerateEvent::ToolCall { .. } | GenerateEvent::TokenLogprob { .. } => {}
            }
        }

        if !received_done {
            anyhow::bail!("generation ended without Done event");
        }

        let p95 = hist.p95();
        if baseline_p95.is_none() {
            baseline_p95 = Some(p95);
        }

        // Check at interval.
        if last_check.elapsed() >= interval {
            last_check = Instant::now();
            let elapsed_min = start.elapsed().as_secs() / 60;
            let rss = ax_serving_api::metrics::current_rss_bytes() as f64;
            let rss_drift = (rss - baseline_rss) / baseline_rss;
            // Guard against zero baseline (empty first burst) — NaN comparisons
            // are always false in IEEE 754, which would silently mask drift.
            let baseline_secs = baseline_p95
                .map(|d| d.as_secs_f64())
                .filter(|secs| *secs > 0.0)
                .unwrap_or(1e-9);
            let p95_drift = (p95.as_secs_f64() - baseline_secs) / baseline_secs;
            let thermal = backend.thermal_state().as_str().to_string();

            println!(
                "t={}min  rss_drift={:+.1}%  p95={:.1}ms  p95_drift={:+.1}%  thermal={}",
                elapsed_min,
                rss_drift * 100.0,
                p95.as_secs_f64() * 1000.0,
                p95_drift * 100.0,
                thermal
            );

            measurements.push(serde_json::json!({
                "elapsed_min": elapsed_min,
                "rss_drift_pct": rss_drift * 100.0,
                "p95_ms": p95.as_secs_f64() * 1000.0,
                "p95_drift_pct": p95_drift * 100.0,
                "thermal": thermal,
            }));

            if rss_drift > max_rss_drift {
                anyhow::bail!(
                    "RSS drift {:.1}% exceeds threshold {:.0}%",
                    rss_drift * 100.0,
                    max_rss_drift * 100.0
                );
            }
            // Only flag P95 drift when thermal is not degraded.
            if matches!(
                backend.thermal_state(),
                ax_serving_engine::ThermalState::Nominal | ax_serving_engine::ThermalState::Fair
            ) && p95_drift > max_p95_drift
            {
                anyhow::bail!(
                    "P95 latency drift {:.1}% exceeds threshold {:.0}%",
                    p95_drift * 100.0,
                    max_p95_drift * 100.0
                );
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    println!("soak PASSED after {}min", duration_min);

    if let Some(path) = json_out {
        let out = serde_json::json!({ "status": "passed", "measurements": measurements });
        std::fs::write(&path, serde_json::to_string_pretty(&out)?)?;
    }

    Ok(())
}
