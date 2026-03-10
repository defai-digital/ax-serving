//! Per-operation timing profile: TTFT and decode inter-token latency.
//!
//! Runs `--warmup` throwaway generations then `MEASURE_ITERS` measured runs.
//!
//! Output columns (all milliseconds):
//!   `ttft`       — time from `generate()` call to the first Token event
//!                  (prefill latency + first decode token).
//!   `decode_itl` — inter-token latency for every subsequent decode token.
//!
//! ```text
//! metric               min_ms     p50_ms     p95_ms     p99_ms
//! ─────────────────────────────────────────────────────────────
//! ttft                  12.34      13.10      18.40      21.00
//! decode_itl             9.80      10.30      11.90      13.20
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use ax_serving_engine::{
    GenerateEvent, GenerateInput, GenerationParams, InferenceBackend, LoadConfig, ModelHandle,
    RouterBackend,
};
use tokio::sync::mpsc;

const MEASURE_ITERS: usize = 5;
const DECODE_TOKENS: usize = 64;

pub async fn run(
    model: PathBuf,
    prompt_length: usize,
    warmup: usize,
    json_out: Option<PathBuf>,
) -> Result<()> {
    // load_model calls block_on internally — must run on a non-tokio thread.
    let (backend, handle, meta) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let backend = RouterBackend::from_env();
        let (handle, meta) = backend.load_model(&model, LoadConfig::default())?;
        Ok((backend, handle, meta))
    })
    .await??;

    let prompt_tokens: Vec<u32> = (1..=(prompt_length as u32)).collect();
    let params = GenerationParams {
        temperature: None, // greedy
        top_p: None,
        top_k: Some(1),
        max_tokens: Some(DECODE_TOKENS),
        stop_seqs: Vec::new(),
        seed: None,
        repeat_penalty: None,
        ..Default::default()
    };

    println!("model: {} (ctx={})", meta.architecture, meta.context_length);
    println!(
        "prompt_length={prompt_length}  decode_tokens={DECODE_TOKENS}  warmup={warmup}  iters={MEASURE_ITERS}"
    );
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10}",
        "metric", "min_ms", "p50_ms", "p95_ms", "p99_ms"
    );
    println!("{}", "─".repeat(62));

    // Warmup — discard results.
    for _ in 0..warmup {
        profile_one_run(&backend, handle, &prompt_tokens, &params).await?;
    }

    // Measurement.
    let mut ttft_samples: Vec<Duration> = Vec::with_capacity(MEASURE_ITERS);
    let mut itl_samples: Vec<Duration> = Vec::with_capacity(MEASURE_ITERS * DECODE_TOKENS);
    for _ in 0..MEASURE_ITERS {
        let pr = profile_one_run(&backend, handle, &prompt_tokens, &params).await?;
        if let Some(t) = pr.ttft {
            ttft_samples.push(t);
        }
        itl_samples.extend_from_slice(&pr.inter_token_latencies);
    }

    print_row("ttft", &mut ttft_samples);
    print_row("decode_itl", &mut itl_samples);

    if let Some(path) = json_out {
        let out = serde_json::json!({
            "model": meta.architecture,
            "prompt_length": prompt_length,
            "decode_tokens": DECODE_TOKENS,
            "warmup": warmup,
            "iters": MEASURE_ITERS,
            "ttft_ms":       stat_map(&mut ttft_samples),
            "decode_itl_ms": stat_map(&mut itl_samples),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&out)?)?;
        println!("\nresults written to {}", path.display());
    }

    Ok(())
}

// ── Internal helpers ──────────────────────────────────────────────────────────

struct ProfileRun {
    /// Time from calling `generate()` to the first Token event.
    ttft: Option<Duration>,
    /// Wall-clock gap between each pair of consecutive Token events.
    inter_token_latencies: Vec<Duration>,
}

async fn profile_one_run(
    backend: &dyn InferenceBackend,
    handle: ModelHandle,
    tokens: &[u32],
    params: &GenerationParams,
) -> Result<ProfileRun> {
    let (tx, mut rx) = mpsc::channel::<GenerateEvent>(512);
    let t_start = Instant::now();

    // generate() spawns onto the backend's internal runtime and returns
    // immediately. Safe to call from an async context.
    backend.generate(
        handle,
        GenerateInput::Tokens(tokens.to_vec()),
        params.clone(),
        tx,
    )?;

    let mut ttft: Option<Duration> = None;
    let mut itl: Vec<Duration> = Vec::new();
    let mut last_tok = t_start;
    let mut generation_done = false;

    while let Some(event) = rx.recv().await {
        match event {
            GenerateEvent::Token(_) => {
                let now = Instant::now();
                if ttft.is_none() {
                    ttft = Some(now.duration_since(t_start));
                } else {
                    itl.push(now.duration_since(last_tok));
                }
                last_tok = now;
            }
            GenerateEvent::Done(_) => {
                generation_done = true;
                break;
            }
            GenerateEvent::Error(e) => anyhow::bail!("generation error: {e}"),
            GenerateEvent::ToolCall { .. } | GenerateEvent::TokenLogprob { .. } => {}
        }
    }

    if !generation_done {
        anyhow::bail!("generation channel closed without Done — backend may have crashed");
    }

    Ok(ProfileRun {
        ttft,
        inter_token_latencies: itl,
    })
}

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
}

fn print_row(label: &str, samples: &mut [Duration]) {
    if samples.is_empty() {
        println!(
            "{:<20} {:>10} {:>10} {:>10} {:>10}",
            label, "n/a", "n/a", "n/a", "n/a"
        );
        return;
    }
    samples.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    println!(
        "{:<20} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        label,
        ms(samples[0]),
        ms(percentile(samples, 50)),
        ms(percentile(samples, 95)),
        ms(percentile(samples, 99)),
    );
}

fn stat_map(samples: &mut [Duration]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::Value::Null;
    }
    samples.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    serde_json::json!({
        "min": ms(samples[0]),
        "p50": ms(percentile(samples, 50)),
        "p95": ms(percentile(samples, 95)),
        "p99": ms(percentile(samples, 99)),
        "n":   samples.len(),
    })
}
