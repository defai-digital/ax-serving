//! Throughput benchmark: prefill tok/s + decode tok/s across prompt lengths.
//!
//! Threading model: `run()` is plain sync. The selected backend owns its own
//! execution path for model loading and generation. A small separate
//! `current_thread` runtime is created only to `.await` the channel drain — it
//! never calls into the backend.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use ax_serving_engine::{
    BackendChoice, GenerateEvent, GenerateInput, GenerationParams, GenerationStats,
    InferenceBackend, LoadConfig, ModelHandle, RouterBackend, RoutingConfig,
};
use tokio::sync::mpsc;

pub fn run(
    model: PathBuf,
    prompt_lengths: Vec<usize>,
    decode_tokens: usize,
    warmup: usize,
    iters: usize,
    json_out: Option<PathBuf>,
) -> Result<()> {
    // load_model is called here from a plain non-tokio thread.
    let backend = RouterBackend::from_env();
    let (handle, meta) = backend.load_model(&model, LoadConfig::default())?;

    // Dedicated drain runtime — only used for channel recv, never calls backend.
    let drain_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    println!("model: {} (ctx={})", meta.architecture, meta.context_length);
    println!("greedy (top_k=1), decode_tokens={decode_tokens}");
    println!(
        "{:<10} {:>14} {:>14}",
        "prompt_len", "prefill_tok/s", "decode_tok/s"
    );
    println!("{}", "─".repeat(40));

    let mut results: Vec<serde_json::Value> = Vec::new();
    let probe_len = prompt_lengths.first().copied().unwrap_or(39).max(1);
    let use_split_phase = should_use_split_phase(
        &model,
        &backend,
        &drain_rt,
        handle,
        probe_len,
        decode_tokens,
    )?;

    if use_split_phase {
        println!("mode: split-phase fallback (prefill-only + decode-focused)");
        let decode_tok_per_sec =
            measure_decode_focused(&backend, &drain_rt, handle, decode_tokens, warmup, iters)?;
        for &plen in &prompt_lengths {
            let prefill_tok_per_sec =
                measure_prefill_only(&backend, &drain_rt, handle, plen, warmup, iters)?;
            println!(
                "{:<10} {:>14.1} {:>14.1}",
                plen, prefill_tok_per_sec, decode_tok_per_sec
            );
            results.push(serde_json::json!({
                "prompt_length": plen,
                "prefill_tok_per_sec": prefill_tok_per_sec,
                "decode_tok_per_sec": decode_tok_per_sec,
            }));
        }
    } else {
        for &plen in &prompt_lengths {
            let prompt_tokens: Vec<u32> = (1..=(plen as u32)).collect();
            let params = GenerationParams {
                temperature: None,
                top_p: None,
                top_k: Some(1),
                max_tokens: Some(decode_tokens),
                stop_seqs: Vec::new(),
                seed: None,
                repeat_penalty: None,
                ..Default::default()
            };

            for _ in 0..warmup {
                one_run(&backend, &drain_rt, handle, &prompt_tokens, &params)?;
            }

            let mut pp_samples = Vec::with_capacity(iters);
            let mut tg_samples = Vec::with_capacity(iters);

            for _ in 0..iters {
                let s = one_run(&backend, &drain_rt, handle, &prompt_tokens, &params)?;
                pp_samples.push(s.prefill_tok_per_sec);
                tg_samples.push(s.decode_tok_per_sec);
            }

            let pp = median(&mut pp_samples);
            let tg = median(&mut tg_samples);

            println!("{:<10} {:>14.1} {:>14.1}", plen, pp, tg);
            results.push(serde_json::json!({
                "prompt_length": plen,
                "prefill_tok_per_sec": pp,
                "decode_tok_per_sec": tg,
            }));
        }
    }

    if let Some(path) = json_out {
        let out = serde_json::json!({ "model": meta.architecture, "results": results });
        std::fs::write(&path, serde_json::to_string_pretty(&out)?)?;
        println!("\nresults written to {}", path.display());
    }

    Ok(())
}

/// Probe whether backend provides native timing stats.
///
/// If both metrics are zero, we switch to split-phase fallback so prefill/decode
/// are measured in separate runs rather than from one mixed wall-time number.
fn should_use_split_phase(
    model: &std::path::Path,
    backend: &dyn InferenceBackend,
    drain_rt: &tokio::runtime::Runtime,
    handle: ModelHandle,
    probe_len: usize,
    decode_tokens: usize,
) -> Result<bool> {
    let cfg = RoutingConfig::load_default();
    if matches!(cfg.resolve(model), BackendChoice::LlamaCpp) {
        return Ok(true);
    }

    let probe_tokens: Vec<u32> = (1..=(probe_len as u32)).collect();
    let probe_decode = decode_tokens.max(1);
    let params = GenerationParams {
        temperature: None,
        top_p: None,
        top_k: Some(1),
        max_tokens: Some(probe_decode),
        stop_seqs: Vec::new(),
        seed: None,
        repeat_penalty: None,
        ..Default::default()
    };
    match one_run(backend, drain_rt, handle, &probe_tokens, &params) {
        Ok(s) => Ok(s.prefill_tok_per_sec <= 0.0 || s.decode_tok_per_sec <= 0.0),
        Err(e) => {
            eprintln!(
                "warning: timing probe failed ({e}); falling back to mixed-run measurement mode"
            );
            Ok(false)
        }
    }
}

fn measure_prefill_only(
    backend: &dyn InferenceBackend,
    drain_rt: &tokio::runtime::Runtime,
    handle: ModelHandle,
    prompt_len: usize,
    warmup: usize,
    iters: usize,
) -> Result<f64> {
    let prompt_tokens: Vec<u32> = (1..=(prompt_len as u32)).collect();
    let params = GenerationParams {
        temperature: None,
        top_p: None,
        top_k: Some(1),
        // Some llama.cpp OpenAI paths reject token-ID requests with max_tokens=0.
        // Use 1 token as a stable near-prefill pass.
        max_tokens: Some(1),
        stop_seqs: Vec::new(),
        seed: None,
        repeat_penalty: None,
        ..Default::default()
    };

    for _ in 0..warmup {
        one_run(backend, drain_rt, handle, &prompt_tokens, &params)?;
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = one_run(backend, drain_rt, handle, &prompt_tokens, &params)?;
        samples.push(s.prefill_tok_per_sec);
    }
    Ok(median(&mut samples))
}

fn measure_decode_focused(
    backend: &dyn InferenceBackend,
    drain_rt: &tokio::runtime::Runtime,
    handle: ModelHandle,
    decode_tokens: usize,
    warmup: usize,
    iters: usize,
) -> Result<f64> {
    let prompt_tokens: Vec<u32> = vec![1u32];
    let params = GenerationParams {
        temperature: None,
        top_p: None,
        top_k: Some(1),
        max_tokens: Some(decode_tokens),
        stop_seqs: Vec::new(),
        seed: None,
        repeat_penalty: None,
        ..Default::default()
    };

    for _ in 0..warmup {
        one_run(backend, drain_rt, handle, &prompt_tokens, &params)?;
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = one_run(backend, drain_rt, handle, &prompt_tokens, &params)?;
        samples.push(s.decode_tok_per_sec);
    }
    Ok(median(&mut samples))
}

/// One prefill+decode run. generate() is called synchronously (no runtime context),
/// then drain_rt awaits the channel — these two steps are strictly sequential.
fn one_run(
    backend: &dyn InferenceBackend,
    drain_rt: &tokio::runtime::Runtime,
    handle: ModelHandle,
    tokens: &[u32],
    params: &GenerationParams,
) -> Result<GenerationStats> {
    let (tx, rx) = mpsc::channel::<GenerateEvent>(512);

    // generate() is sync: acquires backend's blocking_read, spawns task on
    // backend's internal runtime, returns immediately. Must NOT be called from
    // inside any tokio block_on (hence the split from drain below).
    backend.generate(
        handle,
        GenerateInput::Tokens(tokens.to_vec()),
        params.clone(),
        tx,
    )?;

    // Now enter the drain runtime ONLY for channel recv — no backend calls here.
    drain_rt.block_on(drain_channel(rx, tokens.len()))
}

async fn drain_channel(
    mut rx: mpsc::Receiver<GenerateEvent>,
    n_prompt: usize,
) -> Result<GenerationStats> {
    let wall = Instant::now();
    let mut stats = GenerationStats::default();
    let mut first_token_after = None::<f64>;

    let mut got_done = false;
    while let Some(event) = rx.recv().await {
        match event {
            GenerateEvent::Token(_) => {
                first_token_after.get_or_insert_with(|| wall.elapsed().as_secs_f64());
            }
            GenerateEvent::Done(s) => {
                if s.prefill_tok_per_sec > 0.0 && s.decode_tok_per_sec > 0.0 {
                    stats = s;
                } else {
                    let elapsed = wall.elapsed().as_secs_f64().max(1e-9);
                    let (prefill_tps, decode_tps) = fallback_phase_tps(
                        n_prompt,
                        s.completion_tokens,
                        first_token_after,
                        elapsed,
                    );
                    tracing::warn!(
                        "backend did not report split-phase stats; \
                         throughput numbers estimated from first-token timing"
                    );
                    stats = s;
                    stats.prefill_tok_per_sec = prefill_tps;
                    stats.decode_tok_per_sec = decode_tps;
                }
                got_done = true;
                break;
            }
            GenerateEvent::Error(e) => anyhow::bail!("generation error: {e}"),
            GenerateEvent::ToolCall { .. } | GenerateEvent::TokenLogprob { .. } => {}
        }
    }
    if !got_done {
        anyhow::bail!("generation channel closed without Done event");
    }

    Ok(stats)
}

fn fallback_phase_tps(
    prompt_tokens: usize,
    completion_tokens: usize,
    first_token_after_secs: Option<f64>,
    total_elapsed_secs: f64,
) -> (f64, f64) {
    let total_elapsed_secs = total_elapsed_secs.max(1e-9);
    match first_token_after_secs {
        Some(ttft_secs) => {
            let prefill_elapsed = ttft_secs.max(1e-9);
            let decode_elapsed = (total_elapsed_secs - ttft_secs).max(1e-9);
            (
                prompt_tokens as f64 / prefill_elapsed,
                completion_tokens as f64 / decode_elapsed,
            )
        }
        None => (prompt_tokens as f64 / total_elapsed_secs, 0.0),
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::{fallback_phase_tps, median};

    #[test]
    fn median_even_uses_average_of_middle_values() {
        let mut values = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut values), 2.5);
    }

    #[test]
    fn median_odd_uses_middle_value() {
        let mut values = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut values), 2.0);
    }

    #[test]
    fn fallback_phase_tps_uses_first_token_timing() {
        let (prefill_tps, decode_tps) = fallback_phase_tps(1000, 100, Some(0.5), 1.0);
        assert_eq!(prefill_tps, 2000.0);
        assert_eq!(decode_tps, 200.0);
    }

    #[test]
    fn fallback_phase_tps_without_tokens_reports_zero_decode() {
        let (prefill_tps, decode_tps) = fallback_phase_tps(50, 0, None, 2.0);
        assert_eq!(prefill_tps, 25.0);
        assert_eq!(decode_tps, 0.0);
    }
}
