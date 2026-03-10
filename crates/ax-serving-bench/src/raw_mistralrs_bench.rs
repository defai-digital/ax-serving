//! raw-mistralrs-bench: pure mistralrs throughput, zero ax-serving overhead.
//!
//! Uses RequestMessage::CompletionTokens with synthetic token IDs, matching
//! what llama-bench (-p N) and ax-serving-bench do — no chat template overhead,
//! exact token counts.
//!
//! Usage (release build required for fair comparison):
//!   raw-mistralrs-bench -m models/foo.gguf \
//!     --prompt-lengths 64,256,512 --decode-tokens 128 --iters 3 --warmup 1

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("raw-mistralrs-bench only supports aarch64-apple-darwin");

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use mistralrs::{
    Constraint, GgufModelBuilder, NormalRequest, Request, RequestMessage, Response, SamplingParams,
    TokenSource,
};

#[derive(Parser, Debug)]
#[command(
    name = "raw-mistralrs-bench",
    about = "Pure mistralrs throughput benchmark (no ax-serving overhead)"
)]
struct Cli {
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Exact token counts to benchmark (matches llama-bench -p N).
    #[arg(long, value_delimiter = ',', default_value = "64,256,512")]
    prompt_lengths: Vec<usize>,

    /// Number of tokens to generate per run.
    #[arg(long, default_value = "128")]
    decode_tokens: usize,

    /// Warmup iterations (discarded).
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Measurement iterations.
    #[arg(long, default_value = "3")]
    iters: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Suppress mistralrs INFO chatter; MISTRALRS_LOG overrides.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("MISTRALRS_LOG")
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    let cli = Cli::parse();

    let path = &cli.model;
    anyhow::ensure!(path.exists(), "model not found: {}", path.display());

    let parent = path
        .parent()
        .context("no parent dir")?
        .to_str()
        .context("non-UTF8 parent")?;
    let filename = path
        .file_name()
        .context("no filename")?
        .to_str()
        .context("non-UTF8 filename")?;

    eprintln!("Loading {} ...", path.display());
    let load_start = Instant::now();
    let model = GgufModelBuilder::new(parent, vec![filename])
        .with_token_source(TokenSource::None)
        .build()
        .await
        .context("GgufModelBuilder::build failed")?;
    eprintln!("Loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    let sender: tokio::sync::mpsc::Sender<Request> = model
        .inner()
        .get_sender(None)
        .context("get_sender failed")?;

    // Greedy sampling — matches llama-bench and ax-serving-bench defaults.
    let sampling = SamplingParams {
        temperature: None,
        top_k: Some(1),
        top_p: None,
        min_p: None,
        top_n_logprobs: 0,
        frequency_penalty: None,
        presence_penalty: None,
        repetition_penalty: None,
        stop_toks: None,
        max_len: Some(cli.decode_tokens),
        logits_bias: None,
        n_choices: 1,
        dry_params: None,
    };

    println!("\nraw-mistralrs-bench  model={}", filename);
    println!("greedy (top_k=1), decode_tokens={}", cli.decode_tokens);
    println!(
        "{:<12} {:>16} {:>16}",
        "prompt_tokens", "prefill tok/s", "decode tok/s"
    );
    println!("{}", "─".repeat(46));

    for &plen in &cli.prompt_lengths {
        // Synthetic token IDs [1..=plen], same approach as llama-bench and ax-serving-bench.
        // Avoids chat template overhead — token count is exact.
        let tokens: Vec<u32> = (1..=(plen as u32)).collect();

        for _ in 0..cli.warmup {
            run_once(&sender, tokens.clone(), &sampling).await?;
        }

        let mut prefill_samples = Vec::with_capacity(cli.iters);
        let mut decode_samples = Vec::with_capacity(cli.iters);

        for _ in 0..cli.iters {
            let (pp, tg) = run_once(&sender, tokens.clone(), &sampling).await?;
            prefill_samples.push(pp);
            decode_samples.push(tg);
        }

        let pp_med = median(&mut prefill_samples);
        let tg_med = median(&mut decode_samples);

        println!("{:<12} {:>16.1} {:>16.1}", plen, pp_med, tg_med);
    }

    Ok(())
}

/// Send CompletionTokens request (exact token IDs, no template), drain responses,
/// return (prefill_tok_per_sec, decode_tok_per_sec).
async fn run_once(
    sender: &tokio::sync::mpsc::Sender<Request>,
    tokens: Vec<u32>,
    sampling: &SamplingParams,
) -> Result<(f64, f64)> {
    let (resp_tx, mut resp_rx) = tokio::sync::mpsc::channel::<Response>(256);

    let request = Request::Normal(Box::new(NormalRequest {
        messages: RequestMessage::CompletionTokens(tokens),
        sampling_params: sampling.clone(),
        response: resp_tx,
        return_logprobs: false,
        is_streaming: false, // CompletionDone (with usage) only sent on non-streaming path
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: None,
        truncate_sequence: false,
    }));

    sender
        .send(request)
        .await
        .map_err(|e| anyhow::anyhow!("send failed: {e}"))?;

    while let Some(resp) = resp_rx.recv().await {
        match resp {
            Response::Done(done) => {
                return Ok((
                    done.usage.avg_prompt_tok_per_sec as f64,
                    done.usage.avg_compl_tok_per_sec as f64,
                ));
            }
            Response::CompletionDone(done) => {
                return Ok((
                    done.usage.avg_prompt_tok_per_sec as f64,
                    done.usage.avg_compl_tok_per_sec as f64,
                ));
            }
            Response::Chunk(chunk) => {
                for choice in &chunk.choices {
                    if choice.finish_reason.is_some()
                        && let Some(usage) = &chunk.usage
                    {
                        return Ok((
                            usage.avg_prompt_tok_per_sec as f64,
                            usage.avg_compl_tok_per_sec as f64,
                        ));
                    }
                }
            }
            Response::CompletionChunk(_) => {
                // Intermediate chunk — no usage yet; wait for CompletionDone.
            }
            Response::ModelError(e, _) | Response::CompletionModelError(e, _) => {
                anyhow::bail!("model error: {e}");
            }
            Response::InternalError(e) | Response::ValidationError(e) => {
                anyhow::bail!("internal error: {e}");
            }
            _ => {}
        }
    }

    anyhow::bail!("response channel closed without usage stats");
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
    use super::median;

    #[test]
    fn median_even_uses_average_of_middle_values() {
        let mut values = vec![10.0, 2.0, 8.0, 4.0];
        assert_eq!(median(&mut values), 6.0);
    }
}
