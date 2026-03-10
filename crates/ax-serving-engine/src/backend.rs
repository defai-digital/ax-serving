//! MistralrsBackend: InferenceBackend implementation wrapping mistralrs.
//!
//! # Threading model
//!
//! `MistralrsBackend` owns a dedicated `tokio::runtime::Runtime`. This lets
//! sync callers (C API shim, benchmarks) use `runtime.block_on()` to drive
//! async operations.  When called from within an existing tokio runtime (REST
//! handler), callers call `backend.generate()` which fires a task on the
//! backend's internal runtime and communicates via an mpsc channel.
//!
//! # Model handles
//!
//! Each loaded model gets a `ModelHandle` (opaque u64). The backend keeps
//! `Arc<mistralrs::Model>` alive for each handle so the mistralrs engine
//! continues running.

use std::path::Path;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Context;
use either::Either;
use indexmap::IndexMap;
use mistralrs::{
    Constraint, GgufModelBuilder, MessageContent, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, StopTokens, TokenSource,
};
use mistralrs_core::{DetokenizationRequest, TokenizationRequest};
use rustc_hash::FxHashMap;
use tracing::{info, warn};

use crate::{
    BackendType, ChatMessage, GenerateEvent, GenerateInput, GenerationParams, GenerationStats,
    InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, ThermalMonitor, ThermalState,
};

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn next_handle() -> ModelHandle {
    ModelHandle(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
}

/// Per-model entry in the registry.
struct LoadedModel {
    /// mistralrs model — keeps the engine alive.
    model: Arc<mistralrs::Model>,
}

/// Build mistralrs `SamplingParams` from our `GenerationParams`.
///
/// `GenerationParams.temperature = None` means greedy (deterministic).  We
/// map it to `Some(0.0)` here because mistralrs treats `None` as "use the
/// model's default temperature" (typically 1.0), not as greedy sampling.
fn build_sampling_params(params: &GenerationParams) -> SamplingParams {
    SamplingParams {
        temperature: params.temperature.or(Some(0.0)),
        top_k: params.top_k,
        top_p: params.top_p,
        min_p: None,
        top_n_logprobs: 0,
        frequency_penalty: params.frequency_penalty.map(|v| v as f32),
        presence_penalty: params.presence_penalty.map(|v| v as f32),
        repetition_penalty: params.repeat_penalty.map(|r| r as f32),
        stop_toks: if params.stop_seqs.is_empty() {
            None
        } else {
            Some(StopTokens::Seqs(params.stop_seqs.clone()))
        },
        max_len: params.max_tokens,
        logits_bias: None,
        n_choices: 1,
        dry_params: None,
    }
}

/// Extract `GenerationStats` from mistralrs `Usage`.
fn stats_from_usage(usage: &mistralrs::Usage) -> GenerationStats {
    GenerationStats {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        prefill_tok_per_sec: usage.avg_prompt_tok_per_sec as f64,
        decode_tok_per_sec: usage.avg_compl_tok_per_sec as f64,
        stop_reason: String::new(), // mistralrs does not expose finish_reason yet
    }
}

/// `InferenceBackend` implementation wrapping mistralrs.
pub struct MistralrsBackend {
    // std::sync::RwLock is safe to lock from both sync and async contexts
    // (unlike tokio::sync::RwLock whose blocking_read/write panic in async).
    // The lock is only held briefly to clone an Arc or insert/remove an entry —
    // never across an await point.
    models: Arc<RwLock<FxHashMap<ModelHandle, LoadedModel>>>,
    thermal: ThermalMonitor,
    /// Dedicated tokio runtime for sync ↔ async bridging.
    runtime: Arc<tokio::runtime::Runtime>,
}

impl MistralrsBackend {
    pub fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("ax-serving-engine")
            .on_thread_start(|| {
                // Pin decode/prefill threads to P-cores (QOS_CLASS_USER_INTERACTIVE)
                // to avoid scheduling on E-cores which causes P95 latency jitter.
                #[cfg(target_os = "macos")]
                {
                    // libc does not expose pthread_set_qos_class_np; declare manually.
                    unsafe extern "C" {
                        fn pthread_set_qos_class_np(
                            thread: libc::pthread_t,
                            qos_class: libc::qos_class_t,
                            relative_priority: libc::c_int,
                        ) -> libc::c_int;
                    }
                    unsafe {
                        pthread_set_qos_class_np(
                            libc::pthread_self(),
                            libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
                            0,
                        );
                    }
                }
            })
            .build()
            .expect("failed to build ax-serving-engine tokio runtime");
        Self {
            models: Arc::new(RwLock::new(FxHashMap::default())),
            thermal: ThermalMonitor::new(),
            runtime: Arc::new(runtime),
        }
    }
}

impl Default for MistralrsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MistralrsBackend {
    // ── Model lifecycle ───────────────────────────────────────────────────────

    fn load_model(
        &self,
        path: &Path,
        config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)> {
        let rss_before = current_rss_bytes();
        let start = std::time::Instant::now();

        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());
        anyhow::ensure!(
            path.extension().and_then(|e| e.to_str()) == Some("gguf"),
            "only .gguf models are supported"
        );

        let file_size = std::fs::metadata(path)
            .context("cannot stat model file")?
            .len();
        crate::memory::check_memory_budget(file_size)?;

        // Read GGUF metadata before loading — fast (≤ 4096 bytes).
        let gguf_meta = crate::gguf_meta::read_gguf_meta(path).ok();

        // mistralrs GgufModelBuilder: model_id = parent directory, files = [filename].
        let parent = path
            .parent()
            .context("model path has no parent directory")?
            .to_str()
            .context("model parent path is not valid UTF-8")?;
        let filename = path
            .file_name()
            .context("model path has no filename")?
            .to_str()
            .context("model filename is not valid UTF-8")?;

        info!("loading model: {} (file: {})", parent, filename);

        let mut builder =
            GgufModelBuilder::new(parent, vec![filename]).with_token_source(TokenSource::None);

        if config.backend_type == BackendType::Cpu {
            builder = builder.with_force_cpu();
        }

        // Build the model on the caller thread. Metal device initialization has
        // been observed to fail in some environments when performed on backend
        // worker threads.
        let model = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build temporary runtime for model load")?
            .block_on(builder.build())
            .context("mistralrs model load failed")?;

        let max_seq = model.max_sequence_length().ok().flatten().unwrap_or(4096);
        let context_length = if config.context_length > 0 {
            config.context_length
        } else {
            max_seq as u32
        };

        let load_ms = start.elapsed().as_millis() as u64;
        let rss_after = current_rss_bytes();

        let metadata = ModelMetadata {
            architecture: gguf_meta
                .as_ref()
                .map(|m| m.architecture.clone())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "gguf".into()),
            n_layers: gguf_meta.as_ref().map(|m| m.block_count).unwrap_or(0),
            n_heads: gguf_meta.as_ref().map(|m| m.head_count).unwrap_or(0),
            n_kv_heads: gguf_meta.as_ref().map(|m| m.head_count_kv).unwrap_or(0),
            embedding_dim: gguf_meta.as_ref().map(|m| m.embedding_length).unwrap_or(0),
            vocab_size: gguf_meta.as_ref().map(|m| m.vocab_size).unwrap_or(0),
            // mistralrs's max_seq_len is authoritative (may apply its own capping).
            context_length,
            load_time_ms: load_ms,
            peak_rss_bytes: rss_after.saturating_sub(rss_before),
        };

        let handle = next_handle();
        let model_arc = Arc::new(model);

        {
            let mut guard = self.models.write().unwrap();
            guard.insert(handle, LoadedModel { model: model_arc });
        }

        info!(
            "model loaded in {}ms (handle={:?}, ctx={})",
            load_ms, handle, context_length
        );
        Ok((handle, metadata))
    }

    fn unload_model(&self, handle: ModelHandle) -> anyhow::Result<()> {
        let mut guard = self.models.write().unwrap();
        anyhow::ensure!(
            guard.remove(&handle).is_some(),
            "no model loaded with handle {:?}",
            handle
        );
        info!("unloaded model {:?}", handle);
        Ok(())
    }

    // ── Generation ────────────────────────────────────────────────────────────

    fn generate(
        &self,
        handle: ModelHandle,
        input: GenerateInput,
        params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> anyhow::Result<()> {
        // Obtain a cloned Arc<MistralRs> sender — this is cheap and has 'static lifetime.
        // std::sync::RwLock::read() is safe to call from both sync and async contexts.
        let guard = self.models.read().unwrap();
        let loaded = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("invalid model handle {:?}", handle))?;

        let sender = loaded
            .model
            .inner()
            .get_sender(None)
            .map_err(|e| anyhow::anyhow!("get_sender: {e}"))?;

        drop(guard);

        let sampling = build_sampling_params(&params);

        let (messages, is_streaming) = match input {
            GenerateInput::Tokens(tokens) => {
                (RequestMessage::CompletionTokens(tokens), params.stream)
            }
            GenerateInput::Text(text) => {
                // Wrap plain text as a single user chat message so the model's
                // built-in chat template (from GGUF) is applied automatically.
                let mut msg: IndexMap<String, MessageContent> = IndexMap::new();
                msg.insert("role".to_string(), Either::Left("user".to_string()));
                msg.insert("content".to_string(), Either::Left(text));
                (
                    RequestMessage::Chat {
                        messages: vec![msg],
                        enable_thinking: None,
                        reasoning_effort: None,
                    },
                    params.stream,
                )
            }
            GenerateInput::Chat(chat_messages) => {
                let messages = chat_messages
                    .into_iter()
                    .map(|m: ChatMessage| {
                        // mistralrs expects a plain string for content.
                        // For multipart messages (vision), concatenate text parts.
                        let text = match &m.content {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Array(parts) => parts
                                .iter()
                                .filter_map(|p| {
                                    if p["type"].as_str() == Some("text") {
                                        p["text"].as_str().map(|s| s.to_string())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join(""),
                            other => other.to_string(),
                        };
                        let mut msg: IndexMap<String, MessageContent> = IndexMap::new();
                        msg.insert("role".to_string(), Either::Left(m.role));
                        msg.insert("content".to_string(), Either::Left(text));
                        msg
                    })
                    .collect();
                (
                    RequestMessage::Chat {
                        messages,
                        enable_thinking: None,
                        reasoning_effort: None,
                    },
                    params.stream,
                )
            }
        };

        // Spawn the generation loop on the backend runtime.
        self.runtime.spawn(async move {
            let (resp_tx, mut resp_rx) = tokio::sync::mpsc::channel::<Response>(128);

            let request = Request::Normal(Box::new(NormalRequest {
                messages,
                sampling_params: sampling,
                response: resp_tx,
                return_logprobs: false,
                is_streaming,
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

            if let Err(e) = sender.send(request).await {
                let _ = tx.send(GenerateEvent::Error(e.to_string())).await;
                return;
            }

            while let Some(response) = resp_rx.recv().await {
                match response {
                    // Completion-style streaming chunk (for CompletionTokens / Completion input).
                    Response::CompletionChunk(chunk) => {
                        for choice in &chunk.choices {
                            if !choice.text.is_empty()
                                && tx
                                    .send(GenerateEvent::Token(choice.text.clone()))
                                    .await
                                    .is_err()
                            {
                                return;
                            }
                            if choice.finish_reason.is_some() {
                                // Always emit Done so callers receive a clean termination
                                // event even when mistralrs sends CompletionChunk (streaming
                                // completion path) instead of CompletionDone.
                                let _ = tx
                                    .send(GenerateEvent::Done(GenerationStats::default()))
                                    .await;
                                return;
                            }
                        }
                    }
                    // Chat-style streaming chunk.
                    Response::Chunk(chunk) => {
                        for choice in &chunk.choices {
                            if let Some(content) = &choice.delta.content
                                && !content.is_empty()
                                && tx
                                    .send(GenerateEvent::Token(content.clone()))
                                    .await
                                    .is_err()
                            {
                                return;
                            }
                            if choice.finish_reason.is_some() {
                                // mistralrs only includes usage in streaming chunks when
                                // stream_options.include_usage is set; fall back to default
                                // stats if the field is absent so Done is always emitted.
                                let stats = chunk
                                    .usage
                                    .as_ref()
                                    .map(stats_from_usage)
                                    .unwrap_or_default();
                                let _ = tx.send(GenerateEvent::Done(stats)).await;
                                return;
                            }
                        }
                    }
                    // Completion done (non-streaming or final).
                    // `is_streaming=false` means mistralrs accumulates the full
                    // response and delivers it here as one shot — forward all
                    // choice text as Token events before the Done event.
                    Response::CompletionDone(done) => {
                        for choice in &done.choices {
                            if !choice.text.is_empty()
                                && tx
                                    .send(GenerateEvent::Token(choice.text.clone()))
                                    .await
                                    .is_err()
                            {
                                return;
                            }
                        }
                        let _ = tx
                            .send(GenerateEvent::Done(stats_from_usage(&done.usage)))
                            .await;
                        return;
                    }
                    // Chat done.
                    Response::Done(done) => {
                        let _ = tx
                            .send(GenerateEvent::Done(stats_from_usage(&done.usage)))
                            .await;
                        return;
                    }
                    // Error variants.
                    Response::ModelError(msg, _) | Response::CompletionModelError(msg, _) => {
                        let _ = tx.send(GenerateEvent::Error(msg)).await;
                        return;
                    }
                    Response::InternalError(e) | Response::ValidationError(e) => {
                        let _ = tx.send(GenerateEvent::Error(e.to_string())).await;
                        return;
                    }
                    // Ignore irrelevant variants (embeddings, images, speech, raw).
                    _ => {}
                }
            }

            // Channel closed without a Done event — emit empty stats.
            warn!("mistralrs response channel closed without Done");
            let _ = tx
                .send(GenerateEvent::Done(GenerationStats::default()))
                .await;
        });

        Ok(())
    }

    // ── Tokenization ──────────────────────────────────────────────────────────

    fn tokenize(&self, handle: ModelHandle, text: &str, add_bos: bool) -> anyhow::Result<Vec<u32>> {
        let guard = self.models.read().unwrap();
        let loaded = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("invalid model handle {:?}", handle))?;

        // Get a cloned sender with 'static lifetime.
        let sender = loaded
            .model
            .inner()
            .get_sender(None)
            .map_err(|e| anyhow::anyhow!("get_sender: {e}"))?;
        drop(guard);

        let text_owned = text.to_string();
        self.runtime.block_on(async move {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let request = Request::Tokenize(TokenizationRequest {
                text: Either::Right(text_owned),
                tools: None,
                // Respect caller intent: include BOS/special tokens only when requested.
                add_special_tokens: add_bos,
                add_generation_prompt: false,
                response: tx,
                enable_thinking: None,
                reasoning_effort: None,
            });
            sender
                .send(request)
                .await
                .map_err(|e| anyhow::anyhow!("send tokenize request: {e}"))?;
            rx.recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("tokenize channel closed unexpectedly"))?
                .map_err(|e| anyhow::anyhow!("tokenize error: {e}"))
        })
    }

    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> anyhow::Result<String> {
        let guard = self.models.read().unwrap();
        let loaded = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("invalid model handle {:?}", handle))?;

        let sender = loaded
            .model
            .inner()
            .get_sender(None)
            .map_err(|e| anyhow::anyhow!("get_sender: {e}"))?;
        drop(guard);

        let tokens_owned = tokens.to_vec();
        self.runtime.block_on(async move {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let request = Request::Detokenize(DetokenizationRequest {
                tokens: tokens_owned,
                skip_special_tokens: true,
                response: tx,
            });
            sender
                .send(request)
                .await
                .map_err(|e| anyhow::anyhow!("send detokenize request: {e}"))?;
            rx.recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("detokenize channel closed unexpectedly"))?
                .map_err(|e| anyhow::anyhow!("detokenize error: {e}"))
        })
    }

    fn eos_tokens(&self, handle: ModelHandle) -> anyhow::Result<Vec<u32>> {
        let guard = self.models.read().unwrap();
        anyhow::ensure!(
            guard.contains_key(&handle),
            "invalid model handle {:?}",
            handle
        );
        // mistralrs handles EOS internally during generation.
        // Return common defaults; actual EOS is enforced by the engine.
        // LLaMA 3 = 128009, Gemma = 1, Qwen = 151645, Mistral = 2.
        Ok(vec![2])
    }

    fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> anyhow::Result<u32> {
        // Get the tokenize sender before entering block_on (sync RwLock read,
        // safe from any context). Must drop the guard before calling block_on.
        let guard = self.models.read().unwrap();
        let loaded = guard
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("invalid model handle {:?}", handle))?;
        let sender = loaded
            .model
            .inner()
            .get_sender(None)
            .map_err(|e| anyhow::anyhow!("get_sender: {e}"))?;
        drop(guard);

        // Spawn generation (max_tokens=1, greedy) on the backend runtime.
        // generate() calls runtime.spawn() — non-blocking, queues the task.
        let (gen_tx, mut gen_rx) = tokio::sync::mpsc::channel::<GenerateEvent>(32);
        self.generate(
            handle,
            GenerateInput::Tokens(tokens.to_vec()),
            GenerationParams {
                temperature: None, // greedy
                max_tokens: Some(1),
                top_p: None,
                top_k: None,
                stop_seqs: Vec::new(),
                seed: None,
                repeat_penalty: None,
                ..Default::default()
            },
            gen_tx,
        )?;

        // Drive the runtime in one block_on: collect the generated token text,
        // then tokenize it to get the predicted token ID.
        // The spawned generate task runs on the runtime's worker threads
        // concurrently with this block_on closure.
        self.runtime.block_on(async move {
            // Collect generated token text (max_tokens=1 → at most one Token event).
            let mut generated_text = String::new();
            while let Some(event) = gen_rx.recv().await {
                match event {
                    GenerateEvent::Token(text) => {
                        generated_text = text;
                    }
                    GenerateEvent::TokenLogprob { .. } | GenerateEvent::ToolCall { .. } => {}
                    GenerateEvent::Done(_) | GenerateEvent::Error(_) => break,
                }
            }
            if generated_text.is_empty() {
                return Err(anyhow::anyhow!("eval_tokens: no token generated"));
            }

            // Tokenize the generated text to recover the token ID.
            // Done inside the same block_on to avoid nested block_on calls.
            let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel(1);
            let request = Request::Tokenize(TokenizationRequest {
                text: Either::Right(generated_text),
                tools: None,
                add_special_tokens: false,
                add_generation_prompt: false,
                response: tok_tx,
                enable_thinking: None,
                reasoning_effort: None,
            });
            sender
                .send(request)
                .await
                .map_err(|e| anyhow::anyhow!("send tokenize request: {e}"))?;
            let token_ids = tok_rx
                .recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("tokenize channel closed"))?
                .map_err(|e| anyhow::anyhow!("tokenize error: {e}"))?;
            token_ids
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("eval_tokens: empty tokenization result"))
        })
    }

    // ── Thermal ───────────────────────────────────────────────────────────────

    fn thermal_state(&self) -> ThermalState {
        self.thermal.current()
    }

    fn recommended_concurrency(&self) -> usize {
        self.thermal.recommended_concurrency()
    }
}

// ── Memory helper (re-exported for memory.rs) ─────────────────────────────────

/// Current resident set size in bytes (macOS).
pub fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        let mut info = MaybeUninit::<libc::mach_task_basic_info_data_t>::uninit();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info_data_t>()
            / std::mem::size_of::<libc::integer_t>()) as u32;
        #[allow(deprecated)]
        let ret = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                info.as_mut_ptr() as libc::task_info_t,
                &mut count,
            )
        };
        if ret == libc::KERN_SUCCESS {
            return unsafe { info.assume_init() }.resident_size;
        }
    }
    0
}
