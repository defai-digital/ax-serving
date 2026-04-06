//! AxEngineBackend: `InferenceBackend` implementation backed by ax-engine.

use std::path::Path;
use std::sync::{
    Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ax_core::backend::{self, BackendConfig};
use ax_core::chat::{self, ChatRenderOptions, ChatRole};
use ax_core::gguf::MappedModel;
use ax_core::metrics::current_rss_bytes;
use ax_core::model::{
    DecodeControl, LlamaModel, ModelConfig, WeightStore, arch_registry, run_decode,
};
use ax_core::sampling::{LogitBias, SampledTokenInfo, Sampler, SamplingConfig};
use ax_core::tokenizer::Tokenizer;
use rustc_hash::FxHashMap;
use tracing::warn;

use crate::{
    BackendType, ChatMessage, GenerateEvent, GenerateInput, GenerationParams, GenerationStats,
    InferenceBackend, LoadConfig, ModelHandle, ModelMetadata, ThermalMonitor, ThermalState,
};

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(8_000_000);
const DEFAULT_STREAM_TOKEN_BATCH_SIZE: usize = 4;
const MAX_STREAM_TOKEN_BATCH_SIZE: usize = 32;

fn next_handle() -> ModelHandle {
    ModelHandle(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
}

fn stream_token_batch_size() -> usize {
    std::env::var("AXS_STREAM_TOKEN_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STREAM_TOKEN_BATCH_SIZE)
        .clamp(1, MAX_STREAM_TOKEN_BATCH_SIZE)
}

fn flush_stream_token_batch(
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    buffer: &mut String,
    buffered_pieces: &mut usize,
) -> bool {
    *buffered_pieces = 0;
    if buffer.is_empty() {
        return true;
    }
    tx.blocking_send(GenerateEvent::Token(std::mem::take(buffer)))
        .is_ok()
}

fn push_stream_token_piece(
    tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    piece: String,
    batch_size: usize,
    first_chunk_sent: &mut bool,
    buffer: &mut String,
    buffered_pieces: &mut usize,
) -> bool {
    if piece.is_empty() {
        return true;
    }
    if !*first_chunk_sent {
        *first_chunk_sent = true;
        return tx.blocking_send(GenerateEvent::Token(piece)).is_ok();
    }

    buffer.push_str(&piece);
    *buffered_pieces += 1;
    if *buffered_pieces < batch_size {
        return true;
    }

    flush_stream_token_batch(tx, buffer, buffered_pieces)
}

struct LoadedModel {
    mapped: MappedModel,
    tokenizer: Tokenizer,
    model: LlamaModel,
    metadata: ModelMetadata,
    render_architecture: String,
}

// SAFETY: LoadedModel is immutable after load. Per-request mutable state lives
// in fresh KV caches and sampler/history buffers, not in the loaded model.
// `MappedModel` is read-only mmap-backed data, `Tokenizer` is immutable, and
// `LlamaModel` methods take `&self` and operate on caller-owned buffers.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

fn resolve_backend_config(config: &LoadConfig) -> BackendConfig {
    match config.backend_type {
        BackendType::Cpu => BackendConfig::Cpu,
        BackendType::Metal => BackendConfig::Metal,
        BackendType::Auto => match std::env::var("AX_HYBRID_DECODE") {
            Ok(v) if v.trim().eq_ignore_ascii_case("cpu") => BackendConfig::HybridCpuDecode,
            _ => {
                if std::env::var("AX_CPU_ONLY")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                {
                    BackendConfig::Cpu
                } else {
                    BackendConfig::default()
                }
            }
        },
    }
}

fn resolved_backend_type(config: BackendConfig) -> BackendType {
    match config {
        BackendConfig::Cpu => BackendType::Cpu,
        BackendConfig::Metal | BackendConfig::Hybrid | BackendConfig::HybridCpuDecode => {
            BackendType::Metal
        }
    }
}

fn extract_text(content: &serde_json::Value) -> Result<String> {
    match content {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(|v| v.as_str()) {
                    Some("text") => {
                        let part_text = part
                            .get("text")
                            .and_then(|v| v.as_str())
                            .context("chat text part missing string 'text' field")?;
                        text.push_str(part_text);
                    }
                    Some(other) => {
                        anyhow::bail!(
                            "native ax-engine backend only supports text chat content; got part type '{other}'"
                        );
                    }
                    None => anyhow::bail!(
                        "native ax-engine backend requires multipart chat content parts to include a string 'type' field"
                    ),
                }
            }
            Ok(text)
        }
        other => anyhow::bail!(
            "native ax-engine backend only supports string or text-part chat content, got {other}"
        ),
    }
}

fn parse_chat_role(role: &str) -> Result<ChatRole> {
    match role.trim().to_ascii_lowercase().as_str() {
        "system" | "developer" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        "tool" | "function" => {
            anyhow::bail!("native ax-engine backend does not support chat role '{role}'")
        }
        _ => anyhow::bail!("unsupported chat role for native ax-engine backend: {role}"),
    }
}

fn normalize_chat_messages(messages: &[ChatMessage]) -> Result<Vec<(ChatRole, String)>> {
    let mut normalized = Vec::with_capacity(messages.len());
    for message in messages {
        let role = parse_chat_role(&message.role)?;
        let content = extract_text(&message.content)?;
        normalized.push((role, content));
    }
    Ok(normalized)
}

fn infer_render_architecture(model_architecture: &str, chat_template: Option<&str>) -> String {
    if let Some(chat_template) = chat_template {
        if chat_template.contains("<|start_header_id|>") {
            return "llama".to_string();
        }
        if chat_template.contains("<|im_start|>") {
            return "qwen3".to_string();
        }
        if chat_template.contains("<start_of_turn>") {
            return "gemma3".to_string();
        }
        if chat_template.contains("[INST]") && chat_template.contains("[/INST]") {
            return "mistral".to_string();
        }
    }
    model_architecture.to_string()
}

fn render_chat_messages_with_compat(
    messages: &[chat::ChatMessage<'_>],
    architecture: &str,
    options: ChatRenderOptions,
) -> String {
    if matches!(architecture, "mistral" | "mixtral") {
        return render_mistral_chat_messages(messages);
    }
    chat::render_chat_messages(messages, architecture, options)
}

fn render_mistral_chat_messages(messages: &[chat::ChatMessage<'_>]) -> String {
    let mut rendered = String::new();
    let mut pending_system = None;

    for message in messages {
        match message.role {
            chat::ChatRole::System => {
                pending_system = Some(message.content);
            }
            chat::ChatRole::User => {
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str("[INST] ");
                if let Some(system) = pending_system.take() {
                    rendered.push_str("<<SYS>>\n");
                    rendered.push_str(system);
                    rendered.push_str("\n<</SYS>>\n\n");
                }
                rendered.push_str(message.content);
                rendered.push_str(" [/INST]");
            }
            chat::ChatRole::Assistant => {
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str(message.content);
            }
        }
    }

    rendered
}

fn build_sampling_config(params: &GenerationParams) -> SamplingConfig {
    SamplingConfig {
        logit_bias: Vec::<LogitBias>::new(),
        allowed_token_ids: Vec::new(),
        banned_token_ids: Vec::new(),
        temperature: params.temperature.unwrap_or(0.0) as f32,
        top_k: params.top_k.map(|v| v as i32).unwrap_or(40),
        top_p: params.top_p.unwrap_or(0.9) as f32,
        min_p: params.min_p.unwrap_or(0.0) as f32,
        min_keep: 1,
        repeat_penalty: params.repeat_penalty.unwrap_or(1.0) as f32,
        frequency_penalty: params.frequency_penalty.unwrap_or(0.0) as f32,
        presence_penalty: params.presence_penalty.unwrap_or(0.0) as f32,
        repeat_last_n: 64,
        seed: params.seed.unwrap_or(u64::MAX),
    }
}

fn unsupported_generation_features(params: &GenerationParams) -> Vec<&'static str> {
    let mut unsupported = Vec::new();

    if params.grammar.is_some() {
        unsupported.push("grammar");
    }
    if matches!(params.response_format.as_deref(), Some(kind) if kind != "text") {
        unsupported.push("response_format");
    }
    if params.mirostat.unwrap_or(0) > 0
        || params.mirostat_tau.is_some()
        || params.mirostat_eta.is_some()
    {
        unsupported.push("mirostat");
    }
    if params.tools.is_some() || params.tool_choice.is_some() {
        unsupported.push("tools");
    }

    unsupported
}

fn ensure_supported_generation_params(params: &GenerationParams) -> Result<()> {
    let unsupported = unsupported_generation_features(params);
    anyhow::ensure!(
        unsupported.is_empty(),
        "native ax-engine backend does not support {}",
        unsupported.join(", ")
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct StopPieceAction {
    emit: String,
    matched: bool,
}

fn render_token_text(tokenizer: &Tokenizer, token: u32) -> String {
    tokenizer
        .render_token(token)
        .unwrap_or_else(|| tokenizer.decode(&[token]))
}

fn sample_top_logprobs_text(
    tokenizer: &Tokenizer,
    info: &SampledTokenInfo,
    top_logprobs: usize,
) -> Vec<(String, f32)> {
    info.top_logprobs
        .iter()
        .take(top_logprobs)
        .map(|entry| (render_token_text(tokenizer, entry.token), entry.logprob))
        .collect()
}

fn split_keep_tail_chars(text: &str, keep_tail_chars: usize) -> (String, String) {
    if keep_tail_chars == 0 {
        return (text.to_string(), String::new());
    }

    let total_chars = text.chars().count();
    if total_chars <= keep_tail_chars {
        return (String::new(), text.to_string());
    }

    let split_at = text
        .char_indices()
        .nth(total_chars - keep_tail_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    (text[..split_at].to_string(), text[split_at..].to_string())
}

fn consume_stop_piece(pending: &mut String, piece: &str, stop_seqs: &[String]) -> StopPieceAction {
    pending.push_str(piece);

    if stop_seqs.is_empty() {
        let emit = std::mem::take(pending);
        return StopPieceAction {
            emit,
            matched: false,
        };
    }

    if let Some(matched) = stop_seqs
        .iter()
        .filter(|seq| pending.ends_with(seq.as_str()))
        .max_by_key(|seq| seq.len())
    {
        let emit_len = pending.len().saturating_sub(matched.len());
        let emit = pending[..emit_len].to_string();
        pending.clear();
        return StopPieceAction {
            emit,
            matched: true,
        };
    }

    let hold_chars = stop_seqs
        .iter()
        .map(|seq| seq.chars().count().saturating_sub(1))
        .max()
        .unwrap_or(0);
    let (emit, tail) = split_keep_tail_chars(pending, hold_chars);
    *pending = tail;

    StopPieceAction {
        emit,
        matched: false,
    }
}

struct DecodeOutcome {
    completion_tokens: usize,
    decode_duration: Duration,
}

fn stats_from_decode(
    prompt_tokens: usize,
    prefill_duration: std::time::Duration,
    decode_duration: std::time::Duration,
    decode_tokens: usize,
    stop_reason: String,
) -> GenerationStats {
    GenerationStats {
        prompt_tokens,
        completion_tokens: decode_tokens,
        prefill_tok_per_sec: if prefill_duration.as_secs_f64() > 0.0 {
            prompt_tokens as f64 / prefill_duration.as_secs_f64()
        } else {
            0.0
        },
        decode_tok_per_sec: if decode_duration.as_secs_f64() > 0.0 {
            decode_tokens as f64 / decode_duration.as_secs_f64()
        } else {
            0.0
        },
        stop_reason,
    }
}

fn run_generate(
    loaded: Arc<LoadedModel>,
    input: GenerateInput,
    params: GenerationParams,
    tx: tokio::sync::mpsc::Sender<GenerateEvent>,
) -> Result<()> {
    ensure_supported_generation_params(&params)?;

    let prompt = match input {
        GenerateInput::Tokens(tokens) => tokens,
        GenerateInput::Text(text) => loaded.tokenizer.encode(
            &chat::render_user_prompt(&text, &loaded.render_architecture),
            true,
        ),
        GenerateInput::Chat(messages) => {
            let normalized = normalize_chat_messages(&messages)?;
            let chat_messages: Vec<_> = normalized
                .iter()
                .map(|(role, content)| chat::ChatMessage::new(*role, content.as_str()))
                .collect();
            loaded.tokenizer.encode(
                &render_chat_messages_with_compat(
                    &chat_messages,
                    &loaded.render_architecture,
                    ChatRenderOptions::default(),
                ),
                true,
            )
        }
    };

    anyhow::ensure!(
        !prompt.is_empty(),
        "empty token sequence after tokenization"
    );

    let ctx_size = loaded.metadata.context_length as usize;
    anyhow::ensure!(
        prompt.len() < ctx_size,
        "input ({} tokens) exceeds context length ({ctx_size})",
        prompt.len()
    );

    let max_new = params
        .max_tokens
        .unwrap_or(512)
        .min(ctx_size.saturating_sub(prompt.len()));
    let weights = WeightStore::new(&loaded.mapped);
    let mut kv = loaded.model.create_model_kv();
    let mut logits = vec![0.0f32; loaded.metadata.vocab_size as usize];
    let mut sampler = Sampler::new(build_sampling_config(&params));
    let mut history = prompt.clone();
    let emit_logprobs = params.logprobs.unwrap_or(false);
    let top_logprobs = params.top_logprobs.unwrap_or(0) as usize;
    // ax-engine v1.0 only threads `SampledTokenInfo` through `run_decode` when
    // `top_logprobs > 0`. Request at least one candidate internally so
    // `logprobs=true` still yields sampled-token logprobs when the client did
    // not ask for a top-N list.
    let decode_top_logprobs = if emit_logprobs {
        top_logprobs.max(1)
    } else {
        0
    };

    let prefill_started = Instant::now();
    loaded
        .model
        .forward_batch(&prompt, &mut kv, &weights, &mut logits)
        .context("ax-engine prefill failed")?;
    let prefill_duration = prefill_started.elapsed();
    if max_new == 0 {
        let stats = stats_from_decode(
            prompt.len(),
            prefill_duration,
            Duration::ZERO,
            0,
            "length".to_string(),
        );
        let _ = tx.blocking_send(GenerateEvent::Done(stats));
        return Ok(());
    }

    let mut stop_reason = "stop".to_string();
    let mut generated_tokens = 0usize;
    let mut stop_buffer = String::new();
    let mut stopped_on_stop_sequence = false;
    let stream_batch_size = stream_token_batch_size();
    let mut first_stream_chunk_sent = false;
    let mut stream_token_buffer = String::new();
    let mut buffered_stream_tokens = 0usize;
    let decode = if emit_logprobs {
        let first_sample = sampler.sample_with_logprobs(&mut logits, &history, decode_top_logprobs);
        run_decode(
            &loaded.model,
            &weights,
            &loaded.tokenizer,
            &mut kv,
            &mut sampler,
            &mut history,
            first_sample.token,
            Some(first_sample),
            prompt.len(),
            max_new,
            ax_core::model::DecodeRunConfig {
                intent: ax_core::model::DecodeIntent::Throughput,
                allow_pipelined: true,
                top_logprobs: decode_top_logprobs,
            },
            |token, info| {
                generated_tokens += 1;
                let piece = render_token_text(&loaded.tokenizer, token);
                let action = consume_stop_piece(&mut stop_buffer, &piece, &params.stop_seqs);
                if !action.emit.is_empty() {
                    if tx.blocking_send(GenerateEvent::Token(action.emit)).is_err() {
                        anyhow::bail!("receiver dropped");
                    }
                    if let Some(info) = info
                        && tx
                            .blocking_send(GenerateEvent::TokenLogprob {
                                logprob: info.logprob,
                                top: sample_top_logprobs_text(
                                    &loaded.tokenizer,
                                    info,
                                    top_logprobs,
                                ),
                            })
                            .is_err()
                    {
                        anyhow::bail!("receiver dropped");
                    }
                }
                if action.matched {
                    stopped_on_stop_sequence = true;
                    return Ok(DecodeControl::Stop);
                }
                Ok(DecodeControl::Continue)
            },
        )
        .map(|result| DecodeOutcome {
            completion_tokens: generated_tokens,
            decode_duration: result.decode_duration,
        })
    } else {
        let next_token = sampler.sample(&mut logits, &history);
        run_decode(
            &loaded.model,
            &weights,
            &loaded.tokenizer,
            &mut kv,
            &mut sampler,
            &mut history,
            next_token,
            None,
            prompt.len(),
            max_new,
            ax_core::model::DecodeRunConfig {
                intent: ax_core::model::DecodeIntent::Throughput,
                allow_pipelined: true,
                top_logprobs: 0,
            },
            |token, _info| {
                generated_tokens += 1;
                let piece = render_token_text(&loaded.tokenizer, token);

                let action = consume_stop_piece(&mut stop_buffer, &piece, &params.stop_seqs);
                if !push_stream_token_piece(
                    &tx,
                    action.emit,
                    stream_batch_size,
                    &mut first_stream_chunk_sent,
                    &mut stream_token_buffer,
                    &mut buffered_stream_tokens,
                ) {
                    anyhow::bail!("receiver dropped");
                }
                if action.matched {
                    stopped_on_stop_sequence = true;
                    return Ok(DecodeControl::Stop);
                }
                Ok(DecodeControl::Continue)
            },
        )
        .map(|result| DecodeOutcome {
            completion_tokens: generated_tokens,
            decode_duration: result.decode_duration,
        })
    };

    let (completion_tokens, decode_duration) = match decode {
        Ok(outcome) => {
            if !push_stream_token_piece(
                &tx,
                std::mem::take(&mut stop_buffer),
                stream_batch_size,
                &mut first_stream_chunk_sent,
                &mut stream_token_buffer,
                &mut buffered_stream_tokens,
            ) {
                return Ok(());
            }
            if !flush_stream_token_batch(&tx, &mut stream_token_buffer, &mut buffered_stream_tokens)
            {
                return Ok(());
            }
            (outcome.completion_tokens, outcome.decode_duration)
        }
        Err(err) if err.to_string() == "receiver dropped" => return Ok(()),
        Err(err) => return Err(err).context("ax-engine decode failed"),
    };

    if !stopped_on_stop_sequence && max_new > 0 && completion_tokens >= max_new {
        stop_reason = "length".to_string();
    }

    let stats = stats_from_decode(
        prompt.len(),
        prefill_duration,
        decode_duration,
        completion_tokens,
        stop_reason,
    );
    let _ = tx.blocking_send(GenerateEvent::Done(stats));
    Ok(())
}

/// `InferenceBackend` implementation backed by ax-engine (`ax-core`).
pub struct AxEngineBackend {
    models: Arc<RwLock<FxHashMap<ModelHandle, Arc<LoadedModel>>>>,
    thermal: ThermalMonitor,
}

impl AxEngineBackend {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(FxHashMap::default())),
            thermal: ThermalMonitor::new(),
        }
    }

    fn models_read(&self) -> RwLockReadGuard<'_, FxHashMap<ModelHandle, Arc<LoadedModel>>> {
        self.models.read().unwrap_or_else(|err| {
            warn!("ax-engine models rwlock poisoned; recovering from poisoned read lock");
            err.into_inner()
        })
    }

    fn models_write(&self) -> RwLockWriteGuard<'_, FxHashMap<ModelHandle, Arc<LoadedModel>>> {
        self.models.write().unwrap_or_else(|err| {
            warn!("ax-engine models rwlock poisoned; recovering from poisoned write lock");
            err.into_inner()
        })
    }

    fn get_model(&self, handle: ModelHandle) -> Result<Arc<LoadedModel>> {
        self.models_read()
            .get(&handle)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("invalid ax-engine model handle {:?}", handle))
    }
}

impl Default for AxEngineBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for AxEngineBackend {
    fn load_model(&self, path: &Path, config: LoadConfig) -> Result<(ModelHandle, ModelMetadata)> {
        let rss_before = current_rss_bytes();
        let started = Instant::now();

        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());
        anyhow::ensure!(
            path.extension().and_then(|e| e.to_str()) == Some("gguf"),
            "only .gguf models are supported"
        );

        let file_size = std::fs::metadata(path)
            .context("cannot stat model file")?
            .len();
        crate::memory::check_memory_budget(file_size)?;

        let mapped = MappedModel::open(path).context("ax-engine failed to map GGUF model")?;
        let mut model_config = ModelConfig::from_gguf(&mapped.header)
            .context("ax-engine failed to parse GGUF metadata")?;
        arch_registry::forward_for_arch(&model_config.architecture)
            .context("unsupported architecture for ax-engine native backend")?;

        if config.context_length > 0 {
            model_config.context_length = config.context_length;
        }

        if config.enable_embeddings == Some(true) {
            anyhow::bail!("embeddings are not supported by ax-engine native backend");
        }

        let render_architecture = infer_render_architecture(
            &model_config.architecture,
            chat::gguf_chat_template(&mapped.header),
        );
        let tokenizer =
            Tokenizer::from_gguf(&mapped.header).context("ax-engine failed to load tokenizer")?;
        let backend_config = resolve_backend_config(&config);
        let backend = backend::create_backend(backend_config)
            .context("ax-engine failed to create compute backend")?;
        let model = LlamaModel::with_backend(model_config.clone(), backend);

        let rss_after = current_rss_bytes();
        let metadata = ModelMetadata {
            architecture: model_config.architecture.clone(),
            n_layers: model_config.n_layers,
            n_heads: model_config.n_heads,
            n_kv_heads: model_config.n_kv_heads,
            embedding_dim: model_config.embedding_dim,
            vocab_size: model_config.vocab_size,
            context_length: model_config.context_length,
            load_time_ms: started.elapsed().as_millis() as u64,
            peak_rss_bytes: rss_after.saturating_sub(rss_before),
            resolved_backend: resolved_backend_type(backend_config),
        };

        let handle = next_handle();
        let loaded = Arc::new(LoadedModel {
            mapped,
            tokenizer,
            model,
            metadata: metadata.clone(),
            render_architecture,
        });
        self.models_write().insert(handle, loaded);

        Ok((handle, metadata))
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        anyhow::ensure!(
            self.models_write().remove(&handle).is_some(),
            "no model loaded with handle {:?}",
            handle
        );
        Ok(())
    }

    fn generate(
        &self,
        handle: ModelHandle,
        input: GenerateInput,
        params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> Result<()> {
        ensure_supported_generation_params(&params)?;
        let loaded = self.get_model(handle)?;
        std::thread::Builder::new()
            .name("ax-engine-generate".into())
            .spawn(move || {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_generate(loaded, input, params, tx.clone())
                })) {
                    Ok(Err(err)) => {
                        let _ = tx.blocking_send(GenerateEvent::Error(err.to_string()));
                    }
                    Err(_panic) => {
                        let _ = tx.blocking_send(GenerateEvent::Error(
                            "internal panic in ax-engine generate".into(),
                        ));
                    }
                    Ok(Ok(())) => {}
                }
            })
            .context("failed to spawn ax-engine generation thread")?;
        Ok(())
    }

    fn tokenize(&self, handle: ModelHandle, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        Ok(self.get_model(handle)?.tokenizer.encode(text, add_bos))
    }

    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<String> {
        Ok(self.get_model(handle)?.tokenizer.decode(tokens))
    }

    fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
        Ok(vec![self.get_model(handle)?.tokenizer.eos_id()])
    }

    fn thermal_state(&self) -> ThermalState {
        self.thermal.current()
    }

    fn recommended_concurrency(&self) -> usize {
        self.thermal.recommended_concurrency()
    }

    fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<u32> {
        anyhow::ensure!(!tokens.is_empty(), "eval_tokens: empty input");
        let loaded = self.get_model(handle)?;
        let weights = WeightStore::new(&loaded.mapped);
        let mut kv = loaded.model.create_model_kv();
        let mut logits = vec![0.0f32; loaded.metadata.vocab_size as usize];
        loaded
            .model
            .forward_batch(tokens, &mut kv, &weights, &mut logits)
            .context("ax-engine eval prefill failed")?;
        Ok(ax_core::sampling::argmax(&logits))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ax_core::chat::{self, ChatRenderOptions, ChatRole};
    use tokio::sync::mpsc;

    use super::{
        BackendConfig, BackendType, ChatMessage, GenerateEvent, GenerationParams, LoadConfig,
        build_sampling_config, consume_stop_piece, ensure_supported_generation_params,
        flush_stream_token_batch, infer_render_architecture, normalize_chat_messages,
        parse_chat_role, push_stream_token_piece, resolve_backend_config,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn backend_type_cpu_maps_to_cpu_backend() {
        let cfg = LoadConfig {
            backend_type: BackendType::Cpu,
            ..Default::default()
        };
        assert_eq!(resolve_backend_config(&cfg), BackendConfig::Cpu);
    }

    #[test]
    fn backend_type_metal_maps_to_metal_backend() {
        let cfg = LoadConfig {
            backend_type: BackendType::Metal,
            ..Default::default()
        };
        assert_eq!(resolve_backend_config(&cfg), BackendConfig::Metal);
    }

    #[test]
    fn auto_backend_honors_cpu_only_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("AX_CPU_ONLY", "1") };
        unsafe { std::env::remove_var("AX_HYBRID_DECODE") };
        assert_eq!(
            resolve_backend_config(&LoadConfig::default()),
            BackendConfig::Cpu
        );
        unsafe { std::env::remove_var("AX_CPU_ONLY") };
    }

    #[test]
    fn auto_backend_honors_hybrid_cpu_decode_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("AX_CPU_ONLY") };
        unsafe { std::env::set_var("AX_HYBRID_DECODE", "cpu") };
        assert_eq!(
            resolve_backend_config(&LoadConfig::default()),
            BackendConfig::HybridCpuDecode
        );
        unsafe { std::env::remove_var("AX_HYBRID_DECODE") };
    }

    #[test]
    fn stop_sequence_piece_suppresses_matched_suffix() {
        let mut pending = String::new();
        let stop = vec!["END".to_string()];

        let first = consume_stop_piece(&mut pending, "hello EN", &stop);
        assert_eq!(first.emit, "hello ");
        assert!(!first.matched);
        assert_eq!(pending, "EN");

        let second = consume_stop_piece(&mut pending, "D", &stop);
        assert_eq!(second.emit, "");
        assert!(second.matched);
        assert!(pending.is_empty());
    }

    #[test]
    fn stop_sequence_piece_flushes_buffer_when_no_match_occurs() {
        let mut pending = String::new();
        let stop = vec!["END".to_string()];

        let first = consume_stop_piece(&mut pending, "ab", &stop);
        assert_eq!(first.emit, "");
        assert!(!first.matched);

        let second = consume_stop_piece(&mut pending, "cX", &stop);
        assert_eq!(second.emit, "ab");
        assert!(!second.matched);
        assert_eq!(pending, "cX");
    }

    #[test]
    fn unsupported_generation_params_are_rejected() {
        let params = GenerationParams {
            grammar: Some("root ::= 'x'".to_string()),
            ..Default::default()
        };
        let err = ensure_supported_generation_params(&params).unwrap_err();
        assert!(
            err.to_string()
                .contains("native ax-engine backend does not support grammar")
        );
    }

    #[test]
    fn text_response_format_is_allowed() {
        let params = GenerationParams {
            response_format: Some("text".to_string()),
            ..Default::default()
        };
        assert!(ensure_supported_generation_params(&params).is_ok());
    }

    #[test]
    fn first_stream_piece_is_sent_immediately() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut first_chunk_sent = false;
        let mut buffer = String::new();
        let mut buffered_pieces = 0usize;

        assert!(push_stream_token_piece(
            &tx,
            "hello".to_string(),
            4,
            &mut first_chunk_sent,
            &mut buffer,
            &mut buffered_pieces,
        ));

        match rx.try_recv() {
            Ok(GenerateEvent::Token(text)) => assert_eq!(text, "hello"),
            other => panic!("expected first token event, got {other:?}"),
        }
        assert!(buffer.is_empty());
        assert_eq!(buffered_pieces, 0);
    }

    #[test]
    fn steady_state_stream_pieces_are_batched() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut first_chunk_sent = false;
        let mut buffer = String::new();
        let mut buffered_pieces = 0usize;

        assert!(push_stream_token_piece(
            &tx,
            "a".to_string(),
            2,
            &mut first_chunk_sent,
            &mut buffer,
            &mut buffered_pieces,
        ));
        let _ = rx.try_recv();

        assert!(push_stream_token_piece(
            &tx,
            "b".to_string(),
            2,
            &mut first_chunk_sent,
            &mut buffer,
            &mut buffered_pieces,
        ));
        assert!(rx.try_recv().is_err());

        assert!(push_stream_token_piece(
            &tx,
            "c".to_string(),
            2,
            &mut first_chunk_sent,
            &mut buffer,
            &mut buffered_pieces,
        ));
        match rx.try_recv() {
            Ok(GenerateEvent::Token(text)) => assert_eq!(text, "bc"),
            other => panic!("expected batched token event, got {other:?}"),
        }
    }

    #[test]
    fn flush_stream_token_batch_sends_remaining_buffer() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut buffer = "tail".to_string();
        let mut buffered_pieces = 1usize;

        assert!(flush_stream_token_batch(
            &tx,
            &mut buffer,
            &mut buffered_pieces
        ));

        match rx.try_recv() {
            Ok(GenerateEvent::Token(text)) => assert_eq!(text, "tail"),
            other => panic!("expected flushed token event, got {other:?}"),
        }
        assert!(buffer.is_empty());
        assert_eq!(buffered_pieces, 0);
    }

    #[test]
    fn developer_role_maps_to_system() {
        assert_eq!(parse_chat_role("developer").unwrap(), ChatRole::System);
    }

    #[test]
    fn normalized_messages_render_with_ax_core_chat() {
        let normalized = normalize_chat_messages(&[
            ChatMessage {
                role: "developer".into(),
                content: serde_json::Value::String("be precise".into()),
            },
            ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String("hi".into()),
            },
            ChatMessage {
                role: "assistant".into(),
                content: serde_json::Value::String("hello".into()),
            },
        ])
        .unwrap();
        let chat_messages: Vec<_> = normalized
            .iter()
            .map(|(role, content)| chat::ChatMessage::new(*role, content.as_str()))
            .collect();

        let prompt =
            chat::render_chat_messages(&chat_messages, "gemma3", ChatRenderOptions::default());
        assert!(prompt.contains("<start_of_turn>system\nbe precise<end_of_turn>\n"));
        assert!(prompt.contains("<start_of_turn>user\nhi<end_of_turn>\n"));
        assert!(prompt.contains("<start_of_turn>model\nhello<end_of_turn>\n"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn ax_core_mistral_template_uses_inst_format() {
        let prompt = super::render_chat_messages_with_compat(
            &[
                chat::ChatMessage::system("be concise"),
                chat::ChatMessage::user("hello"),
                chat::ChatMessage::assistant("hi"),
                chat::ChatMessage::user("again"),
            ],
            "mistral",
            ChatRenderOptions::default(),
        );
        assert_eq!(
            prompt,
            "[INST] <<SYS>>\nbe concise\n<</SYS>>\n\nhello [/INST] hi [INST] again [/INST]"
        );
    }

    #[test]
    fn tool_role_is_rejected_for_native_templates() {
        let err = normalize_chat_messages(&[ChatMessage {
            role: "tool".into(),
            content: serde_json::Value::String("result".into()),
        }])
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("does not support chat role 'tool'")
        );
    }

    #[test]
    fn non_text_chat_parts_are_rejected() {
        let err = normalize_chat_messages(&[ChatMessage {
            role: "user".into(),
            content: serde_json::json!([
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
            ]),
        }])
        .unwrap_err();

        assert!(err.to_string().contains("only supports text chat content"));
    }

    #[test]
    fn gguf_chat_template_markers_override_architecture_guess() {
        assert_eq!(
            infer_render_architecture("llama", Some("<s>[INST] {{ prompt }} [/INST]")),
            "mistral"
        );
        assert_eq!(
            infer_render_architecture("mistral", Some("<|im_start|>user\n{{ prompt }}<|im_end|>")),
            "qwen3"
        );
    }

    #[test]
    fn frequency_and_presence_penalties_are_allowed() {
        let params = GenerationParams {
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.25),
            ..Default::default()
        };
        assert!(ensure_supported_generation_params(&params).is_ok());
    }

    #[test]
    fn sampling_config_carries_upstream_penalty_fields() {
        let params = GenerationParams {
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.25),
            min_p: Some(0.1),
            ..Default::default()
        };
        let config = build_sampling_config(&params);
        assert!(config.logit_bias.is_empty());
        assert!(config.allowed_token_ids.is_empty());
        assert!(config.banned_token_ids.is_empty());
        assert_eq!(config.frequency_penalty, 0.5);
        assert_eq!(config.presence_penalty, 0.25);
        assert_eq!(config.min_p, 0.1);
        assert_eq!(config.min_keep, 1);
    }

    #[test]
    fn logprobs_are_allowed() {
        let params = GenerationParams {
            logprobs: Some(true),
            top_logprobs: Some(3),
            ..Default::default()
        };
        assert!(ensure_supported_generation_params(&params).is_ok());
    }
}
