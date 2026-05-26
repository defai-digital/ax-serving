//! AxEngineBackend: `InferenceBackend` implementation backed by ax-engine-sdk.
//!
//! AX Engine v4 exposes a session-oriented SDK. The native MLX runtime loads
//! AX model artifact directories (`model-manifest.json`, `config.json`,
//! `tokenizer.json`, and weights) instead of the old GGUF/internal-core API.

use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ax_engine_sdk::{
    CacheGroupId, EmbeddingPooling, EngineSession, EngineSessionConfig,
    GenerateFinishReason as AxGenerateFinishReason, GenerateRequest, GenerateSampling,
    GenerateStreamEvent as AxGenerateStreamEvent, GenerateStreamResponseEvent, KvCompressionConfig,
    PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier,
};
use rustc_hash::FxHashMap;
use serde_json::Value;
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    BackendType, ChatMessage, EmbedConfig, EmbedInput, EmbedResult, GenerateEvent, GenerateInput,
    GenerationParams, GenerationStats, InferenceBackend, LoadConfig, ModelHandle, ModelMetadata,
    ThermalMonitor, ThermalState, current_rss_bytes,
};

const DEFAULT_CONTEXT_LENGTH: u32 = 16 * 1024;
const DEFAULT_BLOCK_SIZE_TOKENS: u32 = 16;
const DEFAULT_MAX_BATCH_TOKENS: u32 = 2048;
const MODEL_MANIFEST_FILE: &str = "model-manifest.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CONFIG_FILE: &str = "config.json";

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(8_000_000);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_handle() -> ModelHandle {
    ModelHandle(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
}

fn next_request_id() -> u64 {
    NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed).max(1)
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
    session: Mutex<EngineSession>,
    tokenizer: Tokenizer,
    metadata: ModelMetadata,
    model_id: String,
    eos_tokens: Vec<u32>,
    render_architecture: String,
    embedding_pooling: EmbeddingPooling,
}

// SAFETY: AX Engine session access is serialized by `session`, and the other
// fields are immutable after load.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

#[derive(Clone, Copy)]
enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, PartialEq, Eq)]
struct StopPieceAction {
    emit: String,
    matched: bool,
}

fn resolve_model_dir(path: &Path) -> Result<PathBuf> {
    anyhow::ensure!(path.exists(), "model path not found: {}", path.display());

    if path.is_dir() {
        return Ok(path.to_path_buf());
    }

    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        anyhow::bail!(
            "native ax-engine v4.10.0 requires an AX MLX model artifact directory, not a .gguf file: {}",
            path.display()
        );
    }

    path.parent().map(Path::to_path_buf).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot resolve model artifact directory for {}",
            path.display()
        )
    })
}

pub fn is_ax_engine_model_artifacts(path: &Path) -> bool {
    let model_dir = if path.is_dir() {
        path
    } else if let Some(parent) = path.parent() {
        parent
    } else {
        return false;
    };

    model_dir.join(MODEL_MANIFEST_FILE).is_file() && model_dir.join(TOKENIZER_FILE).is_file()
}

fn ensure_model_artifacts(model_dir: &Path) -> Result<()> {
    for required in [MODEL_MANIFEST_FILE, TOKENIZER_FILE] {
        let path = model_dir.join(required);
        anyhow::ensure!(
            path.is_file(),
            "native ax-engine v4.10.0 model artifacts require {} at {}",
            required,
            path.display()
        );
    }
    Ok(())
}

fn read_json_file(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(value))
}

fn json_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
}

fn json_str(value: Option<&Value>) -> Option<&str> {
    value.and_then(Value::as_str).filter(|s| !s.is_empty())
}

fn model_id_from_dir(model_dir: &Path) -> String {
    model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("ax-engine-model")
        .to_string()
}

fn infer_render_architecture(model_architecture: &str, chat_template: Option<&str>) -> String {
    if let Some(chat_template) = chat_template {
        if chat_template.contains("<|start_header_id|>") {
            return "llama".to_string();
        }
        if chat_template.contains("<|im_start|>") {
            return "qwen".to_string();
        }
        if chat_template.contains("<start_of_turn>") {
            return "gemma".to_string();
        }
        if chat_template.contains("[INST]") && chat_template.contains("[/INST]") {
            return "mistral".to_string();
        }
    }

    match model_architecture {
        family if family.starts_with("qwen") => "qwen".to_string(),
        family if family.starts_with("gemma") => "gemma".to_string(),
        family if family.starts_with("llama") => "llama".to_string(),
        family if family.starts_with("mistral") || family.starts_with("mixtral") => {
            "mistral".to_string()
        }
        other => other.to_string(),
    }
}

fn tokenizer_chat_template(tokenizer_config: Option<&Value>) -> Option<&str> {
    tokenizer_config.and_then(|value| match value.get("chat_template") {
        Some(Value::String(template)) => Some(template.as_str()),
        Some(Value::Array(templates)) => templates
            .iter()
            .find_map(|entry| entry.get("template").and_then(Value::as_str)),
        _ => None,
    })
}

fn context_length_from_config(config_json: Option<&Value>, load_config: &LoadConfig) -> u32 {
    if load_config.context_length > 0 {
        return load_config.context_length;
    }

    let Some(config_json) = config_json else {
        return DEFAULT_CONTEXT_LENGTH;
    };

    [
        "max_position_embeddings",
        "model_max_length",
        "seq_length",
        "max_sequence_length",
    ]
    .into_iter()
    .find_map(|key| json_u32(config_json.get(key)))
    .unwrap_or(DEFAULT_CONTEXT_LENGTH)
}

fn eos_tokens_from_config(config_json: Option<&Value>, tokenizer: &Tokenizer) -> Vec<u32> {
    let from_config = config_json
        .and_then(|value| value.get("eos_token_id"))
        .map(|value| match value {
            Value::Number(_) => json_u32(Some(value)).into_iter().collect::<Vec<_>>(),
            Value::Array(items) => items
                .iter()
                .filter_map(|item| json_u32(Some(item)))
                .collect(),
            _ => Vec::new(),
        })
        .unwrap_or_default();

    if !from_config.is_empty() {
        return from_config;
    }

    ["<|endoftext|>", "<|im_end|>", "</s>", "<|eot_id|>"]
        .into_iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .next()
        .into_iter()
        .collect()
}

fn pooling_from_load_config(config: &LoadConfig) -> Result<EmbeddingPooling> {
    match config.pooling_type.as_deref().unwrap_or("mean") {
        "mean" | "none" => Ok(EmbeddingPooling::Mean),
        "last" => Ok(EmbeddingPooling::Last),
        "cls" => Ok(EmbeddingPooling::Cls),
        other => anyhow::bail!("native ax-engine backend does not support pooling type '{other}'"),
    }
}

fn metadata_from_artifacts(
    model_dir: &Path,
    config: &LoadConfig,
    load_time_ms: u64,
    peak_rss_bytes: u64,
) -> Result<(ModelMetadata, String)> {
    let manifest = read_json_file(&model_dir.join(MODEL_MANIFEST_FILE))?
        .context("missing model-manifest.json")?;
    let config_json = read_json_file(&model_dir.join(CONFIG_FILE))?;
    let config_json = config_json.as_ref();

    let architecture = json_str(manifest.get("model_family"))
        .or_else(|| config_json.and_then(|value| json_str(value.get("model_type"))))
        .unwrap_or("mlx")
        .to_string();
    let n_layers = json_u32(manifest.get("layer_count"))
        .or_else(|| config_json.and_then(|value| json_u32(value.get("num_hidden_layers"))))
        .unwrap_or(0);
    let n_heads = json_u32(manifest.get("attention_head_count"))
        .or_else(|| config_json.and_then(|value| json_u32(value.get("num_attention_heads"))))
        .unwrap_or(0);
    let n_kv_heads = json_u32(manifest.get("kv_head_count"))
        .or_else(|| config_json.and_then(|value| json_u32(value.get("num_key_value_heads"))))
        .unwrap_or(n_heads);
    let embedding_dim = json_u32(manifest.get("hidden_size"))
        .or_else(|| config_json.and_then(|value| json_u32(value.get("hidden_size"))))
        .unwrap_or(0);
    let vocab_size = json_u32(manifest.get("vocab_size"))
        .or_else(|| config_json.and_then(|value| json_u32(value.get("vocab_size"))))
        .unwrap_or(0);
    let context_length = context_length_from_config(config_json, config);

    Ok((
        ModelMetadata {
            architecture: architecture.clone(),
            n_layers,
            n_heads,
            n_kv_heads,
            embedding_dim,
            vocab_size,
            context_length,
            load_time_ms,
            peak_rss_bytes,
            resolved_backend: BackendType::Metal,
        },
        architecture,
    ))
}

fn session_config_for_model(
    model_dir: &Path,
    load_config: &LoadConfig,
) -> Result<EngineSessionConfig> {
    if load_config.backend_type == BackendType::Cpu {
        anyhow::bail!("native ax-engine v4.10.0 SDK integration supports MLX/Metal, not CPU");
    }

    let context_length = context_length_from_config(
        read_json_file(&model_dir.join(CONFIG_FILE))?.as_ref(),
        load_config,
    );
    let total_blocks = context_length.div_ceil(DEFAULT_BLOCK_SIZE_TOKENS).max(1);
    let max_batch_tokens = context_length.clamp(1, DEFAULT_MAX_BATCH_TOKENS);

    EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        cache_group_id: CacheGroupId(0),
        block_size_tokens: DEFAULT_BLOCK_SIZE_TOKENS,
        total_blocks,
        deterministic: true,
        max_batch_tokens,
        backend_request: PreviewBackendRequest::new(SupportTier::MlxPreview),
        mlx_runtime_artifacts_dir: None,
        mlx_model_artifacts_dir: Some(model_dir.to_path_buf()),
        mlx_disable_ngram_acceleration: false,
        mlx_kv_compression: KvCompressionConfig::disabled(),
        mlx_prefill_chunk: None,
    })
    .context("failed to build ax-engine session config")
}

fn extract_text(content: &Value) -> Result<String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let part_text = part
                            .get("text")
                            .and_then(Value::as_str)
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

fn render_user_prompt(text: &str, architecture: &str) -> String {
    render_chat_messages(&[(ChatRole::User, text.to_string())], architecture)
}

fn render_chat_messages(messages: &[(ChatRole, String)], architecture: &str) -> String {
    match architecture {
        "qwen" => render_qwen_chat(messages),
        "llama" => render_llama_chat(messages),
        "gemma" => render_gemma_chat(messages),
        "mistral" => render_inst_format_chat_messages(messages),
        _ => render_plain_chat(messages),
    }
}

fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    }
}

fn render_qwen_chat(messages: &[(ChatRole, String)]) -> String {
    let mut rendered = String::new();
    for (role, content) in messages {
        rendered.push_str("<|im_start|>");
        rendered.push_str(role_name(*role));
        rendered.push('\n');
        rendered.push_str(content);
        rendered.push_str("<|im_end|>\n");
    }
    rendered.push_str("<|im_start|>assistant\n");
    rendered
}

fn render_llama_chat(messages: &[(ChatRole, String)]) -> String {
    let mut rendered = String::new();
    for (role, content) in messages {
        rendered.push_str("<|start_header_id|>");
        rendered.push_str(role_name(*role));
        rendered.push_str("<|end_header_id|>\n\n");
        rendered.push_str(content);
        rendered.push_str("<|eot_id|>");
    }
    rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    rendered
}

fn render_gemma_chat(messages: &[(ChatRole, String)]) -> String {
    let mut rendered = String::new();
    for (role, content) in messages {
        let role = match role {
            ChatRole::System | ChatRole::User => "user",
            ChatRole::Assistant => "model",
        };
        rendered.push_str("<start_of_turn>");
        rendered.push_str(role);
        rendered.push('\n');
        rendered.push_str(content);
        rendered.push_str("<end_of_turn>\n");
    }
    rendered.push_str("<start_of_turn>model\n");
    rendered
}

fn render_inst_format_chat_messages(messages: &[(ChatRole, String)]) -> String {
    let mut rendered = String::new();
    let mut pending_system = None;

    for (role, content) in messages {
        match role {
            ChatRole::System => {
                pending_system = Some(content.as_str());
            }
            ChatRole::User => {
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str("[INST] ");
                if let Some(system) = pending_system.take() {
                    rendered.push_str("<<SYS>>\n");
                    rendered.push_str(system);
                    rendered.push_str("\n<</SYS>>\n\n");
                }
                rendered.push_str(content);
                rendered.push_str(" [/INST]");
            }
            ChatRole::Assistant => {
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str(content);
            }
        }
    }

    rendered
}

fn render_plain_chat(messages: &[(ChatRole, String)]) -> String {
    let mut rendered = String::new();
    for (role, content) in messages {
        match role {
            ChatRole::System => rendered.push_str("System: "),
            ChatRole::User => rendered.push_str("User: "),
            ChatRole::Assistant => rendered.push_str("Assistant: "),
        }
        rendered.push_str(content);
        rendered.push('\n');
    }
    rendered.push_str("Assistant:");
    rendered
}

fn encode_text(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;
    Ok(encoding.get_ids().to_vec())
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer
        .decode(tokens, false)
        .map_err(|err| anyhow::anyhow!("token decode failed: {err}"))
}

fn build_sampling(params: &GenerationParams) -> GenerateSampling {
    GenerateSampling {
        temperature: params.temperature.unwrap_or(0.0) as f32,
        top_p: params.top_p.unwrap_or(1.0) as f32,
        top_k: params.top_k.map(|value| value as u32).unwrap_or(0),
        min_p: params
            .min_p
            .map(|value| value as f32)
            .filter(|value| *value > 0.0),
        repetition_penalty: params.repeat_penalty.unwrap_or(1.0) as f32,
        repetition_context_size: Some(64),
        seed: params.seed.unwrap_or(0),
        deterministic: None,
        ignore_eos: false,
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
    if params.frequency_penalty.is_some_and(|value| value != 0.0) {
        unsupported.push("frequency_penalty");
    }
    if params.presence_penalty.is_some_and(|value| value != 0.0) {
        unsupported.push("presence_penalty");
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

fn build_generate_request(
    loaded: &LoadedModel,
    input: GenerateInput,
    params: &GenerationParams,
) -> Result<GenerateRequest> {
    let input_tokens = match input {
        GenerateInput::Tokens(tokens) => tokens,
        GenerateInput::Text(text) => encode_text(
            &loaded.tokenizer,
            &render_user_prompt(&text, &loaded.render_architecture),
            true,
        )?,
        GenerateInput::Chat(messages) => {
            let normalized = normalize_chat_messages(&messages)?;
            let rendered = render_chat_messages(&normalized, &loaded.render_architecture);
            encode_text(&loaded.tokenizer, &rendered, true)?
        }
    };

    anyhow::ensure!(
        !input_tokens.is_empty(),
        "empty token sequence after tokenization"
    );
    anyhow::ensure!(
        input_tokens.len() < loaded.metadata.context_length as usize,
        "input ({} tokens) exceeds context length ({})",
        input_tokens.len(),
        loaded.metadata.context_length
    );

    let max_output_tokens = params.max_tokens.unwrap_or(512).min(
        (loaded.metadata.context_length as usize)
            .saturating_sub(input_tokens.len())
            .max(1),
    ) as u32;

    Ok(GenerateRequest {
        model_id: loaded.model_id.clone(),
        input_tokens,
        input_text: None,
        max_output_tokens,
        sampling: build_sampling(params),
        stop_sequences: params.stop_seqs.clone(),
        metadata: None,
    })
}

fn finish_reason(reason: Option<AxGenerateFinishReason>, stopped_on_stop_sequence: bool) -> String {
    if stopped_on_stop_sequence {
        return "stop".to_string();
    }

    match reason {
        Some(AxGenerateFinishReason::MaxOutputTokens) => "length",
        Some(AxGenerateFinishReason::ContentFilter) => "content_filter",
        Some(AxGenerateFinishReason::Cancelled) => "stop",
        Some(AxGenerateFinishReason::Error) => "error",
        Some(AxGenerateFinishReason::Stop) | None => "stop",
    }
    .to_string()
}

fn stats_from_response(
    response: Option<&GenerateStreamResponseEvent>,
    prompt_tokens: usize,
    emitted_tokens: usize,
    started: Instant,
    first_token_at: Option<Instant>,
    stopped_on_stop_sequence: bool,
) -> GenerationStats {
    let elapsed = started.elapsed();
    let prefill_duration = first_token_at
        .map(|instant| instant.saturating_duration_since(started))
        .unwrap_or(elapsed);
    let decode_duration = first_token_at
        .map(|instant| {
            started
                .elapsed()
                .saturating_sub(instant.saturating_duration_since(started))
        })
        .unwrap_or(Duration::ZERO);
    let completion_tokens = response
        .and_then(|event| event.response.known_output_token_count())
        .map(|count| count as usize)
        .unwrap_or(emitted_tokens);
    let prompt_tokens = response
        .and_then(|event| event.response.known_prompt_token_count())
        .map(|count| count as usize)
        .unwrap_or(prompt_tokens);

    GenerationStats {
        prompt_tokens,
        completion_tokens,
        prefill_tok_per_sec: if prefill_duration.as_secs_f64() > 0.0 {
            prompt_tokens as f64 / prefill_duration.as_secs_f64()
        } else {
            0.0
        },
        decode_tok_per_sec: if decode_duration.as_secs_f64() > 0.0 {
            completion_tokens as f64 / decode_duration.as_secs_f64()
        } else {
            0.0
        },
        stop_reason: finish_reason(
            response.and_then(|event| event.response.finish_reason),
            stopped_on_stop_sequence,
        ),
    }
}

fn run_generate(
    loaded: Arc<LoadedModel>,
    input: GenerateInput,
    params: GenerationParams,
    tx: tokio::sync::mpsc::Sender<GenerateEvent>,
) -> Result<()> {
    ensure_supported_generation_params(&params)?;
    let request = build_generate_request(&loaded, input, &params)?;
    let prompt_tokens = request.input_tokens.len();
    let request_id = next_request_id();
    let emit_logprobs = params.logprobs.unwrap_or(false);
    let top_logprobs = params.top_logprobs.unwrap_or(0);
    let stream_batch_size = crate::stream_token_batch_size();

    let mut session = loaded.session.lock().unwrap_or_else(|err| {
        warn!("ax-engine session mutex poisoned; recovering from poisoned state");
        err.into_inner()
    });
    let mut state = session
        .stream_generate_state_with_request_id(request_id, request)
        .context("ax-engine failed to start generation stream")?;

    let started = Instant::now();
    let mut first_token_at = None;
    let mut emitted_tokens = 0usize;
    let mut stop_buffer = String::new();
    let mut stopped_on_stop_sequence = false;
    let mut first_stream_chunk_sent = false;
    let mut stream_token_buffer = String::new();
    let mut buffered_stream_tokens = 0usize;
    let mut response = None;

    loop {
        let Some(event) = session
            .next_stream_event(&mut state)
            .context("ax-engine generation stream failed")?
        else {
            break;
        };

        match event {
            AxGenerateStreamEvent::Request(_) => {}
            AxGenerateStreamEvent::Step(step) => {
                for (idx, token) in step.delta_tokens.iter().copied().enumerate() {
                    emitted_tokens += 1;
                    first_token_at.get_or_insert_with(Instant::now);

                    let piece = decode_tokens(&loaded.tokenizer, &[token])?;
                    let action = consume_stop_piece(&mut stop_buffer, &piece, &params.stop_seqs);
                    if emit_logprobs {
                        if !action.emit.is_empty()
                            && tx.blocking_send(GenerateEvent::Token(action.emit)).is_err()
                        {
                            return Ok(());
                        }
                        let logprob = step
                            .delta_token_logprobs
                            .get(idx)
                            .and_then(|value| *value)
                            .unwrap_or(0.0);
                        if tx
                            .blocking_send(GenerateEvent::TokenLogprob {
                                logprob,
                                top: Vec::with_capacity(top_logprobs as usize),
                            })
                            .is_err()
                        {
                            return Ok(());
                        }
                    } else if !push_stream_token_piece(
                        &tx,
                        action.emit,
                        stream_batch_size,
                        &mut first_stream_chunk_sent,
                        &mut stream_token_buffer,
                        &mut buffered_stream_tokens,
                    ) {
                        return Ok(());
                    }

                    if action.matched {
                        stopped_on_stop_sequence = true;
                        let _ = session.cancel_request(request_id);
                        break;
                    }
                }
                if stopped_on_stop_sequence {
                    break;
                }
            }
            AxGenerateStreamEvent::Response(event) => {
                response = Some(event);
                break;
            }
        }
    }

    if !stopped_on_stop_sequence
        && !push_stream_token_piece(
            &tx,
            std::mem::take(&mut stop_buffer),
            stream_batch_size,
            &mut first_stream_chunk_sent,
            &mut stream_token_buffer,
            &mut buffered_stream_tokens,
        )
    {
        return Ok(());
    }
    if !flush_stream_token_batch(&tx, &mut stream_token_buffer, &mut buffered_stream_tokens) {
        return Ok(());
    }

    let stats = stats_from_response(
        response.as_ref(),
        prompt_tokens,
        emitted_tokens,
        started,
        first_token_at,
        stopped_on_stop_sequence,
    );
    let _ = tx.blocking_send(GenerateEvent::Done(stats));
    Ok(())
}

/// `InferenceBackend` implementation backed by ax-engine-sdk sessions.
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
        let model_dir = resolve_model_dir(path)?;
        ensure_model_artifacts(&model_dir)?;
        crate::memory::check_memory_budget(0)?;

        let tokenizer = Tokenizer::from_file(model_dir.join(TOKENIZER_FILE))
            .map_err(|err| anyhow::anyhow!("ax-engine failed to load tokenizer.json: {err}"))?;
        let tokenizer_config = read_json_file(&model_dir.join("tokenizer_config.json"))?;
        let session_config = session_config_for_model(&model_dir, &config)?;
        let embedding_pooling = pooling_from_load_config(&config)?;

        if config.enable_embeddings == Some(true) {
            // Validate early enough that unsupported models fail during load.
            // The SDK still owns the authoritative support check at embed time.
            let _ = embedding_pooling;
        }

        let session =
            EngineSession::new(session_config).context("ax-engine failed to load model")?;
        let rss_after = current_rss_bytes();
        let (metadata, architecture) = metadata_from_artifacts(
            &model_dir,
            &config,
            started.elapsed().as_millis() as u64,
            rss_after.saturating_sub(rss_before),
        )?;
        let render_architecture = infer_render_architecture(
            &architecture,
            tokenizer_chat_template(tokenizer_config.as_ref()),
        );
        let eos_tokens = eos_tokens_from_config(
            read_json_file(&model_dir.join(CONFIG_FILE))?.as_ref(),
            &tokenizer,
        );
        let model_id = model_id_from_dir(&model_dir);

        let handle = next_handle();
        let loaded = Arc::new(LoadedModel {
            session: Mutex::new(session),
            tokenizer,
            metadata: metadata.clone(),
            model_id,
            eos_tokens,
            render_architecture,
            embedding_pooling,
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
        encode_text(&self.get_model(handle)?.tokenizer, text, add_bos)
    }

    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<String> {
        decode_tokens(&self.get_model(handle)?.tokenizer, tokens)
    }

    fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
        let tokens = self.get_model(handle)?.eos_tokens.clone();
        anyhow::ensure!(
            !tokens.is_empty(),
            "ax-engine tokenizer/config did not expose an EOS token"
        );
        Ok(tokens)
    }

    fn thermal_state(&self) -> ThermalState {
        self.thermal.current()
    }

    fn recommended_concurrency(&self) -> usize {
        self.thermal.recommended_concurrency()
    }

    fn embed(
        &self,
        handle: ModelHandle,
        inputs: &EmbedInput<'_>,
        config: &EmbedConfig,
    ) -> Result<EmbedResult> {
        let loaded = self.get_model(handle)?;
        let mut batch = match inputs {
            EmbedInput::Strings(strings) => strings
                .iter()
                .map(|text| encode_text(&loaded.tokenizer, text, true))
                .collect::<Result<Vec<_>>>()?,
            EmbedInput::Tokens(tokens) => tokens.to_vec(),
        };
        if config.truncate {
            for tokens in &mut batch {
                tokens.truncate(loaded.metadata.context_length as usize);
            }
        }
        let prompt_tokens = saturating_token_count(batch.iter().map(Vec::len));
        let session = loaded.session.lock().unwrap_or_else(|err| {
            warn!("ax-engine session mutex poisoned during embed; recovering");
            err.into_inner()
        });
        let embeddings = session
            .embed_batch(&batch, loaded.embedding_pooling, config.normalize)
            .context("ax-engine embedding failed")?;
        Ok(EmbedResult {
            embeddings,
            prompt_tokens,
        })
    }

    fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> Result<u32> {
        anyhow::ensure!(!tokens.is_empty(), "eval_tokens: empty input");
        let loaded = self.get_model(handle)?;
        let request = GenerateRequest {
            model_id: loaded.model_id.clone(),
            input_tokens: tokens.to_vec(),
            input_text: None,
            max_output_tokens: 1,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        };
        let mut session = loaded.session.lock().unwrap_or_else(|err| {
            warn!("ax-engine session mutex poisoned during eval; recovering");
            err.into_inner()
        });
        let response = session
            .generate_with_request_id(next_request_id(), request)
            .context("ax-engine eval generation failed")?;
        response
            .output_tokens
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("ax-engine eval did not produce a token"))
    }
}

fn saturating_token_count(lengths: impl IntoIterator<Item = usize>) -> u32 {
    lengths
        .into_iter()
        .map(|len| len.min(u32::MAX as usize) as u32)
        .fold(0u32, u32::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::{
        BackendType, ChatMessage, ChatRole, GenerationParams, LoadConfig, StopPieceAction,
        consume_stop_piece, ensure_supported_generation_params, infer_render_architecture,
        normalize_chat_messages, parse_chat_role, render_chat_messages, saturating_token_count,
        session_config_for_model,
    };

    #[test]
    fn gguf_native_loads_are_rejected_for_v4_sdk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, "").unwrap();

        let err = super::resolve_model_dir(&path).unwrap_err();
        assert!(
            err.to_string()
                .contains("requires an AX MLX model artifact directory")
        );
    }

    #[test]
    fn uppercase_gguf_native_loads_are_rejected_for_v4_sdk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.GGUF");
        std::fs::write(&path, "").unwrap();

        let err = super::resolve_model_dir(&path).unwrap_err();
        assert!(
            err.to_string()
                .contains("requires an AX MLX model artifact directory")
        );
    }

    #[test]
    fn cpu_backend_is_rejected_for_v4_native_session() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let cfg = LoadConfig {
            backend_type: BackendType::Cpu,
            ..Default::default()
        };
        let err = session_config_for_model(dir.path(), &cfg).unwrap_err();
        assert!(err.to_string().contains("supports MLX/Metal"));
    }

    #[test]
    fn stop_sequence_piece_suppresses_matched_suffix() {
        let mut pending = String::new();
        let stop = vec!["END".to_string()];

        let first = consume_stop_piece(&mut pending, "hello EN", &stop);
        assert_eq!(
            first,
            StopPieceAction {
                emit: "hello ".to_string(),
                matched: false,
            }
        );
        assert_eq!(pending, "EN");

        let second = consume_stop_piece(&mut pending, "D", &stop);
        assert_eq!(
            second,
            StopPieceAction {
                emit: String::new(),
                matched: true,
            }
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn stop_sequence_piece_flushes_buffer_when_no_match_occurs() {
        let mut pending = String::new();
        let stop = vec!["END".to_string()];

        let action = consume_stop_piece(&mut pending, "hello world", &stop);
        assert_eq!(action.emit, "hello wor");
        assert!(!action.matched);
        assert_eq!(pending, "ld");
    }

    #[test]
    fn unsupported_generation_features_are_reported_together() {
        let params = GenerationParams {
            grammar: Some("root ::= \"x\"".to_string()),
            response_format: Some("json_object".to_string()),
            frequency_penalty: Some(0.2),
            ..Default::default()
        };

        let err = ensure_supported_generation_params(&params).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("grammar"));
        assert!(message.contains("response_format"));
        assert!(message.contains("frequency_penalty"));
    }

    #[test]
    fn chat_role_parsing_accepts_supported_roles() {
        assert!(matches!(
            parse_chat_role("system").unwrap(),
            ChatRole::System
        ));
        assert!(matches!(
            parse_chat_role("developer").unwrap(),
            ChatRole::System
        ));
        assert!(matches!(parse_chat_role("user").unwrap(), ChatRole::User));
        assert!(matches!(
            parse_chat_role("assistant").unwrap(),
            ChatRole::Assistant
        ));
    }

    #[test]
    fn chat_text_parts_are_normalized() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::json!([
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ]),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let normalized = normalize_chat_messages(&messages).unwrap();
        assert_eq!(normalized.len(), 1);
        assert!(matches!(normalized[0].0, ChatRole::User));
        assert_eq!(normalized[0].1, "hello world");
    }

    #[test]
    fn embedding_prompt_token_count_saturates() {
        let count = saturating_token_count([u32::MAX as usize, 10]);

        assert_eq!(count, u32::MAX);
    }

    #[test]
    fn render_architecture_uses_template_markers() {
        assert_eq!(
            infer_render_architecture("unknown", Some("<|im_start|>")),
            "qwen"
        );
        assert_eq!(
            infer_render_architecture("unknown", Some("<start_of_turn>")),
            "gemma"
        );
        assert_eq!(
            infer_render_architecture("unknown", Some("<|start_header_id|>")),
            "llama"
        );
    }

    #[test]
    fn qwen_chat_renderer_adds_generation_prompt() {
        let rendered = render_chat_messages(&[(ChatRole::User, "hello".to_string())], "qwen");
        assert!(rendered.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }
}
