//! OpenAI API request / response schema types.
//!
//! Covers the OpenAI v1 schema surface needed for ax-serving.
//! Unknown fields are silently ignored (no `deny_unknown_fields`).

use serde::{Deserialize, Serialize};

// ── Input validation limits ────────────────────────────────────────────────────

pub const MAX_MESSAGES: usize = 100;
pub const MAX_CONTENT_BYTES: usize = 32 * 1024; // 32 KB per message
pub const MAX_MAX_TOKENS: u32 = 32_768;
pub const MAX_MODEL_ID_BYTES: usize = 256;

use crate::cache::CachePreference;

// ── Message content (string or multipart for vision) ──────────────────────────

/// Message content — either a plain string or a multipart array (vision).
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Flatten to a plain string (image parts are skipped).
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Byte length for validation (image parts count as ~256 bytes each).
    pub fn byte_len(&self) -> usize {
        match self {
            MessageContent::Text(s) => s.len(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::ImageUrl { .. } => 256,
                })
                .sum(),
        }
    }

    /// True if there is at least one image_url part.
    pub fn has_images(&self) -> bool {
        matches!(self, MessageContent::Parts(parts) if parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })))
    }
}

/// A single part of a multipart message (text or image URL).
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

// ── Input message (request) vs output message (response) ─────────────────────

/// Message in a chat request — content may be string or multipart.
#[derive(Debug, Deserialize, Clone)]
pub struct InputMessage {
    pub role: String,
    pub content: MessageContent,
    /// Optional display name for the speaker (used in system prompts).
    #[serde(default)]
    pub name: Option<String>,
}

/// Message in a chat response — content is always a plain string.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Tool calls emitted by the model (non-streaming, per OpenAI spec).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── Tool calling ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Tool {
    /// Always `"function"` in current OpenAI spec.
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema object describing the function parameters.
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// A tool call emitted by the model in a response.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    /// Position of this tool call in the `tool_calls` array (required by OpenAI spec).
    pub index: u32,
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

// ── Response format ───────────────────────────────────────────────────────────

/// `response_format` parameter — `{"type": "json_object"}` or `{"type": "text"}`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

// ── Requests ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<InputMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// Stop sequences (string or array of strings).
    #[serde(default)]
    pub stop: Option<StopSeqs>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Raw BNF grammar string for constrained generation.
    #[serde(default)]
    pub grammar: Option<String>,
    /// `{"type": "json_object"}` enforces JSON grammar output.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Mirostat mode: 0=off, 1=v1, 2=v2.
    #[serde(default)]
    pub mirostat: Option<u8>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    /// Tool definitions for function calling.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// `"auto"`, `"none"`, `"required"`, or `{"type":"function","function":{"name":"..."}}`.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub cache: Option<CachePreference>,
    #[serde(default)]
    pub cache_ttl: Option<String>,
    /// Whether to return log probabilities of output tokens.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return per token (0–20). Requires `logprobs: true`.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
}

/// Stop sequences — either a single string or an array.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum StopSeqs {
    One(String),
    Many(Vec<String>),
}

impl StopSeqs {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSeqs::One(s) => vec![s],
            StopSeqs::Many(v) => v,
        }
    }
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_repeat_penalty() -> f32 {
    1.1
}

// ── Finish reason ─────────────────────────────────────────────────────────────

/// Why the model stopped generating tokens.
///
/// Serialized as lowercase snake_case to match the OpenAI wire format.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model reached a natural stop point or stop sequence.
    Stop,
    /// `max_tokens` was reached before a natural stop.
    Length,
    /// Model called a tool.
    ToolCalls,
    /// Output omitted by a content filter.
    ContentFilter,
}

// ── Response ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ── Logprobs ──────────────────────────────────────────────────────────────────

/// One entry in a top_logprobs list.
#[derive(Debug, Serialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Per-token logprob content (OpenAI logprobs spec).
#[derive(Debug, Serialize, Clone)]
pub struct LogprobContent {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// `choices[i].logprobs` container.
#[derive(Debug, Serialize, Clone)]
pub struct ChoiceLogprobs {
    pub content: Vec<LogprobContent>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: Option<Message>,
    pub delta: Option<Delta>,
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Models list ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

// ── Model management ──────────────────────────────────────────────────────────

/// POST /v1/models — load a model from a GGUF file.
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    /// Model ID to register (1–128 chars, alphanumeric/dash/underscore/dot).
    pub model_id: String,
    /// Absolute or relative path to the `.gguf` file.
    pub path: String,
    /// Override context length (0 / omit = use model default).
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Explicit backend: `"llama_cpp"` | `"lib_llama"` | `"native"` | `"auto"`.
    /// Omit to use the routing config (`backends.yaml`).
    #[serde(default)]
    pub backend: Option<String>,
    /// Path to a multimodal projector (`.gguf`) for vision models (LLaVA etc.).
    /// Passed as `--mmproj` to llama-server when using the llama_cpp backend.
    #[serde(default)]
    pub mmproj_path: Option<String>,
    /// Override GPU layer count (`--n-gpu-layers`).  `None` = backend default.
    #[serde(default)]
    pub n_gpu_layers: Option<i32>,
    /// Force llama-server embedding mode (`--embedding`).
    ///
    /// `None` = auto-detect from GGUF pooling metadata.
    #[serde(default)]
    pub enable_embeddings: Option<bool>,
    /// Override llama-server pooling mode (`--pooling`).
    ///
    /// Typical values: `"none"`, `"mean"`, `"cls"`, `"last"`, `"rank"`.
    #[serde(default)]
    pub pooling_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    pub ready: bool,
    pub model_available: bool,
    pub loaded_model_count: usize,
    pub architecture: String,
    pub context_length: u32,
    pub load_time_ms: u64,
}

/// DELETE /v1/models/:id — unload a model.
#[derive(Debug, Serialize)]
pub struct UnloadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    pub ready: bool,
    pub model_available: bool,
    pub loaded_model_count: usize,
}

/// POST /v1/models/:id/reload — atomically reload from same path and config.
#[derive(Debug, Serialize)]
pub struct ReloadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    pub ready: bool,
    pub model_available: bool,
    pub loaded_model_count: usize,
    pub architecture: String,
    pub load_time_ms: u64,
}

// ── Text completions (POST /v1/completions) ───────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default)]
    pub stop: Option<StopSeqs>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub mirostat: Option<u8>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    /// Whether to return log probabilities of output tokens.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return per token (0–20). Requires `logprobs: true`.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Response cache control (`"enable"` or `"disable"`).  Non-streaming only.
    #[serde(default)]
    pub cache: Option<CachePreference>,
    /// Custom TTL for this entry (e.g. `"5m"`, `"1h"`).
    #[serde(default)]
    pub cache_ttl: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

// ── Embeddings (POST /v1/embeddings) ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    /// Single string, array of strings, single token array, or array of token arrays.
    pub input: EmbeddingsInput,
    /// `"float"` (default) or `"base64"`.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// L2-normalise embeddings (default: `true`).
    #[serde(default)]
    pub normalize: Option<bool>,
    /// Truncate inputs exceeding context length (default: `true`).
    #[serde(default)]
    pub truncate: Option<bool>,
    /// Reduce output dimensionality (OpenAI extension, accepted but ignored).
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// Mirrors llama-server's `input` field: string, string[], int[], or int[][].
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    /// Single pre-tokenized sequence.
    OneTokens(Vec<u32>),
    /// Multiple pre-tokenized sequences.
    ManyTokens(Vec<Vec<u32>>),
    /// Single text string.
    One(String),
    /// Multiple text strings.
    Many(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub model: String,
    pub data: Vec<EmbeddingObject>,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub index: u32,
    /// Float array or base64-encoded string depending on `encoding_format`.
    pub embedding: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ── Health ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub ready: bool,
    pub model_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    pub thermal: String,
    pub loaded_models: Vec<String>,
    pub loaded_model_count: usize,
    pub uptime_secs: u64,
}
