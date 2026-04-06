//! ax-serving-engine: Inference backend adapter.
//!
//! Wraps the native ax-engine backend, llama.cpp, and mlx-lm behind the
//! [`InferenceBackend`] trait.
//! All other crates depend on this trait, not on a concrete inference engine.
//!
//! # Architecture
//!
//! ```text
//! ax-serving-api / ax-serving-shim
//!      │  InferenceBackend trait
//!      ▼
//! AxEngineBackend / LlamaCppBackend / MlxBackend / LibLlamaBackend
//! ```

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-serving-engine only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub mod ax_engine;
pub mod gguf_meta;
#[cfg(feature = "libllama")]
pub mod libllama;
pub mod llamacpp;
pub mod memory;
pub mod mlx;
pub mod routing;
pub mod thermal;

use std::path::Path;

pub use ax_engine_core::metrics::current_rss_bytes;
pub use ax_engine::AxEngineBackend;
#[cfg(feature = "libllama")]
pub use libllama::LibLlamaBackend;
pub use llamacpp::{LlamaCppBackend, LlamaCppConfig};
pub use mlx::{MlxBackend, MlxConfig, is_mlx_model};
pub use routing::{BackendChoice, RouterBackend, RoutingConfig};
pub use thermal::{ThermalMonitor, ThermalState};

// ── Public types ──────────────────────────────────────────────────────────────

/// Opaque handle identifying a loaded model in the backend.
///
/// The inner `u64` is public to allow construction in integration tests and
/// the C API shim. Production code should treat the value as opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelHandle(pub u64);

/// Configuration for loading a model.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Override context length (0 = use model default).
    pub context_length: u32,
    /// Backend selection.
    pub backend_type: BackendType,
    /// Optional llama.cpp `--n-gpu-layers` override.
    ///
    /// `None` means backend default behavior. Typical values:
    /// - `0`  => CPU-only
    /// - `-1` => all layers on GPU (llama.cpp semantics)
    /// - `N`  => first N layers on GPU
    pub llama_cpp_n_gpu_layers: Option<i32>,
    /// Path to a multimodal projector file for vision models (e.g. LLaVA).
    /// Forwarded as `--mmproj <path>` to llama-server.
    pub mmproj_path: Option<String>,
    /// Per-load backend routing override. When `Some`, overrides the global
    /// `backends.yaml` routing config for this specific load call.
    ///
    /// Accepted values: `"llama_cpp"`, `"native"`, `"lib_llama"`, `"auto"`.
    /// `None` = use routing config (`backends.yaml`).
    pub backend_hint: Option<String>,
    /// Explicitly enable embedding mode for llama-server (`--embedding` flag).
    ///
    /// `None` = auto-detect from GGUF `pooling_type` metadata (non-zero → enabled).
    /// `Some(true)` = always enable (useful if pooling_type is absent from GGUF).
    /// `Some(false)` = always disable.
    pub enable_embeddings: Option<bool>,
    /// Override llama-server pooling mode (`--pooling`).
    ///
    /// Common values: `"none"`, `"mean"`, `"cls"`, `"last"`, `"rank"`.
    /// `None` means use model default pooling behavior.
    pub pooling_type: Option<String>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            context_length: 0,
            backend_type: BackendType::Auto,
            llama_cpp_n_gpu_layers: None,
            mmproj_path: None,
            backend_hint: None,
            enable_embeddings: None,
            pooling_type: None,
        }
    }
}

/// Backend hardware selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    #[default]
    Auto,
    Metal,
    Cpu,
}

/// Metadata returned after loading a model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub architecture: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub embedding_dim: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub load_time_ms: u64,
    pub peak_rss_bytes: u64,
    /// The backend hardware type that was actually used to load the model.
    ///
    /// Set by each backend implementation (`AxEngineBackend` → `Metal`/`Cpu`,
    /// `LlamaCppBackend` → `Metal` when GPU layers > 0 else `Cpu`,
    /// `LibLlamaBackend` → `Metal`).  Allows callers to report the resolved
    /// backend rather than echoing the client-supplied `Auto` hint.
    pub resolved_backend: BackendType,
}

impl ModelMetadata {
    /// Estimate the KV cache memory usage in bytes for the full context length.
    ///
    /// Assumes BF16 (2 bytes/element) for K and V tensors across all layers.
    /// Returns 0 if any required field is zero (metadata not fully populated).
    pub fn estimated_kv_bytes(&self) -> u64 {
        if self.n_layers == 0
            || self.n_kv_heads == 0
            || self.n_heads == 0
            || self.embedding_dim == 0
        {
            return 0;
        }
        let head_dim = (self.embedding_dim / self.n_heads) as u64;
        // 2 = K+V, 2 bytes per BF16 element
        let kv_per_layer = 2 * self.n_kv_heads as u64 * head_dim * self.context_length as u64 * 2;
        self.n_layers as u64 * kv_per_layer
    }
}

// ── Generation types ──────────────────────────────────────────────────────────

/// Input for a generation request.
#[derive(Debug, Clone)]
pub enum GenerateInput {
    /// Pre-tokenized input. Goes directly to the model as `CompletionTokens`.
    Tokens(Vec<u32>),
    /// Raw text prompt. Wrapped as a single user message and uses model chat template.
    Text(String),
    /// Pre-structured chat messages. Uses model chat template directly.
    Chat(Vec<ChatMessage>),
}

/// A single chat message.
///
/// `content` is a JSON value so it can carry either a plain string or a
/// multipart array (vision) — matching llama-server's `/v1/chat/completions`
/// message format exactly.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// Plain string or multipart array (e.g. `[{"type":"text",...},{"type":"image_url",...}]`).
    pub content: serde_json::Value,
}

/// Parameters controlling the generation sampler.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Stream incremental token deltas (`true`) or request a buffered response (`false`).
    pub stream: bool,
    /// Sampling temperature (None = greedy / deterministic).
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling (None = disabled).
    pub top_p: Option<f64>,
    /// Min-p sampling threshold relative to the most likely token (None = disabled).
    pub min_p: Option<f64>,
    /// Top-k sampling (None = disabled).
    pub top_k: Option<usize>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
    /// Stop sequences that halt generation.
    pub stop_seqs: Vec<String>,
    /// RNG seed for reproducible outputs (None = random).
    pub seed: Option<u64>,
    /// Repetition penalty (None = backend default; 1.0 = disabled).
    pub repeat_penalty: Option<f64>,
    /// Frequency penalty — reduces repeated tokens proportional to frequency (OpenAI).
    pub frequency_penalty: Option<f64>,
    /// Presence penalty — flat penalty for any previously seen token (OpenAI).
    pub presence_penalty: Option<f64>,
    /// BNF grammar string for constrained generation (llama.cpp `grammar` field).
    pub grammar: Option<String>,
    /// Response format: `"json_object"` enforces JSON grammar; `"text"` is default.
    pub response_format: Option<String>,
    /// Mirostat sampling mode: 0 = off, 1 = Mirostat v1, 2 = Mirostat v2.
    pub mirostat: Option<u8>,
    /// Mirostat target perplexity tau (default 5.0).
    pub mirostat_tau: Option<f64>,
    /// Mirostat learning rate eta (default 0.1).
    pub mirostat_eta: Option<f64>,
    /// Return log probabilities of output tokens.
    pub logprobs: Option<bool>,
    /// Number of top log probabilities per token (0–20, requires `logprobs: true`).
    pub top_logprobs: Option<u32>,
    /// Tool definitions forwarded verbatim to the backend (OpenAI tools array).
    pub tools: Option<serde_json::Value>,
    /// Tool choice forwarded verbatim to the backend (`"auto"`, `"none"`, or a specific tool).
    pub tool_choice: Option<serde_json::Value>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            stream: false,
            temperature: Some(0.7),
            top_p: None,
            min_p: None,
            top_k: None,
            max_tokens: None,
            stop_seqs: Vec::new(),
            seed: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            grammar: None,
            response_format: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
        }
    }
}

/// Usage statistics returned in [`GenerateEvent::Done`].
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prefill throughput in tokens/second.
    pub prefill_tok_per_sec: f64,
    /// Decode throughput in tokens/second.
    pub decode_tok_per_sec: f64,
    /// OpenAI finish_reason string: `"stop"` (default), `"length"`, `"tool_calls"`,
    /// or `"content_filter"`. Empty string is treated as `"stop"` by callers.
    pub stop_reason: String,
}

/// Events emitted during generation, received via the mpsc channel.
#[derive(Debug)]
pub enum GenerateEvent {
    /// A decoded token text piece.
    Token(String),
    /// Log probability for the most recently emitted `Token`.
    ///
    /// Immediately follows each `Token` event when `logprobs: true` was
    /// requested. Consumers that don't need logprobs can ignore this variant.
    TokenLogprob {
        /// Log probability (natural log) of the sampled token.
        logprob: f32,
        /// Top-N alternatives: `(token_text, logprob)` ordered by descending probability.
        /// Empty when `top_logprobs` was not requested.
        top: Vec<(String, f32)>,
    },
    /// A tool call emitted by the model (function calling).
    ToolCall {
        /// Auto-generated call ID (e.g. `call_abc123`).
        id: String,
        /// Function name the model wants to invoke.
        name: String,
        /// JSON-encoded arguments string.
        arguments: String,
    },
    /// Generation finished — statistics available.
    Done(GenerationStats),
    /// Unrecoverable error during generation.
    Error(String),
}

// ── Embedding types ───────────────────────────────────────────────────────────

/// Input to [`InferenceBackend::embed`]: either text strings or pre-tokenized
/// integer sequences (mirrors llama-server's `/v1/embeddings` `input` field).
pub enum EmbedInput<'a> {
    Strings(&'a [String]),
    /// Pre-tokenized sequences — each inner Vec is one sequence.
    Tokens(&'a [Vec<u32>]),
}

/// Tuning parameters forwarded to the embedding backend.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// L2-normalise each embedding vector (default: `true`, matches llama-server).
    pub normalize: bool,
    /// Truncate inputs that exceed the model's context window (default: `true`).
    pub truncate: bool,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            truncate: true,
        }
    }
}

/// Result returned by [`InferenceBackend::embed`].
pub struct EmbedResult {
    /// One float vector per input (same order as input).
    pub embeddings: Vec<Vec<f32>>,
    /// Total prompt tokens processed (0 if the backend cannot report it).
    pub prompt_tokens: u32,
}

// ── Cache telemetry ──────────────────────────────────────────────────────────

/// Cache and batch telemetry reported by a backend for scheduling decisions.
///
/// All fields default to 0 meaning "unknown / not supported". Callers must
/// treat 0 as "no data" and degrade gracefully.
#[derive(Debug, Clone, Default)]
pub struct CacheTelemetry {
    /// KV cache pages currently allocated.
    pub kv_pages_used: u64,
    /// KV cache page budget (0 = unknown).
    pub kv_pages_total: u64,
    /// Tokens in reusable prefix cache (0 = unsupported).
    pub prefix_reusable_tokens: u64,
    /// Current internal batch occupancy.
    pub active_batch_size: u32,
    /// Backend's max batch capacity (0 = unknown).
    pub max_batch_size: u32,
}

// ── InferenceBackend trait ────────────────────────────────────────────────────

/// Core inference interface implemented by concrete backend adapters.
///
/// All serving-layer logic (gRPC, REST, C API) depends only on this trait.
/// The underlying inference engine is not visible outside `ax-serving-engine`.
pub trait InferenceBackend: Send + Sync {
    /// Load a GGUF model and return an opaque handle.
    fn load_model(
        &self,
        path: &Path,
        config: LoadConfig,
    ) -> anyhow::Result<(ModelHandle, ModelMetadata)>;

    /// Unload a previously loaded model, freeing GPU/CPU memory.
    fn unload_model(&self, handle: ModelHandle) -> anyhow::Result<()>;

    /// Begin generating tokens, streaming events via `tx`.
    ///
    /// Returns immediately; the actual generation runs on the backend's
    /// internal tokio runtime. Callers receive events via `rx`.
    fn generate(
        &self,
        handle: ModelHandle,
        input: GenerateInput,
        params: GenerationParams,
        tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    ) -> anyhow::Result<()>;

    /// Encode text to token IDs.
    fn tokenize(&self, handle: ModelHandle, text: &str, add_bos: bool) -> anyhow::Result<Vec<u32>>;

    /// Decode token IDs to a UTF-8 string.
    fn decode_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> anyhow::Result<String>;

    /// Returns the EOS token IDs for the loaded model.
    fn eos_tokens(&self, handle: ModelHandle) -> anyhow::Result<Vec<u32>>;

    /// Current thermal state (affects concurrency recommendations).
    fn thermal_state(&self) -> ThermalState;

    /// Recommended concurrency level given current thermal state.
    fn recommended_concurrency(&self) -> usize;

    /// Return cache and batch telemetry for scheduling decisions.
    ///
    /// Default returns all zeros (unknown). Backends that can report KV
    /// utilization or batch occupancy should override this.
    fn cache_telemetry(&self) -> CacheTelemetry {
        CacheTelemetry::default()
    }

    /// Return the concrete backend currently serving a given outer handle.
    ///
    /// Implemented by `RouterBackend` to distinguish native vs `llama.cpp`.
    ///
    /// Returns `Some("native")`, `Some("llama_cpp")`, or `Some("lib_llama")`.
    /// Backends that cannot resolve handle ownership return `None`.
    fn backend_name_for_handle(&self, _handle: ModelHandle) -> Option<&'static str> {
        None
    }

    /// Embed inputs and return one float vector per input.
    ///
    /// Model must be loaded with embeddings enabled. Default implementation
    /// returns an error — only backends that support embeddings override this.
    fn embed(
        &self,
        handle: ModelHandle,
        inputs: &EmbedInput<'_>,
        config: &EmbedConfig,
    ) -> anyhow::Result<EmbedResult> {
        let _ = (handle, inputs, config);
        Err(anyhow::anyhow!("embed() not supported by this backend"))
    }

    /// Run a forward pass over `tokens` and return the predicted next token ID.
    ///
    /// Tokens accumulate across calls (KV cache is extended). Used by the C API
    /// shim to implement `llama_eval`. The returned ID drives synthetic logits
    /// in `llama_get_logits` (greedy-compatible near-degenerate distribution).
    ///
    /// Default implementation returns an error — only selected backends support this.
    fn eval_tokens(&self, handle: ModelHandle, tokens: &[u32]) -> anyhow::Result<u32> {
        let _ = (handle, tokens);
        Err(anyhow::anyhow!("eval_tokens not supported by this backend"))
    }
}
