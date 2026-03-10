//! Opaque C API handle types.
//!
//! LlamaModel and LlamaContext are never exposed by layout to C callers.
//! They are always passed as raw pointers to heap-allocated Rust structs.

use std::sync::Arc;

use ax_serving_engine::{InferenceBackend, ModelHandle};

/// Opaque model handle (passed as `struct llama_model*` in C).
pub struct LlamaModel {
    pub(crate) handle: ModelHandle,
    pub(crate) backend: Arc<dyn InferenceBackend>,
    pub(crate) vocab_size: i32,
    pub(crate) n_ctx: i32,
    /// BOS token ID (−1 if unknown).
    pub(crate) bos_token: i32,
    /// EOS token ID (−1 if unknown).
    pub(crate) eos_token: i32,
    /// Newline token ID (−1 if unknown).
    pub(crate) nl_token: i32,
}

/// Opaque context handle (passed as `struct llama_context*` in C).
#[allow(dead_code)]
pub struct LlamaContext {
    pub(crate) model: Arc<LlamaModel>,
    pub(crate) position: usize,
    pub(crate) logits: Vec<f32>,
    pub(crate) n_ctx: i32,
    /// Accumulated token IDs passed to `llama_eval`. Extended each call.
    pub(crate) token_buf: Vec<u32>,
}
