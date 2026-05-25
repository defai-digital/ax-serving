//! Opaque C API handle types.
//!
//! LlamaModel and LlamaContext are never exposed by layout to C callers.
//! They are always passed as raw pointers to heap-allocated Rust structs.

use std::sync::Arc;

use ax_serving_engine::{InferenceBackend, ModelHandle};

/// Opaque model handle (passed as `struct llama_model*` in C).
pub struct LlamaModel {
    pub(crate) inner: Arc<LlamaModelInner>,
}

pub(crate) struct LlamaModelInner {
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

impl LlamaModel {
    pub(crate) fn new(
        handle: ModelHandle,
        backend: Arc<dyn InferenceBackend>,
        vocab_size: i32,
        n_ctx: i32,
        bos_token: i32,
        eos_token: i32,
        nl_token: i32,
    ) -> Self {
        Self {
            inner: Arc::new(LlamaModelInner {
                handle,
                backend,
                vocab_size,
                n_ctx,
                bos_token,
                eos_token,
                nl_token,
            }),
        }
    }
}

impl Drop for LlamaModelInner {
    fn drop(&mut self) {
        if let Err(e) = self.backend.unload_model(self.handle) {
            tracing::warn!("llama model drop: backend unload failed: {e}");
        }
    }
}

/// Opaque context handle (passed as `struct llama_context*` in C).
#[allow(dead_code)]
pub struct LlamaContext {
    pub(crate) model: Arc<LlamaModelInner>,
    pub(crate) position: usize,
    pub(crate) logits: Vec<f32>,
    pub(crate) n_ctx: i32,
    /// Accumulated token IDs passed to `llama_eval`. Extended each call.
    pub(crate) token_buf: Vec<u32>,
}
