//! llama_model_* C API functions.

use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

use ax_serving_engine::{BackendType, InferenceBackend, LoadConfig, RouterBackend};

use crate::types::LlamaModel;

/// Parameters for loading a model (mirrors llama_model_params).
#[repr(C)]
pub struct LlamaModelParams {
    /// Number of GPU layers (-1 = all, 0 = CPU only).
    pub n_gpu_layers: i32,
    /// Context length override (0 = use model default).
    pub n_ctx: u32,
    /// Reserved / ignored.
    pub _pad: [u32; 6],
}

/// Load a GGUF model from file. Returns null on failure.
///
/// # Safety
/// `path_bytes` must be a valid null-terminated C string pointing to a file path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_model_load_from_file(
    path_bytes: *const libc::c_char,
    params: LlamaModelParams,
) -> *mut LlamaModel {
    if path_bytes.is_null() {
        tracing::error!("llama_model_load_from_file: null path");
        return std::ptr::null_mut();
    }

    let path = match unsafe { CStr::from_ptr(path_bytes) }.to_str() {
        Ok(s) => Path::new(s).to_path_buf(),
        Err(e) => {
            tracing::error!("llama_model_load_from_file: invalid UTF-8 path: {e}");
            return std::ptr::null_mut();
        }
    };

    let backend_type = if params.n_gpu_layers == 0 {
        BackendType::Cpu
    } else {
        BackendType::Metal
    };

    let config = LoadConfig {
        context_length: params.n_ctx,
        backend_type,
        llama_cpp_n_gpu_layers: Some(params.n_gpu_layers),
        mmproj_path: None,
        backend_hint: None,
        enable_embeddings: None,
        pooling_type: None,
    };

    // Share a single RouterBackend across all model loads (BUG-074).
    static BACKEND: std::sync::OnceLock<Arc<dyn InferenceBackend>> = std::sync::OnceLock::new();
    let backend = BACKEND
        .get_or_init(|| Arc::new(RouterBackend::from_env()))
        .clone();

    match backend.load_model(&path, config) {
        Ok((handle, meta)) => {
            // Probe special tokens from the loaded model.
            let eos_token = backend
                .eos_tokens(handle)
                .ok()
                .and_then(|v| v.first().copied())
                .map(|t| t as i32)
                .unwrap_or(-1);
            // BOS token: query directly from backend (BUG-103).
            let bos_token = backend.bos_token(handle).map(|t| t as i32).unwrap_or(-1);
            // Newline token: tokenize "\n" without BOS.
            let nl_token = backend
                .tokenize(handle, "\n", false)
                .ok()
                .and_then(|v| v.first().copied())
                .map(|t| t as i32)
                .unwrap_or(-1);
            let model = Box::new(LlamaModel {
                handle,
                backend,
                vocab_size: (meta.vocab_size).min(i32::MAX as u32) as i32,
                n_ctx: if meta.context_length > 0 {
                    (meta.context_length).min(i32::MAX as u32) as i32
                } else {
                    4096
                },
                bos_token,
                eos_token,
                nl_token,
            });
            Box::into_raw(model)
        }
        Err(e) => {
            tracing::error!("llama_model_load_from_file: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Return the beginning-of-sequence token ID, or −1 if not available.
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_bos(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    unsafe { (*model).bos_token }
}

/// Return the end-of-sequence token ID, or −1 if not available.
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_eos(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    unsafe { (*model).eos_token }
}

/// Return the newline token ID, or −1 if not available.
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_nl(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    unsafe { (*model).nl_token }
}

/// Return the padding token ID. Returns −1 (no dedicated padding token).
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_pad(_model: *const LlamaModel) -> i32 {
    -1
}

/// Return the training context length for a model.
///
/// For models where `n_ctx` equals the training context length (standard GGUF),
/// this returns the same value as `llama_n_ctx` on a default context.
#[unsafe(no_mangle)]
pub extern "C" fn llama_model_n_ctx_train(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return 0;
    }
    unsafe { (*model).n_ctx }
}

/// Free a loaded model.
///
/// # Safety
/// `model` must be a valid non-null pointer returned by `llama_model_load_from_file`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_free_model(model: *mut LlamaModel) {
    if model.is_null() {
        return;
    }
    let m = unsafe { Box::from_raw(model) };
    if let Err(e) = m.backend.unload_model(m.handle) {
        tracing::warn!("llama_free_model: backend unload failed: {e}");
    }
    // Box dropped here → frees LlamaModel
}

/// Return the vocabulary size.
#[unsafe(no_mangle)]
pub extern "C" fn llama_n_vocab(model: *const LlamaModel) -> i32 {
    if model.is_null() {
        return 0;
    }
    unsafe { (*model).vocab_size }
}
