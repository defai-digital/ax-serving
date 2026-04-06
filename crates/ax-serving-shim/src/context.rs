//! llama_context_* C API functions.

use std::sync::Arc;

use crate::types::{LlamaContext, LlamaModel};

/// Parameters for creating a context (mirrors llama_context_params).
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub seed: u32,
    pub _pad: [u32; 6],
}

/// Create a new inference context for a loaded model.
///
/// # Safety
/// `model` must be a valid non-null pointer returned by `llama_model_load_from_file`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_new_context_with_model(
    model: *mut LlamaModel,
    params: LlamaContextParams,
) -> *mut LlamaContext {
    if model.is_null() {
        tracing::error!("llama_new_context_with_model: null model");
        return std::ptr::null_mut();
    }

    let model_ref = unsafe { &*model };
    let n_ctx = if params.n_ctx > 0 {
        params.n_ctx as i32
    } else {
        model_ref.n_ctx
    };
    let vocab_size = model_ref.vocab_size as usize;

    let ctx = Box::new(LlamaContext {
        model: Arc::new(LlamaModel {
            handle: model_ref.handle,
            backend: model_ref.backend.clone(),
            vocab_size: model_ref.vocab_size,
            n_ctx: model_ref.n_ctx,
            bos_token: model_ref.bos_token,
            eos_token: model_ref.eos_token,
            nl_token: model_ref.nl_token,
        }),
        position: 0,
        logits: vec![0.0_f32; vocab_size],
        n_ctx,
        token_buf: Vec::new(),
    });

    Box::into_raw(ctx)
}

/// Free a context.
///
/// # Safety
/// `ctx` must be a valid non-null pointer returned by `llama_new_context_with_model`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_free(ctx: *mut LlamaContext) {
    if ctx.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(ctx) });
}

/// Return the context's maximum sequence length.
#[unsafe(no_mangle)]
pub extern "C" fn llama_n_ctx(ctx: *const LlamaContext) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    unsafe { (*ctx).n_ctx }
}

/// Run a forward pass over `n_tokens` tokens, extending the KV cache.
///
/// Tokens accumulate in an internal buffer across calls. On success, synthetic
/// logits are written to the context's logit buffer: −20.0 for all tokens,
/// +20.0 at the position of the predicted next token. This near-degenerate
/// distribution is compatible with all sampling functions (`llama_sample_*`).
///
/// Returns 0 on success, −1 on error or invalid arguments.
///
/// # Safety
/// `ctx` must be a valid non-null pointer. `tokens` must point to at least
/// `n_tokens` valid `int32_t` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_eval(
    ctx: *mut LlamaContext,
    tokens: *const i32,
    n_tokens: i32,
    _n_past: i32,
) -> i32 {
    if ctx.is_null() || tokens.is_null() || n_tokens <= 0 {
        return -1;
    }
    let ctx_ref = unsafe { &mut *ctx };
    let i32_slice = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };
    if i32_slice.iter().any(|&t| t < 0) {
        tracing::error!("llama_eval: negative token ID in input");
        return -1;
    }
    let token_slice =
        unsafe { std::slice::from_raw_parts(tokens as *const u32, n_tokens as usize) };
    if ctx_ref.token_buf.len() + token_slice.len() > ctx_ref.n_ctx as usize {
        tracing::error!(
            "llama_eval: token count ({}) would exceed n_ctx ({})",
            ctx_ref.token_buf.len() + token_slice.len(),
            ctx_ref.n_ctx
        );
        return -1;
    }
    ctx_ref.token_buf.extend_from_slice(token_slice);

    match ctx_ref
        .model
        .backend
        .eval_tokens(ctx_ref.model.handle, &ctx_ref.token_buf)
    {
        Ok(next_id) => {
            // Write synthetic logits: -20.0 everywhere, +20.0 at next predicted token.
            for logit in ctx_ref.logits.iter_mut() {
                *logit = -20.0;
            }
            if (next_id as usize) < ctx_ref.logits.len() {
                ctx_ref.logits[next_id as usize] = 20.0;
            } else {
                tracing::warn!(
                    next_id,
                    vocab_size = ctx_ref.logits.len(),
                    "eval: predicted token ID out of vocab range; logits are flat"
                );
            }
            ctx_ref.position = ctx_ref.token_buf.len();
            0
        }
        Err(e) => {
            tracing::error!("llama_eval: {e}");
            -1
        }
    }
}

/// Return a pointer to the logit buffer (one f32 per vocabulary token).
///
/// Valid until the next `llama_eval` call or until the context is freed.
/// Returns null if `ctx` is null.
#[unsafe(no_mangle)]
pub extern "C" fn llama_get_logits(ctx: *mut LlamaContext) -> *mut f32 {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*ctx).logits.as_mut_ptr() }
}
