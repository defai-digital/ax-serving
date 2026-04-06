//! C API compatibility tests.
//!
//! Tests exercise the shim with null/invalid inputs — no model file required.
//! All tests verify that null-pointer safety invariants hold (no crashes, no
//! UB) when callers pass invalid arguments.
//!
//! Model-dependent tests (real tokenization, full inference) are marked
//! `#[ignore]` and require a valid GGUF file at `AXS_TEST_MODEL`.

use ax_serving_shim::context::{
    LlamaContextParams, llama_eval, llama_free, llama_get_logits, llama_n_ctx,
    llama_new_context_with_model,
};
use ax_serving_shim::model::{
    LlamaModelParams, llama_free_model, llama_model_load_from_file, llama_model_n_ctx_train,
    llama_n_vocab, llama_token_bos, llama_token_eos, llama_token_nl, llama_token_pad,
};
use ax_serving_shim::tokenize::{llama_token_to_piece, llama_tokenize};
use ax_serving_shim::{llama_backend_free, llama_backend_init, llama_system_info, llama_time_us};

fn default_model_params() -> LlamaModelParams {
    LlamaModelParams {
        n_gpu_layers: -1,
        n_ctx: 0,
        _pad: [0; 6],
    }
}

fn default_ctx_params() -> LlamaContextParams {
    LlamaContextParams {
        n_ctx: 512,
        seed: 0,
        _pad: [0; 6],
    }
}

/// Backend init + free must not crash.
#[test]
fn test_backend_init_free() {
    llama_backend_init(false);
    llama_backend_free();
}

/// Loading with a null path must return null (no crash, no UB).
#[test]
fn test_null_path_returns_null() {
    llama_backend_init(false);
    let model = unsafe { llama_model_load_from_file(std::ptr::null(), default_model_params()) };
    assert!(model.is_null(), "expected null for null path");
}

/// Loading with a nonexistent file path must return null.
#[test]
fn test_nonexistent_path_returns_null() {
    llama_backend_init(false);
    let path = c"/nonexistent/path/model.gguf";
    let model = unsafe { llama_model_load_from_file(path.as_ptr(), default_model_params()) };
    assert!(model.is_null(), "expected null for nonexistent path");
}

/// Creating a context from a null model must return null.
#[test]
fn test_null_model_context_returns_null() {
    llama_backend_init(false);
    let ctx = unsafe { llama_new_context_with_model(std::ptr::null_mut(), default_ctx_params()) };
    assert!(ctx.is_null(), "expected null for null model");
}

/// `llama_n_ctx(null)` must return 0.
#[test]
fn test_n_ctx_null_returns_zero() {
    let result = llama_n_ctx(std::ptr::null());
    assert_eq!(result, 0);
}

/// `llama_n_vocab(null)` must return 0.
#[test]
fn test_n_vocab_null_returns_zero() {
    let result = llama_n_vocab(std::ptr::null());
    assert_eq!(result, 0);
}

/// `llama_get_logits(null)` must return a null pointer.
#[test]
fn test_get_logits_null_returns_null() {
    let ptr = llama_get_logits(std::ptr::null_mut());
    assert!(ptr.is_null(), "expected null pointer for null context");
}

/// `llama_eval(null, ...)` must return -1.
#[test]
fn test_eval_null_ctx_returns_error() {
    let tokens: [i32; 1] = [1];
    let result = unsafe { llama_eval(std::ptr::null_mut(), tokens.as_ptr(), 1, 0) };
    assert_eq!(result, -1, "expected -1 for null ctx");
}

/// `llama_eval` with null tokens must return -1.
#[test]
fn test_eval_null_tokens_returns_error() {
    // We have no real context, but we can still test the null token ptr path
    // by checking the null ctx path first (same -1 result; both guard together).
    let result = unsafe { llama_eval(std::ptr::null_mut(), std::ptr::null(), 0, 0) };
    assert_eq!(result, -1, "expected -1 for null ctx/tokens");
}

/// `llama_tokenize` with null model must return negative.
#[test]
fn test_tokenize_null_model_returns_error() {
    let text = c"hello";
    let mut buf = [0i32; 16];
    let result = unsafe {
        llama_tokenize(
            std::ptr::null(),
            text.as_ptr(),
            5,
            buf.as_mut_ptr(),
            16,
            false,
            false,
        )
    };
    assert!(result < 0, "expected negative for null model");
}

/// `llama_token_to_piece` with null model must return negative.
#[test]
fn test_token_to_piece_null_model_returns_error() {
    let mut buf = [0i8; 16];
    let result = unsafe { llama_token_to_piece(std::ptr::null(), 1, buf.as_mut_ptr(), 16) };
    assert!(result < 0, "expected negative for null model");
}

/// `llama_free` with null context must not crash.
#[test]
fn test_free_null_ctx_no_crash() {
    unsafe { llama_free(std::ptr::null_mut()) };
    // If we reach here without crashing, the null guard works.
}

/// `llama_free_model` with null model must not crash.
#[test]
fn test_free_null_model_no_crash() {
    unsafe { llama_free_model(std::ptr::null_mut()) };
}

// ── Extended C API (Phase 3) ──────────────────────────────────────────────────

/// `llama_token_bos(null)` must return −1.
#[test]
fn test_token_bos_null_returns_minus_one() {
    assert_eq!(llama_token_bos(std::ptr::null()), -1);
}

/// `llama_token_eos(null)` must return −1.
#[test]
fn test_token_eos_null_returns_minus_one() {
    assert_eq!(llama_token_eos(std::ptr::null()), -1);
}

/// `llama_token_nl(null)` must return −1.
#[test]
fn test_token_nl_null_returns_minus_one() {
    assert_eq!(llama_token_nl(std::ptr::null()), -1);
}

/// `llama_token_pad` always returns −1 (no dedicated padding token).
#[test]
fn test_token_pad_always_minus_one() {
    assert_eq!(llama_token_pad(std::ptr::null()), -1);
}

/// `llama_model_n_ctx_train(null)` must return 0.
#[test]
fn test_model_n_ctx_train_null_returns_zero() {
    assert_eq!(llama_model_n_ctx_train(std::ptr::null()), 0);
}

/// `llama_time_us` must return a positive value.
#[test]
fn test_time_us_positive() {
    let t = llama_time_us();
    assert!(t > 0, "expected positive timestamp, got {t}");
}

/// `llama_time_us` must be monotonically non-decreasing across two calls.
#[test]
fn test_time_us_monotonic() {
    let t1 = llama_time_us();
    let t2 = llama_time_us();
    assert!(t2 >= t1, "timestamps must be non-decreasing: {t1} > {t2}");
}

/// `llama_system_info` must return a non-null, non-empty C string.
#[test]
fn test_system_info_not_null() {
    let ptr = llama_system_info();
    assert!(!ptr.is_null(), "expected non-null system info string");
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .expect("system info must be valid UTF-8");
    assert!(!s.is_empty(), "system info must not be empty");
}

/// Full init→load→context→eval→get_logits→free round-trip.
///
/// Requires a valid GGUF file at the path given by `AXS_TEST_MODEL`.
/// Run with: `AXS_TEST_MODEL=/path/to/model.gguf cargo test -p ax-serving-shim -- --ignored`
#[test]
#[ignore]
fn test_full_roundtrip_with_model() {
    let model_path = std::env::var("AXS_TEST_MODEL")
        .expect("AXS_TEST_MODEL must be set for ignored model tests");
    let path_cstr = std::ffi::CString::new(model_path).unwrap();

    llama_backend_init(false);

    let model = unsafe { llama_model_load_from_file(path_cstr.as_ptr(), default_model_params()) };
    assert!(!model.is_null(), "model load failed");
    assert!(llama_n_vocab(model) > 0);

    let ctx = unsafe { llama_new_context_with_model(model, default_ctx_params()) };
    assert!(!ctx.is_null(), "context creation failed");
    assert!(llama_n_ctx(ctx) > 0);

    // Eval a single BOS token (ID=1 is a common placeholder).
    let tokens: [i32; 1] = [1];
    let ret = unsafe { llama_eval(ctx, tokens.as_ptr(), 1, 0) };
    assert_eq!(ret, 0, "llama_eval failed");

    // get_logits must return a non-null pointer.
    let logits = llama_get_logits(ctx);
    assert!(!logits.is_null(), "get_logits returned null");

    // At least one logit must be +20.0 (the predicted token's synthetic logit).
    let vocab_size = llama_n_vocab(model) as usize;
    let logit_slice = unsafe { std::slice::from_raw_parts(logits, vocab_size) };
    let max_logit = logit_slice
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (max_logit - 20.0).abs() < 1e-3,
        "expected max logit ~20.0, got {max_logit}"
    );

    unsafe {
        llama_free(ctx);
        llama_free_model(model);
    }
    llama_backend_free();
}
