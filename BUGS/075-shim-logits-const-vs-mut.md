# BUG-075: Shim `llama_get_logits` Returns `*const f32` vs llama.cpp's `float*`

**Severity:** Medium
**File:** `crates/ax-serving-shim/src/context.rs:145-149`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
pub extern "C" fn llama_get_logits(ctx: *const LlamaContext) -> *const f32 {
```

llama.cpp's `llama_get_logits` returns `float*` (mutable pointer). Many downstream consumers (e.g., llama.cpp examples, ollama, LM Studio) write logit biases directly into this buffer. The shim returns a `const` pointer, so C callers can't write to it without a cast -- breaking compatibility with code that writes logit biases.

## Impact

Any C code that writes logit biases through `llama_get_logits` (standard llama.cpp usage) will fail to compile or produce undefined behavior when linked against the shim.

## Fix

Return `*mut f32` and take `*mut LlamaContext`:

```rust
pub extern "C" fn llama_get_logits(ctx: *mut LlamaContext) -> *mut f32 {
    if ctx.is_null() { return std::ptr::null_mut(); }
    unsafe { (*ctx).logits.as_mut_ptr() }
}
```

## Fix Applied
Changed `llama_get_logits` signature from `*const LlamaContext -> *const f32` to `*mut LlamaContext -> *mut f32`. Updated test `compat.rs` accordingly.
