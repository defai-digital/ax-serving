# BUG-037: Shim `token_buf` Grows Unbounded With No Context-Length Check

**Severity:** Medium
**File:** `crates/ax-serving-shim/src/context.rs`
**Lines:** 104, 119
**Status:** ✅ FIXED (2026-03-29)

## Description

`ctx_ref.token_buf.extend_from_slice(token_slice)` appends tokens on every `llama_eval` call without ever checking whether the accumulated length exceeds `n_ctx`. There is no error returned and no eviction:

```rust
ctx_ref.token_buf.extend_from_slice(token_slice);

match ctx_ref.model.backend.eval_tokens(ctx_ref.model.handle, &ctx_ref.token_buf) {
    Ok(next_id) => {
        // ...
        ctx_ref.position = ctx_ref.token_buf.len();
        0
    }
    // ...
}
```

## Impact

A runaway or buggy C client that calls `llama_eval` in a tight loop without resetting the context will eventually exhaust memory. Real llama.cpp enforces the context window and errors out.

## Fix

Before extending `token_buf`, check against the context length:

```rust
if ctx_ref.token_buf.len() + token_slice.len() > ctx_ref.n_ctx as usize {
    tracing::error!("llama_eval: context length exceeded ({} + {} > {})",
        ctx_ref.token_buf.len(), token_slice.len(), ctx_ref.n_ctx);
    return -1;
}
```

## Fix Applied
Added a check before `extend_from_slice`: if `token_buf.len() + token_slice.len() > n_ctx`, log an error and return -1.
