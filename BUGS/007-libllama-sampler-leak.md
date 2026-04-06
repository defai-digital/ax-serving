# BUG-007: Sampler Memory Leak on Client Disconnect in LibLlama Backend

**Severity:** High
**File:** `crates/ax-serving-engine/src/libllama.rs`
**Lines:** 1119–1122
**Status:** ✅ FIXED (2026-03-28)

## Description

When the streaming client disconnects during non-logprobs generation, the early-return path at line 1122 leaks the `sampler` raw pointer.

```rust
// Line 1119-1122
if !emit_logprobs
    && !flush_stream_token_batch(tx, &mut stream_token_buffer, &mut buffered_stream_tokens)
{
    return Ok(());  // LEAK: sampler is a raw pointer, never freed on this path
}

// Line 1125-1128 — only reached on the happy path
if !sampler.is_null() {
    unsafe { ffi::llama_sampler_free(sampler) };
}
```

`sampler` is a `*mut ffi::llama_sampler` heap-allocated C struct containing internal state (penalty history, distribution state). The early return bypasses the `llama_sampler_free` call.

## Impact

Memory leak on every client disconnect during streaming generation (non-logprobs path). The sampler's internal state accumulates with each leaked instance.

## Fix

Move sampler cleanup before the early return, or use a guard type:

```rust
struct SamplerGuard(*mut ffi::llama_sampler);
impl Drop for SamplerGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::llama_sampler_free(self.0) };
        }
    }
}
```

## Fix Applied

Added `llama_sampler_free(sampler)` call before the early `return Ok(())` at the disconnect path (line 1122), so the sampler is freed regardless of whether the function exits early or follows the happy path.
