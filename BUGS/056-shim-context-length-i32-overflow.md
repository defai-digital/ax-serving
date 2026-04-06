# BUG-056: Shim `context_length` Silent i32 Overflow Bypasses Capacity Check

**Severity:** High
**File:** `crates/ax-serving-shim/src/model.rs:91-94`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
n_ctx: if meta.context_length > 0 {
    meta.context_length as i32
} else {
    4096
},
```

If `meta.context_length` (likely `u32` or `usize`) exceeds `i32::MAX`, the truncated `n_ctx` wraps negative. A negative `n_ctx` used as `i32` and cast to `usize` elsewhere (e.g., `context.rs:107`) becomes a huge number, defeating the capacity check:

```rust
if ctx_ref.token_buf.len() + token_slice.len() > ctx_ref.n_ctx as usize {
```

A negative `n_ctx as usize` would be ~2^64, so the capacity check **never triggers**, allowing unbounded `token_buf` growth.

## Impact

Models with very large context lengths (> 2B tokens, theoretically possible with future architectures) bypass context window enforcement, leading to unbounded memory allocation in the shim.

## Fix

Use `try_into()` and return null on overflow, or clamp to `i32::MAX`:

```rust
n_ctx: if meta.context_length > 0 {
    i32::try_from(meta.context_length).unwrap_or(i32::MAX)
} else {
    4096
},
```

## Fix Applied
Added `.min(i32::MAX as u32)` before the `as i32` cast for `context_length` in `model.rs`.
