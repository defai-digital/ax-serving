# BUG-045: Shim i32-to-u32 Reinterpret Cast for Token IDs

**Severity:** Low
**File:** `crates/ax-serving-shim/src/context.rs`
**Lines:** 103
**Status:** ✅ FIXED (2026-03-29)

## Description

The code casts `tokens: *const i32` to `*const u32` to reinterpret signed token IDs as unsigned:

```rust
let token_slice = unsafe { std::slice::from_raw_parts(tokens as *const u32, n_tokens as usize) };
```

If any token ID is negative (e.g., -1 is the "invalid token" sentinel returned by `llama_token_bos` when unknown), this produces a very large u32 value (4294967295).

## Impact

A negative sentinel token ID silently becomes a huge unsigned value, potentially causing out-of-bounds logit writes or confusing the backend.

## Fix

Validate that all token IDs are non-negative before the cast, or explicitly filter negative values and return an error.

## Fix Applied
Read the raw i32 slice first and check `if i32_slice.iter().any(|&t| t < 0)`. Return -1 with an error log if any negative token ID is found.
