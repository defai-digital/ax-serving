# BUG-102: `llama_sample_top_k` with `k<=0` leaves `sorted` flag stale

**Severity:** Medium  
**File:** `crates/ax-serving-shim/src/sampling.rs:97`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
if candidates.is_null() || k <= 0 {
    return;
}
```

When `k <= 0`, the function returns immediately without setting `arr.sorted`. If a caller does `top_k(k) → top_k(0)` intending to reset, the stale `sorted` flag from the first call can cause `top_p` to skip sorting on a potentially modified candidate array.

## Why It's A Bug

The `sorted` flag is left stale when `k <= 0`. Subsequent `top_p` calls may skip sorting, producing incorrect sampling results.

## Suggested Fix

When `k <= 0`, explicitly set `arr.sorted = false` to invalidate any prior sorted state.

## Fix Applied
When `k <= 0`, explicitly set `arr.sorted = false` before returning, invalidating any prior sorted state.
