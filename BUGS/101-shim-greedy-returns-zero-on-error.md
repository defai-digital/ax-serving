# BUG-101: `llama_sample_token_greedy` returns token ID `0` on error — ambiguous with valid token

**Severity:** Medium  
**File:** `crates/ax-serving-shim/src/sampling.rs:66-83`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
if candidates.is_null() { return 0; }  // token ID 0 is a valid token!
if arr.data.is_null() || arr.size == 0 { return 0; }
// ...
.unwrap_or(0)  // fallback also returns 0
```

Token ID `0` is a valid token in most vocabularies (often `<unk>` or the first byte). Returning `0` on null/empty inputs is indistinguishable from successfully sampling token `0`. llama.cpp returns `-1` for error cases.

## Why It's A Bug

A C caller cannot distinguish between "successfully sampled token 0" and "error / null input." This can silently produce wrong output.

## Suggested Fix

Return `-1` on error, matching llama.cpp convention.

## Fix Applied
Changed error returns from `0` to `-1` (matching llama.cpp convention) for null/empty candidates.
