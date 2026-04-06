# BUG-120: `llama_sample_top_p` with `p=1.0` can truncate candidates

**Severity:** Low  
**File:** `crates/ax-serving-shim/src/sampling.rs:164-165`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
if cumulative >= p {
    cutoff = i + 1;
    break;
}
```

When `p = 1.0` and floating-point accumulation reaches 1.0 slightly early (due to rounding), `cumulative >= 1.0` can be true after just a few high-probability tokens. `top_p(1.0)` should be a no-op (keep all tokens) but can truncate the candidate list.

## Why It's A Bug

`top_p(1.0)` is conventionally "disable filtering" but doesn't behave that way due to floating-point accumulation.

## Suggested Fix

Add special case: if `p >= 1.0`, return without truncating.

## Fix Applied
Added early return: `if p >= 1.0 { return; }` — p=1.0 means "keep all candidates", no filtering needed.
