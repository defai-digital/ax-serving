# BUG-118: NaN temperature divides all logits by NaN, producing arbitrary sampling

**Severity:** Low  
**File:** `crates/ax-serving-shim/src/sampling.rs:46`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
if temp <= 0.0 { return; }
```

If `temp` is `NaN`, the comparison `temp <= 0.0` is `false`, so the function proceeds to divide logits by NaN, making all logits NaN. Downstream `partial_cmp` on NaN resolves to `Equal` via `unwrap_or`, producing an arbitrary first-element selection.

## Why It's A Bug

NaN temperature silently corrupts all logits. No error is signaled.

## Suggested Fix

Check `temp.is_nan()` and treat as greedy (return early).

## Fix Applied
Added `|| temp.is_nan()` to the guard: `if temp <= 0.0 || temp.is_nan() { return; }`.
