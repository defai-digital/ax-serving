# BUG-013: Softmax Division by Zero in Shim Sampling

**Severity:** Medium
**File:** `crates/ax-serving-shim/src/sampling.rs`
**Lines:** 140–151
**Status:** ✅ FIXED (2026-03-28)

## Description

If all logits are `f32::NEG_INFINITY` (or extremely negative such that `exp()` underflows to 0.0), the softmax computation divides by zero, producing NaN probabilities.

```rust
let max_logit = slice.iter().map(|e| e.logit).fold(f32::NEG_INFINITY, f32::max);
let mut sum = 0.0_f32;
for e in slice.iter_mut() {
    e.p = (e.logit - max_logit).exp();
    sum += e.p;
}
for e in slice.iter_mut() {
    e.p /= sum;  // division by zero if all logits are -inf
}
```

When `sum == 0.0`, dividing produces NaN. Downstream top-p filtering operates on NaN values; `partial_cmp` returns `None` (resolved to `Equal`), silently corrupting the sampling distribution.

## Impact

External C code that injects extreme logits will get silent token selection corruption without any error signal. While the synthetic logits from `llama_eval` use ±20.0 (making this unlikely in normal shim usage), the C API contract makes no guarantees about logit ranges.

## Fix

Add a zero-sum guard:

```rust
for e in slice.iter_mut() {
    e.p /= if sum > 0.0 { sum } else { 1.0 };
}
```

## Fix Applied

Added a zero-sum guard: `let norm = if sum > 0.0 { sum } else { 1.0 };` before the normalization loop. When all exps underflow to 0.0, probabilities become 0.0 (uniform-ish via downstream top-k/top-p) instead of NaN.
