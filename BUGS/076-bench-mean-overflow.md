# BUG-076: Bench `mean()` u64 Sum Overflow

**Severity:** Medium
**File:** `crates/ax-serving-bench/src/bench_common.rs:50`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
values.iter().sum::<u64>() as f64 / values.len() as f64
```

`sum::<u64>()` wraps silently on overflow. With ~18K latency samples at ~1 second each (~1_000_000 ms each as u64), the sum exceeds `u64::MAX` and wraps, producing a completely wrong mean.

## Fix

Sum into `f64`: `values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64`

## Fix Applied
Changed `values.iter().sum::<u64>() as f64` to `values.iter().map(|&v| v as f64).sum::<f64>()` — summing in f64 prevents u64 wrapping.
