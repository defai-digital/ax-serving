# BUG-116: `cache_bench.rs` warm average truncated by integer division, inflating speedup

**Severity:** Low  
**File:** `crates/ax-serving-bench/src/cache_bench.rs:71`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
let warm_avg_ms = warm_slice.iter().sum::<u128>() / warm_slice.len() as u128;
```

Integer division truncates. `[1ms, 2ms]` → average = `3 / 2 = 1ms` instead of `1.5ms`. The speedup calculation `cold_ms / warm_avg_ms` inflates: `100 / 1 = 100x` instead of `100 / 1.5 = 66.7x`.

## Why It's A Bug

The speedup ratio is slightly inaccurate due to truncated denominator.

## Suggested Fix

Compute average in `f64`:
```rust
let warm_avg_ms = warm_slice.iter().map(|&v| v as f64).sum::<f64>() / warm_slice.len() as f64;
```

## Fix Applied
Changed integer division to f64 sum+divide, then round back to u128.
