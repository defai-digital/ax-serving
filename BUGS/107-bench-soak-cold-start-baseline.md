# BUG-107: `soak.rs` P95 drift measured against cold-start baseline, masking gradual degradation

**Severity:** Medium  
**File:** `crates/ax-serving-bench/src/soak.rs:99-115`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

```rust
if baseline_p95.is_none() {
    baseline_p95 = Some(p95);  // set once, never updated
}
let p95_drift = (p95.as_secs_f64() - baseline_secs) / baseline_secs;
```

The P95 drift is always computed against the very first P95 measurement (cold-start). A memory leak that causes 60% latency increase from stable state (e.g., 50ms → 80ms) would show drift = (80 - 100) / 100 = -20% → passes, because the cold-start baseline was 100ms.

## Why It's A Bug

The soak test is designed to catch gradual degradation but measures against the worst-case (cold start) baseline, masking real issues.

## Suggested Fix

Set the baseline after a warmup period (e.g., skip the first N bursts), or use a rolling baseline from the first check interval.
