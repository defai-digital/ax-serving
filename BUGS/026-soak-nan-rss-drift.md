# BUG-026: NaN RSS Drift Silently Masks Memory Leak in Soak Test

**Severity:** High
**File:** `crates/ax-serving-bench/src/soak.rs`
**Lines:** 47, 105
**Status:** ✅ FIXED (2026-03-29)

## Description

`baseline_rss` is computed from `current_rss_bytes()` on line 47. If this returns 0 (possible if the macOS task info call fails, e.g., in certain sandboxed/container environments), then `rss_drift = (rss - 0.0) / 0.0 = NaN`. Since `NaN > max_rss_drift` is always `false`, the RSS drift threshold check **never fires**, silently masking a real memory leak:

```rust
let baseline_rss = ax_serving_api::metrics::current_rss_bytes() as f64; // could be 0.0
// ...
let rss_drift = (rss - baseline_rss) / baseline_rss; // NaN when baseline_rss == 0.0
```

Notably, `baseline_p95` is properly guarded (line 108–111 uses `.filter(|secs| *secs > 0.0).unwrap_or(1e-9)`), but RSS is not.

## Impact

A production soak test could run for 24 hours, RSS could grow from 0 to 10 GB, and the test passes because the drift calculation produces NaN. Memory leaks go undetected.

## Fix

Guard `baseline_rss` the same way `baseline_p95` is guarded:

```rust
let baseline_rss = {
    let raw = ax_serving_api::metrics::current_rss_bytes() as f64;
    if raw > 0.0 { raw } else { 1.0 }
};
```

## Fix Applied
Guarded `baseline_rss`: if raw value is 0, use 1.0 as fallback. Prevents NaN from 0/0 division.
