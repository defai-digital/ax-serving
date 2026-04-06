# BUG-109: TTFT p95 truncated to 0ms by integer division

**Severity:** Low  
**File:** `crates/ax-serving-cli/src/main.rs:739`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
let ttft_p95_ms = layer.scheduler.metrics.ttft_p95_us() / 1000;
```

Integer division truncates. Any p95 value under 1000µs (1ms) becomes 0ms. A value of 500µs is reported as 0ms to the orchestrator.

## Why It's A Bug

The orchestrator may use `ttft_p95_ms` for routing decisions (prefer workers with lower TTFT). 0ms is misleadingly fast and could cause over-routing.

## Suggested Fix

Use `(ttft_p95_us + 500) / 1000` for rounding, or send microseconds and let the orchestrator convert.

## Fix Applied
Changed `ttft_p95_us / 1000` to `(ttft_p95_us + 500) / 1000` for proper rounding instead of truncation.
