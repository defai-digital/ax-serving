# BUG-078: Bench Burst Timing Broken Under Contention

**Severity:** Medium
**File:** `crates/ax-serving-bench/src/service_perf.rs:120-146`
**Status:** ⏳ DEFERRED

## Description

```rust
let start = Instant::now();  // captured BEFORE semaphore wait

for i in 0..cfg.requests {
    let permit = Arc::clone(&sem).acquire_owned().await?;  // can block
    let launch_offset_ms = ((i as u64) * burst_window_ms) / ((cfg.requests - 1) as u64);
    let deadline = start + Duration::from_millis(launch_offset_ms);
    tokio::time::sleep_until(deadline.into()).await;
```

`start` is captured before the semaphore wait. When all concurrency slots are occupied, task `i` may wait seconds for a permit, but its `launch_offset_ms` was computed assuming instant acquisition. The deadline is already in the past, so `sleep_until` returns immediately. The staggered burst pattern collapses to a concurrent flood.

## Fix

Capture `start` after acquiring the permit, or compute offset relative to permit acquisition time.
