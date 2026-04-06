# BUG-108: `compare.rs` / `mixed.rs` RPS includes client-side semaphore dispatch overhead

**Severity:** Medium  
**File:** `crates/ax-serving-bench/src/compare.rs:152`, `crates/ax-serving-bench/src/mixed.rs:176`  
**Status:** ❌ FALSE POSITIVE  
**Introduced:** 2026-03-29

## Description

All `total_requests` tasks are spawned in a loop, but the semaphore limits concurrency. The main loop blocks on `sem.acquire_owned().await` between spawns. `total_ms` (RPS denominator) includes this dispatch overhead:

```rust
let start = Instant::now();
for i in 0..cfg.total_requests {
    let permit = sem.acquire_owned().await?;  // main thread blocks
    handles.push(tokio::spawn(...));
}
let total_ms = start.elapsed().as_millis() as u64;
let computed_rps = rps(total_success, total_ms);
```

## Why It's A Bug

The RPS metric measures throughput including client-side dispatch overhead, not just server throughput. For `--requests 1000 --concurrency 4`, RPS appears lower than actual server capacity.

## Suggested Fix

Compute RPS from the median/mean of individual request latencies instead of wall-clock time, or measure `total_ms` from first response to last response.
