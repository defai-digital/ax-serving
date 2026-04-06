# BUG-104: `service_perf.rs` JoinErrors silently discarded (BUG-089 fix missed this file)

**Severity:** Medium  
**File:** `crates/ax-serving-bench/src/service_perf.rs:186`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

`let _ = handle.await;` silently discards all `JoinError`s, including panics. BUG-089 was filed and marked "FIXED" for `mixed.rs`, `compare.rs`, and `multi_worker.rs`, but `service_perf.rs:186` was missed — it still uses `let _ = handle.await`.

## Why It's A Bug

A panicked benchmark task in `service_perf` goes completely undetected, causing the benchmark to silently report incomplete data.

## Suggested Fix

Replace with the same pattern applied to the other files:
```rust
if let Err(e) = handle.await && e.is_panic() {
    tracing::warn!("service-perf task panicked: {e}");
}
```

## Fix Applied
Replaced `let _ = handle.await` with `if let Err(e) = handle.await && e.is_panic() { tracing::warn!(...) }`, matching the pattern applied in BUG-089.
