# BUG-025: Cache Benchmark Has No HTTP Client Timeout

**Severity:** High
**File:** `crates/ax-serving-bench/src/cache_bench.rs`
**Lines:** 30
**Status:** ✅ FIXED (2026-03-29)

## Description

`Client::new()` is used without setting a timeout. Every other HTTP-based benchmark in this crate (`compare.rs`, `mixed.rs`, `multi_worker.rs`, `service_perf.rs`) uses `Client::builder().timeout(Duration::from_secs(120..300))`. This is the only one missing it.

```rust
let client = Client::new(); // no timeout!
```

## Impact

If the server hangs or becomes unresponsive, the benchmark hangs indefinitely with no feedback to the user. The only escape is a SIGKILL.

## Fix

```rust
let client = Client::builder()
    .timeout(std::time::Duration::from_secs(300))
    .build()?;
```

## Fix Applied
Replaced `Client::new()` with `Client::builder().timeout(Duration::from_secs(300)).build()?`.
