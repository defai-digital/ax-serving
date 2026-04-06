# BUG-077: Bench `concurrency=0` Causes Infinite Hang

**Severity:** Medium
**File:** `crates/ax-serving-bench/src/mixed.rs:170` and `compare.rs:146`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
let sem = Arc::new(Semaphore::new(concurrency));
```

`Semaphore::new(0)` creates a semaphore with zero permits. The call to `sem.acquire_owned().await` blocks forever. There is no validation rejecting `--concurrency 0`. Note: `service_perf.rs:108` correctly clamps with `.max(1)` but mixed and compare do not.

## Fix

Add validation:

```rust
anyhow::ensure!(concurrency > 0, "concurrency must be >= 1");
```

## Fix Applied
Added `anyhow::ensure!(concurrency > 0, "concurrency must be >= 1")` in both `mixed.rs` and `compare.rs` before creating the semaphore.
