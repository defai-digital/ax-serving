# BUG-019: Worker Registration Accepts `max_inflight: 0`

**Severity:** Low
**File:** `crates/ax-serving-api/src/orchestration/registry.rs`
**Lines:** 429–521
**Status:** ✅ FIXED (2026-03-28)

## Description

Worker registration stores `max_inflight` as-is with no validation that it is greater than zero.

```rust
pub fn register(&self, req: RegisterRequest, heartbeat_interval_ms: u64) -> RegisterResponse {
    let RegisterRequest { max_inflight, .. } = req;
    // No check: max_inflight > 0
    self.inner.entry(id)
        .or_insert_with(|| WorkerEntry {
            max_inflight,  // stored as-is
            // ...
        });
}
```

## Impact

If a worker registers with `max_inflight: 0`, the dispatcher's capacity check (`inflight < max_inflight`, i.e. `0 < 0`) always fails. The worker appears healthy in the registry and health checks but never receives any requests — a zombie worker that confuses operators.

## Fix

Clamp to at least 1: `max_inflight: max_inflight.max(1)`, or reject the registration.

## Fix Applied

Added `let max_inflight = max_inflight.max(1);` at the top of `register()`, before the value is stored in either the `and_modify` or `or_insert_with` branches.
