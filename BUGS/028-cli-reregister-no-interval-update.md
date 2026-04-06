# BUG-028: CLI Re-registration Does Not Update Heartbeat Interval

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/main.rs`
**Lines:** 774–779
**Status:** ✅ FIXED (2026-03-29)

## Description

When the orchestrator evicts a worker (404/410) and `heartbeat_loop` re-registers, only `cfg.worker_id` is updated from the new `WorkerReg`. The `cfg.interval_ms` field is left unchanged from the original registration:

```rust
Ok(r) => {
    tracing::info!(new_worker_id = %r.worker_id, "re-registered with orchestrator after eviction");
    cfg.worker_id = r.worker_id;
    // cfg.interval_ms NOT updated
}
```

## Impact

If the orchestrator configuration has changed the `heartbeat_interval_ms` between registrations (e.g., an operator tuned it, or a different orchestrator instance was deployed), the worker continues heartbeating at the old rate. If the new interval is shorter, the orchestrator may evict the worker for missing heartbeats, causing a registration/eviction flapping loop.

## Fix

```rust
Ok(r) => {
    cfg.worker_id = r.worker_id;
    cfg.interval_ms = r.heartbeat_interval_ms;
}
```

## Fix Applied
Added `cfg.interval_ms = r.heartbeat_interval_ms;` after updating `cfg.worker_id` in the re-registration success path.
