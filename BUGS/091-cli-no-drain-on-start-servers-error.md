# BUG-091: Missing drain/drain-complete when `start_servers` errors

**Severity:** High  
**File:** `crates/ax-serving-cli/src/main.rs:1025`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
start_servers(layer, &config).await?;  // line 1025

// Graceful shutdown — ONLY reached on clean shutdown:
if let Some(h) = hb_handle { hb_handle.abort(); }
if let Some(ref r) = reg {
    drain_worker(&client, &r.orchestrator_addr, &r.worker_id, ...).await;
    drain_complete(&client, &r.orchestrator_addr, &r.worker_id, ...).await;
}
```

If `start_servers` returns `Err` (e.g., port binding failure, TLS misconfiguration), the `?` operator short-circuits. The drain/drain-complete cleanup block is **skipped**. The worker remains registered in the orchestrator until its TTL expires (~15 s), during which the orchestrator routes requests to a dead endpoint.

## Why It's a Bug

Missing error-path cleanup. The orchestrator routes to the dead worker for the TTL window, causing connection-refused errors and unnecessary reroute churn.

## Suggested Fix

Use a `scopeguard::defer!` or wrap the drain logic to always execute on exit (both `Ok` and `Err` paths):
```rust
let _guard = scopeguard::guard((), |_| {
    // drain/drain-complete logic
});
```

## Fix Applied
Changed `start_servers(...).await?` to `let server_result = start_servers(...).await;`. Drain/drain-complete cleanup now runs unconditionally. `server_result` is propagated after cleanup.
