# BUG-023: Shutdown Race Between Heartbeat Re-registration and Drain

**Severity:** High
**File:** `crates/ax-thor-agent/src/main.rs`
**Lines:** 50–61
**Status:** ✅ FIXED (2026-03-29)

## Description

During graceful shutdown, there is a window where the heartbeat task can re-register with the control plane, replacing the session, between the `drain()` call and `drain_complete()`:

```rust
let _ = agent::drain(&client, &config, &runtime).await;        // uses session A
// ... wait loop ...
// heartbeat task still running, could re-register as session B via 404/410 response
heartbeat_task.abort();                                           // too late
let _ = agent::drain_complete(&client, &config, &runtime).await; // uses session B
```

## Impact

If the control plane responds to a heartbeat with 404/410 during the drain wait window, `heartbeat_loop` re-registers and updates `runtime.session` with a new worker ID. `drain_complete()` then sends a drain-complete for the new session, but the old session's drain state is inconsistent on the control plane. The new registration also causes the control plane to route traffic to the worker that is shutting down.

## Fix

Abort the heartbeat task **before** calling `drain()`:

```rust
heartbeat_task.abort();
let _ = agent::drain(&client, &config, &runtime).await;
// ... wait for inflight ...
let _ = agent::drain_complete(&client, &config, &runtime).await;
```

## Fix Applied
Moved `heartbeat_task.abort()` to execute **before** `agent::drain()`, preventing the heartbeat loop from re-registering during the drain window.
