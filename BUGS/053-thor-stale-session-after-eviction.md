# BUG-053: Stale Session Persists After Eviction + Failed Re-registration

**Severity:** High
**File:** `crates/ax-thor-agent/src/agent.rs:164-169`
**Status:** ✅ FIXED (2026-03-29)

## Description

When the control plane returns 404/410 (eviction), the heartbeat loop attempts re-registration. On failure, the stale `WorkerSession` remains in `runtime.session` and is **never cleared**. The next loop iteration reads the same evicted session, sends another heartbeat to the old `worker_id`, gets 404/410 again, and retries re-registration -- indefinitely, at a fixed cadence with no backoff.

## Impact

Persistent failure loop with no escalation. If the control plane is down, every heartbeat cycle produces: one heartbeat failure + one re-registration failure, logging noise and wasting resources with no exponential backoff.

## Fix

Clear `runtime.session` on re-registration failure so the loop falls into the `None` branch which has a 1-second sleep. Also add exponential backoff:

```rust
Err(err) => {
    tracing::warn!(%err, "thor agent re-registration failed");
    *runtime.session.write().await = None;
}
```

## Fix Applied
On re-registration failure, clear the stale session: `*runtime.session.write().await = None;`. This prevents the next heartbeat from sending to an evicted worker ID.
