# BUG-032: Non-Atomic Models and Session Update During Thor Re-registration

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/agent.rs`
**Lines:** 162–164
**Status:** ⏳ DEFERRED

## Description

When re-registering after eviction, the heartbeat loop updates `models` and `session` under separate write locks, creating an inconsistency window:

```rust
Ok(registration) => {
    *runtime.models.write().await = registration.models;      // lock 1
    *runtime.session.write().await = Some(registration.session);  // lock 2
}
```

## Impact

A concurrent reader (e.g., another heartbeat iteration, or a request handler inspecting models) can observe the new models with the old session, or the new session with old models. This could cause heartbeats that report the wrong model list for the new worker ID, or the control plane making routing decisions based on stale model information.

## Fix

Use a single combined lock that holds both values atomically, e.g., wrap the pair in a single `RwLock<(Vec<String>, Option<WorkerSession>)>`.
