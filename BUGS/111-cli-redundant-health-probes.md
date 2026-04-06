# BUG-111: Redundant concurrent health probes to same `/health` endpoint

**Severity:** Low  
**File:** `crates/ax-serving-cli/src/thor.rs:763-768`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
let (control, sglang, thor, agent_health, worker) = tokio::join!(
    control_plane_readiness(&client, &control_plane, token),
    probe_sglang(&client, &runtime_url),
    probe_health(&client, &thor_health_url),       // GET /health
    probe_agent_health(&client, &thor_health_url),  // GET /health (same URL!)
    probe_registered_worker(&client, &control_plane, token, advertised_addr),
);
```

`probe_health` and `probe_agent_health` both send GET requests to the same `/health` URL concurrently. One probe can succeed while the other fails (transient network issue), producing contradictory status.

## Why It's A Bug

Wastes resources and can produce inconsistent status.

## Suggested Fix

Remove `probe_health` and derive `thor.ok` from `probe_agent_health` result.

## Fix Applied
Removed `probe_health` from the parallel join; derive `thor.ok` from `agent_health.is_ok()` instead.
