# BUG-095: Thor agent `is_ok()` not checked in readiness condition

**Severity:** Medium  
**File:** `crates/ax-serving-cli/src/thor.rs:794-801`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
let ready = control.status == reqwest::StatusCode::OK
    && thor.ok
    && sglang.ok
    && registration_ok
    && config_mismatch.is_none()
    && worker.as_ref().is_some_and(|w| w.health == "healthy" && !w.drain);
```

`agent_health` is fetched and checked for config mismatches, but `agent_health.is_ok()` (which checks `status == "ok"` in the JSON body) is **never** checked in the `ready` condition. A thor agent returning HTTP 200 with body `{"status":"error","backend":"crashed"}` passes the `thor.ok` check and can be reported as "ready".

## Why It's A Bug

A broken agent (crashed backend) can be reported as ready, leading to routing failures.

## Suggested Fix

Add `&& agent_health.is_ok()` to the `ready` condition.

## Fix Applied
Added `&& agent_health.is_ok()` to the `ready` condition in the thor status check.
