# BUG-096: Heartbeat/registration requests can stall for 300s

**Severity:** Medium  
**File:** `crates/ax-thor-agent/src/agent.rs:160`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

The heartbeat loop and `register()` use the shared `reqwest::Client` configured with a global 300-second timeout. Control plane requests are expected to complete in ~1s. If the control plane is slow or unresponsive, a single heartbeat cycle stalls for up to 300s, causing cascading eviction and re-registration stalls.

Distinct from BUG-055 (which focuses on the timeout being too *short* for streaming); this is the timeout being too *long* for control plane requests.

## Why It's A Bug

During the stall, no heartbeats are sent. The control plane evicts the worker for missing its heartbeat deadline, then the eviction triggers re-registration (which also stalls for 300s). Agent is effectively unresponsive for up to 600s.

## Suggested Fix

Create a separate `reqwest::Client` for control plane requests with a shorter timeout (e.g., 10s), or use per-request `.timeout()` overrides for heartbeat and registration calls.
