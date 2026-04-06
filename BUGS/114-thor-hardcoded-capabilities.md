# BUG-114: Hardcoded registration capabilities don't reflect actual backend

**Severity:** Low  
**File:** `crates/ax-thor-agent/src/agent.rs:63-65`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

The registration body hardcodes `"vision": false`, `"embedding": true`, and `"max_context": null` regardless of what the sglang backend actually supports.

Distinct from BUG-073 (hardcoded heartbeat metrics); this is the registration surface.

## Why It's A Bug

If sglang loads a vision-capable model, the control plane won't route vision requests. If sglang has no embedding models, embedding requests will fail. `max_context: null` prevents prompt-length-aware routing.

## Suggested Fix

Query sglang's `/v1/models` response for model capabilities, or at minimum derive `max_context` from the model's config.
