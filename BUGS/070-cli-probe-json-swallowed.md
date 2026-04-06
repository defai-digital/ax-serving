# BUG-070: CLI `probe_agent_health` Silently Swallows JSON Parse Errors

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/thor.rs:632-637`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
response.json().await.unwrap_or_default()
```

Silently returns a default `AgentHealth` if the response body is not valid JSON. If the agent returns malformed JSON (e.g., HTML error page, truncated response), the status check will report the agent as "not ok" without any indication *why*.

## Impact

A misconfigured agent returning HTML is indistinguishable from one returning `{"status": "error"}`, making debugging very difficult.

## Fix

Log the parse error before falling back to default, or include the error in the `AgentHealth` struct.

## Fix Applied
Changed `.unwrap_or_default()` to explicit `match` with `Err(e)` arm that logs `tracing::warn!` before falling back to default.
