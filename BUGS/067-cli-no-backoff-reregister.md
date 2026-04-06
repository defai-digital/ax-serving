# BUG-067: CLI No Backoff on Orchestrator Re-registration Failure

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/main.rs:786-789`
**Status:** ✅ FIXED (2026-03-29)

## Description

When re-registration fails after eviction (404/410), the heartbeat loop just logs a warning and retries on the *next heartbeat interval*. If the orchestrator is temporarily down, this means re-registration attempts happen at the same fixed interval (e.g., every 5s). There's no exponential backoff.

## Impact

Under sustained orchestrator downtime, the worker spams registration attempts at a constant rate, flooding logs and network.

## Fix

Add exponential backoff for re-registration failures, capped at some max interval (e.g., 30s).

## Fix Applied
Added `rereg_backoff: u32` counter. On re-registration failure, increment and sleep for `2^n seconds` (capped at 30s). On success, reset to 0.
