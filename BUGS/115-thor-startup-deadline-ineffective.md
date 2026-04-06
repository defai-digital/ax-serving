# BUG-115: Startup deadline ineffective when configured timeout < connect_timeout

**Severity:** Low  
**File:** `crates/ax-thor-agent/src/sglang.rs:24-36`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

The deadline check (`Instant::now() > deadline`) is performed *after* the health check attempt, not before. The reqwest client has a 5-second `connect_timeout`. If `AXS_THOR_STARTUP_TIMEOUT_SECS=1`, the first health probe attempt blocks for 5 seconds before the deadline check fires.

## Why It's A Bug

The configured startup timeout is not a hard upper bound. With `AXS_THOR_STARTUP_TIMEOUT_SECS=1`, the actual startup wait is at least 5 seconds.

## Suggested Fix

Add a deadline check *before* each attempt, or wrap each health probe in a per-attempt timeout with `tokio::select!`.
