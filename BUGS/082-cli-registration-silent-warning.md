# BUG-082: CLI Registration Failure Is Silent Warning

**Severity:** Low
**File:** `crates/ax-serving-cli/src/main.rs:984-988`
**Status:** ❌ FALSE POSITIVE

## Description

If the orchestrator is specified but registration fails, the worker starts anyway without registering. It will serve requests locally but won't receive routed traffic. The only indication is a `WARNING` printed to stderr. The heartbeat loop won't start (since `reg` is `None`), so the worker is silently orphaned.

## Fix

Consider making this a hard error by default with a `--allow-registration-failure` flag, or at minimum print a more prominent warning.
