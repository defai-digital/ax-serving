# BUG-066: BlockingExecutor Silently Drops Jobs If Sender Disconnects

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/llamacpp.rs:363-370`
**Status:** ⏳ DEFERRED

## Description

If the `BlockingExecutor` sender is dropped (e.g., during `LlamaCppBackend` drop), `self.tx.send()` fails and returns an error. The caller gets "blocking executor stopped", but the response `tx` channel is never sent a `Done` or `Error` event. The receiver will hang until it times out or is dropped.

## Fix

On `execute` failure, send a `GenerateEvent::Error` via the caller's `tx` before returning.
