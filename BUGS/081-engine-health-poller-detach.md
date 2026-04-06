# BUG-081: Health Poller Thread Detach Prevents Clean Shutdown

**Severity:** Low
**File:** `crates/ax-serving-engine/src/llamacpp.rs:384`
**Status:** ⏳ DEFERRED

## Description

`LlamaCppProcess::_poller` is a `JoinHandle` that is dropped (not joined) in `Drop`. This means the poller thread is detached during process shutdown. If the process is shutting down, the poller thread may continue running briefly and attempt to restart the server after the main thread has begun cleanup.

## Fix

Join the poller thread with a timeout in `Drop`, after setting `stop = true`.
