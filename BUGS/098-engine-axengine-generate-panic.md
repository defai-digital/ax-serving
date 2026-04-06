# BUG-098: AxEngineBackend generate thread panic silently swallows error

**Severity:** Medium  
**File:** `crates/ax-serving-engine/src/ax_engine.rs:756-764`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

The generation thread spawned in `AxEngineBackend::generate` only handles `Err` from `run_generate`. If `run_generate` **panics** (e.g., assertion failure in ax-core), the panic kills the spawned thread silently. The caller of `generate()` already received `Ok(())`, and the `rx` channel never receives a `Done` or `Error` event — the client hangs forever.

Compare with `libllama.rs:527-531` which correctly uses `std::panic::catch_unwind`.

## Why It's A Bug

A panicked generate job leaves the client with a hung response. The libllama backend correctly catches panics; the ax_engine backend does not.

## Suggested Fix

Wrap `run_generate` in `std::panic::catch_unwind(AssertUnwindSafe(|| ...))` and send a `GenerateEvent::Error` on panic.

## Fix Applied
Wrapped `run_generate` in `std::panic::catch_unwind(AssertUnwindSafe(...))`. On panic, sends `GenerateEvent::Error("internal panic in ax-engine generate")` to the client.
