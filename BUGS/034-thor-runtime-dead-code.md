# BUG-034: `runtime.rs` Is Dead Code in `ax-thor-agent`

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/lib.rs` (missing `pub mod runtime`)
**Status:** ✅ FIXED (2026-03-29)

## Description

`lib.rs` only declares `pub mod agent; pub mod config; pub mod proxy; pub mod sglang;`. The file `runtime.rs` is never included in the module tree, meaning:
- Its `wait_for_runtime()` and `get_loaded_models()` functions are dead code (duplicated in `sglang.rs`).
- Its three unit tests (`runtime_ready_uses_v1_models_when_health_is_missing`, `runtime_ready_fails_with_backend_specific_context`, `runtime_ready_uses_trtllm_backend_context`) are **never compiled or run**.
- `runtime.rs` provides backend-aware versions (vllm, trtllm, sglang) of these functions that are more capable than the sglang-only versions in `sglang.rs`, suggesting this was intended to replace `sglang.rs` but was never wired up.

## Impact

Dead code gives a false sense of test coverage. If someone later adds `pub mod runtime;` to `lib.rs` without removing `sglang.rs`, there will be duplicate public functions with different signatures, causing compile errors.

## Fix

Either complete the migration by adding `pub mod runtime;` to `lib.rs` and updating callers, or delete `runtime.rs` if it is truly abandoned.

## Fix Applied
Deleted `crates/ax-thor-agent/src/runtime.rs`. The file was never declared in `lib.rs` and contained dead code with uncompiled tests.
