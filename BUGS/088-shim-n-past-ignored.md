# BUG-088: Shim `_n_past` Parameter Completely Ignored

**Severity:** Low
**File:** `crates/ax-serving-shim/src/context.rs:94`
**Status:** ❌ FALSE POSITIVE

## Description

In llama.cpp, `_n_past` tells the backend the current position in the KV cache, enabling out-of-order or non-contiguous token processing. The shim ignores it entirely. While most callers pass contiguous tokens (making `_n_past` equal to the accumulated count), some advanced usage patterns (e.g., parallel sequence processing, batched inference) rely on `_n_past` for correct positioning.

## Fix

Either document the limitation or validate that `_n_past == token_buf.len()` and log a warning if not.
