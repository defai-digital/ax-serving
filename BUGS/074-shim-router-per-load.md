# BUG-074: Shim New `RouterBackend` Created Per Model Load

**Severity:** Medium
**File:** `crates/ax-serving-shim/src/model.rs:62`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
let backend: Arc<dyn InferenceBackend> = Arc::new(RouterBackend::from_env());
```

Every call to `llama_model_load_from_file` creates a **new** `RouterBackend` instance. If `RouterBackend::from_env()` initializes sub-backends (e.g., GPU contexts, thread pools, shared memory), each model load creates redundant resources. Additionally, if the routing configuration changes between loads (env vars), different models may be served by different backend stacks.

## Impact

Resource waste, and potentially data races if two RouterBackend instances share an underlying GPU context not designed for concurrent access from multiple owners.

## Fix

Use a `OnceLock<Arc<dyn InferenceBackend>>` to share a single backend instance:

```rust
static BACKEND: std::sync::OnceLock<Arc<dyn InferenceBackend>> = std::sync::OnceLock::new();
let backend = BACKEND.get_or_init(|| Arc::new(RouterBackend::from_env())).clone();
```

## Fix Applied
Added `static BACKEND: OnceLock<Arc<dyn InferenceBackend>>` — a single `RouterBackend` is shared across all model loads.
