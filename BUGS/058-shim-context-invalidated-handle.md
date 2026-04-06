# BUG-058: Shim Context Uses Invalidated Handle After `llama_free_model`

**Severity:** High
**File:** `crates/ax-serving-shim/src/context.rs:38-46` and `model.rs:159-168`
**Status:** ❌ FALSE POSITIVE

## Description

In `llama_new_context_with_model`, the context clones all `LlamaModel` fields into a **new** `Arc<LlamaModel>`:

```rust
model: Arc::new(LlamaModel {
    handle: model_ref.handle,        // copied handle
    backend: model_ref.backend.clone(),
    ...
}),
```

When `llama_free_model` is called, it:
1. Calls `m.backend.unload_model(m.handle)` -- **invalidates the handle**
2. Drops the original `LlamaModel` box

The context still holds a clone with the same (now-invalid) `handle`. Any subsequent `llama_eval`, `llama_tokenize`, etc. using that context will pass the invalid handle to the backend -- potentially causing crashes, wrong results, or corrupting backend state.

## Impact

Use-after-free (semantic). After freeing a model, any remaining contexts silently operate on an invalidated handle, causing undefined behavior in the backend.

## Fix

Share the `Arc<LlamaModel>` between the model pointer and all contexts. `llama_free_model` should only call `unload_model` when the Arc ref count drops to 1 (no remaining contexts).
