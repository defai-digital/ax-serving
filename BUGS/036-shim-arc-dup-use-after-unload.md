# BUG-036: Shim Arc Duplication Allows Use-After-Unload of Model Handle

**Severity:** Medium
**File:** `crates/ax-serving-shim/src/context.rs`
**Lines:** 38–46
**Status:** ❌ FALSE POSITIVE

## Description

`llama_new_context_with_model` creates a brand new `Arc<LlamaModel>` by copying the fields from the raw-pointer model. This means the context holds a separate heap allocation with a copy of the `ModelHandle` and a clone of the `Arc<dyn InferenceBackend>`. When `llama_free_model` is later called, it calls `backend.unload_model(m.handle)`, which actually unloads the model from the shared backend. The context still holds the same `handle` value — now pointing to an unloaded model:

```rust
let ctx = Box::new(LlamaContext {
    model: Arc::new(LlamaModel {
        handle: model_ref.handle,     // copies handle value
        backend: model_ref.backend.clone(),  // clones Arc to backend
        ...
    }),
    ...
});
```

## Impact

If a C caller frees the model before freeing the context (violating the C API contract), subsequent `llama_eval` calls on that context silently use an invalid handle, causing panics or undefined behavior in the backend.

## Fix

Reference-count the model pointer via `Arc` from the start. Instead of `Box::into_raw(model)`, store the model as `Arc<LlamaModel>` inside the raw pointer (e.g., `Arc::into_raw`), and have the context increment the refcount. This way `llama_free_model` only decrements the refcount and skips unloading if contexts still exist.
