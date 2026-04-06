# BUG-103: `llama_model_load_from_file` BOS token detection unreliable

**Severity:** Medium  
**File:** `crates/ax-serving-shim/src/model.rs:76-81`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

```rust
let bos_token = backend
    .tokenize(handle, "", true)
    .ok()
    .and_then(|v| v.first().copied())
    .map(|t| t as i32)
    .unwrap_or(-1);
```

The code assumes tokenizing an empty string with `add_bos=true` will return exactly one token (BOS). Some tokenizers return an empty vec for `""` even with `add_bos=true`, or return multiple tokens. If empty: `bos_token = -1`. If multiple: `v.first()` may not be BOS.

## Why It's A Bug

Wrong BOS token silently breaks prompt formatting for callers that prepend BOS.

## Suggested Fix

Query BOS token from model metadata/GGUF directly, or add a dedicated `bos_token()` method to `InferenceBackend`.
