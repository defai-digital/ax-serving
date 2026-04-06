# BUG-012: EOS Token Hardcoded to ID 2 for All Llama.cpp Models

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/llamacpp.rs`
**Lines:** 959–961
**Status:** ✅ FIXED (2026-03-28)

## Description

The llama.cpp backend returns a hardcoded EOS token ID of `2` regardless of the model.

```rust
fn eos_tokens(&self, _handle: ModelHandle) -> Result<Vec<u32>> {
    Ok(vec![2])
}
```

The `_handle` parameter is ignored — the actual model's EOS token is never queried from the llama-server process. The libllama and ax-engine backends both query the actual EOS token from the model's vocabulary.

## Impact

Wrong EOS token for any model that uses a non-standard EOS ID. Models with custom vocabularies (some fine-tuned models, non-LLaMA architectures) may use different EOS tokens, causing premature or missed stop conditions.

## Fix

Query the actual EOS token from the model at load time (from GGUF metadata or llama-server `/props` endpoint) and store it in `LlamaCppProcess`. Return the stored value in `eos_tokens()`.

## Fix Applied

Added `eos_token: u32` field to `LlamaCppProcess`. At load time, the EOS token is queried from the llama-server `/props` endpoint (`default_generation_settings.eos_token_id` or top-level `eos_token_id`), falling back to 2 if the query fails. `eos_tokens()` now looks up the stored per-model value via the models map.
