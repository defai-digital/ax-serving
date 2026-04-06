# BUG-052: `eos_tokens` Silently Returns Default for Invalid Handles

**Severity:** High
**File:** `crates/ax-serving-engine/src/llamacpp.rs:977-981`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
    let guard = self.models_read();
    let eos = guard.get(&handle).map(|p| p.eos_token).unwrap_or(2);
    Ok(vec![eos])
}
```

Unlike every other method in `LlamaCppBackend` that returns an error for invalid handles, `eos_tokens()` silently returns `Ok(vec![2])` (hardcoded EOS=2) when the handle is not found. Callers believe the model exists and use token 2 as EOS, which is incorrect for most non-LLaMA models.

## Impact

If an invalid handle is passed (due to a race with unload, or a programming error), the caller uses EOS token ID 2, which may be a valid but incorrect token (e.g., a punctuation mark or subword), causing premature or missed stop-sequence detection.

## Fix

Return an error for missing handles, consistent with other methods:

```rust
fn eos_tokens(&self, handle: ModelHandle) -> Result<Vec<u32>> {
    let guard = self.models_read();
    let proc = guard.get(&handle)
        .ok_or_else(|| anyhow::anyhow!("invalid llama.cpp model handle {:?}", handle))?;
    Ok(vec![proc.eos_token])
}
```

## Fix Applied
Changed `eos_tokens()` to return `Err("unknown model handle")` when the handle is not found, instead of silently returning `Ok(vec![2])`.
