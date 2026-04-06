# BUG-100: `response_format` parameter silently ignored for llama.cpp backend

**Severity:** Medium  
**File:** `crates/ax-serving-engine/src/llamacpp.rs:1363-1429` (`apply_generation_params`)  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

`GenerationParams` has a `response_format: Option<String>` field documented as `"json_object" enforces JSON grammar`. But `apply_generation_params` never reads `params.response_format`. Only `params.grammar` (with the `"__json__"` sentinel) is translated to `response_format` in the HTTP body. If a caller sets `response_format` directly (without the grammar sentinel), the parameter is silently dropped.

The API validation layer (`resolve_grammar`) maps `response_format: json_object` to `grammar: "__json__"`, so the common path works. However, any programmatic caller that sets `response_format` on `GenerationParams` directly (bypassing the REST layer) gets no JSON enforcement.

## Why It's A Bug

API contract violation — a documented parameter is ignored on a direct code path.

## Suggested Fix

Add handling for `params.response_format` in `apply_generation_params`:
```rust
if let Some(ref fmt) = params.response_format {
    if fmt == "json_object" {
        body["response_format"] = json!({"type": "json_object"});
    }
}
```
