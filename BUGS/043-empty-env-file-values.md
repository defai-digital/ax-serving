# BUG-043: Empty Env-File Values Treated as Present, Causing False Capability Mismatches

**Severity:** Low
**File:** `crates/ax-serving-cli/src/thor.rs`
**Lines:** 127–135, 730
**Status:** ✅ FIXED (2026-03-29)

## Description

`ThorEnvFile::read()` stores key-value pairs where the value is an empty string (e.g., `AXS_THOR_BACKEND=` becomes `key="AXS_THOR_BACKEND", value=""`). Unlike `set()`, which removes empty values, `read()` preserves them. On line 730, `env_file.get("AXS_THOR_BACKEND").unwrap_or("sglang")` returns `Some("")` for an empty value, which is truthy, so the fallback to `"sglang"` is skipped.

## Impact

`thor status` and `thor wait-ready` report `config_mismatch=capability_mismatch` when the env file has an empty `AXS_THOR_BACKEND=` line, even though the agent is running correctly with sglang.

## Fix

In `ThorEnvFile::read()`, skip empty values:
```rust
let value = value.trim().to_string();
if value.is_empty() {
    continue; // treat as unset
}
values.insert(key.trim().to_string(), value);
```

## Fix Applied
In `ThorEnvFile::read()`, skip entries where the trimmed value is empty (`continue` if `trimmed_value.is_empty()`), so `.get()` returns `None` and the fallback default applies.
