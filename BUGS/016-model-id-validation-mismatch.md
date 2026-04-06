# BUG-016: Unicode vs ASCII Model ID Validation Mismatch

**Severity:** Low
**File:** `crates/ax-serving-api/src/rest/validation.rs:39` vs `crates/ax-serving-api/src/registry.rs:563`
**Status:** ✅ FIXED (2026-03-28)

## Description

The REST layer validates model IDs with `is_ascii_alphanumeric()`, but the registry (used by gRPC and internal code) uses `is_alphanumeric()` which accepts Unicode characters.

**REST (`rest/validation.rs` line 39–41):**
```rust
if !model.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
```

**Registry (`registry.rs` line 563–565):**
```rust
if !id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
```

## Impact

A model loaded via gRPC with a Unicode ID (e.g., `模型-1`) passes registry validation and is loadable via gRPC, but REST clients receive a 422 error when trying to access it. Inconsistent API surface.

## Fix

Change `registry.rs` to use `c.is_ascii_alphanumeric()` to match the REST layer's policy.

## Fix Applied

Changed `registry.rs` `validate_model_id()` from `c.is_alphanumeric()` to `c.is_ascii_alphanumeric()`, matching the REST layer's policy in `validation.rs`.
