# BUG-094: URL normalization accepts query params/fragments, breaks downstream URLs

**Severity:** Medium  
**File:** `crates/ax-serving-cli/src/main.rs:531-549`, `crates/ax-serving-cli/src/thor.rs:44-61`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

Both `normalize_http_base_url` and `normalize_control_plane_url` reject URLs containing `/` in the path but do **not** reject `?` or `#`:

```rust
if rest.contains('/') {
    anyhow::bail!("URL must not include a path");
}
// No check for '?' or '#'
```

A URL like `http://host:19090?foo=bar` passes validation. Downstream URL construction produces `http://host:19090?foo=bar/internal/workers/register` — malformed.

## Why It's A Bug

The normalizer's purpose is to produce safe base URLs for path concatenation. Accepting query strings breaks this contract silently.

## Suggested Fix

Reject `rest` if it contains `?` or `#`:
```rust
if rest.contains('?') || rest.contains('#') {
    anyhow::bail!("URL must not include query params or fragments");
}
```

## Fix Applied
Added rejection of `?` and `#` in URL remainder in both `normalize_http_base_url` (main.rs) and `normalize_control_plane_url` (thor.rs).
