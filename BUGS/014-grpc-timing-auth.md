# BUG-014: gRPC TCP Auth Uses Timing-Vulnerable Comparison

**Severity:** Medium
**File:** `crates/ax-serving-api/src/grpc/mod.rs`
**Lines:** 63–81
**Status:** ✅ FIXED (2026-03-28)

## Description

The gRPC TCP interceptor validates API keys using `HashSet::contains()`, which performs standard equality comparison that leaks timing information. The REST layer correctly uses constant-time comparison.

**gRPC (vulnerable):**
```rust
.map(|key| keys.contains(key.trim()))  // timing-vulnerable
```

**REST (correct, in `auth.rs`):**
```rust
fn has_valid_api_key(candidate: &str, keys: &HashSet<String>) -> bool {
    keys.iter()
        .any(|expected| constant_time_eq_str(candidate, expected))
}
```

## Impact

An attacker can exploit timing side-channels to recover API keys byte-by-byte through the gRPC TCP endpoint. The REST layer is not affected.

## Fix

Import `crate::auth::has_valid_api_key` and use it in the gRPC interceptor instead of `keys.contains()`.

## Fix Applied

Made `has_valid_api_key` in `auth.rs` `pub(crate)` and replaced `keys.contains(key.trim())` in the gRPC TCP interceptor with `crate::auth::has_valid_api_key(key.trim(), &keys)`, which uses constant-time string comparison via `constant_time_eq_str`.
