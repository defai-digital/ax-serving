# BUG-065: GGUF Parser Silently Truncates at 512 KV Pairs

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/gguf_meta.rs:119`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
let limit = n_kv.min(512)
```

Silently drops KV pairs beyond index 512. If `general.architecture` appears after position 512 (unusual but valid per the GGUF format), architecture detection fails and defaults to empty, causing routing to fall back to filename heuristics.

## Fix

Increase to 2048 or log a warning when the cap is hit.

## Fix Applied
Increased KV pair iteration cap from 512 to 2048.
