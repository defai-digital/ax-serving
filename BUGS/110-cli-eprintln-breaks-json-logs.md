# BUG-110: `eprintln!` used for operational messages breaks JSON log format

**Severity:** Low  
**File:** `crates/ax-serving-cli/src/main.rs:903, 911, 954, 998`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
eprintln!("[ax-serving] worker starting on {host}:{port}");
eprintln!("[ax-serving] preloaded '{model_id}'");
eprintln!("[ax-serving] WARNING: orchestrator registration failed: {e}");
```

When `AXS_LOG_FORMAT=json` is set, `tracing` events are emitted as JSON. These `eprintln!` calls produce unstructured plain text mixed into the JSON stream, breaking log aggregators. BUG-086 (thor.rs eprintln) was already fixed by switching to `tracing::warn!`, but the same issue persists in `main.rs`.

## Why It's A Bug

Mixed-format output breaks structured log pipelines in production.

## Suggested Fix

Replace `eprintln!("[ax-serving] ...")` with `tracing::info!` / `tracing::warn!` calls.

## Fix Applied
Replaced all operational `eprintln!("[ax-serving] ...")` calls with `tracing::info!` / `tracing::warn!` for compatibility with `AXS_LOG_FORMAT=json`.
