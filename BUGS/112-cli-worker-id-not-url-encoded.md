# BUG-112: `worker_id` interpolated into URL path without encoding

**Severity:** Low  
**File:** `crates/ax-serving-cli/src/thor.rs:905-907`, `crates/ax-serving-cli/src/main.rs:827, 850`  
**Status:** ⏳ DEFERRED  
**Introduced:** 2026-03-29

## Description

```rust
let url = format!("{}/internal/workers/{worker_id}/{action}", ...);
```

`worker_id` is interpolated directly into the URL path without URL-encoding. If it contains `/`, `?`, `#`, or spaces, the URL is malformed.

## Why It's A Bug

While worker_ids are typically server-assigned UUIDs, there is no validation. A buggy orchestrator returning a worker_id with special characters would cause silent routing failures.

## Suggested Fix

Validate that `worker_id` contains only safe characters, or use `urlencoding::encode()`.
