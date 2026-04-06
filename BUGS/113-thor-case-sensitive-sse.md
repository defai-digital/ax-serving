# BUG-113: Case-sensitive SSE content-type detection

**Severity:** Low  
**File:** `crates/ax-thor-agent/src/proxy.rs:135`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
content_type.starts_with("text/event-stream")
```

Case-sensitive comparison. If sglang returns `Content-Type: Text/Event-Stream` (valid per RFC 7231 — header values are case-insensitive), the streaming path is bypassed. The entire SSE response body is buffered into memory via `resp.bytes().await`, defeating streaming.

Combined with BUG-033 (unbounded response body), this amplifies memory consumption.

## Why It's A Bug

Correctly-cased responses from non-conforming servers bypass streaming silently.

## Suggested Fix

```rust
let is_event_stream = content_type.to_ascii_lowercase().starts_with("text/event-stream");
```

## Fix Applied
Changed `content_type.starts_with("text/event-stream")` to `content_type.to_ascii_lowercase().starts_with("text/event-stream")`.
