# BUG-047: Thor Proxy Drops All Client Headers When Forwarding

**Severity:** Low
**File:** `crates/ax-thor-agent/src/proxy.rs`
**Lines:** 75–80
**Status:** ✅ FIXED (2026-03-29)

## Description

The proxy hardcodes `content-type: application/json` and drops all other client headers (including `Authorization`, `X-Request-Id`, etc.) when forwarding to sglang:

```rust
state.client.post(url)
    .header("content-type", "application/json")
    .body(body)
    .send()
    .await
```

## Impact

If sglang requires authentication tokens or if tracing relies on request IDs propagated via headers, these are silently lost. This limits observability and could cause auth failures if sglang is configured to require tokens.

## Fix

Forward relevant headers from the original request (at minimum `Authorization`, `X-Request-Id`, and any custom tracing headers).

## Fix Applied
Added `FORWARDED_HEADERS` constant listing `authorization` and `x-request-id`. `proxy_to` now receives `client_headers: &HeaderMap` and forwards matching headers to the sglang request.
