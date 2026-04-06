# BUG-033: Thor Proxy Unbounded Response Body Read

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/proxy.rs`
**Lines:** 123
**Status:** ⏳ DEFERRED

## Description

In the non-streaming response path, `resp.bytes().await` reads the entire response body into memory with no size limit:

```rust
match resp.bytes().await {  // no size bound
    Ok(bytes) => ...
}
```

While the reqwest client has a 300-second total timeout (set in `main.rs` line 19), there is no response body size limit.

## Impact

A buggy or compromised sglang backend that returns an extremely large response body (e.g., a multi-gigabyte JSON payload) will cause the proxy to allocate all that memory and likely OOM-crash the agent process.

## Fix

Add a response body size limit:

```rust
const MAX_RESPONSE_BODY: usize = 64 * 1024 * 1024; // 64 MiB
let bytes = resp.bytes().await?;
if bytes.len() > MAX_RESPONSE_BODY {
    return (StatusCode::BAD_GATEWAY, "response too large").into_response();
}
```
