# BUG-055: Global reqwest 300s Timeout Kills Streaming Proxies

**Severity:** High
**File:** `crates/ax-thor-agent/src/main.rs:19`
**Status:** ⏳ DEFERRED

## Description

The `reqwest::Client` is built with `.timeout(Duration::from_secs(300))`. This timeout applies to the **entire request-response cycle**, including reading the response body stream. The same client is used in `proxy.rs` to forward requests to sglang.

## Impact

A streaming `/v1/chat/completions` response that takes >300 seconds to fully generate tokens will have its upstream connection killed by reqwest mid-stream. The client sees a truncated response or error. For long-running batch generations or slow models, this is a common occurrence.

## Fix

Remove the global timeout from the client builder. Set per-request timeouts for non-streaming paths (registration, heartbeat) explicitly, or create two separate clients -- one for short-lived control-plane requests and one without a timeout for proxying.
