# BUG-097: Response headers from sglang not forwarded to client

**Severity:** Medium  
**File:** `crates/ax-thor-agent/src/proxy.rs:155-163` (streaming), `167-171` (non-streaming)  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

The proxy constructs the response by forwarding only the `content-type` header from sglang. All other response headers (`x-request-id`, `cache-control`, `vary`, `x-accel-buffering`, etc.) are silently dropped.

BUG-047 covers request-direction header forwarding; this is the response direction.

## Why It's A Bug

Missing `cache-control: no-cache` causes intermediate proxies (nginx, CloudFront) to buffer the SSE response, breaking real-time streaming for clients. Dropped `x-request-id` breaks distributed tracing correlation.

## Suggested Fix

Forward all response headers (or a curated allowlist) from the sglang response to the client response.

## Fix Applied
Forward all response headers from sglang to the client in both streaming and non-streaming paths, instead of only forwarding content-type.
