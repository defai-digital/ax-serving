# BUG-085: Thor Proxy Hardcodes `content-type: application/json`

**Severity:** Low
**File:** `crates/ax-thor-agent/src/proxy.rs:113`
**Status:** ✅ FIXED (2026-03-29)

## Description

The proxy unconditionally sets `content-type: application/json` on the forwarded request to sglang, discarding the original content-type from the client (e.g., `application/json; charset=utf-8`). If a client sends a request with `content-encoding: gzip` and a gzipped body, the proxy would forward the gzipped bytes with `content-type: application/json` and no `content-encoding` header, causing sglang to fail parsing.

## Fix

Forward the original `content-type` from `client_headers`, falling back to `application/json`.

## Fix Applied
Added `content-type` to `FORWARDED_HEADERS`. The hardcoded `application/json` is now only a fallback if the client didn't send a content-type.
