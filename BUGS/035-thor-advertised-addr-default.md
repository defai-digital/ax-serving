# BUG-035: Thor Advertised Address Defaults to `0.0.0.0`

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/config.rs`
**Lines:** 44–47
**Status:** ✅ FIXED (2026-03-29)

## Description

When `AXS_THOR_ADVERTISED_ADDR` is not set, the default is `listen_addr.to_string()`. If the listen address is `0.0.0.0:18081` (the default), the advertised address is also `0.0.0.0:18081`:

```rust
let advertised_addr: SocketAddr = std::env::var("AXS_THOR_ADVERTISED_ADDR")
    .unwrap_or_else(|_| listen_addr.to_string())  // "0.0.0.0:18081"
    .parse()
    .context("invalid AXS_THOR_ADVERTISED_ADDR")?;
```

## Impact

The control plane receives `0.0.0.0:18081` as the worker address. Connecting to `0.0.0.0` is undefined behavior for a client — it typically resolves to `127.0.0.1` on the control plane's host, not the worker's actual IP. The control plane will be unable to route traffic to the worker, causing it to appear permanently unreachable.

## Fix

If `listen_addr` is `0.0.0.0` or `::`, either auto-detect the actual IP and use that as the default advertised address, or require `AXS_THOR_ADVERTISED_ADDR` to be explicitly set when listening on a wildcard address. Emit a warning when defaulting to a non-routable address.

## Fix Applied
Added a warning message via `eprintln!` when the advertised address is a wildcard (unspecified) IP, informing the operator to set `AXS_THOR_ADVERTISED_ADDR`.
