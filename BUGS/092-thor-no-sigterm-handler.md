# BUG-092: No SIGTERM signal handler — graceful shutdown unreachable in containers

**Severity:** High  
**File:** `crates/ax-thor-agent/src/main.rs:43-46`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

Only `tokio::signal::ctrl_c()` (SIGINT) is handled. In containerized deployments (Kubernetes, Docker, systemd), the standard shutdown signal is SIGTERM. Without a SIGTERM handler, the process is killed immediately without triggering the graceful drain sequence (drain → wait for inflight → drain_complete).

## Why It's A Bug

The entire graceful shutdown logic (drain → wait for inflight → drain_complete) is dead code in production deployments. K8s sends SIGTERM, waits `terminationGracePeriodSeconds`, then SIGKILL. The thor agent never drains inflight requests — it's force-killed.

## Suggested Fix

```rust
#[cfg(unix)]
let shutdown = async {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = sigterm.recv() => {},
    }
};
```

## Fix Applied
Added `tokio::signal::unix::signal(SignalKind::terminate())` SIGTERM handler alongside SIGINT via `tokio::select!`. Gated with `#[cfg(unix)]` for portability.
