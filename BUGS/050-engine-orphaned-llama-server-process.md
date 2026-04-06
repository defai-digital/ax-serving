# BUG-050: Orphaned llama-server Process When Restart Fails in Health Poller

**Severity:** High
**File:** `crates/ax-serving-engine/src/llamacpp.rs:1232-1259`
**Status:** ✅ FIXED (2026-03-29)

## Description

When `spawn_server` succeeds (line 1232) but `wait_ready` fails after restart (line 1256), the health poller marks the model as Dead and breaks:

```rust
Err(e) => {
    warn!("llama-server port {port} failed to start after restart: {e}");
    health.store(HealthState::Dead as u8, Ordering::Relaxed);
    break;
}
```

The newly spawned child process is stored in `Arc<Mutex<Option<Child>>>` at line 1233 but never killed. Since the model is Dead, callers may never call `unload_model`, leaving a zombie llama-server running indefinitely.

## Impact

Accumulation of orphaned llama-server processes that consume GPU memory and system resources, eventually causing OOM or device exhaustion.

## Fix

Kill the child before breaking on `wait_ready` failure:

```rust
Err(e) => {
    warn!("llama-server port {port} failed to start after restart: {e}");
    let mut guard = child.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(mut c) = guard.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    health.store(HealthState::Dead as u8, Ordering::Relaxed);
    break;
}
```

## Fix Applied
In the health poller, when `wait_ready` fails after a successful `spawn_server`, kill the child process via `guard.as_mut().kill() + wait()` before marking the model Dead.
