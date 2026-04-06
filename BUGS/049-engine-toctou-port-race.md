# BUG-049: TOCTOU Port Race in llama-server Spawn

**Severity:** High
**File:** `crates/ax-serving-engine/src/llamacpp.rs:1268-1273`
**Status:** ⏳ DEFERRED

## Description

```rust
fn find_free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(format!("{LLAMACPP_LOCAL_HOST}:0"))?;
    Ok(listener.local_addr()?.port())
    // listener drops here, releasing the port.
    // TOCTOU: negligible for local use.
}
```

`find_free_port()` binds to `:0`, reads the assigned port, then **drops the listener**. Between the drop and llama-server's bind, another process can grab the same port. This causes `wait_ready()` to timeout or connect to the wrong process.

The comment acknowledges the TOCTOU but dismisses it. On a busy system with many model loads/unloads (e.g., during a benchmark or rolling restart), the race window is real.

## Impact

Under high concurrency of model loads, port collisions cause spurious `wait_ready` timeouts, leading to false "model dead" states and unnecessary restart cycles.

## Fix

Pass the bound listener fd to the child process via `SO_REUSEADDR`/`SO_REUSEPORT`, or retry with a new port on bind failure.
