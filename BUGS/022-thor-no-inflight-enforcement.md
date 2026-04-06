# BUG-022: Thor Agent Does Not Enforce Inflight Request Limit Locally

**Severity:** High
**File:** `crates/ax-thor-agent/src/proxy.rs`
**Lines:** 29–31, 73
**Status:** ✅ FIXED (2026-03-29)

## Description

The `max_inflight` configuration value is reported to the control plane (in `agent.rs` line 70) but is never enforced locally. `InflightGuard::acquire` unconditionally increments the counter with no check against any upper bound:

```rust
fn acquire(counter: &Arc<AtomicUsize>) -> Self {
    counter.fetch_add(1, Ordering::Relaxed);  // no limit check
    Self(Arc::clone(counter))
}
```

## Impact

The control plane routes traffic assuming the worker handles at most `max_inflight` concurrent requests. If many more arrive, the worker accepts all of them, overloading the sglang backend and causing cascading timeouts. The reported `inflight` value in heartbeats can far exceed `max_inflight`, making the control plane's routing decisions incorrect.

## Fix

Add a compare-and-swap loop that rejects requests when `inflight >= max_inflight`:

```rust
fn try_acquire(counter: &Arc<AtomicUsize>, max: usize) -> Option<Self> {
    loop {
        let current = counter.load(Ordering::Acquire);
        if current >= max {
            return None;
        }
        match counter.compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return Some(Self(Arc::clone(counter))),
            Err(_) => continue,
        }
    }
}
```

Return 503 Service Unavailable when `try_acquire` returns `None`.

## Fix Applied
Replaced `InflightGuard::acquire` (unconditional increment) with `InflightGuard::try_acquire` using a CAS loop that checks `current >= max_inflight`. Returns `None` when at capacity. `proxy_to` returns 503 Service Unavailable when `try_acquire` fails. `ProxyState` now carries `max_inflight` from config.
