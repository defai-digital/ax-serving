# BUG-009: Mutex Held During Blocking HTTP Calls in `cache_telemetry`

**Severity:** High
**File:** `crates/ax-serving-engine/src/llamacpp.rs`
**Lines:** 1050–1081
**Status:** ✅ FIXED (2026-03-28)

## Description

`cache_telemetry()` acquires the models `Mutex` and makes blocking HTTP requests to every loaded llama-server process while holding it. With N loaded models, the mutex is held for up to N seconds (1-second timeout per request).

```rust
fn cache_telemetry(&self) -> CacheTelemetry {
    let guard = self.models_read();  // acquires Mutex
    let mut total = CacheTelemetry::default();
    for proc in guard.values() {
        let url = format!("http://{LLAMACPP_LOCAL_HOST}:{}/health", proc.port);
        let resp = match self.http.get(&url).timeout(Duration::from_secs(1)).send() {
            // ...
        };
        // ...
    }
    total
}  // guard dropped here
```

The `models` field is `Arc<Mutex<HashMap<...>>>` (not `RwLock`). All other operations (`generate`, `load_model`, `unload_model`, `tokenize`, `decode_tokens`, `embed`) acquire the same mutex.

## Impact

Causes head-of-line blocking across all model operations. When telemetry is collected, every concurrent request to any model served by this backend instance blocks for up to N seconds. With 4 loaded models, this is a 4-second latency spike for all inference requests.

## Fix

Clone the port list from under the lock, then make HTTP calls after releasing it:

```rust
fn cache_telemetry(&self) -> CacheTelemetry {
    let ports: Vec<u16> = {
        let guard = self.models_read();
        guard.values().map(|p| p.port).collect()
    }; // lock released
    let mut total = CacheTelemetry::default();
    for port in ports {
        // ... HTTP calls without lock held
    }
    total
}
```

## Fix Applied

Snapshot the port list under the mutex lock, then drop the lock before making any HTTP calls. The HTTP health-check loop now iterates over the cloned `Vec<u16>` of ports instead of holding the models guard.
