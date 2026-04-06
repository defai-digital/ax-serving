# BUG-017: PerModelScheduler DashMap Entries Never Removed on Unload

**Severity:** Low
**File:** `crates/ax-serving-api/src/scheduler.rs`
**Lines:** 759–794
**Status:** ✅ FIXED (2026-03-28)

## Description

`PerModelScheduler::slots` is a `DashMap<String, Arc<Semaphore>>` that grows without bound. Every distinct `model_id` passed to `acquire()` creates a permanent entry. When models are unloaded (idle eviction or explicit unload), the corresponding entry is never removed.

```rust
pub struct PerModelScheduler {
    max_per_model: usize,
    slots: DashMap<String, Arc<Semaphore>>,
}

impl PerModelScheduler {
    pub async fn acquire(&self, model_id: &str, max_wait_ms: u64) -> Result<OwnedSemaphorePermit> {
        let sem = Arc::clone(
            &*self.slots
                .entry(model_id.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(self.max_per_model))),
        );
        // ...
    }
    // No remove() method exists
}
```

## Impact

Repeated load/unload cycles of different models cause unbounded memory growth. Each entry is small (`String` + `Arc<Semaphore>`) but grows without bound.

## Fix

Add a `remove(&self, model_id: &str)` method and call it from the model unload path in the registry.

## Fix Applied

Added `PerModelScheduler::remove(&self, model_id: &str)` method that removes the semaphore entry from the DashMap. Called in two places: (1) `rest_unload_model` after successful unload, (2) the idle eviction loop in `start_servers` after each evicted model ID.
