# BUG-061: Unload-Then-Reinsert Race in RouterBackend

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/routing.rs:530-555`
**Status:** ⏳ DEFERRED

## Description

`unload_model` removes the entry, calls the backend's unload, then re-inserts on failure:

```rust
fn unload_model(&self, handle: ModelHandle) -> Result<()> {
    let entry = self.entries_write().remove(&handle)?;
    let result = match entry.tag { /* backend.unload_model */ };
    if result.is_err() {
        self.entries_write().insert(handle, entry);
    }
    result
}
```

Between the remove and re-insert, the write lock is released twice. Concurrent `generate`/`tokenize` calls on the same handle will fail with "unknown router handle" even though the model is still loaded in the backend.

## Fix

Use a single write-lock scope or mark the entry as "unloading" instead of removing it.
