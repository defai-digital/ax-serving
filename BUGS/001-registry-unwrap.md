# BUG-001: Unwrap on HashMap Remove in Registry Eviction

**Severity:** Low (False Positive)
**File:** `crates/ax-serving-api/src/registry.rs`
**Line:** 203
**Status:** ❌ FALSE POSITIVE — No fix needed

## Description

```rust
if let Some(id) = oldest_id {
    let evicted = guard.remove(&id).unwrap();
    candidates.push(evicted);
}
```

## Analysis

The `oldest_id` comes from `guard.iter().min_by_key(...)` while holding the write lock. Since we hold the write lock, no other thread can modify the map, so the entry must always exist.

## Verdict

**False Positive** — The write lock is held across the entire while-loop body, including both the `iter()` call and the subsequent `remove()` call. No other thread can insert or remove between these two operations. The `.unwrap()` is safe by construction; the invariant is sound.
