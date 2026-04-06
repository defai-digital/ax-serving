# BUG-117: `mixed.rs` worst-case P99 folds with `NEG_INFINITY` when `all_results` empty

**Severity:** Low  
**File:** `crates/ax-serving-bench/src/mixed.rs:122-145`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
.fold(f64::NEG_INFINITY, f64::max)
```

If `all_results` is somehow empty, all P99 fields produce `f64::NEG_INFINITY`, serialized as `-inf` in JSON output.

## Why It's A Bug

`-inf` in JSON output breaks JSON parsers and is semantically incorrect.

## Suggested Fix

Use `.reduce(f64::max)` which returns `None` on empty, and handle the `None` case explicitly.

## Fix Applied
Changed `.fold(f64::NEG_INFINITY, f64::max)` to `.reduce(f64::max)` which returns `None` on empty iterators.
