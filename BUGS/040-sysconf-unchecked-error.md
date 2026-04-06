# BUG-040: `sysconf` Return Value Not Checked for Error in Memory Budget

**Severity:** Medium
**File:** `crates/ax-serving-engine/src/memory.rs`
**Lines:** 51
**Status:** ✅ FIXED (2026-03-29)

## Description

`libc::sysconf` returns `c_long` and returns `-1` on error. Without checking, `-1i64 as u64` produces `18446744073709551615`, which when multiplied by the page count produces an astronomically large "available memory" value:

```rust
let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) as u64 };
let free = stats.free_count as u64 + stats.inactive_count as u64;
return free * page_size;
```

## Impact

The memory budget check `available < required` always passes, allowing potentially OOM-inducing model loads. On macOS with `_SC_PAGE_SIZE`, this should never fail in practice, but if it does, the safety check becomes a no-op.

## Fix

```rust
let page_size_raw = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) };
if page_size_raw <= 0 {
    return 8 * 1024 * 1024 * 1024; // fallback: assume 8GB
}
let page_size = page_size_raw as u64;
```

## Fix Applied
Check `sysconf` return value: if <= 0, use 4096 as fallback page size instead of reinterpreting -1 as a large unsigned value.
