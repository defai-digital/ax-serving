# BUG-046: Shim i32 Overflow in Tokenize Buffer-Size-Hint Return

**Severity:** Low
**File:** `crates/ax-serving-shim/src/tokenize.rs`
**Lines:** 46, 87
**Status:** ✅ FIXED (2026-03-29)

## Description

`return -(ids.len() as i32)`: if `ids.len()` exceeds `i32::MAX` (2,147,483,647), the cast overflows and the negative value wraps, producing a positive number instead of the intended negative hint:

```rust
if ids.len() > n_tokens_max as usize {
    return -(ids.len() as i32); // overflow if ids.len() > i32::MAX
}
```

## Impact

A C caller that sends an extremely long text could receive a positive return value and believe it succeeded, reading uninitialized memory. In practice, tokenizing 2+ billion tokens is unrealistic, but the overflow is undefined behavior per Rust rules (in release mode it wraps, in debug it panics).

## Fix

Clamp: `return -((ids.len().min(i32::MAX as usize)) as i32);`

## Fix Applied
Clamped with `.min(i32::MAX as usize)` before the `as i32` cast in both `llama_tokenize` and `llama_token_to_piece` overflow paths.
