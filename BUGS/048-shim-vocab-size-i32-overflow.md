# BUG-048: Shim `vocab_size` Silent i32 Overflow Causes OOM Panic

**Severity:** Critical
**File:** `crates/ax-serving-shim/src/model.rs:90`
**Status:** ✅ FIXED (2026-03-29)

## Description

```rust
vocab_size: meta.vocab_size as i32,
```

If `meta.vocab_size` (likely `u32` or `usize`) exceeds `i32::MAX` (2,147,483,647), the `as i32` cast wraps to a **negative** value. Downstream in `context.rs:35`, this negative `i32` is cast to `usize`:

```rust
let vocab_size = model_ref.vocab_size as usize;
```

On 64-bit platforms, a negative `i32` becomes a huge `usize` (e.g., -1 -> 18,446,744,073,709,515,615). The `vec![0.0_f32; vocab_size]` on `context.rs:48` then attempts to allocate hundreds of exabytes, causing an **OOM panic** (abort) in production.

## Impact

Any model whose vocabulary metadata reports a value > `i32::MAX` will crash the process immediately on context creation.

## Fix

```rust
let vocab_size: i32 = meta.vocab_size.try_into()
    .context("vocab_size exceeds i32 range")?;
```

## Fix Applied
Added `.min(i32::MAX as u32)` before the `as i32` cast for `vocab_size` in `model.rs`.
