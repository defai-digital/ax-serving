# BUG-042: LibLLama Unbounded Stop-Sequence Accumulation String

**Severity:** Low
**File:** `crates/ax-serving-engine/src/libllama.rs`
**Lines:** 1034, 1044, 1049
**Status:** ✅ FIXED (2026-03-29)

## Description

`accumulated` grows unboundedly with every generated token:

```rust
let mut accumulated = String::new();
// ... in loop:
accumulated.push_str(&piece);
if params.stop_seqs.iter().any(|seq| accumulated.ends_with(seq.as_str())) { ... }
```

For long generations (e.g., `max_tokens = 16384`), this wastes memory proportional to output size. The `ax_engine.rs` backend only keeps a small tail buffer (`consume_stop_piece`), bounded by the longest stop sequence length.

## Impact

Wasted memory (potentially megabytes for very long outputs without stop sequences). Not a crash risk.

## Fix

Only keep the tail of `accumulated` up to the length of the longest stop sequence plus one token, similar to `consume_stop_piece` in `ax_engine.rs`.

## Fix Applied
After the stop-sequence check, trim `accumulated` to keep only the last `max_stop_len * 2` bytes (using `ceil_char_boundary` for UTF-8 safety). Bounded memory usage proportional to the longest stop sequence, not output length.
