# BUG-008: Stop Sequence Text Leaks Into Output in LibLlama Backend

**Severity:** High
**File:** `crates/ax-serving-engine/src/libllama.rs`
**Lines:** 1043–1083
**Status:** ✅ FIXED (2026-03-28)

## Description

The libllama backend emits token text to the client **before** checking stop sequences. If a stop sequence spans multiple tokens, the partial stop sequence text is sent to the client.

```rust
// Line 1043-1044: Token text emitted immediately
let piece = unsafe { token_to_piece(vocab, next_tok) };
accumulated.push_str(&piece);

// Line 1046-1072: Token is SENT to client
if emit_logprobs {
    if tx.blocking_send(GenerateEvent::Token(piece.clone())).is_err() {
        break;
    }
    // ...
} else if !push_stream_token_piece(tx, piece, ...) {
    break;
}

// Line 1076-1083: Stop sequence checked AFTER emission
if params.stop_seqs.iter().any(|seq| accumulated.ends_with(seq.as_str())) {
    break;
}
```

Contrast with `ax_engine.rs` which uses `consume_stop_piece` to buffer potential stop sequence prefixes and only emit text that cannot be part of a stop sequence.

## Impact

When both `stop` sequences and streaming are used via the libllama backend, partial stop sequence text is exposed to the client. This violates the expected contract that stop sequences are not included in output. Behavioral inconsistency between backends.

## Fix

Port the `consume_stop_piece` buffering logic from `ax_engine.rs` into the libllama generation loop, or restructure to check stop sequences before emitting tokens.

## Fix Applied

Moved the stop-sequence check (`accumulated.ends_with(seq)`) before token emission in the decode loop. If a stop sequence is detected, the loop breaks without emitting the final token piece that completed the stop sequence.
