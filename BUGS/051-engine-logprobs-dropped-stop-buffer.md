# BUG-051: Logprobs Dropped During Stop-Sequence Buffering

**Severity:** High
**File:** `crates/ax-serving-engine/src/ax_engine.rs:515-532`
**Status:** ⏳ DEFERRED

## Description

In the logprobs branch of the decode callback, when `consume_stop_piece` buffers the token text (returns `emit: ""`), the token's logprob event is also suppressed:

```rust
let action = consume_stop_piece(&mut stop_buffer, &piece, &params.stop_seqs);
if !action.emit.is_empty() {
    if tx.blocking_send(GenerateEvent::Token(action.emit)).is_err() {
        anyhow::bail!("receiver dropped");
    }
    if let Some(info) = info
        && tx
            .blocking_send(GenerateEvent::TokenLogprob { ... })
            .is_err()
    {
        anyhow::bail!("receiver dropped");
    }
}
```

When the buffer later flushes, the accumulated text is emitted with only the *current* token's logprob. Tokens whose text was held in the stop buffer have their logprobs permanently lost.

## Impact

Violates the documented `Token -> TokenLogprob` 1:1 invariant. Clients requesting `logprobs` receive fewer `TokenLogprob` events than `Token` events, causing index misalignment and incorrect probability reporting.

## Fix

Always emit `TokenLogprob` for the current token regardless of stop-buffer state. Buffer only the token text for stop matching, not the logprob metadata.
