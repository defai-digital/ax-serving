# BUG-005: Silent NATS Message Acknowledgment on Parse-Error Path

**Severity:** Low
**File:** `crates/ax-serving-api/src/orchestration/nats_worker.rs`
**Line:** 246
**Status:** ✅ FIXED (2026-03-22)

## Code (before fix)

```rust
let _ = msg.ack().await;
```

## Analysis

The parse-error path used `let _ = msg.ack().await` while the success and nack paths in the same function both used `if let Err(e)` with `warn!`. This inconsistency meant that if the ACK itself failed on the parse-error path, the malformed message would be redelivered indefinitely with no indication that ACK was also failing.

## Fix Applied

```rust
if let Err(ack_err) = msg.ack().await {
    warn!(%model_id, %ack_err, "NatsWorker: ack failed for malformed message");
}
```

Now consistent with the success path (`warn!` on ack failure) and the nack path (`warn!` on nack failure).
