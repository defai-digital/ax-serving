# Bug Summary — Round 9 (Resolved 2026-05-24)

**Original report:** 2026-04-11, head commit `375e842f`  
**Resolved:** 2026-05-24

## Fixed Bugs

| ID | File(s) | Description | Fix |
|----|---------|-------------|-----|
| 137 | `llamacpp.rs:935`, `mlx.rs:915` | Circuit breaker `was_half_open` read with `Relaxed` ordering; stale read could re-trip after successful recovery | Changed to `Ordering::Acquire` |
| 138 | `llamacpp.rs:1206,1253` | Health poller stop-flag reads used `Relaxed`; missing stop check between kill and spawn could orphan a child process | Changed to `Ordering::Acquire`; added stop check after kill, before spawn |
| 139 | `inference.rs:904` | Streaming tool-call chunk sent `finish_reason: "tool_calls"` alongside the delta, creating double-finish (protocol violation) | Changed to `finish_reason: None`; Done event remains the sole finish chunk |
| 140 | `nats_worker.rs:549` | NATS chunk publish failure broke without setting `stream_error`; done sentinel reported 200 on truncated streams | Set `stream_error` before `break` |
| 141 | `nats_worker.rs:526` | Per-chunk timeout reset on every iteration allowed unbounded total streaming duration | Replaced with sliding deadline (`Instant::now() + request_timeout`) |
| 142 | `inference.rs:551,1341` | `cache_follower_waiting` metric leaked +1 on client disconnect at `rx.recv().await` | Added `FollowerGuard` RAII struct; removed manual `fetch_sub` calls |
| 145 | `llamacpp.rs`, `mlx.rs` | `BlockingExecutor._workers` JoinHandles dropped without joining; threads detached on shutdown | Changed `tx` to `Option<Sender>`, added `impl Drop` that closes channel and joins all workers |
| 151 | `libllama.rs` | `llama_backend_init()` called with no matching `llama_backend_free()`; GPU/metal resources leaked on backend drop | Added `BackendGuard` RAII struct stored in `LibLlamaBackend` |

## Closed as False Positive / Won't Fix

| ID | Reason |
|----|--------|
| 136 | False positive — `llama_get_logits` is already before `mod tests` in current code |
| 143 | Refactoring request (magic number extraction), not a correctness bug |
| 144 | Mitigated by BUG-138 fix (Acquire ordering); `std::thread` has no `join_timeout` in stable Rust |
| 146 | Code duplication refactoring, not a correctness bug |
| 147 | Complex refactoring; scope too large for targeted fix; overlaps 144/145 which are resolved |
| 148 | Low severity observability improvement; thermal fallbacks are intentional and safe |
| 149 | Partially addressed by BUG-145 (executor join); libllama generate threads are fire-and-forget by design — pool acquire/release ensures correct cleanup |
| 150 | Low severity observability improvement; silent parse fallbacks are safe and documented |
