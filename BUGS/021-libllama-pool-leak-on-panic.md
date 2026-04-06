# BUG-021: LibLLama Context Pool Leak on FFI Panic Causes Unload Deadlock

**Severity:** High
**File:** `crates/ax-serving-engine/src/libllama.rs`
**Lines:** 523–528 (generate), 614–696 (embed)
**Status:** ✅ FIXED (2026-03-29)

## Description

If the FFI calls inside `run_generate` or the embed closure panic (e.g., due to a bug in llama.cpp), the context is never released back to the pool. The `pool.release(ctx)` call is never reached:

```rust
// generate (line 523-528):
let ctx = holder.pool.acquire();
let result = run_generate(holder.model.0, ctx.0, &holder.meta, &input, &params, &tx);
holder.pool.release(ctx); // never reached if run_generate panics

// embed (line 614-696):
let ctx = holder.pool.acquire();
let result = (|| -> Result<EmbedResult> { ... })();
holder.pool.release(ctx); // never reached if closure panics
result
```

Subsequently, `drain_all()` (called from `LlamaModelHolder::drop` during unload) will wait forever because `guard.len() < self.total` is never satisfied.

## Impact

A single FFI panic causes the model to be permanently unloadable and blocks the calling thread indefinitely, effectively requiring a process restart.

## Fix

Wrap the acquire/use/release in a `scopeguard::guard` or use a `defer!` pattern to guarantee release on panic:

```rust
let ctx = holder.pool.acquire();
let ctx = scopeguard::guard(ctx, |ctx| holder.pool.release(ctx));
let result = run_generate(holder.model.0, ctx.0, ...);
// ctx released via scopeguard on panic or normal exit
```

Alternatively, use `std::panic::catch_unwind` around the FFI calls.

## Fix Applied
Wrapped `run_generate` in `std::panic::catch_unwind(AssertUnwindSafe(...))`. On panic, the context is still released via `pool.release(ctx)` which now executes unconditionally after the catch_unwind. Panic payload is forwarded as `GenerateEvent::Error`.
