# BUG-002: Expect on Tokio Runtime Build

**Severity:** Low (Startup)
**File:** `crates/ax-serving-engine/src/backend.rs`
**Line:** 139
**Status:** ❌ FALSE POSITIVE — No fix needed (startup-only, acceptable panic)

## Code

```rust
.build()
.expect("failed to build ax-serving-engine tokio runtime");
```

## Analysis

If tokio runtime fails to build (extremely rare), the application panics. Runtime creation failure indicates a system-level problem (resource exhaustion or OS thread limits). If the tokio runtime cannot be built, there is no other runtime to propagate the error to — there is no viable recovery path.

## Verdict

**False Positive / Acceptable** — Startup-only constructor called once. `.expect()` with a clear message is idiomatic Rust for unrecoverable startup failures. Established as acceptable in PRD-BUG-CORRECTION-PLAN-v3.0 (BUG-023).
