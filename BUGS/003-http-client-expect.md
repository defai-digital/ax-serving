# BUG-003: Expect on HTTP Client Build

**Severity:** Low (Startup)
**File:** `crates/ax-serving-engine/src/llamacpp.rs` line 390; `crates/ax-serving-api/src/orchestration/direct.rs` line 128
**Status:** ❌ FALSE POSITIVE — No fix needed (startup-only, acceptable panic)

## Code

```rust
// llamacpp.rs:390
.build()
.expect("failed to build reqwest blocking client");

// direct.rs:128
.build()
.expect("failed to build reqwest client"),
```

## Analysis

`reqwest::Client::builder().build()` returns `Err` only if TLS library initialisation fails — practically never, and unrecoverable at runtime since the application cannot make HTTP calls without a client. Both are single startup calls in constructors.

## Verdict

**False Positive / Acceptable** — Startup-only constructor called once per process lifetime. `.expect()` with a clear message is idiomatic Rust for this pattern. Same class as BUG-002. Established as acceptable in PRD-BUG-CORRECTION-PLAN-v3.0 (BUG-023).
