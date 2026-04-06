# BUG-080: Seed Defaults to `u64::MAX` in ax-engine Sampling

**Severity:** Low
**File:** `crates/ax-serving-engine/src/ax_engine.rs:257`
**Status:** ❌ FALSE POSITIVE

## Description

`seed: params.seed.unwrap_or(u64::MAX)` uses a fixed seed when none is provided, making all identical requests deterministically produce the same output. This is surprising for users who expect random sampling by default.

## Fix

Use a random seed source or the current timestamp as default.
