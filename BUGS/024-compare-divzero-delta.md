# BUG-024: Division by Zero in `delta_str` When Endpoint A Has Zero Latency

**Severity:** High
**File:** `crates/ax-serving-bench/src/compare.rs`
**Lines:** 230–236
**Status:** ✅ FIXED (2026-03-29)

## Description

`delta_str` computes `(b - a) / a * 100.0`. The guard on line 231 only handles the case where both `a` and `b` are zero. If `a == 0.0` and `b != 0.0` (e.g., endpoint A has 0.0ms P99 due to no successful requests in a latency class), this divides by zero:

```rust
fn delta_str(a: f64, b: f64) -> String {
    if a == 0.0 && b == 0.0 {
        return "  N/A".to_string();
    }
    let pct = (b - a) / a * 100.0;  // inf when a == 0.0 && b != 0.0
    format!("{pct:+.0}%")
}
```

## Impact

Produces `+inf%` or `-inf%` in the comparison table output, corrupting the delta display and potentially the JSON output.

## Fix

```rust
fn delta_str(a: f64, b: f64) -> String {
    if a == 0.0 && b == 0.0 {
        return "  N/A".to_string();
    }
    if a == 0.0 {
        return "  new".to_string();
    }
    let pct = (b - a) / a * 100.0;
    format!("{pct:+.0}%")
}
```

## Fix Applied
Added early return: `if a == 0.0 { return "  new".to_string(); }` before the division.
