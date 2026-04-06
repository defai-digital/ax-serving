# BUG-093: Thermal parse fails on `%` suffix, reports "no throttling"

**Severity:** Medium  
**File:** `crates/ax-serving-cli/src/doctor.rs:222-249`  
**Status:** ✅ FIXED (2026-03-29)  
**Introduced:** 2026-03-29

## Description

```rust
if let Some(val) = line.trim().strip_prefix("CPU_Speed_Limit") {
    let val = val.trim().trim_start_matches('=').trim();
    if let Ok(limit) = val.parse::<u32>() {  // fails on "75%"
        return if limit >= 100 { ... };
    }
}
// Falls through to:
CheckResult { status: CheckStatus::Pass, detail: "no thermal throttling detected" }
```

If `pmset -g therm` outputs `CPU_Speed_Limit = 75%` (with `%` suffix), `val.parse::<u32>()` fails. The function returns **Pass** with "no thermal throttling detected" when the system IS throttled at 75%.

## Why It's A Bug

`ax-serving doctor` is meant to detect thermal issues before serving. Silently reporting PASS when the CPU is throttled defeats the purpose of the check.

## Suggested Fix

Strip trailing non-digit characters before parsing:
```rust
val.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>()
```

## Fix Applied
Added `.trim_end_matches(|c: char| !c.is_ascii_digit())` to strip trailing `%` or other non-digit characters before `parse::<u32>()`.
