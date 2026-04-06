# BUG-084: CLI f32 to f64 Precision Loss for Sampling Parameters

**Severity:** Low
**File:** `crates/ax-serving-cli/src/main.rs:59, 443`
**Status:** ❌ FALSE POSITIVE

## Description

CLI sampling parameters (`temp`, `top_p`, `repeat_penalty`) are parsed as `f32` by clap and then cast to `f64`. The default `0.7` as f32 is `0.699999988079071` in f64, causing slightly different sampling behavior than intended.

## Fix

Parse temperature, top_p, and repeat_penalty as `f64` directly by changing the CLI struct field types.
