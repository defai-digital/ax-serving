# BUG-027: TOML Content Written to `.yaml` File Extension

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/tune.rs`
**Lines:** 206, 212
**Status:** ✅ FIXED (2026-03-29)

## Description

The `run_tune()` function generates TOML-formatted content via `rec.to_toml()`, but the default output path is `PathBuf::from("serving.yaml")`. The fallback path on line 212 is `PathBuf::from("serving.tune.yaml")` — also `.yaml`:

```rust
let path = output.unwrap_or_else(|| PathBuf::from("serving.yaml"));
if path.exists() {
    let alt = PathBuf::from("serving.tune.yaml");
    std::fs::write(&alt, &toml).context("failed to write serving.tune.yaml")?;
}
```

The content is unambiguously TOML (key-value pairs, `#` comments, no YAML markers).

## Impact

Users running `ax-serving tune` get a config file with a `.yaml` extension containing TOML syntax. If `ax-serving serve` later tries to load this file, it may parse it as YAML (based on the `.yaml` extension), producing a parse error or silently misinterpreting the config.

## Fix

Change the default extension from `.yaml` to `.toml`:

```rust
let path = output.unwrap_or_else(|| PathBuf::from("serving.toml"));
// and fallback:
let alt = PathBuf::from("serving.tune.toml");
```

## Fix Applied
Changed default output path from `serving.yaml` to `serving.toml` and fallback from `serving.tune.yaml` to `serving.tune.toml`.
