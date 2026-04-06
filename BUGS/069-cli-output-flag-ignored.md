# BUG-069: CLI `--output` Flag Ignored When File Exists

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/tune.rs:206-219`
**Status:** ✅ FIXED (2026-03-29)

## Description

When the user runs `ax-serving tune --output custom.toml` and `custom.toml` already exists, the code writes to `serving.tune.toml` instead -- ignoring the user's explicit output path. The fallback logic checks `path.exists()` but the alternate path is hardcoded to `"serving.tune.toml"`, losing the user's intended filename.

## Impact

If a user specifies `--output /etc/ax-serving/production.toml` and that file exists, the config is silently written to `./serving.tune.toml` instead. Surprising and potentially dangerous in production.

## Fix

When `--output` is explicitly provided, either overwrite the file (with warning) or error out. Only use the `serving.tune.toml` fallback when the default `serving.toml` path is used.

## Fix Applied
Track whether `--output` was explicitly provided. Only use the `serving.tune.toml` fallback when the default path was used; explicit paths are respected (overwritten if they exist).
