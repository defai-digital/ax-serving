# BUG-068: CLI `$HOME` Unset Falls Back to CWD

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/thor.rs:442-447`
**Status:** ✅ FIXED (2026-03-29)

## Description

If `HOME` env var is not set, `default_thor_env_path()` falls back to `PathBuf::from(".")`, resulting in a path like `./.config/ax-serving/thor.env`. This writes the config file relative to the current working directory, which is almost certainly not what the user wants.

## Impact

In production environments (containers, systemd), `HOME` might not be set. Silently writing to CWD is confusing and creates unexpected files.

## Fix

Return an error or use a sensible fallback like `/tmp/ax-serving-thor.env` with a warning.

## Fix Applied
When `$HOME` is unset, print a WARNING and fall back to `/tmp` instead of `.` (CWD).
