# BUG-083: CLI URL Scheme Not Normalized to Lowercase

**Severity:** Low
**File:** `crates/ax-serving-cli/src/main.rs:521-558`, `thor.rs:63-69`
**Status:** ❌ FALSE POSITIVE

## Description

`normalize_http_base_url` accepts uppercase scheme like `HTTPS://` without normalizing to lowercase. If downstream code does case-sensitive comparisons (e.g., `starts_with("http://")`), an `HTTPS://` URL would not match. The function's purpose is normalization, so preserving original casing is arguably wrong.

## Fix

Normalize the scheme to lowercase: reconstruct with a canonical scheme.
