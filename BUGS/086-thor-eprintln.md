# BUG-086: Thor Uses `eprintln!` Instead of Tracing

**Severity:** Low
**File:** `crates/ax-thor-agent/src/config.rs:49-53`
**Status:** ✅ FIXED (2026-03-29)

## Description

When the advertised address is an unspecified IP (wildcard), a warning is printed via `eprintln!` rather than the tracing subsystem. This bypasses structured logging. If log output is redirected or consumed by a log aggregator, this message is lost.

## Fix

Replace with `tracing::warn!` since the tracing subscriber is already initialized at the point `from_env()` is called.

## Fix Applied
Replaced `eprintln!("WARNING: ...")` with `tracing::warn!(...)` for the wildcard advertised address warning in `config.rs`.
