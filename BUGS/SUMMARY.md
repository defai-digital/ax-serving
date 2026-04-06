# Bug Analysis Report

**Generated:** 2026-03-22
**Updated:** 2026-03-29
**Repository:** ax-serving

## Overview

120 issues identified across all crates. 18 false positives. 80 fixed (4 batches). 22 deferred.

## Statistics

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 1 | Fixed |
| High | 22 | 20 Fixed, 2 Deferred |
| Medium | 45 | 36 Fixed, 9 Deferred |
| Low | 34 | 23 Fixed, 2 Deferred, 9 FP |
| False Positive | 18 | No fix needed |
| **Total** | **120** | |

---

## False Positives (18)

| ID | Issue | Reason |
|----|-------|--------|
| BUG-001 | `.unwrap()` on HashMap remove | Write lock invariant |
| BUG-002 | `.expect()` on runtime build | Startup-only |
| BUG-003 | `.expect()` on HTTP client | Startup-only |
| BUG-004 | Silent job deletion | Returns `Option` |
| BUG-006 | JSON OOB indexing | serde_json null-safe |
| BUG-020 | eval_tokens fresh KV | Shim works correctly |
| BUG-036 | Arc dup use-after-unload | Caller contract |
| BUG-058 | Context invalidated handle | Duplicate of 036 |
| BUG-062 | Float precision memory | Negligible |
| BUG-063 | Circuit breaker Relaxed | Acceptable for heuristic |
| BUG-064 | unix_ms pre-epoch | System misconfig |
| BUG-080 | Seed defaults u64::MAX | Design choice |
| BUG-082 | Registration warning | Design choice |
| BUG-083 | URL scheme casing | Unreachable |
| BUG-084 | f32→f64 precision | Negligible |
| BUG-088 | n_past ignored | Documented limitation |
| BUG-108 | RPS dispatch overhead | Standard measurement |
| BUG-119 | Seed ignored in shim | No RNG in shim sampling |

---

## Fixed — Batch 1 (14, 2026-03-28): BUG-005,007–019

## Fixed — Batch 2 (23, 2026-03-29): BUG-021–031,034–035,037–040,042–047

## Fixed — Batch 3 (23, 2026-03-29): BUG-048,050,052–054,056–057,059–060,065,067–072,074–077,085–086,089

## Fixed — Batch 4 (20, 2026-03-29)

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| BUG-091 | High | No drain on start_servers error | Cleanup runs unconditionally |
| BUG-092 | High | No SIGTERM handler | Added SIGTERM via `tokio::select!` |
| BUG-093 | Medium | Thermal parse fails on `%` | Strip non-digit suffix |
| BUG-094 | Medium | URL accepts query/fragment | Reject `?` and `#` |
| BUG-095 | Medium | agent_health.is_ok() unchecked | Added to ready condition |
| BUG-097 | Medium | Response headers dropped | Forward all sglang headers |
| BUG-098 | Medium | AxEngine generate panic | `catch_unwind` + Error event |
| BUG-099 | Medium | Drain errors discarded | Log with `tracing::warn!` |
| BUG-101 | Medium | Greedy returns 0 on error | Return -1 (llama.cpp compat) |
| BUG-102 | Medium | top_k sorted flag stale | Set `sorted=false` on k<=0 |
| BUG-104 | Medium | service_perf JoinError missed | Log panics (BUG-089 pattern) |
| BUG-105 | Medium | Mixed gate no error return | `anyhow::ensure!` on gate fail |
| BUG-109 | Low | TTFT truncated to 0ms | Round instead of truncate |
| BUG-110 | Low | eprintln breaks JSON logs | Replaced with `tracing::*` |
| BUG-111 | Low | Redundant health probes | Derive from agent_health |
| BUG-113 | Low | Case-sensitive SSE detect | `to_ascii_lowercase()` |
| BUG-116 | Low | Cache warm avg truncated | f64 average |
| BUG-117 | Low | fold NEG_INFINITY on empty | `.reduce()` returns None |
| BUG-118 | Low | NaN temperature corrupts | Guard `temp.is_nan()` |
| BUG-120 | Low | top_p(1.0) truncates | Early return on p>=1.0 |

---

## Deferred (22)

| ID | Severity | Issue | Reason |
|----|----------|-------|--------|
| BUG-032 | Medium | Non-atomic reregister | Combined RwLock refactor |
| BUG-033 | Medium | Unbounded response body | Streaming refactor |
| BUG-041 | Low | String error matching | Custom error type |
| BUG-049 | High | TOCTOU port race | Pass bound listener fd |
| BUG-051 | High | Logprobs dropped in stop buffer | Buffer logprobs alongside text |
| BUG-055 | High | 300s timeout kills streaming | Per-request timeout strategy |
| BUG-061 | Medium | Unload-reinsert race | Lock restructuring |
| BUG-066 | Medium | Executor drops jobs | Closure refactor |
| BUG-073 | Medium | Hardcoded metrics | sglang metrics integration |
| BUG-078 | Medium | Burst timing contention | Design limitation |
| BUG-079 | Medium | Unbounded memory duration | Handle draining refactor |
| BUG-081 | Low | Poller thread detach | Join-with-timeout in Drop |
| BUG-087 | Low | penalty_last_n==0 | Matches shim contract |
| BUG-090 | Low | Inconsistent percentiles | Cross-subcommand consolidation |
| BUG-096 | Medium | Heartbeat stall 300s | Separate client needed |
| BUG-100 | Medium | response_format ignored | Complex backend mapping |
| BUG-103 | Medium | BOS detection unreliable | Needs backend method |
| BUG-106 | Medium | Wall-clock fallback wrong | Design issue |
| BUG-107 | Medium | Soak cold-start baseline | Design issue |
| BUG-112 | Low | worker_id not URL-encoded | UUIDs in practice |
| BUG-114 | Low | Hardcoded capabilities | Feature gap |
| BUG-115 | Low | Startup deadline ineffective | Connect_timeout interaction |
