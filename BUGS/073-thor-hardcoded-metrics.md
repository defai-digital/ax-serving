# BUG-073: Thor Hardcoded Placeholder Metrics in Heartbeat

**Severity:** Medium
**File:** `crates/ax-thor-agent/src/agent.rs:138-148`
**Status:** ⏳ DEFERRED

## Description

The heartbeat body reports all metrics as hardcoded zeros: `thermal_state: "nominal"`, `rss_bytes: 0`, `decode_tok_per_sec: 0.0`, `ttft_p95_ms: 0`, `queue_depth: 0`, `error_rate: 0.0`. None of these are actually collected from sglang.

## Impact

The control plane receives misleading data. Scheduling decisions based on these metrics (e.g., load balancing, auto-scaling, eviction) will be wrong. The control plane thinks this worker is idle, has no memory usage, and performs perfectly -- all of which are false.

## Fix

Query sglang's metrics endpoint (SGLang exposes Prometheus metrics) and populate these fields, or omit the fields entirely and let the control plane use defaults.
