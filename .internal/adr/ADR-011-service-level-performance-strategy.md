# ADR-011: Service-Level Performance Strategy

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

## Context
ax-serving uses mistralrs as the inference backend through `InferenceBackend`. Single-stream tok/s is primarily determined by backend kernels and quantization path. For production, the critical outcomes are throughput under concurrency, p95/p99 latency, and overload behavior.

Recent benchmarks show that wrapper overhead is small compared with backend behavior, but tail latency and queueing under real load are not yet first-class optimized.

## Decision
Optimize performance at the serving layer rather than attempting to beat raw mistralrs per-stream token speed.

The service-level strategy includes:
1. Central admission queue with bounded depth and explicit overload policy.
2. Dynamic decode batching with bounded wait windows.
3. Prefix-aware cache reuse in the serving layer.
4. Warm model pool with memory-constrained eviction.
5. Backpressure and fairness controls per model/client.
6. Full latency-path observability (queue wait to final token).

## Rationale
1. Highest ROI: production bottlenecks are queueing and contention, not single-kernel speed.
2. Architectural fit: these controls sit naturally in `ax-serving-api` and scheduler paths.
3. Lower risk: avoids forking/maintaining backend kernel logic.
4. Clear SLO impact: directly targets p95/p99 and throughput under load.

## Consequences
### Positive
- Better throughput and lower tail latency for multi-tenant workloads.
- Predictable overload behavior instead of timeout storms.
- Clear operational tuning via `AXS_*` config knobs.

### Negative
- More scheduler complexity and tuning surface.
- Potential fairness/batching regressions if guardrails are weak.
- Additional metrics/storage overhead.

## Non-Goals
- Beating raw mistralrs in single-stream microbenchmarks.
- Custom Metal or Candle kernel development in ax-serving.

## Implementation Boundaries
- Backend contract remains `InferenceBackend`; no backend-specific logic leaks into API layer.
- Keep batching and queue policy in serving/scheduler modules.
- Keep cache policy pluggable and bounded by explicit memory budget.

## Rollout Plan
1. Ship admission queue + overload policy + metrics first.
2. Add batching behind feature/config flag.
3. Add prefix cache and warm pool with conservative defaults.
4. Run staged load tests; enable gradually in production.

## Validation
- Compare against baseline on fixed hardware and model set.
- Required checks:
1. Throughput gain at 8+ concurrent clients.
2. p95/p99 latency improvements under sustained load.
3. Stable error rate and bounded queue under overload.
