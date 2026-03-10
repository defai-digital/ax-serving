# Service Tuning Guide (M3)

This guide covers production-oriented tuning for the new admission controls:

- `AXS_SCHED_MAX_INFLIGHT`
- `AXS_SCHED_MAX_QUEUE`
- `AXS_SCHED_MAX_WAIT_MS`

These settings protect p95/p99 latency while preserving throughput under bursty load.

## Recommended starting defaults

| SKU | `AXS_SCHED_MAX_INFLIGHT` | `AXS_SCHED_MAX_QUEUE` | `AXS_SCHED_MAX_WAIT_MS` |
|---|---:|---:|---:|
| M3 / M3 Air (8-10 GPU cores) | 4 | 24 | 150 |
| M3 Pro | 8 | 64 | 250 |
| M3 Max | 12 | 96 | 300 |
| M3 Ultra | 16 | 128 | 400 |

Notes:
- Prefer lower queue wait for interactive apps; increase only if your clients tolerate extra latency.
- Queue should be large enough for short bursts, but not so large that requests sit and time out.

## Runbook

1. Start server with tuned env vars:

```bash
AXS_SCHED_MAX_INFLIGHT=8 \
AXS_SCHED_MAX_QUEUE=64 \
AXS_SCHED_MAX_WAIT_MS=250 \
cargo run -p ax-serving-cli --release -- serve -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --port 18080
```

2. Run load test:

```bash
python3 scripts/load_test_chat.py \
  --url http://127.0.0.1:18080/v1/chat/completions \
  --model default \
  --requests 200 \
  --concurrency 8 \
  --prompt-tokens 39 \
  --max-tokens 64
```

3. Tune iteratively:
- If `429` is high and p95 is low: raise `AXS_SCHED_MAX_INFLIGHT` by +1 to +2.
- If `429` is high and p95 is high: reduce concurrency or model size; do not just raise queue.
- If p99 spikes with low `429`: reduce `AXS_SCHED_MAX_WAIT_MS`.
- If thermal throttling appears: reduce `AXS_SCHED_MAX_INFLIGHT`.

## Acceptance targets

- Error rate (`429` + `5xx`) < 1% under expected peak load.
- p95 latency meets product SLO.
- p99 remains stable across 30+ minute sustained test.

