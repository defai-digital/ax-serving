# Service tuning and evidence guide

This guide separates gateway admission tuning from runtime tuning. AX Serving
does not know the runtime's token scheduler well enough to replace it, and a
larger gateway queue can reduce rather than improve runtime goodput.

## 1. Establish the workload contract

Before changing a setting, record:

- AX Serving commit and release profile;
- gateway/agent/runtime versions and immutable image digests;
- model revision, artifact, tokenizer, template, and quantization identity;
- machine/GPU SKU, memory, power and thermal state;
- exact request mix, prompt bytes, sampling, input/output token accounting;
- concurrency, warmup, repetitions, and run duration;
- direct runtime, same-runtime-through-gateway, and mixed-fleet scenarios.

Do not use a summary with null or missing raw samples as evidence.

## 2. Gateway admission controls

| Variable | Default | Effect |
| --- | ---: | --- |
| `AXS_GLOBAL_QUEUE_MAX` | `128` | Active requests accepted by one gateway |
| `AXS_GLOBAL_QUEUE_DEPTH` | `256` | Waiting requests |
| `AXS_GLOBAL_QUEUE_WAIT_MS` | `10000` | Maximum queue wait |
| `AXS_GLOBAL_QUEUE_POLICY` | `queue` | `queue`, `reject`, or `shed_oldest` |
| `AXS_TENANT_MAX_CONCURRENT` | `0` | Per-tenant active cap; zero disables |
| `AXS_MAX_DISPATCH_ATTEMPTS` | `2` | Maximum attempts; safe conditions still apply |
| `AXS_FIRST_BYTE_TIMEOUT_MS` | `120000` | Headers-to-first-byte deadline |
| `AXS_STREAM_IDLE_TIMEOUT_MS` | `30000` | Maximum gap between stream bytes |
| `AXS_DISPATCHER_TIMEOUT_SECS` | `300` | Absolute configured request ceiling |

Starting values are not performance claims. Derive them from the total
advertised runtime capacity and target tail latency:

1. Set `GLOBAL_QUEUE_MAX` no higher than the measured aggregate concurrent
   capacity that maintains the target TTFT.
2. Begin with a small queue. Increase it only when short bursts are absorbed
   without violating the end-to-end deadline.
3. Use tenant caps to prevent one tenant from consuming all active slots.
4. Use priority for service classes, not to bypass hard capacity.
5. Reduce admission when runtime queue/KV pressure rises; do not blindly add
   gateway concurrency.

## 3. Endpoint policy

`inference_aware` is the production-candidate default. It hard-filters
incompatible endpoints, then scores reported capacity, queue depth, KV
pressure, batch headroom, TTFT, recent errors, freshness, and bounded jitter.
Unknown or stale telemetry receives a penalty.

Use compatibility policies only for controlled rollback or an experiment:

- `least_inflight` when runtime telemetry is unavailable;
- `weighted_round_robin` for static capacity weighting;
- `model_affinity` for coarse model locality;
- `token_cost` for TTFT/sequence weighting;
- `cache_affinity` with a configured tenant-scoped affinity secret.

Compare policies under the same request trace. A policy is better only if it
improves SLO goodput or tail latency without unsafe routing or starvation.

## 4. Tuning loop

1. Build release binaries.
2. Warm every deployment using a declared procedure.
3. Run direct runtime traffic and retain raw samples.
4. Run the same traffic through one gateway and agent.
5. Add mixed-fleet load, drain, failure, stale heartbeat, and overload events.
6. Change one setting at a time.
7. Repeat enough runs to report dispersion, not only the best run.
8. Inspect gateway and runtime metrics together.

Useful commands:

```bash
cargo run -p ax-serving-bench --release -- bench \
  -m ./models/replace-with-supported-model

python3 scripts/load_test_chat.py \
  --url http://127.0.0.1:18080/v1/chat/completions \
  --model replace-with-logical-model \
  --requests 200 \
  --concurrency 8 \
  --prompt-tokens 39 \
  --max-tokens 64
```

These commands do not by themselves prove equivalence or a production gate.
Attach the full configuration and raw artifact.

## 5. Required measurements

- admission and endpoint-selection latency;
- time to response headers and TTFT;
- inter-token and end-to-end latency;
- SLO goodput, not only tokens per second;
- admitted, rejected, attempted, retried, cancelled, completed, and failed;
- duplicate commitments (required value: zero);
- worker queue, active sequences, KV/cache pressure, and batch headroom;
- gateway CPU, RSS, connections, and allocator growth;
- lease exclusion/recovery and gateway restart time;
- distribution across pools and bounded route reasons.

Scrape admin-authenticated `GET /metrics` for normalized gateway signals and
runtime metrics from the runtime independently. High-cardinality request
evidence belongs in bounded logs/traces, never metric labels.

## 6. Production release gates

The PRD currently requires, among other gates:

- same-LAN added gateway setup latency: p50 <= 5 ms and p95 <= 15 ms;
- endpoint selection at 256 candidates: p99 <= 2 ms;
- less than 3% normalized goodput loss at the production envelope;
- stale lease exclusion within configured TTL;
- no retry after first committed byte;
- shared-state recovery within one heartbeat plus propagation time;
- at least 60 minutes of soak with bounded memory and no leaked queue,
  reservation, or lease state.

These are release criteria, not results for the current checkout. Publish a
pass only when retained artifacts contain all required samples and metadata.

## 7. Runtime comparison policy

For AX Engine versus llama.cpp, match the source model, tokenizer, prompt,
template, sampler, stop conditions, input/output accounting, hardware, warmup,
and run count. Use the same artifact only when both runtimes support it.
Disclose artifact format, digest, quantizer implementation, scheme, and
effective precision whenever they differ.

For AX Serving, report separately:

1. direct runtime baseline;
2. the same endpoint through gateway and agent;
3. a mixed AX Engine/CUDA fleet under normal and fault conditions.

Do not compare AX Serving's routing layer with vLLM's token engine as though
they perform the same work.
