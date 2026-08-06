# Live evidence: heterogeneous compatibility multi-worker fleet

**Date:** 2026-08-05 (America/Toronto) / 2026-08-06 (UTC)

**Claim level:** live laboratory evidence for **compatibility runtime endpoints**

**Not a claim:** production certification, Dynamo domain qualification, multi-Mac cluster, or live mixed-domain failover

**Repository retention:** narrative summary and digests only; raw JSON/JSONL remain in the local
ignored build tree and are not independently downloadable from this repository

## What was demonstrated

AX Serving routed one logical model (`qwen3-edge`) across four heterogeneous workers registered as
`compatibility_runtime_endpoint` agents. Every successful completion was a real inference response
with an `x-ax-routing-trace`.

| Worker | Runtime | Model artifact |
| --- | --- | --- |
| `df-rtx5090` | vLLM 0.25.1 | Qwen3-0.6B BF16 |
| `df-thor-01` | TensorRT Edge-LLM 0.9.1 | Qwen3-0.6B FP16 |
| `df-thor-02` | TensorRT Edge-LLM 0.9.1 | Qwen3-0.6B FP16 |
| `macstudio-m2u` | llama.cpp `1-091a46c` | Qwen3-0.6B GGUF Q8_0 |

This is multi-worker federation over direct/compatibility runtimes. It is **not** a Dynamo domain
result, **not** a pinned AX Engine result, and **not** a physical multi-Mac cluster result.

## Serial fairness (concurrency 1)

- 40 / 40 successful completions
- equal-weight round-robin: **10 requests per worker**
- 40 / 40 routing traces
- latency: p50 41.873 ms, p95 73.748 ms, max 85.348 ms

## Concurrent soak (300 s, concurrency 4)

Each worker used `max_inflight=1`. The load generator applied 10–250 ms bounded exponential backoff
on capacity HTTP 503 responses.

| Worker | Successful requests | Share | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 / vLLM | 12,915 | 43.42% | 15.214 ms | 16.421 ms |
| Thor-01 / Edge-LLM | 5,032 | 16.92% | 29.066 ms | 34.753 ms |
| Thor-02 / Edge-LLM | 5,640 | 18.96% | 28.945 ms | 34.400 ms |
| Mac Studio / llama.cpp | 6,154 | 20.69% | 31.414 ms | 35.572 ms |
| **Total** | **29,741** | **100%** | **28.431 ms** | **33.882 ms** |

Additional soak facts:

- 29,741 / 29,741 logical requests succeeded (0 final failure)
- 29,741 / 29,741 routing traces
- ~99.08 successful requests/s over ~300.2 s
- aggregate p99 299.772 ms; max 7,518.474 ms (capacity-retry tail, not steady model latency)
- fleet snapshot observed all four workers simultaneously at `inflight=1`
- post-run: all four `healthy`, `runtime_ready=true`, `inflight=0`

Non-uniform shares under concurrency are expected and work-conserving: the serial test proved fair
round-robin; faster workers free capacity sooner and accept more subsequent work.

## Capacity / backpressure observation

With all workers busy, the gateway returned HTTP 503 immediately rather than queuing at the gateway:

- formal soak: 33,453 HTTP attempts, of which 3,712 were capacity retries (**11.10%**)
- after bounded client backoff, every logical request completed successfully
- an exploratory run without backoff produced a 503 storm (tens of thousands of retries in ~37 s)

Implication for operators and SDKs: treat capacity 503 as normal backpressure. Pair busy fleets with
bounded client retry, gateway admission/queue policy, or both. See
[service tuning](../perf/service-tuning.md).

## Thor smoke (compatibility path)

Separate TensorRT Edge-LLM smokes on Thor and through the gateway reported `status: pass` for short
non-stream and stream completions against `qwen3-edge`. These support the **experimental Thor
compatibility** path only. They do **not** certify `nvidia_dynamo_thor` or any Dynamo-on-Thor
deployment.

## Explicit non-claims

This evidence does **not** establish:

| Still open | Why |
| --- | --- |
| Live mixed-domain **failover** / drain / worker death | Soak kept workers healthy; no intentional domain failure injection |
| Multi-Mac **cluster** (`ax-mac-cluster-adapter` physical cert) | One Mac Studio endpoint only |
| Pinned live **AX Engine** Mac domain | Mac path used llama.cpp, not AX Engine |
| **Dynamo** PC domain production path | RTX path used direct vLLM via runtime agent |
| **Dynamo Thor** production path | Thor used TensorRT Edge-LLM compatibility endpoint |
| Output equivalence / quality floor across backends | Completions validated as JSON success, not semantic equivalence |
| Mac unattended production ops | Background Mac agent LAN reachability required an SSH control-plane tunnel workaround for heartbeat; data plane still used LAN |

## Operator notes retained from the run

- Mac control-plane heartbeat used an SSH loopback tunnel because macOS Local Network policy blocked
  some background processes from LAN; inference data path remained LAN
  (`gateway → Mac agent → llama.cpp`).
- Formal unattended Mac deployment still needs Local Network permission, a Docker/VM network domain,
  or a productized control tunnel with monitoring and reconnect.

## Artifact retention

Raw JSON/JSONL from the laboratory run were produced under
`target/runtime-qualification/` (build/output tree; not published as repo release artifacts).

Locally verified raw-artifact digests from that run:

| Artifact | SHA-256 |
| --- | --- |
| `serial-40-summary.json` | `6c59be0d325e05b33ee25326c13ecea8b0b254811d24077a85e41266569463fb` |
| `soak-300s-c4-retry-summary.json` | `452820a106640bfe688721a1f7998e2285989f48944b734cd718e0acfa69b473` |
| `serial-40-summary.jsonl` | `4ff9573218cfff066a6cd10edcc3060b89bfc8e720c7458e595922490248d3fd` |
| `soak-300s-c4-retry-summary.jsonl` | `db0231d0c5054530ac501be51537f054df8bebc256174f78bb69c1f0d850e485` |
| `tensorrt-edge-llm-gateway.json` | `9c8e27c76ced106a671d3c405ff9a801ea034155e223b351cfc1312464e022cc` |
| `df-thor-01-edge-direct.json` | `af635b3476060d2d7e310067db1d9f0e5315867d4f704fa65abcfa180fcbab0c` |
| `df-thor-02-edge-direct.json` | `8448effbf479bc1b8eb82aaad07bfd5cd4c18c34ee7aec5b5014c27de10449ff` |

A Chinese operator write-up of the same run may exist locally as
`target/runtime-qualification/heterogeneous/RESULTS.zh-TW.md`; this document is the English
retained public summary for status language in README and docs.

## How to cite this

Safe public wording:

> Live laboratory evidence shows AX Serving can route and soak a single logical model across
> heterogeneous compatibility workers (vLLM, TensorRT Edge-LLM on Thor, llama.cpp on Mac) with
> routing traces and zero final logical failures under the recorded load. The repository publishes
> the bounded summary and raw-artifact digests, not the raw samples.

Unsafe / overclaim wording to avoid:

> AX Serving is production-certified for mixed-domain failover, multi-Mac clusters, or Dynamo Thor.
