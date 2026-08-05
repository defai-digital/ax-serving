# AX Serving v2.3.0 benchmark evidence

These measurements were collected on 2026-08-05 from commit
`72077118d5fdf0412c69b912ee614d7a7029a555`, before the evidence-only release
commit.

## Reference system

- Apple Mac Studio with M2 Ultra
- 24 CPU cores: 16 performance and 8 efficiency cores
- 192 GB unified memory
- macOS 26.6 (25G72)
- Rust release binaries built from the commit above
- AX Engine 4.10.0
- llama.cpp build 9020 (`a4701c98f`) with Metal and Accelerate

Local authentication was explicitly disabled only for the loopback benchmark
services by setting `AXS_ALLOW_NO_AUTH=true`.

## Native AX Engine measurements

The throughput runner used prompt lengths 39, 209, 509, and 1024, two warmup
iterations, five measured iterations, and 128 decode tokens. The release
baseline uses the minimum throughput across those prompt lengths. The profile
runner used a 128-token prompt, two warmups, five measured iterations, and 64
decode tokens; the baseline uses P50 TTFT.

| Artifact | Manifest SHA-256 | Decode tok/s | Prefill tok/s | P50 TTFT |
| --- | --- | ---: | ---: | ---: |
| `AX-MiniCPM5-1B-MLX-AXQ-4bit` | `7e51ebe6c2ce5c2276105322a5026eb653542f7b3b7da30aab221bd8b43c0b01` | 265.782 | 3,239.135 | 9.411 ms |
| `AX-gemma-4-12b-MLX-AXQ-4bit` | `80a72cdf0ad436e1038fa2d3b736246ceae7387757040b6cb69f708f9d75d71c` | 48.956 | 201.258 | 35.566 ms |
| `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit` | `7d4f882cbbb8d785c646a6baacfd99b5b6fff764864b1b742ee483e6d8237568` | 32.394 | 263.469 | 33.501 ms |

Representative commands:

```console
target/release/ax-serving-bench bench \
  --model /path/to/AX-model \
  --prompt-lengths 39,209,509,1024 \
  --decode-tokens 128 \
  --warmup 2 \
  --iters 5 \
  --json native-model-throughput.json

target/release/ax-serving-bench profile \
  --model /path/to/AX-model \
  --prompt-length 128 \
  --warmup 2 \
  --json native-model-profile.json
```

Candidate artifacts that did not produce a complete throughput and TTFT pair
under the pinned AX Engine runtime were not copied into the populated release
baseline.

## llama.cpp measurements

The llama.cpp artifact was Qwen3.5-9B IQ4_XS (4.25 bpw), 5,157,685,248 bytes,
with SHA-256
`7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429`.
Five `llama-bench` samples were recorded at each prompt length and for a
128-token generation. Streaming TTFT was measured through `llama-server` with
two warmups and five samples.

| Decode tok/s | Minimum prefill tok/s | P50 TTFT |
| ---: | ---: | ---: |
| 67.784 | 400.754 | 75.221 ms |

## Mixed workload

The mixed benchmark used the MiniCPM5 worker over loopback with 60 requests,
concurrency 4, 128 maximum output tokens, and equal short, medium, and long
prompt classes.

```console
target/release/ax-serving-bench mixed \
  --url http://127.0.0.1:18101 \
  --concurrency 4 \
  --requests 60 \
  --model minicpm5-1b \
  --max-tokens 128 \
  --target-p99-ms 10000 \
  --json mixed-minicpm5-1b-c4.json
```

All 60 requests succeeded. Overall P99 was 6,247 ms. A separate concurrency
sweep showed the default 10-second boundary between concurrency 4 (pass) and
concurrency 8 (11,139 ms, fail), so concurrency 4 is the recorded release
baseline.

## Multi-worker scope

The v2.3.0 release evidence covers the direct `least_inflight` and
`weighted_round_robin` policies with four independently registered embedded
workers, 16 concurrent clients, 128 decode tokens, and 30-minute runs. A
100-request direct-worker control was measured at concurrency 1 for sequential
latency and concurrency 4 for the scaling comparison. Concurrency 4 matches
the per-worker load in the four-worker runs.

| Policy | Success | Errors | Requests/s | P50 | P95 | P99 | Throughput/control | P95/control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `least_inflight` | 4,757 | 0 | 2.632 | 6,259.898 ms | 8,372.742 ms | 9,417.474 ms | 3.994x | 1.330x |
| `weighted_round_robin` | 4,358 | 0 | 2.410 | 6,323.761 ms | 12,530.034 ms | 14,917.430 ms | 3.656x | 1.991x |

The scaling gates are at least 3.2x control throughput (80% of ideal
four-worker scaling), P95 no more than 1.5x the concurrency-matched control,
and error rate below 1%. The runner's `overhead_p50_lt_5ms` field is not used
as a release gate here because it measures end-to-end LLM request latency, not
isolated gateway dispatch overhead.

Both policies pass the throughput and error-rate gates. `least_inflight` passes
the P95 ratio gate; `weighted_round_robin` does not. That failed gate is
retained in `baseline-multi-worker.json` as measured release evidence.

The benchmark CLI does not currently drive a NATS dispatcher or alternate
models within one run. NATS and two-model affinity placeholders from the old
unmeasured template are therefore outside this release baseline rather than
being represented by invented results.
