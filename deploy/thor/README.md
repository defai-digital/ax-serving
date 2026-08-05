# Jetson Thor TensorRT Edge-LLM compatibility deployment

This path integrates NVIDIA TensorRT Edge-LLM's **experimental**
OpenAI-compatible server with `ax-runtime-agent`. It is distinct from
TensorRT-LLM and from an NVIDIA Dynamo production domain.

The pinned qualification tuple is:

| Component | Pin |
|---|---|
| Jetson Linux / JetPack | `39.2.0-20260601141651` / `7.2-b187` |
| CUDA | 13.2 |
| TensorRT | `10.16.2.10-1+cuda13.2` |
| TensorRT Edge-LLM | v0.9.1 / `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| uv | 0.12.2 |
| Model | `Qwen/Qwen3-0.6B` |
| Model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| AX runtime identity | `tensorrt_edge_llm` |
| AX hardware class | `thor-jetpack-7.2` |

The model is downloaded at the exact revision into an operator-owned install
root. The server receives a stable local alias, so `/v1/models` reports
`qwen3-edge` instead of an unpinned Hugging Face branch.

This compatibility profile deliberately sets Edge-LLM batch size and AX
inflight capacity to `1`. On v0.9.1, concurrent requests to the experimental
server can fail inside TensorRT Myelin while switching execution profiles.
Treat serialization as a required compatibility guard until NVIDIA qualifies
a later release for concurrent serving on Thor.

## Install and start

The host manager allow-lists the two SSH aliases and encodes the currently
observed alias/hostname inversion:

```bash
scripts/qualification/runtime/manage_thor_edge_llm.sh preflight --target all
scripts/qualification/runtime/manage_thor_edge_llm.sh system-deps --target all
scripts/qualification/runtime/manage_thor_edge_llm.sh install --target all
scripts/qualification/runtime/manage_thor_edge_llm.sh start --target all
```

`df-thor-01` is guarded as hostname `df-thor-02`; `df-thor-02` is guarded as
hostname `df-thor-01`. The installer uses passwordless `sudo` only for apt
packages. Source, `uv`, `.venv`, model, engine, logs, and state remain under
`~/.local/share/ax-serving/tensorrt-edge-llm/v0.9.1`.

`install` is fail-closed: it verifies R39.2, CUDA 13.2, TensorRT 10.16, the
Edge-LLM commit, immutable model revision, and expected build products. It
does not reset or replace an unexpected source checkout.

The native server binds to loopback by default because the experimental API
does not implement AX authentication. The host-networked AX agent is the
authenticated LAN-facing boundary. Probe the native endpoint on a Thor:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

scripts/qualification/runtime/manage_thor_edge_llm.sh qualify-direct \
  --target df-thor-01 \
  --output-dir target/runtime-qualification
```

The manager sends the checked-in smoke test over SSH and runs it with the
pinned remote `uv`; it then copies the JSON evidence back to `--output-dir`.
Do not pass `--host 0.0.0.0` unless a separate firewall and trusted-network
policy protect port 8000.

The first start exports ONNX and builds a TensorRT engine, so its readiness
deadline is longer than subsequent starts.

## Register with AX Serving

Copy the environment example to an ignored host-local file, replace both
credentials, and set the values for that alias:

```bash
cp deploy/thor/tensorrt-edge-llm.env.example .env.edge-thor
docker compose \
  --env-file .env.edge-thor \
  -f deploy/thor/tensorrt-edge-llm-agent.compose.yaml \
  up -d --build
```

The standalone Linux-host-network Compose file builds only the portable
runtime agent. The Edge-LLM server remains a native JetPack process. Use the
gateway's private port (normally 19090) for `AXS_CONTROL_PLANE_URL`; the
gateway must be able to reach the advertised Thor port 18081.

Qualify the gateway path with the canonical runtime hint:

```bash
uv run scripts/qualification/runtime/smoke_openai_runtime.py \
  --base-url http://gateway.example:18080 \
  --model qwen3-edge \
  --runtime tensorrt_edge_llm \
  --stability-seconds 45 \
  --requests 8 \
  --concurrency 1 \
  --allow-insecure-http \
  --output target/runtime-qualification/df-thor-01-edge-gateway.json
```

Keep each Thor in its own node/domain identity. Never reuse a PC
TensorRT-LLM engine, performance calibration, compatibility manifest, or
certification state for TensorRT Edge-LLM.
