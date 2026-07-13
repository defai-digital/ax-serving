# AX Serving quick start

This guide starts the portable runtime-neutral gateway. The embedded
Apple-Silicon server is a separate compatibility path described at the end.

## 1. Build

Requirements:

- Rust 1.88 or newer;
- macOS arm64 or Linux x86_64/arm64 for the portable gateway and agent;
- an independently running OpenAI-compatible AX Engine, vLLM, or SGLang
  endpoint.

```bash
cargo build --release \
  -p ax-serving-cli --bin ax-serving-api \
  -p ax-thor-agent --bin ax-runtime-agent
```

The gateway build must remain runtime-SDK-free:

```bash
cargo tree -p ax-serving-cli --no-default-features --features gateway \
  | rg 'ax-serving-engine|mlx-rs|llama-cpp'
```

No output is expected.

## 2. Start a local development gateway

The checked-in default binds both APIs to loopback and uses in-memory fleet
state:

```bash
AXS_ALLOW_NO_AUTH=true target/release/ax-serving-api
```

Verify process health. Readiness is intentionally `503` until a runtime is
registered and ready:

```bash
curl -i http://127.0.0.1:18080/livez
curl -i http://127.0.0.1:18080/readyz
curl -sS http://127.0.0.1:18080/health | jq .
```

## 3. Attach a runtime

Start the runtime first. It must expose:

- `GET /health` or an adapter-supported readiness endpoint;
- `GET /v1/models`;
- the OpenAI-compatible inference operations it advertises.

For a local vLLM endpoint on port 8000:

```bash
AXS_CONTROL_PLANE_URL=http://127.0.0.1:19090 \
AXS_NODE_RUNTIME=vllm \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=127.0.0.1:18081 \
AXS_NODE_ADVERTISED_ADDR=127.0.0.1:18081 \
AXS_NODE_HARDWARE_CLASS=pc-cuda \
AXS_NODE_WORKER_POOL=cuda \
AXS_NODE_MAX_INFLIGHT=16 \
target/release/ax-runtime-agent
```

For AX Engine on Apple Silicon, point the same agent at the AX Engine server:

```bash
AXS_CONTROL_PLANE_URL=http://127.0.0.1:19090 \
AXS_NODE_RUNTIME=ax_engine \
AXS_NODE_RUNTIME_URL=http://127.0.0.1:8000 \
AXS_NODE_LISTEN_ADDR=127.0.0.1:18081 \
AXS_NODE_ADVERTISED_ADDR=127.0.0.1:18081 \
AXS_NODE_HARDWARE_CLASS=mac \
AXS_NODE_WORKER_POOL=mac-mlx \
AXS_NODE_MAX_INFLIGHT=8 \
target/release/ax-runtime-agent
```

The agent does not link the runtime SDK. It discovers readiness, inventory,
capabilities, and common metrics over HTTP, then proxies request and stream
bytes. Runtime credentials use `AXS_RUNTIME_API_KEY`; client credentials are
never reused for that hop.

Check registration:

```bash
curl -sS http://127.0.0.1:19090/internal/workers | jq .
curl -i http://127.0.0.1:18080/readyz
curl -sS http://127.0.0.1:18080/v1/models | jq .
```

## 4. Send requests

In the default `legacy_compat` deployment mode, use a model ID reported by the
runtime:

```bash
export MODEL_ID='replace-with-runtime-model-id'

curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain hybrid inference briefly.\"}],
    \"max_tokens\": 96,
    \"stream\": false
  }"
```

Streaming:

```bash
curl -N http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Count to five.\"}],
    \"max_tokens\": 32,
    \"stream\": true
  }"
```

Embeddings use the same gateway and require a runtime that advertises the
embedding operation:

```bash
curl -sS http://127.0.0.1:18080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"input\":\"hello\"}"
```

## 5. Configure safe hybrid routing

`legacy_compat` is for migration and one-pool setups. Certified cross-runtime
failover requires `deployment_mode: explicit` with:

- homogeneous pools;
- logical-model-to-runtime deployment mappings;
- exact runtime/model/tokenizer/template/quantization identity;
- an operator-approved equivalence class and retained certification artifact.

Copy the example, replace every placeholder, then validate startup:

```bash
cp config/serving.hybrid.example.yaml /secure/path/serving.yaml
AXS_CONFIG=/secure/path/serving.yaml \
AXS_API_KEY='public-client-key' \
AXS_ADMIN_API_KEY='admin-key' \
AXS_INTERNAL_API_TOKEN='worker-control-key' \
AXS_DISPATCH_TOKEN='gateway-agent-key' \
target/release/ax-serving-api
```

The example binds remotely, so `trusted_mesh` requires TLS or mTLS from the
deployment environment. It is not an in-process TLS server switch.

Do not certify different tokenizer or template digests merely because two
deployments share a public model name. Missing identity is eligible only for
explicitly pinned single-pool use, not cross-runtime failover.

## 6. Authentication and trust boundaries

Use different random credentials for each boundary:

```text
client -> gateway             AXS_API_KEY
operator -> admin API         AXS_ADMIN_API_KEY
agent -> worker control API   AXS_WORKER_TOKEN / AXS_INTERNAL_API_TOKEN
gateway -> agent dispatch     AXS_DISPATCH_TOKEN
agent -> runtime              AXS_RUNTIME_API_KEY
gateway -> Redis/Valkey       AXS_REDIS_URL
```

Public inference calls use:

```bash
curl -sS http://gateway.example/v1/models \
  -H 'Authorization: Bearer public-client-key'
```

Admin calls require the admin key, not the public key:

```bash
curl -sS http://gateway.example/admin/v1/deployments \
  -H 'Authorization: Bearer admin-key'
```

The gateway strips public authorization, cookies, proxy credentials,
hop-by-hop headers, and AX internal headers before dispatch.

## 7. Active-active gateways

Use Redis or Valkey for shared leases, desired deployment state, jobs, probe
ownership, and worker-capacity reservations:

```bash
AXS_CONFIG=/secure/path/serving.yaml \
AXS_FLEET_STORE=redis \
AXS_REDIS_URL='rediss://user:password@redis.example:6379/0' \
AXS_GATEWAY_ID='gateway-a' \
AXS_API_KEY='public-client-key' \
AXS_ADMIN_API_KEY='admin-key' \
AXS_INTERNAL_API_TOKEN='worker-control-key' \
AXS_DISPATCH_TOKEN='gateway-agent-key' \
target/release/ax-serving-api
```

Start another replica with a unique `AXS_GATEWAY_ID` and the same Redis key
prefix. Use a durable Redis topology appropriate to the availability target;
an ephemeral cache is not fleet-state durability.

The Kubernetes baseline in [deploy/kubernetes](deploy/kubernetes/README.md)
contains two gateway replicas, probes, security contexts, a disruption budget,
and a runtime-agent sidecar example.

## 8. Admission, priorities, and affinity

Useful controls:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `AXS_GLOBAL_QUEUE_MAX` | `128` | Active gateway requests |
| `AXS_GLOBAL_QUEUE_DEPTH` | `256` | Waiting requests |
| `AXS_GLOBAL_QUEUE_WAIT_MS` | `10000` | Admission deadline |
| `AXS_TENANT_MAX_CONCURRENT` | `0` | Per-tenant active quota; zero disables |
| `AXS_DISPATCH_POLICY` | `inference_aware` | Endpoint scoring after hard filters |
| `AXS_MAX_DISPATCH_ATTEMPTS` | `2` | Hard maximum; safe conditions only |

Clients may send `x-ax-priority: low|normal|high`. Priority aging and
cross-client fairness prevent indefinite starvation, but runtime token
scheduling remains inside the runtime.

An opaque `x-ax-cache-affinity` hint is accepted only when
`AXS_CACHE_AFFINITY_SECRET` contains at least 32 random bytes. The gateway
derives a tenant-scoped keyed digest and discards the raw value. It never hashes
or logs prompt text for affinity.

## 9. Lifecycle and drain

In explicit mode, mutations return asynchronous job records:

```bash
curl -sS -X PATCH \
  http://127.0.0.1:18080/admin/v1/deployments/DEPLOYMENT_ID \
  -H 'Authorization: Bearer admin-key' \
  -H 'Content-Type: application/json' \
  -d '{"action":"drain"}'

curl -sS http://127.0.0.1:18080/admin/v1/jobs/JOB_ID \
  -H 'Authorization: Bearer admin-key'
```

Deployment jobs change and observe control-plane desired state. External
runtime/orchestrator adapters still own process creation, model download, and
GPU allocation. A rolling replacement is enabled and proven ready before the
source is disabled; timeout rolls desired state back.

## 10. Observability

The portable `ax-servingctl` binary can collect the public and operator views
without installing the embedded macOS runtime:

```bash
ax-servingctl status --url http://127.0.0.1:18080 \
  --api-key public-key --admin-key admin-key --diagnostics --json
```

```bash
curl -sS http://127.0.0.1:18080/v1/metrics \
  -H 'Authorization: Bearer admin-key' | jq .
curl -sS http://127.0.0.1:18080/metrics \
  -H 'Authorization: Bearer admin-key'
curl -sS http://127.0.0.1:18080/v1/admin/diagnostics \
  -H 'Authorization: Bearer admin-key' | jq .
```

Prometheus metrics use bounded `axs_gateway_*` names and avoid worker IDs,
request IDs, prompts, model paths, and credentials as labels.

W3C trace context is propagated through gateway, agent, and runtime whether
or not spans are exported. To export OTLP/HTTP JSON traces, set a collector
endpoint and optional sensitive headers on both gateway and agents:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.example \
OTEL_EXPORTER_OTLP_PROTOCOL=http/json \
OTEL_EXPORTER_OTLP_HEADERS='authorization=Bearer%20replace-me' \
target/release/ax-serving-api
```

Prompt text, generated output, tool arguments, images, model paths, and
credentials are not captured by the built-in spans.

## 11. Embedded compatibility on macOS

The local inference CLI and server require the non-default feature and link
the embedded engine stack:

```bash
cargo build --release -p ax-serving-cli \
  --no-default-features --features embedded-compat --bin ax-serving

AXS_ALLOW_NO_AUTH=true target/release/ax-serving serve \
  -m ./models/replace-with-supported-artifact \
  --model-id local-model \
  --port 18080
```

Use this path for migration, local diagnostics, and compatibility. New hybrid
deployment work belongs in runtime agents and the versioned protocol rather
than new gateway engine integrations.

## 12. Validation

```bash
cargo fmt --all -- --check
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 AXS_ALLOW_NO_AUTH=true \
  cargo test --workspace --all-features
```

Live runtime certification and production performance evidence are separate
release gates. A passing mock suite does not certify a runtime image, model
identity, latency SLO, goodput target, or 60-minute soak.
