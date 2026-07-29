# Mac AX Engine cluster coordinator

AX Serving can represent one model-parallel Mac cluster as one execution domain. The
`ax-mac-cluster-adapter` validates an immutable rank manifest, coordinates gang readiness, exposes
rank bootstrap/control endpoints, registers one aggregate protocol-v1.2 endpoint, and proxies
OpenAI requests to rank 0.

This is currently a source/mock-tested control-plane implementation. AX Engine does not yet
implement the required cross-Mac partial layer loader, pipeline executor, activation transport, or
generation-wide cancellation. Consequently, this guide can validate coordination and an external
rank-0 mock, but it cannot run a 405B model across Macs today.

## Ownership boundary

| Concern | Owner |
| --- | --- |
| Public API, logical model, domain choice, reservation, decision evidence | AX Serving gateway |
| Manifest validation, gang state, aggregate observation, rank bootstrap | Mac cluster adapter |
| Layer loading, embeddings/head, KV, activation transfer, sampling, cancellation | AX Engine |
| Rank discovery and transport trust | Operator plus the future AX Engine cluster runtime |

Internal ranks never register as AX workers and never become gateway routing candidates.

## Build and configure

```bash
cargo build \
  -p ax-serving-cli --bin ax-serving-api \
  -p ax-mac-cluster-adapter --bin ax-mac-cluster-adapter
```

Start the loopback development gateway:

```bash
AXS_CONFIG=config/serving.mac-cluster.example.yaml \
AXS_ALLOW_NO_AUTH=true \
target/debug/ax-serving-api
```

The example manifest and adapter environment are:

- [`config/mac-cluster-manifest.example.json`](../../../config/mac-cluster-manifest.example.json)
- [`config/mac-cluster-adapter.example.env`](../../../config/mac-cluster-adapter.example.env)
- [`config/serving.mac-cluster.example.yaml`](../../../config/serving.mac-cluster.example.yaml)

Replace every placeholder digest, node identity, memory number, runtime version, and credential.
Then start the adapter with the documented environment values. `AXS_MAC_CLUSTER_RANK0_URL` must be
an OpenAI-compatible frontend for the complete cluster; a single-node AX Engine server does not
become distributed merely by using this URL.

The adapter requires three distinct credentials in authenticated profiles:

- `AXS_WORKER_TOKEN`: adapter to gateway registration/control;
- `AXS_DISPATCH_TOKEN`: gateway to adapter request dispatch;
- `AXS_MAC_CLUSTER_CONTROL_TOKEN`: rank to coordinator bootstrap/heartbeat.

Unauthenticated dispatch is accepted only when the listener is loopback, the TLS profile is
`loopback_dev`, and `AXS_ALLOW_NO_AUTH=true` is explicit. Remote plaintext origins require the
operator-declared `trusted_mesh` profile; production qualification must use an actually trusted
and authenticated transport.

## Rank control contract

Each rank retrieves exactly its own layer, memory, topology, and artifact plan:

```text
GET /internal/cluster/ranks/{rank}/plan
x-ax-cluster-control-token: <rank-control-token>
```

It reports lifecycle and bounded diagnostics:

```text
POST /internal/cluster/ranks/{rank}/heartbeat
x-ax-cluster-control-token: <rank-control-token>
content-type: application/json
```

The JSON observation must bind the same cluster ID, generation, manifest digest, and rank. A ready
observation must also meet the manifest's minimum peer bandwidth and maximum latency. The operator
can inspect:

```text
GET /internal/cluster/status
x-ax-cluster-control-token: <rank-control-token>
```

The domain is ready only when every required rank is fresh and ready. A failed rank fails the
entire generation and cannot recover inside that generation. Recovery requires a new immutable
manifest with a higher generation and an adapter/rank restart.

## What remains before real multi-Mac inference

The AX Engine repository must add and certify:

1. manifest-bound partial model loading for one decoder-only family and quantization;
2. direct stage-to-stage activation transport and per-rank KV ownership;
3. ordered streaming and idempotent generation-wide cancellation;
4. rank-loss, partition, memory-pressure, drain, and restart behavior;
5. numerical correctness and retained topology-specific performance/soak evidence.

Until those gates pass, use the adapter only for control-plane development and do not advertise the
Mac cluster path as runnable or production supported.
