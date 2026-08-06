# Mac AX Engine cluster coordinator

AX Serving can represent one model-parallel Mac cluster as one execution domain. The
`ax-mac-cluster-adapter` validates an immutable rank manifest, coordinates gang readiness, exposes
rank bootstrap/control endpoints, registers one aggregate protocol-v1.2 endpoint, and proxies
OpenAI requests to rank 0.

AX Engine now has an initial dense-Llama-3 static pipeline runtime: generation-fenced topology and
activation contracts, file-selective stage loading, stage-local global-indexed KV, bounded binary
activation frames, request replay/cancellation fencing, and an authenticated rank HTTP service.
The two-rank numeric test crosses a serialized activation boundary and matches monolithic forward.

The AX Engine branch also includes a rank-chain client, tokenizer-aware generation CLI, a
rank-0 OpenAI-compatible greedy completions/chat gateway with SSE and Llama 3 chat templating,
bootstrap artifact verification, generation-bound endpoint preflight, ready heartbeats, and a
final draining/failed observation. Automatic artifact download, non-greedy distributed sampling,
live two-Mac 405B qualification, and production fault/soak evidence are not complete. The path is
therefore an internal runtime bring-up surface, not a production-supported 405B API.

## Ownership boundary

| Concern | Owner |
| --- | --- |
| Public API, logical model, domain choice, reservation, pre-dispatch decision evidence | AX Serving gateway |
| Manifest validation, gang state, aggregate observation, rank bootstrap | Mac cluster adapter |
| Layer loading, embeddings/head, KV, activation transfer, greedy final-rank sampling, cancellation | AX Engine |
| Rank discovery and transport trust | Operator plus AX Engine pipeline rank runtime |

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

The adapter requires four distinct credentials in authenticated profiles:

- `AXS_WORKER_TOKEN`: adapter to gateway registration/control;
- `AXS_DISPATCH_TOKEN`: gateway to adapter request dispatch;
- `AXS_MAC_CLUSTER_CONTROL_TOKEN`: rank to coordinator bootstrap/heartbeat;
- `AXS_MAC_CLUSTER_RANK0_TOKEN`: adapter to the rank-0 OpenAI frontend.

Unauthenticated dispatch is accepted only when the listener is loopback, the TLS profile is
`loopback_dev`, and `AXS_ALLOW_NO_AUTH=true` is explicit. Remote plaintext origins require the
operator-declared `trusted_mesh` profile; production qualification must use an actually trusted
and authenticated transport. The example `http://` URLs assume a private encrypted overlay such as
WireGuard; bearer tokens and activation payloads must never traverse an untrusted plaintext LAN.

## Rank control contract

Each rank retrieves exactly its own layer, memory, topology, and artifact plan:

```text
GET /internal/cluster/ranks/{rank}/plan
x-ax-cluster-control-token: <rank-control-token>
```

The runtime-neutral topology projection can be saved directly as the AX Engine rank input:

```text
GET /internal/cluster/engine-topology
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

## AX Engine rank data plane

On each Mac, build `ax-engine-pipeline-rank` from the matching AX Engine revision. Start it with the
saved topology, the rank-specific artifact directory, its numeric rank, and a distinct worker
data-plane token:

```bash
ax-engine-pipeline-rank \
  --topology ./engine-topology.json \
  --bootstrap-plan ./rank-0-plan.json \
  --model-dir ./model \
  --rank 0 \
  --worker-token '<separate-worker-data-plane-token>' \
  --coordinator-url http://adapter:9200 \
  --control-token '<rank-control-token>' \
  --peer-bandwidth-bytes-per-second 1000000000 \
  --peer-latency-micros 1000 \
  --listen 0.0.0.0:9300
```

Save `rank-0-plan.json` from `GET /internal/cluster/ranks/0/plan` (and use the corresponding
rank-specific plan on every other Mac). When coordinator integration is enabled, AX Engine requires
this plan, verifies cluster/generation/manifest/rank identity, rejects unsafe or duplicate artifact
paths, streams SHA-256 verification for every declared file, and proves that
`model-manifest.json` plus every stage-selected safetensor file is covered before loading weights.

The initial authenticated endpoints are:

```text
POST /internal/pipeline/tokens
POST /internal/pipeline/activation
POST /internal/pipeline/requests/{request_id}/close
x-ax-cluster-worker-token: <worker-data-plane-token>
```

Activation requests use `application/x-ax-pipeline-frame`. The binary frame binds cluster,
generation, manifest, artifact, request sequence, source/destination rank, layer boundary, token
offset, tensor shape/dtype, payload length, and SHA-256. A receiver validates every field and its
configured byte ceiling before reconstructing the MLX array. Rank 0 accepts token IDs; each
non-final rank accepts only the immediately preceding activation; the final rank currently returns
a greedy token.

After all rank services are healthy, start the rank-0 frontend:

```bash
ax-engine-pipeline-gateway \
  --topology ./engine-topology.json \
  --model-dir ./model \
  --endpoints http://mac-a:9300,http://mac-b:9300 \
  --worker-token '<worker-data-plane-token>' \
  --api-key '<rank0-runtime-token>' \
  --model-id llama-405b-int4-pp2 \
  --listen 0.0.0.0:9400
```

`--model-id` must equal `model.runtime_model_id` in the cluster manifest. AX Serving rewrites the
logical public model to this runtime model ID before dispatch. The gateway preflights every endpoint
against the topology's rank order, cluster, generation, manifest digest, and model artifact digest
before it starts accepting traffic.

Set `AXS_MAC_CLUSTER_RANK0_URL` to this gateway and set `AXS_MAC_CLUSTER_RANK0_TOKEN` to the same
value as the gateway API key. Keep it distinct from `AXS_DISPATCH_TOKEN`. The gateway currently
implements authenticated health, model inventory,
greedy `/v1/completions`, and Llama 3 `/v1/chat/completions` (streaming and non-streaming). It
propagates each prefill/decode step through every rank in order, enforces an overall generation
deadline, and closes request KV on every rank at termination, timeout, error, or when a disconnected
stream stops accepting deltas. Rank services can post generation-bound ready heartbeats directly
to the adapter once weights are loaded.

## In-repo phase status

| Phase | In-repo surface | Physical / Engine pin |
| --- | --- | --- |
| 0–1 | Protocol 1.2 cluster domain, registration, heartbeat, drain, generation fence, domain reservation | Live Redis HA exercised when `AXS_TEST_REDIS_URL` is set |
| 2 | Static PP manifest, gang lifecycle, rank-0 proxy, typed pre-admission / non-retry ambiguous failure, generation restart policy | Real-weight two-Mac correctness still external |
| 3 | Shard-aware artifact prepare/verify, advisory placement, async reconcile, micro-batch contracts, multi-replica aggregation, operator status, metrics, evidence hooks | Load/fault/soak evidence retained via hooks; 60-minute hardware soak external |
| 4 | TP/hybrid plan validation, model-parallel topology projection, chunking profile contracts | Model-native Engine TP/hybrid kernels external |
| 5 | Adaptive federation config + live dispatch selection with shadow/canary/active/rollback and decision retention | Production policy rollout external |

## Remaining qualification gates

1. add richer OpenAI sampling/tool/structured-output contracts on the Engine rank-0 gateway;
2. automate bootstrap artifact download and live generation replacement;
3. add temperature/top-k/top-p sampling with deterministic generation ownership;
4. run retained real-weight, two-Mac memory/throughput/latency, link-loss, memory-pressure, restart,
   and long-soak evidence on pinned hardware;
5. expand the certified matrix beyond dense Llama 3 static pipeline parallelism once Engine TP/hybrid
   kernels land.

Until those physical and Engine gates pass, do not advertise the Mac cluster path as production
supported.

## Multi-process rehearsal

A compose template for gateway + Redis + adapter (+ optional Engine rank profile) lives at
[`deploy/compose/mac-cluster.compose.example.yaml`](../../../deploy/compose/mac-cluster.compose.example.yaml).
Operator install, upgrade, rank-loss, credential rotation, and rollback steps are documented in
[`docs/runbooks/mac-cluster-operations.md`](../../runbooks/mac-cluster-operations.md).
