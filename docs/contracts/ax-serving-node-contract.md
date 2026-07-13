# AX Serving runtime-agent protocol contract

| Field | Value |
| --- | --- |
| Status | Protocol v1 source contract |
| Wire types | `crates/ax-serving-protocol` |
| Current version | `1.0` |
| Last updated | 2026-07-12 |

This contract defines how a runtime agent joins the portable AX Serving fleet.
The Rust protocol crate and its JSON fixtures are authoritative when this
document and code differ.

## 1. Runtime ownership

The runtime owns model loading, tokenization, templates, batching, cache,
generation, distributed execution, and hardware kernels. The agent owns
runtime discovery, readiness normalization, credential isolation,
cancellation, and byte-preserving HTTP/SSE proxying. The gateway owns fleet
leases, admission, endpoint selection, equivalence, retry, and operations.

An agent must not parse model files or invent model identity from a filename.
Unknown runtime facts remain unknown.

## 2. Transport and trust

Control-plane endpoints:

```text
POST /internal/workers/register
POST /internal/workers/{worker_id}/heartbeat
POST /internal/workers/{worker_id}/drain
POST /internal/workers/{worker_id}/drain-complete
```

Inference dispatch uses the agent's advertised OpenAI-compatible HTTP endpoint.
Streaming uses incremental SSE bytes over direct HTTP. A durable broker is not
the primary token-stream transport.

Credential boundaries:

- worker control: `X-Internal-Token` from `AXS_WORKER_TOKEN` or
  `AXS_INTERNAL_API_TOKEN`;
- worker lease: registration-issued `X-Ax-Lease-Token`;
- inference dispatch: `X-Ax-Dispatch-Token`;
- runtime authentication: agent-owned `Authorization` derived from
  `AXS_RUNTIME_API_KEY`.

The client's `Authorization`, cookies, proxy authorization, hop-by-hop
headers, and AX internal headers must not reach the runtime. Non-loopback
control and dispatch require authenticated channels and the `trusted_mesh`
transport profile.

## 3. Protocol negotiation

Registration carries:

```json
{
  "protocol": {
    "version": {"major": 1, "minor": 0},
    "capabilities": ["control.drain", "dispatch.typed-admission"]
  }
}
```

Rules:

- different major versions are incompatible;
- the negotiated minor is the lower supported minor, subject to minimums;
- optional capabilities are an intersection, not runtime-name checks;
- unknown future capabilities and fields must survive tolerant decoding where
  the protocol type permits it;
- a missing required capability makes the endpoint ineligible for that
  operation, not optimistically compatible.

Known protocol capabilities include:

```text
control.drain
control.deployment-jobs
control.inventory-delta
dispatch.cancel
dispatch.typed-admission
telemetry.capacity
telemetry.kv-cache
telemetry.prefix-cache
```

## 4. Registration

`RegisterWorkerRequest` contains six descriptors:

| Object | Required content |
| --- | --- |
| `protocol` | Version and optional capabilities |
| `agent` | Agent name, version, optional build SHA |
| `worker` | Stable worker ID, process instance ID, advertised URL, pool, trust domain, bounded labels |
| `runtime` | Runtime kind, version, and API family |
| `hardware` | Platform, accelerator, device count, optional memory and hardware class |
| `observation` | Runtime-authoritative status, inventory generation, model descriptors, optional capacity |

Worker identity has two levels:

- stable `worker.id`, chosen by the operator;
- unique `worker.instance_id`, regenerated for each agent process.

Re-registering a stable worker ID creates a new registration and fences the old
instance. The advertised URL must use HTTP or HTTPS, contain an IP address and
port, contain no credentials/path/query/fragment, and avoid wildcard,
multicast, or link-local destinations.

The response returns:

- registration ID;
- opaque lease token, redacted by `Debug`;
- negotiated protocol;
- heartbeat interval and lease TTL;
- optional inventory-resync directive.

## 5. Runtime observation

Runtime status includes:

| Field | Meaning |
| --- | --- |
| `ready` | Runtime can accept an operation now |
| `state` | `starting`, `ready`, `degraded`, `draining`, `unavailable`, or `unknown` |
| `reason_code` | Bounded machine-readable reason |
| `message` | Safe diagnostic text without secrets or model paths |
| `probe_latency_ms` | Optional observation latency |

`ready` must agree with state. Startup, unknown, unavailable, and draining
observations are not eligible. Observation timestamps outside the configured
clock-skew envelope are rejected.

Each model descriptor contains:

- runtime model ID;
- runtime kind/version, revision/artifact digest, tokenizer digest, template
  digest, and quantization where known;
- supported operations and capabilities;
- optional context and output limits.

Supported operation names are currently `chat_completions`,
`text_completions`, and `embeddings`. `responses` exists as a future protocol
operation but is not a released public endpoint until adapter certification.

Digests use lowercase `sha256:<64 hex>` or `blake3:<64 hex>`. Missing identity
is not synthesized. It can support a pinned single-pool deployment when policy
allows, but cannot authorize cross-runtime equivalence.

## 6. Capacity observation

All capacity fields are optional. Available fields include:

- active and maximum concurrent requests;
- waiting requests and process RSS;
- recent error rate;
- KV-cache used ratio and prefix-cache hit ratio;
- batch token capacity and use;
- TTFT and inter-token EWMA;
- generated tokens per second and observation window.

Ratios must be finite in `[0,1]`; latency/throughput values must be finite and
non-negative; active/batch use cannot exceed an advertised maximum. Unknown
signals remain absent. The gateway penalizes absent or stale telemetry instead
of treating it as idle.

The generic agent can translate selected stable AX/vLLM/SGLang Prometheus and
JSON aliases. Those translations are best-effort adapter behavior, not a
guarantee that every runtime version exports every signal. Pin and test each
certified runtime image.

## 7. Heartbeat and fencing

Each heartbeat contains:

- registration ID and instance ID;
- strictly increasing sequence;
- runtime observation time/status;
- inventory generation;
- optional full model inventory and capacity;
- optional deployment-job observations.

It carries `X-Ax-Lease-Token`. The gateway rejects:

- missing or expired leases;
- a different registration or process instance;
- replayed or decreasing sequence numbers;
- malformed/inconsistent observations.

Workers become ineligible when fresh authoritative observations stop and are
removed after lease TTL. In HA mode, registration and heartbeat are written to
shared state atomically, old registrations are fenced, and only one gateway
owns an active probe lease for a worker at a time.

Heartbeat responses may request:

- begin drain;
- drain complete;
- inventory resync;
- re-registration;
- negotiated deployment commands when supported.

The agent must apply a begin-drain directive immediately: stop new admission,
allow current streams to finish, then report completion at zero inflight or the
operator deadline.

## 8. Dispatch contract

Gateway requests include opaque request and attempt IDs and, on remote
profiles, a dispatch credential. The agent:

1. authenticates dispatch before reading an unbounded body;
2. rejects while draining or locally saturated with a typed pre-admission
   response;
3. forwards only approved headers and the runtime credential;
4. preserves unknown JSON request fields and SSE byte order;
5. propagates body drop/client cancellation to the runtime connection;
6. sanitizes runtime transport errors.

Typed non-admission uses `X-Ax-Admission-State: not-admitted` and a stable AX
error code. The gateway trusts that marker only on the authenticated agent
channel. A generic runtime `5xx` is not a safe retry signal.

## 9. Deployment identity and failover

The gateway resolves a logical model to explicit deployments and pools. It
checks runtime readiness, lease freshness, drain, pool/trust policy, operation,
capabilities, context/output limits, identity, equivalence, and shared capacity
before scoring.

Cross-pool retry additionally requires:

- source and target in the same operator-certified equivalence class;
- both deployment IDs listed in the certification artifact;
- every required identity field present and matching.

The protocol does not promise bit-identical output. Equivalence is an explicit
operator policy backed by a retained workload artifact.

## 10. Lifecycle jobs

Agents may advertise `control.deployment-jobs` and report job observations.
AX Serving's admin jobs always represent desired and observed control-plane
state. Runtime process creation, image rollout, model download, and GPU
allocation remain external unless a certified lifecycle adapter explicitly
implements them.

## 11. Conformance requirements

A runtime adapter is not certified until tests cover:

1. protocol negotiation and future-field tolerance;
2. startup unavailable and later ready;
3. runtime failure after registration;
4. inventory add/remove/resync;
5. every claimed operation and modality;
6. blocking and streaming byte preservation;
7. client cancellation;
8. drain with inflight work;
9. public credential non-forwarding;
10. typed local non-admission;
11. generic runtime `5xx` remaining non-retryable;
12. exact runtime image/model identity and retained result metadata.

Mock adapter tests validate protocol logic but do not certify a live runtime
version or model artifact.

## 12. Legacy compatibility

The same internal paths still decode the pre-v1 registration and heartbeat
shape for migration. Legacy workers lack registration fencing, complete
deployment identity, and negotiated capabilities. They are accepted only in
`legacy_compat` routing and cannot participate in certified cross-runtime
failover.

New integrations must use the protocol crate rather than adding runtime-name
conditionals or more fields to the legacy DTOs.
