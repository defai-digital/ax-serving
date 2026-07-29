# AX Serving runtime-agent protocol contract

| Field | Value |
| --- | --- |
| Status | Protocol v1.2 source foundation; Dynamo and Mac cluster certification pending |
| Wire types | `crates/ax-serving-protocol` |
| Current version | `1.2` |
| Last updated | 2026-07-28 |

This contract defines how a runtime agent or domain adapter joins the portable AX Serving fleet.
The Rust protocol crate and its JSON fixtures are authoritative when this
document and code differ.

Protocol v1.1 extends the v1.0 worker contract additively so a Dynamo deployment can register as one
execution domain. The types, validation, gateway state propagation, fixtures, and desired-domain
catalog and Dynamo adapter are implemented; live certification is not.

Protocol v1.2 additively defines `mac_ax_engine_cluster` and
`control.mac-cluster.v1`. A source/mock-tested runtime-neutral coordinator adapter implements this
control contract, but there is no distributed AX Engine implementation or live support evidence.

## 1. Runtime ownership

The runtime/domain owner owns model loading, tokenization, templates, batching, cache,
generation, distributed execution, and hardware kernels. The agent owns
runtime discovery, readiness normalization, credential isolation,
cancellation, and byte-preserving HTTP/SSE proxying. The gateway owns fleet
leases, cross-domain admission/selection, equivalence, retry, and operations.

For NVIDIA target deployments, Dynamo owns worker routing, KV-aware placement, disaggregation,
planner/scaling, and backend execution. The AX Dynamo adapter represents the whole deployment and
must not expose internal workers as AX candidates.

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
    "version": {"major": 1, "minor": 1},
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
control.execution-domain.v1
control.inventory-delta
dispatch.cancel
dispatch.typed-admission
telemetry.capacity
telemetry.domain-capacity.v1
telemetry.kv-cache
telemetry.prefix-cache
```

## 4. Registration

`RegisterWorkerRequest` contains six base descriptors plus optional v1.1 domain fields:

| Object | Required content |
| --- | --- |
| `protocol` | Version and optional capabilities |
| `agent` | Agent name, version, optional build SHA |
| `worker` | Stable worker ID, process instance ID, advertised URL, pool, trust domain, bounded labels |
| `runtime` | Runtime kind, version, and API family |
| `hardware` | Platform, accelerator, device count, optional memory and hardware class |
| `domain` | Optional stable execution-domain descriptor; requires `control.execution-domain.v1` |
| `domain_observation` | Optional aggregate domain state, inventory, manifest, and capacity |
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

The current generic agent can translate selected stable AX/vLLM/SGLang Prometheus and
JSON aliases. Direct CUDA use is a migration/testing compatibility path in the final architecture.
Those translations are best-effort adapter behavior, not a
guarantee that every runtime version exports every signal. Pin and test each
certified runtime image.

The Dynamo adapter reports only documented aggregate domain telemetry. Dynamo worker costs,
KV indexes, KVBM ownership, and NIXL transfer metadata do not enter AX state.
When `domain_observation.aggregate_capacity` is present, it is authoritative for gateway admission
and scoring; the generic runtime capacity field is only a fallback. This prevents a domain from
appearing idle when its aggregate view reports pressure.

## 7. Heartbeat and fencing

Each heartbeat contains:

- registration ID and instance ID;
- strictly increasing sequence;
- runtime observation time/status;
- inventory generation;
- optional full model inventory and capacity;
- optional deployment-job observations.

For a registration that declared `control.execution-domain.v1`, `domain_observation` is mandatory
on every new heartbeat sequence. Omitting it rejects that heartbeat; the gateway never refreshes a
domain lease using runtime-only state. This requirement does not apply to v1.0 migration workers.

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

Cross-pool/domain retry additionally requires:

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
conditionals or more fields to the legacy DTOs. Direct vLLM/SGLang agents become
`compatibility_runtime_endpoint` targets after the Dynamo adapter is certified.

## 13. Execution-domain v1.1 and Mac cluster v1.2 extensions

Protocol minor 1 adds optional:

```text
control.execution-domain.v1
telemetry.domain-capacity.v1
ExecutionDomainDescriptor
DomainObservation
```

Protocol minor 2 adds `control.mac-cluster.v1` and the
`mac_ax_engine_cluster` domain kind. A cluster descriptor requires that
capability and cannot be advertised through the v1.0 Mac migration path.

The descriptor includes a stable domain ID, kind, endpoint scope, execution owner, qualification,
pool, trust domain, hardware class, architecture, and compatibility-manifest digest.
Desired `DomainSpec.selector` may pin that digest with `compatibility_manifest` (or
`compatibility_manifest_digest`); a missing or different observed digest then fails eligibility.

Required mappings:

| Domain kind | Scope | Meaning |
| --- | --- | --- |
| `mac_ax_engine` | `node` | One Mac AX Engine endpoint |
| `mac_ax_engine_cluster` | `domain` | One complete model-parallel AX Engine cluster |
| `nvidia_dynamo_pc` | `domain` | One complete NVIDIA PC Dynamo deployment |
| `nvidia_dynamo_thor` | `domain` | One separately qualified Thor Dynamo deployment |
| `compatibility_runtime_endpoint` | `node` | Time-bounded direct runtime migration path |

PC and Thor cannot share a domain/pool by default. Dynamo domain observations are aggregate and must
not contain internal worker IDs or KV state. A Dynamo adapter returns typed non-admission only when
it rejects locally before upstream dispatch or a pinned Dynamo contract proves that execution did
not start.

Current `ax-runtime-agent` registrations advertise the domain capability only when
`AXS_NODE_DOMAIN_ID` is configured. The agent derives a safe kind: AX Engine becomes
`mac_ax_engine`, while every other direct runtime becomes
`compatibility_runtime_endpoint`; it cannot claim a Dynamo or Mac cluster
domain. `AXS_NODE_DOMAIN_QUALIFICATION` defaults to `unverified`, and an
optional `AXS_NODE_DOMAIN_COMPATIBILITY_MANIFEST` carries the retained manifest
digest. Without a domain ID, operators can use the explicit Mac/compatibility
migration mapping. NVIDIA production domains require the separate adapter, a
valid v1.1 descriptor, and a ready aggregate observation.

The protocol types, fixtures, and validation rules in `crates/ax-serving-protocol` are normative.
The public [Mac cluster integration guide](../integrations/mac/CLUSTER.md) documents the current
v1.2 source-level setup and its limitations.
