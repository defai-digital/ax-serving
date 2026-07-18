# Technical Specification: Federated Inference Control Plane

| Field | Value |
| --- | --- |
| Status | Approved target design; implementation is incremental |
| Last updated | 2026-07-15 |
| Target | AX Serving 3.x |
| Product requirements | [AX Serving PRD](../prd/PRD-AX-SERVING.md) |
| Decision | [ADR-016](../adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md) |
| Evidence | [Implementation and certification status](../IMPLEMENTATION-STATUS.md) |

## 1. Purpose

This specification turns ADR-016 into an incremental implementation plan. It defines how the
existing portable AX Serving gateway becomes a federation plane over:

- Mac AX Engine endpoints registered by `ax-runtime-agent`;
- NVIDIA GPU PC deployments managed internally by upstream NVIDIA Dynamo;
- NVIDIA Thor deployments managed by a separate Dynamo domain.

The key implementation rule is:

> AX Serving selects and governs an execution domain. Dynamo selects NVIDIA workers. AX Engine
> executes on Mac. No layer makes the same placement decision twice.

The specification is normative for target behavior. The implementation ledger is authoritative for
what the current source and released artifacts have actually proved.

## 2. Verified starting point

The current repository already implements foundations that must be extended, not replaced:

| Existing capability | Current source |
| --- | --- |
| Portable OpenAI JSON/SSE gateway | `crates/ax-serving-api` |
| Runtime-neutral wire DTOs and fixtures | `crates/ax-serving-protocol` |
| Explicit pools/deployments/equivalence | `orchestration/deployment.rs`, protocol `deployment.rs` |
| Bounded request classification | `orchestration/request_profile.rs` |
| Hard eligibility and endpoint scoring | `orchestration/registry.rs`, `policy.rs` |
| Typed pre-admission, at-most-one retry, SSE, cancellation | `orchestration/direct.rs`, `proxy_handlers.rs` |
| Worker lease, inventory, drain, capacity | `orchestration/registry.rs`, protocol `worker.rs` |
| Desired deployment and async job state | `deployment_lifecycle.rs`, `jobs.rs`, protocol `lifecycle.rs` |
| Memory and Redis/Valkey HA state | `orchestration/fleet_state.rs` |
| AX Engine/OpenAI runtime agent | `crates/ax-thor-agent`, binary `ax-runtime-agent` |
| CPU-only container, Compose, Kustomize, Helm | `packaging/container`, `deploy/` |
| Control-plane `/readyz` and capacity `/routablez` | gateway operations/routes and deployment config |

Important current gaps:

- protocol v1.1 domain/decision types, tolerant v1.0 fixtures, catalog resolution, and bounded
  in-process decision diagnostics are implemented;
- direct vLLM/SGLang agent registration exists, but the target NVIDIA path is a Dynamo domain;
- no Dynamo Domain Adapter or pinned Dynamo compatibility manifest exists;
- `RequestProfile` carries a versioned decision profile, but authenticated cost, locality, privacy,
  quality, and routing-profile policy inputs are not wired yet;
- decision records currently contain eligible domain candidates in a bounded in-process journal;
  rejected-candidate evidence, immutable policy config, durable storage, and offline replay remain;
- agent-session design exists only in superseded documentation and is not implemented;
- source tests do not certify a live AX Engine + Dynamo + Thor fleet or prove product value.

Implementation must preserve current public APIs and protocol fixtures while introducing the new
domain model additively.

## 3. Release-blocking invariants

1. The portable gateway dependency graph contains no AX Engine, MLX, Metal, Dynamo, CUDA, vLLM,
   SGLang, TensorRT-LLM, NIXL, or model-runtime SDK.
2. A Dynamo deployment appears to AX Serving as one domain endpoint. Dynamo workers never become AX
   endpoint-picker candidates.
3. PC and Thor use separate domain IDs, pools, qualification, artifacts, telemetry calibration,
   and rollout state.
4. One AX attempt enters one execution domain and remains there.
5. Hard policy, capability, identity, equivalence, readiness, and capacity checks precede scoring.
6. Missing or stale observations never mean ready, idle, equivalent, or cheap.
7. AX retries only before admission/commitment; Dynamo owns in-domain retry and migration.
8. Request bodies and SSE bytes remain preserved except for the existing top-level runtime-model
   rewrite and explicit bounded validation.
9. Decision/audit/fleet state contains no prompt, output, raw session/affinity ID, credential, KV
   state, or Dynamo worker index.
10. A learned policy cannot affect production until offline replay, shadow, canary, and rollback
    gates pass.

## 4. Target topology

### 4.1 Logical topology

```text
                        Clients
                           |
                     public TLS/auth
                           |
              +------------v-------------+
              | AX Serving gateways (HA) |
              | logical model + policy   |
              | domain picker + audit    |
              +------+-----------+-------+
                     |           |
                Redis/Valkey     | authenticated dispatch
                                 |
              +------------------+------------------+
              |                  |                  |
        Mac pool endpoints  PC Dynamo adapter  Thor Dynamo adapter
              |                  |                  |
       ax-runtime-agent       Dynamo frontend    Dynamo frontend
              |                  |                  |
         AX Engine          Dynamo PC graph     Dynamo Thor graph
                             |  |  |              |  |  |
                           backend workers      backend workers
```

### 4.2 Data paths

Mac:

```text
client -> AX gateway -> ax-runtime-agent -> ax-engine-server -> agent -> gateway -> client
```

NVIDIA:

```text
client -> AX gateway -> ax-dynamo-adapter -> Dynamo frontend/router -> backend worker(s)
       <- incremental JSON/SSE over the same admitted domain attempt <-
```

The adapter may be colocated with or near the Dynamo frontend, but it is an AX trust and protocol
boundary. It is not a second Dynamo router.

### 4.3 Control paths

- AX gateway to AX adapters: registration, lease, observation, drain, and optional deployment-job
  protocol.
- Dynamo internal: discovery, endpoints/CRDs, NATS/etcd where required, planner, operator, KV events,
  KVBM, and NIXL. AX does not read or mutate these stores directly.
- Optional lifecycle bridge: desired AX domain/deployment state to a certified Dynamo Kubernetes
  controller. It is asynchronous and separate from inference dispatch.
- AX active-active state: AX-owned Redis/Valkey only.

## 5. Terminology and domain model

### 5.1 Strong identifiers

Add to `ax-serving-protocol`:

```rust
pub struct DomainId(String);
pub struct PolicyId(String);
pub struct PolicyVersion(String);
pub struct CompatibilityManifestDigest(Digest);
```

Use the same bounded identifier rules as existing `PoolId` and `DeploymentId`. IDs are opaque,
stable operator identities and are never derived from mutable URLs.

### 5.2 Domain kind and scope

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDomainKind {
    MacAxEngine,
    NvidiaDynamoPc,
    NvidiaDynamoThor,
    CompatibilityRuntimeEndpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointScope {
    Node,
    Domain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationState {
    Unverified,
    Experimental,
    Certified,
    Suspended,
}
```

Required combinations:

| Kind | Scope | Execution owner |
| --- | --- | --- |
| `mac_ax_engine` | `node` | `ax_engine` |
| `nvidia_dynamo_pc` | `domain` | `dynamo` |
| `nvidia_dynamo_thor` | `domain` | `dynamo` |
| `compatibility_runtime_endpoint` | `node` | configured legacy runtime |

Invalid combinations fail registration/config validation.

### 5.3 Domain descriptor

Protocol v1.1 adds an optional descriptor:

```rust
pub struct ExecutionDomainDescriptor {
    pub id: DomainId,
    pub kind: ExecutionDomainKind,
    pub endpoint_scope: EndpointScope,
    pub execution_owner: String,
    pub qualification: QualificationState,
    pub pool_id: PoolId,
    pub trust_domain: TrustDomainId,
    pub hardware_class: String,
    pub architecture: String,
    pub compatibility_manifest: Option<CompatibilityManifestDigest>,
    pub labels: BTreeMap<String, String>,
}
```

All strings and labels use existing protocol bounds and allowlists. `labels` may describe region,
zone, rack, energy class, or operator group; they must not contain credentials, user data, or
unbounded hardware inventory.

### 5.4 Desired domain specification

Add to gateway config/catalog:

```rust
pub struct DomainSpec {
    pub id: DomainId,
    pub kind: ExecutionDomainKind,
    pub pool: PoolId,
    pub trust_domain: TrustDomainId,
    pub hardware_class: String,
    pub required_qualification: QualificationState,
    pub selector: BTreeMap<String, String>,
    pub enabled: bool,
}
```

Extend `DeploymentSpec` additively with `domain: Option<DomainId>`. In final explicit mode, every
deployment names a domain. During migration, a missing domain is accepted only when its pool maps
to exactly one enabled domain; ambiguous resolution fails startup.

`PoolSpec` remains a policy/maintenance grouping. A domain is an execution and failure boundary.
Several Mac nodes can share one pool; each Dynamo deployment normally has its own domain and pool.
The desired selector key `compatibility_manifest` pins the descriptor digest and fails closed when
the adapter omits or changes it.

### 5.5 Observed domain state

```rust
pub struct DomainObservation {
    pub observed_at: OffsetDateTime,
    pub generation: u64,
    pub ready: bool,
    pub state: RuntimeState,
    pub reason_code: Option<String>,
    pub frontend_instances_ready: Option<u32>,
    pub aggregate_capacity: Option<CapacityObservation>,
    pub manifest_digest: Option<CompatibilityManifestDigest>,
    pub models: Vec<RuntimeModelDescriptor>,
}
```

The observation is aggregate at the Dynamo domain boundary. It must not include Dynamo worker IDs,
per-worker KV keys, transfer handles, or placement decisions.
When aggregate domain capacity is present, the gateway uses it ahead of the generic endpoint
capacity view for admission and scoring.

## 6. Workspace and component layout

### 6.1 `ax-serving-protocol`

Add `domain.rs` and `decision.rs` containing serialization, validation, and redaction only. Extend
registration/heartbeat DTOs with optional domain fields under protocol v1.1. Keep all v1.0 fixtures
and tolerant decoding tests.

The protocol crate remains platform-neutral and has no HTTP client, Kubernetes, Dynamo, or runtime
dependency.

### 6.2 `ax-serving-api`

Extend existing modules rather than create a parallel orchestrator:

| Module | Change |
| --- | --- |
| `config.rs` | Parse/validate `domains`, policy mode, and compatibility manifest references. |
| `orchestration/deployment.rs` | Add desired domain catalog and deployment-to-domain resolution. |
| `orchestration/registry.rs` | Store observed domain metadata alongside existing endpoint state. |
| `orchestration/request_profile.rs` | Add bounded `DecisionProfile` fields without prompt parsing. |
| `orchestration/policy.rs` | Select domains/endpoints after hard filtering; retain current policies for rollback. |
| `orchestration/direct.rs` | Record selected domain and enforce retry-owner semantics. |
| `orchestration/fleet_state.rs` | Persist domain lease/observation and decision-policy versions in HA state. |
| `orchestration/gateway_ops.rs` | Expose domain diagnostics and readiness/routability summaries. |

Internal type names such as `WorkerRegistry` may remain during migration. Public/admin responses and
new code must use `endpoint` or `domain` accurately rather than calling a whole Dynamo deployment a
GPU worker.

### 6.3 Adapter core

Extract the portable common logic from `crates/ax-thor-agent` into a focused internal crate only
when doing so can preserve behavior through existing tests:

```text
crates/ax-serving-adapter-core
  registration / lease / heartbeat
  drain and typed local admission
  dispatch authentication
  header allowlist and SSE proxy helpers
  bounded observation helpers
```

Do not block the first Dynamo prototype on this refactor. Duplication may be temporarily accepted
behind conformance tests, then removed before production qualification.

### 6.4 Mac runtime agent

Keep the released binary name `ax-runtime-agent`. The package may later be renamed from
`ax-thor-agent` through a compatibility-preserving workspace change. Its target runtime kind is
`ax_engine`; direct vLLM/SGLang modes become compatibility-only after Dynamo PC certification.

### 6.5 New `ax-dynamo-adapter`

Add a portable Rust binary/crate:

```text
crates/ax-dynamo-adapter/
  src/config.rs
  src/domain_observer.rs
  src/inventory.rs
  src/metrics.rs
  src/proxy.rs
  src/registration.rs
  src/manifest.rs
  tests/conformance.rs
```

It uses HTTP/OpenAI/Prometheus and optional Kubernetes API boundaries. It must not import Dynamo's
Python packages or link CUDA/NIXL/backend libraries. A later lifecycle-controller binary may use
Kubernetes CRDs, but the data-plane adapter remains usable without Kubernetes.

## 7. Protocol negotiation and migration

Current protocol remains major 1. Add minor 1 and capability:

```text
control.execution-domain.v1
telemetry.domain-capacity.v1
```

Rules:

- v1.0 registrations decode unchanged.
- A v1.1 domain adapter must advertise `control.execution-domain.v1` and provide a valid domain
  descriptor.
- Every new heartbeat sequence for a registered v1.1 domain must carry a fresh aggregate domain
  observation; runtime-only heartbeats cannot refresh domain eligibility.
- `nvidia_dynamo_pc` and `nvidia_dynamo_thor` registrations without the capability are rejected.
- A v1.0 endpoint can be mapped only to `compatibility_runtime_endpoint` or an explicit Mac
  migration declaration; runtime-name inference cannot silently create a production domain.
- Unknown future domain kinds/capabilities make the endpoint ineligible while preserving safe
  diagnostic fields.
- Old gateway replicas that cannot decode the new minor must reject registration cleanly; rolling
  upgrade proceeds gateway-first only after fixture compatibility is proven.

## 8. Configuration contract

### 8.1 Example final topology

```yaml
orchestrator:
  deployment_mode: explicit
  dispatch_policy: domain_aware
  decision_policy:
    id: explicit-safe-v1
    version: "1"
    mode: active
  domains:
    - id: mac-local
      kind: mac_ax_engine
      pool: mac-mlx
      trust_domain: private-local
      hardware_class: apple-silicon
      required_qualification: certified
      selector:
        worker_pool: mac-mlx
      enabled: true
    - id: nvidia-pc-main
      kind: nvidia_dynamo_pc
      pool: nvidia-pc
      trust_domain: private-dc
      hardware_class: nvidia-pc-cuda
      required_qualification: certified
      selector:
        domain_id: nvidia-pc-main
      enabled: true
    - id: nvidia-thor-lab
      kind: nvidia_dynamo_thor
      pool: nvidia-thor
      trust_domain: private-edge
      hardware_class: nvidia-thor
      required_qualification: experimental
      selector:
        domain_id: nvidia-thor-lab
      enabled: false
```

Deployment declarations additionally name `domain`. PC and Thor never share a pool in the default
schema examples.

### 8.2 Policy configuration

P0 supports deterministic policies:

- `explicit_safe`: required/preferred domain and existing inference-aware signals;
- `domain_aware`: hard domain filters, then bounded domain scoring;
- existing policies remain available only for Mac-node selection and rollback.

P1 adds:

```yaml
decision_policy:
  id: cost-slo-router
  version: "2026-07-15.1"
  mode: shadow # shadow | canary | active
  canary_percent: 0
  max_cost_usd: null
  latency_slo_ms: null
  quality_floor: null
  rollback_guard:
    error_rate_delta: 0.01
    p95_latency_delta: 0.10
```

Secrets are environment/secret-store only. Config files contain references and non-secret policy.

### 8.3 Dynamo adapter configuration

```text
AXS_DYNAMO_DOMAIN_ID
AXS_DYNAMO_DOMAIN_KIND=nvidia_dynamo_pc|nvidia_dynamo_thor
AXS_DYNAMO_FRONTEND_URL
AXS_DYNAMO_METRICS_URL
AXS_DYNAMO_MANIFEST_PATH
AXS_DYNAMO_PROBE_INTERVAL_MS
AXS_DYNAMO_REQUEST_TIMEOUT_SECS
AXS_DYNAMO_API_KEY
```

The adapter also uses existing AX control/dispatch credentials. URLs are validated against the same
SSRF, scheme, credential, wildcard, link-local, and trusted-mesh rules as current runtime
endpoints.

## 9. Dynamo compatibility manifest

Every NVIDIA domain loads a signed or digest-pinned manifest:

```json
{
  "schema_version": 1,
  "domain_kind": "nvidia_dynamo_pc",
  "dynamo": {
    "repository": "https://github.com/ai-dynamo/dynamo",
    "tag": "v1.2.1",
    "commit": "<full commit>",
    "release_url": "https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1"
  },
  "components": {
    "frontend": "nvcr.io/nvidia/ai-dynamo/dynamo-frontend@sha256:<digest>",
    "planner": "nvcr.io/nvidia/ai-dynamo/dynamo-planner@sha256:<digest>",
    "operator": "nvcr.io/nvidia/ai-dynamo/kubernetes-operator@sha256:<digest>",
    "runtime": "nvcr.io/nvidia/ai-dynamo/<backend>-runtime@sha256:<digest>"
  },
  "backend": {"kind": "vllm", "version": "<version>"},
  "platform": {"arch": "amd64", "os": "ubuntu-24.04", "cuda": "<version>"},
  "graph_config_digest": "sha256:<digest>",
  "model_certifications": ["sha256:<digest>"],
  "issued_at": "<rfc3339>",
  "evidence": "<immutable artifact reference>"
}
```

Validation rules:

- repository must exactly match the canonical upstream unless a separately approved vendor mirror
  is recorded;
- tag, commit, and every deployed image digest are mandatory;
- backend and platform must match observations and domain kind;
- Thor requires `arm64`, its approved Ubuntu/CUDA baseline, and Thor-specific evidence;
- floating tags, `main`, mutable chart values, or missing graph digest fail qualification;
- changing any identity field increments deployment/domain generation and requires requalification.

The adapter advertises only the manifest digest. Detailed contents remain in operator/audit storage
and are not sent on every heartbeat.

## 10. Dynamo Domain Adapter behavior

### 10.1 Startup state machine

```text
LoadConfig
  -> ValidateManifest
  -> ProbeFrontend
  -> DiscoverInventory
  -> RegisterNotReady
  -> Ready
  -> Draining | Degraded | Unavailable
```

The adapter process may be live while the domain is not routable. Registration can occur in
`starting`/not-ready state so diagnostics exist during Dynamo rollout.

### 10.2 Readiness

The domain is ready only when all required conditions are true:

- manifest is valid and matches configured domain kind;
- at least one configured Dynamo frontend endpoint is ready;
- the required logical deployment inventory is observed;
- adapter-to-Dynamo authentication and transport pass;
- aggregate admission status is not draining or unavailable;
- observation is within freshness and clock-skew bounds;
- qualification meets the desired domain's minimum state.

Do not infer readiness from a Kubernetes Pod being `Running`, a TCP port opening, or the adapter
process health alone.

### 10.3 Inventory

The adapter queries the stable OpenAI model surface or a version-pinned documented Dynamo surface.
It maps runtime model IDs, operations, limits, and capabilities only when the selected backend and
manifest conformance prove them.

Model identity comes from the manifest/certification record, not a filename or the `/v1/models`
string alone. Inventory mismatch makes the affected deployment ineligible; it does not silently
rewrite desired identity.

### 10.4 Aggregate capacity and telemetry

Allowed normalized signals:

- domain pending/admitted request count;
- documented frontend overload/load-shed state;
- aggregate TTFT/ITL/throughput/error observations with window and freshness;
- optional domain capacity or planner target when semantics are documented;
- optional aggregate KV hit ratio only as a domain-level score.

Forbidden AX inputs:

- individual Dynamo worker ranking or active slot cost;
- per-worker KV prefix hashes/index;
- KVBM block ownership or NIXL transfer metadata;
- planner coefficients copied into AX scoring;
- a sum/average of heterogeneous metrics whose units or windows do not match.

Every mapping is keyed by Dynamo release/backend in the compatibility manifest. Unknown metrics
remain absent and receive the gateway's conservative penalty.

### 10.5 Request proxy

The adapter:

1. authenticates AX dispatch before reading a large body;
2. enforces local drain and bounded inflight admission;
3. validates request/attempt/domain IDs and trace context;
4. adds only configured Dynamo credential and documented headers;
5. preserves the rewritten OpenAI body and unknown fields;
6. sends one request to the Dynamo frontend service;
7. streams response bytes incrementally with backpressure;
8. propagates body drop, cancellation, and deadline;
9. sanitizes upstream transport errors and strips credentials/internal headers.

It does not choose a Dynamo component or worker, compute KV overlap, or direct-route to an internal
worker. Dynamo frontend/router is always the admitted execution boundary.

### 10.6 Typed non-admission

The adapter returns AX `not-admitted` only if:

- it rejects locally before sending upstream because it is draining or its bounded local admission
  limit is full; or
- the pinned Dynamo contract supplies an authenticated, unambiguous pre-admission rejection that
  conformance tests prove means no execution started.

Timeout, connection reset after request write, backend `5xx`, malformed upstream response, or
unknown Dynamo error is admission-ambiguous and cannot trigger AX cross-domain retry.

### 10.7 Lifecycle

P0 lifecycle is observational: adapter drain and readiness plus operator-managed Dynamo deployment.

P1 may add a separate controller that maps AX async jobs to documented Dynamo Kubernetes resources
such as a pinned `DynamoGraphDeployment`. Requirements:

- idempotent desired state with generation/fencing;
- separate controller credential and RBAC;
- no Kubernetes/Dynamo control API call in the request path;
- observe replacement ready before disabling source;
- rollback to the prior immutable graph/manifest;
- report job progress/failure without claiming AX allocates GPUs itself.

## 11. Mac AX Engine adapter behavior

Retain current protocol semantics:

- agent process liveness and AX Engine readiness are separate;
- `/v1/models`/runtime metadata and pinned fixtures define observed capabilities;
- the agent reports `ExecutionDomainKind::MacAxEngine`, scope `Node`, architecture `arm64`, and a
  stable domain/pool mapping;
- model identity is operator/runtime supplied and validated against explicit deployment config;
- inference-aware Mac endpoint selection uses fresh queue/KV/batch/TTFT/error signals when present;
- missing signals are penalized;
- the agent handles typed local non-admission, drain, cancellation, and SSE preservation;
- no public credential or raw affinity/session hint reaches AX Engine.

The gateway does not select AX Engine Direct/MTP/n-gram/precision modes in P0. Such execution policy
belongs to AX Engine or a future explicit, typed AX Engine policy contract.

## 12. PC and Thor domain profiles

### 12.1 NVIDIA PC

Production candidate requirements:

- AMD64 or separately certified ARM64 PC/server class;
- pinned supported OS, driver, CUDA, Dynamo, backend, NIXL, charts, graph, and model artifacts;
- stable frontend/data and controller endpoints;
- PC-specific profiling and capacity model;
- one certified backend/model combination for P0; other Dynamo backends are not inherited claims.

### 12.2 NVIDIA Thor

Thor defaults:

```text
kind = nvidia_dynamo_thor
scope = domain
architecture = arm64
qualification = experimental
pool != any PC pool
enabled = false until operator opt-in
```

Promotion requires:

- exact Thor device/firmware/JetPack or OS baseline;
- compatible CUDA 13/driver/container runtime;
- pinned Dynamo ARM64 component/runtime images;
- backend/model capability and artifact validation;
- memory pressure, unified/shared memory behavior, thermal/power, restart cleanup, long-context,
  stream/cancel, load-shed, and 60-minute soak evidence;
- proof that no PC-only TensorRT engine, profiling curve, or planner coefficient is reused;
- safe removal of the entire Thor domain without PC/Mac impairment.

Generic Dynamo ARM64 + Blackwell support is a prerequisite, not this evidence.

## 13. Request profile and domain admission

### 13.1 P0 profile

Keep existing `RequestProfile` and add a nested, versioned structure:

```rust
pub struct DecisionProfileV1 {
    pub routing_profile: Option<String>,
    pub required_domain: Option<DomainId>,
    pub preferred_domain: Option<DomainId>,
    pub privacy_class: Option<String>,
    pub locality: Option<String>,
    pub max_cost_microusd: Option<u64>,
    pub latency_slo_ms: Option<u64>,
    pub quality_floor: Option<String>,
}
```

All values are authenticated policy inputs or bounded client hints validated against tenant policy.
Do not accept arbitrary free-form strings into logs/metrics. P0 may leave cost/quality predictions
unset; their presence in the type is not a support claim.

### 13.2 Eligibility order

For each request:

1. authenticate client and resolve tenant/project;
2. validate unique routing-sensitive JSON/header fields and size/deadline limits;
3. resolve explicit logical model or approved routing profile;
4. enumerate desired deployments and domains;
5. filter domain enabled/qualification/lease/readiness/drain/freshness;
6. filter tenant, trust, privacy, locality, residency, and explicit constraints;
7. filter operation, modality, capabilities, context/output limits;
8. validate observed versus desired deployment identity;
9. apply equivalence policy for fallback candidates;
10. acquire shared domain/endpoint capacity reservation;
11. score only remaining candidates;
12. record selection and dispatch.

Empty candidate sets produce a stable reason-specific error before dispatch.

## 14. Domain selection and decision records

### 14.1 P0 deterministic score

Adapt the current inference-aware policy to domain observations without importing Dynamo worker
signals:

```text
score =
  active_capacity_pressure
  + aggregate_queue_pressure
  + aggregate_ttft_pressure
  + recent_error_pressure
  + stale_or_missing_penalty
  + configured_domain_cost_penalty
  + stable_tie_jitter
  - explicit_locality_bonus
```

Lower is better. Weights are versioned configuration. Hard constraints are never represented as
finite penalties. Mac-node selection may then use the existing endpoint score within the chosen Mac
pool; a Dynamo choice is dispatched to its adapter without another AX worker selection.

### 14.2 Decision record

```rust
pub struct DecisionRecordV1 {
    pub request_id: RequestId,
    pub operation: Operation,
    pub logical_model: LogicalModelId,
    pub routing_profile: Option<String>,
    pub policy_id: PolicyId,
    pub policy_version: PolicyVersion,
    pub policy_mode: PolicyMode,
    pub candidate_summary: Vec<CandidateDecision>,
    pub selected_domain: DomainId,
    pub selected_deployment: DeploymentId,
    pub reason_codes: BTreeSet<DecisionReasonCode>,
    pub observation_generations: BTreeMap<DomainId, u64>,
    pub predicted_cost_microusd: Option<u64>,
    pub predicted_latency_ms: Option<u64>,
    pub decided_at: OffsetDateTime,
}
```

`CandidateDecision` contains only domain/deployment ID, eligible boolean, bounded rejection reasons,
and optional normalized score. Cap candidate records at the configured maximum. Store detailed
records in bounded audit storage; emit only low-cardinality aggregates to metrics.

### 14.3 Shadow and canary

- `shadow`: active explicit policy selects; candidate policy records a counterfactual decision only.
- `canary`: deterministic request hashing selects the candidate policy for a configured percentage;
  hard safety filters remain shared and cannot be bypassed.
- `active`: candidate policy is primary after gates pass.
- rollback: atomically restore a prior immutable policy version without deleting decision evidence.

Policy inputs, models, datasets, calibration, and feature definitions are immutable artifacts.

## 15. Identity, equivalence, and failover

Extend existing `DeploymentIdentity` rather than replace it. Domain and compatibility-manifest
identity live in the domain/deployment catalog; model semantic identity remains:

- runtime/backend kind and version;
- model revision and artifact digest;
- tokenizer and template digest;
- quantization;
- operations, capabilities, context/output limits.

Strict cross-domain equivalence requires all operator-configured fields present and matching plus:

- both deployment IDs in the equivalence policy;
- non-empty immutable certification artifact;
- same requested operation/modality/capability;
- quality floor and safety tests for the target workload;
- compatible trust/residency policy.

Different formats or quantizers may be certified as behaviorally acceptable but must not be called
artifact-identical. Certification records disclose the difference and test tolerance.

Cross-domain retry candidate order is fixed from the request's initial catalog/policy snapshot. A
mid-request config change cannot introduce a new target.

## 16. Dispatch state machine and retry ownership

```text
Classified
  -> AdmittedByAX
  -> DomainReserved
  -> ConnectingAdapter
  -> SentToDomain
  -> DomainAdmitted | AdmissionAmbiguous | NotAdmitted
  -> HeadersCommitted
  -> BodyStreaming
  -> Completed | Cancelled | Failed
```

Second AX attempt is allowed only from `ConnectingAdapter` connect failure or trusted
`NotAdmitted`, before `HeadersCommitted`, within deadline, and to an equivalent eligible target.

`SentToDomain` without a proven non-admission acknowledgment becomes `AdmissionAmbiguous`; AX must
not retry. Dynamo's internal request migration or backend retry remains one AX domain attempt and is
visible only through trace/aggregate diagnostics.

The adapter and gateway propagate:

```text
x-ax-request-id
x-ax-attempt-id
traceparent / tracestate / bounded baggage
```

They never use these as credentials or metrics labels.

## 17. Fleet state and HA

Extend the existing `FleetStateStore` with versioned records:

```text
DomainLeaseRecord
DomainObservationRecord
DomainReservationRecord
PolicyActivationRecord
DecisionAuditPointer (optional; not the full prompt/request)
```

Requirements:

- atomic registration/fencing by stable domain ID plus process instance/generation;
- heartbeat sequence monotonicity and observation freshness;
- shared domain reservations before dispatch and bounded release/reconciliation;
- one active probe owner per adapter endpoint;
- policy activation uses compare-and-set generation;
- expired/suspended domain remains diagnostic but ineligible;
- Redis failure fails closed for new HA admissions; existing streams continue when possible;
- Dynamo internal etcd/NATS/Kubernetes state never shares the AX Redis namespace or credentials.

Memory mode implements identical semantics for one-gateway evaluation without HA claims.

## 18. Health, drain, and lifecycle

### 18.1 Gateway probes

- `/livez`: process event loop/listener alive.
- `/readyz`: control plane initialized, accepting registration, and required fleet store fresh;
  independent of execution capacity in default `control_plane` mode.
- `/routablez`: at least one deployment/domain is currently eligible for configured routing scope.
- `/health`: JSON summary that distinguishes control-plane health from capacity.

Legacy `readyz_mode=eligible_workers` remains migration-only.

### 18.2 Adapter/domain drain

1. mark local adapter not admitting;
2. send draining observation before stopping heartbeat;
3. reject new dispatch with typed local non-admission;
4. allow admitted streams to finish until deadline;
5. ask Dynamo/AX lifecycle owner to drain through its documented boundary;
6. report drain complete at zero AX-visible inflight or hard deadline;
7. expire/fence registration on restart.

AX gateway shutdown separately fails readiness, stops admission, waits propagation, drains accepted
requests, cancels at hard deadline, releases reservations, and exits before platform termination.

## 19. Security design

### 19.1 Trust zones

| Zone | Credential |
| --- | --- |
| Client -> AX public API | `AXS_API_KEY` or configured identity provider |
| Operator/admin/metrics | separate admin/monitoring identity |
| Adapter -> AX registration/heartbeat | worker-control identity and lease token |
| AX -> adapter dispatch | dispatch identity |
| Dynamo adapter -> Dynamo frontend | Dynamo data-plane identity |
| Lifecycle controller -> Dynamo/Kubernetes | separate controller/RBAC identity |
| Mac agent -> AX Engine | runtime identity |
| AX gateways -> Redis/Valkey | fleet-store identity |

No credential crosses zones by reuse or forwarding. Rotation is independently testable.

### 19.2 Header policy

Deny by default:

- public `Authorization`, cookies, proxy authorization;
- hop-by-hop headers;
- AX control/lease/dispatch headers from clients;
- Kubernetes/Dynamo internal routing headers;
- raw session/affinity identifiers;
- upstream `Set-Cookie` and authentication challenges not intended for clients.

Allow only documented OpenAI/content/tracing headers and adapter-generated bounded internal fields.

### 19.3 Threat tests

- malicious registration/domain kind/manifest mismatch;
- replayed lease/sequence and stale generation takeover;
- SSRF and DNS rebinding on adapter/frontend URLs;
- duplicate routing JSON/header fields and request smuggling;
- forged typed non-admission to trigger duplicate work;
- Dynamo metric poisoning, NaN/overflow, stale timestamps, and cardinality attack;
- cross-tenant model/profile/domain pinning;
- credential leakage in logs, errors, traces, diagnostics, and child processes;
- policy artifact tampering and rollback to an unsigned version;
- cross-domain retry after admission/commitment.

## 20. Observability

### 20.1 Metrics

Add bounded families following existing `axs_gateway_*` naming:

```text
axs_gateway_domain_state{domain_kind,state,qualification}
axs_gateway_domain_observation_age_seconds{domain_kind}
axs_gateway_domain_requests_total{domain_kind,result}
axs_gateway_domain_selection_total{policy_id,reason,domain_kind}
axs_gateway_domain_retry_total{from_kind,to_kind,reason}
axs_gateway_policy_decisions_total{policy_id,version,mode,result}
axs_gateway_policy_prediction_error{policy_id,metric}
axs_gateway_dynamo_adapter_requests_total{domain_kind,phase,result}
axs_gateway_dynamo_adapter_observation_total{domain_kind,result}
```

Do not label by request, raw domain ID, worker ID, tenant, session, prompt, free-form model, image
digest, or error message. Operator diagnostics can expose bounded IDs under authentication.

### 20.2 Tracing

Trace spans:

```text
ax.request
  ax.classify
  ax.domain.filter
  ax.domain.select
  ax.domain.reserve
  ax.dispatch.attempt
    ax.adapter.proxy
      dynamo.frontend/... (upstream trace when supported)
```

Record policy/version, domain kind, deployment class, reason enums, phase, commitment, and retry
owner. Preserve W3C context without accepting unbounded baggage.

### 20.3 Audit

Audit events cover domain registration/fencing, manifest change, policy activation/rollback,
deployment/equivalence mutation, drain, lifecycle job, credential/transport failure class, and
cross-domain retry. They use immutable generations/digests and safe bounded detail.

## 21. Deployment and packaging

### 21.1 AX Helm ownership

The AX chart owns:

- AX gateway Deployment/Services/ServiceAccount;
- config and existing-Secret references;
- PDB, topology, autoscaling, ingress/Gateway API, NetworkPolicy, and ServiceMonitor options;
- external Redis/Valkey configuration;
- optional AX adapter deployment templates only when they remain CPU-only and do not install
  runtimes.

It does not own Dynamo CRDs/operator/runtime graphs, NVIDIA GPU operator, device plugins, AX Engine,
model weights, NATS/etcd for Dynamo, or accelerator resources.

### 21.2 Network surfaces

Render distinct public and private worker-control Services. Admin/metrics remain private by default.
NetworkPolicy separately allows:

- client ingress to public API;
- adapter ingress to worker-control;
- gateway egress to adapters;
- Dynamo adapter egress to Dynamo frontend/metrics;
- gateway egress to Redis/telemetry;
- DNS and required mesh endpoints.

External Macs use an operator-provided private network/mesh. AX does not claim to build that
network.

### 21.3 Artifacts

Release gateway and adapter images as non-root AMD64/ARM64 manifests with immutable digests, SBOM,
provenance, signatures, and vulnerability results. The release manifest links the AX Helm chart and
all approved Dynamo compatibility manifests; it does not mirror Dynamo images under AX tags.

## 22. Backward compatibility and quarantine

- Existing public REST/SSE endpoints and explicit model semantics remain.
- Existing v1.0 worker fixtures remain supported for the documented window.
- Direct vLLM/SGLang registration through `ax-runtime-agent` is marked
  `compatibility_runtime_endpoint`; it cannot be used in the production Dynamo profile after
  Dynamo certification.
- Embedded macOS inference, llama.cpp/MLX backends, synchronous model mutation, and gRPC v1 remain
  behind `embedded-compat` and are not federation contracts.
- Kustomize remains an example/baseline; Helm is the supported Kubernetes contract after release
  qualification.
- Removing a compatibility path requires a migration note, replacement adapter, conformance
  evidence, and deprecation window.

## 23. Test strategy

### 23.1 Unit and property tests

- domain ID/kind/scope/qualification/manifest validation;
- v1.0/v1.1 protocol compatibility and unknown-field tolerance;
- desired/observed domain matching and ambiguous pool mapping rejection;
- PC/Thor separation and forbidden merged-domain cases;
- hard filters before score and stale/missing telemetry penalty;
- deterministic policy decision/replay and bounded record/redaction;
- identity/equivalence and cross-domain retry candidate freezing;
- retry/commitment state-machine properties;
- URL/header/credential/metric validation;
- memory/Redis semantic parity and fencing.

### 23.2 Mock integration tests

- Mac node plus PC Dynamo domain plus disabled Thor domain;
- domain starting -> ready -> degraded -> drain -> expired;
- inventory/manifest mismatch removes only affected deployment;
- typed local non-admission permits one equivalent retry;
- ambiguous Dynamo error never retries;
- SSE byte fragmentation/order, cancellation, timeouts, and body limits;
- gateway restart/partition and reservation reconciliation;
- shadow decision never changes active route; canary hashing is deterministic;
- `/readyz` 200 with no domains and `/routablez` 503 in control-plane mode.

### 23.3 Live conformance matrix

| Target | Required for initial production | Notes |
| --- | --- | --- |
| Pinned AX Engine on one Mac | Yes | API, identity, stream, cancel, drain, metrics, fault |
| Pinned Dynamo v1.2.1-compatible PC domain with one backend | Yes | Exact release/image manifest; other backends not inherited |
| Direct vLLM/SGLang runtime agent | Migration only | Does not certify target NVIDIA architecture |
| Pinned Dynamo Thor domain | Separate gate | Experimental until full Thor row passes |

Every target runs registration/readiness/inventory, logical model rewrite, blocking/streaming,
cancellation, drain, credential, generic `5xx`, malformed response, restart, and load-shed tests.

### 23.4 Fault matrix

- adapter process death/restart and stale lease;
- Dynamo frontend unreachable before and after request write;
- backend worker failure handled within Dynamo;
- Dynamo planner/operator/NATS/etcd degradation without AX state corruption;
- Mac Engine death/restart;
- PC domain loss, Thor domain loss, and simultaneous domain drain;
- AX gateway rolling upgrade/restart;
- Redis restart/partition/failover;
- policy activation race/rollback;
- client disconnect at connect, pre-header, first-byte, and mid-stream phases.

### 23.5 Performance and value tests

Retain raw artifacts for:

1. direct AX Engine versus through Mac agent/AX gateway;
2. direct Dynamo frontend versus through Dynamo adapter/AX gateway;
3. mixed-domain healthy, saturated, drain, and outage workloads;
4. PC and Thor separately under their certified envelopes;
5. 256-candidate decision latency;
6. two gateways, Redis, production concurrency, and 60-minute soak;
7. one PRD value gate: cost/load reduction, policy-correct availability, privacy/locality, or
   operator workflow improvement;
8. shadow/canary policy regret, prediction error, quality-floor violations, and rollback trigger.

## 24. CI and release evidence

Required checks include repository guidelines plus:

```text
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --tests -- -D warnings
cargo test --workspace --lib
AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test orchestration
```

Also require:

- gateway/adapter forbidden-dependency inspection;
- Linux AMD64/ARM64 and macOS portable builds;
- OCI non-root/SBOM/signature/provenance checks;
- Helm schema/render/install/upgrade/rollback matrix;
- protocol fixture compatibility;
- live runtime jobs on dedicated Mac/PC hardware;
- manual or scheduled Thor qualification;
- Redis HA and mixed-domain soak;
- release manifest with no null or placeholder evidence.

Hardware-dependent tests are not silently skipped in a release job. Missing hardware/evidence blocks
the corresponding support claim.

## 25. Implementation sequence

### D0: contract consolidation

- Land ADR-016, PRD, this specification, updated public boundary docs, and status ledger.
- Mark legacy direct CUDA and agent-session packages superseded.

### D1: execution-domain foundation

- Source foundation implemented on 2026-07-15; live certification, durable replay, rejected-candidate
  evidence, and domain-keyed HA reservations remain tracked in the status ledger.
- Add protocol v1.1 domain/decision types and fixtures.
- Add domain catalog/config resolution and state records.
- Map Mac workers explicitly and add domain-aware diagnostics/decision records.
- Preserve v1.0 behavior under compatibility mode.

### D2: Dynamo PC adapter

- Add manifest validation, readiness/inventory observer, registration, and proxy.
- Start with shadow inventory and no production dispatch.
- Enable canary dispatch after mock/security tests.
- Certify one PC Dynamo/backend/model stack and disable direct CUDA agents in production config.

### D3: HA, lifecycle, and deployment qualification

- Extend Redis state/fencing/reservations for domains.
- Add optional asynchronous Dynamo lifecycle controller.
- Finish Helm/NetworkPolicy/monitoring/runbook profiles.
- Run upgrade, rollback, credential, partition, and 60-minute soak gates.

### D4: Thor experimental integration

- Produce Thor-specific compatibility manifest and deployment profile.
- Run live functional, memory/thermal, performance, fault, and soak suites.
- Promote qualification independently; never inherit PC evidence.

### D5: adaptive policy v1

- Add bounded DecisionProfile inputs and offline replay dataset.
- Deploy one deterministic cost/SLO policy in shadow.
- Canary only after quality/value gates; add automated rollback guard.
- Keep explicit routing and prior policy as immediate rollback.

## 26. Definition of done

The final architecture is implemented only when:

- domain types and two-stage ownership are enforced in code and conformance tests;
- Mac AX Engine and PC Dynamo each pass a pinned live certification;
- Thor is either explicitly experimental/disabled or has independent live certification;
- direct NVIDIA worker routing is absent from the production profile;
- every enabled cross-domain transition has identity/equivalence evidence;
- decision records are bounded, replayable, redacted, and linked to immutable policy/observations;
- no fault test produces duplicate commitment or conflicting retry ownership;
- HA, packaging, security, performance, soak, upgrade, and rollback gates pass;
- at least one product value gate passes;
- public documentation accurately distinguishes target, implemented, certified, experimental, and
  compatibility behavior.

## 27. References

- [AX Serving PRD](../prd/PRD-AX-SERVING.md)
- [ADR-016](../adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md)
- [AX Serving node contract](../../docs/contracts/ax-serving-node-contract.md)
- [Runtime responsibility inventory](../../docs/contracts/ax-serving-runtime-responsibility-inventory.md)
- [Public contract inventory](../../docs/contracts/ax-serving-public-contract-inventory.md)
- [NVIDIA Dynamo overall architecture](https://docs.nvidia.com/dynamo/design-docs/overall-architecture)
- [NVIDIA Dynamo routing concepts](https://docs.nvidia.com/dynamo/latest/components/router/routing-concepts)
- [NVIDIA Dynamo support matrix](https://docs.nvidia.com/dynamo/latest/resources/support-matrix)
- [NVIDIA Dynamo feature matrix](https://docs.nvidia.com/dynamo/latest/resources/feature-matrix)
- [NVIDIA Dynamo release artifacts](https://docs.nvidia.com/dynamo/latest/resources/release-artifacts)
- [NVIDIA Dynamo unified backend guide](https://docs.nvidia.com/dynamo/latest/backends/custom-backend/writing-unified-backends)
- [NVIDIA Jetson AGX Thor CUDA setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html)
