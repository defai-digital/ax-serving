# ADR-013: Runtime-Neutral Hybrid Inference Control Plane

| Field | Value |
| --- | --- |
| Status | Accepted |
| Decision date | 2026-07-12 |
| Owners | AX Serving maintainers |
| Scope | Gateway, runtime agents, worker protocol, and embedded compatibility path |
| Supersedes | Any design assumption that the API gateway must link an inference runtime SDK |
| Related PRD | [AX Serving product requirements](../prd/PRD-AX-SERVING.md) |
| Implementation | [Hybrid runtime control-plane spec](../specs/TECH-SPEC-HYBRID-RUNTIME-CONTROL-PLANE.md) |
| Deployment extension | [ADR-014: CPU-only OCI and Helm deployment](ADR-014-CPU-ONLY-OCI-AND-HELM-DEPLOYMENT.md) |

## Context

AX Serving must expose a stable API over a fleet that includes AX Engine on Apple Silicon and
vLLM or SGLang on CUDA. These runtimes do not share an SDK, platform, model format, threading
model, scheduling implementation, or release cadence.

The repository's intended multi-worker direction is already close to the right abstraction: the
gateway registers workers, receives heartbeats, chooses endpoints, and proxies requests through a
runtime agent. The remaining embedded path creates several structural problems:

- `ax-serving-api` depends on `ax-serving-engine`, so a gateway that should hold no weights inherits
  the platform and build requirements of a model runtime.
- `ax-serving-engine` pins an old AX Engine SDK release. The current AX Engine request and session
  contracts have changed, so upgrading is not a tag-only change.
- An embedded `EngineSession` is shared behind a mutex and generation is run on per-request OS
  threads. Current AX Engine MLX sessions require construction, execution, and destruction on one
  dedicated owner thread.
- The embedded adapter duplicates chat-template and request-feature logic that belongs to the
  runtime. It cannot reliably keep pace with AX Engine model, multimodal, tool, structured-output,
  and speculative-decoding support.
- The current runtime agent can report its own process as healthy while its upstream runtime is
  unavailable, and a failed inventory refresh may leave stale models eligible.
- The gateway can retry broad failures without a protocol-level distinction between “not admitted”
  and “accepted but failed,” which is unsafe for generated and streamed responses.

Modern inference gateways make endpoint decisions from runtime readiness, capabilities, queue and
cache pressure, while leaving token scheduling and distributed execution inside the runtime. AX
Serving needs the same separation without requiring Kubernetes or adopting a CUDA-specific runtime
as its control plane.

## Decision drivers

- AX Engine must be able to evolve quickly without forcing lockstep gateway releases.
- The gateway must run on macOS and Linux without MLX, Metal, CUDA, Python, or model weights.
- AX Engine and CUDA runtimes must participate through the same control-plane abstraction.
- Runtime-specific scheduling, cache, and model semantics must not be reimplemented in the gateway.
- Routing and failover must be safe when two deployments are not semantically equivalent.
- Streaming retries and cancellation must have clear commitment semantics.
- The architecture must support a laptop-scale deployment and an HA fleet without two different
  products.

## Decision

AX Serving will be a **runtime-neutral inference gateway and endpoint picker**. Runtime integration
will occur through a versioned wire protocol implemented by thin runtime agents. The gateway will
not link AX Engine, vLLM, SGLang, llama.cpp, or another inference-runtime SDK.

### Ownership boundary

| AX Serving owns | Runtime and runtime agent own |
| --- | --- |
| Public API and authentication | Model process lifecycle and execution readiness |
| Logical model and deployment catalog | Model loading and artifact validation |
| Tenant admission and policy | Tokenizer and chat-template semantics |
| Worker leases and drain state | Continuous batching and sequence scheduling |
| Capability filtering and endpoint selection | KV and prefix-cache allocation |
| Request attempt policy and safe failover | Speculative decoding and MTP |
| SSE proxying, deadlines, and cancellation | Tensor, pipeline, data, and expert parallelism |
| Fleet-level metrics, traces, and audit | Hardware kernels and runtime-specific metrics |

The agent is deliberately thin. It performs runtime discovery, readiness normalization,
capability and metric translation, authentication, cancellation, and byte-preserving proxying. It
does not tokenize, render templates, batch sequences, own a second KV cache, or schedule tokens.

### Protocol boundary

A new cross-platform `ax-serving-protocol` crate will define serialization-only types for:

- registration and protocol negotiation;
- heartbeat, lease, runtime readiness, and observed model inventory;
- model deployment identity and equivalence metadata;
- normalized capacity and telemetry;
- typed pre-admission and dispatch outcomes;
- drain and lifecycle job state.

The wire protocol has its own semantic version. The gateway accepts compatible protocol versions
and negotiates optional features. It does not branch on a runtime name to infer a capability.

### Data path

Direct HTTP with streaming SSE remains the default request path:

```text
client -> AX Serving gateway -> runtime agent -> runtime
```

This path is simple, supports backpressure and cancellation, and avoids placing token streams in a
durable message system. NATS or another broker may be used for asynchronous deployment jobs,
events, or workflow integration, but not as the primary inference stream.

Each dispatch attempt executes entirely on one runtime endpoint. AX Serving does not split prefill,
decode, model layers, or KV state between an MLX runtime and a CUDA runtime, and it does not migrate
an admitted request mid-stream.

### Fleet model

The fleet is heterogeneous, but each endpoint pool is homogeneous enough to make routing safe. A
pool identifies one deployment class: runtime family, artifact or revision, tokenizer, template,
quantization, hardware class, trust domain, and supported operations.

Clients use a logical model alias. The alias resolves to one or more deployment pools. Cross-pool
failover is permitted only when the deployments share an explicit, operator-approved equivalence
class. Missing identity is not treated as equivalence.

This follows the endpoint-pool pattern used by inference-aware gateways while retaining an AX
Serving-native protocol and deployment model.

### Routing boundary

AX Serving performs request-level endpoint selection in two stages:

1. Apply hard filters for readiness, lease freshness, drain state, identity, operation, capability,
   limits, trust policy, and admission capacity.
2. Score eligible endpoints from normalized signals such as queue depth, active-to-capacity ratio,
   KV pressure, batch headroom, TTFT EWMA, recent failures, locality, and cache affinity.

Missing or stale telemetry is penalized and never interpreted as an idle endpoint. Once selected,
the runtime controls batching and token-level scheduling.

### Retry and stream commitment

Every public request receives one stable request ID. Each dispatch receives a distinct attempt ID.
The gateway may retry at most once only when:

- connection establishment fails before the runtime can admit the request; or
- the agent returns a typed, authenticated pre-admission rejection; and
- no response headers or bytes have been committed to the client.

The gateway will not retry arbitrary `5xx` responses or any stream after its first committed byte.
Client disconnect and deadlines are propagated through the agent to the runtime.

### Readiness and leases

Agent process liveness and runtime readiness are separate signals. A worker is eligible only when
the most recent authoritative runtime observation is ready and its lease is fresh. Failed runtime
discovery makes the worker unavailable; stale model inventory may be retained for diagnostics but
not routing.

The worker state model is:

```text
Discovered -> RegisteredNotReady -> Ready -> Draining -> Unavailable/Expired
```

Recovery from `Unavailable` or `Expired` requires a fresh compatible registration or heartbeat.

### Trust boundaries

AX Serving terminates public client authentication. Worker-control identity and runtime credentials
are independent. The agent denies public `Authorization`, cookie, proxy credential, and hop-by-hop
header forwarding by default. Non-loopback links use TLS, with mTLS preferred for worker control.

### Packaging boundary

- `ax-serving-api` becomes a cross-platform control-plane crate with no engine dependency.
- `ax-serving-protocol` contains cross-platform wire types and no runtime dependency.
- `ax-runtime-agent` is the preferred portable worker integration; the legacy `ax-thor-agent`
  binary name may remain temporarily as an alias.
- Embedded inference moves behind a macOS-only compatibility crate or feature and is not linked
  into the gateway binary.
- The existing `ax.serving.v1` gRPC service remains with the embedded compatibility product. Its
  local model paths, Metal/CPU backend enum, and token-ID stream are not runtime-neutral and must not
  be translated lossily into the hybrid gateway. Any portable gRPC successor is a new v2 contract.
- A native AX Engine integration uses its supported HTTP/server contract through the agent. If a
  future embedding use case requires direct SDK access, the session must live on a dedicated owner
  thread inside a runtime-specific compatibility process, not inside the gateway.
- CPU-only OCI images, the first-party Helm chart, runtime-agent placement, probe semantics, and
  container lifecycle are governed by ADR-014. Those artifacts must preserve this runtime-neutral
  dependency boundary and must not request accelerator resources.

## Consequences

### Positive

- AX Engine can add models and change SDK internals without a gateway rebuild when the wire
  contract remains compatible.
- macOS and Linux workers participate through one lifecycle and routing model.
- Gateway deployment is smaller and no longer inherits runtime platform or Python constraints.
- Runtime authors keep control of the execution features they can optimize best.
- Readiness, failover, and retry behavior become explicit and conformance-testable.
- Operators can compare runtime performance separately from gateway overhead and mixed-fleet SLOs.
- The same design can later integrate with Kubernetes endpoint discovery without making
  Kubernetes mandatory.

### Negative

- The runtime agent and protocol become compatibility surfaces that require versioning and tests.
- An additional local or network hop can add latency and operational configuration.
- Existing gRPC v1 clients remain tied to the compatibility product until a justified v2 exists.
- Runtime metrics require maintained adapters because names and semantics vary by version.
- Safe equivalence requires artifact metadata and testing that a simple model-name registry does
  not provide.
- HA requires shared lease and fleet state, introducing an external state-store option.

### Neutral trade-offs

- A thin agent duplicates some HTTP plumbing, but avoids duplicating inference semantics.
- Routing can improve fleet-level utilization, but cannot compensate for a poorly configured
  runtime scheduler.
- Exact output equivalence across runtimes is not guaranteed; the architecture makes the policy
  visible instead of hiding the uncertainty.

## Alternatives considered

### A. Embed AX Engine and CUDA runtimes in AX Serving

Rejected. It would turn AX Serving into a multi-runtime host, inherit incompatible platform and
threading constraints, and force it to duplicate model lifecycle and execution behavior.

### B. Link the AX Engine SDK while proxying vLLM over HTTP

Rejected as the target architecture. It creates asymmetric ownership, prevents a portable gateway,
and couples AX Serving to AX Engine's fast-moving SDK and MLX thread-affinity contract. The current
embedded path may remain temporarily as compatibility-only code.

### C. Use one runtime agent and versioned wire protocol for every runtime

Accepted. It produces one symmetric control-plane contract while leaving runtime-specific work in
adapter modules.

### D. Adopt the Kubernetes Gateway API Inference Extension as the product API

Deferred. Its inference-pool and endpoint-picker patterns are valuable and should inform AX
Serving, but requiring Kubernetes would exclude the local and small private deployments AX Serving
must support. A future integration can map AX deployment pools to Kubernetes resources.

### E. Use vLLM and Ray as the universal serving layer

Rejected. This fits CUDA clusters but does not provide the intended native MLX execution path on
Apple Silicon. It would also make a runtime-specific distributed stack the fleet control plane.

### F. Route every inference request through NATS

Rejected for the primary data path. Durable brokers are useful for jobs and events but complicate
stream backpressure, first-byte latency, cancellation, and response commitment.

## Migration strategy

1. Introduce `ax-serving-protocol` without changing the existing v1 endpoints.
2. Add protocol version, runtime readiness, identity, and safe header fields to agent registration
   and heartbeat using backward-compatible optional decoding.
3. Remove `ax-serving-engine` from the API gateway's dependency graph and CI-build the gateway on
   macOS and Linux.
4. Place embedded backends behind an explicit compatibility feature and binary.
5. Add deployment pools, equivalence classes, request profiles, and routing v2 behind a feature or
   configuration gate.
6. Add typed admission outcomes and commitment-aware dispatch, then make it the default.
7. Add shared fleet state for HA deployments.
8. Deprecate synchronous fleet model mutation and introduce asynchronous deployment jobs.
9. Remove the embedded compatibility path only after a published deprecation period and equivalent
   local-agent workflow exist.

## Compliance checks

Future changes comply with this ADR only if all answers below are yes:

- Can the gateway build and run without a model runtime SDK or accelerator library?
- Does the runtime remain authoritative for tokenization, templates, batching, and caches?
- Is every new routing input carried by a versioned protocol field with freshness semantics?
- Does missing identity or telemetry fail closed or receive a conservative penalty?
- Can a request be proven not to retry after response commitment?
- Are public, worker-control, and runtime credentials still separate?
- Can both an AX Engine deployment and a CUDA deployment implement the feature through the same
  control-plane abstraction?

If any answer is no, the change requires a new ADR or an amendment to this one.

## References

- [AX Serving product requirements](../prd/PRD-AX-SERVING.md)
- [Hybrid runtime control-plane technical specification](../specs/TECH-SPEC-HYBRID-RUNTIME-CONTROL-PLANE.md)
- [ADR-014: CPU-only OCI and Helm deployment](ADR-014-CPU-ONLY-OCI-AND-HELM-DEPLOYMENT.md)
- [CPU-only OCI and Helm technical specification](../specs/TECH-SPEC-CPU-ONLY-OCI-AND-HELM-DEPLOYMENT.md)
- [AX Serving runtime responsibility inventory](../../docs/contracts/ax-serving-runtime-responsibility-inventory.md)
- [Kubernetes Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension)
- [InferencePool API](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
