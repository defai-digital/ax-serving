# ADR-014: CPU-Only OCI and Helm Deployment

| Field | Value |
| --- | --- |
| Status | Accepted |
| Decision date | 2026-07-14 |
| Owners | AX Serving maintainers |
| Scope | Release artifacts, containers, Kubernetes packaging, runtime-agent placement, and deployment lifecycle |
| Extends | [ADR-013: Runtime-neutral hybrid inference control plane](ADR-013-RUNTIME-NEUTRAL-HYBRID-INFERENCE-CONTROL-PLANE.md) |
| Related PRD | [CPU-only container deployment requirements](../prd/PRD-CPU-ONLY-CONTAINER-DEPLOYMENT.md) |
| Implementation | [CPU-only OCI and Helm deployment spec](../specs/TECH-SPEC-CPU-ONLY-OCI-AND-HELM-DEPLOYMENT.md) |

## Context

ADR-013 separates the AX Serving gateway from AX Engine, vLLM, SGLang, llama.cpp, and other
inference runtimes. The portable gateway now owns request admission, endpoint selection, fleet
state, streaming proxying, and operations without linking an accelerator runtime. A CPU-only
runtime agent translates runtime health, inventory, identity, capacity, and OpenAI-compatible data
paths into the AX Serving worker protocol.

The deployment surface has not yet caught up with that architecture. The repository contains a
multi-stage Dockerfile and a Kustomize baseline, but:

- no first-party Helm chart exists;
- release automation does not publish multi-platform images or a chart;
- the Kubernetes baseline can deadlock because gateway readiness depends on an eligible worker,
  while worker registration is sent only to ready gateway endpoints;
- the base NetworkPolicy and ClusterIP control Service do not describe external Apple Silicon
  agents or managed Redis;
- `legacy_compat` remains the example deployment mode;
- container shutdown grace is shorter than the configured request timeout;
- the runtime-agent advertised endpoint accepts only a numeric socket address, which is awkward in
  Docker and service-discovery environments;
- the runtime-agent example places a CPU-only agent beside a GPU runtime without clearly defining
  the accelerator-isolation guarantee.

The intended product distribution is now Docker/OCI and Helm. A durable decision is required so
packaging work does not reintroduce the runtime coupling removed by ADR-013.

## Decision drivers

- AX Serving must run on general-purpose CPU infrastructure.
- The gateway must never require accelerator drivers, devices, SDKs, model weights, or runtime
  process ownership.
- One artifact set must support Linux AMD64 and ARM64.
- Kubernetes operators need a versioned, configurable, testable installation surface.
- Small deployments need a workflow that does not require Kubernetes.
- Runtime owners must remain free to deploy AX Engine, vLLM, or SGLang independently.
- A chart must support external macOS AX Engine nodes and in-cluster CUDA runtimes in one fleet.
- Fresh installation, rolling update, and runtime outage behavior must be unambiguous.
- Security and supply-chain controls must be enforceable from rendered artifacts and immutable
  release metadata.

## Decision

AX Serving will use **CPU-only OCI images as its canonical executable artifacts** and a
**first-party Helm chart as its canonical production Kubernetes installation surface**. Docker
Compose will be the supported local and evaluation surface.

The core chart will deploy only the AX Serving gateway. The runtime-agent image will be published
separately for integration into runtime-owned deployments. The core chart will not install AX
Engine, vLLM, SGLang, accelerator operators, device plugins, drivers, or model weights.

### 1. Artifact boundary

The release creates these portable artifacts:

| Artifact | Purpose | Accelerator access |
| --- | --- | --- |
| `ax-serving-gateway` image | Public/admin APIs, admission, routing, fleet state, streaming proxy | Forbidden |
| `ax-runtime-agent` image | Runtime discovery, normalized health/capacity, authenticated proxy | Forbidden |
| `ax-serving` Helm chart | CPU-only gateway installation | Forbidden |
| Docker Compose example | Local/evaluation gateway, Redis, and agent wiring | Forbidden for AX Serving containers |

Embedded macOS compatibility binaries remain separately named and separately packaged. They are not
part of the CPU-only image or chart contract.

The gateway and agent images must not compile or contain AX Engine, MLX, Metal, CUDA, llama.cpp,
PyO3, embedded gRPC compatibility, model files, or accelerator-specific native libraries. CI and
SBOM policy enforce the boundary.

### 2. Helm owns the gateway, not inference runtimes

The core chart owns:

- gateway Deployment;
- public and private worker-control Services;
- ServiceAccount;
- ConfigMap references or rendering;
- Secret references;
- PodDisruptionBudget;
- optional autoscaling, ingress, Gateway API, monitoring, and NetworkPolicy resources;
- lifecycle, health, resource, and scheduling configuration.

The chart does not own:

- runtime Deployments or StatefulSets;
- model download or storage;
- accelerator resources, operators, drivers, or runtime classes;
- runtime-specific scheduling arguments;
- runtime rollout and model-load semantics;
- semantic equivalence certification.

Runtime-agent examples and optional integration documentation may show sidecar and separate-
Deployment patterns. They do not make runtime resources part of the core chart release.

### 3. CPU-only guarantee

The gateway is scheduled on a general-purpose CPU node pool. The chart has no accelerator resource,
device mount, GPU toleration, or accelerator runtime-class default.

A runtime agent remains CPU-only even when it is a sidecar in a Pod whose runtime container requests
a GPU. The agent container must not request a GPU or receive accelerator device configuration.
Where organizational policy forbids any AX Serving process from running on a GPU node, the agent is
deployed separately on CPU nodes and connects to the runtime through a private Service.

### 4. Readiness and routability are separate states

Gateway process readiness and downstream runtime capacity are different conditions.

- `/livez` means the process can make progress.
- `/readyz` means configuration is accepted, listeners are operational, and required control-plane
  dependencies such as the fleet store are usable.
- `/routablez` means at least one runtime deployment is eligible for the requested or configured
  routing scope.

The gateway remains ready when a runtime pool is empty or unavailable. Inference requests receive a
structured `503` with `Retry-After`, and routability metrics and alerts identify the outage. This
prevents downstream runtime failure from removing the gateway control plane or blocking runtime
registration.

The private worker-control Service keeps `publishNotReadyAddresses: false`. Decoupling readiness
from runtime routability breaks the bootstrap cycle without forcing Kubernetes to advertise
starting or draining Pods as ready endpoints. Runtime agents retry registration with bounded
exponential backoff and jitter and reconnect to another ready gateway during rollout.

A future headless peer-discovery Service may use different endpoint publication semantics, but it
must not be reused for worker-control traffic without a separate decision and termination review.

### 5. Public and private network surfaces are separate

The chart renders distinct Services:

1. A public API Service, normally ClusterIP behind an ingress or Gateway.
2. A private worker-control Service for registration and heartbeat.

Admin and metrics paths are private by default. Until the application provides a dedicated admin
listener, the public ingress or Gateway must exclude those paths.

The NetworkPolicy model supports:

- namespace and Pod selectors for in-cluster agents;
- explicit CIDRs for external runtime agents;
- explicit CIDRs or selectors for Redis/Valkey;
- DNS egress;
- optional telemetry egress;
- no unrestricted control-plane ingress by default.

External agents use a private load balancer, private Gateway, service mesh spanning the relevant
network, or an operator-supplied private overlay. AX Serving does not claim to create that network.

### 6. Runtime-agent endpoints are service-discovery friendly

The runtime agent will advertise a validated URI that may contain an IP address or DNS hostname.
The scheme remains constrained by the configured transport profile. Every gateway replica must be
able to resolve and reach an advertised endpoint before it becomes eligible.

This replaces the assumption that every deployment can inject a stable numeric `SocketAddr`.

### 7. Lifecycle is bounded and stream-aware

On SIGTERM, a gateway:

1. marks itself draining and fails readiness;
2. stops new admission;
3. permits accepted blocking requests and streams to finish;
4. waits until inflight work reaches zero or the configured drain deadline;
5. cancels remaining upstream work at the drain deadline;
6. releases reservations and exits no later than the hard shutdown deadline.

Kubernetes termination grace must exceed the hard application shutdown deadline. The hard deadline
must exceed the endpoint-propagation and drain windows combined. Chart schema or template validation
rejects an invalid relationship.

### 8. Helm is the production contract; Compose is not HA

The Helm chart is versioned, schema-validated, tested, and published as an OCI artifact. It supports
immutable image digests, existing Secrets, external Redis, HA replicas, topology, disruption,
autoscaling, monitoring, ingress, and network policy.

Docker Compose provides one reproducible evaluation topology. It may include Redis and mock or
external agent wiring, but it does not claim production HA, durable storage, or cluster security.

Kustomize remains a checked integration baseline or generated example. It does not evolve as a
second independently supported configuration API.

### 9. Release artifacts are immutable and linked

One release version links:

- source revision;
- AMD64 and ARM64 image manifests and digests;
- chart package and digest;
- SBOMs;
- provenance attestations;
- signatures;
- vulnerability results;
- compatibility and certification status.

Release workflows may publish semantic tags for usability, but installation and rollback guidance
uses immutable digests.

## Detailed consequences

### Positive consequences

- AX Serving has a clear, testable CPU-only claim.
- Gateway releases are independent of AX Engine and CUDA runtime releases.
- The same gateway images run on ordinary AMD64 and ARM64 Linux nodes.
- Operators receive one first-party, versioned Kubernetes configuration contract.
- Runtime owners retain control of accelerator lifecycle and tuning.
- External Mac and in-cluster CUDA nodes can participate in one explicit fleet.
- Runtime outages no longer remove the control plane from service discovery.
- Image and chart rollback can be reproduced from immutable evidence.
- The chart can satisfy restricted Pod security without accelerator exceptions.

### Negative consequences

- Runtime integration requires a separately deployed agent and routable private network.
- Operators must provide Redis/Valkey for active-active gateways.
- The project must maintain chart compatibility, values schema, release automation, and cluster
  tests in addition to Rust code.
- Sidecar and separate-agent deployment patterns require explicit documentation.
- Existing readiness tests and monitoring assumptions must change.
- Native TLS is still not created by the chart; trusted transport remains a platform prerequisite.
- Kustomize users must migrate to Helm values or accept the example-only support level.

### Neutral consequences

- AX Serving can still run on macOS for development or compatibility, but the production gateway
  contract is portable Linux containers.
- A runtime-agent sidecar may be located on an accelerator node while remaining CPU-only at the
  container-resource and dependency boundary.
- The chart can expose optional platform integrations, but no optional integration becomes a hard
  dependency of the gateway.

## Alternatives considered

### A. Keep Kustomize as the only Kubernetes surface

Rejected. Kustomize is useful for environment overlays, but the product needs a versioned values
contract, schema validation, packaged releases, dependency-free installation, reusable templates,
and chart tests. Maintaining raw manifests alone would push too much composition work to every
operator.

### B. Publish Docker images but no chart

Rejected. Images solve executable distribution but not HA replicas, Services, network policy,
secret references, probes, lifecycle, topology, monitoring, or safe upgrade defaults.

### C. Put gateway, agents, runtimes, Redis, and GPU operators in one chart

Rejected. This would make the gateway release cadence depend on accelerator vendors and runtime
projects, blur security ownership, enlarge the failure domain, and contradict ADR-013.

### D. Build one image containing gateway and embedded runtimes

Rejected. A universal image would include platform-specific native dependencies, fail the CPU-only
guarantee, increase attack surface and image size, and make Linux gateway releases depend on macOS
or CUDA runtime code.

### E. Make a Kubernetes operator the first production surface

Rejected for the initial release. An operator could later manage deployment custom resources, but
it introduces CRDs, reconciliation semantics, RBAC, upgrade compatibility, and a larger operational
surface before the basic image and Helm contract is proven.

### F. Keep worker-dependent Kubernetes readiness

Rejected. It couples control-plane discovery to downstream capacity, can block fresh registration,
causes Helm `--wait` to fail when runtimes are intentionally deployed later, and can amplify a
runtime outage into a gateway outage. Routability remains observable and enforceable separately.

### G. Require all runtime agents to run as sidecars

Rejected. Sidecars are effective for in-cluster CUDA runtimes but do not cover native macOS AX
Engine nodes and may violate organizational policies that prohibit AX Serving processes on GPU
nodes. Both sidecar and remote-agent patterns are supported.

### H. Require all agents to run on separate CPU nodes

Rejected. This adds a network hop and weakens lifecycle coupling for every in-cluster runtime. The
CPU-only guarantee is defined per container and dependency boundary, while a stricter node-
separation profile remains available.

## Security implications

- Core images have a smaller dependency and device-access surface than embedded runtime artifacts.
- Existing Secrets prevent credentials from becoming chart defaults or release metadata.
- Separate Services and ingress path policy reduce accidental control/admin exposure.
- External-agent support increases network-boundary complexity and therefore requires explicit
  transport, CIDR, and authentication configuration.
- Public and worker-control Services exclude unready endpoints. Readiness remains independent of
  runtime availability so this does not recreate the registration bootstrap cycle.
- Generic value escape hatches remain constrained by safe defaults, schema validation, and rendered
  policy tests.
- Signatures and provenance identify artifact origin; they do not replace vulnerability policy or
  runtime conformance tests.

## Migration plan

1. Add `/routablez`, redefine `/readyz`, and preserve compatibility aliases in diagnostics.
2. Set private control-service bootstrap behavior and add a fresh-install registration test.
3. Add graceful gateway drain and validate termination deadlines.
4. Change runtime-agent advertised endpoint configuration from `SocketAddr` to validated URI.
5. Harden and publish multi-platform gateway and agent images.
6. Add the Compose evaluation stack.
7. Implement the first-party chart and values schema.
8. Reconcile or generate the Kustomize example from the chart contract.
9. Add OCI chart/image release, SBOM, provenance, signing, and verification.
10. Complete live external-Mac, in-cluster-CUDA, HA, upgrade, partition, and soak certification.

During migration, the old `/readyz` behavior may be available behind an explicitly named
compatibility setting, but it must not remain the production chart default.

## Acceptance criteria

The decision is implemented when:

- gateway and agent images are multi-platform and pass the forbidden-dependency policy;
- the core chart renders no accelerator resource or dependency;
- a fresh chart install becomes ready before a runtime registers;
- a runtime can register through the private control Service during bootstrap;
- routability transitions independently from process readiness;
- external Redis and external runtime-agent CIDRs are representable in NetworkPolicy values;
- graceful replacement preserves accepted streams within the declared deadline;
- chart defaults use explicit deployment mode and existing Secret references;
- Compose and Helm installation paths pass their documented smoke tests;
- image and chart digests, SBOMs, provenance, signatures, and source revision verify together;
- production certification evidence satisfies the related PRD release gates.

## Follow-up decisions

The following may require separate ADRs after the base deployment is proven:

- adopting Gateway API as the preferred public ingress integration;
- introducing a dedicated admin listener;
- introducing an AX Serving Kubernetes operator and CRDs;
- standardizing SPIFFE/SPIRE identities across external runtime agents;
- selecting a default custom-metric adapter or event-driven autoscaler;
- deprecating the hand-maintained Kustomize baseline completely.

## References

- [Kubernetes EndpointSlices](https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/)
- [Kubernetes Pod lifecycle and termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Kubernetes NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Helm chart best practices](https://docs.helm.sh/docs/chart_best_practices/)
- [Helm OCI registries](https://docs.helm.sh/docs/topics/registries/)
- [Helm 4 overview](https://docs.helm.sh/docs/overview/)
