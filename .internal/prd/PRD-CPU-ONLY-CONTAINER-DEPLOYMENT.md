# Product Requirements: CPU-Only Container Deployment

| Field | Value |
| --- | --- |
| Status | Canonical deployment target; implementation pending |
| Owner | AX Serving maintainers |
| Last updated | 2026-07-14 |
| Applies to | AX Serving 3.x deployment and release surfaces |
| Parent product requirements | [AX Serving hybrid inference control plane](PRD-AX-SERVING.md) |
| Architecture decision | [ADR-014](../adr/ADR-014-CPU-ONLY-OCI-AND-HELM-DEPLOYMENT.md) |
| Technical specification | [CPU-only OCI and Helm deployment spec](../specs/TECH-SPEC-CPU-ONLY-OCI-AND-HELM-DEPLOYMENT.md) |

## 1. Executive summary

AX Serving will be distributed and operated as a CPU-only inference gateway and control plane.
The supported deployment artifacts are Open Container Initiative images and a first-party Helm
chart. Docker Compose is the supported local and evaluation workflow; Helm is the supported
production Kubernetes workflow.

AX Serving containers do not execute model kernels and do not require access to Metal, MLX,
CUDA, NVIDIA devices, model weights, or an embedded inference-runtime SDK. AX Engine remains the
Apple Silicon and MLX inference runtime. vLLM, SGLang, or another certified runtime remains the
CUDA inference runtime. CPU-only runtime agents normalize those runtimes into AX Serving's
versioned worker protocol.

The product boundary is therefore:

```text
clients -> CPU-only AX Serving gateways -> CPU-only runtime agents -> inference runtimes
                                              |                         |
                                              |                         +-- AX Engine / MLX
                                              +---------------------------- vLLM / SGLang / CUDA
```

The gateway may run on ordinary Linux CPU nodes. A runtime agent may run on an ordinary CPU node
or beside a runtime on an accelerator node, but it must not request, receive, initialize, or use an
accelerator device. The core Helm chart does not install or manage GPU runtimes.

## 2. Problem statement

The repository has a portable gateway and runtime-agent implementation, a multi-stage Dockerfile,
and a Kustomize integration baseline. These are useful foundations but do not yet constitute a
supported container product:

1. There is no first-party Helm chart, values schema, chart test, or chart release artifact.
2. Container images are built in CI but are not published as signed multi-platform release images.
3. The current Kubernetes readiness probe depends on an eligible runtime, while runtime agents
   register through a Service that normally publishes only ready gateway endpoints. A fresh
   installation can therefore deadlock.
4. The base NetworkPolicy and ClusterIP control Service do not model managed Redis or external
   Apple Silicon runtime nodes.
5. The Kubernetes baseline defaults to `legacy_compat`, although production hybrid routing requires
   explicit pools, deployments, identities, and equivalence policy.
6. Gateway termination grace is shorter than the configured request timeout and can truncate long
   inference streams during an upgrade.
7. Docker users do not have a Compose example, an image health check, or a stable way to advertise
   a runtime-agent DNS address.
8. The release workflow does not bind image digests, chart versions, SBOMs, provenance, signatures,
   and source revisions into one verifiable release record.

Without a clear deployment contract, operators must invent security, networking, rollout, and
resource behavior independently. This increases the likelihood of exposing a control endpoint,
misrouting a model, scheduling the gateway on expensive GPU nodes, or losing active streams during
an upgrade.

## 3. Product decision

AX Serving adopts the following deployment hierarchy:

1. **OCI images are the canonical executable artifacts.**
2. **Helm is the canonical production Kubernetes installation surface.**
3. **Docker Compose is the canonical local and evaluation surface.**
4. **Kustomize manifests remain examples and migration aids, not an independently versioned
   product surface.**
5. **The core chart installs only the CPU-only AX Serving gateway.**
6. **The runtime-agent image is published separately and integrated into runtime deployments by
   runtime owners.**
7. **Inference runtimes and accelerator operators are not dependencies of the core chart.**

ADR-014 records the decision and alternatives. The technical specification defines the artifact,
chart, networking, lifecycle, security, and verification contracts.

## 4. Product principles

### 4.1 CPU-only means enforceable isolation

CPU-only is not merely a documentation claim. The gateway and runtime-agent release artifacts must:

- contain no AX Engine, MLX, Metal, CUDA, llama.cpp, PyO3, or embedded-runtime dependency;
- contain no model weights or model download behavior;
- request no Kubernetes extended accelerator resource;
- mount no accelerator device, driver socket, or runtime library path;
- require no GPU-specific node label, runtime class, toleration, or environment variable;
- run on both `linux/amd64` and `linux/arm64` general-purpose nodes;
- pass an automated dependency-boundary and rendered-manifest policy test.

### 4.2 Runtime ownership remains external

AX Serving owns public APIs, authentication, admission, routing, fleet state, worker leases,
streaming proxying, audit, and deployment desired state. Runtime owners continue to own image
rollout, model download, model loading, accelerator allocation, batching, KV cache, tokenization,
templates, and inference execution.

### 4.3 Installation must not depend on runtime availability

A gateway Pod can be operational even when no runtime is eligible. Kubernetes process readiness
must not be coupled to downstream model capacity. Runtime routability is exposed separately and
used for alerts, diagnostics, and optional load-balancer policy.

### 4.4 Production defaults fail closed

The production chart defaults to explicit deployment identity, authenticated public/admin/control
and dispatch channels, external durable fleet state for multiple replicas, non-root containers,
read-only filesystems, least-privilege networking, and immutable image selection.

### 4.5 One release has one evidence chain

Images, chart, source, SBOMs, provenance, signatures, and release notes must resolve to the same
version and source revision. A mutable tag alone is not release evidence.

## 5. Users and primary workflows

### 5.1 Platform operator

The platform operator installs two or more CPU-only gateways, supplies Redis/Valkey and secrets,
connects runtime pools, configures ingress and network policy, monitors routing health, and performs
rolling upgrades without losing accepted requests.

### 5.2 Runtime owner

The runtime owner deploys AX Engine, vLLM, or SGLang, adds the CPU-only runtime agent, supplies
runtime identity and capability metadata, exposes a private dispatch address, and verifies
registration and drain behavior.

### 5.3 Application developer

The application developer runs a Compose stack or a single gateway container, points it at one or
more runtime agents, and uses one OpenAI-compatible endpoint without installing an accelerator SDK
inside AX Serving.

### 5.4 Security and release engineer

The security or release engineer verifies signed image and chart digests, SBOMs, provenance,
vulnerability results, non-root configuration, secret references, and the absence of accelerator
dependencies in the core artifacts.

## 6. Goals

### 6.1 P0 goals

- Publish CPU-only gateway and runtime-agent images for Linux AMD64 and ARM64.
- Provide a first-party Helm chart for an HA gateway deployment.
- Provide a Docker Compose example for local and evaluation use.
- Prevent fresh-install registration deadlock.
- Support in-cluster and externally hosted runtime agents, including native macOS AX Engine nodes.
- Support an external managed Redis or Valkey service without embedding credentials in chart values.
- Preserve active blocking and streaming requests during planned gateway replacement up to a
  configured drain deadline.
- Keep accelerator runtimes outside the core chart and image dependency graph.
- Publish immutable, signed, source-linked release artifacts with SBOM and provenance.
- Make default production configuration explicit, authenticated, and fail closed.

### 6.2 P1 goals

- Provide optional Ingress, Gateway API, ServiceMonitor, HPA, PDB, and NetworkPolicy
  resources.
- Support CPU-node placement, topology spreading, affinity, tolerations, priority classes, and
  scheduling constraints through values.
- Support external secret controllers through `existingSecret` references without requiring one
  specific secrets implementation.
- Provide chart tests and a kind-based installation smoke test.
- Support custom-metric autoscaling from queue depth, active requests, or request rate.
- Retain Kustomize as a rendered example generated from or checked against the Helm contract.

## 7. Non-goals

- Installing AX Engine, vLLM, SGLang, NVIDIA GPU Operator, device plugins, drivers, model servers,
  or model weights from the core Helm chart.
- Scheduling or partitioning accelerator resources.
- Running MLX or Metal inside Linux containers.
- Splitting one inference request, token stream, model, or KV cache across runtime nodes.
- Replacing a service mesh, private overlay network, ingress controller, certificate manager,
  external secrets controller, Prometheus operator, or managed Redis service.
- Providing an AX Serving Kubernetes operator in the first Helm release.
- Treating Docker Compose as an HA or production certification surface.
- Automatically declaring two runtime artifacts equivalent.

## 8. Supported deployment modes

### 8.1 Docker evaluation mode

The operator runs one gateway container, one Redis/Valkey container when shared state is desired,
and one or more runtime agents. Runtime processes may run on the host, in another container, or on
a reachable remote host.

This mode optimizes for clarity and reproducibility. It is not an HA claim.

### 8.2 Helm production mode

The operator installs at least two gateway replicas on CPU nodes, points them at external durable
Redis/Valkey, supplies existing secrets, exposes a public API through a trusted ingress, and exposes
the control channel only to trusted runtime agents.

Runtime agents may be:

- sidecars beside in-cluster runtime containers;
- separate CPU Deployments that point at runtime Services;
- native processes or containers on external Apple Silicon hosts;
- containers on external CUDA hosts.

### 8.3 Compatibility mode

Embedded macOS inference remains a separately named compatibility artifact. It is not installed by
the CPU-only chart and is not included in the gateway or runtime-agent images.

## 9. Functional requirements

### 9.1 Artifact and CPU-isolation requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| DEP-001 | P0 | Publish separate `ax-serving-gateway` and `ax-runtime-agent` OCI images. |
| DEP-002 | P0 | Each image must provide `linux/amd64` and `linux/arm64` variants under one manifest digest. |
| DEP-003 | P0 | Gateway and agent images must be built only from portable features and must not compile or contain embedded-runtime dependencies. |
| DEP-004 | P0 | Gateway and agent containers must run as a fixed non-root UID/GID with no added Linux capabilities. |
| DEP-005 | P0 | Core manifests must request only CPU, memory, storage, and ordinary network resources. |
| DEP-006 | P0 | CI must reject `nvidia.com/gpu`, Metal/MLX/CUDA devices, accelerator runtime classes, or GPU tolerations in rendered core-chart resources. |
| DEP-007 | P0 | Runtime-agent resources must be independently configurable and must never include a GPU request. |
| DEP-008 | P0 | Embedded compatibility binaries must be released separately and clearly excluded from the CPU-only deployment guarantee. |
| DEP-009 | P1 | Images should minimize packages, expose OCI source/version labels, and support a read-only root filesystem. |

### 9.2 Docker requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| DEP-010 | P0 | The Dockerfile must provide named `gateway` and `agent` targets and use `Cargo.lock`. |
| DEP-011 | P0 | Publish a Compose example with gateway, Redis/Valkey, secrets by environment or file, and documented external-runtime connectivity. |
| DEP-012 | P0 | Images must provide or document an executable liveness probe that does not require public or admin credentials. |
| DEP-013 | P0 | Runtime agents must accept a routable advertised URI or DNS hostname, not only a numeric IP socket address. |
| DEP-014 | P0 | The Compose example must work with runtime agents on the Compose network and with an AX Engine server on a macOS host. |
| DEP-015 | P1 | Build stages should use reproducible base-image digests and BuildKit cache mounts or equivalent dependency caching. |
| DEP-016 | P1 | Docker documentation must include `--read-only`, capability drop, resource limit, secret, log, and shutdown guidance. |

### 9.3 Helm chart requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| DEP-020 | P0 | Publish a Helm API v2 chart named `ax-serving` with semantic chart and application versions, validated against the supported Helm 3 and Helm 4 client matrix. |
| DEP-021 | P0 | The core chart must deploy only the gateway; it must not install an inference runtime or accelerator operator. |
| DEP-022 | P0 | The chart must include `values.yaml`, `values.schema.json`, templates, notes, tests, and an operator README. |
| DEP-023 | P0 | Production credentials must be supplied through an existing Secret; plaintext secret values must not have functional defaults. |
| DEP-024 | P0 | Production values must default to `deployment_mode: explicit`; `legacy_compat` must be an explicit opt-in. |
| DEP-025 | P0 | Public and worker-control Services must be separate and independently configurable. |
| DEP-026 | P0 | The control Service must expose every control-plane-ready gateway independent of runtime routability, while excluding starting and draining gateways. |
| DEP-027 | P0 | NetworkPolicy values must support in-cluster selectors and external CIDR blocks for Redis and runtime agents. |
| DEP-028 | P0 | The chart must support a private control-plane Service or Gateway for external macOS and CUDA agents without exposing it publicly by default. |
| DEP-029 | P0 | Multiple gateway replicas must require Redis/Valkey shared state and unique gateway identities. |
| DEP-030 | P0 | Values must expose resources, node selector, affinity, tolerations, topology spread, priority class, and runtime class without accelerator defaults. |
| DEP-031 | P0 | ConfigMap and Secret identity changes must trigger a controlled Pod rollout. |
| DEP-032 | P0 | Rolling updates must preserve availability with `maxUnavailable: 0` by default when replica count is greater than one. |
| DEP-033 | P1 | Optional HPA, PDB, ServiceMonitor, ingress, Gateway API, and NetworkPolicy resources must be independently enabled. |
| DEP-034 | P1 | The chart must support image selection by immutable digest and warn or fail in production mode when only a mutable tag is supplied. |
| DEP-035 | P1 | The chart must expose extra labels, annotations, environment variables, volumes, volume mounts, and image pull secrets without requiring a fork. |

### 9.4 Health, readiness, and lifecycle requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| DEP-040 | P0 | `/livez` must report only whether the gateway process can make progress. |
| DEP-041 | P0 | `/readyz` must report gateway configuration, listeners, and required fleet-store readiness; it must not require an eligible runtime. |
| DEP-042 | P0 | `/routablez` must report whether at least one compatible runtime is eligible for a defined routing scope. |
| DEP-043 | P0 | Runtime unavailability must produce a structured `503` and `Retry-After` without removing a healthy gateway from worker-control discovery. |
| DEP-044 | P0 | On SIGTERM, the gateway must stop new admission, transition out of readiness, drain accepted requests, and exit by a configured hard deadline. |
| DEP-045 | P0 | Kubernetes termination grace must exceed the application hard shutdown deadline. |
| DEP-046 | P0 | A fresh Helm installation with no runtime agents must complete successfully and expose not-routable diagnostics. |
| DEP-047 | P0 | A runtime agent must be able to register through the private control Service while public routability is false. |
| DEP-048 | P1 | Chart tests must cover readiness, routability, authenticated status, and control-plane reachability. |

### 9.5 Security requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| DEP-050 | P0 | Public, admin, worker-control, dispatch, runtime, and affinity credentials must remain independent. |
| DEP-051 | P0 | The chart must never render a usable default credential. |
| DEP-052 | P0 | The public ingress must not expose admin or metrics paths unless explicitly enabled and protected. |
| DEP-053 | P0 | The worker-control Service must default to cluster-private access. External exposure must require explicit internal load-balancer or private Gateway configuration. |
| DEP-054 | P0 | Non-loopback control and dispatch channels must require authenticated trusted transport. |
| DEP-055 | P0 | Containers must set `allowPrivilegeEscalation: false`, drop all capabilities, use `RuntimeDefault` seccomp, and support read-only root filesystems. |
| DEP-056 | P0 | Service-account token automount must be disabled unless a separately documented Kubernetes API integration requires it. |
| DEP-057 | P1 | The chart must support namespace and Pod security labels without attempting to weaken cluster policy. |

### 9.6 Release and supply-chain requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| DEP-060 | P0 | Release CI must build both image platforms from the tagged source revision. |
| DEP-061 | P0 | Release CI must publish image SBOMs and provenance attestations and run vulnerability policy before publication. |
| DEP-062 | P0 | Release CI must sign immutable image and chart digests using the project release identity. |
| DEP-063 | P0 | The Helm chart must be published as an OCI artifact and reference the matching application version. |
| DEP-064 | P0 | Release notes must list chart digest, image manifest digests, source revision, SBOMs, provenance, and known compatibility constraints. |
| DEP-065 | P0 | CI must verify that chart defaults render only CPU resources and portable images. |
| DEP-066 | P1 | A release manifest should provide one machine-readable mapping from version to source, images, chart, signatures, and evidence. |

## 10. Configuration contract

The chart exposes structured values for stable deployment controls. Application configuration is
rendered into a ConfigMap or supplied through an existing ConfigMap. Secrets are referenced from
an existing Secret and injected as environment variables.

The following separation is mandatory:

| Surface | Configuration owner |
| --- | --- |
| Gateway replicas, image, resources, Services, probes | Helm values |
| Pools, deployments, equivalence, admission policy | AX Serving configuration |
| Public/admin/control/dispatch credentials | Existing Secret |
| Redis URL and credentials | Existing Secret |
| Ingress certificates and mesh identity | Cluster platform |
| Runtime image, model, accelerator, runtime arguments | Runtime owner |

The chart must not evaluate arbitrary configuration with Helm `tpl`. It may accept a structured
map or an existing ConfigMap reference. Values schema validation must catch invalid types, missing
required references, unsupported deployment modes, invalid port collisions, and unsafe replica/
fleet-store combinations where expressible.

## 11. Networking contract

### 11.1 Public channel

Clients reach the public API through a ClusterIP Service and an optional trusted ingress or
Gateway. Only documented public inference and model-discovery paths are exposed externally by
default. Public client credentials are not valid for admin routes.

### 11.2 Admin and metrics channel

Admin and metrics access is private. Until the application has a separate admin listener, ingress
and Gateway rules must deny admin and metrics paths on the public route. Operators access them
through a private Service, authenticated proxy, port-forward, or service mesh policy.

### 11.3 Worker-control channel

Runtime agents initiate registration and heartbeat requests to the private worker-control Service.
The Service remains discoverable during gateway bootstrap. External agents use an explicitly
configured private load balancer, private Gateway, or overlay network.

### 11.4 Dispatch channel

Gateways initiate inference dispatch requests to runtime agents. Every advertised agent endpoint
must be routable from every gateway replica that can select it. NetworkPolicy supports both Pod
selectors and explicit external CIDRs.

### 11.5 Fleet-store channel

HA gateways use a durable Redis or Valkey endpoint with authentication and trusted transport.
Production mode does not create an unauthenticated in-cluster cache by default.

## 12. Availability and autoscaling

Production defaults are:

- two gateway replicas;
- rolling update with `maxUnavailable: 0` and `maxSurge: 1`;
- one PodDisruptionBudget preserving at least one replica;
- hostname and zone topology preferences;
- Redis/Valkey shared fleet state;
- unique gateway identity derived from the Pod identity;
- no runtime-agent or GPU dependency in the gateway Pod.

HPA is optional until a deployment has measured capacity. CPU utilization may be used only when
CPU requests are set. Preferred primary signals are active admission count, queue depth, request
rate, or event-loop/connection pressure. Runtime queue and GPU saturation are not gateway scaling
signals; they trigger runtime capacity actions instead.

## 13. Non-functional requirements

| ID | Area | Requirement |
| --- | --- | --- |
| NFR-DEP-001 | Portability | Gateway and agent images run on Linux AMD64 and ARM64 without accelerator hardware. |
| NFR-DEP-002 | Isolation | No core image or chart resource contains an accelerator dependency or request. |
| NFR-DEP-003 | Availability | One gateway replica can be replaced without interrupting new admission when another healthy replica exists. |
| NFR-DEP-004 | Stream safety | Accepted streams survive planned replacement up to the configured drain deadline. |
| NFR-DEP-005 | Bootstrap | A chart install with zero runtimes reaches gateway readiness and reports routability false. |
| NFR-DEP-006 | Recovery | A newly eligible runtime becomes routable within two heartbeat intervals plus fleet reconciliation delay. |
| NFR-DEP-007 | Security | Default chart output passes restricted Pod security expectations without privileged exceptions. |
| NFR-DEP-008 | Reproducibility | Every release image and chart resolves to immutable digests and one source revision. |
| NFR-DEP-009 | Observability | Gateway, chart, image, fleet-store kind, readiness, routability, queue, and worker state are observable without high-cardinality labels. |
| NFR-DEP-010 | Operability | Configuration and secret identity changes produce controlled, diagnosable rollouts. |
| NFR-DEP-011 | Scale evidence | Certification includes at least two gateways, 32 runtime agents, and 256 concurrent streams unless a stricter release profile supersedes it. |
| NFR-DEP-012 | Upgrade evidence | Certification includes rolling update, rollback, Redis restart, runtime loss, and external-agent reconnect scenarios. |

Resource defaults are starting points, not performance claims. Production values must be backed by
retained CPU, RSS, connection, queue, latency, and stream-duration evidence.

## 14. Observability requirements

The deployment must expose and document:

- image version and source revision;
- chart version and release name;
- gateway identity and fleet-store kind;
- process liveness and dependency readiness;
- fleet routability and eligible worker count;
- active and queued requests;
- rejection, cancellation, retry, and failure rates;
- endpoint-selection, first-byte, attempt, and stream duration;
- container CPU, RSS, open files, connections, restarts, and throttling;
- rollout generation, desired replicas, available replicas, and disruption state.

Prometheus integration is optional and uses a ServiceMonitor when the operator has the Prometheus
Operator. Metrics credentials must be supplied through an existing Secret or an authenticated
scrape proxy; they must not be committed in a monitor resource.

## 15. Release gates

A container and chart release is not production-qualified until all P0 requirements pass and the
following evidence is attached to the source revision:

1. Both image platforms build and pass startup, liveness, readiness, and non-root smoke tests.
2. SBOM policy confirms no embedded AX Engine, MLX, Metal, CUDA, llama.cpp, or model artifact.
3. Rendered default and production chart profiles contain no accelerator resource or device access.
4. `helm lint --strict`, schema validation, template matrices, and Kubernetes schema validation pass.
5. A disposable cluster installs the chart with no runtime and reaches gateway readiness.
6. A runtime agent registers through the private control Service and changes routability to true.
7. Public inference, streaming, cancellation, drain, and structured no-runtime errors pass.
8. Two gateway replicas reconcile shared fleet state through Redis/Valkey.
9. Rolling upgrade and rollback preserve accepted traffic within the declared drain deadline.
10. NetworkPolicy tests cover in-cluster agents, external-agent CIDRs, DNS, and external Redis.
11. Image and chart signatures, provenance, SBOMs, digests, and source revision verify.
12. A retained scale and soak artifact satisfies the release profile.

Passing build and mock tests is necessary but not sufficient for production certification.

## 16. Success metrics

- One documented command installs the chart from an OCI registry using immutable artifacts.
- One documented command starts the Docker evaluation stack.
- Zero accelerator dependencies or resources appear in core artifacts.
- A fresh chart installation cannot deadlock on runtime registration.
- Runtime loss does not restart or undiscover the gateway control plane.
- Planned gateway upgrades complete without unexplained accepted-request loss.
- External Apple Silicon and in-cluster CUDA agents can coexist in one explicit deployment catalog.
- Operators can identify source, image, chart, configuration, and runtime deployment revisions from
  diagnostics without inspecting mutable tags.

## 17. Delivery stages

### Stage A: Correctness and lifecycle

- Split readiness from routability.
- Preserve worker-control discovery during bootstrap.
- Add gateway drain state and hard shutdown deadline.
- Accept advertised agent URIs or DNS names.
- Add external network-policy configuration.

### Stage B: OCI product artifacts

- Harden Docker targets.
- Add Compose evaluation stack and health probes.
- Build and publish multi-platform images.
- Attach SBOM, provenance, signatures, and release manifest.

### Stage C: First-party Helm chart

- Add values schema and core templates.
- Add production-safe profiles and existing-secret integration.
- Add ingress, monitoring, PDB, HPA, topology, and network-policy options.
- Publish the chart as an OCI artifact.

### Stage D: Qualification

- Add disposable-cluster tests.
- Test external macOS and in-cluster CUDA runtime agents.
- Run HA, partition, rolling-upgrade, rollback, scale, and soak evidence suites.
- Promote the deployment status only after all release gates pass.

## 18. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Readiness tracks runtime capacity | Bootstrap deadlock and cascading gateway removal | Separate readiness and routability; publish control endpoints during bootstrap. |
| Gateway scheduled on GPU nodes | Cost waste and unclear isolation claim | CPU-node selectors/affinity; no GPU tolerations or resources; rendered policy test. |
| Agent sidecar is assumed to use the GPU | Compliance confusion | State that GPU resources are container-scoped; provide separate CPU Deployment pattern. |
| External Mac agents are not routable | Hybrid fleet cannot operate | Private control exposure, routable advertised URI, overlay network, CIDR policy tests. |
| Managed Redis blocked by policy | HA gateways fail or diverge | Explicit external Redis CIDRs and connectivity preflight. |
| Long streams exceed Pod grace | Truncated responses during upgrades | Application drain deadline, Pod grace validation, active-stream rollout tests. |
| Chart owns runtime lifecycle | Tight coupling to CUDA and MLX release cadence | Keep runtime deployments and accelerator operators outside the core chart. |
| Mutable tags drift | Unreproducible rollbacks | Digest support, signatures, provenance, and release manifest. |
| Generic chart escapes weaken security | Unsafe operator overrides | Safe defaults, schema validation, explicit opt-ins, rendered policy tests. |

## 19. Documentation requirements

Public documentation must include:

- the CPU-only guarantee and its exact scope;
- Docker, Compose, and Helm installation paths;
- supported platforms and architectures;
- secret and certificate prerequisites;
- Redis/Valkey prerequisites for HA;
- external AX Engine and CUDA runtime-agent networking;
- readiness versus routability semantics;
- upgrade, drain, rollback, and recovery procedures;
- observability and autoscaling guidance;
- immutable artifact verification;
- limitations and current certification status.

Documentation must not describe the Kustomize baseline, a successful image build, or a mock runtime
test as a production-certified deployment.
