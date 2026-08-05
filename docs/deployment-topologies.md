# Control-plane placement and mixed-fleet topologies

AX Serving separates **where the control plane runs** from **where inference executes**. The
portable, CPU-only `ax-serving-api` gateway can govern Mac AX Engine pools and NVIDIA Dynamo
domains without running on the same architecture or host as those runtimes.

The intended control-plane host classes are:

- Apple Silicon macOS;
- Linux AMD64;
- Linux ARM64, including a Thor host when co-location is appropriate.

From any of these host classes, a reachable gateway can govern local or remote Mac, NVIDIA PC, and
NVIDIA Thor execution domains. Network reachability, trust, and qualification determine
eligibility—not the gateway's CPU architecture.

## Placement matrix

| Gateway placement | Mac AX Engine pool | AMD64 NVIDIA PC domain | ARM64 NVIDIA Thor domain |
| --- | --- | --- | --- |
| Apple Silicon Mac | Yes, through `ax-runtime-agent` | Yes, through a Dynamo Domain Adapter | Yes, through a separate Dynamo Domain Adapter |
| Linux AMD64 | Yes, through `ax-runtime-agent` | Yes, through a Dynamo Domain Adapter | Yes, through a separate Dynamo Domain Adapter |
| Linux ARM64 server | Yes, through `ax-runtime-agent` | Yes, through a Dynamo Domain Adapter | Yes, through a separate Dynamo Domain Adapter |
| NVIDIA Thor host | Technically the same as Linux ARM64 | Technically reachable | Yes, but gateway/runtime co-location shares resources and a failure domain |

Running the gateway on Thor does not certify Dynamo-on-Thor inference. Control-plane portability and
execution-domain qualification are independent claims.

## Component placement

| Component | Placement | Responsibility |
| --- | --- | --- |
| `ax-serving-api` | One or more CPU hosts | Public API, authentication, logical models, policy, domain selection, safe retry boundary, and audit |
| `ax-runtime-agent` | Beside each Mac AX Engine endpoint | Mac readiness, inventory, identity, capacity, cancellation, and byte-preserving proxy |
| `ax-mac-cluster-adapter` | Beside one future AX Engine cluster coordinator | Cluster generation, gang readiness, bounded rank control, AX registration, and byte-preserving rank-0 proxy |
| `ax-dynamo-adapter` | Beside or near one Dynamo frontend/domain | Domain identity, aggregate observation, AX registration, admission boundary, and byte-preserving proxy |
| AX Engine | Apple Silicon Mac | Tokenization, templates, batching, caches, speculation, and execution |
| NVIDIA Dynamo and backend | NVIDIA PC or Thor domain | NVIDIA worker selection, KV-aware routing, disaggregation, planning, scaling, retry, and execution |
| Redis/Valkey | Reachable trusted infrastructure for the HA profile | Shared AX-owned gateway/fleet state |

The gateway and adapters do not require a GPU and do not link AX Engine, MLX, Metal, CUDA, Dynamo,
vLLM, SGLang, or TensorRT-LLM runtime SDKs.

## Adapter rules

Mac and NVIDIA use different adapter boundaries:

- Every independently registered Mac endpoint runs `ax-runtime-agent` beside AX Engine.
- A model-parallel Mac cluster registers once through `ax-mac-cluster-adapter`; its ranks are never
  independently routable. The adapter is experimental and does not provide the missing distributed
  AX Engine runtime.
- One `ax-dynamo-adapter` represents one independently operated Dynamo domain, not one GPU worker.
- NVIDIA PC and Thor always use separate domain IDs, adapters, manifests, artifacts, capacity
  calibration, rollout state, and qualification evidence.
- AX Serving selects a domain. Dynamo selects the NVIDIA worker inside that domain.

The direct vLLM/SGLang mode in `ax-runtime-agent` remains available for migration and testing. It is
`compatibility_runtime_endpoint`, not the target NVIDIA production path. This remains true on both
AMD64 PCs and Thor: the legacy `ax-thor-agent` executable is an alias for `ax-runtime-agent` and
cannot register a Dynamo domain.

## Local-office topology

A small office can run the gateway on an existing Mac, a Linux AMD64 server, a Linux ARM64 server,
or—when resource and failure isolation are acceptable—a Thor host.

```text
                       office LAN
                           |
                  AX Serving gateway
                  (Mac/AMD64/ARM64)
                     /           \
                    /             \
      Mac + ax-runtime-agent    PC/Thor Dynamo domain
               |                     |
           AX Engine          ax-dynamo-adapter
                                      |
                                  Dynamo workers
```

A single gateway with in-memory fleet state is suitable for evaluation. It is not an HA claim.

## Cross-network topology

Domains can reside in different offices, private data centers, or trusted network segments:

```text
Office A                 Office B                 Private data center
Mac AX Engine pool       Thor Dynamo domain       PC Dynamo domain
        \                     |                         /
         +---------- operator-provided trusted network --------+
                                    |
                           AX Serving gateway(s)
                                    |
                              Redis/Valkey (HA)
```

AX Serving does not create the WAN, VPN, ingress, certificates, or service mesh. A remote
deployment must provide:

- private routing or a trusted mesh/VPN;
- TLS/mTLS appropriate to the deployment;
- stable, reachable advertised adapter addresses;
- DNS and firewall policy;
- separate public, admin, worker-control, dispatch, runtime, Dynamo, and fleet-store credentials;
- latency and failure budgets that include the gateway-to-domain network path.

Do not expose worker-control or adapter interfaces directly to the public Internet. A wide-area
disconnect makes an affected domain ineligible; it must never be interpreted as spare capacity.

## Where to place the gateway

Prefer an ordinary, independently managed CPU host or CPU-only Kubernetes node when available. This
keeps the control plane outside an inference domain's resource and failure envelope.

Co-location on a Mac or Thor can be useful for a compact office or evaluation deployment, but it
has trade-offs:

- inference load and the gateway share CPU and memory;
- a host restart removes both control and execution capacity;
- maintenance and credential boundaries are harder to isolate.

Gateway placement should usually follow client/network reachability and control-plane availability,
not accelerator locality. Place `ax-runtime-agent` and `ax-dynamo-adapter` close to their runtimes.

## When to bypass AX Serving

Use the execution system directly when federation adds no measurable value:

| Situation | Recommended entry point |
| --- | --- |
| One CUDA/Dynamo domain with one policy | Call Dynamo directly |
| One simple Mac/AX Engine endpoint | Call AX Engine directly |
| Two or more independently operated domains with shared policy, identity, audit, or failover | Evaluate AX Serving |

An all-CUDA fleet can still need AX Serving when it contains separate Dynamo domains for regions,
trust boundaries, failure isolation, PC versus Thor qualification, or independent rollout. The
same is true for an all-Mac fleet with several independently operated endpoints. Conversely, mixed
hardware alone does not justify AX Serving. The product boundary is multi-domain governance, not
hardware mixing by itself.

## Current claim boundary

The repository currently checks portable gateway source on Linux AMD64 and Linux ARM64 and packages
Apple Silicon macOS release binaries. CPU-only container, Compose, Kubernetes, and Helm sources are
present. Published, production-qualified multi-architecture Linux artifacts and live mixed-domain
certification remain release gates.

The Dynamo Domain Adapter and compatibility-manifest validation have source/mock conformance.
NVIDIA PC live qualification is pending, and Thor remains experimental until its independent
hardware, correctness, fault, performance, thermal, and soak gates pass.

Production support claims require retained qualification artifacts for every applicable gate;
source and mock conformance alone are not sufficient.
