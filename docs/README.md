# AX Serving documentation

Use this page to choose the shortest path for your task. The repository contains user guides,
operator runbooks, public contracts, and historical design records; they are intentionally not all
entry-point documentation.

## New to AX Serving

Read these in order:

1. [Project overview](../README.md) — product boundary, architecture, status, and supported paths.
2. [Quick start](../QUICKSTART.md) — run a gateway, attach one runtime, and send a completion.
3. [Use cases and trade-offs](advantages-and-use-cases.md) — when federation earns its extra hop.
4. [Deployment topologies](deployment-topologies.md) — where gateways, agents, and domain adapters
   belong.

The central distinction is:

```text
compatibility runtime endpoint        production NVIDIA domain
--------------------------------      --------------------------------
AX gateway -> runtime agent ->        AX gateway -> Dynamo adapter ->
vLLM/SGLang/TRT-LLM/etc.              Dynamo -> certified backend workers

evaluation and migration              target production architecture
```

A working compatibility endpoint does not certify a Dynamo domain. A shared model name does not
prove that two deployments are semantically equivalent.

## Evaluate a runtime

| Runtime or platform | Guide | Qualification boundary |
| --- | --- | --- |
| Generic OpenAI-compatible runtime | [Quick start](../QUICKSTART.md#3-attach-a-runtime) | Compatibility endpoint |
| vLLM on one NVIDIA PC | [Compose profiles](../deploy/compose/README.md#nvidia-runtime-profiles-on-one-pc) | Compatibility endpoint |
| SGLang on one NVIDIA PC | [Compose profiles](../deploy/compose/README.md#nvidia-runtime-profiles-on-one-pc) | Compatibility endpoint |
| TensorRT-LLM on one NVIDIA PC | [Compose profiles](../deploy/compose/README.md#nvidia-runtime-profiles-on-one-pc) | Compatibility endpoint |
| TensorRT Edge-LLM on Jetson Thor | [Thor guide](../deploy/thor/README.md) | Experimental Thor compatibility endpoint |
| AX Engine on one Mac | [Quick start](../QUICKSTART.md#3-attach-a-runtime) | Mac endpoint; exact deployment still needs evidence |
| Mac model-parallel coordinator | [Mac cluster integration](integrations/mac/CLUSTER.md) | Source-level coordinator; physical certification pending |
| NVIDIA Dynamo | [Dynamo integration](integrations/nvidia/DYNAMO.md) | Production target; exact stack must be qualified |

Runtime qualification tools are documented in
[`scripts/qualification/runtime`](../scripts/qualification/runtime/README.md) and
[`scripts/qualification/nvidia`](../scripts/qualification/nvidia/README.md).

## Deploy and operate

| Task | Guide |
| --- | --- |
| Single-gateway Compose evaluation | [Compose](../deploy/compose/README.md) |
| Kubernetes baseline | [Kubernetes](../deploy/kubernetes/README.md) |
| Helm deployment | [Helm chart](../deploy/helm/ax-serving/README.md) |
| Multi-worker and active-active operations | [Federated fleet runbook](runbooks/multi-worker.md) |
| Mac cluster operations | [Mac cluster runbook](runbooks/mac-cluster-operations.md) |
| Monitoring and alerts | [Monitoring](../deploy/monitoring/README.md) |
| Performance and capacity tuning | [Service tuning](perf/service-tuning.md) |
| Releases and rollback | [Release runbook](runbooks/releases.md) |

Production deployments use separate public, admin, worker-control, dispatch, runtime/Dynamo, and
Redis credentials. Evaluation defaults are not production secrets or transport policy.

## Integrate clients and systems

- [Python client](python-sdk.md)
- [JavaScript SDK](../sdk/javascript/README.md)
- [AX Code integration](ax-code-integration.md)
- [Public HTTP and configuration contract inventory](contracts/ax-serving-public-contract-inventory.md)
- [Runtime-agent protocol](contracts/ax-serving-node-contract.md)
- [Runtime responsibility inventory](contracts/ax-serving-runtime-responsibility-inventory.md)
- [AX Fabric integration contract](contracts/ax-fabric-runtime-contract.md)

## Understand status language

Documentation uses the following terms deliberately:

| Term | Meaning |
| --- | --- |
| Source implemented | Code and automated tests exist in this repository |
| Mock/conformance tested | Contract behavior passed without proving a live hardware stack |
| Live qualified | The exact runtime, model, artifact, hardware, and configuration produced retained evidence |
| Certified | The applicable correctness, fault, security, performance, and soak gates passed |
| Experimental | Useful for evaluation, but not a production support claim |

Portable compilation does not qualify every runtime on that architecture. Likewise, a successful
smoke test is not a performance, failover, or production certification result.

## Product and strategy context

- [Positioning and claim boundary](market-positioning.md)
- [Competitive category map](competitive-landscape.md)
- [Ideal customer and demand qualification](icp-and-demand.md)

## Maintainers and design history

Files under `docs/designs/`, dated design documents, and refactor plans explain decisions and
planned work. They are useful to maintainers, but may describe historical baselines or future
gates. For current operator behavior, prefer the quick start, deployment guides, runbooks, and
public contracts above.
