# AX Serving Roadmap

## Positioning

AX Serving is not the primary top-level product. AX Serving is the execution
and model-serving control plane for AX Fabric.

- AX Fabric: product surface for retrieval, memory, ingestion, and grounded AI
  workflows
- AX Serving: runtime/control plane for model access, orchestration, routing,
  scheduling, APIs, and deployment operations

Roadmap priority for AX Serving is therefore:

```text
make AX Fabric deployable and operable as an offline enterprise LLM and AI system
```

## Target Outcome

AX Serving should enable AX Fabric to be deployed as:

- an offline-first LLM runtime
- an enterprise AI serving/control plane
- a governed and observable local or air-gapped system
- a secure execution layer for grounded AI workloads

This means roadmap decisions should optimize for:

- offline and air-gapped deployment
- operational reliability
- enterprise governance and policy control
- secure local model execution
- clear commercial differentiation for enterprise environments

## Product Principles

- Offline first: the system must work without cloud dependence
- Enterprise operability: logs, metrics, health, and admin surfaces must be
  usable by real ops teams
- Predictable runtime behavior: model lifecycle, scheduling, and worker control
  must be deterministic and debuggable
- Secure by default: auth, license control, policy enforcement, and tenant
  boundaries must be treated as core product features
- Infrastructure focus: AX Serving should power AX Fabric, not compete with it

## Version Direction

### v1.2.x

Goal: positioning cleanup and trusted offline release baseline

- Align README, docs, and release language around "for AX Fabric"
- Complete licensing, packaging, and release cleanup
- Harden request validation and config correctness
- Improve test reliability and release hygiene
- Clarify single-worker versus gateway/worker operating modes
- Remove accidental public/internal leakage from repo and release surface
- Tighten offline packaging assumptions and local-only operating guidance

### v1.3

Goal: become the standard offline runtime for AX Fabric model access

- Stabilize embeddings and core OpenAI-compatible APIs
- Tighten model lifecycle behavior: load, unload, reload, health
- Add stronger AX Fabric integration coverage
- Improve startup configuration, metrics, and health semantics
- Reduce setup friction for local AX Fabric plus AX Serving deployments
- Improve local deployment ergonomics for single-node enterprise pilots
- Clarify supported offline model/runtime paths and operational defaults

### v1.4

Goal: strengthen orchestration, scheduling, and predictable service behavior

- Implement real batching improvements where practical
- Reduce request-lifetime slot pinning
- Improve queue fairness and worker dispatch behavior
- Add token-cost-aware routing inputs
- Improve cache coordination and duplicate-request handling
- Improve deterministic failure handling under constrained local hardware
- Make latency and throughput behavior more stable for enterprise workloads

### v1.5

Goal: become production-grade enterprise control-plane infrastructure

- Expand observability: metrics, tracing, dashboards, operational status
- Improve admin and control-plane APIs
- Harden failure handling and worker lifecycle management
- Clean up business and enterprise edition gating
- Improve deployment and operations documentation
- Add stronger policy, governance, and operational guardrails
- Improve authentication, role separation, and auditability assumptions
- Support real enterprise runbooks for offline and controlled-network setups

### v1.6+

Goal: differentiated offline enterprise infrastructure for advanced deployments

- Mature Mac Grid multi-node operation
- Evaluate broader backend support beyond current runtime assumptions
- Improve Apple Silicon-specific performance paths
- Add stronger policy, admission, and multi-tenant controls
- Deepen one-command deployment experience with AX Fabric
- Add higher-trust deployment patterns for air-gapped and regulated environments
- Improve fleet-level operations for enterprise-scale local infrastructure

## What Not To Do

- Do not position AX Serving as a separate flagship product
- Do not optimize roadmap around competing directly with Ollama on single-node
  simplicity
- Do not duplicate AX Fabric's product-layer responsibilities inside AX Serving
- Do not grow broad framework features that weaken execution-layer focus
- Do not make cloud dependence a requirement for core product value
- Do not optimize for hobbyist convenience at the expense of enterprise control

## Success Criteria

AX Serving is on the right roadmap if each version makes AX Fabric:

- easier to deploy
- easier to integrate
- easier to operate
- more reliable under load
- more secure in offline environments
- more governable in enterprise environments
- more differentiated in business and enterprise deployments
