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
make AX Fabric easier to run, scale, operate, and commercialize
```

## Version Direction

### v1.2.x

Goal: positioning cleanup and stable release baseline

- Align README, docs, and release language around "for AX Fabric"
- Complete licensing, packaging, and release cleanup
- Harden request validation and config correctness
- Improve test reliability and release hygiene
- Clarify single-worker versus gateway/worker operating modes

### v1.3

Goal: become the standard runtime for AX Fabric model access

- Stabilize embeddings and core OpenAI-compatible APIs
- Tighten model lifecycle behavior: load, unload, reload, health
- Add stronger AX Fabric integration coverage
- Improve startup configuration, metrics, and health semantics
- Reduce setup friction for local AX Fabric plus AX Serving deployments

### v1.4

Goal: strengthen orchestration and scheduler quality

- Implement real batching improvements where practical
- Reduce request-lifetime slot pinning
- Improve queue fairness and worker dispatch behavior
- Add token-cost-aware routing inputs
- Improve cache coordination and duplicate-request handling

### v1.5

Goal: become production-grade control plane infrastructure

- Expand observability: metrics, tracing, dashboards, operational status
- Improve admin and control-plane APIs
- Harden failure handling and worker lifecycle management
- Clean up business and enterprise edition gating
- Improve deployment and operations documentation

### v1.6+

Goal: differentiated infrastructure for advanced AX Fabric deployments

- Mature Mac Grid multi-node operation
- Evaluate broader backend support beyond current runtime assumptions
- Improve Apple Silicon-specific performance paths
- Add stronger policy, admission, and multi-tenant controls
- Deepen one-command deployment experience with AX Fabric

## What Not To Do

- Do not position AX Serving as a separate flagship product
- Do not optimize roadmap around competing directly with Ollama on single-node
  simplicity
- Do not duplicate AX Fabric's product-layer responsibilities inside AX Serving
- Do not grow broad framework features that weaken execution-layer focus

## Success Criteria

AX Serving is on the right roadmap if each version makes AX Fabric:

- easier to deploy
- easier to integrate
- easier to operate
- more reliable under load
- more differentiated in business and enterprise deployments
