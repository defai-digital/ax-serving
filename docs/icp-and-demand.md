# Ideal deployment profile

AX Serving is worth evaluating when a team answers yes to several of these:

- We operate more than one inference endpoint, pool, runtime, or hardware type.
- Clients should use logical model names rather than know runtime addresses.
- We need to drain or replace runtimes without changing client configuration.
- Failover must not cross tokenizer/template/quantization identity silently.
- Public, admin, worker, dispatch, and runtime credentials need separate trust
  boundaries.
- We need active-active gateways and shared capacity accounting.
- We need tenant admission, priorities, diagnostics, or audit above runtimes.
- Apple Silicon/MLX and CUDA capacity should coexist in one managed fleet.

AX Serving is probably unnecessary when one stable runtime endpoint already
meets availability, security, and operations needs. Another gateway may be a
better choice when the organization is standardized on its APIs and does not
need AX Serving's explicit identity/equivalence model.

## Adoption prerequisites

- a runtime endpoint with reliable readiness and model inventory;
- immutable or at least traceable runtime/model artifacts;
- an owner for deployment identity and equivalence certification;
- workload traces and SLO definitions;
- a secret and transport-security system for remote deployments;
- durable Redis/Valkey for active-active state;
- an incident owner willing to fail closed rather than bypass compatibility.

## Evaluation sequence

1. Run one runtime directly and through one gateway/agent.
2. Validate streaming, cancellation, credential isolation, and drain.
3. Add explicit deployment identity without cross-runtime equivalence.
4. Certify a second deployment and test only the approved equivalence class.
5. Add a second gateway and validate restart/partition behavior.
6. Run target load, failure scenarios, and the 60-minute soak.
7. Adopt only if overhead, goodput, safety, and operational burden meet the
   team's stated criteria.

Demand should be measured from completed evaluations and retained deployment
evidence, not assumed from company size, model parameter count, or hardware
brand.
