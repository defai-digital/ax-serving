# Ideal customer and demand qualification

AX Serving is worth evaluating when a team answers yes to several of these:

- We operate both Apple Silicon and NVIDIA inference capacity.
- NVIDIA PC and Thor must be separate deployment, trust, or failure domains.
- Clients should use logical models rather than runtime/domain addresses.
- We need tenant, privacy, residency, locality, budget, or SLO policy above runtimes.
- Failover must not silently cross tokenizer, template, quantization, revision, or trust boundaries.
- We need central admission, audit, drain, rollout, diagnostics, or active-active state.
- We can measure a representative workload and define a quality/safety floor.

## Disqualifiers

- One Dynamo endpoint satisfies all traffic and policy.
- One Mac/AX Engine endpoint satisfies all traffic and policy.
- The team expects AX to improve Dynamo's internal GPU scheduling.
- The team wants an agent planner/tool/memory framework.
- No owner can maintain model identity/equivalence and pinned compatibility manifests.
- No workload or value metric exists to justify the extra hop/control plane.

## Required proof before adoption

Choose at least one measurable outcome:

- at least 20% lower NVIDIA load or measured cost while maintaining quality/SLO;
- policy-correct availability during a domain outage/drain with no duplicate commitment;
- zero privacy/locality violations while using one API;
- a defined reduction in endpoint-specific client/operations work.

Also measure direct-domain versus through-AX overhead. If value does not exceed the operational and
latency cost, deploy AX Engine or Dynamo directly and do not expand AX policy scope.

## Current availability

The current source contains the gateway and Mac-capable agent foundations. The final Dynamo domain
adapter and Thor qualification remain roadmap work. See the
[status ledger](../.internal/IMPLEMENTATION-STATUS.md) before planning a deployment.
