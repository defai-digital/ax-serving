# AX Serving positioning

AX Serving is a federated heterogeneous inference control plane for private AI fleets.

It sits above execution domains:

- AX Engine owns Apple Silicon execution;
- NVIDIA Dynamo owns NVIDIA PC/Thor domain-local routing, KV, planning, scaling, and backend
  execution;
- AX Serving owns the one-API federation policy across Mac, NVIDIA PC, and a separately qualified
  NVIDIA Thor domain.

## Core promise

One authenticated OpenAI-compatible endpoint can enforce logical-model identity, tenant policy,
privacy/locality, capability, SLO/cost constraints, audit, and safe cross-domain failover without
linking runtime SDKs or competing with Dynamo's worker scheduler.

AX Serving selects a domain. Dynamo selects NVIDIA workers. AX Engine executes on Mac.

## Primary user

The primary user is a platform team with at least two hardware/trust domains or a real need for
central identity, admission, audit, or lifecycle. A single Dynamo deployment with one policy is not
a strong fit; that user should call Dynamo directly.

## Differentiators that must be proved

- Mac + NVIDIA PC + separate Thor federation under one logical model/API;
- explicit, fail-closed deployment identity and cross-domain equivalence;
- tenant, privacy, residency, locality, budget, and SLO policy above runtimes;
- conservative pre-commit retry with clear AX-versus-Dynamo retry ownership;
- bounded, versioned, replayable decision records;
- active-active AX state, capacity fencing, drain, rollout, diagnostics, and audit;
- a measurable value gate: cost/load, policy-correct availability, privacy/locality, or simpler
  operations.

These are product hypotheses until pinned Mac/Dynamo conformance, performance, HA, fault, soak, and
value evidence passes.

## Anti-scope

AX Serving is not:

- a token engine, CUDA runtime, or Dynamo replacement;
- a second NVIDIA worker/KV router or planner;
- an attempt to put PC and Thor in one homogeneous pool;
- a tensor, prefill/decode, or KV splitter across domains;
- an agent planner, tool/MCP executor, sandbox, or durable memory system;
- an online self-learning production scheduler;
- useful by default for one model on one endpoint.

## Message hierarchy

1. “AX Serving federates execution domains; it does not replace them.”
2. “Dynamo manages NVIDIA workers; AX manages cross-domain policy.”
3. “PC and Thor are independently deployed and must be independently certified.”
4. “Homogeneous NVIDIA users should use Dynamo directly.”
5. “AX Serving earns its place only through retained value and safety evidence.”

## Current claim boundary

The repository implements the portable gateway and Mac-capable runtime-agent foundations. The
Dynamo Domain Adapter and final domain protocol are target design work, and Thor is experimental
until live qualification. Public ownership boundaries are defined by the
[runtime responsibility inventory](contracts/ax-serving-runtime-responsibility-inventory.md).
