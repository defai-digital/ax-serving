# AX Serving positioning

AX Serving is a federated multi-domain inference control plane for private AI fleets. Heterogeneous
hardware is a strong use case, but the product boundary is independently operated execution
domains—not hardware mixing by itself.

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

The CPU-only gateway can run independently on Apple Silicon, Linux AMD64, or Linux ARM64 and govern
local or remote domains over an operator-provided trusted network. See
[Control-plane placement and mixed-fleet topologies](deployment-topologies.md).

The AX Serving v2.3 release line is the Apache-2.0 open infrastructure layer. AX Fabric and AX Trust may provide
separate orchestration, governance, trust, managed-service, and enterprise value through the public
contracts. They are not required to operate AX Serving and do not unlock or relicense it.

## Primary user

The primary user is a platform team with at least two execution, trust, region, rollout, or failure
domains, or a real need for central identity, admission, audit, or lifecycle. The domains may mix
Mac and NVIDIA, be all Mac, or be separate CUDA/Dynamo domains. Hardware heterogeneity alone is
neither required nor sufficient. A single Dynamo deployment with one policy is not a strong fit;
that user should call Dynamo directly.

## Differentiators that must be proved

- Mac + NVIDIA PC + separate Thor federation under one logical model/API;
- explicit, fail-closed deployment identity and cross-domain equivalence;
- tenant, privacy, residency, locality, budget, and SLO policy above runtimes;
- conservative pre-commit retry with clear AX-versus-Dynamo retry ownership;
- bounded, versioned, replayable pre-dispatch decision records;
- active-active AX state, capacity fencing, drain, rollout, diagnostics, and audit;
- a measurable value gate: cost/load, policy-correct availability, privacy/locality, or simpler
  operations.

These are product hypotheses until pinned Mac/Dynamo conformance, performance, HA, fault, soak, and
value evidence passes.

The current decision records prove what the gateway selected from the policy inputs it observed.
They do not prove runtime admission, hardware placement, response completion, or execution-stack
attestation.

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
4. “Single-domain NVIDIA users should use Dynamo directly; separate CUDA domains may still need
   AX-level federation.”
5. “AX Serving earns its place only through retained value and safety evidence.”
6. “AX Serving is the open foundation; AX Fabric and AX Trust are separate products.”

## Current claim boundary

The repository implements the portable gateway, protocol v1.2 execution-domain
foundation, Mac-capable runtime agent, runtime-SDK-free Dynamo Domain Adapter,
and runtime-neutral Mac cluster coordinator with source/mock conformance. The
Mac cluster control plane is not a distributed AX Engine runtime claim.

**Live laboratory evidence** exists for heterogeneous **compatibility multi-worker** federation
(vLLM + Thor TensorRT Edge-LLM + Mac llama.cpp under one logical model, including soak). That is
not production certification, not Dynamo-domain qualification, not multi-Mac cluster certification,
and not live mixed-domain failover evidence. Published multi-architecture Linux production artifacts
and Dynamo PC/Thor domain gates remain open. Thor Edge-LLM remains an experimental compatibility
path. See the
[retained evidence summary](qualification/2026-08-05-heterogeneous-compatibility-fleet.md) and
[deployment topology guide](deployment-topologies.md).
