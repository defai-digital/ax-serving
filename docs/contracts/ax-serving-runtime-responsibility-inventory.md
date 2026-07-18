# AX Serving runtime responsibility inventory

| Field | Value |
| --- | --- |
| Status | Active target boundary |
| Last updated | 2026-07-15 |
| Authority | [ADR-016](../../.internal/adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md) |

## Product boundary

- AX Serving is the cross-domain federation and policy plane.
- NVIDIA Dynamo is the NVIDIA PC/Thor domain-local distributed inference system.
- AX Engine is the Apple Silicon inference runtime.
- NVIDIA PC and Thor are separate Dynamo domains and qualification classes.

AX Serving selects a domain. Dynamo selects NVIDIA workers. AX Engine executes on Mac.

## Responsibility matrix

| Concern | AX Serving | Dynamo NVIDIA domain | AX Engine/Mac path |
| --- | --- | --- | --- |
| Public OpenAI API/auth | Owns | Not public through AX | Not public through AX |
| Tenant quota, trust, privacy, residency, locality | Owns | Receives admitted request | Receives admitted request |
| Logical model and routing profile | Owns | Receives runtime model | Receives runtime model |
| Model identity and cross-domain equivalence | Owns/certifies | Reports pinned inventory | Reports pinned inventory |
| Cross-domain admission/selection | Owns | Does not override | Does not override |
| NVIDIA worker selection and KV-aware routing | Does not own | Owns | Not applicable |
| NVIDIA prefill/decode disaggregation, KVBM, NIXL | Does not own | Owns | Not applicable |
| NVIDIA planner/scaling/backend process graph | Desired global intent only | Owns | Not applicable |
| Mac node selection | Owns after hard filters | Not applicable | Agent reports capacity/readiness |
| Tokenization/templates/batching/KV/speculation | Does not own | Backend/Dynamo own | AX Engine owns |
| Cross-domain retry | Owns only before admission/commitment | Does not own | Does not own |
| In-domain retry/migration | Observes one domain attempt | Dynamo owns | AX Engine/agent semantics |
| Streaming/cancellation/deadlines | End-to-end transport owner | Must propagate/execute | Must propagate/execute |
| Decision reason/audit/policy version | Owns | Supplies bounded observation | Supplies bounded observation |

## Source classification

| Area | Current role | Final role | State |
| --- | --- | --- | --- |
| OpenAI REST/SSE gateway | Portable API/control plane | Keep | Implemented |
| `ax-serving-protocol` v1.1 | Worker/domain wire contract with v1.0 tolerance | Keep additive within major 1 | Domain foundation implemented; live certification pending |
| Registry, leases, admission, catalog, equivalence | Domain-aware endpoint control plane | Add certified adapters and full domain policy | Foundation implemented / adapter pending |
| Redis/Valkey state | Worker/deployment HA | Extend with domain/policy records | Implemented / extension pending |
| `ax-runtime-agent` binary | Generic OpenAI runtime adapter | Mac AX Engine adapter | Implemented; live certification pending |
| Direct vLLM/SGLang agent modes | CUDA endpoint adapter | Migration/testing compatibility only | Implemented compatibility |
| Future `ax-dynamo-adapter` | None | One adapter per Dynamo execution domain | Designed, not implemented |
| Future Dynamo lifecycle controller | None | Optional async desired-state bridge | Designed, not implemented |
| Embedded AX/MLX/llama.cpp backends | Local inference | `embedded-compat` only | Compatibility |
| `ax.serving.v1` gRPC | Local paths/backend/token IDs | `embedded-compat` only | Compatibility |
| LAN mDNS `_ax-engine._tcp` | Lab/home discovery | Optional Mac bootstrap, never auth/state | Implemented optional bootstrap |
| Helm/Compose/Kustomize | AX gateway packaging | AX federation plane only; no Dynamo/runtime install | Source implemented, release qualification pending |

## Dynamo boundary rules

- Use canonical upstream `https://github.com/ai-dynamo/dynamo` with exact released tag/commit and
  immutable image digests.
- Do not fork/vendor Dynamo by default or link its runtime SDKs into the gateway.
- Register one Dynamo deployment as one AX domain endpoint; do not register Dynamo GPU workers.
- Do not copy Dynamo worker/KV/router/planner state into AX fleet state.
- Do not direct-route around the Dynamo frontend to an internal worker.
- Do not translate a generic Dynamo/backend `5xx` into AX typed non-admission.
- Keep PC and Thor manifests, pools, calibration, artifacts, rollout, and evidence separate.
- Contribute general improvements upstream; keep AX tenant/equivalence/federation policy in AX.

## Compatibility rules

- Compatibility paths must not become new product architecture.
- The default gateway graph remains free of runtime/accelerator SDKs.
- Direct CUDA agents cannot participate in the production Dynamo profile after its certification.
- Removing a compatibility backend requires a replacement path, migration note, conformance
  evidence, and deprecation window.
- `AXS_EMBEDDED_RUNTIME_POLICY=deny` remains the production direction for the portable gateway.

## Removal and promotion gates

Before direct CUDA agents are disabled in production:

1. The Dynamo Domain Adapter passes the pinned PC conformance matrix.
2. Model identity/equivalence and retry-owner tests pass.
3. Direct Dynamo versus through-AX overhead is retained.
4. Drain, fault, upgrade, and rollback procedures pass.
5. A migration document maps current pools/deployments to domains.

Thor cannot be promoted from experimental until its separate ARM64/CUDA/backend, memory/thermal,
fault, performance, and soak evidence passes.
