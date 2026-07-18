# AX Serving implementation and certification status

| Field | Value |
| --- | --- |
| Snapshot | 2026-07-15 architecture/source review |
| Final architecture | Accepted in [ADR-016](adr/ADR-016-FEDERATED-DYNAMO-AND-AX-ENGINE-CONTROL-PLANE.md) |
| Product state | Portable gateway and execution-domain source foundation implemented; Dynamo adapter and live federation pending |
| Production claim | Blocked until every applicable PRD release and value gate has retained evidence |
| Canonical requirements | [PRD](prd/PRD-AX-SERVING.md) |
| Detailed design | [Technical specification](specs/TECH-SPEC-FEDERATED-INFERENCE-CONTROL-PLANE.md) |

This ledger separates four states:

- **Designed**: required by the canonical documents but not present in source.
- **Implemented**: present in source with local/mock tests.
- **Live certified**: passed a pinned runtime/domain conformance suite.
- **Production qualified**: also passed HA, security, performance, soak, operations, and value gates.

No lower state implies a higher one.

## Architecture status

| Area | Source state | Certification state | Next required work |
| --- | --- | --- | --- |
| Portable OpenAI REST/SSE gateway | Implemented | Not production qualified | Retain direct-versus-gateway and live mixed-domain evidence. |
| Runtime-SDK-free dependency boundary | Implemented and CI-checked in prior snapshots | Release evidence pending | Re-run forbidden-dependency checks for every image/release. |
| Protocol registration/lease/readiness/inventory/drain | Additive protocol v1.1 domain descriptors/observations and tolerant v1.0 fixtures implemented | Live AX/Dynamo federation certification pending | Implement a conforming adapter and run rolling-version/live-domain tests. |
| Explicit pools/domains/deployments/equivalence | `DomainSpec`, deployment mapping, PC/Thor separation, and fail-closed domain matching implemented | Cross-domain artifacts pending | Add manifest-backed qualification records and live-domain evidence. |
| Request profile and hard admission | `DecisionProfileV1` source type and domain filters implemented; authenticated policy inputs are not wired | Production workload limits pending | Add bounded tenant/routing-profile inputs without prompt parsing. |
| Inference-aware endpoint policy | Existing endpoint scoring plus hard desired/observed domain eligibility implemented | 256-candidate/domain evidence pending | Add a distinct domain-selection stage before Mac endpoint selection. |
| Safe retry, streaming, cancellation, phased deadlines | Implemented with mock/fault tests | Live AX/Dynamo/ingress fault evidence pending | Enforce retry-owner semantics at the Dynamo boundary. |
| Decision records and replay | Bounded, redacted in-process `DecisionRecordV1` journal and admin diagnostics implemented for successful eligible choices | Durable replay and rejected-candidate evidence absent | Persist immutable policy/observation evidence, add rejection reasons, shadow data, and offline replay. |
| Memory and Redis/Valkey HA state | Worker registration records carry v1.1 domain descriptor/observation through reconciliation | Domain-keyed reservation and two-gateway envelope pending | Add domain reservations/fencing and run restart/partition/soak. |
| Async desired deployment/jobs | Implemented as AX state | External lifecycle not certified | Add optional Dynamo/AX lifecycle controllers outside request path. |
| Probes and shutdown | `/livez`, control-plane `/readyz`, `/routablez`, and bounded drain implemented | Deployment drills pending | Align all public/runbook wording and retain rollout evidence. |
| CPU-only OCI/Compose/Kustomize/Helm source | Implemented in repository | Published production artifacts not qualified | Run multi-arch, install/upgrade/rollback, SBOM/sign/signature gates. |

## Execution-domain status

| Domain/path | Source state | Support status |
| --- | --- | --- |
| Mac -> `ax-runtime-agent` -> AX Engine | Optional v1.1 Mac node-domain advertisement plus explicit catalog and fail-closed v1.0 migration mapping implemented | Live pinned AX Engine/domain certification pending |
| Direct vLLM/SGLang -> `ax-runtime-agent` | Existing compatibility path implemented | Migration/testing only; not the final NVIDIA production architecture |
| NVIDIA PC -> Dynamo Domain Adapter -> Dynamo | Designed in ADR/spec | Adapter not implemented; no AX certification |
| NVIDIA Thor -> separate Dynamo Domain Adapter -> Dynamo | Designed in ADR/spec | Experimental design only; no Dynamo-on-Thor certification |
| Embedded AX/MLX/llama.cpp and gRPC v1 | Existing `embedded-compat` path | Compatibility-only, outside federation product |

The canonical upstream Dynamo repository is
[`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo). The initial integration reference is
release `v1.2.1`, but support requires an AX compatibility manifest with exact commit, component
image digests, backend, CUDA/platform, graph config, model identity, and retained test evidence.

Upstream ARM64/Ubuntu 24.04 and Blackwell support plus Thor CUDA 13 container support make Thor
integration plausible. They do not certify the combined stack. Thor remains disabled or
`experimental` until the separate qualification row passes.

## Current source evidence

The reviewed source includes:

- `crates/ax-serving-protocol` v1.1 domain/decision DTOs, validation, and tolerant v1.0/v1.1 fixtures;
- `crates/ax-serving-api` domain-aware catalog, desired/observed filtering, worker state propagation,
  bounded decision diagnostics, fleet, dispatch, HA state, jobs, and probes;
- `crates/ax-thor-agent` with the `ax-runtime-agent` binary and OpenAI runtime proxy;
- separate portable gateway and embedded compatibility features;
- gateway/agent container targets;
- Compose, Kubernetes/Kustomize, first-party Helm, monitoring, and alerting files.

The previous ledger recorded a full local Apple Silicon run of formatting, all-feature Rust
check/clippy/tests, Python, JavaScript, YAML, and monitoring JSON. It reported 1,054 Rust cases,
1,052 passed, and two ignored hardware/model-dependent cases. That is historical source evidence,
not a result of this documentation rewrite and not a live federation certification.

### D1 local source validation (2026-07-15)

- `cargo check --workspace`: passed.
- `cargo test -p ax-serving-protocol`: 27 unit and five fixture tests passed.
- `cargo test -p ax-serving-api --lib -- --test-threads=1`: 345 passed.
- `AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test orchestration --
  --test-threads=1`: 65 passed, including domain dispatch and decision diagnostics.
- `cargo test -p ax-thor-agent -- --test-threads=1`: 56 library and three e2e tests passed,
  including opt-in node-domain registration/heartbeat.
- `cargo clippy -p ax-serving-protocol -p ax-serving-api -p ax-thor-agent --tests --no-deps --
  -D warnings`: passed.
- Local Markdown link resolution and `git diff --check`: passed.

The broader `cargo test --workspace --lib` run reaches `ax-serving-py` and cannot start its test
binary on this host because `libpython3.12.dylib` is absent. Workspace-wide formatting and Clippy
also expose pre-existing drift/lint in files outside this change. These environment/repository
issues are not counted as D1 passes, and none of the local results are live runtime certification.

## Canonical requirement status

| Requirement group | State |
| --- | --- |
| API-001 through API-005, API-007 | Current REST/SSE/compatibility behavior implemented; live domain conformance pending. |
| DOM-001 through DOM-008 | Protocol/catalog/eligibility source foundation implemented; adapter, manifest-backed qualification, HA reservation, and live certification remain. |
| MAC-001 through MAC-005 | Source path implemented; pinned live certification pending. |
| DYN-001 through DYN-009 | Designed; compatibility manifest and adapter implementation pending. |
| MOD-001 through MOD-005 | Logical model/equivalence/domain mapping foundation implemented; compatibility-manifest identity and live equivalence evidence remain. |
| REQ-001 through REQ-004 | Bounded profile and hard domain admission foundation implemented; authenticated cost/privacy/locality/quality inputs and domain scoring remain. |
| REQ-005 through REQ-008 | `DecisionRecordV1` and bounded in-memory selected-candidate diagnostics implemented; rejected candidates, durable replay, shadow/canary, and learned-policy controls remain. |
| DSP-001 through DSP-003, DSP-005, DSP-006 | Existing dispatch implementation present; live evidence pending. |
| DSP-004 and DSP-007 | Dynamo retry-owner and cross-domain rules designed; not implemented. |
| OPS-001 through OPS-005 | Source implementation substantially present; domain extension and deployment drills pending. |
| OPS-006 | Designed; external lifecycle controllers not implemented/certified. |
| PKG-001 through PKG-004 | Source packaging present; production publication/qualification pending. |
| PKG-005 | Complete linked release manifest not yet certified. |
| ADP-001 through ADP-005 | Optional P1/P2; not implemented and not a P0 blocker. |

## Superseded design packages

The 2026-07-15 consolidation removed the prior split documents from the working tree:

- ADR-013 runtime-neutral hybrid control plane;
- ADR-014 CPU-only OCI/Helm deployment;
- ADR-015 agent-aware inference fabric;
- their separate deployment and agent-session PRDs/specifications.

Their still-valid requirements were folded into ADR-016, the canonical PRD, and the consolidated
technical specification. In particular, runtime-SDK isolation, CPU-only packaging,
`/readyz`/`/routablez`, graceful drain, safe retry, privacy-preserving affinity, and the rule that AX
is not an agent orchestrator remain in force.

## Release blockers

A production federated release remains blocked until all applicable items are attached to one AX
source digest:

1. Complete D1 beyond its source foundation: retain protocol v1.0/v1.1 compatibility evidence,
   add rejected-candidate decision evidence, immutable policy configuration, and domain-keyed HA
   reservations without regressing existing routing.
2. One pinned Mac AX Engine deployment passes API, identity, streaming, cancellation, drain,
   credential, fault, and performance conformance.
3. One pinned NVIDIA PC Dynamo deployment passes the same federation contract; the compatibility
   manifest identifies every upstream component/backend/image/config artifact.
4. Direct per-worker CUDA registration is absent from the production profile and remains clearly
   compatibility-only.
5. Thor is either disabled/experimental in the release or passes its own ARM64/CUDA/backend,
   memory/thermal, fault, performance, and soak qualification.
6. Cross-domain equivalence artifacts exist for every enabled failover/retry transition.
7. Gateway/adapter images and Helm pass multi-architecture, dependency, security, install, upgrade,
   rollback, SBOM, provenance, signature, and vulnerability gates.
8. Two gateways plus Redis/Valkey pass restart, partition, fencing, reservation, drain, rollout,
   rollback, overload, and at least 60 minutes of mixed-domain soak.
9. Direct-versus-federated overhead passes NFR-001/NFR-002/NFR-006 and no duplicate commitment is
   observed.
10. At least one PRD value gate proves cost/load, availability, privacy/locality, or operator-workflow
    value on a representative workload.
11. Transport, credential rotation, monitoring, alerting, incident, Dynamo upgrade, and rollback
    drills have retained evidence.

Only after these gates pass may public documentation say “production-ready federated inference
control plane.” Only after the Thor row passes may it say “Thor supported” without an experimental
qualifier.

## Next implementation milestone

The D1 source foundation now exists. The next product milestone is **D2: NVIDIA PC Dynamo adapter**,
with a small D1 hardening track in parallel:

1. Define and verify an immutable Dynamo compatibility manifest before registration becomes ready.
2. Implement one adapter endpoint representing one pinned Dynamo deployment; never register its
   internal GPU workers with AX Serving.
3. Normalize aggregate readiness, inventory, capacity, cancellation, and typed pre-admission.
4. Begin in shadow inventory mode, then canary one PC/backend/model combination after fault and
   credential-isolation tests.
5. Complete D1 rejected-candidate evidence, durable decision storage/replay, and domain-keyed HA
   reservations before calling the federation path production ready.

Adaptive routing remains after the adapter, live certification, HA, and value gates.
