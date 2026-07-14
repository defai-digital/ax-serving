# AX Serving agent-aware inference implementation status

| Field | Value |
| --- | --- |
| Status | Approved design; product implementation not started |
| Last reviewed | 2026-07-14 |
| Repository | `/Users/akiralam/code/ax-serving` |
| Baseline branch | `main` |
| Baseline commit | `7d3636eea9a5552ee9ea868fffa863b1761e02b0` |
| Baseline working tree | Clean before this documentation package |
| Baseline remote CI | CI run `29312964367` failed; CodeQL run `29312963889` passed |
| Peer repository | `/Users/akiralam/code/ax-engine` |
| Peer baseline commit | `fdad304af58707f7f06a9a5ae23acbbc98c0bd27` |
| PRD | [Agent-aware inference fabric](prd/PRD-AGENT-AWARE-INFERENCE-FABRIC.md) |
| ADR | [ADR-015](adr/ADR-015-AGENT-AWARE-INFERENCE-FABRIC.md) |
| Technical spec | [Agent Session Contract v1](specs/TECH-SPEC-AGENT-SESSION-FABRIC-CONTRACT.md) |

This ledger is the handoff point between the AX Serving and AX Engine coders. It records evidence,
not intent. Inspect both repositories directly before trusting any row.

## Existing baseline blockers

The main-branch CI failure predates agent-aware implementation and must be separated from new work:

- Rust formatting failed in the portable gateway jobs.
- deployment YAML parsing failed.
- macOS all-feature clippy failed with `items_after_test_module` in
  `crates/ax-serving-cli/src/support.rs`.

The assigned coder must confirm the live CI result and exact current source before starting. Fixing
these blockers is a prerequisite for trustworthy new CI evidence, but unrelated refactoring must
not be mixed into the agent-session contract unless separately scoped.

## Required session-start audit

Before every coding session that can affect the shared contract, update:

```text
AX Serving HEAD:
AX Serving git status --short:
AX Serving latest CI run/conclusion:
AX Engine HEAD:
AX Engine git status --short:
Peer status ledger last reviewed:
Contract fixture digest/version:
Observed peer implementation relevant to this session:
```

Read `/Users/akiralam/code/ax-engine` directly. Verify its core type, HTTP mapping, capability
metadata, and tests; do not infer them from this ledger or a copied fixture.

## Milestone ledger

| Milestone | State | Required evidence |
| --- | --- | --- |
| S0: Protocol types and public parsing | Not started | Strict header tests, keyed derivation, redaction, shared fixture comparison |
| S1: Request profile and capability negotiation | Not started | Profile/deployment filter tests, optional/required runtime behavior |
| S2: Session affinity store and routing | Not started | memory/Redis TTL tests, HA convergence, eligibility/fallback tests |
| S3: Runtime-agent normalized transport | Not started | header scrub/allowlist, AX Engine capability fixture, SSE/error preservation |
| S4: Diagnostics and observability | Not started | bounded metrics, audit/redaction, response header tests |
| S5: Cross-repo conformance and benchmarks | Not started | exact peer commit, live AX path, baseline/gateway comparison artifacts |

Allowed states are `Not started`, `In progress`, `Blocked`, `Implemented`, and `Certified`.
`Implemented` requires passing local tests. `Certified` additionally requires retained HA,
cross-repository, and performance evidence from the PRD.

## Contract change log

| Date | Change | AX Serving commit | AX Engine commit | Compatibility |
| --- | --- | --- | --- | --- |
| 2026-07-14 | Agent Session Contract v1 design approved | `7d3636e` | `fdad304a` | Initial additive contract |

## Current risks

- Public `x-ax-session-id` is sensitive correlation data. It must be tenant-scoped through the
  configured secret and must not be retained, forwarded, or logged.
- Session affinity is a soft performance preference. If implementation makes it correctness state,
  it violates ADR-015 and the peer AX Engine ADR.
- Existing cache-affinity and new session affinity must share routing signals deliberately; two
  uncoordinated worker-sticky maps would create unpredictable policy.
- Runtime capability is model/deployment-specific. An agent process being new enough does not prove
  that the selected runtime/model honors the v1 headers.
- `FleetStateStore` is already a large compatibility surface. Add session methods with bounded TTL
  and Redis tests; do not place per-request streams or prompt data in shared state.

## End-of-session update

Before handing work back, record:

```text
Completed requirement IDs:
Files/types changed:
Contract fields or semantics changed:
Tests run and exact result:
CI run URL/result:
Benchmark/evidence paths:
Peer repository re-reviewed at commit:
Blockers/risks:
Next concrete step:
```

Any shared field, default, error code, capability name, or fixture change requires a separately
authorized matching AX Engine documentation/status update before the milestone is complete.

