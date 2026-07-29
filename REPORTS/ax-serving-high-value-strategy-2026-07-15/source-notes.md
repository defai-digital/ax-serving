# AX Serving high-value strategy — source and method notes

## Decision frame

- Audience: AX Serving product and technical leadership.
- Decision: whether AX Serving can create enough user value to justify continued investment, especially when an operator could deploy an all-CUDA vLLM stack.
- Baseline: direct vLLM or the vLLM Production Stack for homogeneous CUDA fleets.
- Date reviewed: 2026-07-15.
- Recommendation status: provisional because no customer-interview dataset, production usage telemetry, willingness-to-pay evidence, or retained mixed-fleet TCO benchmark was available.

## Evidence-maturity rubric

The report chart is an evidence audit, not a product score, market size, or customer-adoption measure.

- 0 — no retained evidence found in the reviewed repository/source set.
- 1 — hypothesis or documented plan exists, but no comparative validation was found.
- 2 — implemented or locally verified evidence exists, but production/customer validation is incomplete.
- 3 — production- or customer-validated evidence exists and is retained in an auditable artifact.

## Reviewed evidence and assigned levels

| Dimension | Level | Basis |
| --- | ---: | --- |
| Architecture and runtime boundary | 2 | The canonical PRD and implementation ledger describe a runtime-neutral gateway, protocol, fleet state, routing, identity and safety boundaries as implemented in source; release certification remains pending. |
| Local correctness and contract tests | 2 | The implementation ledger records passing local source checks and tests, while explicitly distinguishing them from live runtime certification. |
| Live AX Engine plus CUDA certification | 0 | The implementation ledger lists pinned AX Engine and CUDA runtime conformance as a release blocker. |
| Mixed-fleet cost and SLO advantage | 0 | The PRD requires direct, through-gateway and mixed-fleet evidence; the implementation ledger says no qualifying goodput or mixed-fleet artifact exists. |
| Customer demand and willingness to pay | 0 | No customer interview, signed design-partner, usage, purchasing or willingness-to-pay dataset was found in the reviewed source set. This means “not evidenced here,” not “no demand exists.” |
| Differentiation versus all-CUDA alternatives | 1 | The repository documents a mixed-runtime and semantic-safety wedge, but vLLM Production Stack already offers Kubernetes deployment, routing, observability, autoscaling and semantic-router integration. vLLM also exposes a hardware-plugin architecture, and the vLLM project now hosts a community-maintained Apple Silicon plugin backed by MLX. No retained win/loss or comparative adoption evidence was found. |

## Strategic segment table method

The segment table is a recommendation synthesized from the repository's stated strong/poor fits, the official vLLM Production Stack feature set, and the absence of customer/economic validation. It is not observed market segmentation.

## Chart map

- Section: engineering evidence versus value proof.
- Question: which major claims have retained evidence at the review date?
- Family/type: comparison, horizontal bar.
- Fields: `dimension`, `evidence_level`, `evidence_label`, `basis`.
- Scale: ordinal evidence maturity from 0 to 3, rendered numerically for comparison.
- Palette: single-root preferred; no redundant color grouping or legend.
- Caveat: the chart measures evidence availability, not market attractiveness.

## Executive-report structure mapping

- Title: `AX Serving 的高價值產品策略`.
- Executive Summary: direct recommendation and product principle.
- Key findings with visual evidence: evidence-maturity chart, competitive substitution analysis and segment decision table.
- Recommended next steps: staged product scope plus provisional go/no-go gates.
- Further questions: customer, workload, budget-owner and substitution questions.
- Caveats and assumptions: missing demand/TCO evidence and limits of vendor claims.

## Primary sources

- `README.md` — canonical product boundaries, goals, release gates and benchmark policy.
- `docs/deployment-topologies.md` — current implementation and certification ledger.
- `docs/competitive-landscape.md` — intended wedge and explicit poor fits.
- `docs/advantages-and-use-cases.md` — strong fits, poor fits and tradeoffs.
- vLLM Production Stack official documentation — current all-CUDA substitute capabilities.
- vLLM Semantic Router official documentation — model selection, PII/prompt guard, semantic cache and tool-selection scope.
- vLLM installation documentation and `vllm-project/vllm-metal` — hardware-plugin strategy and current Apple Silicon/MLX expansion.
- Apple M3 Ultra/Mac Studio newsroom material — official unified-memory capability claim; not independent performance or TCO evidence.
