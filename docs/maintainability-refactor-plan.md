# AX Serving Maintainability Refactor Plan

> **Status**: Draft  
> **Date**: 2026-03-29  
> **Type**: Engineering Refactor Plan  
> **Scope**: Supports [docs/prd/PRD-AX-SERVING-v3.0.md](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-v3.0.md) without changing product positioning

## 1. Purpose

This plan defines the refactor work needed to reduce code duplication, shrink
oversized modules, and improve maintainability before significant new v3.0
feature work begins.

It is intentionally an engineering plan, not a new product PRD. The canonical
product PRD remains:

- [docs/prd/PRD-AX-SERVING-v3.0.md](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-v3.0.md)

## 2. Problem Statement

The current codebase already demonstrates the product shape we want, but it is
carrying several maintainability risks:

- Large multi-responsibility modules make changes slow and risky.
- Similar request-shaping, audit, reporting, and helper logic exists in more
  than one serving surface.
- Test support code is duplicated across unit and integration suites.
- CLI command handling is too centralized, which makes the operational surface
  harder to extend cleanly.
- Some files are now large enough that ownership boundaries are unclear.

If left as-is, v3.0 feature delivery will slow down and regressions will become
harder to contain.

## 3. Current Baseline

Snapshot from 2026-03-29:

### Largest production modules

- `crates/ax-serving-api/src/rest/routes.rs`: 3487 LOC
- `crates/ax-serving-api/src/orchestration/registry.rs`: 1782 LOC
- `crates/ax-serving-api/src/scheduler.rs`: 1252 LOC
- `crates/ax-serving-cli/src/main.rs`: 1233 LOC
- `crates/ax-serving-api/src/registry.rs`: 1229 LOC
- `crates/ax-serving-api/src/orchestration/mod.rs`: 1228 LOC
- `crates/ax-serving-api/src/config.rs`: 1125 LOC
- `crates/ax-serving-cli/src/thor.rs`: 1014 LOC
- `crates/ax-serving-api/src/orchestration/policy.rs`: 1005 LOC

### Largest test modules

- `crates/ax-serving-api/tests/model_management.rs`: 3328 LOC
- `crates/ax-serving-api/tests/orchestration.rs`: 2861 LOC

### Confirmed duplication / maintainability hotspots

- Prompt-token estimation logic exists in both:
  - `crates/ax-serving-api/src/rest/routes.rs`
  - `crates/ax-serving-api/src/orchestration/mod.rs`
- Audit helper patterns exist in both:
  - `crates/ax-serving-api/src/rest/routes.rs`
  - `crates/ax-serving-api/src/orchestration/mod.rs`
- Startup-report shaping exists in both:
  - `crates/ax-serving-api/src/rest/routes.rs`
  - `crates/ax-serving-api/src/orchestration/mod.rs`
- Test backends and GGUF test fixtures are duplicated in:
  - `crates/ax-serving-api/src/registry.rs`
  - `crates/ax-serving-api/tests/model_management.rs`
  - `crates/ax-serving-api/tests/graceful_shutdown.rs`
- Environment-variable test locking appears in:
  - `crates/ax-serving-api/src/config.rs`
  - `crates/ax-serving-api/src/license.rs`
  - `crates/ax-serving-api/tests/orchestration.rs`
  - `crates/ax-serving-engine/src/ax_engine.rs`

## 4. Goals

- Reduce duplication in shared serving/orchestration helpers.
- Split oversized files into modules with clear ownership.
- Centralize test support primitives used repeatedly across suites.
- Make CLI command flow easier to read and extend.
- Keep behavior stable while improving internal structure.
- Establish size and structure guardrails so the codebase does not drift back.

## 5. Non-Goals

- No product repositioning or PRD rewrite.
- No API contract redesign unless required for pure internal extraction.
- No scheduler-policy redesign in this refactor plan.
- No backend-engine rewrite of `llamacpp.rs` or `libllama.rs` in this phase.
- No performance-tuning project beyond parity checks required for safe refactor.

## 6. Refactor Principles

- Behavior-preserving first, structure-changing second.
- Extract shared logic before rewriting call paths.
- Prefer module decomposition over clever abstractions.
- Avoid introducing generic frameworks that obscure the serving/control-plane
  domain.
- Keep public interfaces stable unless a clear maintainability gain justifies a
  targeted internal API change.

## 7. Target Architecture Outcomes

By the end of this refactor, the codebase should move toward these boundaries.

### 7.1 REST surface

`crates/ax-serving-api/src/rest/routes.rs` should stop being the single place
for inference, model lifecycle, admin, audit, license, dashboard, metrics, and
cache behavior.

Target split:

- `rest/inference.rs`
- `rest/models.rs`
- `rest/admin.rs`
- `rest/license.rs`
- `rest/cache.rs`
- `rest/reporting.rs`

`rest/mod.rs` should remain router composition only.

### 7.2 Orchestrator surface

`crates/ax-serving-api/src/orchestration/mod.rs` should become composition and
bootstrap logic, not a second giant route implementation file.

Target split:

- `orchestration/proxy.rs`
- `orchestration/admin.rs`
- `orchestration/fleet.rs`
- `orchestration/reporting.rs`
- `orchestration/request_shape.rs`

### 7.3 Shared serving/orchestration helpers

Shared internal helpers should move into focused modules instead of being copied
between serving and orchestrator paths.

Initial candidates:

- request metadata parsing
- prompt-token estimation
- audit actor helpers
- startup-report helpers
- reusable JSON/error response helpers

### 7.4 CLI composition

`crates/ax-serving-cli/src/main.rs` should be reduced to CLI parsing and command
dispatch only.

Target split:

- `cli/commands/infer.rs`
- `cli/commands/serve.rs`
- `cli/commands/worker.rs`
- `cli/commands/thor.rs`
- `cli/runtime.rs`
- `cli/host_info.rs`

The exact module names can vary, but command ownership must be explicit.

### 7.5 Test support

Shared testing support should be centralized instead of reimplemented.

Target structure:

- `crates/ax-serving-api/tests/common/`
- `crates/ax-serving-api/src/test_support.rs` or `src/test_support/` gated by `cfg(test)`
- equivalent support for engine tests only if truly shared

Initial shared fixtures:

- `NullBackend`
- `FailingUnloadBackend`
- temp GGUF fixture builder
- env-var lock / scoped env helpers
- serving-layer builders for integration tests

## 8. Workstreams

### WS-1: Decompose REST handlers

Scope:

- Split `rest/routes.rs` by responsibility.
- Keep route registration in `rest/mod.rs`.
- Extract common helper functions out of route handlers.

Primary outcomes:

- inference handlers separated from admin/model-management code
- smaller review radius per change
- clearer ownership for future features

### WS-2: Decompose orchestrator public surface

Scope:

- Split `orchestration/mod.rs` into routing, proxying, admin, fleet, and
  reporting modules.
- Extract request-shape parsing and prompt estimation to shared helpers.

Primary outcomes:

- less duplicated logic across worker proxy entry points
- cleaner boundary between dispatcher logic and HTTP surface

### WS-3: Centralize duplicated helper logic

Scope:

- Move prompt estimation into a shared internal helper module.
- Move audit actor / audit-query defaults into reusable helper modules.
- Normalize startup-report construction patterns where practical.
- Consolidate repeated response-construction helpers.

Primary outcomes:

- fewer cross-surface inconsistencies
- easier test coverage for shared behavior

### WS-4: Build shared test support

Scope:

- Extract duplicated backend test doubles.
- Extract temp GGUF fixture creation.
- Extract env-lock / scoped env mutation helpers.
- Split giant integration tests into thematic files.

Primary outcomes:

- smaller tests with clearer purpose
- lower maintenance cost for new test coverage
- fewer inconsistent test helpers

### WS-5: Reduce CLI command sprawl

Scope:

- Move command implementations out of `main.rs`.
- Keep clap argument declarations near dispatch, but move operational logic into
  submodules.
- Reuse shared worker-registration / heartbeat / drain helpers where applicable.

Primary outcomes:

- shorter main entrypoint
- clearer operational code ownership
- safer future additions to worker and Thor flows

### WS-6: Add maintainability guardrails

Scope:

- Define file-size guardrails for production and test modules.
- Add a lightweight check or script to report oversized Rust files.
- Document expectations for future extractions in code review.

Primary outcomes:

- the codebase does not regress into new giant files
- maintainability becomes an explicit engineering constraint

## 9. Phased Delivery Plan

### Phase 0: Baseline and safety net

- confirm current test and build baseline
- identify target file splits and ownership
- list all reused helpers before moving code

Exit criteria:

- build/test baseline recorded
- module split map agreed

### Phase 1: Shared helper extraction

- extract prompt estimation
- extract audit helper primitives
- extract response/reporting helpers that are obviously shared

Exit criteria:

- no duplicated prompt-estimation logic remains in serving + orchestrator paths
- shared helpers have direct unit coverage

### Phase 2: REST and orchestrator decomposition

- split `rest/routes.rs`
- split `orchestration/mod.rs`
- keep behavior stable with existing tests

Exit criteria:

- route composition files are visibly smaller and focused
- diff risk per module is reduced

### Phase 3: Test refactor

- introduce shared test support
- split giant integration test files by topic
- remove redundant fixture implementations

Exit criteria:

- shared test fixtures are reused by multiple suites
- integration test files are thematic and smaller

### Phase 4: CLI and config cleanup

- split `main.rs`
- optionally split `config.rs` parsing by concern if Phase 1-3 reveal strong value

Exit criteria:

- `main.rs` is primarily argument parsing and dispatch
- config parsing is easier to navigate and review

## 10. Acceptance Criteria

This plan is complete when all of the following are true:

- The canonical product PRD remains unchanged:
  - [docs/prd/PRD-AX-SERVING-v3.0.md](/Users/akiralam/code/ax-serving/docs/prd/PRD-AX-SERVING-v3.0.md)
- No single production Rust file exceeds 1500 LOC, except engine adapter files
  explicitly exempted for FFI/backend reasons.
- No single integration-test Rust file exceeds 1500 LOC.
- Prompt estimation is implemented once and reused.
- Audit helper logic is no longer duplicated across serving and orchestrator
  route layers where the behavior is the same.
- Shared backend test doubles and GGUF test fixtures are reused from a single
  support location.
- `cargo fmt --all`, `cargo clippy --workspace --tests -- -D warnings`, and
  `cargo test --workspace` pass after the refactor.
- Existing API and orchestration behavior remain compatible with the v3.0
  product scope.

## 11. Risks

- Over-abstracting too early can make the code harder to understand than the
  current duplication.
- Refactoring giant route files without strong tests can create subtle behavior
  regressions.
- Moving shared test support into the wrong layer can make tests more coupled
  rather than less.
- Chasing every duplication pattern at once will stall delivery.

## 12. Recommended Execution Order

1. Extract the obviously shared helper logic first.
2. Split REST and orchestrator route files second.
3. Refactor tests only after the production seams are stable.
4. Split CLI command logic after the API/orchestrator shape is cleaner.
5. Add guardrails last so the new structure becomes the default expectation.

## 13. Immediate Next Tasks

Recommended first implementation batch:

1. Create a shared internal helper for prompt estimation used by both REST and
   orchestrator request parsing.
2. Extract audit helper primitives shared by serving and orchestrator surfaces.
3. Split `rest/routes.rs` into `inference`, `models`, and `admin` modules.
4. Create `tests/common/` fixtures for `NullBackend`, temp GGUF files, and env
   locking.
5. Split `crates/ax-serving-api/tests/model_management.rs` into smaller suites.
