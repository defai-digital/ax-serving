# ADR-001: Target Market And Product Scope

- Status: Accepted
- Date: 2026-03-28

## Context

AX Serving needs a clear product boundary before further PRD and code work.

Without a clear boundary, the project risks drifting into one of several
misleading categories:

- local end-user AI app
- generic OpenAI-compatible wrapper
- raw inference engine
- hyperscale GPU serving framework

Those categories do not accurately reflect the product being built.

## Decision

AX Serving is positioned as a **department-scale private AI fleet control
plane**.

Target market:

- SMEs
- enterprise departments
- typical deployment scope under roughly 100 users or operators

Target product shape:

- multi-model serving
- multi-worker orchestration
- model lifecycle management
- operational control surfaces
- auth, audit, metrics, diagnostics, and policy visibility

Non-goals:

- consumer desktop chat workflows
- hobby local runtime convenience as the primary value proposition
- generic hyperscale CUDA-first serving as the primary value proposition
- replacing low-level inference engines as the main product mission

## Consequences

Future PRDs and implementation work should prioritize:

- fleet operation over single-user convenience
- control-plane capabilities over raw engine differentiation
- mixed-worker operability over generic platform breadth
- product clarity for team-operated private AI deployments

Future PRDs and implementation work should avoid:

- drifting into desktop-app positioning
- claiming raw hardware economics as the core software value
- optimizing first for hyperscale cluster narratives
