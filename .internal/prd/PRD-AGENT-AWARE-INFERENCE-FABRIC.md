# Product Requirements: AX Serving Agent-Aware Inference Fabric

| Field | Value |
| --- | --- |
| Status | Approved implementation target |
| Owner | AX Serving maintainers |
| Last updated | 2026-07-14 |
| Product position | Runtime-neutral private agent-aware inference fabric |
| Extends | [AX Serving PRD](PRD-AX-SERVING.md), [ADR-013](../adr/ADR-013-RUNTIME-NEUTRAL-HYBRID-INFERENCE-CONTROL-PLANE.md) |
| Decision | [ADR-015](../adr/ADR-015-AGENT-AWARE-INFERENCE-FABRIC.md) |
| Technical specification | [Agent Session Contract v1](../specs/TECH-SPEC-AGENT-SESSION-FABRIC-CONTRACT.md) |
| Evidence status | [Agent-aware status](../AGENTIC-INFERENCE-STATUS.md) |
| Peer runtime | `/Users/akiralam/code/ax-engine` |

## 1. Decision summary

AX Serving will become an **agent-aware inference fabric**: a runtime-neutral control plane that can
recognize a bounded agent session, keep sequential turns near useful runtime-local cache, negotiate
runtime support, preserve priorities/deadlines, and expose end-to-end session performance signals.

It will not become an agent orchestrator. Planning, tool/MCP execution, permissions, sandboxes,
handoffs, durable memory, and workflow/task lifecycle remain in the external harness. Every
inference request remains independently replayable with complete model input. Session affinity is a
performance optimization, never hidden correctness state.

The shared boundary with AX Engine is **Agent Session Contract v1**. AX Serving accepts a raw
client session ID at its authenticated public boundary, derives a tenant-scoped opaque key, and
forwards only normalized performance hints to a compatible runtime through `ax-runtime-agent`.

## 2. Market and architecture rationale

Agent frameworks increasingly own stateful orchestration themselves. The
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/sessions/) stores session history
and its [tracing model](https://openai.github.io/openai-agents-python/tracing/) covers runs, tool
calls, handoffs, and guardrails. [MCP](https://modelcontextprotocol.io/specification/2025-06-18/server/index)
defines application/model primitives for tools, resources, and prompts, while
[A2A 1.0](https://a2a-protocol.org/latest/specification/) defines agent task and context lifecycle.
Those layers should integrate with AX Serving, not be reimplemented inside it.

Inference infrastructure is moving in a complementary direction. NVIDIA Dynamo describes an
agent-native interface across frontend hints, KV-aware routing, priority scheduling, and selective
cache retention; its [KV router](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)
scores cache overlap and load rather than round-robin placement. vLLM documents automatic prefix
caching as a direct optimization for repeated multi-round conversations. These trends validate the
AX portfolio split:

- AX Engine optimizes inference execution and runtime-local KV/prefix state.
- AX Serving optimizes admission, capability-safe placement, session affinity, and fleet behavior.
- The harness owns the agent and its durable semantics.

## 3. Users and jobs

### 3.1 Agent platform developer

Send one opaque session ID and workload hints through an OpenAI-compatible API without knowing the
worker, runtime, or cache implementation. Require runtime support when the workload depends on a
specific performance capability.

### 3.2 Private fleet operator

Run agent workloads across Apple Silicon and CUDA pools, retain security boundaries, observe
affinity/cache outcomes, and drain or fail over workers without corrupting conversations.

### 3.3 Runtime integrator

Advertise a versioned per-model capability, receive normalized opaque hints, and return safe
diagnostics without exposing runtime cache contents or linking the gateway to an SDK.

## 4. Goals

### 4.1 P0 goals

- Define and validate the public Agent Session Contract v1 header surface.
- Derive a 128-bit tenant-scoped opaque session key with the existing operator secret; never retain
  or forward the raw session ID.
- Extend `RequestProfile` with typed workload, expected output, retention, and runtime-support
  requirement fields.
- Prefer an eligible prior worker for sequential turns while preserving readiness, drain, capacity,
  deployment identity, equivalence, deadline, and retry rules.
- Provide bounded in-memory and Redis/Valkey session-affinity records for single-gateway and HA
  deployments.
- Negotiate the per-model capability `inference.agent-session.v1`.
- Forward normalized internal headers only through a trusted runtime agent to a capable runtime.
- Provide safe response diagnostics, bounded metrics, and trace correlation.
- Preserve identical behavior for clients that do not send agent headers.
- Pass cross-repository conformance with the exact AX Engine commit recorded in both ledgers.

### 4.2 P1 goals

- Consume block-level KV lifecycle events from compatible runtimes for exact cache-aware placement,
  with prediction/TTL fallback when events are unavailable.
- Add measured workload-class-aware queue policy without starving background work.
- Support session-affinity policy across larger gateway/router topologies and Gateway API Inference
  Extension integrations.
- Surface tool-call and structured-output conformance across runtime pools.

### 4.3 P2 research

- Cross-runtime cache transfer or portable continuation handles.
- Cost/SLO policy across local Apple Silicon and remote accelerators using agent-loop objectives.
- Durable asynchronous inference operations, governed by a separate public API and ADR.

## 5. Non-goals

- Running the agent loop, planner, tools, MCP servers, A2A tasks, sandboxes, or approvals.
- Storing conversation history, prompts, tool results, artifacts, or durable workflow memory.
- Tokenizing or rendering prompts in the gateway to calculate cache hashes.
- Owning KV blocks, prefix caches, continuous batching, speculation, or model output parsing.
- Keeping public HTTP streams open during tool execution.
- Making session affinity a correctness requirement or rejecting safe failover solely because a
  cache is cold.
- Migrating KV state between AX Engine and vLLM/SGLang in P0.
- Replacing existing priority, tenant, deadline, typed admission, commitment, or equivalence rules.

## 6. Functional requirements

| ID | Priority | Requirement |
| --- | --- | --- |
| ASF-001 | P0 | Accept v1 only on chat/completions and completions using the exact public headers in the technical specification. |
| ASF-002 | P0 | Require `x-ax-agent-contract-version: 1` and `x-ax-session-id` when any v1 hint is present; reject unknown versions and malformed/duplicate values with stable errors. |
| ASF-003 | P0 | Derive the opaque session key from secret, tenant, and raw ID with domain separation. If no secret is configured, reject session use instead of hashing unkeyed or forwarding raw data. |
| ASF-004 | P0 | Raw session IDs never leave request classification and never appear in logs, traces, metrics, audits, fleet state, dispatch headers, or responses. |
| ASF-005 | P0 | Extend request classification without parsing prompt content or changing the byte-preserving unknown-field behavior. |
| ASF-006 | P0 | If runtime support is required, eligible deployments must advertise `inference.agent-session.v1`; absence produces a stable no-compatible-deployment error. |
| ASF-007 | P0 | If runtime support is optional, Serving may use session affinity on any eligible deployment but forwards normalized runtime hints only to a deployment that advertises v1. |
| ASF-008 | P0 | A live compatible session binding is a strong preference but never bypasses hard filters, active capacity, tenant policy, or an operator pool pin. |
| ASF-009 | P0 | If a bound worker is unavailable, draining, stale, incompatible, or saturated, select another eligible worker under existing policy and update the binding only after trusted admission. |
| ASF-010 | P0 | Store only the opaque key, logical/deployment identity, worker ID, timestamps, and generation in a bounded TTL record. |
| ASF-011 | P0 | Single-gateway memory and HA Redis stores implement the same expiry and last-writer/version semantics. |
| ASF-012 | P0 | Gateway-generated normalized headers cannot be overridden by public client headers; runtime-agent forwarding uses an explicit allowlist and preserves body/SSE bytes. |
| ASF-013 | P0 | Existing no-retry-after-commitment and at-most-one typed pre-admission retry rules remain unchanged. A retry may reselect/rebind only before commitment. |
| ASF-014 | P0 | Responses indicate affinity and runtime-hint results without returning the raw or opaque session key. |
| ASF-015 | P0 | Metrics and traces distinguish affinity hit/miss/rebind, runtime capability, and cache diagnostics using bounded labels. |
| ASF-016 | P0 | Existing `x-ax-cache-affinity` remains backward compatible. When both hints exist, agent session affinity is evaluated first and legacy cache affinity remains a secondary score. |
| ASF-017 | P1 | Exact KV events are authenticated, deployment-scoped, bounded, expiry-aware, and cannot contain prompts or token contents at the gateway. |

## 7. Quality and security requirements

| ID | Requirement |
| --- | --- |
| ASF-NFR-001 | Agent-header parsing and session lookup add less than 100 microseconds p99 gateway setup overhead in the 256-candidate benchmark. |
| ASF-NFR-002 | With agent headers absent, gateway goodput and setup latency remain within 3% of the same commit's baseline. |
| ASF-NFR-003 | A healthy, compatible, non-saturated bound worker is selected for at least 99% of sequential-turn requests within TTL. |
| ASF-NFR-004 | Memory store capacity is bounded and eviction deterministic; Redis records always have TTL and no global key scan in the request path. |
| ASF-NFR-005 | Two active gateways converge on the latest successful binding and remain correct under Redis restart, partition, expiry, and stale writes. |
| ASF-NFR-006 | Fuzz malformed/duplicate headers, large IDs, integer boundaries, and client attempts to inject normalized internal headers. |
| ASF-NFR-007 | No metric label or default log contains raw/opaque session IDs, prompt data, tool payloads, or free-form headers. |
| ASF-NFR-008 | Cross-repository contract fixtures are semantically identical at the commits recorded in both status ledgers. |

## 8. Success measures and release gates

The feature may merge behind configuration, but “agent-aware inference fabric” is a supported claim
only after:

1. All P0 requirements pass unit, integration, Redis, runtime-agent, and security tests.
2. Main-branch baseline CI blockers recorded in the status ledger are resolved and the exact feature
   commit is green on portable Linux and macOS compatibility jobs.
3. A 100-session, 10-turn mixed agent-loop benchmark demonstrates at least 25% lower p95
   time-to-next-action for cache-eligible sequential turns versus session affinity disabled.
4. Non-agent gateway overhead/goodput regression is no more than 3%.
5. Affinity selection reaches 99% under healthy capacity and degrades safely under saturation,
   drain, expiry, worker death, gateway restart, and Redis failure.
6. A pinned AX Engine path passes public gateway -> runtime agent -> AX Engine conformance, including
   capability, normalized headers, SSE, cancellation, and diagnostics.
7. Security tests prove raw public IDs and internal keys are absent from exported logs, metrics,
   traces, errors, and fleet records not explicitly designed for the opaque key.
8. Public docs state that sessions are replayable performance hints, not server-side agent memory.

## 9. Delivery sequence

| Milestone | Outcome |
| --- | --- |
| S0 | Protocol types/constants, public parser, keyed derivation, fixture, redaction |
| S1 | Request profile, required/optional capability negotiation, deployment filtering |
| S2 | Memory/Redis binding store and affinity-first eligible routing |
| S3 | Gateway/runtime-agent normalized transport and AX Engine capability discovery |
| S4 | Safe diagnostics, metrics, trace/audit rules, operator configuration |
| S5 | Cross-repo conformance, HA/fault tests, benchmark, public release decision |

At each milestone the coder must update the [status ledger](../AGENTIC-INFERENCE-STATUS.md), inspect
the actual peer implementation at `/Users/akiralam/code/ax-engine`, and record both commits.

