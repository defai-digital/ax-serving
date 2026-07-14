# Technical Specification: AX Serving Agent Session Fabric Contract v1

| Field | Value |
| --- | --- |
| Status | Approved for implementation |
| Last updated | 2026-07-14 |
| Target | AX Serving 3.x additive experimental capability |
| PRD | [Agent-aware inference fabric](../prd/PRD-AGENT-AWARE-INFERENCE-FABRIC.md) |
| Decision | [ADR-015](../adr/ADR-015-AGENT-AWARE-INFERENCE-FABRIC.md) |
| Extends | [Hybrid runtime control-plane spec](TECH-SPEC-HYBRID-RUNTIME-CONTROL-PLANE.md) |
| Evidence status | [Implementation ledger](../AGENTIC-INFERENCE-STATUS.md) |
| Peer spec | `/Users/akiralam/code/ax-engine/.internal/specs/TECH-SPEC-AGENT-SESSION-RUNTIME-CONTRACT.md` |

## 1. Purpose and invariants

This specification defines the AX Serving half of Agent Session Contract v1: public parsing,
tenant-scoped key derivation, request profiling, capability negotiation, soft session affinity,
shared state, normalized runtime-agent transport, diagnostics, and verification.

Release-blocking invariants:

1. Every inference request remains independently replayable with complete model input.
2. Session affinity never bypasses hard eligibility, capacity, policy, deadline, or equivalence.
3. Raw public session IDs leave no request-classification boundary.
4. The gateway never tokenizes prompts, calculates KV hashes, or owns runtime cache contents.
5. Requests without agent headers follow the existing code path and behavior.
6. Existing typed pre-admission and no-retry-after-commitment rules remain authoritative.

## 2. Verified starting point

Extend these current paths rather than creating a parallel gateway:

- `crates/ax-serving-api/src/orchestration/proxy_handlers.rs` parses bounded body/header metadata,
  derives tenant-scoped keyed `x-ax-cache-affinity`, and builds `RequestProfile`.
- `crates/ax-serving-api/src/orchestration/request_profile.rs` contains runtime-neutral admission and
  routing metadata.
- `crates/ax-serving-api/src/orchestration/deployment.rs` performs capability and equivalence
  filtering.
- `crates/ax-serving-api/src/orchestration/policy.rs` implements inference/cache-affinity scoring.
- `crates/ax-serving-api/src/orchestration/direct.rs` owns attempt selection, typed retry,
  streaming, and worker header filtering.
- `crates/ax-serving-api/src/orchestration/fleet_state.rs` provides in-memory and Redis/Valkey shared
  state with fencing/reservation primitives.
- `crates/ax-serving-protocol/src` defines runtime-neutral versioned DTOs and capabilities.
- `crates/ax-thor-agent/src/agent.rs` reports protocol/model capability and
  `crates/ax-thor-agent/src/proxy.rs` controls runtime header forwarding.

Re-inspect all paths and the peer AX Engine code at the session's recorded commits. Do not treat this
snapshot as proof after implementation begins.

## 3. Protocol types and constants

Add `crates/ax-serving-protocol/src/agent_session.rs` with serialization/validation only:

```rust
pub const AGENT_SESSION_CONTRACT_V1: u16 = 1;
pub const AGENT_SESSION_MODEL_CAPABILITY: &str = "inference.agent-session.v1";

#[derive(Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct AgentSessionKey([u8; 16]);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentWorkloadClass {
    #[default]
    Interactive,
    Background,
    ToolResume,
    Subagent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentSessionHintsV1 {
    pub session_key: AgentSessionKey,
    pub workload_class: AgentWorkloadClass,
    pub expected_output_tokens: Option<u32>,
    pub cache_retention_ms: Option<u32>,
    pub require_runtime_support: bool,
}
```

`AgentSessionKey` uses exactly 32 lowercase hex characters at wire/fixture boundaries, rejects zero,
has redacted `Debug`, and has no `Display`. Production logs must not use alternate encoding helpers.

Add header constants to the protocol crate so gateway and runtime agent do not duplicate strings.

## 4. Public contract

Accepted on `/v1/chat/completions` and `/v1/completions`:

| Public header | Required | Validation/default |
| --- | --- | --- |
| `x-ax-agent-contract-version` | Yes when any v1 field is present | exactly `1` |
| `x-ax-session-id` | Yes for v1 | visible ASCII after trim, 1..=256 bytes |
| `x-ax-workload-class` | No | `interactive`, `background`, `tool_resume`, `subagent`; default `interactive` |
| `x-ax-expected-output-tokens` | No | positive `u32`, not above request hard max when present |
| `x-ax-cache-retention-ms` | No | `u32`, clamped to operator maximum |
| `x-ax-runtime-session-required` | No | strict `true` or `false`; default `false` |

Any agent header on embeddings returns HTTP 422. Duplicate/comma-joined values, invalid ASCII,
invalid ranges, missing version/session ID, and unknown versions fail before queue admission with:

```text
AXS_AGENT_CONTRACT_INVALID
AXS_AGENT_CONTRACT_UNSUPPORTED
AXS_AGENT_SESSION_DISABLED
```

Public requests containing `x-ax-agent-session-key` or another reserved normalized internal header
are rejected with `AXS_RESERVED_HEADER`; silently accepting and later overriding them is prohibited.

The public body remains byte-preserved except for the existing model rewrite. Do not add an
`ax_agent` JSON field or parse messages/prompt beyond existing bounded request classification.

## 5. Tenant-scoped key derivation

Reuse the configured `AXS_CACHE_AFFINITY_SECRET` (32..=4096 bytes) with a distinct domain. If it is
absent and `x-ax-session-id` is present, return `AXS_AGENT_SESSION_DISABLED`.

Use the repository's SHA-256 keyed-prefix pattern:

```text
SHA256(
  "ax-serving-agent-session-v1\0" ||
  u64be(secret.len) || secret ||
  u64be(tenant.len) || tenant ||
  u64be(raw_session_id.len) || raw_session_id
)[0..16]
```

The tenant is the post-authenticated/effective project used by existing policy, not an untrusted
forwarded proxy header. Apply exact length prefixes. Add tests proving changes in secret, tenant, or
raw ID change the key and that legacy cache-affinity uses a different domain.

The raw ID must be borrowed only during derivation and dropped before constructing `RequestProfile`.
Never place it in an error message.

## 6. Request profile and capability filtering

Extend `RequestProfile`:

```rust
pub agent_session: Option<AgentSessionHintsV1>,
```

Do not duplicate individual fields at the top level. `AgentSessionHintsV1` contains the derived key,
never the raw session ID.

When `require_runtime_support` is true, insert
`ProtocolCapability::new("inference.agent-session.v1")` into `required_capabilities`. Existing
deployment/model filtering then fails closed.

When false:

- do not make the capability a hard filter;
- use session affinity across otherwise eligible candidates;
- forward normalized runtime hints only when the selected model descriptor advertises the
  capability;
- return a safe diagnostic that runtime hints were unsupported/bypassed.

Capability is per model/deployment. Do not infer it from runtime kind, version string, agent binary
version, or AX Engine branding.

## 7. Configuration

Add to `OrchestratorConfig` with environment and config-file support:

```text
AXS_AGENT_SESSION_ENABLED=false
AXS_AGENT_SESSION_DEFAULT_TTL_MS=30000
AXS_AGENT_SESSION_MAX_RETENTION_MS=120000
AXS_AGENT_SESSION_MAX_BINDINGS=100000
```

Rules:

- `enabled=false` rejects public v1 use with `AXS_AGENT_SESSION_DISABLED`; it does not alter legacy
  requests.
- Default TTL applies when the request omits `cache_retention_ms`.
- Effective binding TTL is `min(requested_or_default, max_retention)`; zero disables binding after
  the current request while still allowing normalized runtime forwarding.
- `max_bindings` applies to memory mode. Redis relies on per-record TTL and must not scan all keys in
  the request path.
- Enabling the feature requires a valid `AXS_CACHE_AFFINITY_SECRET`.

The experimental default remains off until S5 certification. Release packaging may turn it on only
after the PRD gates pass.

## 8. Session affinity store

Define an internal serde type in `fleet_state.rs` or a focused sibling module:

```rust
pub struct SessionBindingRecord {
    pub session_key: AgentSessionKey,
    pub logical_model_id: LogicalModelId,
    pub deployment_id: DeploymentId,
    pub worker_id: WorkerId,
    pub generation: u64,
    pub updated_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}
```

Do not store tenant separately because it is already cryptographically scoped into the key. Do not
store pool/runtime strings, prompt hashes, request IDs, or response data unless a later ADR proves
they are required.

Extend `FleetStateStore` (or a narrow `SessionAffinityStore` implemented by the same backends) with:

```text
get_session_binding(session_key, logical_model_id)
put_session_binding_if_newer(record)
remove_session_binding(session_key, logical_model_id)
```

Requirements:

- Memory store uses bounded deterministic LRU/oldest-expiry eviction and prunes expired records on
  read/write without an unbounded scan per request.
- Redis key includes a versioned prefix and digest/model-safe key; value is versioned JSON or a
  stable compact encoding and is always written with TTL.
- `put_if_newer` rejects a record with older `(updated_at, generation)` to reduce HA stale writes.
- A store timeout/error does not fail inference. It records `store_error`, skips affinity, and uses
  normal eligible routing.
- Bindings are advisory and may disappear at any time.

## 9. Endpoint selection

Add the session key and optional binding to `DispatchContext` without exposing it to generic debug
output. Selection order:

1. Build candidates with existing readiness, lease, drain, model, operation, capability, limit,
   trust, pool, equivalence, and reservation filters.
2. Read the unexpired binding for `(session_key, logical_model)`.
3. If its worker is still in the eligible, non-saturated candidate set and deployment identity is
   compatible, select it as a strong preference.
4. Otherwise call the configured existing dispatch policy. Agent session affinity precedes legacy
   `cache_affinity_key`; legacy affinity remains a secondary scoring signal among fallback
   candidates.
5. Do not persist a provisional selection.
6. After the trusted agent proves admission or a successful response begins, write/update the
   binding with the selected deployment/worker and effective TTL.
7. A typed pre-admission rejection may select a different worker under existing retry rules; only a
   successfully admitted attempt may replace the binding.

Never bypass worker reservation/capacity to preserve affinity. If the bound worker is saturated,
fallback is expected and counted as `rebind_capacity`, not an error.

Concurrent requests with one session key are allowed. Last successfully admitted write wins. V1
does not serialize turns or enforce parent/child ordering; W3C `traceparent` supplies trace lineage.

## 10. Normalized gateway-to-runtime contract

The gateway generates and signs/authenticates the dispatch as today, then adds only for a capable
selected deployment:

| Internal header | Value |
| --- | --- |
| `x-ax-agent-contract-version` | `1` |
| `x-ax-agent-session-key` | 32 lowercase hex |
| `x-ax-agent-workload-class` | normalized enum |
| `x-ax-agent-expected-output-tokens` | optional canonical integer |
| `x-ax-agent-cache-retention-ms` | optional effective/clamped integer |

The public raw session ID and `x-ax-runtime-session-required` are never forwarded.

`ax-runtime-agent` must:

- advertise `inference.agent-session.v1` for a model only when its runtime adapter has discovered
  and conformance-tested support;
- accept normalized headers only on an authenticated dispatch;
- reject duplicates and invalid normalized values rather than forwarding them;
- use an explicit allowlist for the five headers;
- deny any raw public `x-ax-session-id`, public credentials, cookies, proxy credentials, and
  hop-by-hop headers;
- forward request body and runtime SSE bytes unchanged;
- allow safe AX Engine response diagnostics through while continuing to strip admission/dispatch
  secrets.

For AX Engine, capability discovery consumes the AX-specific `/v1/models`/runtime metadata defined
by the peer spec. Pin fixtures to the peer commit. vLLM/SGLang must not advertise the capability
until an adapter-specific mapping is implemented and tested; generic prefix caching alone is not
proof that the custom headers are honored.

## 11. Public diagnostics

Add gateway response headers when v1 is accepted:

| Header | Values |
| --- | --- |
| `x-ax-agent-contract-version` | `1` |
| `x-ax-agent-affinity-result` | `hit`, `miss`, `rebind`, `disabled`, `store_error` |
| `x-ax-agent-runtime-result` | `honored`, `partially_honored`, `unsupported` |

Forward safe runtime headers from AX Engine:

```text
x-ax-agent-cache-result
x-ax-agent-reused-prefix-tokens
x-ax-agent-hint-result
```

Never forward/echo the raw or normalized session key. If a runtime sends a key-bearing header, strip
it. Diagnostics are best effort and must not delay or buffer SSE.

## 12. Observability

Add bounded metrics using the repository naming convention:

```text
ax_serving_agent_requests_total{workload_class,runtime_required}
ax_serving_agent_affinity_total{result,reason}
ax_serving_agent_binding_store_total{operation,result}
ax_serving_agent_runtime_hint_total{result,runtime_kind}
ax_serving_agent_contract_errors_total{code}
```

No session key, request ID, tenant string, model string, worker ID, prompt, tool data, or free-form
header may be a metric label. Use existing bounded worker/deployment diagnostics outside metrics for
operator drill-down.

Trace spans may include workload class, required flag, result enums, and reused-token counts. Use
the existing request/attempt IDs and W3C trace context for correlation; do not add session identity.

## 13. Shared fixtures and tests

Add:

```text
crates/ax-serving-protocol/tests/fixtures/agent-session/v1/
├── valid-minimal.json
├── valid-complete.json
├── invalid-session-keys.json
├── invalid-ranges.json
└── expected-ax-engine-capability.json
```

`valid-complete.json` semantic content must match AX Engine:

```json
{
  "version": 1,
  "session_key": "0123456789abcdef0123456789abcdef",
  "workload_class": "tool_resume",
  "expected_output_tokens": 64,
  "cache_retention_ms": 30000
}
```

The Serving-only public fixture may additionally include `require_runtime_support`; it is not
forwarded to Engine. Record fixture digest/version and both commits in the status ledger.

Required test groups:

- protocol serialization, strict key parsing, redacted debug, unknown future fields;
- public absent/minimal/complete/duplicate/malformed/unknown-version/reserved-header cases;
- tenant/secret/domain-separated derivation and raw-value leak negative tests;
- request profile and optional/required capability filtering;
- binding TTL, capacity bound, expiry, stale write, deployment change, drain, worker death, and
  concurrent requests;
- memory/Redis semantic parity and two-gateway convergence under restart/partition;
- affinity precedence over legacy cache affinity without bypassing capacity;
- retry/commitment behavior and binding update only after trusted admission;
- gateway and runtime-agent header scrub/allowlist tests;
- AX Engine capability fixture and end-to-end JSON/SSE/error/cancel conformance;
- bounded metrics/trace/audit fields and no sensitive values;
- complete existing no-agent regression suite.

## 14. Benchmark and fault contract

Extend the existing gateway benchmark runner with:

- 100 sessions, 10 or more turns per session;
- 8K/32K/supported long shared prefixes and small appended tool results;
- 100 ms, 2 s, 30 s, and 120 s tool waits;
- interactive/background/tool-resume/subagent traffic;
- one and two gateways, memory and Redis stores;
- 32-worker and 256-stream production envelope where available;
- affinity disabled, legacy cache affinity, and Agent Session Contract v1 modes;
- healthy, saturation, drain, worker death, gateway restart, Redis restart/partition, and TTL expiry;
- gateway setup latency, affinity outcome, runtime cache result, TTFT, time-to-next-action, goodput,
  retry/commit outcome, and store latency.

Run the pinned AX Engine trace direct, through one runtime agent, and through AX Serving. Record
model/runtime/peer commit, configuration, raw JSON, and summaries under the existing evidence
layout. The PRD release thresholds are authoritative.

## 15. Implementation sequence and cross-repo handoff

1. **S0:** protocol types/constants, public parser, derivation, redaction, fixtures.
2. **S1:** request profile and required/optional per-model capability behavior.
3. **S2:** memory/Redis session binding store and affinity-first eligible selection.
4. **S3:** normalized dispatcher/runtime-agent headers and AX Engine capability discovery.
5. **S4:** diagnostics, metrics/traces, configuration, operator docs.
6. **S5:** live cross-repo conformance, HA/fault suite, benchmark, release decision.

Before each step, the AX Serving coder must inspect `/Users/akiralam/code/ax-engine`, read its
agentic status ledger, record both HEADs, and verify actual types/capability/tests. If AX Engine has
changed a field, default, capability, error, or fixture, reconcile both specs before continuing. Do
not rely on a copied status row or implement a private translation.

## 16. Required verification commands

At minimum before handoff:

```text
cargo fmt --all -- --check
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --workspace --all-targets --all-features
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --workspace --all-targets --all-features -- -D warnings
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --workspace --all-features
```

Also run the focused Redis conformance, runtime-agent proxy, security/header, orchestration
integration, and cross-repository live AX tests. Record exact commands/results, CI URL, artifacts,
local commit, and peer commit in [`AGENTIC-INFERENCE-STATUS.md`](../AGENTIC-INFERENCE-STATUS.md).
