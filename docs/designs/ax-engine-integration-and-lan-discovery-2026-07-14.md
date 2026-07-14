# Design: AX Engine integration best practices + LAN discovery

| Field | Value |
| --- | --- |
| Status | Active (phase-1 implemented) |
| Date | 2026-07-14 |
| Owners | AX Serving + AX Engine |
| Related | `docs/contracts/ax-serving-runtime-responsibility-inventory.md`, `ax-engine/docs/LOCAL-ENGINE-CLIENTS.md` |

## Problem

1. **Integration model drift.** AX Serving still embeds `ax-engine-sdk` (pinned at `v4.10.0`) for Mac native inference, while both products’ target architecture is **gateway + HTTP runtime node**. The embedded path violates engine thread-ownership guidance and lags engine releases.
2. **LAN discovery gap.** Operators must hard-code `AXS_CONTROL_PLANE_URL` and `AXS_NODE_RUNTIME_URL`. On a home or lab LAN with several Macs running `ax-engine-server`, zero-config discovery is missing.
3. **Exo comparison.** [exo](https://github.com/exo-explore/exo) auto-discovers peers and shards one model across devices (libp2p mesh + topology-aware parallel). AX’s product boundary is different: **whole-model workers** behind a **control-plane gateway**, not tensor/pipeline sharding of a single graph.

## Goals

- Make **HTTP runtime-node** the default recommended Mac path; keep embedded SDK as `embedded-compat` only.
- Add **opt-in LAN discovery** so Serving can find Engine (and agents can find gateways) without hand-written IPs.
- Stay compatible with existing worker register/heartbeat contracts (no link-local advertise URLs).
- Learn from exo’s UX (zero config join) without adopting its distributed execution model.

## Non-goals (phase 1)

- Model sharding / tensor parallel across Macs (exo’s core product).
- Auto-join without operator consent on untrusted networks.
- Replacing Redis fleet state or multi-region mesh.
- Silently bumping the embedded SDK pin to 6.x without a certification matrix.

## Residual: embedded SDK pin

`ax-serving-engine` remains pinned to **ax-engine-sdk git tag `v4.10.0`**.
A best-effort bump to workspace 6.9 was **not** completed in this workstream:
the pin needs a Mac certification matrix (load/stream/embed/cancel) and API
field migration (`multimodal_inputs`, etc.). Production Mac path is HTTP
`ax-engine-server` + agent, which does not use this pin.

## Key decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Runtime ownership | Engine process owns session, templates, batching, Metal thread | Matches engine `NativeGenerationService` contract |
| Serving role | Portable gateway + agent proxy only for production Mac | Matches runtime responsibility inventory |
| Discovery transport | **DNS-SD / mDNS** (`_ax-engine._tcp`, `_ax-serving-gateway._tcp`) | Native on macOS, no central broker, Apple-friendly |
| Trust model | Discovery **never** carries secrets; join is opt-in | mDNS is unauthenticated multicast |
| Advertise address | Prefer **private unicast IPv4** (RFC1918); reject link-local for fleet register | Node contract forbids link-local advertise URLs |
| Exo vs AX | Peer **discovery** yes; peer **model sharding** no | Different product; fleet routes whole models |
| Cluster isolation | Optional `cluster` / `namespace` TXT label | Analogous to exo’s `EXO_LIBP2P_NAMESPACE` |
| Embedded path | Quarantine + warn; prefer agent | Avoid deepening wrong thread model |

## Architecture

```text
                    ┌─────────────────────────────┐
  Clients ─────────►│ ax-serving-api (gateway)    │
                    │  mDNS: _ax-serving-gateway  │  (opt-in)
                    └──────────────┬──────────────┘
                                   │ register / heartbeat / dispatch
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ax-runtime-agent     ax-runtime-agent      (explicit config)
     mDNS browse engine   explicit URLs
              │
              ▼
     ax-engine-server
     mDNS: _ax-engine._tcp   (opt-in --advertise-lan)
     GET /v1/discovery       (verify after browse)
```

### Layers of discovery (ordered)

1. **Explicit config** (production default): env / YAML URLs.
2. **mDNS browse** (lab / home LAN): opt-in flags and `ax-servingctl discover`.
3. **HTTP verify**: after browse, `GET {base}/v1/discovery` (engine) or `/health` (gateway) before trust.
4. **Control-plane register**: only after verify; uses existing protocol crate.

Exo’s “everyone is a peer” is mapped to “everyone can *find* peers”; **routing authority stays at the gateway**.

## Wire contract: DNS-SD

### Service types

| Service | Type | Advertiser |
| --- | --- | --- |
| Engine runtime | `_ax-engine._tcp` | `ax-engine-server --advertise-lan` |
| Serving gateway | `_ax-serving-gateway._tcp` | `ax-serving-api --advertise-lan` (phase 1 optional) |

### TXT keys (ASCII, no secrets)

| Key | Example | Meaning |
| --- | --- | --- |
| `proto` | `1` | Discovery protocol major |
| `kind` | `ax_engine` / `ax_serving_gateway` | Role |
| `version` | `6.9.0` | Product version |
| `model` | `qwen3` | Primary model id if any |
| `auth` | `required` / `open` | Whether API key required |
| `scheme` | `http` | Base URL scheme |
| `path` | `/v1` | API root hint |
| `cluster` | `home-lab` | Optional isolation namespace |
| `instance` | uuid | Process instance |
| `platform` | `macos-aarch64` | Host class |

Instance name: hostname or operator-provided `--lan-instance-name`.

### HTTP verify: `GET /v1/discovery` (engine)

Unauthenticated, small JSON (no model weights paths with home dirs if redaction applies; no secrets):

```json
{
  "schema": "ax.engine.discovery.v1",
  "service": "ax-engine-server",
  "version": "6.9.0",
  "model_id": "qwen3",
  "auth_required": true,
  "openai_base": "http://192.168.1.20:8080/v1",
  "operations": ["chat_completions", "completions", "embeddings"],
  "cluster": "home-lab",
  "instance_id": "..."
}
```

## Operator UX

```bash
# Mac Studio (engine)
ax-engine-server --host 0.0.0.0 --port 8080 \
  --mlx --mlx-model-artifacts-dir /models/qwen \
  --api-key "$AX_ENGINE_API_KEY" \
  --advertise-lan --lan-cluster home-lab

# Laptop (ops)
ax-servingctl discover --timeout-secs 3 --cluster home-lab --json

# Runtime agent next to engine (or on same LAN)
AXS_DISCOVER_LAN=1 \
AXS_DISCOVER_LAN_CLUSTER=home-lab \
AXS_CONTROL_PLANE_URL=http://gateway:19090 \   # or also discoverable later
AXS_NODE_RUNTIME=ax_engine \
AXS_RUNTIME_API_KEY=$AX_ENGINE_API_KEY \
ax-runtime-agent
```

When `AXS_NODE_RUNTIME_URL` is unset and `AXS_DISCOVER_LAN=1`, the agent browses for `_ax-engine._tcp`, filters by cluster, verifies `/v1/discovery`, and picks a single candidate (error if zero or many without `AXS_DISCOVER_LAN_INSTANCE`).

## Security

- mDNS is spoofable; never auto-register into production without `trusted_mesh` + tokens.
- Prefer `auth=required` when advertising on non-loopback.
- Refuse to use discovered link-local or multicast IPs for worker `advertise_url`.
- Discovery results expire (TTL); re-browse on agent startup and optional periodic refresh (phase 2).

## Integration best practices (normative)

1. **Production Mac inference** = `ax-engine-server` + `ax-runtime-agent` + portable `ax-serving-api`. Do not link `ax-engine-sdk` into the gateway binary.
2. **Pin** engine binary min version for agents; pin SDK **tag** only for `embedded-compat` and document the certified commit.
3. **Embedded path** must not invent a third thread model: if kept, own session on one worker thread (engine-server pattern). Until then, treat embedded as migration-only.
4. **Chat templates / tokenize** live in engine; serving gateway and agents proxy OpenAI bytes.
5. **LAN discovery** is for lab/bootstrap; production fleets use explicit registration and fleet store.

## PR Plan

| PR | Repo | Scope |
| --- | --- | --- |
| PR-E1 | ax-engine | `/v1/discovery`, `--advertise-lan`, docs |
| PR-S1 | ax-serving | `ax-serving-discovery` crate, `ax-servingctl discover` |
| PR-S2 | ax-serving | Agent `AXS_DISCOVER_LAN*` resolution for runtime URL |
| PR-S3 | ax-serving | Gateway optional mDNS advertise (follow-up) |
| PR-E2 | ax-serving | Embedded SDK pin certification to engine 6.x (follow-up) |
| PR-E3 | ax-serving | Dedicated owner thread for embedded path or remove native load for multi-request |

## Open questions

- Should gateway auto-spawn agents for discovered engines, or only print join commands? **Phase 1: print + agent resolve only.**
- Multi-engine same cluster: sticky selection by instance name vs least-loaded? **Phase 1: require disambiguation.**
