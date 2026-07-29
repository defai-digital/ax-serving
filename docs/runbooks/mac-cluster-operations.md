# Mac cluster operations runbook

Operator procedures for one `mac_ax_engine_cluster` domain. Clients continue to
call `POST /v1/chat/completions` with a logical model; they never pass rank
metadata.

## Preconditions

- Immutable cluster manifest with generation, digests, memory plan, and rank map
- Distinct credentials:
  - public client API key
  - gateway internal/worker registration token
  - dispatch token (gateway → adapter)
  - rank-control token (ranks → adapter)
  - rank-0 runtime token (adapter → Engine OpenAI frontend)
- AX Engine rank binaries pinned to the manifest `runtime.build_digest`
- Optional Redis/Valkey for multi-gateway HA

## Install / first ready

1. Start gateway with `config/serving.mac-cluster.example.yaml`.
2. Start `ax-mac-cluster-adapter` with `config/mac-cluster-adapter.example.env`.
3. On each Mac rank:
   - `GET /internal/cluster/ranks/{rank}/plan`
   - prepare artifacts (shard-aware download/verify)
   - start `ax-engine-pipeline-rank`
   - heartbeat ready with generation + manifest digest + topology measurements
4. Start rank-0 OpenAI gateway and point `AXS_MAC_CLUSTER_RANK0_URL` at it.
5. Confirm operator status is `ready` and gateway can admit the logical model.

## Upgrade / generation replacement

1. Publish a new manifest with a **strictly higher generation** and new digest.
2. Drain the adapter (`begin_drain`) so new admission stops.
3. Stop ranks for the old generation (they are generation-fenced).
4. Restart adapter with the new manifest; ranks bootstrap from the new plan.
5. Ready requires the complete gang at the new generation.

Never live-reshard an admitted generation.

## Rank loss / fault

- One required rank failure fails the whole generation.
- Adapter marks the domain unready; AX does not route new attempts.
- Recovery = higher generation restart, not partial-rank continue.
- Multi-replica HA: deploy **two complete clusters** behind the same domain/pool;
  degraded is allowed only while at least one complete replica remains ready.

## Credential rotation

Rotate one plane at a time with a dual-accept window where possible:

1. rank-control token on adapter, then ranks
2. rank-0 runtime token on Engine gateway, then adapter
3. dispatch token on gateway, then adapter
4. internal registration token on gateway, then adapter
5. public API keys via normal key rotation

Do not reuse credentials across planes.

## Rollback

- Adapter/gateway: redeploy previous binary + previous generation manifest.
- Adaptive federation: set `adaptive_federation.mode: rollback` to force the
  baseline domain while retaining decision evidence.
- Never roll back to a partial-rank plan.

## Evidence retention

Retain for every certification run:

- topology label, OS, AX Engine build, model/quantization digests
- transport class and measured bandwidth/latency
- load / fault / restart / soak outcomes via evidence journal hooks
- adaptive decision records (prompt-free) from the fleet store

Physical 60-minute soak and usefulness thresholds remain external release gates.
