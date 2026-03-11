# AX Serving v1.8.0

`v1.8.0` adds the first AX Fabric-oriented integration slice on top of the fleet control work from `v1.7.0`.

## Highlights

- Added worker registration metadata for integrated deployments:
  - `worker_pool`
  - `node_class`
- Added pool-aware dispatch preference:
  - request header `X-AX-Worker-Pool`
- Added authenticated fleet diagnostics endpoint:
  - `GET /v1/admin/fleet`
- Extended worker auto-registration so workers can advertise:
  - `AXS_WORKER_POOL`
  - `AXS_WORKER_NODE_CLASS`

## Why It Matters

This release improves AX Fabric integration and multi-worker operability:

- Fabric-side or operator-side tooling can steer traffic toward a preferred worker pool
- fleet inventory can now be summarized by pool, node class, and backend
- the control plane has better topology visibility for offline business deployments

## Verification

- `cargo test --workspace`
- `cargo clippy --workspace --tests -- -D warnings`
