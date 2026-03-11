# AX Serving v1.6.0

`v1.6.0` adds a business trust-baseline slice to the public `ax-serving` control plane.

## Highlights

- Added authenticated admin startup reports:
  - `GET /v1/admin/startup-report`
- Added authenticated diagnostics bundles:
  - `GET /v1/admin/diagnostics`
- Added authenticated in-process audit log access:
  - `GET /v1/admin/audit`
- Added audit recording for:
  - serving startup
  - orchestrator startup
  - model load / unload / reload
  - license changes
  - worker drain / drain-complete / delete

## Why It Matters

This release improves operator trust and supportability without changing `ax-serving`'s role in the stack:

- `ax-serving` remains the serving and control plane for `AX Fabric`
- the new surfaces make offline and business deployments easier to inspect and troubleshoot
- audit visibility is now available through stable authenticated admin APIs

## Verification

- `cargo test -p ax-serving-api --test model_management`
- `cargo test -p ax-serving-api --test orchestration`
- `cargo clippy --workspace --tests -- -D warnings`
