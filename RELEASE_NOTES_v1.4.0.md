# ax-serving v1.4.0

## Summary

`v1.4.0` completes the remaining `v1.3` acceptance gaps and delivers the first scheduler-focused `v1.4` runtime improvements for AX Fabric deployments.

This release strengthens:

- AX Fabric runtime contract coverage
- embeddings success and failure behavior
- startup/readiness lifecycle verification
- token-aware scheduler instrumentation
- operator-facing runtime controls for offline enterprise deployments

## Highlights

### v1.3 Completion

- Added embeddings success-path integration coverage
- Added embeddings backend-failure integration coverage
- Added startup readiness sequence coverage:
  - process up
  - `/health` degraded with no models
  - model load
  - `/health` returns `ok`
- Added failed-load regression coverage to ensure health remains degraded with no loaded models
- Expanded the AX Fabric runtime contract documentation

### v1.4 Scheduler and Runtime

- Fixed a TTFT accounting bug where a request could record TTFT more than once
- Added token-aware scheduler admission inputs for:
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
- Added split-scheduler observability signals:
  - `prefill_tokens_active`
  - `decode_sequences_active`
  - `split_scheduler_enabled`
- Exposed `AXS_MISTRALRS_MAX_SEQS` to control `mistralrs` continuous-batching depth
- Documented `AXS_SPLIT_SCHEDULER` and `AXS_MISTRALRS_MAX_SEQS` in operator docs

### CI and Documentation

- Added GitHub Actions test summary output for unit and integration suites
- Updated README and Quickstart to reflect:
  - AX Fabric runtime contract
  - `v1.4` runtime controls
  - current integration test coverage

## Verification

Verified with:

- `cargo test -p ax-serving-api --test model_management`
- `cargo test -p ax-serving-api scheduler::tests -- --nocapture`
- `cargo clippy --workspace --tests -- -D warnings`

## Upgrade Notes

- `AXS_SPLIT_SCHEDULER` is optional and disabled by default
- `AXS_MISTRALRS_MAX_SEQS` is optional; default remains `32`
- AX Fabric integrations can now rely on the expanded contract in `docs/contracts/ax-fabric-runtime-contract.md`
