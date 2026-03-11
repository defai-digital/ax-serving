# ax-serving v1.4.1

## Summary

`v1.4.1` is a validation follow-up to `v1.4.0`.

It does not change the runtime contract shape introduced in `v1.4.0`. Instead, it closes the remaining acceptance-test gaps for the new scheduler, cache-coordination, orchestration, and backend-config work.

## Added Validation Coverage

### Scheduler and Metrics

- Added `/metrics` assertions for:
  - `axs_cache_follower_waiting`
  - `axs_ttft_p50_us`
  - `axs_ttft_p95_us`
  - `axs_ttft_p99_us`
- Added `/v1/metrics` assertions for:
  - `cache_follower_waiting`
  - `ttft_p50_us`
  - `ttft_p95_us`
  - `ttft_p99_us`

### Cache Coordination

- Added an integration test proving cache followers wait without consuming an extra scheduler permit
- Verifies the key `v1.4` invariant:
  - follower wait does not increase `inflight_count`
  - follower wait is reflected in `cache_follower_waiting`

### Token-Cost Dispatch

- Added orchestration integration coverage for `token_cost` dispatch
- Verifies real worker selection prefers the lower-cost worker when extended heartbeat telemetry is present

### Extended Heartbeat Telemetry

- Added internal API round-trip coverage for:
  - `active_sequences`
  - `decode_tok_per_sec`
  - `ttft_p95_ms`
- Verifies these fields are transmitted through the internal heartbeat API and persisted in worker snapshots

### Backend Config Parsing

- Added unit coverage for `AXS_MISTRALRS_MAX_SEQS`
- Verifies:
  - default value
  - valid positive override
  - rejection of zero / invalid values

## Verification

Verified with:

- `cargo test -p ax-serving-api --test model_management`
- `cargo test -p ax-serving-api --test orchestration`
- `cargo test -p ax-serving-engine backend::tests`
- `cargo clippy --workspace --tests -- -D warnings`
