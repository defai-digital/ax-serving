# ADR-009: Configuration Strategy — TOML Config + Environment Variable Overrides

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

ax-engine uses **environment variables exclusively** for configuration. This works for
development (easy to toggle, no file to edit) but has limitations for production:

- No persistent configuration between invocations
- No documentation of current config state
- No config validation at startup
- Env vars are global; cannot vary per-model

mistralrs-cli uses **TOML config files** (`from-config <file>`) for complex multi-model
setups. This is better for production but less ergonomic for interactive development.

ax-serving needs both: a stable config file for operators, with env var overrides for
developers and CI.

---

## Decision

ax-serving uses a **layered configuration system**:

```
Priority (highest to lowest):
1. CLI flags          (--port 9090)
2. Environment vars   (AXS_REST_PORT=9090)
3. Config file        (serving.toml)
4. Built-in defaults
```

### Config File Format (`serving.toml`)

```toml
[server]
rest_port = 8080
rest_host = "127.0.0.1"
grpc_socket = "/tmp/ax-serving.sock"

[backend]
type = "metal"                    # metal | cpu | auto
thermal_monitoring = true

[models]
max_loaded = 4
default_context_length = 4096

[kv]
block_size_tokens = 16
max_blocks = 512
prefix_cache_max_entries = 1024   # 0 = disabled

[metrics]
rss_polling_interval_secs = 10
latency_histogram_capacity = 4096

[[models.preload]]
path = "/models/llama3-8b-q4.gguf"
id = "llama3"
context_length = 8192             # override
```

Config file location (in priority order):
1. `--config <path>` CLI flag
2. `AXS_CONFIG` environment variable
3. `./serving.toml` (current directory)
4. `~/.config/ax-serving/serving.toml`
5. (built-in defaults only)

### Environment Variable Reference

All `AXS_*` — no collision with ax-engine `AX_*` in dual-install environments.

| Variable | Type | Default | Description |
|---|---|---|---|
| `AXS_CONFIG` | path | (auto) | Config file location |
| `AXS_REST_PORT` | u16 | 8080 | HTTP REST port |
| `AXS_REST_HOST` | str | 127.0.0.1 | HTTP bind address |
| `AXS_GRPC_SOCKET` | path | /tmp/ax-serving.sock | gRPC UDS path |
| `AXS_GRPC_PORT` | u16 | 50051 | gRPC TCP port (if using TCP) |
| `AXS_BACKEND` | str | auto | metal \| cpu \| auto |
| `AXS_THERMAL` | bool | on | Thermal monitoring on/off |
| `AXS_MAX_LOADED` | u32 | 4 | Max simultaneous models |
| `AXS_PAGED_KV` | bool | off | Enable paged KV (Phase 2) |
| `AXS_PREFIX_CACHE` | bool | off | Enable prefix cache (Phase 2) |
| `AXS_PREFIX_CACHE_MAX` | u32 | 1024 | Max prefix cache entries |
| `AXS_DEBUG_LOGITS` | bool | off | Dump top-5 logits per step |
| `AXS_LOG` | str | info | Log level (trace/debug/info/warn/error) |

### Startup Config Dump

On startup, ax-serving logs the resolved configuration:

```
[INFO] ax-serving v0.1.0 starting
[INFO] config source: ./serving.toml + env overrides
[INFO] backend: metal (Apple M3 Pro, 18GB UMA)
[INFO] rest: http://127.0.0.1:8080
[INFO] grpc: /tmp/ax-serving.sock
[INFO] thermal: on (current: Nominal)
[INFO] models: max_loaded=4, kv_block_size=16
[INFO] preloading: llama3 from /models/llama3-8b-q4.gguf
```

This makes the configuration observable without digging through env vars.

### Config Validation

At startup, before loading any models:
- Validate all paths exist and are readable
- Check `rest_port` is in valid range and not in use
- Check `max_loaded` ≥ 1 and ≤ 16
- Check `kv.block_size_tokens` is a power of 2 ≥ 1
- Warn (not error) on unknown TOML keys

---

## Migration from ax-engine env vars

| ax-engine env var | ax-serving equivalent |
|---|---|
| `AX_CPU_ONLY=1` | `AXS_BACKEND=cpu` |
| `AX_METAL_F16_KV_CACHE=auto` | (default behavior via mistralrs) |
| `AX_PAGED_KV=1` | `AXS_PAGED_KV=on` |
| `AX_PREFIX_CACHE=1` | `AXS_PREFIX_CACHE=on` |
| `AX_DEBUG_LOGITS=1` | `AXS_DEBUG_LOGITS=on` |
| `AX_SERIAL_PREFILL=1` | Not supported (mistralrs batches by default) |
| `AX_METAL_BATCH_SIMD=1` | Not applicable (no custom Metal kernels) |

---

## Consequences

### Positive

- **Operator-friendly**: TOML file is inspectable, version-controlled, self-documented
- **Developer-friendly**: env var overrides work for quick iteration without editing files
- **Validated**: startup catches misconfiguration before serving requests
- **Observable**: startup log shows exactly what config is active

### Negative

- **Slightly more code**: TOML parsing + validation layer (~200 lines, using `toml` crate)
- **Breaking change from ax-engine**: `AX_*` vars become `AXS_*`; migration required

---

## Crate Used for Config Parsing

```toml
# Cargo.toml
toml = "0.8"
serde = { version = "1", features = ["derive"] }
```

No additional config crate (figment, config-rs) to keep dependencies minimal. The
layering (file → env → defaults) is implemented in ~100 lines of explicit merge logic.
