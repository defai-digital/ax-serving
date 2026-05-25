# ADR-008: Thermal-Aware Scheduling

**Status**: Accepted
**Date**: 2026-03-01
**Deciders**: AutomatosX Team (DEFAI Private Limited)

---

## Context

Apple Silicon M3 chips aggressively throttle under sustained compute load. Without
thermal awareness, an inference server running at full capacity for extended periods
will experience:
- Progressive decode tok/s degradation (10–40% over 30 minutes)
- Unpredictable latency spikes (2–5× at throttle onset)
- P95 latency drift > 5% (violates soak test requirement)

ax-engine addresses this via `ThermalMonitor` (polls macOS `notify_register_dispatch`
for thermal pressure notifications) and a `Scheduler` that adjusts thread count based
on thermal state:

| Thermal State | Action |
|---|---|
| Nominal | Full thread count |
| Fair | Full thread count (monitor) |
| Serious | Half thread count (min 1) |
| Critical | 1 thread |

This integration was effective in ax-engine's 24h soak tests.

mistralrs-core has no thermal integration.

---

## Decision

Port ax-engine's `ThermalMonitor` to `ax-serving-engine/src/thermal.rs`.

The thermal state is:
1. **Exposed in health API** — gRPC `Health` RPC and REST `/health` return current state
2. **Applied to Tokio spawn_blocking concurrency** — `InferenceBackend::recommended_concurrency()`
3. **Logged on state change** — `tracing::warn!` with state name

### ThermalMonitor Port

```rust
// ax-serving-engine/src/thermal.rs

use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal  = 0,
    Fair     = 1,
    Serious  = 2,
    Critical = 3,
}

pub struct ThermalMonitor {
    state: Arc<AtomicU8>,
}

impl ThermalMonitor {
    pub fn new() -> Self { ... }  // registers notify_register_dispatch callback

    pub fn current(&self) -> ThermalState {
        match self.state.load(Ordering::Relaxed) {
            0 => ThermalState::Nominal,
            1 => ThermalState::Fair,
            2 => ThermalState::Serious,
            _ => ThermalState::Critical,
        }
    }
}
```

### Concurrency Integration

mistralrs-core uses Tokio for its engine loop. The thermal state influences how many
`tokio::task::spawn_blocking` tasks ax-serving allows concurrently (via a semaphore):

```rust
// ax-serving-engine/src/backend.rs

pub struct MistralrsBackend {
    pipeline: Arc<dyn mistralrs_core::Pipeline>,
    thermal: ThermalMonitor,
    max_concurrent: usize,      // sysctl hw.perflevel0.physicalcpu
    decode_semaphore: Arc<Semaphore>,
}

impl MistralrsBackend {
    fn concurrency_limit(&self) -> usize {
        match self.thermal.current() {
            ThermalState::Nominal | ThermalState::Fair => self.max_concurrent,
            ThermalState::Serious => (self.max_concurrent / 2).max(1),
            ThermalState::Critical => 1,
        }
    }
}
```

For Phase 1 (single-request serving), the semaphore has capacity 1. Multi-request
concurrency (Phase 3) will use this semaphore to bound parallelism.

### P-core Preference

Retain ax-engine's QoS hint for inference threads:

```rust
// In spawn_blocking closure:
unsafe {
    libc::pthread_set_qos_class_self_np(
        libc::QOS_CLASS_USER_INTERACTIVE, 0
    );
}
```

This is a hint to macOS; it prefers P-cores but is not a hard guarantee.

### Env Override

```
AXS_THERMAL=off    # disable thermal monitoring (for benchmarks)
AXS_THERMAL=on     # enable (default)
```

---

## Soak Test Integration

`ax-serving-bench soak` records thermal state at each measurement interval:

```json
{
  "interval_secs": 600,
  "measurements": [
    {"t": 0, "decode_tps": 104.2, "thermal": "Nominal"},
    {"t": 600, "decode_tps": 103.8, "thermal": "Nominal"},
    {"t": 1200, "decode_tps": 99.1, "thermal": "Fair"},
    ...
  ]
}
```

A thermal-induced drop is distinguished from a genuine regression: if thermal state is
Serious or Critical, the measurement is flagged but does not trigger a drift alert.

---

## Consequences

### Positive

- **Predictable long-run performance** — thermal-adaptive concurrency prevents
  progressive degradation
- **Soak test compliance** — P95 drift < 5% achievable with thermal management
- **Operator visibility** — thermal state in health endpoint enables alerting

### Negative

- **macOS-specific** — `notify_register_dispatch` is a Darwin API; reinforces
  Apple Silicon-only positioning (acceptable given platform constraint)
- **Latency on state change** — brief latency spike at throttle onset before
  semaphore adjustment takes effect (typically < 1 token generation)

---

## Alternatives Considered

### A: No thermal management

Rejected. Without it, 24h soak tests show >10% P95 drift on M3 Pro under sustained load.

### B: Rely on OS scheduler

Rejected. macOS will throttle the CPU regardless, but without our semaphore we may
have many queued tasks competing for reduced resources. Proactive reduction is better.

### C: Use mistralrs-core's scheduler

mistralrs-core has no macOS thermal integration. We cannot rely on it.
