//! Background task that drives the worker health state machine.
//!
//! Runs every `heartbeat_ms / 2` ms.  On each tick:
//! 1. [`WorkerRegistry::tick`] derives health state from heartbeat age and
//!    evicts workers whose age exceeds `ttl_ms`.
//! 2. For each worker that is already `Unhealthy`, an active TCP connect probe
//!    is attempted (1 s timeout).  If the probe fails the worker is evicted
//!    immediately rather than waiting for the full TTL to expire.
//!
//! # Why active probing?
//!
//! TTL-only eviction can take up to `ttl_ms` (default 15 s) to remove a dead
//! worker.  During that window the dispatcher may attempt to forward requests
//! to the dead addr, triggering reroutes and latency spikes.  Active TCP
//! probing on *already-unhealthy* workers reduces the window to one tick
//! interval (~2.5 s with defaults) with negligible overhead (only probed after
//! a missed heartbeat, not on every tick for healthy workers).

use std::net::SocketAddr;
use std::time::Duration;

use tokio::sync::watch;
use tracing::{info, warn};

use super::registry::{WorkerId, WorkerRegistry};

pub struct HealthTicker {
    registry: WorkerRegistry,
    tick_interval: Duration,
    ttl_ms: u64,
}

impl HealthTicker {
    /// `heartbeat_ms` — how often workers are expected to heartbeat.
    /// `ttl_ms`       — age after which a worker is evicted.
    pub fn new(registry: WorkerRegistry, heartbeat_ms: u64, ttl_ms: u64) -> Self {
        Self {
            registry,
            // Halve the heartbeat period for the tick interval so we catch
            // a missed beat promptly.  Clamp to ≥ 1 ms so that a zero or
            // near-zero heartbeat_ms does not produce Duration::ZERO, which
            // causes tokio::time::interval to panic.
            tick_interval: Duration::from_millis((heartbeat_ms / 2).max(1)),
            ttl_ms,
        }
    }

    /// Run the ticker until `shutdown` emits `true`.
    pub async fn run(self, mut shutdown: watch::Receiver<bool>) {
        let mut interval = tokio::time::interval(self.tick_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        info!(
            tick_ms = self.tick_interval.as_millis(),
            ttl_ms = self.ttl_ms,
            "health ticker started"
        );

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Phase 1: TTL-based health transitions + eviction.
                    let evicted = self.registry.tick(self.ttl_ms);
                    for id in &evicted {
                        warn!(%id, ttl_ms = self.ttl_ms, "worker evicted: TTL expired");
                    }

                    // Phase 2: active TCP probe for unhealthy workers.
                    // Only probe workers that have already missed at least one
                    // heartbeat — healthy workers are left untouched to keep
                    // overhead minimal.
                    let candidates = self.registry.list_unhealthy_addrs();
                    if !candidates.is_empty() {
                        probe_and_evict(&self.registry, candidates).await;
                    }
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        info!("health ticker shutting down");
                        break;
                    }
                }
            }
        }
    }
}

/// Attempt TCP connects to `candidates` concurrently (1 s timeout each).
/// Any worker whose probe fails is evicted from the registry immediately.
async fn probe_and_evict(registry: &WorkerRegistry, candidates: Vec<(WorkerId, SocketAddr)>) {
    let probe_timeout = Duration::from_secs(1);

    let tasks: Vec<_> = candidates
        .into_iter()
        .map(|(id, addr)| {
            tokio::spawn(async move {
                let result =
                    tokio::time::timeout(probe_timeout, tokio::net::TcpStream::connect(addr)).await;
                let reachable = matches!(result, Ok(Ok(_)));
                (id, addr, reachable)
            })
        })
        .collect();

    for task in tasks {
        if let Ok((id, addr, reachable)) = task.await
            && !reachable
        {
            warn!(%id, %addr, "worker evicted: TCP probe failed (unreachable)");
            registry.evict(id);
        }
    }
}
