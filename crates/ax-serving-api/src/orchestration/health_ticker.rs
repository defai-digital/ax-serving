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
use tokio::task::JoinSet;
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

/// Timeout for each TCP liveness probe. Short enough that a dead worker is
/// detected within one heartbeat interval; long enough for a loaded host to accept.
const TCP_PROBE_TIMEOUT_SECS: u64 = 1;
/// Cap concurrent liveness probes so a churny worker pool cannot fan out an
/// unbounded number of probe tasks on a single health-ticker tick.
const MAX_CONCURRENT_TCP_PROBES: usize = 32;

async fn probe_candidate(
    id: WorkerId,
    addr: SocketAddr,
    probe_timeout: Duration,
) -> (WorkerId, SocketAddr, bool) {
    let result = tokio::time::timeout(probe_timeout, tokio::net::TcpStream::connect(addr)).await;
    let reachable = matches!(result, Ok(Ok(_)));
    (id, addr, reachable)
}

async fn probe_candidates(
    candidates: Vec<(WorkerId, SocketAddr)>,
    probe_timeout: Duration,
    max_concurrency: usize,
) -> Vec<(WorkerId, SocketAddr, bool)> {
    let limit = max_concurrency.max(1);
    let mut pending = candidates.into_iter();
    let mut probes = JoinSet::new();
    let mut results = Vec::new();

    loop {
        while probes.len() < limit {
            let Some((id, addr)) = pending.next() else {
                break;
            };
            probes.spawn(probe_candidate(id, addr, probe_timeout));
        }

        let Some(joined) = probes.join_next().await else {
            break;
        };

        if let Ok(result) = joined {
            results.push(result);
        }
    }

    results
}

/// Attempt TCP connects to `candidates` concurrently (1 s timeout each).
/// Any worker whose probe fails is evicted from the registry immediately.
async fn probe_and_evict(registry: &WorkerRegistry, candidates: Vec<(WorkerId, SocketAddr)>) {
    let probe_timeout = Duration::from_secs(TCP_PROBE_TIMEOUT_SECS);

    for (id, addr, reachable) in
        probe_candidates(candidates, probe_timeout, MAX_CONCURRENT_TCP_PROBES).await
    {
        if !reachable {
            warn!(%id, %addr, "worker evicted: TCP probe failed (unreachable)");
            registry.evict(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::time::Duration;

    use super::{probe_and_evict, probe_candidates};
    use crate::orchestration::registry::{
        RegisterCapabilities, RegisterRequest, WorkerId, WorkerRegistry,
    };

    fn register_worker(registry: &WorkerRegistry, addr: SocketAddr) -> WorkerId {
        let response = registry.register(
            RegisterRequest {
                worker_id: None,
                addr: addr.to_string(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".to_string()]),
                backend: "auto".to_string(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
            },
            5_000,
        );
        WorkerId::parse(&response.worker_id).expect("registry must return a valid worker id")
    }

    #[tokio::test]
    async fn probe_candidates_reports_reachability() {
        let reachable_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("reachable listener");
        let reachable_addr = reachable_listener
            .local_addr()
            .expect("reachable listener local addr");

        let closed_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("closed listener");
        let closed_addr = closed_listener
            .local_addr()
            .expect("closed listener local addr");
        drop(closed_listener);

        let reachable_id = WorkerId::new();
        let closed_id = WorkerId::new();
        let results = probe_candidates(
            vec![(reachable_id, reachable_addr), (closed_id, closed_addr)],
            Duration::from_millis(100),
            1,
        )
        .await;

        let by_id: HashMap<WorkerId, bool> = results
            .into_iter()
            .map(|(id, _addr, reachable)| (id, reachable))
            .collect();
        assert_eq!(by_id.get(&reachable_id), Some(&true));
        assert_eq!(by_id.get(&closed_id), Some(&false));
    }

    #[tokio::test]
    async fn probe_and_evict_removes_only_unreachable_workers() {
        let reachable_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("reachable listener");
        let reachable_addr = reachable_listener
            .local_addr()
            .expect("reachable listener local addr");

        let closed_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("closed listener");
        let closed_addr = closed_listener
            .local_addr()
            .expect("closed listener local addr");
        drop(closed_listener);

        let registry = WorkerRegistry::new();
        let reachable_id = register_worker(&registry, reachable_addr);
        let closed_id = register_worker(&registry, closed_addr);
        registry.mark_unhealthy(reachable_id);
        registry.mark_unhealthy(closed_id);

        probe_and_evict(&registry, registry.list_unhealthy_addrs()).await;

        assert!(
            registry.get_snapshot(reachable_id).is_some(),
            "reachable worker should remain registered"
        );
        assert!(
            registry.get_snapshot(closed_id).is_none(),
            "unreachable worker should be evicted"
        );
    }
}
