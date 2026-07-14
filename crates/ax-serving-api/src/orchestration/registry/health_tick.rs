//! Health ticker: TTL-based health transitions and eviction paths.
//!
//! [`WorkerRegistry::tick`] derives health from heartbeat age and removes
//! Dead workers. Probe-driven eviction uses
//! [`WorkerRegistry::evict_if_unhealthy_at_addr`] so stale probe snapshots
//! cannot remove a recovered worker.

use super::super::worker_endpoint::WorkerEndpoint;
use super::WorkerRegistry;
use super::types::{WorkerHealth, WorkerId};

impl WorkerRegistry {
    /// Remove a worker entirely (drain-complete or explicit eviction).
    pub fn evict(&self, id: WorkerId) {
        self.inner.remove(&id);
        self.protocol_sessions
            .retain(|_, session| session.internal_id != id);
    }

    /// Remove a worker only if it still matches a stale unhealthy probe snapshot.
    ///
    /// Active TCP probes are launched from a point-in-time list of unhealthy
    /// workers. A heartbeat or re-registration can make that snapshot stale
    /// before the probe result returns, so failed probes must not evict a worker
    /// that has already recovered or moved to a different address.
    pub fn evict_if_unhealthy_at_addr(&self, id: WorkerId, addr: &WorkerEndpoint) -> bool {
        let removed = self
            .inner
            .remove_if(&id, |_, entry| {
                &entry.addr == addr && matches!(entry.health, WorkerHealth::Unhealthy { .. })
            })
            .is_some();
        if removed {
            self.protocol_sessions
                .retain(|_, session| session.internal_id != id);
        }
        removed
    }

    // ── Health ticker ─────────────────────────────────────────────────────────

    /// Derive health state from heartbeat age and evict Dead workers.
    ///
    /// Called by [`HealthTicker`] on each tick.  Returns the IDs of any
    /// workers that were evicted in this call.
    ///
    /// [`HealthTicker`]: crate::orchestration::health_ticker::HealthTicker
    pub fn tick(&self, ttl_ms: u64) -> Vec<WorkerId> {
        // First pass: update health states and collect dead IDs.
        // DashMap's `iter_mut` locks one shard at a time — concurrent heartbeats
        // on other shards proceed in parallel.
        let mut evicted = Vec::new();

        for mut r in self.inner.iter_mut() {
            let entry = r.value_mut();
            let age_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
            if entry.drain {
                // Draining workers are normally removed via drain-complete, but
                // if the worker crashes before calling it we must still evict.
                if age_ms > ttl_ms {
                    evicted.push(entry.id);
                }
                continue;
            }

            entry.health = if age_ms <= ttl_ms / 3 {
                if matches!(entry.health, WorkerHealth::Unhealthy { .. }) {
                    entry.health.clone()
                } else {
                    WorkerHealth::Healthy
                }
            } else if age_ms <= (2 * ttl_ms) / 3 {
                WorkerHealth::Unhealthy { missed: 1 }
            } else if age_ms <= ttl_ms {
                WorkerHealth::Unhealthy { missed: 2 }
            } else {
                evicted.push(entry.id);
                WorkerHealth::Dead // removed below
            };
        }

        // Second pass: remove only entries that are still stale. This closes the
        // race where a heartbeat or re-registration refreshes the worker after
        // the first pass but before removal.
        for id in &evicted {
            self.inner.remove_if(id, |_, entry| {
                let age_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
                if entry.drain {
                    age_ms > ttl_ms
                } else {
                    age_ms > ttl_ms && matches!(entry.health, WorkerHealth::Dead)
                }
            });
        }

        if !evicted.is_empty() {
            self.protocol_sessions
                .retain(|_, session| !evicted.contains(&session.internal_id));
        }

        evicted
    }
}
