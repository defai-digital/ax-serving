//! Registry snapshot and count queries.
//!
//! Read-only listings for admin/internal endpoints and health probes.
//! Snapshot view construction helpers live in [`super::normalize`].

use crate::orchestration::worker_endpoint::WorkerEndpoint;

use super::WorkerRegistry;
use super::normalize::snapshot_of;
use super::types::{WorkerHealth, WorkerId, WorkerSnapshot};

impl WorkerRegistry {
    /// All workers — for the `/internal/workers` listing endpoint.
    pub fn list_all(&self) -> Vec<WorkerSnapshot> {
        self.inner.iter().map(|r| snapshot_of(r.value())).collect()
    }

    /// Single worker — for the `/internal/workers/{id}` endpoint.
    pub fn get_snapshot(&self, id: WorkerId) -> Option<WorkerSnapshot> {
        self.inner.get(&id).map(|r| snapshot_of(r.value()))
    }

    /// Workers currently in `Unhealthy` state — used by the health ticker for
    /// active TCP probing.  Returns `(WorkerId, WorkerEndpoint)` pairs.
    pub fn list_unhealthy_addrs(&self) -> Vec<(WorkerId, WorkerEndpoint)> {
        self.inner
            .iter()
            .filter(|r| matches!(r.value().health, WorkerHealth::Unhealthy { .. }))
            .map(|r| (r.value().id, r.value().addr.clone()))
            .collect()
    }

    /// Count workers that are healthy AND not draining.
    ///
    /// This mirrors [`eligible_workers`] — only these workers can actually
    /// receive dispatched requests.  Use this for the orchestrator health
    /// `status` field so `"ok"` means "at least one worker can serve traffic",
    /// not "at least one worker exists but may be draining".
    ///
    /// [`eligible_workers`]: Self::eligible_workers
    pub fn eligible_healthy_count(&self) -> usize {
        self.inner
            .iter()
            .filter(|r| {
                let e = r.value();
                !e.drain && matches!(e.health, WorkerHealth::Healthy)
            })
            .count()
    }

    /// Count workers by health state and drain flag (for /health endpoint).
    ///
    /// Returns `(healthy, unhealthy, draining)`.  The `draining` count is
    /// orthogonal to health state — a draining worker may be healthy or unhealthy.
    ///
    /// Note: there is no `dead` count because `tick()` removes Dead workers in a
    /// second pass after the `iter_mut` sweep.  Between the two passes a Dead
    /// worker is briefly visible but excluded from `eligible_workers` and
    /// `eligible_healthy_count` because its health is `Dead`.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut healthy = 0usize;
        let mut unhealthy = 0usize;
        let mut draining = 0usize;
        for r in self.inner.iter() {
            let e = r.value();
            if e.drain {
                draining += 1;
            }
            match e.health {
                WorkerHealth::Healthy => healthy += 1,
                WorkerHealth::Unhealthy { .. } => unhealthy += 1,
                WorkerHealth::Dead => {} // briefly visible between tick() passes; ignore
            }
        }
        (healthy, unhealthy, draining)
    }
}
