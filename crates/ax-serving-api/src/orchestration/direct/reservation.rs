//! Shared fleet-state reservations for multi-gateway dispatch.
//!
//! Each reserved attempt spawns a per-attempt renew loop. Instrumentation on
//! [`DispatchMetrics`] records active renew tasks and renew outcomes so a future
//! shared renewer can be justified by measured task churn / fence rates.
//!
//! # Drop semantics
//!
//! - On drop, the renew loop is stopped (best-effort oneshot) and
//!   `release_reservation` is spawned on the current runtime if available.
//! - A fenced renew (`ReservationResult::Saturated`) ends the renew loop; the
//!   reservation is still released on drop when the attempt ends.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use tracing::warn;

use super::metrics::DispatchMetrics;
use crate::orchestration::fleet_state::{FleetStateStore, ReservationResult};

/// RAII guard that holds a shared reservation and renews it until dropped.
pub(super) struct SharedReservationGuard {
    store: Arc<dyn FleetStateStore>,
    worker_id: ax_serving_protocol::WorkerId,
    attempt_id: ax_serving_protocol::AttemptId,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

/// Tracks one active renew task so the gauge decrements on every exit path.
struct RenewTaskGuard(Arc<DispatchMetrics>);

impl Drop for RenewTaskGuard {
    fn drop(&mut self) {
        let _ = self.0.reservation_renew_tasks.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_sub(1),
        );
    }
}

impl SharedReservationGuard {
    pub(super) fn new(
        store: Arc<dyn FleetStateStore>,
        worker_id: ax_serving_protocol::WorkerId,
        attempt_id: ax_serving_protocol::AttemptId,
        max_concurrent: usize,
        ttl_ms: u64,
        metrics: Arc<DispatchMetrics>,
    ) -> Self {
        let (stop, mut stopped) = tokio::sync::oneshot::channel();
        let renew_store = Arc::clone(&store);
        let renew_worker = worker_id.clone();
        let renew_metrics = Arc::clone(&metrics);
        let renew_every = std::time::Duration::from_millis((ttl_ms / 3).max(250));
        renew_metrics
            .reservation_renew_tasks
            .fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            let _task_guard = RenewTaskGuard(Arc::clone(&renew_metrics));
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(renew_every) => {
                        match renew_store
                            .try_reserve(
                                &renew_worker,
                                attempt_id,
                                max_concurrent,
                                ttl_ms,
                            )
                            .await
                        {
                            Ok(ReservationResult::Reserved) => {
                                renew_metrics
                                    .reservation_renew_ok_total
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(ReservationResult::Saturated) => {
                                renew_metrics
                                    .reservation_renew_fenced_total
                                    .fetch_add(1, Ordering::Relaxed);
                                warn!(%renew_worker, %attempt_id, "shared dispatch reservation renewal was fenced");
                                break;
                            }
                            Err(error) => {
                                renew_metrics
                                    .reservation_renew_error_total
                                    .fetch_add(1, Ordering::Relaxed);
                                warn!(%renew_worker, %attempt_id, %error, "shared dispatch reservation renewal failed");
                            }
                        }
                    }
                    _ = &mut stopped => break,
                }
            }
        });
        Self {
            store,
            worker_id,
            attempt_id,
            stop: Some(stop),
        }
    }
}

impl Drop for SharedReservationGuard {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let store = Arc::clone(&self.store);
        let worker_id = self.worker_id.clone();
        let attempt_id = self.attempt_id;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(error) = store.release_reservation(&worker_id, attempt_id).await {
                    warn!(%worker_id, %attempt_id, %error, "shared dispatch reservation release failed");
                }
            });
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ReservationAcquireError {
    #[error("worker reservation capacity exhausted")]
    Saturated,
    #[error("shared fleet state is unavailable")]
    Store(#[source] anyhow::Error),
}

/// Try to acquire a shared reservation when a fleet store is configured.
pub(super) async fn reserve_attempt(
    fleet_store: Option<&Arc<dyn FleetStateStore>>,
    metrics: &Arc<DispatchMetrics>,
    reservation_ttl_ms: u64,
    worker_id: Option<ax_serving_protocol::WorkerId>,
    attempt_id: ax_serving_protocol::AttemptId,
    max_concurrent: usize,
) -> Result<Option<SharedReservationGuard>, ReservationAcquireError> {
    let (Some(store), Some(worker_id)) = (fleet_store, worker_id) else {
        return Ok(None);
    };
    match store
        .try_reserve(&worker_id, attempt_id, max_concurrent, reservation_ttl_ms)
        .await
        .map_err(ReservationAcquireError::Store)?
    {
        ReservationResult::Reserved => Ok(Some(SharedReservationGuard::new(
            Arc::clone(store),
            worker_id,
            attempt_id,
            max_concurrent,
            reservation_ttl_ms,
            Arc::clone(metrics),
        ))),
        ReservationResult::Saturated => Err(ReservationAcquireError::Saturated),
    }
}
