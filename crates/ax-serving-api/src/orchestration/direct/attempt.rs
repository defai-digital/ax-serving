//! Inflight and attempt RAII guards for dispatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::reservation::SharedReservationGuard;

/// RAII guard: increments a counter on creation, decrements on drop.
///
/// Acquisition uses a CAS loop so `max_inflight` is respected without a lock.
pub(super) struct InflightGuard(pub(super) Arc<AtomicUsize>);

impl InflightGuard {
    #[cfg(test)]
    pub(super) fn acquire(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(counter))
    }

    pub(super) fn try_acquire(counter: &Arc<AtomicUsize>, max_inflight: usize) -> Option<Self> {
        let max_inflight = max_inflight.max(1);
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current >= max_inflight {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Self(Arc::clone(counter))),
                Err(actual) => current = actual,
            }
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let _ = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            });
    }
}

/// Holds inflight + optional shared reservation for the lifetime of one dispatch attempt.
///
/// Dropping this guard releases both local capacity and the shared reservation
/// (best-effort release on the shared store).
pub(super) struct AttemptGuard {
    pub(super) _inflight: InflightGuard,
    pub(super) _reservation: Option<SharedReservationGuard>,
}
