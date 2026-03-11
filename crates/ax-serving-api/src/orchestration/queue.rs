//! Global request queue: admission control + concurrency cap.
//!
//! # Design
//!
//! `GlobalQueue` acts as a bounded semaphore with a wait list.
//!
//! When `active >= max_concurrent`:
//! - If `waiters < max_queue_depth`: caller waits (up to `wait_ms`).
//! - If queue depth is exceeded:
//!   - `Reject` policy → immediate HTTP 429.
//!   - `ShedOldest` policy → oldest waiter receives `false` (→ 503),
//!     new request takes its slot in the queue.
//!
//! On `QueuePermit` drop the slot is handed to the first live waiter.
//! Dead waiters (whose `rx` was dropped due to timeout) are skipped lazily.
//!
//! # Environment variables
//!
//! | Variable                   | Default | Description                             |
//! |----------------------------|---------|-----------------------------------------|
//! | `AXS_GLOBAL_QUEUE_MAX`     | `128`   | Max concurrent active requests          |
//! | `AXS_GLOBAL_QUEUE_DEPTH`   | `256`   | Max requests waiting in queue           |
//! | `AXS_GLOBAL_QUEUE_WAIT_MS` | `10000` | Wait timeout before 503 (ms)            |
//! | `AXS_GLOBAL_QUEUE_POLICY`  | `queue` | `"queue"`, `"reject"`, or `"shed_oldest"` |

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;
use tokio::time::timeout;

// ── OverloadPolicy ────────────────────────────────────────────────────────────

/// What to do when both the concurrency limit and queue depth are exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverloadPolicy {
    /// Queue requests; return HTTP 429 when queue overflows (default).
    Queue,
    /// Alias for `Queue` — same behavior, kept for config backward-compat.
    Reject,
    /// Evict the oldest queued waiter (send them 503) to make room.
    ShedOldest,
}

// ── GlobalQueueConfig ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GlobalQueueConfig {
    /// Max concurrent active requests (semaphore width).
    pub max_concurrent: usize,
    /// Max requests waiting for a slot before policy kicks in.
    pub max_queue_depth: usize,
    /// How long a queued request waits before it times out (ms).
    pub wait_ms: u64,
    pub overload_policy: OverloadPolicy,
}

impl Default for GlobalQueueConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 128,
            max_queue_depth: 256,
            wait_ms: 10_000,
            overload_policy: OverloadPolicy::Queue,
        }
    }
}

// ── GlobalQueueMetrics ────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct GlobalQueueMetrics {
    pub permit_total: AtomicU64,
    pub rejected_total: AtomicU64,
    pub shed_total: AtomicU64,
    pub timeout_total: AtomicU64,
}

// ── QueuePermit ───────────────────────────────────────────────────────────────

/// RAII permit.  Dropping it releases one concurrency slot and wakes the
/// next live queued waiter (if any).
pub struct QueuePermit {
    /// Shared active-request counter — decremented if no waiter claims the slot.
    active: Arc<AtomicUsize>,
    /// Waiter queue — locked only when transferring a slot to a queued request.
    waiters: Arc<Mutex<VecDeque<WaiterEntry>>>,
    last_served_client: Arc<Mutex<Option<String>>>,
    /// The client that owns this permit. Used as a fairness hint in `drop()`:
    /// when `last_served_client` is `None` (e.g. first handoff), we seed the
    /// hint with this permit's own client key so the next waiter from a
    /// *different* client is preferred.
    client_key: String,
}

impl Drop for QueuePermit {
    fn drop(&mut self) {
        // Use `ok()` instead of `unwrap()` — panicking inside `drop` during
        // an already-unwinding stack causes an immediate process abort.
        let Ok(mut waiters) = self.waiters.lock() else {
            // Mutex poisoned; decrement active so the slot is not permanently lost.
            self.active.fetch_sub(1, Ordering::Release);
            return;
        };
        // Fast exit when no one is queued — avoids locking last_served_client
        // in the common (no-queue) case.
        if waiters.is_empty() {
            self.active.fetch_sub(1, Ordering::Release);
            return;
        }
        let mut last_served = self.last_served_client.lock().ok();
        // Hand the slot to the next live waiter; skip dead ones
        // (rx was dropped because the request timed out). When multiple clients
        // are queued, prefer a different client than the one most recently
        // served to prevent a single hot client from monopolizing the queue.
        //
        // Hint priority: (1) last_served_client (set on previous handoffs),
        // (2) this permit's own client_key (first handoff — permit was obtained
        //     on the fast path without updating last_served_client).
        let hint: Option<&str> = last_served
            .as_deref()
            .and_then(|v| v.as_deref())
            .or(if self.client_key.is_empty() {
                None
            } else {
                Some(self.client_key.as_str())
            });
        loop {
            let next = select_next_waiter(&mut waiters, hint);
            match next {
                Some(entry) => {
                    if entry.tx.send(true).is_ok() {
                        if let Some(ref mut client) = last_served {
                            **client = Some(entry.client_key);
                        }
                        // Slot transferred — active count is unchanged (permit
                        // passes from this request to the woken waiter).
                        return;
                    }
                    // Dead waiter — try the next one.
                }
                None => {
                    // No live waiter; return the slot to the pool.
                    self.active.fetch_sub(1, Ordering::Release);
                    return;
                }
            }
        }
    }
}

// ── AcquireResult ─────────────────────────────────────────────────────────────

pub enum AcquireResult {
    /// Concurrency slot acquired — proceed with the request.
    Permit(QueuePermit),
    /// Queue depth exceeded and policy is `Reject` → respond 429.
    Rejected,
    /// This request was shed by a newer one → respond 503.
    Shed,
    /// Waited longer than `wait_ms` → respond 503.
    Timeout,
}

// ── GlobalQueue ───────────────────────────────────────────────────────────────

/// Shared admission gate for the orchestrator's public proxy.
///
/// # Design: split-lock semaphore
///
/// `active` is an `AtomicUsize` so the fast path (slot available) is a single
/// CAS with no mutex contention.  The `Mutex<VecDeque>` is only taken when the
/// concurrency limit is reached and a request must be queued, or when a permit
/// drops and needs to wake a waiter.  At low-to-moderate load the mutex is
/// never touched.
///
/// Clone is cheap (all fields are `Arc`-backed).
#[derive(Clone)]
pub struct GlobalQueue {
    /// Number of currently active (permitted) requests.
    active: Arc<AtomicUsize>,
    /// Requests waiting for a slot; locked only on the slow path.
    waiters: Arc<Mutex<VecDeque<WaiterEntry>>>,
    /// Last client that received a handed-off slot. Used to spread queued
    /// wakeups across clients rather than purely serving FIFO by enqueue order.
    last_served_client: Arc<Mutex<Option<String>>>,
    config: Arc<GlobalQueueConfig>,
    pub metrics: Arc<GlobalQueueMetrics>,
}

impl GlobalQueue {
    pub fn new(config: GlobalQueueConfig) -> Self {
        Self {
            active: Arc::new(AtomicUsize::new(0)),
            waiters: Arc::new(Mutex::new(VecDeque::new())),
            last_served_client: Arc::new(Mutex::new(None)),
            config: Arc::new(config),
            metrics: Arc::new(GlobalQueueMetrics::default()),
        }
    }

    /// Build a `QueuePermit` that shares the same `active` counter and
    /// `waiters` queue as this `GlobalQueue`.
    fn make_permit(&self, client_key: String) -> QueuePermit {
        QueuePermit {
            active: Arc::clone(&self.active),
            waiters: Arc::clone(&self.waiters),
            last_served_client: Arc::clone(&self.last_served_client),
            client_key,
        }
    }

    /// Try to acquire a concurrency permit.
    ///
    /// **Fast path** (no mutex): if `active < max_concurrent`, claims the slot
    /// via a CAS loop and returns immediately.  The client key is stored in the
    /// permit so that `drop()` can seed the fairness hint on the first handoff.
    ///
    /// **Slow path** (mutex): if at capacity, queues the caller (up to
    /// `max_queue_depth`) and waits up to `wait_ms`.  Applies the overload
    /// policy when the queue is also full.
    pub async fn acquire(&self, client_key: String) -> AcquireResult {
        // ── Fast path: lock-free CAS loop ────────────────────────────────────
        let max = self.config.max_concurrent;
        let mut current = self.active.load(Ordering::Acquire);
        loop {
            if current >= max {
                break; // At capacity — fall to slow path.
            }
            match self
                .active
                .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // Fast path: no mutex acquired — last_served_client is not
                    // updated here.  The client key is stored in the permit so
                    // drop() can use it as a fallback hint on the first handoff.
                    self.metrics.permit_total.fetch_add(1, Ordering::Relaxed);
                    return AcquireResult::Permit(self.make_permit(client_key));
                }
                Err(actual) => current = actual, // Retry (contention or spurious).
            }
        }

        // ── Slow path: take the waiters mutex ────────────────────────────────
        let rx = {
            let mut waiters = self.waiters.lock().unwrap();

            // Re-check under the mutex: a permit may have dropped (and
            // decremented `active`) between our CAS failure and this lock.
            // If so, claim the freed slot without queueing.
            let mut a = self.active.load(Ordering::Acquire);
            loop {
                if a >= max {
                    break;
                }
                match self.active.compare_exchange_weak(
                    a,
                    a + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.metrics.permit_total.fetch_add(1, Ordering::Relaxed);
                        return AcquireResult::Permit(self.make_permit(client_key));
                    }
                    Err(actual) => a = actual,
                }
            }

            // Still at capacity — prune timed-out waiters (rx dropped) before
            // checking depth.  Without this, timed-out entries remain in the
            // deque and inflate the apparent queue depth, causing new requests
            // to be incorrectly rejected or shed as "queue full".
            waiters.retain(|entry| !entry.tx.is_closed());

            if waiters.len() >= self.config.max_queue_depth {
                match self.config.overload_policy {
                    OverloadPolicy::Reject | OverloadPolicy::Queue => {
                        self.metrics.rejected_total.fetch_add(1, Ordering::Relaxed);
                        return AcquireResult::Rejected;
                    }
                    OverloadPolicy::ShedOldest => {
                        // Evict oldest waiter to make room; active stays the same.
                        if let Some(oldest) = waiters.pop_front() {
                            let _ = oldest.tx.send(false); // false = "you are shed"
                            self.metrics.shed_total.fetch_add(1, Ordering::Relaxed);
                            // Fall through: enqueue current request.
                        } else {
                            // Queue was empty (max_queue_depth=0) — nothing to shed.
                            // Treat as Reject so the depth=0 constraint is honoured.
                            self.metrics.rejected_total.fetch_add(1, Ordering::Relaxed);
                            return AcquireResult::Rejected;
                        }
                    }
                }
            }

            let (tx, rx) = oneshot::channel::<bool>();
            // Clone key into the waiter entry (for fairness tracking during
            // handoff); the original moves into the permit when we wake up.
            waiters.push_back(WaiterEntry { client_key: client_key.clone(), tx });
            rx
        }; // ← waiters mutex released here

        // Wait for a permit, with timeout.
        let wait_dur = std::time::Duration::from_millis(self.config.wait_ms);
        match timeout(wait_dur, rx).await {
            Ok(Ok(true)) => {
                // Slot handed off from a completing request.
                self.metrics.permit_total.fetch_add(1, Ordering::Relaxed);
                AcquireResult::Permit(self.make_permit(client_key))
            }
            Ok(Ok(false)) => {
                // Shed by a newer incoming request.
                AcquireResult::Shed
            }
            Ok(Err(_)) | Err(_) => {
                // Sender dropped (shouldn't happen) or wait_ms elapsed.
                self.metrics.timeout_total.fetch_add(1, Ordering::Relaxed);
                AcquireResult::Timeout
            }
        }
    }

    /// Current number of active (permitted) requests.
    ///
    /// Lock-free — reads the `AtomicUsize` directly.
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Current number of requests waiting in queue.
    pub fn queued(&self) -> usize {
        self.waiters.lock().unwrap().len()
    }
}

#[derive(Debug)]
struct WaiterEntry {
    client_key: String,
    tx: oneshot::Sender<bool>,
}

fn select_next_waiter(
    waiters: &mut VecDeque<WaiterEntry>,
    last_served_client: Option<&str>,
) -> Option<WaiterEntry> {
    let preferred_idx = last_served_client.and_then(|last| {
        waiters
            .iter()
            .position(|entry| !entry.tx.is_closed() && entry.client_key != last)
    });

    if let Some(idx) = preferred_idx {
        waiters.remove(idx)
    } else {
        while let Some(entry) = waiters.pop_front() {
            if !entry.tx.is_closed() {
                return Some(entry);
            }
        }
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &'static str) -> String {
        name.to_string()
    }

    fn cfg(max: usize, depth: usize, wait_ms: u64, policy: OverloadPolicy) -> GlobalQueueConfig {
        GlobalQueueConfig {
            max_concurrent: max,
            max_queue_depth: depth,
            wait_ms,
            overload_policy: policy,
        }
    }

    #[tokio::test]
    async fn permit_immediate_when_under_limit() {
        let q = GlobalQueue::new(cfg(2, 4, 1000, OverloadPolicy::Reject));
        let r = q.acquire(key("a")).await;
        assert!(matches!(r, AcquireResult::Permit(_)));
        assert_eq!(q.active(), 1);
    }

    #[tokio::test]
    async fn permit_released_on_drop() {
        let q = GlobalQueue::new(cfg(1, 0, 100, OverloadPolicy::Reject));
        {
            let r = q.acquire(key("a")).await;
            assert!(matches!(r, AcquireResult::Permit(_)));
            assert_eq!(q.active(), 1);
        }
        assert_eq!(q.active(), 0);
    }

    #[tokio::test]
    async fn reject_when_queue_full() {
        let q = GlobalQueue::new(cfg(1, 0, 100, OverloadPolicy::Reject));
        let _permit = q.acquire(key("a")).await; // occupies the sole slot
        let r = q.acquire(key("a")).await;
        assert!(matches!(r, AcquireResult::Rejected));
        assert_eq!(q.metrics.rejected_total.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn queued_request_gets_permit_on_release() {
        let q = Arc::new(GlobalQueue::new(cfg(1, 4, 1000, OverloadPolicy::Reject)));
        let permit = q.acquire(key("a")).await;
        assert!(matches!(permit, AcquireResult::Permit(_)));

        let q2 = Arc::clone(&q);
        let handle = tokio::spawn(async move { q2.acquire(key("b")).await });

        // Give the waiter time to enqueue.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert_eq!(q.queued(), 1);

        // Release — waiter should receive the slot.
        drop(permit);
        let result = handle.await.unwrap();
        assert!(matches!(result, AcquireResult::Permit(_)));
        assert_eq!(q.metrics.permit_total.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn timeout_when_no_release() {
        let q = Arc::new(GlobalQueue::new(cfg(1, 4, 50, OverloadPolicy::Reject)));
        let _permit = q.acquire(key("a")).await;

        let q2 = Arc::clone(&q);
        let handle = tokio::spawn(async move { q2.acquire(key("b")).await });

        let result = handle.await.unwrap();
        assert!(matches!(result, AcquireResult::Timeout));
        assert_eq!(q.metrics.timeout_total.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn timed_out_waiter_does_not_block_subsequent_enqueue() {
        // Regression for: timed-out waiters whose rx was dropped kept their
        // tx in q.waiters, inflating the apparent depth.  New requests were
        // incorrectly rejected as "queue full" even when the queue was empty.
        //
        // Scenario: max_concurrent=1, max_queue_depth=1
        //   Request A holds the permit.
        //   Request B queues (waiters=[tx_B]).
        //   B times out — rx_B dropped, tx_B remains (dead).
        //   Request C arrives: without the fix, sees len=1 >= depth=1 → Rejected.
        //                      with the fix, dead tx_B is pruned → C queues.
        let q = Arc::new(GlobalQueue::new(cfg(1, 1, 50, OverloadPolicy::Reject)));

        let permit = q.acquire(key("a")).await;
        assert!(matches!(permit, AcquireResult::Permit(_)));

        // Waiter B queues with a very short timeout.
        let q2 = Arc::clone(&q);
        let waiter_b = tokio::spawn(async move { q2.acquire(key("b")).await });

        // Let B time out.
        let r_b = waiter_b.await.unwrap();
        assert!(matches!(r_b, AcquireResult::Timeout));

        // After B times out, C must be allowed to enqueue (not rejected).
        // Release A's permit so C can eventually receive it.
        drop(permit);
        let r_c = q.acquire(key("c")).await;
        assert!(
            matches!(r_c, AcquireResult::Permit(_)),
            "expected Permit for C after timed-out B was pruned, got Rejected"
        );
    }

    #[tokio::test]
    async fn shed_oldest_evicts_first_waiter() {
        // max_concurrent=1, max_queue_depth=1, ShedOldest.
        let q = Arc::new(GlobalQueue::new(cfg(
            1,
            1,
            2000,
            OverloadPolicy::ShedOldest,
        )));

        let permit = q.acquire(key("a")).await;
        assert!(matches!(permit, AcquireResult::Permit(_)));

        // First waiter fills the queue depth slot.
        let q2 = Arc::clone(&q);
        let waiter1 = tokio::spawn(async move { q2.acquire(key("b")).await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(q.queued(), 1);

        // Second request arrives: queue full → shed waiter1, enqueue waiter2.
        let q3 = Arc::clone(&q);
        let waiter2 = tokio::spawn(async move { q3.acquire(key("c")).await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // waiter1 should have been shed.
        let r1 = waiter1.await.unwrap();
        assert!(matches!(r1, AcquireResult::Shed));

        // Release the original permit — waiter2 should receive it.
        drop(permit);
        let r2 = waiter2.await.unwrap();
        assert!(matches!(r2, AcquireResult::Permit(_)));
        assert_eq!(q.metrics.shed_total.load(Ordering::Relaxed), 1);
        assert_eq!(q.metrics.permit_total.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn queued_handoff_prefers_different_client_when_possible() {
        let q = Arc::new(GlobalQueue::new(cfg(1, 4, 1000, OverloadPolicy::Reject)));

        let permit = match q.acquire(key("client-a")).await {
            AcquireResult::Permit(p) => p,
            other => panic!("expected initial permit, got {:?}", core::mem::discriminant(&other)),
        };

        let q2 = Arc::clone(&q);
        let waiter_a2 = tokio::spawn(async move { q2.acquire(key("client-a")).await });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let q3 = Arc::clone(&q);
        let waiter_b = tokio::spawn(async move { q3.acquire(key("client-b")).await });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        drop(permit);

        let permit_b = waiter_b.await.unwrap();
        assert!(matches!(permit_b, AcquireResult::Permit(_)));
        let AcquireResult::Permit(permit_b) = permit_b else { unreachable!() };
        drop(permit_b);

        let permit_a2 = waiter_a2.await.unwrap();
        assert!(matches!(permit_a2, AcquireResult::Permit(_)));
    }
}
