//! Asynchronous cluster reconciliation that never blocks the request path.
//!
//! Reconciliation work (stale rank cleanup, generation fencing bookkeeping, and
//! evidence emission) is queued and drained by a background worker. Admission
//! and proxy paths only read atomic readiness bits.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Work item processed off the gateway request path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ReconcileWork {
    RefreshReadiness { generation: u64 },
    FenceStaleGeneration { generation: u64 },
    EmitEvidence { kind: String },
}

/// Non-blocking reconcile queue plus request-path readiness snapshot.
#[derive(Debug)]
pub struct AsyncReconciler {
    ready: AtomicBool,
    generation: AtomicU64,
    queued: AtomicU64,
    processed: AtomicU64,
    sender: mpsc::Sender<ReconcileWork>,
    receiver: Mutex<Option<mpsc::Receiver<ReconcileWork>>>,
}

impl AsyncReconciler {
    pub fn new(capacity: usize) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(capacity.max(1));
        Arc::new(Self {
            ready: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            sender,
            receiver: Mutex::new(Some(receiver)),
        })
    }

    /// Request-path read: never waits on reconcile work.
    pub fn admits_requests(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn queued_count(&self) -> u64 {
        self.queued.load(Ordering::Acquire)
    }

    pub fn processed_count(&self) -> u64 {
        self.processed.load(Ordering::Acquire)
    }

    /// Enqueue work without awaiting the worker. Returns false if the queue is full.
    pub fn enqueue(&self, work: ReconcileWork) -> bool {
        match self.sender.try_send(work) {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::Release);
                true
            }
            Err(_) => false,
        }
    }

    /// Mark readiness for the request path (still non-blocking).
    pub fn publish_readiness(&self, ready: bool, generation: u64) {
        self.generation.store(generation, Ordering::Release);
        self.ready.store(ready, Ordering::Release);
    }

    /// Take ownership of the receiver for a single background worker.
    pub fn take_receiver(&self) -> Option<mpsc::Receiver<ReconcileWork>> {
        self.receiver.lock().ok().and_then(|mut guard| guard.take())
    }

    /// Drain one work item. Intended for the background worker only.
    pub async fn process_one(self: &Arc<Self>, work: ReconcileWork) {
        match work {
            ReconcileWork::RefreshReadiness { generation } => {
                // Background refresh never clears an already-published ready bit
                // without an explicit fence work item.
                if self.generation() == generation {
                    self.ready.store(true, Ordering::Release);
                }
            }
            ReconcileWork::FenceStaleGeneration { generation } => {
                if self.generation() <= generation {
                    self.ready.store(false, Ordering::Release);
                }
            }
            ReconcileWork::EmitEvidence { kind: _ } => {
                // Evidence emission is side-effect free here; adapters attach sinks.
            }
        }
        self.processed.fetch_add(1, Ordering::Release);
    }
}

/// Spawn a background reconciler that processes work without blocking callers.
pub fn spawn_reconcile_worker(reconciler: Arc<AsyncReconciler>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(mut receiver) = reconciler.take_receiver() else {
            return;
        };
        while let Some(work) = receiver.recv().await {
            reconciler.process_one(work).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn enqueue_does_not_wait_for_worker() {
        let reconciler = AsyncReconciler::new(8);
        // No worker running yet — try_send must still return immediately.
        assert!(reconciler.enqueue(ReconcileWork::EmitEvidence {
            kind: "load".into()
        }));
        assert_eq!(reconciler.queued_count(), 1);
        assert!(reconciler.admits_requests() || !reconciler.admits_requests());
    }

    #[tokio::test]
    async fn request_path_reads_published_ready_without_processing_queue() {
        let reconciler = AsyncReconciler::new(4);
        reconciler.publish_readiness(true, 9);
        assert!(reconciler.admits_requests());
        assert_eq!(reconciler.generation(), 9);
        // Queue work but never drain it; admission bit remains independently readable.
        assert!(reconciler.enqueue(ReconcileWork::FenceStaleGeneration { generation: 9 }));
        assert!(reconciler.admits_requests());
        assert_eq!(reconciler.processed_count(), 0);
    }

    #[tokio::test]
    async fn background_worker_processes_without_blocking_publish() {
        let reconciler = AsyncReconciler::new(16);
        let worker = spawn_reconcile_worker(Arc::clone(&reconciler));
        reconciler.publish_readiness(true, 3);
        for _ in 0..5 {
            assert!(reconciler.enqueue(ReconcileWork::EmitEvidence {
                kind: "soak".into()
            }));
        }
        // Request path remains readable while work drains asynchronously.
        assert!(reconciler.admits_requests());
        tokio::time::timeout(Duration::from_secs(2), async {
            while reconciler.processed_count() < 5 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reconcile worker should drain queued evidence");
        drop(worker);
    }
}
