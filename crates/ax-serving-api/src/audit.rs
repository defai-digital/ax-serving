//! Lightweight in-process audit log for admin and control-plane actions.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tracing::warn;

const DEFAULT_AUDIT_CAPACITY: usize = 256;

#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    pub ts_unix_secs: u64,
    pub actor: String,
    pub action: String,
    pub target_type: String,
    pub target_id: Option<String>,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct AuditLog {
    capacity: usize,
    events: Mutex<VecDeque<AuditEvent>>,
}

impl AuditLog {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity: capacity.max(1),
            events: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
        })
    }

    pub fn default_shared() -> Arc<Self> {
        Self::new(DEFAULT_AUDIT_CAPACITY)
    }

    pub fn record(
        &self,
        actor: impl Into<String>,
        action: impl Into<String>,
        target_type: impl Into<String>,
        target_id: Option<String>,
        outcome: impl Into<String>,
        detail: Option<serde_json::Value>,
    ) {
        let mut events = self.events_lock();
        if events.len() >= self.capacity {
            events.pop_front();
        }
        events.push_back(AuditEvent {
            ts_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            actor: actor.into(),
            action: action.into(),
            target_type: target_type.into(),
            target_id,
            outcome: outcome.into(),
            detail,
        });
    }

    pub fn tail(&self, limit: usize) -> Vec<AuditEvent> {
        let events = self.events_lock();
        let len = events.len();
        let take = limit.min(len);
        events
            .iter()
            .skip(len.saturating_sub(take))
            .cloned()
            .collect()
    }

    fn events_lock(&self) -> MutexGuard<'_, VecDeque<AuditEvent>> {
        match self.events.lock() {
            Ok(guard) => guard,
            Err(err) => {
                warn!(%err, "audit log lock poisoned; continuing with poisoned state");
                err.into_inner()
            }
        }
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_AUDIT_CAPACITY,
            events: Mutex::new(VecDeque::with_capacity(DEFAULT_AUDIT_CAPACITY)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_returns_recent_entries_in_order() {
        let log = AuditLog::new(3);
        log.record("a", "one", "x", None, "ok", None);
        log.record("a", "two", "x", None, "ok", None);
        log.record("a", "three", "x", None, "ok", None);
        log.record("a", "four", "x", None, "ok", None);

        let tail = log.tail(3);
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0].action, "two");
        assert_eq!(tail[2].action, "four");
    }
}
