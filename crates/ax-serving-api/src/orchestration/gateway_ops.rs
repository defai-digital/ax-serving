//! Process-wide gateway operational state: readiness, routability, and drain.
//!
//! Keep this module free of Axum handlers so unit tests can exercise the pure
//! state machine without starting listeners.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// How `/readyz` decides process readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyzMode {
    /// Configuration, listeners, and fleet store only (production default).
    ControlPlane,
    /// Legacy: also require at least one eligible worker (Fabric migration).
    EligibleWorkers,
}

impl ReadyzMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "control_plane" | "control-plane" | "controlplane" => Ok(Self::ControlPlane),
            "eligible_workers" | "eligible-workers" | "eligibleworkers" | "legacy" => {
                Ok(Self::EligibleWorkers)
            }
            other => Err(format!(
                "unknown readyz mode '{other}'; expected control_plane or eligible_workers"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FleetStoreHealthState {
    Ready,
    Stale,
    Unavailable,
}

#[derive(Debug)]
pub struct FleetStoreHealth {
    pub last_success_unix_ms: AtomicU64,
    pub consecutive_failures: AtomicU64,
    /// When true, the in-memory store is always ready after init.
    pub always_ready: AtomicBool,
}

impl FleetStoreHealth {
    pub fn memory() -> Self {
        Self {
            last_success_unix_ms: AtomicU64::new(0),
            consecutive_failures: AtomicU64::new(0),
            always_ready: AtomicBool::new(true),
        }
    }

    pub fn redis() -> Self {
        Self {
            last_success_unix_ms: AtomicU64::new(0),
            consecutive_failures: AtomicU64::new(0),
            always_ready: AtomicBool::new(false),
        }
    }

    pub fn record_success(&self, now_unix_ms: u64) {
        self.last_success_unix_ms
            .store(now_unix_ms, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn evaluate(&self, now_unix_ms: u64, max_stale_ms: u64) -> FleetStoreHealthState {
        if self.always_ready.load(Ordering::Relaxed) {
            return FleetStoreHealthState::Ready;
        }
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        let last = self.last_success_unix_ms.load(Ordering::Relaxed);
        if last == 0 {
            return if failures > 0 {
                FleetStoreHealthState::Unavailable
            } else {
                FleetStoreHealthState::Stale
            };
        }
        let age = now_unix_ms.saturating_sub(last);
        if age > max_stale_ms {
            if failures >= 3 {
                FleetStoreHealthState::Unavailable
            } else {
                FleetStoreHealthState::Stale
            }
        } else {
            FleetStoreHealthState::Ready
        }
    }
}

/// Validated shutdown deadline triple (milliseconds-friendly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownDeadlines {
    pub propagation_ms: u64,
    pub drain_secs: u64,
    pub hard_secs: u64,
}

impl ShutdownDeadlines {
    pub fn new(propagation_ms: u64, drain_secs: u64, hard_secs: u64) -> Result<Self, String> {
        let combined = propagation_ms.saturating_add(drain_secs.saturating_mul(1000));
        let hard_ms = hard_secs.saturating_mul(1000);
        if hard_ms <= combined {
            return Err(format!(
                "shutdown_hard_secs ({hard_secs}) must exceed propagation_ms + drain_secs \
                 ({propagation_ms}ms + {drain_secs}s = {combined}ms)"
            ));
        }
        Ok(Self {
            propagation_ms,
            drain_secs,
            hard_secs,
        })
    }

    pub fn validate_termination_grace(self, termination_grace_secs: u64) -> Result<(), String> {
        if termination_grace_secs <= self.hard_secs {
            return Err(format!(
                "terminationGracePeriodSeconds ({termination_grace_secs}) must be greater than \
                 shutdown_hard_secs ({})",
                self.hard_secs
            ));
        }
        Ok(())
    }

    /// Hard process-exit deadline measured from when shutdown begins (SIGTERM/SIGINT),
    /// never from process start. Callers must pass the Instant captured at drain start.
    pub fn hard_deadline_at(self, shutdown_started_at: Instant) -> Instant {
        shutdown_started_at + std::time::Duration::from_secs(self.hard_secs)
    }

    /// Remaining time until [`Self::hard_deadline_at`], clamped to zero if already past.
    pub fn remaining_until_hard(self, shutdown_started_at: Instant, now: Instant) -> std::time::Duration {
        self.hard_deadline_at(shutdown_started_at)
            .saturating_duration_since(now)
    }
}

#[derive(Debug)]
pub struct GatewayOperationalState {
    pub draining: AtomicBool,
    pub accepted_inflight: AtomicU64,
    pub listeners_ready: AtomicBool,
    pub config_validated: AtomicBool,
    pub fleet_store_health: FleetStoreHealth,
    pub started_at: Instant,
    pub readyz_mode: ReadyzMode,
    pub fleet_store_ready_max_stale_ms: u64,
    pub shutdown: ShutdownDeadlines,
}

impl GatewayOperationalState {
    pub fn new(
        fleet_store_kind: &str,
        readyz_mode: ReadyzMode,
        fleet_store_ready_max_stale_ms: u64,
        shutdown: ShutdownDeadlines,
    ) -> Self {
        let fleet_store_health = if fleet_store_kind.eq_ignore_ascii_case("redis") {
            FleetStoreHealth::redis()
        } else {
            FleetStoreHealth::memory()
        };
        Self {
            draining: AtomicBool::new(false),
            accepted_inflight: AtomicU64::new(0),
            listeners_ready: AtomicBool::new(false),
            config_validated: AtomicBool::new(false),
            fleet_store_health,
            started_at: Instant::now(),
            readyz_mode,
            fleet_store_ready_max_stale_ms: fleet_store_ready_max_stale_ms.max(1),
            shutdown,
        }
    }

    pub fn mark_config_validated(&self) {
        self.config_validated.store(true, Ordering::Relaxed);
    }

    pub fn mark_listeners_ready(&self) {
        self.listeners_ready.store(true, Ordering::Relaxed);
    }

    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::SeqCst);
    }

    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    pub fn inflight(&self) -> u64 {
        self.accepted_inflight.load(Ordering::Relaxed)
    }

    pub fn live_response(&self) -> LiveResponse {
        LiveResponse {
            status: "live".into(),
        }
    }

    pub fn ready_assessment(
        &self,
        now_unix_ms: u64,
        eligible_workers: usize,
    ) -> ReadyAssessment {
        if !self.config_validated.load(Ordering::Relaxed) {
            return ReadyAssessment::not_ready("starting", 1);
        }
        if !self.listeners_ready.load(Ordering::Relaxed) {
            return ReadyAssessment::not_ready("starting", 1);
        }
        if self.is_draining() {
            return ReadyAssessment::not_ready("draining", 5);
        }
        match self
            .fleet_store_health
            .evaluate(now_unix_ms, self.fleet_store_ready_max_stale_ms)
        {
            FleetStoreHealthState::Ready => {}
            FleetStoreHealthState::Stale => {
                return ReadyAssessment::not_ready("fleet_store_stale", 5);
            }
            FleetStoreHealthState::Unavailable => {
                return ReadyAssessment::not_ready("fleet_store_unavailable", 5);
            }
        }
        if self.readyz_mode == ReadyzMode::EligibleWorkers && eligible_workers == 0 {
            return ReadyAssessment::not_ready("no_eligible_workers", 5);
        }
        ReadyAssessment {
            ready: true,
            status: "ready".into(),
            reason: None,
            retry_after_seconds: None,
            fleet_store: "ready".into(),
            draining: false,
            eligible_workers,
        }
    }

    pub fn routable_assessment(&self, eligible_workers: usize) -> RoutableAssessment {
        if eligible_workers > 0 {
            RoutableAssessment {
                routable: true,
                status: "routable".into(),
                retry_after_seconds: None,
                eligible_workers,
            }
        } else {
            RoutableAssessment {
                routable: false,
                status: "not_routable".into(),
                retry_after_seconds: Some(5),
                eligible_workers: 0,
            }
        }
    }
}

/// RAII guard that tracks accepted inference work for drain.
///
/// Owned (`Arc`) so it can be moved into a streaming response body and held
/// until the stream ends — not only until the Axum handler returns.
pub struct AcceptedRequestGuard {
    state: Arc<GatewayOperationalState>,
}

impl AcceptedRequestGuard {
    /// Admit one request if the gateway is not draining.
    ///
    /// Uses a post-increment drain re-check so concurrent `begin_drain` cannot
    /// leave newly accepted work counted after the drain bit is set.
    pub fn try_admit(state: &Arc<GatewayOperationalState>) -> Option<Self> {
        if state.is_draining() {
            return None;
        }
        state.accepted_inflight.fetch_add(1, Ordering::Relaxed);
        // Re-check after the increment so a concurrent begin_drain is observed.
        if state.is_draining() {
            state.accepted_inflight.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(Self {
            state: Arc::clone(state),
        })
    }
}

impl Drop for AcceptedRequestGuard {
    fn drop(&mut self) {
        let _ = self.state.accepted_inflight.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_sub(1),
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LiveResponse {
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ReadyAssessment {
    pub ready: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
    pub fleet_store: String,
    pub draining: bool,
    #[serde(skip_serializing_if = "is_zero_usize")]
    pub eligible_workers: usize,
}

fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

impl ReadyAssessment {
    fn not_ready(reason: &str, retry_after_seconds: u64) -> Self {
        Self {
            ready: false,
            status: "not_ready".into(),
            reason: Some(reason.into()),
            retry_after_seconds: Some(retry_after_seconds),
            fleet_store: if reason.starts_with("fleet_store") {
                reason.to_string()
            } else {
                "unknown".into()
            },
            draining: reason == "draining",
            eligible_workers: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RoutableAssessment {
    pub routable: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
    #[serde(skip_serializing_if = "is_zero_usize")]
    pub eligible_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops(mode: ReadyzMode) -> GatewayOperationalState {
        GatewayOperationalState::new(
            "memory",
            mode,
            15_000,
            ShutdownDeadlines::new(5_000, 300, 330).unwrap(),
        )
    }

    #[test]
    fn control_plane_ready_without_workers() {
        let state = ops(ReadyzMode::ControlPlane);
        state.mark_config_validated();
        state.mark_listeners_ready();
        let assessment = state.ready_assessment(1_000_000, 0);
        assert!(assessment.ready);
        assert_eq!(assessment.status, "ready");
        let routable = state.routable_assessment(0);
        assert!(!routable.routable);
    }

    #[test]
    fn legacy_readyz_requires_workers() {
        let state = ops(ReadyzMode::EligibleWorkers);
        state.mark_config_validated();
        state.mark_listeners_ready();
        let assessment = state.ready_assessment(1_000_000, 0);
        assert!(!assessment.ready);
        assert_eq!(assessment.reason.as_deref(), Some("no_eligible_workers"));
        let with_workers = state.ready_assessment(1_000_000, 2);
        assert!(with_workers.ready);
    }

    #[test]
    fn drain_blocks_admission_and_readyz() {
        let state = Arc::new(ops(ReadyzMode::ControlPlane));
        state.mark_config_validated();
        state.mark_listeners_ready();
        let guard = AcceptedRequestGuard::try_admit(&state).expect("admit");
        assert_eq!(state.inflight(), 1);
        state.begin_drain();
        assert!(AcceptedRequestGuard::try_admit(&state).is_none());
        let assessment = state.ready_assessment(1_000_000, 0);
        assert!(!assessment.ready);
        assert_eq!(assessment.reason.as_deref(), Some("draining"));
        drop(guard);
        assert_eq!(state.inflight(), 0);
    }

    #[test]
    fn admission_guard_is_owned_and_survives_handler_return_semantics() {
        // Streaming responses must keep the guard after the handler returns.
        // An owned Arc-backed guard can be moved into a stream task; a borrow
        // tied to the handler stack cannot.
        let state = Arc::new(ops(ReadyzMode::ControlPlane));
        let guard = AcceptedRequestGuard::try_admit(&state).expect("admit");
        assert_eq!(state.inflight(), 1);
        let moved = std::thread::spawn(move || {
            // Guard still owns the slot across the move (stream body analogue).
            guard
        })
        .join()
        .expect("join");
        assert_eq!(state.inflight(), 1);
        drop(moved);
        assert_eq!(state.inflight(), 0);
    }

    #[test]
    fn shutdown_deadline_validation() {
        assert!(ShutdownDeadlines::new(5_000, 300, 330).is_ok());
        assert!(ShutdownDeadlines::new(5_000, 300, 300).is_err());
        let d = ShutdownDeadlines::new(5_000, 300, 330).unwrap();
        assert!(d.validate_termination_grace(360).is_ok());
        assert!(d.validate_termination_grace(330).is_err());
    }

    #[test]
    fn hard_deadline_is_relative_to_shutdown_start_not_process_start() {
        let d = ShutdownDeadlines::new(5_000, 300, 330).unwrap();
        // Simulate a process that has already been up for an hour before SIGTERM.
        let process_start = Instant::now();
        let shutdown_started = process_start + std::time::Duration::from_secs(3_600);
        let deadline = d.hard_deadline_at(shutdown_started);

        assert_eq!(
            deadline.duration_since(shutdown_started),
            std::time::Duration::from_secs(330),
            "hard deadline must be hard_secs after shutdown begins"
        );
        // A bug that starts the hard timer at process start would expire ~330s after
        // process_start, which is far in the past relative to shutdown_started.
        let wrong_process_start_deadline = process_start + std::time::Duration::from_secs(330);
        assert!(
            deadline > wrong_process_start_deadline,
            "hard deadline must not be measured from process start"
        );
        assert_eq!(
            d.remaining_until_hard(shutdown_started, shutdown_started),
            std::time::Duration::from_secs(330)
        );
        assert_eq!(
            d.remaining_until_hard(shutdown_started, deadline),
            std::time::Duration::ZERO
        );
    }

    #[test]
    fn redis_fleet_store_stale_without_success() {
        let health = FleetStoreHealth::redis();
        assert_eq!(
            health.evaluate(10_000, 15_000),
            FleetStoreHealthState::Stale
        );
        health.record_success(10_000);
        assert_eq!(
            health.evaluate(20_000, 15_000),
            FleetStoreHealthState::Ready
        );
        assert_eq!(
            health.evaluate(30_000, 15_000),
            FleetStoreHealthState::Stale
        );
    }
}
