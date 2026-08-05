//! Operator-facing cluster status projection.
//!
//! Distinguishes configured/incomplete/ready/degraded(complete-replica HA)/
//! draining/failed and experimental versus certified without exposing ranks as
//! routable AX workers.

use ax_serving_protocol::{ClusterLifecycleState, QualificationState};
use serde::Serialize;

use crate::coordinator::ClusterStatus;
use crate::replicas::DomainReplicaAggregate;

/// Coarse operator status for dashboards and CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorClusterPhase {
    ConfiguredNoCoordinator,
    CoordinatorGangIncomplete,
    Downloading,
    Connecting,
    Loading,
    Warming,
    Ready,
    /// At least one complete replica remains ready; never partial-rank admit.
    DegradedCompleteReplicaHa,
    Draining,
    FailedOrFenced,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OperatorClusterView {
    pub phase: OperatorClusterPhase,
    pub qualification: QualificationState,
    pub experimental: bool,
    pub certified: bool,
    pub admits_requests: bool,
    pub generation: u64,
    pub ready_ranks: usize,
    pub required_ranks: usize,
    pub ready_replicas: Option<u32>,
    pub total_replicas: Option<u32>,
    pub reason: &'static str,
}

/// Map one coordinator gang into an operator-facing phase.
pub fn operator_view_from_gang(
    status: &ClusterStatus,
    qualification: QualificationState,
    coordinator_present: bool,
) -> OperatorClusterView {
    let phase = if !coordinator_present {
        OperatorClusterPhase::ConfiguredNoCoordinator
    } else {
        match status.state {
            ClusterLifecycleState::Planned if status.ready_ranks == 0 => {
                OperatorClusterPhase::CoordinatorGangIncomplete
            }
            ClusterLifecycleState::Planned => OperatorClusterPhase::CoordinatorGangIncomplete,
            ClusterLifecycleState::Downloading => OperatorClusterPhase::Downloading,
            ClusterLifecycleState::Connecting => OperatorClusterPhase::Connecting,
            ClusterLifecycleState::Loading => OperatorClusterPhase::Loading,
            ClusterLifecycleState::Warming => OperatorClusterPhase::Warming,
            ClusterLifecycleState::Ready => OperatorClusterPhase::Ready,
            ClusterLifecycleState::Draining => OperatorClusterPhase::Draining,
            ClusterLifecycleState::Stopped | ClusterLifecycleState::Failed => {
                OperatorClusterPhase::FailedOrFenced
            }
        }
    };
    let experimental = !matches!(qualification, QualificationState::Certified);
    let certified = matches!(qualification, QualificationState::Certified);
    OperatorClusterView {
        phase,
        qualification,
        experimental,
        certified,
        admits_requests: status.state.admits_requests() && phase == OperatorClusterPhase::Ready,
        generation: status.generation,
        ready_ranks: status.ready_ranks,
        required_ranks: status.required_ranks,
        ready_replicas: None,
        total_replicas: None,
        reason: phase_reason(phase),
    }
}

/// Map multi-replica aggregation into operator status. Partial ranks never admit.
pub fn operator_view_from_replicas(
    aggregate: &DomainReplicaAggregate,
    generation: u64,
    qualification: QualificationState,
) -> OperatorClusterView {
    let phase = if aggregate.total_replicas == 0 {
        OperatorClusterPhase::ConfiguredNoCoordinator
    } else if aggregate.ready_replicas == 0 {
        OperatorClusterPhase::FailedOrFenced
    } else if aggregate.degraded_complete_replica_ha {
        OperatorClusterPhase::DegradedCompleteReplicaHa
    } else {
        OperatorClusterPhase::Ready
    };
    let experimental = !matches!(qualification, QualificationState::Certified);
    let certified = matches!(qualification, QualificationState::Certified);
    OperatorClusterView {
        phase,
        qualification,
        experimental,
        certified,
        admits_requests: aggregate.admits_requests,
        generation,
        ready_ranks: 0,
        required_ranks: 0,
        ready_replicas: Some(aggregate.ready_replicas),
        total_replicas: Some(aggregate.total_replicas),
        reason: phase_reason(phase),
    }
}

const fn phase_reason(phase: OperatorClusterPhase) -> &'static str {
    match phase {
        OperatorClusterPhase::ConfiguredNoCoordinator => "configured_but_no_coordinator",
        OperatorClusterPhase::CoordinatorGangIncomplete => "coordinator_present_gang_incomplete",
        OperatorClusterPhase::Downloading => "downloading",
        OperatorClusterPhase::Connecting => "connecting",
        OperatorClusterPhase::Loading => "loading",
        OperatorClusterPhase::Warming => "warming",
        OperatorClusterPhase::Ready => "ready",
        OperatorClusterPhase::DegradedCompleteReplicaHa => {
            "degraded_complete_replica_ha_still_safe"
        }
        OperatorClusterPhase::Draining => "draining",
        OperatorClusterPhase::FailedOrFenced => "failed_or_fenced",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_serving_protocol::{ClusterLifecycleState, DomainId};
    use crate::replicas::{ClusterReplicaObservation, aggregate_replicas};

    #[test]
    fn experimental_flag_is_derived_from_qualification() {
        let status = ClusterStatus {
            cluster_id: "mac".into(),
            generation: 1,
            manifest_digest: "sha256:aa".into(),
            state: ClusterLifecycleState::Ready,
            ready_ranks: 2,
            required_ranks: 2,
            ranks: vec![],
        };
        let view =
            operator_view_from_gang(&status, QualificationState::Experimental, true);
        assert!(view.experimental);
        assert!(!view.certified);
        assert_eq!(view.phase, OperatorClusterPhase::Ready);
    }

    #[test]
    fn degraded_requires_complete_replica_not_partial_ranks() {
        let domain = DomainId::new("mac-cluster-main").unwrap();
        let replicas = vec![
            ClusterReplicaObservation {
                replica_id: "a".into(),
                domain_id: domain.clone(),
                generation: 3,
                state: ClusterLifecycleState::Ready,
                required_ranks: 2,
                ready_ranks: 2,
                max_concurrent_requests: 2,
                active_requests: 0,
            },
            ClusterReplicaObservation {
                replica_id: "b".into(),
                domain_id: domain.clone(),
                generation: 3,
                state: ClusterLifecycleState::Failed,
                required_ranks: 2,
                ready_ranks: 0,
                max_concurrent_requests: 2,
                active_requests: 0,
            },
        ];
        let aggregate = aggregate_replicas(&domain, &replicas).unwrap();
        let view = operator_view_from_replicas(&aggregate, 3, QualificationState::Certified);
        assert_eq!(view.phase, OperatorClusterPhase::DegradedCompleteReplicaHa);
        assert!(view.admits_requests);
        assert!(view.certified);
    }
}
