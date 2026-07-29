//! Multi-replica aggregation for complete Mac cluster gangs.
//!
//! A replica is one independently runnable complete cluster (all required ranks
//! ready at one generation). Partial-rank gangs never contribute admission
//! capacity. Degraded-but-safe means at least one complete replica remains ready.

use ax_serving_protocol::{ClusterLifecycleState, DomainId};
use serde::{Deserialize, Serialize};

/// Bounded observation for one complete cluster replica behind a domain/pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterReplicaObservation {
    pub replica_id: String,
    pub domain_id: DomainId,
    pub generation: u64,
    pub state: ClusterLifecycleState,
    pub required_ranks: u16,
    pub ready_ranks: u16,
    pub max_concurrent_requests: u64,
    pub active_requests: u64,
}

/// Aggregate readiness for one logical Mac cluster domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DomainReplicaAggregate {
    pub domain_id: DomainId,
    pub ready_replicas: u32,
    pub total_replicas: u32,
    pub admits_requests: bool,
    /// True when some but not all complete replicas are ready.
    pub degraded_complete_replica_ha: bool,
    pub aggregate_max_concurrent_requests: u64,
    pub aggregate_active_requests: u64,
    pub ready_generations: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplicaAggregateError {
    #[error("replica_id must be a non-empty bounded token")]
    InvalidReplicaId,
    #[error("required_ranks must be greater than zero")]
    ZeroRequiredRanks,
    #[error("ready_ranks cannot exceed required_ranks")]
    ReadyExceedsRequired,
    #[error("mixed domain ids cannot be aggregated")]
    MixedDomains,
}

impl ClusterReplicaObservation {
    pub fn validate(&self) -> Result<(), ReplicaAggregateError> {
        if self.replica_id.is_empty()
            || self.replica_id.len() > 128
            || !self.replica_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
            })
        {
            return Err(ReplicaAggregateError::InvalidReplicaId);
        }
        if self.required_ranks == 0 {
            return Err(ReplicaAggregateError::ZeroRequiredRanks);
        }
        if self.ready_ranks > self.required_ranks {
            return Err(ReplicaAggregateError::ReadyExceedsRequired);
        }
        Ok(())
    }

    /// A replica admits only when its complete gang is ready.
    pub fn is_complete_ready(&self) -> bool {
        self.state == ClusterLifecycleState::Ready
            && self.ready_ranks == self.required_ranks
            && self.required_ranks > 0
    }
}

/// Aggregate multiple complete-cluster replicas. Partial gangs contribute zero capacity.
pub fn aggregate_replicas(
    domain_id: &DomainId,
    replicas: &[ClusterReplicaObservation],
) -> Result<DomainReplicaAggregate, ReplicaAggregateError> {
    for replica in replicas {
        replica.validate()?;
        if &replica.domain_id != domain_id {
            return Err(ReplicaAggregateError::MixedDomains);
        }
    }

    let ready: Vec<&ClusterReplicaObservation> = replicas
        .iter()
        .filter(|replica| replica.is_complete_ready())
        .collect();
    let aggregate_max = ready
        .iter()
        .try_fold(0_u64, |total, replica| {
            total.checked_add(replica.max_concurrent_requests)
        })
        .unwrap_or(u64::MAX);
    let aggregate_active = ready
        .iter()
        .try_fold(0_u64, |total, replica| {
            total.checked_add(replica.active_requests)
        })
        .unwrap_or(u64::MAX);
    let ready_generations = ready
        .iter()
        .map(|replica| replica.generation)
        .collect::<Vec<_>>();

    Ok(DomainReplicaAggregate {
        domain_id: domain_id.clone(),
        ready_replicas: ready.len() as u32,
        total_replicas: replicas.len() as u32,
        admits_requests: !ready.is_empty(),
        degraded_complete_replica_ha: !ready.is_empty() && ready.len() < replicas.len(),
        aggregate_max_concurrent_requests: aggregate_max,
        aggregate_active_requests: aggregate_active,
        ready_generations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_serving_protocol::DomainId;

    fn domain() -> DomainId {
        DomainId::new("mac-cluster-main").unwrap()
    }

    fn replica(id: &str, ready: bool, ready_ranks: u16) -> ClusterReplicaObservation {
        ClusterReplicaObservation {
            replica_id: id.into(),
            domain_id: domain(),
            generation: 3,
            state: if ready {
                ClusterLifecycleState::Ready
            } else {
                ClusterLifecycleState::Failed
            },
            required_ranks: 2,
            ready_ranks,
            max_concurrent_requests: 4,
            active_requests: 1,
        }
    }

    #[test]
    fn partial_rank_gang_never_admits() {
        let partial = replica("a", true, 1);
        assert!(!partial.is_complete_ready());
        let aggregate = aggregate_replicas(&domain(), &[partial]).unwrap();
        assert!(!aggregate.admits_requests);
        assert_eq!(aggregate.ready_replicas, 0);
    }

    #[test]
    fn degraded_means_some_complete_replicas_remain() {
        let replicas = vec![
            replica("a", true, 2),
            replica("b", false, 0),
            replica("c", true, 2),
        ];
        let aggregate = aggregate_replicas(&domain(), &replicas).unwrap();
        assert!(aggregate.admits_requests);
        assert!(aggregate.degraded_complete_replica_ha);
        assert_eq!(aggregate.ready_replicas, 2);
        assert_eq!(aggregate.aggregate_max_concurrent_requests, 8);
    }

    #[test]
    fn mixed_domains_fail_closed() {
        let mut other = replica("a", true, 2);
        other.domain_id = DomainId::new("other").unwrap();
        assert_eq!(
            aggregate_replicas(&domain(), &[other]),
            Err(ReplicaAggregateError::MixedDomains)
        );
    }
}
