use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{DeploymentId, DeploymentSpec, JobId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentDesiredState {
    Enabled,
    Disabled,
    Absent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentObservedState {
    Pending,
    Ready,
    Draining,
    Absent,
    ExternallyManaged,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentJobAction {
    Create,
    Update,
    Roll,
    Drain,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentJobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
}

/// Durable desired-state record shared by active gateway replicas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentControlRecord {
    pub deployment: DeploymentSpec,
    pub generation: u64,
    pub desired_state: DeploymentDesiredState,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

/// Bounded operator-visible lifecycle result. Runtime-specific stack traces,
/// URLs, credentials, and model paths never belong in this record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentJobRecord {
    pub id: JobId,
    pub deployment_id: DeploymentId,
    pub action: DeploymentJobAction,
    pub status: DeploymentJobStatus,
    pub desired_state: DeploymentDesiredState,
    pub observed_state: DeploymentObservedState,
    pub progress_percent: u8,
    pub generation: u64,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub completed_at: Option<OffsetDateTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentCommand {
    pub job_id: JobId,
    pub action: DeploymentJobAction,
    pub deployment: DeploymentSpec,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentJobObservation {
    pub job_id: JobId,
    pub status: DeploymentJobStatus,
    pub observed_state: DeploymentObservedState,
    pub progress_percent: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_message: Option<String>,
}

impl DeploymentJobRecord {
    pub fn queued(
        deployment_id: DeploymentId,
        action: DeploymentJobAction,
        desired_state: DeploymentDesiredState,
        generation: u64,
    ) -> Self {
        let now = OffsetDateTime::now_utc();
        Self {
            id: JobId::new(),
            deployment_id,
            action,
            status: DeploymentJobStatus::Queued,
            desired_state,
            observed_state: DeploymentObservedState::Pending,
            progress_percent: 0,
            generation,
            created_at: now,
            updated_at: now,
            completed_at: None,
            failure_code: None,
            failure_message: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{
        DeploymentDesiredState, DeploymentId, DeploymentJobAction, DeploymentJobRecord,
        DeploymentSpec, IdentityPolicy, LogicalModelId, PoolId, RuntimeModelId,
    };

    #[test]
    fn lifecycle_records_round_trip_without_losing_identity() {
        let deployment = DeploymentSpec {
            id: DeploymentId::new("qwen-cuda").unwrap(),
            logical_model: LogicalModelId::new("qwen/code").unwrap(),
            pool: PoolId::new("cuda").unwrap(),
            domain: None,
            runtime_model_id: RuntimeModelId::new("Qwen/Qwen3").unwrap(),
            equivalence_class: None,
            expected_identity: None,
            required_identity: IdentityPolicy {
                required_matching_fields: BTreeSet::new(),
            },
            required_capabilities: BTreeSet::new(),
            enabled: true,
        };
        let job = DeploymentJobRecord::queued(
            deployment.id,
            DeploymentJobAction::Create,
            DeploymentDesiredState::Enabled,
            1,
        );
        let encoded = serde_json::to_vec(&job).unwrap();
        let decoded: DeploymentJobRecord = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, job);
    }
}
