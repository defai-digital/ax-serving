use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{
    DeploymentCommand, DeploymentJobObservation, DomainObservation, ExecutionDomainDescriptor,
    PoolId, ProtocolCapability, ProtocolDescriptor, RegistrationId, RuntimeModelDescriptor,
    TrustDomainId, WorkerId, WorkerInstanceId,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDescriptor {
    pub name: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_sha: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerDescriptor {
    pub id: WorkerId,
    pub instance_id: WorkerInstanceId,
    pub advertise_url: String,
    pub pool_id: PoolId,
    pub trust_domain: TrustDomainId,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDescriptor {
    pub kind: String,
    pub version: String,
    pub api: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareDescriptor {
    pub platform: String,
    pub accelerator: String,
    #[serde(default)]
    pub device_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_class: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeState {
    Starting,
    Ready,
    Degraded,
    Draining,
    Unavailable,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub ready: bool,
    pub state: RuntimeState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_latency_ms: Option<u64>,
}

impl RuntimeStatus {
    pub fn ready() -> Self {
        Self {
            ready: true,
            state: RuntimeState::Ready,
            reason_code: None,
            message: None,
            probe_latency_ms: None,
        }
    }

    pub fn unavailable(reason_code: impl Into<String>) -> Self {
        Self {
            ready: false,
            state: RuntimeState::Unavailable,
            reason_code: Some(reason_code.into()),
            message: None,
            probe_latency_ms: None,
        }
    }

    pub fn is_consistent(&self) -> bool {
        self.ready == matches!(self.state, RuntimeState::Ready | RuntimeState::Degraded)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CapacityObservation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waiting_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_rss_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recent_error_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_used_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache_hit_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_token_capacity: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_tokens_in_use: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ewma_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inter_token_ewma_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_tokens_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_window_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CapacityError {
    #[error("capacity field '{field}' must be finite and non-negative")]
    InvalidNonNegative { field: &'static str },
    #[error("capacity ratio '{field}' must be finite and in [0, 1]")]
    InvalidRatio { field: &'static str },
    #[error("active requests exceed declared maximum")]
    ActiveExceedsMaximum,
    #[error("batch tokens in use exceed declared capacity")]
    BatchExceedsCapacity,
}

impl CapacityObservation {
    pub fn validate(&self) -> Result<(), CapacityError> {
        validate_ratio("kv_cache_used_ratio", self.kv_cache_used_ratio)?;
        validate_ratio("prefix_cache_hit_ratio", self.prefix_cache_hit_ratio)?;
        validate_ratio("recent_error_rate", self.recent_error_rate)?;
        validate_non_negative("ttft_ewma_ms", self.ttft_ewma_ms)?;
        validate_non_negative("inter_token_ewma_ms", self.inter_token_ewma_ms)?;
        validate_non_negative(
            "generated_tokens_per_second",
            self.generated_tokens_per_second,
        )?;
        if matches!(
            (self.active_requests, self.max_concurrent_requests),
            (Some(active), Some(maximum)) if active > maximum
        ) {
            return Err(CapacityError::ActiveExceedsMaximum);
        }
        if matches!(
            (self.batch_tokens_in_use, self.batch_token_capacity),
            (Some(in_use), Some(capacity)) if in_use > capacity
        ) {
            return Err(CapacityError::BatchExceedsCapacity);
        }
        Ok(())
    }
}

fn validate_ratio(field: &'static str, value: Option<f64>) -> Result<(), CapacityError> {
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(CapacityError::InvalidRatio { field });
    }
    Ok(())
}

fn validate_non_negative(field: &'static str, value: Option<f64>) -> Result<(), CapacityError> {
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(CapacityError::InvalidNonNegative { field });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeObservation {
    #[serde(with = "time::serde::rfc3339")]
    pub observed_at: OffsetDateTime,
    pub runtime: RuntimeStatus,
    pub inventory_generation: u64,
    #[serde(default)]
    pub models: Vec<RuntimeModelDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<CapacityObservation>,
}

impl RuntimeObservation {
    pub fn validate(&self) -> Result<(), ObservationError> {
        if !self.runtime.is_consistent() {
            return Err(ObservationError::InconsistentRuntimeStatus);
        }
        if let Some(capacity) = &self.capacity {
            capacity.validate()?;
        }
        let mut ids = BTreeSet::new();
        for model in &self.models {
            if !ids.insert(model.runtime_model_id.clone()) {
                return Err(ObservationError::DuplicateModel(
                    model.runtime_model_id.to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ObservationError {
    #[error("runtime ready flag and runtime state are inconsistent")]
    InconsistentRuntimeStatus,
    #[error(transparent)]
    InvalidCapacity(#[from] CapacityError),
    #[error("duplicate runtime model id '{0}'")]
    DuplicateModel(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterWorkerRequest {
    pub protocol: ProtocolDescriptor,
    pub agent: AgentDescriptor,
    pub worker: WorkerDescriptor,
    pub runtime: RuntimeDescriptor,
    pub hardware: HardwareDescriptor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<ExecutionDomainDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_observation: Option<DomainObservation>,
    pub observation: RuntimeObservation,
}

impl RegisterWorkerRequest {
    /// Validate additive protocol-v1.1 execution-domain and v1.2 Mac-cluster semantics.
    ///
    /// Runtime-observation validation remains available separately so v1.0
    /// callers do not need to adopt domain fields.
    pub fn validate_domain_contract(&self) -> Result<(), DomainContractError> {
        let declares_control =
            self.protocol.capabilities.iter().any(|capability| {
                capability.as_str() == ProtocolCapability::CONTROL_EXECUTION_DOMAIN
            });
        let declares_domain_capacity =
            self.protocol.capabilities.iter().any(|capability| {
                capability.as_str() == ProtocolCapability::TELEMETRY_DOMAIN_CAPACITY
            });
        let declares_mac_cluster = self
            .protocol
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == ProtocolCapability::CONTROL_MAC_CLUSTER);

        if (self.domain.is_some() || self.domain_observation.is_some() || declares_control)
            && (self.protocol.version.major != 1 || self.protocol.version.minor < 1)
        {
            return Err(DomainContractError::ProtocolMinorTooOld);
        }
        if declares_control != self.domain.is_some() {
            return Err(if declares_control {
                DomainContractError::CapabilityWithoutDescriptor
            } else {
                DomainContractError::DescriptorWithoutCapability
            });
        }
        if declares_domain_capacity && !declares_control {
            return Err(DomainContractError::CapacityCapabilityWithoutDomain);
        }

        let Some(descriptor) = &self.domain else {
            if declares_mac_cluster {
                return Err(DomainContractError::MacClusterCapabilityWithoutDomain);
            }
            if self.domain_observation.is_some() {
                return Err(DomainContractError::ObservationWithoutDescriptor);
            }
            return Ok(());
        };
        descriptor
            .validate()
            .map_err(|error| DomainContractError::InvalidDescriptor(error.to_string()))?;
        let is_mac_cluster = descriptor.kind == crate::ExecutionDomainKind::MacAxEngineCluster;
        if is_mac_cluster && (self.protocol.version.major != 1 || self.protocol.version.minor < 2) {
            return Err(DomainContractError::MacClusterProtocolMinorTooOld);
        }
        if is_mac_cluster != declares_mac_cluster {
            return Err(if is_mac_cluster {
                DomainContractError::MacClusterDescriptorWithoutCapability
            } else {
                DomainContractError::MacClusterCapabilityForOtherDomain
            });
        }
        if descriptor.pool_id != self.worker.pool_id {
            return Err(DomainContractError::PoolMismatch);
        }
        if descriptor.trust_domain != self.worker.trust_domain {
            return Err(DomainContractError::TrustDomainMismatch);
        }
        if let Some(hardware_class) = &self.hardware.hardware_class
            && hardware_class != &descriptor.hardware_class
        {
            return Err(DomainContractError::HardwareClassMismatch);
        }

        if let Some(observation) = &self.domain_observation {
            observation
                .validate()
                .map_err(|error| DomainContractError::InvalidObservation(error.to_string()))?;
            if observation.aggregate_capacity.is_some() && !declares_domain_capacity {
                return Err(DomainContractError::CapacityWithoutCapability);
            }
            if descriptor.compatibility_manifest != observation.manifest_digest {
                return Err(DomainContractError::ManifestMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DomainContractError {
    #[error("execution-domain fields require protocol major 1 minor 1 or newer")]
    ProtocolMinorTooOld,
    #[error("mac_ax_engine_cluster requires protocol major 1 minor 2 or newer")]
    MacClusterProtocolMinorTooOld,
    #[error("control.execution-domain.v1 requires an execution-domain descriptor")]
    CapabilityWithoutDescriptor,
    #[error("an execution-domain descriptor requires control.execution-domain.v1")]
    DescriptorWithoutCapability,
    #[error("telemetry.domain-capacity.v1 requires control.execution-domain.v1")]
    CapacityCapabilityWithoutDomain,
    #[error("control.mac-cluster.v1 requires an execution-domain descriptor")]
    MacClusterCapabilityWithoutDomain,
    #[error("mac_ax_engine_cluster requires control.mac-cluster.v1")]
    MacClusterDescriptorWithoutCapability,
    #[error("control.mac-cluster.v1 is valid only for mac_ax_engine_cluster")]
    MacClusterCapabilityForOtherDomain,
    #[error("a domain observation requires an execution-domain descriptor")]
    ObservationWithoutDescriptor,
    #[error("domain aggregate capacity requires telemetry.domain-capacity.v1")]
    CapacityWithoutCapability,
    #[error("invalid execution-domain descriptor: {0}")]
    InvalidDescriptor(String),
    #[error("invalid execution-domain observation: {0}")]
    InvalidObservation(String),
    #[error("execution-domain pool does not match worker pool")]
    PoolMismatch,
    #[error("execution-domain trust boundary does not match worker trust boundary")]
    TrustDomainMismatch,
    #[error("execution-domain hardware class does not match worker hardware class")]
    HardwareClassMismatch,
    #[error("execution-domain observation manifest does not match its descriptor")]
    ManifestMismatch,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LeaseToken(String);

impl LeaseToken {
    pub fn new(value: impl Into<String>) -> Result<Self, LeaseTokenError> {
        let value = value.into();
        if value.len() < 16 {
            return Err(LeaseTokenError::TooShort);
        }
        if value.len() > 4096 {
            return Err(LeaseTokenError::TooLong);
        }
        if value.chars().any(char::is_control) {
            return Err(LeaseTokenError::ControlCharacter);
        }
        Ok(Self(value))
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for LeaseToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LeaseToken([REDACTED])")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LeaseTokenError {
    #[error("lease token must contain at least 16 bytes")]
    TooShort,
    #[error("lease token exceeds 4096 bytes")]
    TooLong,
    #[error("lease token must not contain control characters")]
    ControlCharacter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterWorkerResponse {
    pub registration_id: RegistrationId,
    pub lease_token: LeaseToken,
    pub protocol: ProtocolDescriptor,
    pub heartbeat_interval_ms: u64,
    pub lease_ttl_ms: u64,
    #[serde(default)]
    pub inventory_resync: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub registration_id: RegistrationId,
    pub instance_id: WorkerInstanceId,
    pub sequence: u64,
    #[serde(with = "time::serde::rfc3339")]
    pub observed_at: OffsetDateTime,
    pub runtime: RuntimeStatus,
    pub inventory_generation: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<RuntimeModelDescriptor>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<CapacityObservation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_observation: Option<DomainObservation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deployment_jobs: Vec<DeploymentJobObservation>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrainDirective {
    #[default]
    None,
    Begin,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HeartbeatResponse {
    #[serde(default)]
    pub drain: DrainDirective,
    #[serde(default)]
    pub inventory_resync: bool,
    #[serde(default)]
    pub reregister: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deployment_commands: Vec<DeploymentCommand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    RegisteredNotReady,
    Ready,
    Degraded,
    Draining,
    Unavailable,
    Expired,
}

#[cfg(test)]
mod tests {
    use time::OffsetDateTime;

    use super::{CapacityObservation, RuntimeObservation, RuntimeStatus};
    use crate::LeaseToken;

    #[test]
    fn lease_token_debug_is_redacted() {
        let token = LeaseToken::new("0123456789abcdef-secret").unwrap();
        assert_eq!(format!("{token:?}"), "LeaseToken([REDACTED])");
        assert_eq!(token.expose(), "0123456789abcdef-secret");
    }

    #[test]
    fn capacity_rejects_nan_and_overcommit() {
        assert!(
            CapacityObservation {
                kv_cache_used_ratio: Some(f64::NAN),
                ..CapacityObservation::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            CapacityObservation {
                active_requests: Some(3),
                max_concurrent_requests: Some(2),
                ..CapacityObservation::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn runtime_status_must_match_ready_flag() {
        let observation = RuntimeObservation {
            observed_at: OffsetDateTime::UNIX_EPOCH,
            runtime: RuntimeStatus::unavailable("connect_failed"),
            inventory_generation: 0,
            models: Vec::new(),
            capacity: None,
        };
        assert!(observation.validate().is_ok());

        let mut inconsistent = observation;
        inconsistent.runtime.ready = true;
        assert!(inconsistent.validate().is_err());
    }

    #[test]
    fn rfc3339_observation_round_trips() {
        let observation = RuntimeObservation {
            observed_at: OffsetDateTime::UNIX_EPOCH,
            runtime: RuntimeStatus::ready(),
            inventory_generation: 1,
            models: Vec::new(),
            capacity: None,
        };
        let json = serde_json::to_string(&observation).unwrap();
        let decoded: RuntimeObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, observation);
    }
}
