//! Execution-domain identity, desired state, and bounded observations.
//!
//! A domain is an independently operated routing and failure boundary. AX
//! Serving selects a domain; the execution owner named by the descriptor owns
//! scheduling inside that boundary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{
    CapacityError, CapacityObservation, Digest, DigestError, DomainId, PoolId,
    RuntimeModelDescriptor, RuntimeState, TrustDomainId,
};

const MAX_OWNER_BYTES: usize = 64;
const MAX_METADATA_BYTES: usize = 128;
const MAX_LABEL_KEY_BYTES: usize = 64;
const MAX_LABEL_VALUE_BYTES: usize = 256;
const MAX_LABELS: usize = 32;

/// Digest of the immutable compatibility manifest used to qualify a domain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompatibilityManifestDigest(Digest);

impl CompatibilityManifestDigest {
    pub fn new(value: impl Into<String>) -> Result<Self, DigestError> {
        Digest::new(value).map(Self)
    }

    pub const fn as_digest(&self) -> &Digest {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for CompatibilityManifestDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for CompatibilityManifestDigest {
    type Err = DigestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Execution system represented by a domain endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDomainKind {
    MacAxEngine,
    MacAxEngineCluster,
    NvidiaDynamoPc,
    NvidiaDynamoThor,
    CompatibilityRuntimeEndpoint,
    /// A future kind understood by a newer peer. It remains diagnosable but
    /// cannot become eligible until this gateway understands its contract.
    #[serde(other)]
    Unknown,
}

impl ExecutionDomainKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MacAxEngine => "mac_ax_engine",
            Self::MacAxEngineCluster => "mac_ax_engine_cluster",
            Self::NvidiaDynamoPc => "nvidia_dynamo_pc",
            Self::NvidiaDynamoThor => "nvidia_dynamo_thor",
            Self::CompatibilityRuntimeEndpoint => "compatibility_runtime_endpoint",
            Self::Unknown => "unknown",
        }
    }

    pub const fn is_dynamo(self) -> bool {
        matches!(self, Self::NvidiaDynamoPc | Self::NvidiaDynamoThor)
    }
}

/// Whether the advertised endpoint represents one node or a complete domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointScope {
    Node,
    Domain,
    #[serde(other)]
    Unknown,
}

impl EndpointScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Domain => "domain",
            Self::Unknown => "unknown",
        }
    }
}

/// Operator qualification attached to an observed execution domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationState {
    Unverified,
    Experimental,
    Certified,
    Suspended,
    #[serde(other)]
    Unknown,
}

impl QualificationState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::Experimental => "experimental",
            Self::Certified => "certified",
            Self::Suspended => "suspended",
            Self::Unknown => "unknown",
        }
    }

    /// Return whether an observed state satisfies a desired minimum.
    pub const fn meets(self, required: Self) -> bool {
        let observed_rank = match self {
            Self::Unverified => 0,
            Self::Experimental => 1,
            Self::Certified => 2,
            Self::Suspended | Self::Unknown => return false,
        };
        let required_rank = match required {
            Self::Unverified => 0,
            Self::Experimental => 1,
            Self::Certified => 2,
            Self::Suspended | Self::Unknown => return false,
        };
        observed_rank >= required_rank
    }
}

/// Observed identity of one registered execution domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionDomainDescriptor {
    pub id: DomainId,
    pub kind: ExecutionDomainKind,
    pub endpoint_scope: EndpointScope,
    pub execution_owner: String,
    pub qualification: QualificationState,
    pub pool_id: PoolId,
    pub trust_domain: TrustDomainId,
    pub hardware_class: String,
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility_manifest: Option<CompatibilityManifestDigest>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

impl ExecutionDomainDescriptor {
    pub fn validate(&self) -> Result<(), DomainValidationError> {
        validate_token("execution_owner", &self.execution_owner, MAX_OWNER_BYTES)?;
        validate_metadata("hardware_class", &self.hardware_class, MAX_METADATA_BYTES)?;
        validate_metadata("architecture", &self.architecture, MAX_METADATA_BYTES)?;
        validate_labels(&self.labels)?;

        let valid_contract = match self.kind {
            ExecutionDomainKind::MacAxEngine => {
                self.endpoint_scope == EndpointScope::Node && self.execution_owner == "ax_engine"
            }
            ExecutionDomainKind::MacAxEngineCluster => {
                self.endpoint_scope == EndpointScope::Domain && self.execution_owner == "ax_engine"
            }
            ExecutionDomainKind::NvidiaDynamoPc | ExecutionDomainKind::NvidiaDynamoThor => {
                self.endpoint_scope == EndpointScope::Domain && self.execution_owner == "dynamo"
            }
            ExecutionDomainKind::CompatibilityRuntimeEndpoint => {
                self.endpoint_scope == EndpointScope::Node
            }
            ExecutionDomainKind::Unknown => false,
        };
        if !valid_contract {
            return Err(DomainValidationError::InvalidKindScopeOwner {
                kind: self.kind.as_str(),
                scope: self.endpoint_scope.as_str(),
                owner: self.execution_owner.clone(),
            });
        }
        if self.qualification == QualificationState::Unknown {
            return Err(DomainValidationError::UnknownQualification);
        }
        Ok(())
    }
}

/// Desired execution-domain declaration loaded by the gateway.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSpec {
    pub id: DomainId,
    pub kind: ExecutionDomainKind,
    pub pool: PoolId,
    pub trust_domain: TrustDomainId,
    pub hardware_class: String,
    pub required_qualification: QualificationState,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub selector: BTreeMap<String, String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl DomainSpec {
    pub fn validate(&self) -> Result<(), DomainValidationError> {
        if self.kind == ExecutionDomainKind::Unknown {
            return Err(DomainValidationError::UnknownDomainKind);
        }
        if matches!(
            self.required_qualification,
            QualificationState::Suspended | QualificationState::Unknown
        ) {
            return Err(DomainValidationError::InvalidRequiredQualification);
        }
        validate_metadata("hardware_class", &self.hardware_class, MAX_METADATA_BYTES)?;
        validate_labels(&self.selector)?;
        Ok(())
    }

    /// Match immutable desired identity and selectors against an observation.
    pub fn matches_descriptor(&self, descriptor: &ExecutionDomainDescriptor) -> bool {
        self.id == descriptor.id
            && self.kind == descriptor.kind
            && self.pool == descriptor.pool_id
            && self.trust_domain == descriptor.trust_domain
            && self.hardware_class == descriptor.hardware_class
            && descriptor.qualification.meets(self.required_qualification)
            && self
                .selector
                .iter()
                .all(|(key, expected)| descriptor_value(descriptor, key) == Some(expected.as_str()))
    }
}

const fn default_enabled() -> bool {
    true
}

/// Aggregate, bounded state reported at an execution-domain boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainObservation {
    #[serde(with = "time::serde::rfc3339")]
    pub observed_at: OffsetDateTime,
    pub generation: u64,
    pub ready: bool,
    pub state: RuntimeState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontend_instances_ready: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate_capacity: Option<CapacityObservation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_digest: Option<CompatibilityManifestDigest>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<RuntimeModelDescriptor>,
}

impl DomainObservation {
    pub fn validate(&self) -> Result<(), DomainValidationError> {
        if self.ready != matches!(self.state, RuntimeState::Ready | RuntimeState::Degraded) {
            return Err(DomainValidationError::InconsistentObservationState);
        }
        if self.ready && self.frontend_instances_ready == Some(0) {
            return Err(DomainValidationError::NoReadyFrontend);
        }
        if let Some(reason_code) = &self.reason_code {
            validate_token("reason_code", reason_code, MAX_METADATA_BYTES)?;
        }
        if let Some(capacity) = &self.aggregate_capacity {
            capacity.validate()?;
        }
        let mut ids = BTreeSet::new();
        for model in &self.models {
            if !ids.insert(model.runtime_model_id.clone()) {
                return Err(DomainValidationError::DuplicateModel(
                    model.runtime_model_id.to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn descriptor_value<'a>(descriptor: &'a ExecutionDomainDescriptor, key: &str) -> Option<&'a str> {
    match key {
        "domain_id" => Some(descriptor.id.as_str()),
        "domain_kind" => Some(descriptor.kind.as_str()),
        "endpoint_scope" => Some(descriptor.endpoint_scope.as_str()),
        "execution_owner" => Some(descriptor.execution_owner.as_str()),
        "worker_pool" | "pool_id" => Some(descriptor.pool_id.as_str()),
        "trust_domain" => Some(descriptor.trust_domain.as_str()),
        "hardware_class" => Some(descriptor.hardware_class.as_str()),
        "architecture" => Some(descriptor.architecture.as_str()),
        "compatibility_manifest" | "compatibility_manifest_digest" => descriptor
            .compatibility_manifest
            .as_ref()
            .map(CompatibilityManifestDigest::as_str),
        label => descriptor.labels.get(label).map(String::as_str),
    }
}

fn validate_token(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), DomainValidationError> {
    if value.is_empty() || value.len() > maximum {
        return Err(DomainValidationError::InvalidMetadata { field });
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'.' | b'_' | b':' | b'-')
    }) {
        return Err(DomainValidationError::InvalidMetadata { field });
    }
    Ok(())
}

fn validate_metadata(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), DomainValidationError> {
    if value.is_empty()
        || value.len() > maximum
        || value.trim() != value
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b' ' | b'.' | b'_' | b':' | b'/' | b'@' | b'+' | b'-')
        })
    {
        return Err(DomainValidationError::InvalidMetadata { field });
    }
    Ok(())
}

fn validate_labels(labels: &BTreeMap<String, String>) -> Result<(), DomainValidationError> {
    if labels.len() > MAX_LABELS {
        return Err(DomainValidationError::TooManyLabels);
    }
    for (key, value) in labels {
        if key.is_empty()
            || key.len() > MAX_LABEL_KEY_BYTES
            || !key.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'_' | b'-')
            })
        {
            return Err(DomainValidationError::InvalidLabelKey);
        }
        validate_metadata("label_value", value, MAX_LABEL_VALUE_BYTES)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DomainValidationError {
    #[error("invalid execution-domain kind/scope/owner combination: {kind}/{scope}/{owner}")]
    InvalidKindScopeOwner {
        kind: &'static str,
        scope: &'static str,
        owner: String,
    },
    #[error("unknown execution-domain kind is not eligible")]
    UnknownDomainKind,
    #[error("unknown qualification state is not eligible")]
    UnknownQualification,
    #[error("required qualification must be unverified, experimental, or certified")]
    InvalidRequiredQualification,
    #[error("domain observation ready flag and state are inconsistent")]
    InconsistentObservationState,
    #[error("a ready domain must not report zero ready frontend instances")]
    NoReadyFrontend,
    #[error("domain metadata field '{field}' is empty, too long, or contains invalid characters")]
    InvalidMetadata { field: &'static str },
    #[error("execution-domain metadata contains too many labels")]
    TooManyLabels,
    #[error("execution-domain label key is empty, too long, or contains invalid characters")]
    InvalidLabelKey,
    #[error("duplicate runtime model id '{0}' in domain observation")]
    DuplicateModel(String),
    #[error(transparent)]
    InvalidCapacity(#[from] CapacityError),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        DomainSpec, EndpointScope, ExecutionDomainDescriptor, ExecutionDomainKind,
        QualificationState,
    };
    use crate::{DomainId, PoolId, TrustDomainId};

    fn descriptor(kind: ExecutionDomainKind) -> ExecutionDomainDescriptor {
        let domain_scoped = kind.is_dynamo() || kind == ExecutionDomainKind::MacAxEngineCluster;
        ExecutionDomainDescriptor {
            id: DomainId::new("domain-main").unwrap(),
            kind,
            endpoint_scope: if domain_scoped {
                EndpointScope::Domain
            } else {
                EndpointScope::Node
            },
            execution_owner: if kind.is_dynamo() {
                "dynamo".into()
            } else {
                "ax_engine".into()
            },
            qualification: QualificationState::Certified,
            pool_id: PoolId::new("pool-main").unwrap(),
            trust_domain: TrustDomainId::new("private").unwrap(),
            hardware_class: "nvidia-pc-cuda".into(),
            architecture: "x86_64".into(),
            compatibility_manifest: None,
            labels: BTreeMap::new(),
        }
    }

    #[test]
    fn dynamo_domains_require_domain_scope_and_dynamo_owner() {
        let mut value = descriptor(ExecutionDomainKind::NvidiaDynamoPc);
        assert!(value.validate().is_ok());
        value.endpoint_scope = EndpointScope::Node;
        assert!(value.validate().is_err());
    }

    #[test]
    fn mac_cluster_domains_require_domain_scope_and_ax_engine_owner() {
        let mut value = descriptor(ExecutionDomainKind::MacAxEngineCluster);
        assert!(value.validate().is_ok());
        value.endpoint_scope = EndpointScope::Node;
        assert!(value.validate().is_err());
        value.endpoint_scope = EndpointScope::Domain;
        value.execution_owner = "dynamo".into();
        assert!(value.validate().is_err());
    }

    #[test]
    fn suspended_domains_never_meet_a_minimum() {
        assert!(!QualificationState::Suspended.meets(QualificationState::Unverified));
        assert!(QualificationState::Certified.meets(QualificationState::Experimental));
        assert!(!QualificationState::Experimental.meets(QualificationState::Certified));
    }

    #[test]
    fn desired_domain_matches_identity_and_selector() {
        let mut observed = descriptor(ExecutionDomainKind::NvidiaDynamoPc);
        observed.compatibility_manifest = Some(
            super::CompatibilityManifestDigest::new(format!("sha256:{}", "a".repeat(64))).unwrap(),
        );
        observed.labels.insert("zone".into(), "dc-a".into());
        let desired = DomainSpec {
            id: observed.id.clone(),
            kind: observed.kind,
            pool: observed.pool_id.clone(),
            trust_domain: observed.trust_domain.clone(),
            hardware_class: observed.hardware_class.clone(),
            required_qualification: QualificationState::Experimental,
            selector: BTreeMap::from([
                ("domain_id".into(), observed.id.to_string()),
                (
                    "compatibility_manifest".into(),
                    observed
                        .compatibility_manifest
                        .as_ref()
                        .unwrap()
                        .to_string(),
                ),
                ("zone".into(), "dc-a".into()),
            ]),
            enabled: true,
        };
        assert!(desired.matches_descriptor(&observed));
    }

    #[test]
    fn unknown_future_kind_decodes_but_fails_validation() {
        let mut value =
            serde_json::to_value(descriptor(ExecutionDomainKind::NvidiaDynamoPc)).unwrap();
        value["kind"] = serde_json::Value::String("future_accelerator".into());
        let decoded: ExecutionDomainDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.kind, ExecutionDomainKind::Unknown);
        assert!(decoded.validate().is_err());
    }
}
