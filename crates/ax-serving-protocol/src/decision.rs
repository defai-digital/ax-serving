//! Bounded domain-decision contracts.
//!
//! These records intentionally exclude prompts, tenant identifiers, URLs, and
//! free-form diagnostics. They are safe control-plane evidence, not request
//! transcripts. Durable storage and deterministic replay remain responsibilities
//! of the control-plane layer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{
    DeploymentId, DomainId, LogicalModelId, Operation, PolicyId, PolicyVersion, RequestId,
};

const MAX_PROFILE_VALUE_BYTES: usize = 96;
const MAX_CANDIDATES: usize = 128;

/// Versioned policy inputs used for deterministic domain admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DecisionProfileV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_domain: Option<DomainId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_domain: Option<DomainId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cost_microusd: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_slo_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_floor: Option<String>,
}

impl DecisionProfileV1 {
    pub fn validate(&self) -> Result<(), DecisionValidationError> {
        for (field, value) in [
            ("routing_profile", self.routing_profile.as_deref()),
            ("privacy_class", self.privacy_class.as_deref()),
            ("locality", self.locality.as_deref()),
            ("quality_floor", self.quality_floor.as_deref()),
        ] {
            if let Some(value) = value {
                validate_bounded_token(field, value)?;
            }
        }
        if self.latency_slo_ms == Some(0) {
            return Err(DecisionValidationError::ZeroLatencySlo);
        }
        if matches!(
            (&self.required_domain, &self.preferred_domain),
            (Some(required), Some(preferred)) if required != preferred
        ) {
            return Err(DecisionValidationError::ConflictingDomainConstraints);
        }
        Ok(())
    }
}

/// Activation mode for an immutable decision policy version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyMode {
    Shadow,
    Canary,
    Active,
    /// Explicit operator rollback to the previous active baseline.
    Rollback,
}

/// Bounded reason for accepting or preferring a selected candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionReasonCode {
    RequiredDomain,
    PreferredDomain,
    OnlyEligible,
    LocalityMatch,
    LowestNormalizedScore,
    StableTieBreak,
    ExplicitDeployment,
}

/// Bounded hard-filter reason attached to an ineligible candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejectionReason {
    DomainDisabled,
    DomainUnknown,
    QualificationInsufficient,
    ObservationMissing,
    ObservationStale,
    DomainNotReady,
    DomainDraining,
    ProtocolIncompatible,
    ManifestMismatch,
    TrustPolicy,
    LocalityPolicy,
    ExplicitDomainConstraint,
    OperationUnsupported,
    CapabilityUnsupported,
    LimitUnsupported,
    IdentityMismatch,
    EquivalenceMissing,
    CapacityUnavailable,
}

/// One bounded candidate result retained in a decision record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateDecision {
    pub domain: DomainId,
    pub deployment: DeploymentId,
    pub eligible: bool,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub rejection_reasons: BTreeSet<CandidateRejectionReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_score_microunits: Option<i64>,
}

/// Versioned evidence for one active or counterfactual domain decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRecordV1 {
    pub request_id: RequestId,
    pub operation: Operation,
    pub logical_model: LogicalModelId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_profile: Option<String>,
    pub policy_id: PolicyId,
    pub policy_version: PolicyVersion,
    pub policy_mode: PolicyMode,
    pub candidate_summary: Vec<CandidateDecision>,
    pub selected_domain: DomainId,
    pub selected_deployment: DeploymentId,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub reason_codes: BTreeSet<DecisionReasonCode>,
    pub observation_generations: BTreeMap<DomainId, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_cost_microusd: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_latency_ms: Option<u64>,
    #[serde(with = "time::serde::rfc3339")]
    pub decided_at: OffsetDateTime,
}

impl DecisionRecordV1 {
    pub fn validate(&self) -> Result<(), DecisionValidationError> {
        if self.candidate_summary.is_empty() {
            return Err(DecisionValidationError::EmptyCandidates);
        }
        if self.candidate_summary.len() > MAX_CANDIDATES {
            return Err(DecisionValidationError::TooManyCandidates);
        }
        if let Some(profile) = &self.routing_profile {
            validate_bounded_token("routing_profile", profile)?;
        }
        let mut candidates = BTreeSet::new();
        for candidate in &self.candidate_summary {
            if !candidates.insert((&candidate.domain, &candidate.deployment)) {
                return Err(DecisionValidationError::DuplicateCandidate);
            }
            if candidate.eligible != candidate.rejection_reasons.is_empty() {
                return Err(DecisionValidationError::InconsistentCandidate);
            }
        }
        if !self.candidate_summary.iter().any(|candidate| {
            candidate.eligible
                && candidate.domain == self.selected_domain
                && candidate.deployment == self.selected_deployment
        }) {
            return Err(DecisionValidationError::SelectedCandidateMissing);
        }
        if !self
            .observation_generations
            .contains_key(&self.selected_domain)
        {
            return Err(DecisionValidationError::SelectedGenerationMissing);
        }
        Ok(())
    }
}

fn validate_bounded_token(field: &'static str, value: &str) -> Result<(), DecisionValidationError> {
    if value.is_empty()
        || value.len() > MAX_PROFILE_VALUE_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
    {
        return Err(DecisionValidationError::InvalidProfileValue { field });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecisionValidationError {
    #[error("decision field '{field}' is empty, too long, or contains invalid characters")]
    InvalidProfileValue { field: &'static str },
    #[error("latency SLO must be greater than zero")]
    ZeroLatencySlo,
    #[error("required and preferred domains cannot conflict")]
    ConflictingDomainConstraints,
    #[error("decision record must contain at least one candidate")]
    EmptyCandidates,
    #[error("decision record exceeds the bounded candidate limit")]
    TooManyCandidates,
    #[error("candidate eligibility and rejection reasons are inconsistent")]
    InconsistentCandidate,
    #[error("decision record contains a duplicate domain/deployment candidate")]
    DuplicateCandidate,
    #[error("selected domain/deployment is not an eligible recorded candidate")]
    SelectedCandidateMissing,
    #[error("selected domain observation generation is missing")]
    SelectedGenerationMissing,
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use time::OffsetDateTime;

    use super::{
        CandidateDecision, DecisionProfileV1, DecisionReasonCode, DecisionRecordV1, PolicyMode,
    };
    use crate::{
        DeploymentId, DomainId, LogicalModelId, Operation, PolicyId, PolicyVersion, RequestId,
    };

    #[test]
    fn selected_candidate_must_be_eligible_and_recorded() {
        let domain = DomainId::new("mac-local").unwrap();
        let deployment = DeploymentId::new("qwen-mac").unwrap();
        let record = DecisionRecordV1 {
            request_id: RequestId::new(),
            operation: Operation::chat_completions(),
            logical_model: LogicalModelId::new("qwen/code").unwrap(),
            routing_profile: None,
            policy_id: PolicyId::new("explicit-safe").unwrap(),
            policy_version: PolicyVersion::new("1").unwrap(),
            policy_mode: PolicyMode::Active,
            candidate_summary: vec![CandidateDecision {
                domain: domain.clone(),
                deployment: deployment.clone(),
                eligible: true,
                rejection_reasons: BTreeSet::new(),
                normalized_score_microunits: Some(0),
            }],
            selected_domain: domain.clone(),
            selected_deployment: deployment,
            reason_codes: BTreeSet::from([DecisionReasonCode::OnlyEligible]),
            observation_generations: BTreeMap::from([(domain, 7)]),
            predicted_cost_microusd: None,
            predicted_latency_ms: None,
            decided_at: OffsetDateTime::UNIX_EPOCH,
        };
        record.validate().unwrap();

        let mut duplicate = record;
        duplicate
            .candidate_summary
            .push(duplicate.candidate_summary[0].clone());
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn decision_profile_rejects_conflicting_domain_constraints() {
        let profile = DecisionProfileV1 {
            required_domain: Some(DomainId::new("mac-local").unwrap()),
            preferred_domain: Some(DomainId::new("nvidia-pc").unwrap()),
            ..DecisionProfileV1::default()
        };
        assert!(profile.validate().is_err());
    }
}
