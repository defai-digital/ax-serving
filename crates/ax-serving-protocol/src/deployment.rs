use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    DeploymentId, EquivalenceClassId, LogicalModelId, Operation, PoolId, ProtocolCapability,
    RuntimeModelId, TrustDomainId,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DigestError {
    #[error("digest must use algorithm:hex format")]
    InvalidFormat,
    #[error("unsupported digest algorithm '{0}'")]
    UnsupportedAlgorithm(String),
    #[error("digest has an invalid length for {algorithm}: expected {expected} hex characters")]
    InvalidLength { algorithm: String, expected: usize },
    #[error("digest value must contain lowercase hexadecimal characters")]
    InvalidHex,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest(String);

impl Digest {
    pub fn new(value: impl Into<String>) -> Result<Self, DigestError> {
        let value = value.into();
        let (algorithm, hex) = value.split_once(':').ok_or(DigestError::InvalidFormat)?;
        let expected = match algorithm {
            "sha256" | "blake3" => 64,
            other => return Err(DigestError::UnsupportedAlgorithm(other.to_string())),
        };
        if hex.len() != expected {
            return Err(DigestError::InvalidLength {
                algorithm: algorithm.to_string(),
                expected,
            });
        }
        if !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(DigestError::InvalidHex);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for Digest {
    type Err = DigestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityField {
    RuntimeKind,
    RuntimeVersion,
    Revision,
    ArtifactDigest,
    TokenizerDigest,
    TemplateDigest,
    Quantization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentIdentity {
    pub runtime_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_digest: Option<Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_digest: Option<Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_digest: Option<Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityPolicy {
    #[serde(default)]
    pub required_matching_fields: BTreeSet<IdentityField>,
}

impl IdentityPolicy {
    pub fn strict_cross_runtime() -> Self {
        Self {
            required_matching_fields: BTreeSet::from([
                IdentityField::Revision,
                IdentityField::ArtifactDigest,
                IdentityField::TokenizerDigest,
                IdentityField::TemplateDigest,
                IdentityField::Quantization,
            ]),
        }
    }

    pub fn identities_match(&self, left: &DeploymentIdentity, right: &DeploymentIdentity) -> bool {
        self.required_matching_fields
            .iter()
            .all(|field| match field {
                IdentityField::RuntimeKind => nonempty_equal(
                    Some(left.runtime_kind.as_str()),
                    Some(right.runtime_kind.as_str()),
                ),
                IdentityField::RuntimeVersion => {
                    option_nonempty_equal(&left.runtime_version, &right.runtime_version)
                }
                IdentityField::Revision => option_nonempty_equal(&left.revision, &right.revision),
                IdentityField::ArtifactDigest => {
                    left.artifact_digest.is_some() && left.artifact_digest == right.artifact_digest
                }
                IdentityField::TokenizerDigest => {
                    left.tokenizer_digest.is_some()
                        && left.tokenizer_digest == right.tokenizer_digest
                }
                IdentityField::TemplateDigest => {
                    left.template_digest.is_some() && left.template_digest == right.template_digest
                }
                IdentityField::Quantization => {
                    option_nonempty_equal(&left.quantization, &right.quantization)
                }
            })
    }

    pub fn identity_is_complete(&self, identity: &DeploymentIdentity) -> bool {
        self.identities_match(identity, identity)
    }
}

impl Default for IdentityPolicy {
    fn default() -> Self {
        Self::strict_cross_runtime()
    }
}

fn option_nonempty_equal(left: &Option<String>, right: &Option<String>) -> bool {
    nonempty_equal(left.as_deref(), right.as_deref())
}

fn nonempty_equal(left: Option<&str>, right: Option<&str>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if !left.is_empty() && left == right)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquivalencePolicy {
    pub id: EquivalenceClassId,
    pub identity_policy: IdentityPolicy,
    #[serde(default)]
    pub certified_deployments: BTreeSet<DeploymentId>,
    pub certification_artifact: String,
}

impl EquivalencePolicy {
    pub fn permits_failover(
        &self,
        source_id: &DeploymentId,
        source: &DeploymentIdentity,
        target_id: &DeploymentId,
        target: &DeploymentIdentity,
    ) -> bool {
        self.certified_deployments.contains(source_id)
            && self.certified_deployments.contains(target_id)
            && !self.certification_artifact.trim().is_empty()
            && self.identity_policy.identities_match(source, target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolSpec {
    pub id: PoolId,
    pub runtime_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_class: Option<String>,
    pub trust_domain: TrustDomainId,
    #[serde(default)]
    pub selector: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub id: DeploymentId,
    pub logical_model: LogicalModelId,
    pub pool: PoolId,
    pub runtime_model_id: RuntimeModelId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equivalence_class: Option<EquivalenceClassId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_identity: Option<DeploymentIdentity>,
    #[serde(default)]
    pub required_identity: IdentityPolicy,
    #[serde(default)]
    pub required_capabilities: BTreeSet<ProtocolCapability>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

const fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeModelDescriptor {
    pub runtime_model_id: RuntimeModelId,
    pub identity: DeploymentIdentity,
    #[serde(default)]
    pub operations: BTreeSet<Operation>,
    #[serde(default)]
    pub capabilities: BTreeSet<ProtocolCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
}

impl RuntimeModelDescriptor {
    pub fn supports(&self, operation: &Operation, required: &BTreeSet<ProtocolCapability>) -> bool {
        self.operations.contains(operation) && required.is_subset(&self.capabilities)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{DeploymentIdentity, Digest, EquivalencePolicy, IdentityPolicy};
    use crate::{DeploymentId, EquivalenceClassId};

    fn digest(byte: char) -> Digest {
        Digest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
    }

    fn identity(runtime: &str) -> DeploymentIdentity {
        DeploymentIdentity {
            runtime_kind: runtime.to_string(),
            runtime_version: Some("1.0.0".into()),
            revision: Some("model-rev-1".into()),
            artifact_digest: Some(digest('a')),
            tokenizer_digest: Some(digest('b')),
            template_digest: Some(digest('c')),
            quantization: Some("q4_k_m".into()),
        }
    }

    #[test]
    fn strict_identity_allows_different_runtimes_when_semantics_match() {
        assert!(
            IdentityPolicy::strict_cross_runtime()
                .identities_match(&identity("ax_engine"), &identity("vllm"))
        );
    }

    #[test]
    fn missing_required_identity_fails_closed() {
        let left = identity("ax_engine");
        let mut right = identity("vllm");
        right.template_digest = None;
        assert!(!IdentityPolicy::strict_cross_runtime().identities_match(&left, &right));
    }

    #[test]
    fn identity_completeness_uses_the_configured_policy() {
        let mut incomplete = identity("vllm");
        incomplete.template_digest = None;
        assert!(!IdentityPolicy::strict_cross_runtime().identity_is_complete(&incomplete));
        assert!(
            IdentityPolicy {
                required_matching_fields: BTreeSet::new(),
            }
            .identity_is_complete(&incomplete)
        );
    }

    #[test]
    fn failover_requires_both_certification_membership_and_artifact() {
        let source = DeploymentId::new("mac-qwen").unwrap();
        let target = DeploymentId::new("cuda-qwen").unwrap();
        let policy = EquivalencePolicy {
            id: EquivalenceClassId::new("qwen-certified").unwrap(),
            identity_policy: IdentityPolicy::strict_cross_runtime(),
            certified_deployments: BTreeSet::from([source.clone(), target.clone()]),
            certification_artifact: "benchmarks/qwen-equivalence.json".into(),
        };
        assert!(policy.permits_failover(
            &source,
            &identity("ax_engine"),
            &target,
            &identity("vllm")
        ));
    }

    #[test]
    fn digest_rejects_uppercase_and_wrong_length() {
        assert!(Digest::new(format!("sha256:{}", "A".repeat(64))).is_err());
        assert!(Digest::new(format!("sha256:{}", "a".repeat(63))).is_err());
    }
}
