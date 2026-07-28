//! Immutable compatibility-manifest loading and fail-closed validation.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{CompatibilityManifestDigest, Digest, ExecutionDomainKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use time::OffsetDateTime;

const CANONICAL_DYNAMO_REPOSITORY: &str = "https://github.com/ai-dynamo/dynamo";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamoCompatibilityManifest {
    pub schema_version: u32,
    pub domain_kind: ExecutionDomainKind,
    pub dynamo: DynamoRelease,
    pub components: BTreeMap<String, String>,
    pub backend: BackendRelease,
    pub platform: PlatformRelease,
    pub graph_config_digest: Digest,
    pub model_certifications: Vec<Digest>,
    #[serde(with = "time::serde::rfc3339")]
    pub issued_at: OffsetDateTime,
    pub evidence: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamoRelease {
    pub repository: String,
    pub tag: String,
    pub commit: String,
    pub release_url: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendRelease {
    pub kind: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformRelease {
    pub arch: String,
    pub os: String,
    pub cuda: String,
}

#[derive(Clone, Debug)]
pub struct ValidatedManifest {
    pub manifest: DynamoCompatibilityManifest,
    pub digest: CompatibilityManifestDigest,
}

impl ValidatedManifest {
    pub fn load(path: &Path, expected_kind: ExecutionDomainKind) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read Dynamo manifest {}", path.display()))?;
        let manifest = serde_json::from_slice::<DynamoCompatibilityManifest>(&bytes)
            .with_context(|| format!("invalid Dynamo manifest JSON {}", path.display()))?;
        manifest.validate(expected_kind)?;

        let raw_digest = format!("sha256:{}", hex::encode(Sha256::digest(&bytes)));
        let digest = CompatibilityManifestDigest::new(raw_digest)
            .context("failed to construct Dynamo manifest digest")?;
        Ok(Self { manifest, digest })
    }
}

impl DynamoCompatibilityManifest {
    pub fn validate(&self, expected_kind: ExecutionDomainKind) -> Result<()> {
        if self.schema_version != 1 {
            bail!(
                "unsupported Dynamo manifest schema_version {}; expected 1",
                self.schema_version
            );
        }
        if !expected_kind.is_dynamo() || self.domain_kind != expected_kind {
            bail!("Dynamo manifest domain_kind does not match configured execution domain");
        }
        if self.dynamo.repository != CANONICAL_DYNAMO_REPOSITORY {
            bail!("Dynamo manifest repository must be {CANONICAL_DYNAMO_REPOSITORY}");
        }
        validate_release_tag(&self.dynamo.tag)?;
        validate_git_commit(&self.dynamo.commit)?;
        validate_release_url(&self.dynamo)?;
        validate_token("backend.kind", &self.backend.kind)?;
        validate_token("backend.version", &self.backend.version)?;
        validate_token("platform.os", &self.platform.os)?;
        validate_token("platform.cuda", &self.platform.cuda)?;
        validate_architecture(self.domain_kind, &self.platform.arch)?;

        if self.components.is_empty() {
            bail!("Dynamo manifest components must not be empty");
        }
        for required in ["frontend", "runtime"] {
            if !self.components.contains_key(required) {
                bail!("Dynamo manifest components must include '{required}'");
            }
        }
        for (component, image) in &self.components {
            validate_token("component name", component)?;
            validate_image_digest(component, image)?;
        }
        if self.model_certifications.is_empty() {
            bail!("Dynamo manifest must include at least one model certification digest");
        }
        validate_evidence_reference(&self.evidence)?;
        Ok(())
    }
}

fn validate_release_tag(value: &str) -> Result<()> {
    validate_token("dynamo.tag", value)?;
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "main" | "master" | "latest" | "nightly" | "dev"
    ) || value.contains('*')
    {
        bail!("Dynamo manifest tag must be an immutable released tag");
    }
    Ok(())
}

fn validate_git_commit(value: &str) -> Result<()> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("Dynamo manifest commit must be a full 40-character hexadecimal Git commit");
    }
    Ok(())
}

fn validate_release_url(release: &DynamoRelease) -> Result<()> {
    let url = reqwest::Url::parse(&release.release_url)
        .context("Dynamo manifest release_url is not a valid URL")?;
    let expected_path = format!("/ai-dynamo/dynamo/releases/tag/{}", release.tag);
    if url.scheme() != "https"
        || url.host_str() != Some("github.com")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != expected_path
    {
        bail!("Dynamo manifest release_url does not match the canonical repository and tag");
    }
    Ok(())
}

fn validate_architecture(kind: ExecutionDomainKind, raw: &str) -> Result<()> {
    let lowercase = raw.trim().to_ascii_lowercase();
    let normalized = match lowercase.as_str() {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        value => value,
    };
    let valid = match kind {
        ExecutionDomainKind::NvidiaDynamoPc => normalized == "amd64",
        ExecutionDomainKind::NvidiaDynamoThor => normalized == "arm64",
        _ => false,
    };
    if !valid {
        bail!("Dynamo manifest platform architecture does not match the domain kind");
    }
    Ok(())
}

fn validate_image_digest(component: &str, value: &str) -> Result<()> {
    let Some((image, digest)) = value.rsplit_once("@sha256:") else {
        bail!("Dynamo component '{component}' must use an immutable @sha256 image digest");
    };
    if image.is_empty()
        || image.chars().any(char::is_whitespace)
        || digest.len() != 64
        || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("Dynamo component '{component}' has an invalid immutable image reference");
    }
    Ok(())
}

fn validate_evidence_reference(value: &str) -> Result<()> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() > 2048
        || trimmed.contains('<')
        || trimmed.contains('>')
        || trimmed.to_ascii_lowercase().contains("todo")
    {
        bail!("Dynamo manifest evidence must be a bounded immutable artifact reference");
    }
    Ok(())
}

fn validate_token(field: &'static str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value.trim() != value
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'+' | b'-')
        })
    {
        bail!("Dynamo manifest field '{field}' is empty or contains invalid characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ax_serving_protocol::{Digest, ExecutionDomainKind};
    use time::OffsetDateTime;

    use super::{BackendRelease, DynamoCompatibilityManifest, DynamoRelease, PlatformRelease};

    fn digest(byte: char) -> Digest {
        Digest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
    }

    fn manifest(kind: ExecutionDomainKind) -> DynamoCompatibilityManifest {
        DynamoCompatibilityManifest {
            schema_version: 1,
            domain_kind: kind,
            dynamo: DynamoRelease {
                repository: "https://github.com/ai-dynamo/dynamo".into(),
                tag: "v1.2.1".into(),
                commit: "a".repeat(40),
                release_url: "https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1".into(),
            },
            components: BTreeMap::from([
                (
                    "frontend".into(),
                    format!("nvcr.io/nvidia/dynamo/frontend@sha256:{}", "b".repeat(64)),
                ),
                (
                    "runtime".into(),
                    format!("nvcr.io/nvidia/dynamo/vllm@sha256:{}", "c".repeat(64)),
                ),
            ]),
            backend: BackendRelease {
                kind: "vllm".into(),
                version: "0.25.1".into(),
            },
            platform: PlatformRelease {
                arch: if kind == ExecutionDomainKind::NvidiaDynamoThor {
                    "arm64"
                } else {
                    "amd64"
                }
                .into(),
                os: "ubuntu-24.04".into(),
                cuda: "13.0".into(),
            },
            graph_config_digest: digest('d'),
            model_certifications: vec![digest('e')],
            issued_at: OffsetDateTime::UNIX_EPOCH,
            evidence: "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                .into(),
        }
    }

    #[test]
    fn accepts_exact_pc_manifest() {
        manifest(ExecutionDomainKind::NvidiaDynamoPc)
            .validate(ExecutionDomainKind::NvidiaDynamoPc)
            .unwrap();
    }

    #[test]
    fn rejects_cross_architecture_or_floating_release() {
        let mut invalid = manifest(ExecutionDomainKind::NvidiaDynamoThor);
        invalid.platform.arch = "amd64".into();
        assert!(
            invalid
                .validate(ExecutionDomainKind::NvidiaDynamoThor)
                .is_err()
        );

        let mut floating = manifest(ExecutionDomainKind::NvidiaDynamoPc);
        floating.dynamo.tag = "main".into();
        floating.dynamo.release_url =
            "https://github.com/ai-dynamo/dynamo/releases/tag/main".into();
        assert!(
            floating
                .validate(ExecutionDomainKind::NvidiaDynamoPc)
                .is_err()
        );
    }

    #[test]
    fn rejects_noncanonical_release_url() {
        let mut invalid = manifest(ExecutionDomainKind::NvidiaDynamoPc);
        invalid.dynamo.release_url =
            "https://github.com/ai-dynamo/dynamo/releases/tag/not-v1.2.1?tag=v1.2.1".into();
        assert!(
            invalid
                .validate(ExecutionDomainKind::NvidiaDynamoPc)
                .is_err()
        );
    }

    #[test]
    fn rejects_mutable_component_image() {
        let mut invalid = manifest(ExecutionDomainKind::NvidiaDynamoPc);
        invalid
            .components
            .insert("runtime".into(), "nvcr.io/nvidia/dynamo/vllm:latest".into());
        assert!(
            invalid
                .validate(ExecutionDomainKind::NvidiaDynamoPc)
                .is_err()
        );
    }
}
