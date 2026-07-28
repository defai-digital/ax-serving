//! Conservative OpenAI inventory mapping at the Dynamo frontend boundary.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use ax_serving_adapter_core::openai_runtime::ModelInfo;
use ax_serving_protocol::{
    DeploymentIdentity, Operation, ProtocolCapability, RuntimeModelDescriptor, RuntimeModelId,
};

use crate::manifest::ValidatedManifest;

pub fn protocol_models(
    observed: &[ModelInfo],
    manifest: &ValidatedManifest,
) -> Result<Vec<RuntimeModelDescriptor>> {
    observed
        .iter()
        .map(|model| protocol_model(model, manifest))
        .collect()
}

fn protocol_model(
    model: &ModelInfo,
    manifest: &ValidatedManifest,
) -> Result<RuntimeModelDescriptor> {
    let mut operations = BTreeSet::new();
    let mut capabilities = BTreeSet::new();
    for operation in &model.supported_operations {
        match operation.as_str() {
            "embedding" | "embeddings" => {
                operations.insert(Operation::embeddings());
            }
            "vision" | "image" | "multimodal" => {
                capabilities.insert(ProtocolCapability::new("inference.vision")?);
                operations.insert(Operation::chat_completions());
            }
            "llm" => {
                operations.insert(Operation::chat_completions());
                operations.insert(Operation::text_completions());
            }
            _ => {}
        }
    }
    if operations.is_empty() {
        bail!(
            "Dynamo model '{}' did not advertise a recognized inference operation",
            model.id
        );
    }
    if model
        .modalities
        .iter()
        .any(|modality| matches!(modality.as_str(), "image" | "video"))
    {
        capabilities.insert(ProtocolCapability::new("inference.vision")?);
    }

    Ok(RuntimeModelDescriptor {
        runtime_model_id: RuntimeModelId::new(model.id.clone())
            .with_context(|| format!("invalid Dynamo runtime model id '{}'", model.id))?,
        identity: DeploymentIdentity {
            runtime_kind: "dynamo".into(),
            runtime_version: Some(manifest.manifest.dynamo.tag.clone()),
            revision: None,
            // The top-level manifest pins certification digests but does not
            // claim a one-to-one mapping to model entries. Keep these absent
            // rather than deriving identity from /v1/models.
            artifact_digest: None,
            tokenizer_digest: None,
            template_digest: None,
            quantization: None,
        },
        operations,
        capabilities,
        max_context_tokens: model.max_model_len.map(u64::from),
        max_output_tokens: model.max_output_tokens.map(u64::from),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ax_serving_adapter_core::openai_runtime::ModelInfo;
    use ax_serving_protocol::{CompatibilityManifestDigest, Digest, ExecutionDomainKind};
    use time::OffsetDateTime;

    use crate::manifest::{
        BackendRelease, DynamoCompatibilityManifest, DynamoRelease, PlatformRelease,
        ValidatedManifest,
    };

    use super::protocol_models;

    fn validated_manifest() -> ValidatedManifest {
        let digest =
            |value: char| Digest::new(format!("sha256:{}", value.to_string().repeat(64))).unwrap();
        ValidatedManifest {
            manifest: DynamoCompatibilityManifest {
                schema_version: 1,
                domain_kind: ExecutionDomainKind::NvidiaDynamoPc,
                dynamo: DynamoRelease {
                    repository: "https://github.com/ai-dynamo/dynamo".into(),
                    tag: "v1.2.1".into(),
                    commit: "a".repeat(40),
                    release_url: "https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1".into(),
                },
                components: BTreeMap::new(),
                backend: BackendRelease {
                    kind: "vllm".into(),
                    version: "0.25.1".into(),
                },
                platform: PlatformRelease {
                    arch: "amd64".into(),
                    os: "ubuntu-24.04".into(),
                    cuda: "13.0".into(),
                },
                graph_config_digest: digest('b'),
                model_certifications: vec![digest('c')],
                issued_at: OffsetDateTime::UNIX_EPOCH,
                evidence: "sha256:evidence".into(),
            },
            digest: CompatibilityManifestDigest::new(format!("sha256:{}", "d".repeat(64))).unwrap(),
        }
    }

    #[test]
    fn maps_frontend_models_without_inventing_artifact_identity() {
        let models = protocol_models(
            &[ModelInfo {
                id: "org/model".into(),
                max_model_len: Some(32_768),
                max_output_tokens: Some(4_096),
                quantization: Some("awq".into()),
                artifact_format: Some("safetensors".into()),
                modalities: vec!["text".into(), "image".into()],
                supported_operations: vec!["llm".into(), "vision".into()],
            }],
            &validated_manifest(),
        )
        .unwrap();

        assert_eq!(models[0].identity.runtime_kind, "dynamo");
        assert_eq!(
            models[0].identity.runtime_version.as_deref(),
            Some("v1.2.1")
        );
        assert_eq!(models[0].identity.artifact_digest, None);
        assert!(
            models[0]
                .capabilities
                .iter()
                .any(|capability| capability.as_str() == "inference.vision")
        );
    }

    #[test]
    fn rejects_unknown_operation_instead_of_inventing_text_support() {
        let result = protocol_models(
            &[ModelInfo {
                id: "org/reranker".into(),
                max_model_len: None,
                max_output_tokens: None,
                quantization: None,
                artifact_format: None,
                modalities: vec![],
                supported_operations: vec!["rerank".into()],
            }],
            &validated_manifest(),
        );
        assert!(result.is_err());
    }
}
