//! Runtime-neutral contracts for a model-parallel Mac AX Engine domain.
//!
//! These types describe and validate the immutable parallel plan and bounded
//! rank lifecycle observations. They deliberately contain no model tensors,
//! prompts, outputs, KV state, transport credentials, or runtime SDK types.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{Digest, DomainId, RuntimeModelId, RuntimeState};

/// Parallel execution strategy owned by AX Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismKind {
    Pipeline,
    Tensor,
    Hybrid,
}

/// Certified artifact role used by shard-aware rank preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFileKind {
    Weight,
    Tokenizer,
    Config,
    Other,
}

/// One immutable file in the certified model artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFilePlan {
    pub relative_path: String,
    pub digest: Digest,
    pub size_bytes: u64,
    pub kind: ArtifactFileKind,
}

/// Gang lifecycle for one immutable cluster generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterLifecycleState {
    Planned,
    Downloading,
    Connecting,
    Loading,
    Warming,
    Ready,
    Draining,
    Stopped,
    Failed,
}

impl ClusterLifecycleState {
    /// Whether this state permits new inference admission.
    pub const fn admits_requests(self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Map the cluster lifecycle to the existing bounded runtime state.
    pub const fn runtime_state(self) -> RuntimeState {
        match self {
            Self::Planned
            | Self::Downloading
            | Self::Connecting
            | Self::Loading
            | Self::Warming => RuntimeState::Starting,
            Self::Ready => RuntimeState::Ready,
            Self::Draining => RuntimeState::Draining,
            Self::Stopped | Self::Failed => RuntimeState::Unavailable,
        }
    }
}

/// Half-open model-layer range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    /// Whether the half-open range contains no layers.
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }

    /// Number of layers in the half-open range.
    pub const fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }
}

/// Immutable model identity used by the distributed plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterModelSpec {
    pub runtime_model_id: RuntimeModelId,
    pub artifact_digest: Digest,
    pub revision: String,
    pub tokenizer_digest: Digest,
    pub template_digest: Digest,
    pub quantization: String,
    pub architecture: String,
    pub total_layers: u32,
    pub max_context_tokens: u64,
    pub max_output_tokens: u64,
}

/// Exact AX Engine and host baseline for the plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterRuntimeSpec {
    pub ax_engine_version: String,
    pub build_digest: Digest,
    pub mlx_version: String,
    pub os_baseline: String,
}

/// Parallel dimensions and bounded pipeline scheduling settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelismPlan {
    pub kind: ParallelismKind,
    pub pp_size: u16,
    pub tp_size: u16,
    pub micro_batch_limit: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunking_profile_digest: Option<Digest>,
}

/// Integrity-bound data-plane profile. Credentials are never stored here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportPlan {
    pub kind: String,
    pub security_profile: String,
    pub topology_digest: Digest,
    pub minimum_bandwidth_bytes_per_second: u64,
    pub maximum_latency_micros: u64,
}

/// Per-rank memory admission plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankMemoryPlan {
    pub assigned_weight_bytes: u64,
    pub non_layer_weight_bytes: u64,
    pub quantization_metadata_bytes: u64,
    pub kv_budget_bytes: u64,
    pub activation_budget_bytes: u64,
    pub communication_buffer_bytes: u64,
    pub allocator_reserve_bytes: u64,
    pub os_headroom_bytes: u64,
    pub certified_usable_bytes: u64,
}

impl RankMemoryPlan {
    /// Calculate the complete rank demand, failing on integer overflow.
    pub fn demand_bytes(&self) -> Result<u64, ClusterManifestError> {
        [
            self.assigned_weight_bytes,
            self.non_layer_weight_bytes,
            self.quantization_metadata_bytes,
            self.kv_budget_bytes,
            self.activation_budget_bytes,
            self.communication_buffer_bytes,
            self.allocator_reserve_bytes,
            self.os_headroom_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            total
                .checked_add(value)
                .ok_or(ClusterManifestError::MemoryDemandOverflow)
        })
    }
}

/// One required AX Engine rank in the immutable gang.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankPlan {
    pub rank: u16,
    pub node_identity_digest: Digest,
    pub stage: u16,
    pub tensor_rank: u16,
    pub layers: LayerRange,
    pub owns_embeddings: bool,
    pub owns_output_head: bool,
    #[serde(default)]
    pub required_weight_files: Vec<Digest>,
    pub memory: RankMemoryPlan,
}

/// Complete immutable model-parallel plan retained outside AX fleet state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelismManifestV1 {
    pub schema_version: u32,
    pub cluster_id: DomainId,
    pub generation: u64,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    pub model: ClusterModelSpec,
    pub runtime: ClusterRuntimeSpec,
    pub parallelism: ParallelismPlan,
    pub transport: TransportPlan,
    pub artifacts: Vec<ArtifactFilePlan>,
    pub ranks: Vec<RankPlan>,
}

impl ParallelismManifestV1 {
    /// Validate the manifest without loading a model or contacting any rank.
    pub fn validate(&self) -> Result<(), ClusterManifestError> {
        if self.schema_version != 1 {
            return Err(ClusterManifestError::UnsupportedSchema(self.schema_version));
        }
        if self.generation == 0 {
            return Err(ClusterManifestError::ZeroGeneration);
        }
        validate_token("model revision", &self.model.revision)?;
        validate_token("quantization", &self.model.quantization)?;
        validate_token("model architecture", &self.model.architecture)?;
        validate_token("AX Engine version", &self.runtime.ax_engine_version)?;
        validate_token("MLX version", &self.runtime.mlx_version)?;
        validate_token("OS baseline", &self.runtime.os_baseline)?;
        validate_token("transport kind", &self.transport.kind)?;
        validate_token(
            "transport security profile",
            &self.transport.security_profile,
        )?;
        if self.model.total_layers == 0 {
            return Err(ClusterManifestError::ZeroLayers);
        }
        if self.model.max_context_tokens == 0 || self.model.max_output_tokens == 0 {
            return Err(ClusterManifestError::ZeroTokenLimit);
        }
        if self.parallelism.micro_batch_limit == 0 {
            return Err(ClusterManifestError::ZeroMicroBatchLimit);
        }

        let (pp_size, tp_size) = (self.parallelism.pp_size, self.parallelism.tp_size);
        match self.parallelism.kind {
            ParallelismKind::Pipeline if pp_size < 2 || tp_size != 1 => {
                return Err(ClusterManifestError::InvalidParallelDimensions);
            }
            ParallelismKind::Tensor if pp_size != 1 || tp_size < 2 => {
                return Err(ClusterManifestError::InvalidParallelDimensions);
            }
            ParallelismKind::Hybrid if pp_size < 2 || tp_size < 2 => {
                return Err(ClusterManifestError::InvalidParallelDimensions);
            }
            _ => {}
        }
        // Profile-driven chunking is optional for static PP, but required once
        // tensor or hybrid parallelism is certified so stage collectives stay
        // bound to a measured execution profile.
        if matches!(
            self.parallelism.kind,
            ParallelismKind::Tensor | ParallelismKind::Hybrid
        ) && self.parallelism.chunking_profile_digest.is_none()
        {
            return Err(ClusterManifestError::MissingChunkingProfile);
        }

        let expected_ranks = usize::from(pp_size)
            .checked_mul(usize::from(tp_size))
            .ok_or(ClusterManifestError::InvalidParallelDimensions)?;
        if self.ranks.len() != expected_ranks {
            return Err(ClusterManifestError::RankCount {
                expected: expected_ranks,
                actual: self.ranks.len(),
            });
        }

        let mut artifact_paths = BTreeSet::new();
        let mut artifact_digests = BTreeSet::new();
        for artifact in &self.artifacts {
            validate_relative_path(&artifact.relative_path)?;
            if artifact.size_bytes == 0 {
                return Err(ClusterManifestError::ZeroArtifactSize);
            }
            if !artifact_paths.insert(artifact.relative_path.as_str()) {
                return Err(ClusterManifestError::DuplicateArtifactPath);
            }
            if !artifact_digests.insert(artifact.digest.clone()) {
                return Err(ClusterManifestError::DuplicateArtifactDigest);
            }
        }
        if !self
            .artifacts
            .iter()
            .any(|artifact| artifact.kind == ArtifactFileKind::Weight)
        {
            return Err(ClusterManifestError::MissingWeightArtifact);
        }

        let mut ranks = BTreeSet::new();
        let mut nodes = BTreeSet::new();
        let mut stages = BTreeMap::<u16, Vec<&RankPlan>>::new();
        for rank in &self.ranks {
            if !ranks.insert(rank.rank) {
                return Err(ClusterManifestError::DuplicateRank(rank.rank));
            }
            if !nodes.insert(rank.node_identity_digest.clone()) {
                return Err(ClusterManifestError::DuplicateNode);
            }
            if rank.stage >= pp_size || rank.tensor_rank >= tp_size {
                return Err(ClusterManifestError::RankCoordinateOutOfRange { rank: rank.rank });
            }
            if rank.layers.start >= rank.layers.end || rank.layers.end > self.model.total_layers {
                return Err(ClusterManifestError::InvalidLayerRange { rank: rank.rank });
            }
            if rank.required_weight_files.is_empty()
                || rank
                    .required_weight_files
                    .iter()
                    .any(|digest| !artifact_digests.contains(digest))
            {
                return Err(ClusterManifestError::UnknownRankArtifact { rank: rank.rank });
            }
            let demand = rank.memory.demand_bytes()?;
            if demand > rank.memory.certified_usable_bytes {
                return Err(ClusterManifestError::InsufficientRankMemory {
                    rank: rank.rank,
                    demand,
                    usable: rank.memory.certified_usable_bytes,
                });
            }
            stages.entry(rank.stage).or_default().push(rank);
        }
        if ranks.iter().copied().ne(0..u16::try_from(expected_ranks)
            .map_err(|_| ClusterManifestError::InvalidParallelDimensions)?)
        {
            return Err(ClusterManifestError::NonContiguousRanks);
        }

        let mut expected_layer = 0_u32;
        for stage in 0..pp_size {
            let Some(stage_ranks) = stages.get(&stage) else {
                return Err(ClusterManifestError::MissingStage(stage));
            };
            if stage_ranks.len() != usize::from(tp_size) {
                return Err(ClusterManifestError::StageWidth {
                    stage,
                    expected: usize::from(tp_size),
                    actual: stage_ranks.len(),
                });
            }
            let first_range = stage_ranks[0].layers;
            let tensor_ranks = stage_ranks
                .iter()
                .map(|rank| rank.tensor_rank)
                .collect::<BTreeSet<_>>();
            if tensor_ranks.iter().copied().ne(0..tp_size)
                || stage_ranks.iter().any(|rank| rank.layers != first_range)
            {
                return Err(ClusterManifestError::InconsistentTensorGroup(stage));
            }
            if first_range.start != expected_layer {
                return Err(ClusterManifestError::LayerCoverage);
            }
            expected_layer = first_range.end;
        }
        if expected_layer != self.model.total_layers {
            return Err(ClusterManifestError::LayerCoverage);
        }
        if !self.ranks.iter().any(|rank| rank.owns_embeddings) {
            return Err(ClusterManifestError::MissingEmbeddingOwner);
        }
        if !self.ranks.iter().any(|rank| rank.owns_output_head) {
            return Err(ClusterManifestError::MissingOutputOwner);
        }
        Ok(())
    }
}

/// Bounded coordinator-facing observation for one required rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterRankObservation {
    pub cluster_id: DomainId,
    pub generation: u64,
    pub manifest_digest: Digest,
    pub rank: u16,
    pub state: ClusterLifecycleState,
    #[serde(with = "time::serde::rfc3339")]
    pub observed_at: OffsetDateTime,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_used_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_bandwidth_bytes_per_second: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_latency_micros: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

impl ClusterRankObservation {
    pub fn validate_for(
        &self,
        manifest: &ParallelismManifestV1,
        manifest_digest: &Digest,
    ) -> Result<(), ClusterManifestError> {
        if self.cluster_id != manifest.cluster_id {
            return Err(ClusterManifestError::ObservationClusterMismatch);
        }
        if self.generation != manifest.generation {
            return Err(ClusterManifestError::ObservationGenerationMismatch);
        }
        if &self.manifest_digest != manifest_digest {
            return Err(ClusterManifestError::ObservationManifestMismatch);
        }
        if usize::from(self.rank) >= manifest.ranks.len() {
            return Err(ClusterManifestError::ObservationUnknownRank(self.rank));
        }
        if self.state == ClusterLifecycleState::Ready
            && (self.peer_bandwidth_bytes_per_second.is_none_or(|observed| {
                observed < manifest.transport.minimum_bandwidth_bytes_per_second
            }) || self
                .peer_latency_micros
                .is_none_or(|observed| observed > manifest.transport.maximum_latency_micros))
        {
            return Err(ClusterManifestError::ObservationTopologyInsufficient);
        }
        if let Some(reason) = &self.reason_code {
            validate_token("rank reason code", reason)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClusterManifestError {
    #[error("unsupported cluster manifest schema version {0}")]
    UnsupportedSchema(u32),
    #[error("cluster generation must be greater than zero")]
    ZeroGeneration,
    #[error("model total_layers must be greater than zero")]
    ZeroLayers,
    #[error("model context and output token limits must be greater than zero")]
    ZeroTokenLimit,
    #[error("micro_batch_limit must be greater than zero")]
    ZeroMicroBatchLimit,
    #[error("parallel dimensions do not match the selected parallelism kind")]
    InvalidParallelDimensions,
    #[error("expected {expected} ranks but found {actual}")]
    RankCount { expected: usize, actual: usize },
    #[error("rank {0} is duplicated")]
    DuplicateRank(u16),
    #[error("a node identity is assigned to more than one rank")]
    DuplicateNode,
    #[error("rank ids must be contiguous from zero")]
    NonContiguousRanks,
    #[error("rank {rank} has an out-of-range stage or tensor coordinate")]
    RankCoordinateOutOfRange { rank: u16 },
    #[error("rank {rank} has an invalid layer range")]
    InvalidLayerRange { rank: u16 },
    #[error("stage {0} is missing")]
    MissingStage(u16),
    #[error("stage {stage} expected {expected} ranks but found {actual}")]
    StageWidth {
        stage: u16,
        expected: usize,
        actual: usize,
    },
    #[error("stage {0} has inconsistent tensor ranks or layer ranges")]
    InconsistentTensorGroup(u16),
    #[error("pipeline stage layer ranges must be gap-free and cover every layer")]
    LayerCoverage,
    #[error("the manifest has no embedding owner")]
    MissingEmbeddingOwner,
    #[error("the manifest has no output-head owner")]
    MissingOutputOwner,
    #[error("rank memory demand overflowed u64")]
    MemoryDemandOverflow,
    #[error("rank {rank} requires {demand} bytes but only {usable} are certified")]
    InsufficientRankMemory { rank: u16, demand: u64, usable: u64 },
    #[error("artifact paths must be bounded safe relative paths")]
    InvalidArtifactPath,
    #[error("artifact size must be greater than zero")]
    ZeroArtifactSize,
    #[error("artifact relative paths must be unique")]
    DuplicateArtifactPath,
    #[error("artifact digests must be unique")]
    DuplicateArtifactDigest,
    #[error("the manifest must contain at least one weight artifact")]
    MissingWeightArtifact,
    #[error("rank {rank} references a missing artifact or has no required weight files")]
    UnknownRankArtifact { rank: u16 },
    #[error("manifest field '{0}' is empty, unbounded, or contains invalid characters")]
    InvalidToken(&'static str),
    #[error("rank observation cluster does not match the manifest")]
    ObservationClusterMismatch,
    #[error("rank observation generation does not match the manifest")]
    ObservationGenerationMismatch,
    #[error("rank observation manifest digest does not match the manifest")]
    ObservationManifestMismatch,
    #[error("rank observation references unknown rank {0}")]
    ObservationUnknownRank(u16),
    #[error("ready rank observation does not meet the certified topology profile")]
    ObservationTopologyInsufficient,
    #[error("tensor and hybrid plans require a chunking_profile_digest")]
    MissingChunkingProfile,
}

fn validate_token(field: &'static str, value: &str) -> Result<(), ClusterManifestError> {
    if value.is_empty()
        || value.len() > 128
        || value.trim() != value
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'+' | b'-')
        })
    {
        return Err(ClusterManifestError::InvalidToken(field));
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), ClusterManifestError> {
    if value.is_empty()
        || value.len() > 512
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.split(['/', '\\']).any(|part| {
            part.is_empty() || matches!(part, "." | "..") || part.chars().any(char::is_control)
        })
    {
        return Err(ClusterManifestError::InvalidArtifactPath);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Digest {
        Digest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
    }

    fn memory() -> RankMemoryPlan {
        RankMemoryPlan {
            assigned_weight_bytes: 100,
            non_layer_weight_bytes: 10,
            quantization_metadata_bytes: 5,
            kv_budget_bytes: 20,
            activation_budget_bytes: 20,
            communication_buffer_bytes: 10,
            allocator_reserve_bytes: 10,
            os_headroom_bytes: 25,
            certified_usable_bytes: 250,
        }
    }

    fn pipeline_manifest() -> ParallelismManifestV1 {
        ParallelismManifestV1 {
            schema_version: 1,
            cluster_id: DomainId::new("mac-cluster").unwrap(),
            generation: 7,
            created_at: OffsetDateTime::UNIX_EPOCH,
            model: ClusterModelSpec {
                runtime_model_id: RuntimeModelId::new("llama-405b").unwrap(),
                artifact_digest: digest('a'),
                revision: "rev-1".into(),
                tokenizer_digest: digest('b'),
                template_digest: digest('c'),
                quantization: "int4".into(),
                architecture: "llama".into(),
                total_layers: 8,
                max_context_tokens: 8_192,
                max_output_tokens: 2_048,
            },
            runtime: ClusterRuntimeSpec {
                ax_engine_version: "4.10.0".into(),
                build_digest: digest('d'),
                mlx_version: "0.29.0".into(),
                os_baseline: "macos-26.0".into(),
            },
            parallelism: ParallelismPlan {
                kind: ParallelismKind::Pipeline,
                pp_size: 2,
                tp_size: 1,
                micro_batch_limit: 1,
                chunking_profile_digest: None,
            },
            transport: TransportPlan {
                kind: "tcp".into(),
                security_profile: "trusted_mesh".into(),
                topology_digest: digest('e'),
                minimum_bandwidth_bytes_per_second: 1_000_000_000,
                maximum_latency_micros: 1_000,
            },
            artifacts: vec![
                ArtifactFilePlan {
                    relative_path: "weights/rank-0.safetensors".into(),
                    digest: digest('1'),
                    size_bytes: 100,
                    kind: ArtifactFileKind::Weight,
                },
                ArtifactFilePlan {
                    relative_path: "weights/rank-1.safetensors".into(),
                    digest: digest('2'),
                    size_bytes: 100,
                    kind: ArtifactFileKind::Weight,
                },
            ],
            ranks: vec![
                RankPlan {
                    rank: 0,
                    node_identity_digest: digest('f'),
                    stage: 0,
                    tensor_rank: 0,
                    layers: LayerRange { start: 0, end: 4 },
                    owns_embeddings: true,
                    owns_output_head: false,
                    required_weight_files: vec![digest('1')],
                    memory: memory(),
                },
                RankPlan {
                    rank: 1,
                    node_identity_digest: digest('0'),
                    stage: 1,
                    tensor_rank: 0,
                    layers: LayerRange { start: 4, end: 8 },
                    owns_embeddings: false,
                    owns_output_head: true,
                    required_weight_files: vec![digest('2')],
                    memory: memory(),
                },
            ],
        }
    }

    #[test]
    fn valid_pipeline_plan_covers_every_layer_and_rank() {
        pipeline_manifest().validate().unwrap();
    }

    #[test]
    fn aggregate_memory_cannot_hide_an_oversized_rank() {
        let mut manifest = pipeline_manifest();
        manifest.ranks[0].memory.certified_usable_bytes = 199;
        assert!(matches!(
            manifest.validate(),
            Err(ClusterManifestError::InsufficientRankMemory { rank: 0, .. })
        ));
    }

    #[test]
    fn layer_gap_fails_closed() {
        let mut manifest = pipeline_manifest();
        manifest.ranks[1].layers.start = 5;
        assert_eq!(
            manifest.validate(),
            Err(ClusterManifestError::LayerCoverage)
        );
    }

    #[test]
    fn artifact_paths_and_rank_shards_fail_closed() {
        let mut manifest = pipeline_manifest();
        manifest.artifacts[0].relative_path = "../rank-0.safetensors".into();
        assert_eq!(
            manifest.validate(),
            Err(ClusterManifestError::InvalidArtifactPath)
        );

        let mut manifest = pipeline_manifest();
        manifest.ranks[0].required_weight_files = vec![digest('9')];
        assert_eq!(
            manifest.validate(),
            Err(ClusterManifestError::UnknownRankArtifact { rank: 0 })
        );
    }

    #[test]
    fn observation_is_generation_and_manifest_fenced() {
        let manifest = pipeline_manifest();
        let manifest_digest = digest('9');
        let mut observation = ClusterRankObservation {
            cluster_id: manifest.cluster_id.clone(),
            generation: manifest.generation,
            manifest_digest: manifest_digest.clone(),
            rank: 0,
            state: ClusterLifecycleState::Ready,
            observed_at: OffsetDateTime::UNIX_EPOCH,
            memory_used_bytes: Some(100),
            peer_bandwidth_bytes_per_second: Some(1_000_000_000),
            peer_latency_micros: Some(1_000),
            reason_code: None,
        };
        observation
            .validate_for(&manifest, &manifest_digest)
            .unwrap();
        observation.generation += 1;
        assert_eq!(
            observation.validate_for(&manifest, &manifest_digest),
            Err(ClusterManifestError::ObservationGenerationMismatch)
        );
    }

    fn hybrid_manifest() -> ParallelismManifestV1 {
        let mut manifest = pipeline_manifest();
        manifest.parallelism.kind = ParallelismKind::Hybrid;
        manifest.parallelism.pp_size = 2;
        manifest.parallelism.tp_size = 2;
        manifest.parallelism.chunking_profile_digest = Some(digest('c'));
        manifest.artifacts = vec![
            ArtifactFilePlan {
                relative_path: "weights/r0.safetensors".into(),
                digest: digest('1'),
                size_bytes: 100,
                kind: ArtifactFileKind::Weight,
            },
            ArtifactFilePlan {
                relative_path: "weights/r1.safetensors".into(),
                digest: digest('2'),
                size_bytes: 100,
                kind: ArtifactFileKind::Weight,
            },
            ArtifactFilePlan {
                relative_path: "weights/r2.safetensors".into(),
                digest: digest('3'),
                size_bytes: 100,
                kind: ArtifactFileKind::Weight,
            },
            ArtifactFilePlan {
                relative_path: "weights/r3.safetensors".into(),
                digest: digest('4'),
                size_bytes: 100,
                kind: ArtifactFileKind::Weight,
            },
        ];
        // node digests must stay unique across ranks (hex-only chars)
        let nodes = [digest('f'), digest('0'), digest('a'), digest('b')];
        let weight = [digest('1'), digest('2'), digest('3'), digest('4')];
        manifest.ranks = vec![
            RankPlan {
                rank: 0,
                node_identity_digest: nodes[0].clone(),
                stage: 0,
                tensor_rank: 0,
                layers: LayerRange { start: 0, end: 4 },
                owns_embeddings: true,
                owns_output_head: false,
                required_weight_files: vec![weight[0].clone()],
                memory: memory(),
            },
            RankPlan {
                rank: 1,
                node_identity_digest: nodes[1].clone(),
                stage: 0,
                tensor_rank: 1,
                layers: LayerRange { start: 0, end: 4 },
                owns_embeddings: false,
                owns_output_head: false,
                required_weight_files: vec![weight[1].clone()],
                memory: memory(),
            },
            RankPlan {
                rank: 2,
                node_identity_digest: nodes[2].clone(),
                stage: 1,
                tensor_rank: 0,
                layers: LayerRange { start: 4, end: 8 },
                owns_embeddings: false,
                owns_output_head: true,
                required_weight_files: vec![weight[2].clone()],
                memory: memory(),
            },
            RankPlan {
                rank: 3,
                node_identity_digest: nodes[3].clone(),
                stage: 1,
                tensor_rank: 1,
                layers: LayerRange { start: 4, end: 8 },
                owns_embeddings: false,
                owns_output_head: false,
                required_weight_files: vec![weight[3].clone()],
                memory: memory(),
            },
        ];
        manifest
    }

    #[test]
    fn hybrid_plan_validates_model_native_pp_tp_grid() {
        hybrid_manifest().validate().unwrap();
    }

    #[test]
    fn hybrid_and_tensor_plans_require_chunking_profile() {
        let mut hybrid = hybrid_manifest();
        hybrid.parallelism.chunking_profile_digest = None;
        assert_eq!(
            hybrid.validate(),
            Err(ClusterManifestError::MissingChunkingProfile)
        );

        let mut tensor = hybrid_manifest();
        tensor.parallelism.kind = ParallelismKind::Tensor;
        tensor.parallelism.pp_size = 1;
        tensor.parallelism.tp_size = 2;
        tensor.parallelism.chunking_profile_digest = Some(digest('c'));
        tensor.ranks = tensor.ranks.into_iter().take(2).collect();
        for rank in &mut tensor.ranks {
            rank.stage = 0;
            rank.layers = LayerRange { start: 0, end: 8 };
        }
        tensor.ranks[0].tensor_rank = 0;
        tensor.ranks[1].tensor_rank = 1;
        tensor.ranks[0].owns_embeddings = true;
        tensor.ranks[0].owns_output_head = true;
        tensor.ranks[1].owns_embeddings = false;
        tensor.ranks[1].owns_output_head = false;
        tensor.validate().unwrap();
        tensor.parallelism.chunking_profile_digest = None;
        assert_eq!(
            tensor.validate(),
            Err(ClusterManifestError::MissingChunkingProfile)
        );
    }

    #[test]
    fn inconsistent_tensor_group_layer_ranges_fail_closed() {
        let mut hybrid = hybrid_manifest();
        hybrid.ranks[1].layers.end = 3;
        assert_eq!(
            hybrid.validate(),
            Err(ClusterManifestError::InconsistentTensorGroup(0))
        );
    }
}
