//! Conservative, advisory-only placement for a future cluster generation.
//!
//! The planner consumes an integrity-identified per-layer model profile and
//! fresh coordinator-only link observations. It never changes the active
//! manifest or rank assignments.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{
    ClusterRankObservation, CompatibilityManifestDigest, Digest, ParallelismKind,
    ParallelismManifestV1,
};
use serde::{Deserialize, Serialize};

const PROFILE_SCHEMA_VERSION: u32 = 1;
const MAX_PROFILE_LAYERS: usize = 4_096;
const SCORE_SCALE: u128 = 1_000_000;

/// Retained model measurements for one representative workload.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementProfileV1 {
    pub schema_version: u32,
    pub profile_digest: Digest,
    pub manifest_digest: CompatibilityManifestDigest,
    pub model_artifact_digest: Digest,
    pub activation_bytes_per_micro_batch: u64,
    pub layers: Vec<LayerExecutionProfile>,
}

/// Bounded execution and memory profile for one model layer.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerExecutionProfile {
    pub layer: u32,
    pub weight_bytes: u64,
    pub representative_compute_micros: u64,
}

/// Deterministic next-generation candidate returned to an operator.
#[derive(Debug, Clone, Serialize)]
pub struct AdvisoryPipelinePlan {
    pub schema_version: u32,
    pub active_generation: u64,
    pub candidate_generation: u64,
    pub manifest_digest: CompatibilityManifestDigest,
    pub profile_digest: Digest,
    pub predicted_iteration_micros: u64,
    pub predicted_pipeline_bubble_micros: u64,
    pub minimum_memory_headroom_bytes: u64,
    pub assignments: Vec<AdvisoryStageAssignment>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdvisoryStageAssignment {
    pub stage: u16,
    pub rank: u16,
    pub node_identity_digest: Digest,
    pub layer_start: u32,
    pub layer_end: u32,
    pub predicted_weight_bytes: u64,
    pub predicted_memory_demand_bytes: u64,
    pub memory_headroom_bytes: u64,
    pub predicted_compute_micros: u64,
    pub predicted_transfer_micros: u64,
    pub measured_bandwidth_bytes_per_second: Option<u64>,
    pub measured_latency_micros: Option<u64>,
}

#[derive(Clone)]
struct Candidate {
    maximum_stage_micros: u128,
    minimum_headroom_bytes: u64,
    boundaries: Vec<usize>,
}

/// Build a deterministic advisory plan. Unknown or stale topology data must be
/// filtered by the coordinator before calling this function.
pub fn build_advisory_plan(
    manifest: &ParallelismManifestV1,
    manifest_digest: &CompatibilityManifestDigest,
    observations: &[ClusterRankObservation],
    profile: &PlacementProfileV1,
) -> Result<AdvisoryPipelinePlan> {
    validate_profile(manifest, manifest_digest, profile)?;
    if manifest.parallelism.kind != ParallelismKind::Pipeline || manifest.parallelism.tp_size != 1 {
        bail!("automatic placement currently accepts pipeline-only manifests");
    }
    let stage_count = usize::from(manifest.parallelism.pp_size);
    if stage_count < 2 || stage_count > profile.layers.len() {
        bail!("pipeline stage count is incompatible with the model profile");
    }
    if observations.len() != stage_count {
        bail!("advisory placement requires one fresh observation per pipeline rank");
    }
    let observations_by_rank = observations
        .iter()
        .map(|observation| (observation.rank, observation))
        .collect::<BTreeMap<_, _>>();
    if observations_by_rank.len() != observations.len() {
        bail!("advisory topology observations contain duplicate ranks");
    }
    let mut stage_observations = Vec::with_capacity(stage_count);
    for stage in 0..stage_count {
        let rank = stage_rank(manifest, stage)?;
        let observation = observations_by_rank
            .get(&rank.rank)
            .copied()
            .context("advisory topology observation is missing a planned rank")?;
        if observation.generation != manifest.generation
            || &observation.manifest_digest != manifest_digest.as_digest()
        {
            bail!("advisory topology observation is incomplete or identity-mismatched");
        }
        if observation.peer_bandwidth_bytes_per_second.is_none()
            || observation.peer_latency_micros.is_none()
        {
            bail!("advisory placement requires measured bandwidth and latency for every rank");
        }
        stage_observations.push(observation);
    }

    let mut weight_prefix = vec![0_u128; profile.layers.len() + 1];
    let mut compute_prefix = vec![0_u128; profile.layers.len() + 1];
    for (index, layer) in profile.layers.iter().enumerate() {
        weight_prefix[index + 1] = weight_prefix[index]
            .checked_add(u128::from(layer.weight_bytes))
            .context("profile weight sum overflow")?;
        compute_prefix[index + 1] = compute_prefix[index]
            .checked_add(u128::from(layer.representative_compute_micros))
            .context("profile compute sum overflow")?;
    }

    let mut dynamic = vec![vec![None::<Candidate>; profile.layers.len() + 1]; stage_count + 1];
    dynamic[0][0] = Some(Candidate {
        maximum_stage_micros: 0,
        minimum_headroom_bytes: u64::MAX,
        boundaries: vec![0],
    });
    for completed_stages in 1..=stage_count {
        let stage = completed_stages - 1;
        let minimum_end = completed_stages;
        let maximum_end = profile.layers.len() - (stage_count - completed_stages);
        for end in minimum_end..=maximum_end {
            for start in (completed_stages - 1)..end {
                let Some(previous) = dynamic[completed_stages - 1][start].as_ref() else {
                    continue;
                };
                let Some((demand, headroom)) =
                    stage_memory(manifest, stage, start, end, &weight_prefix)?
                else {
                    continue;
                };
                let _ = demand;
                let compute = compute_prefix[end] - compute_prefix[start];
                let transfer = transfer_micros(
                    stage,
                    stage_count,
                    &stage_observations,
                    profile.activation_bytes_per_micro_batch,
                )?;
                let stage_micros = compute
                    .checked_add(u128::from(transfer))
                    .context("predicted stage duration overflow")?;
                let mut boundaries = previous.boundaries.clone();
                boundaries.push(end);
                let candidate = Candidate {
                    maximum_stage_micros: previous.maximum_stage_micros.max(stage_micros),
                    minimum_headroom_bytes: previous.minimum_headroom_bytes.min(headroom),
                    boundaries,
                };
                let replace = dynamic[completed_stages][end]
                    .as_ref()
                    .is_none_or(|current| better_candidate(&candidate, current));
                if replace {
                    dynamic[completed_stages][end] = Some(candidate);
                }
            }
        }
    }
    let candidate = dynamic[stage_count][profile.layers.len()]
        .take()
        .context("no per-rank memory-safe contiguous pipeline placement exists")?;

    let mut assignments = Vec::with_capacity(stage_count);
    let mut stage_sum = 0_u128;
    for stage in 0..stage_count {
        let start = candidate.boundaries[stage];
        let end = candidate.boundaries[stage + 1];
        let (demand, headroom) = stage_memory(manifest, stage, start, end, &weight_prefix)?
            .context("selected placement unexpectedly exceeds rank memory")?;
        let compute = compute_prefix[end] - compute_prefix[start];
        let transfer = transfer_micros(
            stage,
            stage_count,
            &stage_observations,
            profile.activation_bytes_per_micro_batch,
        )?;
        stage_sum = stage_sum
            .checked_add(compute)
            .and_then(|value| value.checked_add(u128::from(transfer)))
            .context("predicted pipeline duration overflow")?;
        let rank = stage_rank(manifest, stage)?;
        let (bandwidth, latency) = if stage + 1 < stage_count {
            let next = stage_observations[stage + 1];
            (
                Some(
                    stage_observations[stage]
                        .peer_bandwidth_bytes_per_second
                        .expect("validated topology measurement")
                        .min(
                            next.peer_bandwidth_bytes_per_second
                                .expect("validated topology measurement"),
                        ),
                ),
                Some(
                    stage_observations[stage]
                        .peer_latency_micros
                        .expect("validated topology measurement")
                        .max(
                            next.peer_latency_micros
                                .expect("validated topology measurement"),
                        ),
                ),
            )
        } else {
            (None, None)
        };
        assignments.push(AdvisoryStageAssignment {
            stage: stage as u16,
            rank: rank.rank,
            node_identity_digest: rank.node_identity_digest.clone(),
            layer_start: start as u32,
            layer_end: end as u32,
            predicted_weight_bytes: u64::try_from(weight_prefix[end] - weight_prefix[start])
                .context("predicted stage weight exceeds u64")?,
            predicted_memory_demand_bytes: demand,
            memory_headroom_bytes: headroom,
            predicted_compute_micros: u64::try_from(compute)
                .context("predicted stage compute exceeds u64")?,
            predicted_transfer_micros: transfer,
            measured_bandwidth_bytes_per_second: bandwidth,
            measured_latency_micros: latency,
        });
    }
    let ideal_capacity = candidate
        .maximum_stage_micros
        .checked_mul(stage_count as u128)
        .context("predicted pipeline capacity overflow")?;
    let bubble = ideal_capacity.saturating_sub(stage_sum);
    Ok(AdvisoryPipelinePlan {
        schema_version: 1,
        active_generation: manifest.generation,
        candidate_generation: manifest
            .generation
            .checked_add(1)
            .context("cluster generation overflow")?,
        manifest_digest: manifest_digest.clone(),
        profile_digest: profile.profile_digest.clone(),
        predicted_iteration_micros: u64::try_from(candidate.maximum_stage_micros)
            .context("predicted iteration duration exceeds u64")?,
        predicted_pipeline_bubble_micros: u64::try_from(bubble)
            .context("predicted pipeline bubble exceeds u64")?,
        minimum_memory_headroom_bytes: candidate.minimum_headroom_bytes,
        assignments,
    })
}

fn validate_profile(
    manifest: &ParallelismManifestV1,
    manifest_digest: &CompatibilityManifestDigest,
    profile: &PlacementProfileV1,
) -> Result<()> {
    if profile.schema_version != PROFILE_SCHEMA_VERSION {
        bail!("unsupported placement profile schema version");
    }
    if &profile.manifest_digest != manifest_digest
        || profile.model_artifact_digest != manifest.model.artifact_digest
    {
        bail!("placement profile identity does not match the active manifest");
    }
    if profile.activation_bytes_per_micro_batch == 0 {
        bail!("placement profile activation size must be greater than zero");
    }
    if profile.layers.len() != manifest.model.total_layers as usize
        || profile.layers.is_empty()
        || profile.layers.len() > MAX_PROFILE_LAYERS
    {
        bail!("placement profile must contain exactly one bounded entry per model layer");
    }
    for (expected, layer) in profile.layers.iter().enumerate() {
        if layer.layer as usize != expected
            || layer.weight_bytes == 0
            || layer.representative_compute_micros == 0
        {
            bail!("placement profile layers must be dense with nonzero weight and compute");
        }
    }
    Ok(())
}

fn stage_memory(
    manifest: &ParallelismManifestV1,
    stage: usize,
    start: usize,
    end: usize,
    weight_prefix: &[u128],
) -> Result<Option<(u64, u64)>> {
    let rank = stage_rank(manifest, stage)?;
    let current_demand = rank.memory.demand_bytes()?;
    let fixed = current_demand
        .checked_sub(rank.memory.assigned_weight_bytes)
        .context("rank assigned weight exceeds its total memory demand")?;
    let weights = u64::try_from(weight_prefix[end] - weight_prefix[start])
        .context("predicted stage weight exceeds u64")?;
    let demand = fixed
        .checked_add(weights)
        .context("predicted rank memory demand overflow")?;
    if demand > rank.memory.certified_usable_bytes {
        return Ok(None);
    }
    Ok(Some((demand, rank.memory.certified_usable_bytes - demand)))
}

fn stage_rank(
    manifest: &ParallelismManifestV1,
    stage: usize,
) -> Result<&ax_serving_protocol::RankPlan> {
    let stage = u16::try_from(stage).context("pipeline stage exceeds u16")?;
    manifest
        .ranks
        .iter()
        .find(|rank| rank.stage == stage && rank.tensor_rank == 0)
        .context("pipeline manifest is missing a stage rank")
}

fn transfer_micros(
    stage: usize,
    stage_count: usize,
    observations: &[&ClusterRankObservation],
    activation_bytes: u64,
) -> Result<u64> {
    if stage + 1 == stage_count {
        return Ok(0);
    }
    let next = observations[stage + 1];
    let bandwidth = observations[stage]
        .peer_bandwidth_bytes_per_second
        .context("missing source bandwidth measurement")?
        .min(
            next.peer_bandwidth_bytes_per_second
                .context("missing destination bandwidth measurement")?,
        );
    let latency = observations[stage]
        .peer_latency_micros
        .context("missing source latency measurement")?
        .max(
            next.peer_latency_micros
                .context("missing destination latency measurement")?,
        );
    if bandwidth == 0 {
        bail!("measured bandwidth must be greater than zero");
    }
    let serialization = u128::from(activation_bytes)
        .checked_mul(SCORE_SCALE)
        .context("activation transfer estimate overflow")?
        .div_ceil(u128::from(bandwidth));
    u64::try_from(u128::from(latency) + serialization)
        .context("activation transfer estimate exceeds u64")
}

fn better_candidate(candidate: &Candidate, current: &Candidate) -> bool {
    candidate.maximum_stage_micros < current.maximum_stage_micros
        || (candidate.maximum_stage_micros == current.maximum_stage_micros
            && (candidate.minimum_headroom_bytes > current.minimum_headroom_bytes
                || (candidate.minimum_headroom_bytes == current.minimum_headroom_bytes
                    && candidate.boundaries < current.boundaries)))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ax_serving_protocol::ClusterLifecycleState;

    use super::*;
    use crate::manifest::ValidatedManifest;

    fn fixture_manifest() -> ValidatedManifest {
        ValidatedManifest::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../config/mac-cluster-manifest.example.json"),
        )
        .unwrap()
    }

    fn ready_observation(manifest: &ValidatedManifest, rank: u16) -> ClusterRankObservation {
        ClusterRankObservation {
            cluster_id: manifest.manifest.cluster_id.clone(),
            generation: manifest.manifest.generation,
            manifest_digest: manifest.digest.as_digest().clone(),
            rank,
            state: ClusterLifecycleState::Ready,
            observed_at: time::OffsetDateTime::now_utc(),
            memory_used_bytes: Some(1_000),
            peer_bandwidth_bytes_per_second: Some(1_000_000_000),
            peer_latency_micros: Some(1_000),
            reason_code: None,
        }
    }

    #[test]
    fn advisory_planner_balances_profiles_without_mutating_active_generation() {
        let validated = fixture_manifest();
        let observations = vec![
            ready_observation(&validated, 0),
            ready_observation(&validated, 1),
        ];
        let profile = PlacementProfileV1 {
            schema_version: 1,
            profile_digest: Digest::new(format!("sha256:{}", "9".repeat(64))).unwrap(),
            manifest_digest: validated.digest.clone(),
            model_artifact_digest: validated.manifest.model.artifact_digest.clone(),
            activation_bytes_per_micro_batch: 16 * 1024 * 1024,
            layers: (0..126)
                .map(|layer| LayerExecutionProfile {
                    layer,
                    weight_bytes: 1_000_000_000,
                    representative_compute_micros: if layer < 42 { 10 } else { 100 },
                })
                .collect(),
        };
        let plan = build_advisory_plan(
            &validated.manifest,
            &validated.digest,
            &observations,
            &profile,
        )
        .unwrap();
        assert_eq!(plan.active_generation, 1);
        assert_eq!(plan.candidate_generation, 2);
        assert_eq!(plan.assignments.len(), 2);
        assert_eq!(plan.assignments[0].layer_start, 0);
        assert_ne!(plan.assignments[0].layer_end, 63);
        assert_eq!(
            plan.assignments[1].layer_start,
            plan.assignments[0].layer_end
        );
        assert_eq!(plan.assignments[1].layer_end, 126);
        assert!(plan.assignments.iter().all(|stage| {
            stage.predicted_memory_demand_bytes
                <= validated.manifest.ranks[usize::from(stage.rank)]
                    .memory
                    .certified_usable_bytes
        }));
        assert_eq!(validated.manifest.ranks[0].layers.end, 63);
    }

    #[test]
    fn advisory_planner_fails_closed_without_topology_measurements() {
        let validated = fixture_manifest();
        let mut observations = vec![
            ready_observation(&validated, 0),
            ready_observation(&validated, 1),
        ];
        observations[1].peer_bandwidth_bytes_per_second = None;
        let profile = PlacementProfileV1 {
            schema_version: 1,
            profile_digest: Digest::new(format!("sha256:{}", "9".repeat(64))).unwrap(),
            manifest_digest: validated.digest.clone(),
            model_artifact_digest: validated.manifest.model.artifact_digest.clone(),
            activation_bytes_per_micro_batch: 1,
            layers: (0..126)
                .map(|layer| LayerExecutionProfile {
                    layer,
                    weight_bytes: 1,
                    representative_compute_micros: 1,
                })
                .collect(),
        };
        assert!(
            build_advisory_plan(
                &validated.manifest,
                &validated.digest,
                &observations,
                &profile
            )
            .is_err()
        );
    }
}
