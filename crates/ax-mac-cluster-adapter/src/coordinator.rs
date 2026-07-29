//! Generation-fenced gang coordinator and bounded rank-control API.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{
    ArtifactFileKind, ArtifactFilePlan, CapacityObservation, ClusterLifecycleState,
    ClusterModelSpec, ClusterRankObservation, ClusterRuntimeSpec, CompatibilityManifestDigest,
    DeploymentIdentity, DomainId, DomainObservation, Operation, ParallelismPlan,
    ProtocolCapability, RankPlan, RuntimeModelDescriptor, RuntimeObservation, RuntimeState,
    RuntimeStatus, TransportPlan,
};
use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use tokio::sync::RwLock;

use crate::manifest::ValidatedManifest;
use crate::planner::{AdvisoryPipelinePlan, PlacementProfileV1, build_advisory_plan};

#[derive(Clone, Debug)]
pub struct ObservationSnapshot {
    pub runtime: RuntimeObservation,
    pub domain: DomainObservation,
}

#[derive(Clone)]
pub struct ClusterCoordinator {
    manifest: Arc<ValidatedManifest>,
    observations: Arc<RwLock<BTreeMap<u16, ClusterRankObservation>>>,
    stale_after: Duration,
    max_inflight: usize,
    pub inflight: Arc<AtomicUsize>,
    pub draining: Arc<AtomicBool>,
    pub ready: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterStatus {
    pub cluster_id: String,
    pub generation: u64,
    pub manifest_digest: String,
    pub state: ClusterLifecycleState,
    pub ready_ranks: usize,
    pub required_ranks: usize,
    pub ranks: Vec<ClusterRankObservation>,
}

/// Integrity-bound bootstrap payload for exactly one rank.
#[derive(Debug, Clone, Serialize)]
pub struct RankBootstrapPlan {
    pub cluster_id: DomainId,
    pub generation: u64,
    pub manifest_digest: CompatibilityManifestDigest,
    pub model: ClusterModelSpec,
    pub runtime: ClusterRuntimeSpec,
    pub parallelism: ParallelismPlan,
    pub transport: TransportPlan,
    pub rank: RankPlan,
    pub artifacts: Vec<ArtifactFilePlan>,
}

/// AX Engine-owned projection consumed by `ax-engine-pipeline-rank`.
///
/// This intentionally mirrors the runtime-neutral AX Engine JSON contract
/// without making AX Serving depend on an MLX/runtime crate.
#[derive(Debug, Clone, Serialize)]
pub struct EnginePipelineTopology {
    pub cluster_id: String,
    pub generation: u64,
    pub manifest_digest: String,
    pub model_artifact_digest: String,
    pub total_layers: u32,
    pub micro_batch_limit: u16,
    pub ranks: Vec<EnginePipelineRank>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnginePipelineRank {
    pub rank: u16,
    pub node_identity_digest: String,
    pub layers: EngineLayerRange,
    pub owns_embeddings: bool,
    pub owns_output_head: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EngineLayerRange {
    pub start: u32,
    pub end: u32,
}

impl ClusterCoordinator {
    pub fn new(manifest: ValidatedManifest, max_inflight: usize, stale_after: Duration) -> Self {
        Self {
            manifest: Arc::new(manifest),
            observations: Arc::new(RwLock::new(BTreeMap::new())),
            stale_after,
            max_inflight: max_inflight.max(1),
            inflight: Arc::new(AtomicUsize::new(0)),
            draining: Arc::new(AtomicBool::new(false)),
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn manifest(&self) -> &ValidatedManifest {
        &self.manifest
    }

    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Release);
        self.ready.store(false, Ordering::Release);
    }

    pub async fn update_rank(&self, observation: ClusterRankObservation) -> Result<()> {
        observation
            .validate_for(&self.manifest.manifest, self.manifest.digest.as_digest())
            .context("rank observation failed generation/manifest fencing")?;
        let now = time::OffsetDateTime::now_utc();
        if observation.observed_at > now + time::Duration::minutes(5) {
            bail!("rank observation is too far in the future");
        }
        let plan = &self.manifest.manifest.ranks[usize::from(observation.rank)];
        if observation
            .memory_used_bytes
            .is_some_and(|used| used > plan.memory.certified_usable_bytes)
        {
            bail!("rank observation exceeds its certified usable memory");
        }

        let mut observations = self.observations.write().await;
        if let Some(previous) = observations.get(&observation.rank) {
            if observation.observed_at < previous.observed_at {
                bail!("rank observation timestamp moved backwards");
            }
            if !valid_transition(previous.state, observation.state) {
                bail!(
                    "rank lifecycle transition {:?} -> {:?} is invalid in one generation",
                    previous.state,
                    observation.state
                );
            }
        }
        observations.insert(observation.rank, observation);
        drop(observations);
        self.refresh_ready().await;
        Ok(())
    }

    pub async fn snapshot(&self) -> ObservationSnapshot {
        let status = self.status().await;
        let observed_at = time::OffsetDateTime::now_utc();
        let ready = status.state.admits_requests() && !self.draining.load(Ordering::Acquire);
        let runtime_state = if self.draining.load(Ordering::Acquire) {
            RuntimeState::Draining
        } else {
            status.state.runtime_state()
        };
        let reason_code = match status.state {
            ClusterLifecycleState::Ready if ready => None,
            ClusterLifecycleState::Failed => Some("mac_cluster_rank_failed".into()),
            ClusterLifecycleState::Draining => Some("mac_cluster_draining".into()),
            _ => Some("mac_cluster_gang_incomplete".into()),
        };
        let model = model_descriptor(&self.manifest);
        let capacity = CapacityObservation {
            active_requests: Some(
                self.inflight.load(Ordering::Acquire).min(u64::MAX as usize) as u64
            ),
            max_concurrent_requests: Some(self.max_inflight as u64),
            ..Default::default()
        };
        let runtime_status = RuntimeStatus {
            ready,
            state: runtime_state,
            reason_code: reason_code.clone(),
            message: None,
            probe_latency_ms: None,
        };
        let runtime = RuntimeObservation {
            observed_at,
            runtime: runtime_status,
            inventory_generation: self.manifest.manifest.generation,
            models: vec![model.clone()],
            capacity: Some(capacity.clone()),
        };
        let domain = DomainObservation {
            observed_at,
            generation: self.manifest.manifest.generation,
            ready,
            state: runtime_state,
            reason_code,
            frontend_instances_ready: Some(u32::from(ready)),
            aggregate_capacity: Some(capacity),
            manifest_digest: Some(self.manifest.digest.clone()),
            models: vec![model],
        };
        ObservationSnapshot { runtime, domain }
    }

    pub async fn status(&self) -> ClusterStatus {
        let now = time::OffsetDateTime::now_utc();
        let observations = self.observations.read().await;
        let mut ranks = observations.values().cloned().collect::<Vec<_>>();
        ranks.sort_by_key(|observation| observation.rank);
        let fresh = |observation: &&ClusterRankObservation| {
            now - observation.observed_at
                <= time::Duration::try_from(self.stale_after).unwrap_or(time::Duration::MAX)
        };
        let ready_ranks = ranks
            .iter()
            .filter(fresh)
            .filter(|observation| observation.state == ClusterLifecycleState::Ready)
            .count();
        let required_ranks = self.manifest.manifest.ranks.len();
        let state = if self.draining.load(Ordering::Acquire) {
            ClusterLifecycleState::Draining
        } else if ranks
            .iter()
            .any(|observation| observation.state == ClusterLifecycleState::Failed)
        {
            ClusterLifecycleState::Failed
        } else if ready_ranks == required_ranks && ranks.len() == required_ranks {
            // Ready only when every required rank is present, fresh, and ready.
            ClusterLifecycleState::Ready
        } else {
            // Partial observation sets may include Ready ranks; never promote the
            // incomplete gang to Ready (partial-rank admission is forbidden).
            match least_progress(&ranks, now, self.stale_after) {
                ClusterLifecycleState::Ready => ClusterLifecycleState::Warming,
                other => other,
            }
        };
        ClusterStatus {
            cluster_id: self.manifest.manifest.cluster_id.to_string(),
            generation: self.manifest.manifest.generation,
            manifest_digest: self.manifest.digest.to_string(),
            state,
            ready_ranks,
            required_ranks,
            ranks,
        }
    }

    pub fn rank_bootstrap_plan(&self, rank: u16) -> Option<RankBootstrapPlan> {
        let rank_plan = self.manifest.manifest.ranks.get(usize::from(rank))?.clone();
        let required = rank_plan
            .required_weight_files
            .iter()
            .collect::<BTreeSet<_>>();
        let artifacts = self
            .manifest
            .manifest
            .artifacts
            .iter()
            .filter(|artifact| {
                artifact.kind != ArtifactFileKind::Weight || required.contains(&artifact.digest)
            })
            .cloned()
            .collect();
        Some(RankBootstrapPlan {
            cluster_id: self.manifest.manifest.cluster_id.clone(),
            generation: self.manifest.manifest.generation,
            manifest_digest: self.manifest.digest.clone(),
            model: self.manifest.manifest.model.clone(),
            runtime: self.manifest.manifest.runtime.clone(),
            parallelism: self.manifest.manifest.parallelism.clone(),
            transport: self.manifest.manifest.transport.clone(),
            rank: rank_plan,
            artifacts,
        })
    }

    pub fn engine_topology(&self) -> EnginePipelineTopology {
        let manifest = &self.manifest.manifest;
        EnginePipelineTopology {
            cluster_id: manifest.cluster_id.to_string(),
            generation: manifest.generation,
            manifest_digest: self.manifest.digest.to_string(),
            model_artifact_digest: manifest.model.artifact_digest.to_string(),
            total_layers: manifest.model.total_layers,
            micro_batch_limit: manifest.parallelism.micro_batch_limit,
            ranks: manifest
                .ranks
                .iter()
                .map(|rank| EnginePipelineRank {
                    rank: rank.rank,
                    node_identity_digest: rank.node_identity_digest.to_string(),
                    layers: EngineLayerRange {
                        start: rank.layers.start,
                        end: rank.layers.end,
                    },
                    owns_embeddings: rank.owns_embeddings,
                    owns_output_head: rank.owns_output_head,
                })
                .collect(),
        }
    }

    /// Produce an advisory candidate for a higher generation from fresh,
    /// measured rank topology. This never mutates the active manifest.
    pub async fn advisory_plan(
        &self,
        profile: &PlacementProfileV1,
    ) -> Result<AdvisoryPipelinePlan> {
        let now = time::OffsetDateTime::now_utc();
        let observations = self.observations.read().await;
        if observations.len() != self.manifest.manifest.ranks.len() {
            bail!("advisory placement requires a complete measured gang");
        }
        let mut ordered = observations.values().cloned().collect::<Vec<_>>();
        ordered.sort_by_key(|observation| observation.rank);
        if ordered.iter().any(|observation| {
            observation.state != ClusterLifecycleState::Ready
                || now - observation.observed_at
                    > time::Duration::try_from(self.stale_after).unwrap_or(time::Duration::MAX)
        }) {
            bail!("advisory placement requires fresh ready observations from every rank");
        }
        build_advisory_plan(
            &self.manifest.manifest,
            &self.manifest.digest,
            &ordered,
            profile,
        )
    }

    async fn refresh_ready(&self) {
        let ready = self.status().await.state == ClusterLifecycleState::Ready;
        self.ready.store(ready, Ordering::Release);
    }
}

fn model_descriptor(manifest: &ValidatedManifest) -> RuntimeModelDescriptor {
    let model = &manifest.manifest.model;
    RuntimeModelDescriptor {
        runtime_model_id: model.runtime_model_id.clone(),
        identity: DeploymentIdentity {
            runtime_kind: "ax_engine".into(),
            runtime_version: Some(manifest.manifest.runtime.ax_engine_version.clone()),
            revision: Some(model.revision.clone()),
            artifact_digest: Some(model.artifact_digest.clone()),
            tokenizer_digest: Some(model.tokenizer_digest.clone()),
            template_digest: Some(model.template_digest.clone()),
            quantization: Some(model.quantization.clone()),
        },
        operations: BTreeSet::from([Operation::chat_completions(), Operation::text_completions()]),
        capabilities: BTreeSet::<ProtocolCapability>::new(),
        max_context_tokens: Some(model.max_context_tokens),
        max_output_tokens: Some(model.max_output_tokens),
    }
}

fn valid_transition(from: ClusterLifecycleState, to: ClusterLifecycleState) -> bool {
    use ClusterLifecycleState as State;
    from == to
        || to == State::Failed
        || matches!(
            (from, to),
            (State::Planned, State::Downloading)
                | (State::Downloading, State::Connecting)
                | (State::Connecting, State::Loading)
                | (State::Loading, State::Warming)
                | (State::Warming, State::Ready)
                | (State::Ready, State::Draining)
                | (State::Draining, State::Stopped)
        )
}

fn least_progress(
    ranks: &[ClusterRankObservation],
    now: time::OffsetDateTime,
    stale_after: Duration,
) -> ClusterLifecycleState {
    let stale_after = time::Duration::try_from(stale_after).unwrap_or(time::Duration::MAX);
    if ranks.is_empty()
        || ranks
            .iter()
            .any(|observation| now - observation.observed_at > stale_after)
    {
        return ClusterLifecycleState::Planned;
    }
    ranks
        .iter()
        .map(|observation| observation.state)
        .min_by_key(|state| lifecycle_order(*state))
        .unwrap_or(ClusterLifecycleState::Planned)
}

const fn lifecycle_order(state: ClusterLifecycleState) -> u8 {
    match state {
        ClusterLifecycleState::Planned => 0,
        ClusterLifecycleState::Downloading => 1,
        ClusterLifecycleState::Connecting => 2,
        ClusterLifecycleState::Loading => 3,
        ClusterLifecycleState::Warming => 4,
        ClusterLifecycleState::Ready => 5,
        ClusterLifecycleState::Draining => 6,
        ClusterLifecycleState::Stopped => 7,
        ClusterLifecycleState::Failed => 8,
    }
}

#[derive(Clone)]
struct RankApiState {
    coordinator: ClusterCoordinator,
    token: Arc<str>,
}

pub fn router(coordinator: ClusterCoordinator, token: String) -> Router {
    let state = RankApiState {
        coordinator,
        token: Arc::from(token),
    };
    Router::new()
        .route(
            "/internal/cluster/ranks/{rank}/heartbeat",
            post(rank_heartbeat),
        )
        .route(
            "/internal/cluster/ranks/{rank}/plan",
            get(rank_bootstrap_plan),
        )
        .route(
            "/internal/cluster/engine-topology",
            get(engine_pipeline_topology),
        )
        .route("/internal/cluster/advisory-plan", post(advisory_placement))
        .route("/internal/cluster/status", get(cluster_status))
        .route_layer(middleware::from_fn_with_state(state.clone(), rank_auth))
        .with_state(state)
}

async fn rank_heartbeat(
    State(state): State<RankApiState>,
    Path(rank): Path<u16>,
    Json(observation): Json<ClusterRankObservation>,
) -> Response {
    if rank != observation.rank {
        return (StatusCode::CONFLICT, "rank path does not match observation").into_response();
    }
    match state.coordinator.update_rank(observation).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()).into_response(),
    }
}

async fn cluster_status(State(state): State<RankApiState>) -> Json<ClusterStatus> {
    Json(state.coordinator.status().await)
}

async fn engine_pipeline_topology(
    State(state): State<RankApiState>,
) -> Json<EnginePipelineTopology> {
    Json(state.coordinator.engine_topology())
}

async fn advisory_placement(
    State(state): State<RankApiState>,
    Json(profile): Json<PlacementProfileV1>,
) -> Response {
    match state.coordinator.advisory_plan(&profile).await {
        Ok(plan) => Json(plan).into_response(),
        Err(error) => (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()).into_response(),
    }
}

async fn rank_bootstrap_plan(State(state): State<RankApiState>, Path(rank): Path<u16>) -> Response {
    match state.coordinator.rank_bootstrap_plan(rank) {
        Some(plan) => Json(plan).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn rank_auth(State(state): State<RankApiState>, request: Request, next: Next) -> Response {
    let provided = request
        .headers()
        .get("x-ax-cluster-control-token")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !constant_time_eq(provided, &state.token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    next.run(request).await
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ax_serving_protocol::ClusterLifecycleState;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt as _;

    use super::*;

    fn fixture_manifest() -> ValidatedManifest {
        ValidatedManifest::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../config/mac-cluster-manifest.example.json"),
        )
        .unwrap()
    }

    fn observation(
        coordinator: &ClusterCoordinator,
        rank: u16,
        state: ClusterLifecycleState,
    ) -> ClusterRankObservation {
        ClusterRankObservation {
            cluster_id: coordinator.manifest.manifest.cluster_id.clone(),
            generation: coordinator.manifest.manifest.generation,
            manifest_digest: coordinator.manifest.digest.as_digest().clone(),
            rank,
            state,
            observed_at: time::OffsetDateTime::now_utc(),
            memory_used_bytes: Some(1_000),
            peer_bandwidth_bytes_per_second: Some(1_000_000_000),
            peer_latency_micros: Some(1_000),
            reason_code: None,
        }
    }

    #[tokio::test]
    async fn complete_ready_gang_is_one_ready_domain() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        for rank in 0..2 {
            for state in [
                ClusterLifecycleState::Planned,
                ClusterLifecycleState::Downloading,
                ClusterLifecycleState::Connecting,
                ClusterLifecycleState::Loading,
                ClusterLifecycleState::Warming,
                ClusterLifecycleState::Ready,
            ] {
                coordinator
                    .update_rank(observation(&coordinator, rank, state))
                    .await
                    .unwrap();
            }
        }
        let snapshot = coordinator.snapshot().await;
        assert!(snapshot.domain.ready);
        assert_eq!(snapshot.domain.generation, 1);
        assert_eq!(snapshot.domain.frontend_instances_ready, Some(1));
    }

    #[tokio::test]
    async fn failed_rank_fails_the_generation_and_cannot_recover() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        coordinator
            .update_rank(observation(&coordinator, 0, ClusterLifecycleState::Failed))
            .await
            .unwrap();
        assert!(
            coordinator
                .update_rank(observation(&coordinator, 0, ClusterLifecycleState::Planned,))
                .await
                .is_err()
        );
        assert_eq!(
            coordinator.status().await.state,
            ClusterLifecycleState::Failed
        );
    }

    #[tokio::test]
    async fn stale_rank_clears_gang_readiness() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_millis(10));
        for rank in 0..2 {
            let mut ready = observation(&coordinator, rank, ClusterLifecycleState::Ready);
            ready.observed_at -= time::Duration::seconds(1);
            coordinator.update_rank(ready).await.unwrap();
        }

        let status = coordinator.status().await;
        assert_eq!(status.state, ClusterLifecycleState::Planned);
        assert_eq!(status.ready_ranks, 0);
        assert!(!coordinator.snapshot().await.domain.ready);
    }

    #[tokio::test]
    async fn ready_rank_must_meet_certified_topology() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        let mut ready = observation(&coordinator, 0, ClusterLifecycleState::Ready);
        ready.peer_bandwidth_bytes_per_second = Some(999_999_999);

        assert!(coordinator.update_rank(ready).await.is_err());
        assert_eq!(
            coordinator.status().await.state,
            ClusterLifecycleState::Planned
        );
    }

    #[tokio::test]
    async fn rank_control_api_requires_its_distinct_credential() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        let app = router(coordinator, "rank-control-token".into());

        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/internal/cluster/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorized = app
            .oneshot(
                Request::builder()
                    .uri("/internal/cluster/status")
                    .header("x-ax-cluster-control-token", "rank-control-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
    }

    #[test]
    fn rank_bootstrap_contains_shared_files_and_only_its_weight_shard() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        let plan = coordinator.rank_bootstrap_plan(0).unwrap();
        assert!(
            plan.artifacts
                .iter()
                .any(|artifact| { artifact.relative_path == "weights/rank-0.safetensors" })
        );
        assert!(
            !plan
                .artifacts
                .iter()
                .any(|artifact| { artifact.relative_path == "weights/rank-1.safetensors" })
        );
        assert!(
            plan.artifacts
                .iter()
                .any(|artifact| artifact.relative_path == "tokenizer.json")
        );
    }

    #[test]
    fn engine_topology_projection_matches_rank_and_layer_plan() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        let topology = coordinator.engine_topology();
        assert_eq!(topology.generation, 1);
        assert_eq!(topology.total_layers, 126);
        assert_eq!(topology.micro_batch_limit, 1);
        assert_eq!(topology.ranks.len(), 2);
        assert_eq!(topology.ranks[0].layers.start, 0);
        assert_eq!(topology.ranks[0].layers.end, 63);
        assert!(topology.ranks[0].owns_embeddings);
        assert!(!topology.ranks[0].owns_output_head);
        assert_eq!(topology.ranks[1].layers.start, 63);
        assert_eq!(topology.ranks[1].layers.end, 126);
        assert!(topology.ranks[1].owns_output_head);
    }

    #[tokio::test]
    async fn drain_stops_admission_before_ranks_shut_down() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        for rank in 0..2 {
            for state in [
                ClusterLifecycleState::Planned,
                ClusterLifecycleState::Downloading,
                ClusterLifecycleState::Connecting,
                ClusterLifecycleState::Loading,
                ClusterLifecycleState::Warming,
                ClusterLifecycleState::Ready,
            ] {
                coordinator
                    .update_rank(observation(&coordinator, rank, state))
                    .await
                    .unwrap();
            }
        }
        assert!(coordinator.snapshot().await.domain.ready);
        coordinator.begin_drain();
        let snapshot = coordinator.snapshot().await;
        assert!(!snapshot.domain.ready);
        assert_eq!(
            coordinator.status().await.state,
            ClusterLifecycleState::Draining
        );
    }

    #[tokio::test]
    async fn incomplete_gang_never_reports_ready() {
        let coordinator = ClusterCoordinator::new(fixture_manifest(), 2, Duration::from_secs(30));
        coordinator
            .update_rank(observation(
                &coordinator,
                0,
                ClusterLifecycleState::Ready,
            ))
            .await
            .unwrap();
        let status = coordinator.status().await;
        assert_ne!(status.state, ClusterLifecycleState::Ready);
        assert!(!coordinator.snapshot().await.domain.ready);
        assert_eq!(status.ready_ranks, 1);
        assert_eq!(status.required_ranks, 2);
    }
}
