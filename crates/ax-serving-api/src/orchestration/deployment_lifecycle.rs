//! Asynchronous desired-state deployment administration.

use std::sync::Arc;

use ax_serving_protocol::{
    DeploymentControlRecord, DeploymentDesiredState, DeploymentId, DeploymentJobAction,
    DeploymentJobRecord, DeploymentJobStatus, DeploymentObservedState, DeploymentSpec, JobId,
};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tracing::{info, warn};

use super::deployment::DeploymentMode;
use super::fleet_state::FleetMutationResult;
use super::{OrchestratorLayer, error::ax_error_response};
use crate::auth::AxRequestId;

const DEPLOYMENT_JOB_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_ROLLOUT_TIMEOUT_MS: u64 = 5 * 60 * 1_000;
const MAX_ROLLOUT_TIMEOUT_MS: u64 = 30 * 60 * 1_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateDeploymentRequest {
    pub deployment: DeploymentSpec,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchDeploymentRequest {
    #[serde(default)]
    pub action: Option<DeploymentJobAction>,
    #[serde(default)]
    pub deployment: Option<DeploymentSpec>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub replacement_deployment_id: Option<DeploymentId>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

const fn default_list_limit() -> usize {
    100
}

#[derive(Debug, Serialize)]
struct DeploymentView {
    desired: DeploymentControlRecord,
    observed_state: DeploymentObservedState,
    ready_endpoints: usize,
    inflight_requests: usize,
    externally_managed: bool,
}

fn request_id(value: Option<Extension<AxRequestId>>) -> ax_serving_protocol::RequestId {
    value.map(|Extension(value)| value.0).unwrap_or_default()
}

fn lifecycle_error(
    status: StatusCode,
    request_id: ax_serving_protocol::RequestId,
    code: &'static str,
    message: impl AsRef<str>,
    retryable: bool,
) -> Response {
    ax_error_response(
        status,
        request_id,
        code,
        message,
        retryable,
        ax_serving_protocol::AdmissionPhase::Admission,
    )
}

fn explicit_mode(layer: &OrchestratorLayer) -> bool {
    layer.deployment_catalog.snapshot().mode() == DeploymentMode::Explicit
}

fn observed_view(layer: &OrchestratorLayer, record: DeploymentControlRecord) -> DeploymentView {
    let catalog = layer.deployment_catalog.snapshot();
    let (ready_endpoints, inflight_requests) =
        catalog.observed_endpoint_summary(&layer.registry, &record.deployment);
    let observed_state = match record.desired_state {
        DeploymentDesiredState::Enabled if ready_endpoints > 0 => DeploymentObservedState::Ready,
        DeploymentDesiredState::Enabled => DeploymentObservedState::ExternallyManaged,
        DeploymentDesiredState::Disabled if inflight_requests > 0 => {
            DeploymentObservedState::Draining
        }
        DeploymentDesiredState::Disabled => DeploymentObservedState::ExternallyManaged,
        DeploymentDesiredState::Absent if ready_endpoints == 0 => DeploymentObservedState::Absent,
        DeploymentDesiredState::Absent => DeploymentObservedState::ExternallyManaged,
    };
    DeploymentView {
        desired: record,
        observed_state,
        ready_endpoints,
        inflight_requests,
        externally_managed: true,
    }
}

fn replace_record(
    records: &mut Vec<DeploymentControlRecord>,
    replacement: DeploymentControlRecord,
) {
    if let Some(existing) = records
        .iter_mut()
        .find(|record| record.deployment.id == replacement.deployment.id)
    {
        *existing = replacement;
    } else {
        records.push(replacement);
    }
}

async fn queue_mutation(
    layer: Arc<OrchestratorLayer>,
    request_id: ax_serving_protocol::RequestId,
    action: DeploymentJobAction,
    record: DeploymentControlRecord,
    expected_generation: Option<u64>,
) -> Result<DeploymentJobRecord, Response> {
    let mut records = layer.fleet_store.list_deployments().await.map_err(|_| {
        lifecycle_error(
            StatusCode::SERVICE_UNAVAILABLE,
            request_id,
            "AXS_FLEET_STATE_UNAVAILABLE",
            "shared deployment state is temporarily unavailable",
            true,
        )
    })?;
    replace_record(&mut records, record.clone());
    layer
        .deployment_catalog
        .catalog_for_records(&records)
        .map_err(|error| {
            lifecycle_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                request_id,
                "AXS_INVALID_DEPLOYMENT",
                error.to_string(),
                false,
            )
        })?;

    let job = DeploymentJobRecord::queued(
        record.deployment.id.clone(),
        action,
        record.desired_state,
        record.generation,
    );
    layer
        .fleet_store
        .put_deployment_job(&job, DEPLOYMENT_JOB_TTL_MS)
        .await
        .map_err(|_| {
            lifecycle_error(
                StatusCode::SERVICE_UNAVAILABLE,
                request_id,
                "AXS_FLEET_STATE_UNAVAILABLE",
                "deployment job state is temporarily unavailable",
                true,
            )
        })?;

    let queued = job.clone();
    tokio::spawn(async move {
        run_mutation(layer, job, record, expected_generation).await;
    });
    Ok(queued)
}

async fn persist_job(layer: &OrchestratorLayer, job: &DeploymentJobRecord) {
    if let Err(error) = layer
        .fleet_store
        .put_deployment_job(job, DEPLOYMENT_JOB_TTL_MS)
        .await
    {
        warn!(job_id = %job.id, %error, "failed to persist deployment job update");
    }
}

async fn run_mutation(
    layer: Arc<OrchestratorLayer>,
    mut job: DeploymentJobRecord,
    record: DeploymentControlRecord,
    expected_generation: Option<u64>,
) {
    job.status = DeploymentJobStatus::Running;
    job.progress_percent = 25;
    job.updated_at = OffsetDateTime::now_utc();
    persist_job(&layer, &job).await;

    let result = layer
        .fleet_store
        .put_deployment_if_generation(&record, expected_generation)
        .await;
    match result {
        Ok(FleetMutationResult::Applied) => {}
        Ok(FleetMutationResult::Missing | FleetMutationResult::Fenced) => {
            fail_job(
                &layer,
                &mut job,
                "AXS_DEPLOYMENT_CONFLICT",
                "deployment changed while the job was queued",
            )
            .await;
            return;
        }
        Ok(FleetMutationResult::StaleSequence) => {
            fail_job(
                &layer,
                &mut job,
                "AXS_DEPLOYMENT_CONFLICT",
                "deployment generation is stale",
            )
            .await;
            return;
        }
        Err(error) => {
            warn!(job_id = %job.id, %error, "deployment desired-state mutation failed");
            fail_job(
                &layer,
                &mut job,
                "AXS_FLEET_STATE_UNAVAILABLE",
                "shared deployment state is temporarily unavailable",
            )
            .await;
            return;
        }
    }

    if let Err(error) = layer.reconcile_deployment_state().await {
        warn!(job_id = %job.id, %error, "deployment catalog reconciliation failed");
        fail_job(
            &layer,
            &mut job,
            "AXS_DEPLOYMENT_RECONCILE_FAILED",
            "desired state was saved but local reconciliation failed",
        )
        .await;
        return;
    }

    let view = observed_view(&layer, record);
    job.status = DeploymentJobStatus::Succeeded;
    job.observed_state = view.observed_state;
    job.progress_percent = 100;
    let now = OffsetDateTime::now_utc();
    job.updated_at = now;
    job.completed_at = Some(now);
    persist_job(&layer, &job).await;
    layer.audit.record(
        "operator",
        format!("deployment.{:?}", job.action).to_ascii_lowercase(),
        job.deployment_id.to_string(),
        None,
        "ok",
        Some(serde_json::json!({
            "job_id": job.id,
            "generation": job.generation,
            "desired_state": job.desired_state,
            "observed_state": job.observed_state,
        })),
    );
    info!(job_id = %job.id, deployment_id = %job.deployment_id, "deployment job completed");
}

async fn fail_job(
    layer: &OrchestratorLayer,
    job: &mut DeploymentJobRecord,
    code: &str,
    message: &str,
) {
    job.status = DeploymentJobStatus::Failed;
    job.observed_state = DeploymentObservedState::Failed;
    job.failure_code = Some(code.to_string());
    job.failure_message = Some(message.to_string());
    let now = OffsetDateTime::now_utc();
    job.updated_at = now;
    job.completed_at = Some(now);
    persist_job(layer, job).await;
}

pub async fn create_deployment(
    State(layer): State<Arc<OrchestratorLayer>>,
    request_id_extension: Option<Extension<AxRequestId>>,
    Json(request): Json<CreateDeploymentRequest>,
) -> Response {
    let request_id = request_id(request_id_extension);
    if !explicit_mode(&layer) {
        return lifecycle_error(
            StatusCode::CONFLICT,
            request_id,
            "AXS_EXPLICIT_DEPLOYMENTS_REQUIRED",
            "deployment lifecycle requires deployment_mode=explicit",
            false,
        );
    }
    let current = match layer
        .fleet_store
        .get_deployment(&request.deployment.id)
        .await
    {
        Ok(current) => current,
        Err(_) => {
            return lifecycle_error(
                StatusCode::SERVICE_UNAVAILABLE,
                request_id,
                "AXS_FLEET_STATE_UNAVAILABLE",
                "shared deployment state is temporarily unavailable",
                true,
            );
        }
    };
    if current
        .as_ref()
        .is_some_and(|record| record.desired_state != DeploymentDesiredState::Absent)
    {
        return lifecycle_error(
            StatusCode::CONFLICT,
            request_id,
            "AXS_DEPLOYMENT_EXISTS",
            "deployment already exists",
            false,
        );
    }
    let generation = current.as_ref().map_or(1, |record| record.generation + 1);
    let desired_state = if request.deployment.enabled {
        DeploymentDesiredState::Enabled
    } else {
        DeploymentDesiredState::Disabled
    };
    let record = DeploymentControlRecord {
        deployment: request.deployment,
        generation,
        desired_state,
        updated_at: OffsetDateTime::now_utc(),
    };
    match queue_mutation(
        Arc::clone(&layer),
        request_id,
        DeploymentJobAction::Create,
        record,
        current.map(|record| record.generation),
    )
    .await
    {
        Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Err(response) => response,
    }
}

pub async fn patch_deployment(
    State(layer): State<Arc<OrchestratorLayer>>,
    Path(raw_id): Path<String>,
    request_id_extension: Option<Extension<AxRequestId>>,
    Json(request): Json<PatchDeploymentRequest>,
) -> Response {
    let request_id = request_id(request_id_extension);
    let deployment_id = match DeploymentId::new(raw_id) {
        Ok(id) => id,
        Err(_) => {
            return lifecycle_error(
                StatusCode::BAD_REQUEST,
                request_id,
                "AXS_INVALID_DEPLOYMENT_ID",
                "invalid deployment id",
                false,
            );
        }
    };
    let current = match layer.fleet_store.get_deployment(&deployment_id).await {
        Ok(Some(record)) if record.desired_state != DeploymentDesiredState::Absent => record,
        Ok(_) => {
            return lifecycle_error(
                StatusCode::NOT_FOUND,
                request_id,
                "AXS_DEPLOYMENT_NOT_FOUND",
                "deployment not found",
                false,
            );
        }
        Err(_) => {
            return lifecycle_error(
                StatusCode::SERVICE_UNAVAILABLE,
                request_id,
                "AXS_FLEET_STATE_UNAVAILABLE",
                "shared deployment state is temporarily unavailable",
                true,
            );
        }
    };
    let action = request.action.unwrap_or(DeploymentJobAction::Update);
    if action == DeploymentJobAction::Roll {
        return queue_rollout(layer, current, request, request_id).await;
    }
    if !matches!(
        action,
        DeploymentJobAction::Update | DeploymentJobAction::Drain
    ) {
        return lifecycle_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            request_id,
            "AXS_INVALID_DEPLOYMENT_ACTION",
            "PATCH supports update, roll, or drain",
            false,
        );
    }
    let mut deployment = request.deployment.unwrap_or(current.deployment.clone());
    if deployment.id != deployment_id {
        return lifecycle_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            request_id,
            "AXS_DEPLOYMENT_ID_MISMATCH",
            "deployment body id must match the request path",
            false,
        );
    }
    let desired_state = if action == DeploymentJobAction::Drain {
        DeploymentDesiredState::Disabled
    } else if request.enabled.unwrap_or(deployment.enabled) {
        DeploymentDesiredState::Enabled
    } else {
        DeploymentDesiredState::Disabled
    };
    deployment.enabled = desired_state == DeploymentDesiredState::Enabled;
    let record = DeploymentControlRecord {
        deployment,
        generation: current.generation + 1,
        desired_state,
        updated_at: OffsetDateTime::now_utc(),
    };
    match queue_mutation(
        Arc::clone(&layer),
        request_id,
        action,
        record,
        Some(current.generation),
    )
    .await
    {
        Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Err(response) => response,
    }
}

async fn queue_rollout(
    layer: Arc<OrchestratorLayer>,
    source: DeploymentControlRecord,
    request: PatchDeploymentRequest,
    request_id: ax_serving_protocol::RequestId,
) -> Response {
    let Some(target_id) = request.replacement_deployment_id else {
        return lifecycle_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            request_id,
            "AXS_ROLLOUT_TARGET_REQUIRED",
            "roll requires replacement_deployment_id",
            false,
        );
    };
    let target = match layer.fleet_store.get_deployment(&target_id).await {
        Ok(Some(record)) if record.desired_state != DeploymentDesiredState::Absent => record,
        Ok(_) => {
            return lifecycle_error(
                StatusCode::NOT_FOUND,
                request_id,
                "AXS_DEPLOYMENT_NOT_FOUND",
                "replacement deployment not found",
                false,
            );
        }
        Err(_) => {
            return lifecycle_error(
                StatusCode::SERVICE_UNAVAILABLE,
                request_id,
                "AXS_FLEET_STATE_UNAVAILABLE",
                "shared deployment state is temporarily unavailable",
                true,
            );
        }
    };
    let catalog = layer.deployment_catalog.snapshot();
    if source.deployment.logical_model != target.deployment.logical_model
        || !catalog.permits_failover(&source.deployment, &target.deployment)
    {
        return lifecycle_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            request_id,
            "AXS_UNSAFE_ROLLOUT",
            "rollout source and replacement are not certified equivalent",
            false,
        );
    }
    let mut enabled_target = target.clone();
    enabled_target.generation += 1;
    enabled_target.desired_state = DeploymentDesiredState::Enabled;
    enabled_target.deployment.enabled = true;
    enabled_target.updated_at = OffsetDateTime::now_utc();
    let timeout_ms = request
        .timeout_ms
        .unwrap_or(DEFAULT_ROLLOUT_TIMEOUT_MS)
        .clamp(100, MAX_ROLLOUT_TIMEOUT_MS);
    let job = DeploymentJobRecord::queued(
        source.deployment.id.clone(),
        DeploymentJobAction::Roll,
        DeploymentDesiredState::Disabled,
        source.generation + 1,
    );
    if layer
        .fleet_store
        .put_deployment_job(&job, DEPLOYMENT_JOB_TTL_MS)
        .await
        .is_err()
    {
        return lifecycle_error(
            StatusCode::SERVICE_UNAVAILABLE,
            request_id,
            "AXS_FLEET_STATE_UNAVAILABLE",
            "deployment job state is temporarily unavailable",
            true,
        );
    }
    let queued = job.clone();
    tokio::spawn(async move {
        run_rollout(layer, job, source, target, enabled_target, timeout_ms).await;
    });
    (StatusCode::ACCEPTED, Json(queued)).into_response()
}

async fn run_rollout(
    layer: Arc<OrchestratorLayer>,
    mut job: DeploymentJobRecord,
    source: DeploymentControlRecord,
    target_before: DeploymentControlRecord,
    target_enabled: DeploymentControlRecord,
    timeout_ms: u64,
) {
    job.status = DeploymentJobStatus::Running;
    job.progress_percent = 10;
    job.updated_at = OffsetDateTime::now_utc();
    persist_job(&layer, &job).await;
    if !matches!(
        layer
            .fleet_store
            .put_deployment_if_generation(&target_enabled, Some(target_before.generation))
            .await,
        Ok(FleetMutationResult::Applied)
    ) {
        fail_job(
            &layer,
            &mut job,
            "AXS_DEPLOYMENT_CONFLICT",
            "replacement deployment changed before rollout activation",
        )
        .await;
        return;
    }
    if layer.reconcile_deployment_state().await.is_err() {
        fail_job(
            &layer,
            &mut job,
            "AXS_DEPLOYMENT_RECONCILE_FAILED",
            "replacement desired state could not be reconciled",
        )
        .await;
        return;
    }

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    loop {
        let catalog = layer.deployment_catalog.snapshot();
        let (ready, _) =
            catalog.observed_endpoint_summary(&layer.registry, &target_enabled.deployment);
        if ready > 0 {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            let mut rollback = target_before.clone();
            rollback.generation = target_enabled.generation + 1;
            rollback.updated_at = OffsetDateTime::now_utc();
            let _ = layer
                .fleet_store
                .put_deployment_if_generation(&rollback, Some(target_enabled.generation))
                .await;
            let _ = layer.reconcile_deployment_state().await;
            fail_job(
                &layer,
                &mut job,
                "AXS_ROLLOUT_TIMEOUT",
                "replacement deployment did not become ready before the rollout deadline",
            )
            .await;
            return;
        }
        job.progress_percent = 50;
        job.updated_at = OffsetDateTime::now_utc();
        persist_job(&layer, &job).await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let mut disabled_source = source.clone();
    disabled_source.generation += 1;
    disabled_source.desired_state = DeploymentDesiredState::Disabled;
    disabled_source.deployment.enabled = false;
    disabled_source.updated_at = OffsetDateTime::now_utc();
    if !matches!(
        layer
            .fleet_store
            .put_deployment_if_generation(&disabled_source, Some(source.generation))
            .await,
        Ok(FleetMutationResult::Applied)
    ) {
        fail_job(
            &layer,
            &mut job,
            "AXS_DEPLOYMENT_CONFLICT",
            "source deployment changed during rollout",
        )
        .await;
        return;
    }
    let _ = layer.reconcile_deployment_state().await;
    job.status = DeploymentJobStatus::Succeeded;
    job.observed_state = DeploymentObservedState::Ready;
    job.progress_percent = 100;
    let now = OffsetDateTime::now_utc();
    job.updated_at = now;
    job.completed_at = Some(now);
    persist_job(&layer, &job).await;
}

pub async fn delete_deployment(
    State(layer): State<Arc<OrchestratorLayer>>,
    Path(raw_id): Path<String>,
    request_id_extension: Option<Extension<AxRequestId>>,
) -> Response {
    let request_id = request_id(request_id_extension);
    let deployment_id = match DeploymentId::new(raw_id) {
        Ok(id) => id,
        Err(_) => {
            return lifecycle_error(
                StatusCode::BAD_REQUEST,
                request_id,
                "AXS_INVALID_DEPLOYMENT_ID",
                "invalid deployment id",
                false,
            );
        }
    };
    let current = match layer.fleet_store.get_deployment(&deployment_id).await {
        Ok(Some(record)) if record.desired_state != DeploymentDesiredState::Absent => record,
        Ok(_) => {
            return lifecycle_error(
                StatusCode::NOT_FOUND,
                request_id,
                "AXS_DEPLOYMENT_NOT_FOUND",
                "deployment not found",
                false,
            );
        }
        Err(_) => {
            return lifecycle_error(
                StatusCode::SERVICE_UNAVAILABLE,
                request_id,
                "AXS_FLEET_STATE_UNAVAILABLE",
                "shared deployment state is temporarily unavailable",
                true,
            );
        }
    };
    let mut tombstone = current.clone();
    tombstone.generation += 1;
    tombstone.desired_state = DeploymentDesiredState::Absent;
    tombstone.deployment.enabled = false;
    tombstone.updated_at = OffsetDateTime::now_utc();
    match queue_mutation(
        Arc::clone(&layer),
        request_id,
        DeploymentJobAction::Delete,
        tombstone,
        Some(current.generation),
    )
    .await
    {
        Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Err(response) => response,
    }
}

pub async fn get_deployment(
    State(layer): State<Arc<OrchestratorLayer>>,
    Path(raw_id): Path<String>,
    request_id_extension: Option<Extension<AxRequestId>>,
) -> Response {
    let request_id = request_id(request_id_extension);
    let deployment_id = match DeploymentId::new(raw_id) {
        Ok(id) => id,
        Err(_) => {
            return lifecycle_error(
                StatusCode::BAD_REQUEST,
                request_id,
                "AXS_INVALID_DEPLOYMENT_ID",
                "invalid deployment id",
                false,
            );
        }
    };
    match layer.fleet_store.get_deployment(&deployment_id).await {
        Ok(Some(record)) if record.desired_state != DeploymentDesiredState::Absent => {
            Json(observed_view(&layer, record)).into_response()
        }
        Ok(_) => lifecycle_error(
            StatusCode::NOT_FOUND,
            request_id,
            "AXS_DEPLOYMENT_NOT_FOUND",
            "deployment not found",
            false,
        ),
        Err(_) => lifecycle_error(
            StatusCode::SERVICE_UNAVAILABLE,
            request_id,
            "AXS_FLEET_STATE_UNAVAILABLE",
            "shared deployment state is temporarily unavailable",
            true,
        ),
    }
}

pub async fn list_deployments(State(layer): State<Arc<OrchestratorLayer>>) -> Response {
    match layer.fleet_store.list_deployments().await {
        Ok(mut records) => {
            records.retain(|record| record.desired_state != DeploymentDesiredState::Absent);
            records.sort_by(|left, right| left.deployment.id.cmp(&right.deployment.id));
            let deployments = records
                .into_iter()
                .map(|record| observed_view(&layer, record))
                .collect::<Vec<_>>();
            Json(serde_json::json!({ "deployments": deployments })).into_response()
        }
        Err(_) => lifecycle_error(
            StatusCode::SERVICE_UNAVAILABLE,
            ax_serving_protocol::RequestId::new(),
            "AXS_FLEET_STATE_UNAVAILABLE",
            "shared deployment state is temporarily unavailable",
            true,
        ),
    }
}

pub async fn get_job(
    State(layer): State<Arc<OrchestratorLayer>>,
    Path(raw_id): Path<String>,
    request_id_extension: Option<Extension<AxRequestId>>,
) -> Response {
    let request_id = request_id(request_id_extension);
    let job_id = match raw_id.parse::<JobId>() {
        Ok(id) => id,
        Err(_) => {
            return lifecycle_error(
                StatusCode::BAD_REQUEST,
                request_id,
                "AXS_INVALID_JOB_ID",
                "invalid deployment job id",
                false,
            );
        }
    };
    match layer.fleet_store.get_deployment_job(job_id).await {
        Ok(Some(job)) => Json(job).into_response(),
        Ok(None) => lifecycle_error(
            StatusCode::NOT_FOUND,
            request_id,
            "AXS_JOB_NOT_FOUND",
            "deployment job not found",
            false,
        ),
        Err(_) => lifecycle_error(
            StatusCode::SERVICE_UNAVAILABLE,
            request_id,
            "AXS_FLEET_STATE_UNAVAILABLE",
            "deployment job state is temporarily unavailable",
            true,
        ),
    }
}

pub async fn list_jobs(
    State(layer): State<Arc<OrchestratorLayer>>,
    Query(query): Query<ListQuery>,
) -> Response {
    match layer.fleet_store.list_deployment_jobs().await {
        Ok(mut jobs) => {
            jobs.sort_by_key(|job| std::cmp::Reverse(job.created_at));
            jobs.truncate(query.limit.clamp(1, 500));
            Json(serde_json::json!({ "jobs": jobs })).into_response()
        }
        Err(_) => lifecycle_error(
            StatusCode::SERVICE_UNAVAILABLE,
            ax_serving_protocol::RequestId::new(),
            "AXS_FLEET_STATE_UNAVAILABLE",
            "deployment job state is temporarily unavailable",
            true,
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use ax_serving_protocol::{
        AgentDescriptor, CURRENT_PROTOCOL, CapacityObservation, DeploymentDesiredState,
        DeploymentId, DeploymentIdentity, DeploymentJobRecord, DeploymentJobStatus, DeploymentSpec,
        Digest, EquivalenceClassId, EquivalencePolicy, HardwareDescriptor, IdentityPolicy,
        LogicalModelId, NegotiatedProtocol, Operation, PoolId, PoolSpec, ProtocolDescriptor,
        RegisterWorkerRequest, RuntimeDescriptor, RuntimeModelDescriptor, RuntimeModelId,
        RuntimeObservation, RuntimeStatus, TrustDomainId, WorkerDescriptor, WorkerId,
        WorkerInstanceId,
    };
    use axum::body::Body;
    use axum::http::Request;
    use time::OffsetDateTime;
    use tower::ServiceExt;

    use super::super::{OrchestratorConfig, OrchestratorLayer, ProjectPolicyConfig, proxy_router};

    fn deployment(id: &str, logical_model: &str, enabled: bool) -> DeploymentSpec {
        DeploymentSpec {
            id: DeploymentId::new(id).unwrap(),
            logical_model: LogicalModelId::new(logical_model).unwrap(),
            pool: PoolId::new("cuda").unwrap(),
            domain: None,
            runtime_model_id: RuntimeModelId::new(format!("runtime/{id}")).unwrap(),
            equivalence_class: None,
            expected_identity: None,
            required_identity: Default::default(),
            required_capabilities: BTreeSet::new(),
            enabled,
        }
    }

    fn digest(value: char) -> Digest {
        Digest::new(format!("sha256:{}", value.to_string().repeat(64))).unwrap()
    }

    fn identity() -> DeploymentIdentity {
        DeploymentIdentity {
            runtime_kind: "vllm".into(),
            runtime_version: Some("1.0.0".into()),
            revision: Some("revision-1".into()),
            artifact_digest: Some(digest('a')),
            tokenizer_digest: Some(digest('b')),
            template_digest: Some(digest('c')),
            quantization: Some("int4".into()),
        }
    }

    async fn layer() -> Arc<OrchestratorLayer> {
        let config = OrchestratorConfig {
            deployment_mode: "explicit".into(),
            pools: vec![PoolSpec {
                id: PoolId::new("cuda").unwrap(),
                runtime_kind: "vllm".into(),
                hardware_class: Some("cuda".into()),
                trust_domain: TrustDomainId::new("private").unwrap(),
                selector: BTreeMap::new(),
            }],
            deployments: vec![deployment("baseline", "public/baseline", true)],
            ..OrchestratorConfig::default()
        };
        let layer =
            Arc::new(OrchestratorLayer::new(config, ProjectPolicyConfig::default()).unwrap());
        layer.reconcile_deployment_state().await.unwrap();
        layer
    }

    async fn wait_for_job(layer: &OrchestratorLayer, id: &str) -> DeploymentJobRecord {
        let id = id.parse().unwrap();
        for _ in 0..100 {
            if let Some(job) = layer.fleet_store.get_deployment_job(id).await.unwrap()
                && matches!(
                    job.status,
                    DeploymentJobStatus::Succeeded | DeploymentJobStatus::Failed
                )
            {
                return job;
            }
            tokio::task::yield_now().await;
        }
        panic!("deployment job did not complete");
    }

    #[tokio::test]
    async fn create_and_drain_jobs_update_shared_desired_state() {
        let layer = layer().await;
        let app = proxy_router(Arc::clone(&layer));
        let create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/v1/deployments")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "deployment": deployment("dynamic", "public/dynamic", true)
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(create.status(), axum::http::StatusCode::ACCEPTED);
        let create: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(create.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let created = wait_for_job(&layer, create["id"].as_str().unwrap()).await;
        assert_eq!(created.status, DeploymentJobStatus::Succeeded);
        assert!(
            layer
                .deployment_catalog
                .snapshot()
                .resolve("public/dynamic")
                .is_ok()
        );

        let drain = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/v1/deployments/dynamic")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"action":"drain"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(drain.status(), axum::http::StatusCode::ACCEPTED);
        let drain: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(drain.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let drained = wait_for_job(&layer, drain["id"].as_str().unwrap()).await;
        assert_eq!(drained.status, DeploymentJobStatus::Succeeded);
        assert!(
            layer
                .deployment_catalog
                .snapshot()
                .resolve("public/dynamic")
                .is_err()
        );
    }

    #[tokio::test]
    async fn duplicate_create_is_rejected_before_a_job_is_queued() {
        let layer = layer().await;
        let response = proxy_router(layer)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/v1/deployments")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "deployment": deployment("baseline", "public/baseline", true)
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn observed_summary_counts_only_protocol_leased_endpoints() {
        // Regression: `observed_endpoint_summary` counted endpoints that
        // `route_candidates` would never select because they lack a fenced
        // protocol-v1 lease. The rollout readiness gate and "Ready" reporting
        // must only count explicitly routable endpoints.
        use super::super::registry::{RegisterCapabilities, RegisterRequest};

        let layer = layer().await;

        // Legacy (non-protocol) worker serving the deployment's runtime model.
        let _ = layer.registry.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:18082".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["runtime/baseline".into()]),
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                hardware_class: Some("cuda".into()),
                worker_pool: Some("cuda".into()),
                max_inflight: 4,
                ..Default::default()
            },
            5_000,
        );

        // Protocol-v1 worker with a fenced lease serving the same model.
        layer
            .registry
            .register_protocol(
                RegisterWorkerRequest {
                    protocol: ProtocolDescriptor::current(BTreeSet::new()),
                    agent: AgentDescriptor {
                        name: "test-agent".into(),
                        version: "1".into(),
                        build_sha: None,
                    },
                    worker: WorkerDescriptor {
                        id: WorkerId::new("summary-worker").unwrap(),
                        instance_id: WorkerInstanceId::new(),
                        advertise_url: "http://127.0.0.1:18083".into(),
                        pool_id: PoolId::new("cuda").unwrap(),
                        trust_domain: TrustDomainId::new("private").unwrap(),
                        labels: BTreeMap::new(),
                    },
                    runtime: RuntimeDescriptor {
                        kind: "vllm".into(),
                        version: "1.0.0".into(),
                        api: "openai-v1".into(),
                    },
                    hardware: HardwareDescriptor {
                        platform: "linux".into(),
                        accelerator: "nvidia-gpu".into(),
                        device_count: 1,
                        memory_bytes: None,
                        hardware_class: Some("cuda".into()),
                    },
                    domain: None,
                    domain_observation: None,
                    observation: RuntimeObservation {
                        observed_at: OffsetDateTime::now_utc(),
                        runtime: RuntimeStatus::ready(),
                        inventory_generation: 1,
                        models: vec![RuntimeModelDescriptor {
                            runtime_model_id: RuntimeModelId::new("runtime/baseline").unwrap(),
                            identity: identity(),
                            operations: BTreeSet::from([Operation::chat_completions()]),
                            capabilities: BTreeSet::new(),
                            max_context_tokens: Some(32_768),
                            max_output_tokens: Some(4_096),
                        }],
                        capacity: Some(CapacityObservation {
                            active_requests: Some(0),
                            max_concurrent_requests: Some(4),
                            ..Default::default()
                        }),
                    },
                },
                "127.0.0.1:18083".parse().unwrap(),
                NegotiatedProtocol {
                    version: CURRENT_PROTOCOL,
                    capabilities: BTreeSet::new(),
                },
                5_000,
                15_000,
            )
            .unwrap();

        let catalog = layer.deployment_catalog.snapshot();
        let spec = catalog
            .deployment(&DeploymentId::new("baseline").unwrap())
            .unwrap();
        let (ready, _inflight) = catalog.observed_endpoint_summary(&layer.registry, spec);
        assert_eq!(
            ready, 1,
            "only the protocol-leased endpoint may count toward rollout readiness"
        );
    }

    #[tokio::test]
    async fn rollout_enables_ready_replacement_before_disabling_source() {
        let source_id = DeploymentId::new("source").unwrap();
        let target_id = DeploymentId::new("target").unwrap();
        let class_id = EquivalenceClassId::new("certified").unwrap();
        let logical_model = LogicalModelId::new("public/rollout").unwrap();
        let source = DeploymentSpec {
            id: source_id.clone(),
            logical_model: logical_model.clone(),
            pool: PoolId::new("source-pool").unwrap(),
            domain: None,
            runtime_model_id: RuntimeModelId::new("runtime/source").unwrap(),
            equivalence_class: Some(class_id.clone()),
            expected_identity: Some(identity()),
            required_identity: IdentityPolicy::strict_cross_runtime(),
            required_capabilities: BTreeSet::new(),
            enabled: true,
        };
        let target = DeploymentSpec {
            id: target_id.clone(),
            logical_model,
            pool: PoolId::new("target-pool").unwrap(),
            domain: None,
            runtime_model_id: RuntimeModelId::new("runtime/target").unwrap(),
            equivalence_class: Some(class_id.clone()),
            expected_identity: Some(identity()),
            required_identity: IdentityPolicy::strict_cross_runtime(),
            required_capabilities: BTreeSet::new(),
            enabled: false,
        };
        let config = OrchestratorConfig {
            deployment_mode: "explicit".into(),
            pools: vec![
                PoolSpec {
                    id: PoolId::new("source-pool").unwrap(),
                    runtime_kind: "vllm".into(),
                    hardware_class: Some("cuda".into()),
                    trust_domain: TrustDomainId::new("private").unwrap(),
                    selector: BTreeMap::new(),
                },
                PoolSpec {
                    id: PoolId::new("target-pool").unwrap(),
                    runtime_kind: "vllm".into(),
                    hardware_class: Some("cuda".into()),
                    trust_domain: TrustDomainId::new("private").unwrap(),
                    selector: BTreeMap::new(),
                },
            ],
            deployments: vec![source, target.clone()],
            equivalence_classes: vec![EquivalencePolicy {
                id: class_id,
                identity_policy: IdentityPolicy::strict_cross_runtime(),
                certified_deployments: BTreeSet::from([source_id.clone(), target_id.clone()]),
                certification_artifact: "tests/rollout-certification.json".into(),
            }],
            ..OrchestratorConfig::default()
        };
        let layer =
            Arc::new(OrchestratorLayer::new(config, ProjectPolicyConfig::default()).unwrap());
        layer.reconcile_deployment_state().await.unwrap();
        let worker_addr = "127.0.0.1:18081".parse().unwrap();
        layer
            .registry
            .register_protocol(
                RegisterWorkerRequest {
                    protocol: ProtocolDescriptor::current(BTreeSet::new()),
                    agent: AgentDescriptor {
                        name: "test-agent".into(),
                        version: "1".into(),
                        build_sha: None,
                    },
                    worker: WorkerDescriptor {
                        id: WorkerId::new("target-worker").unwrap(),
                        instance_id: WorkerInstanceId::new(),
                        advertise_url: "http://127.0.0.1:18081".into(),
                        pool_id: PoolId::new("target-pool").unwrap(),
                        trust_domain: TrustDomainId::new("private").unwrap(),
                        labels: BTreeMap::new(),
                    },
                    runtime: RuntimeDescriptor {
                        kind: "vllm".into(),
                        version: "1.0.0".into(),
                        api: "openai-v1".into(),
                    },
                    hardware: HardwareDescriptor {
                        platform: "linux".into(),
                        accelerator: "nvidia-gpu".into(),
                        device_count: 1,
                        memory_bytes: None,
                        hardware_class: Some("cuda".into()),
                    },
                    domain: None,
                    domain_observation: None,
                    observation: RuntimeObservation {
                        observed_at: OffsetDateTime::now_utc(),
                        runtime: RuntimeStatus::ready(),
                        inventory_generation: 1,
                        models: vec![RuntimeModelDescriptor {
                            runtime_model_id: target.runtime_model_id,
                            identity: identity(),
                            operations: BTreeSet::from([Operation::chat_completions()]),
                            capabilities: BTreeSet::new(),
                            max_context_tokens: Some(32_768),
                            max_output_tokens: Some(4_096),
                        }],
                        capacity: Some(CapacityObservation {
                            active_requests: Some(0),
                            max_concurrent_requests: Some(4),
                            ..Default::default()
                        }),
                    },
                },
                worker_addr,
                NegotiatedProtocol {
                    version: CURRENT_PROTOCOL,
                    capabilities: BTreeSet::new(),
                },
                5_000,
                15_000,
            )
            .unwrap();

        let response = proxy_router(Arc::clone(&layer))
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/v1/deployments/source")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"action":"roll","replacement_deployment_id":"target","timeout_ms":500}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        let response: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let job = wait_for_job(&layer, response["id"].as_str().unwrap()).await;
        assert_eq!(job.status, DeploymentJobStatus::Succeeded);
        let source = layer
            .fleet_store
            .get_deployment(&source_id)
            .await
            .unwrap()
            .unwrap();
        let target = layer
            .fleet_store
            .get_deployment(&target_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(source.desired_state, DeploymentDesiredState::Disabled);
        assert_eq!(target.desired_state, DeploymentDesiredState::Enabled);
    }
}
