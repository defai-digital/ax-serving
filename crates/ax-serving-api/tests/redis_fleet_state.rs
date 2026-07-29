use std::collections::{BTreeMap, BTreeSet};

use ax_serving_api::orchestration::fleet_state::{
    DomainReservationResult, FleetMutationResult, FleetStateStore, RedisFleetStateStore,
    ReservationResult,
};
use ax_serving_protocol::{
    AttemptId, CandidateDecision, DecisionReasonCode, DecisionRecordV1, DeploymentControlRecord,
    DeploymentDesiredState, DeploymentId, DeploymentJobAction, DeploymentJobRecord, DeploymentSpec,
    DomainId, IdentityPolicy, LogicalModelId, Operation, PolicyId, PolicyMode, PolicyVersion,
    PoolId, RequestId, RuntimeModelId, WorkerId,
};

fn deployment() -> DeploymentSpec {
    DeploymentSpec {
        id: DeploymentId::new("redis-deployment").unwrap(),
        logical_model: LogicalModelId::new("public/redis").unwrap(),
        pool: PoolId::new("redis-pool").unwrap(),
        domain: None,
        runtime_model_id: RuntimeModelId::new("runtime/redis").unwrap(),
        equivalence_class: None,
        expected_identity: None,
        required_identity: IdentityPolicy {
            required_matching_fields: BTreeSet::new(),
        },
        required_capabilities: BTreeSet::new(),
        enabled: true,
    }
}

fn decision() -> DecisionRecordV1 {
    let domain = DomainId::new("redis-domain").unwrap();
    let deployment = DeploymentId::new("redis-deployment").unwrap();
    DecisionRecordV1 {
        request_id: RequestId::new(),
        operation: Operation::chat_completions(),
        logical_model: LogicalModelId::new("public/redis").unwrap(),
        routing_profile: None,
        policy_id: PolicyId::new("explicit-catalog").unwrap(),
        policy_version: PolicyVersion::new("1").unwrap(),
        policy_mode: PolicyMode::Active,
        candidate_summary: vec![CandidateDecision {
            domain: domain.clone(),
            deployment: deployment.clone(),
            eligible: true,
            rejection_reasons: BTreeSet::new(),
            normalized_score_microunits: None,
        }],
        selected_domain: domain.clone(),
        selected_deployment: deployment,
        reason_codes: BTreeSet::from([DecisionReasonCode::OnlyEligible]),
        observation_generations: BTreeMap::from([(domain, 4)]),
        predicted_cost_microusd: None,
        predicted_latency_ms: None,
        decided_at: time::OffsetDateTime::now_utc(),
    }
}

#[tokio::test]
async fn redis_store_enforces_reservations_generations_and_job_round_trips() {
    let Ok(url) = std::env::var("AXS_TEST_REDIS_URL") else {
        eprintln!("AXS_TEST_REDIS_URL is unset; skipping Redis fleet-state conformance test");
        return;
    };
    let prefix = format!("axs:test:{}", uuid::Uuid::new_v4().simple());
    let store = RedisFleetStateStore::new(&url, &prefix).unwrap();

    let worker_id = WorkerId::new("redis-worker").unwrap();
    let first = AttemptId::new();
    let second = AttemptId::new();
    assert_eq!(
        store
            .try_reserve(&worker_id, first, 1, 5_000)
            .await
            .unwrap(),
        ReservationResult::Reserved
    );
    assert_eq!(
        store
            .try_reserve(&worker_id, second, 1, 5_000)
            .await
            .unwrap(),
        ReservationResult::Saturated
    );
    store.release_reservation(&worker_id, first).await.unwrap();
    assert_eq!(
        store
            .try_reserve(&worker_id, second, 1, 5_000)
            .await
            .unwrap(),
        ReservationResult::Reserved
    );

    let domain_id = DomainId::new("redis-domain").unwrap();
    let domain_first = AttemptId::new();
    let domain_second = AttemptId::new();
    assert_eq!(
        store
            .try_reserve_domain(&domain_id, 3, domain_first, 1, 5_000)
            .await
            .unwrap(),
        DomainReservationResult::Reserved
    );
    assert_eq!(
        store
            .try_reserve_domain(&domain_id, 4, domain_second, 1, 5_000)
            .await
            .unwrap(),
        DomainReservationResult::GenerationFenced
    );
    store
        .release_domain_reservation(&domain_id, domain_first)
        .await
        .unwrap();
    assert_eq!(
        store
            .try_reserve_domain(&domain_id, 4, domain_second, 1, 5_000)
            .await
            .unwrap(),
        DomainReservationResult::Reserved
    );
    assert!(
        store
            .try_acquire_probe_lease(&worker_id, "gateway-a", 5_000)
            .await
            .unwrap()
    );
    assert!(
        !store
            .try_acquire_probe_lease(&worker_id, "gateway-b", 5_000)
            .await
            .unwrap()
    );
    assert!(
        store
            .try_acquire_probe_lease(&worker_id, "gateway-a", 5_000)
            .await
            .unwrap()
    );

    let mut control = DeploymentControlRecord {
        deployment: deployment(),
        generation: 1,
        desired_state: DeploymentDesiredState::Enabled,
        updated_at: time::OffsetDateTime::now_utc(),
    };
    assert_eq!(
        store
            .put_deployment_if_generation(&control, None)
            .await
            .unwrap(),
        FleetMutationResult::Applied
    );
    control.generation = 2;
    assert_eq!(
        store
            .put_deployment_if_generation(&control, Some(0))
            .await
            .unwrap(),
        FleetMutationResult::Fenced
    );
    assert_eq!(
        store
            .put_deployment_if_generation(&control, Some(1))
            .await
            .unwrap(),
        FleetMutationResult::Applied
    );

    let job = DeploymentJobRecord::queued(
        control.deployment.id.clone(),
        DeploymentJobAction::Update,
        DeploymentDesiredState::Enabled,
        control.generation,
    );
    store.put_deployment_job(&job, 5_000).await.unwrap();
    assert_eq!(
        store.get_deployment_job(job.id).await.unwrap().unwrap(),
        job
    );

    let decision = decision();
    store.put_decision(&decision, 5_000).await.unwrap();
    assert_eq!(store.list_decisions(10).await.unwrap(), vec![decision]);
}
