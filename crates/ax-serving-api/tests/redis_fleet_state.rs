use std::collections::BTreeSet;

use ax_serving_api::orchestration::fleet_state::{
    FleetMutationResult, FleetStateStore, RedisFleetStateStore, ReservationResult,
};
use ax_serving_protocol::{
    AttemptId, DeploymentControlRecord, DeploymentDesiredState, DeploymentId, DeploymentJobAction,
    DeploymentJobRecord, DeploymentSpec, IdentityPolicy, LogicalModelId, PoolId, RuntimeModelId,
    WorkerId,
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
}
