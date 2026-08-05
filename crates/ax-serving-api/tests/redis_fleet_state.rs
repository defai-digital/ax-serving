use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ax_serving_api::orchestration::{
    OrchestratorConfig, OrchestratorLayer, ProjectPolicyConfig,
    fleet_state::{
        DomainReservationResult, FleetMutationResult, FleetStateStore, RedisFleetStateStore,
        ReservationResult,
    },
    proxy_router,
    worker_endpoint::WorkerEndpoint,
};
use ax_serving_protocol::{
    AgentDescriptor, AttemptId, CURRENT_PROTOCOL, CandidateDecision, CapacityObservation,
    CompatibilityManifestDigest, DecisionReasonCode, DecisionRecordV1, DeploymentControlRecord,
    DeploymentDesiredState, DeploymentId, DeploymentJobAction, DeploymentJobRecord, DeploymentSpec,
    DomainId, DomainObservation, DomainSpec, EndpointScope, ExecutionDomainDescriptor,
    ExecutionDomainKind, HardwareDescriptor, IdentityPolicy, LogicalModelId, NegotiatedProtocol,
    Operation, PolicyId, PolicyMode, PolicyVersion, PoolId, PoolSpec, ProtocolCapability,
    ProtocolDescriptor, QualificationState, RegisterWorkerRequest, RequestId, RuntimeDescriptor,
    RuntimeModelDescriptor, RuntimeModelId, RuntimeObservation, RuntimeState, RuntimeStatus,
    TrustDomainId, WorkerDescriptor, WorkerId, WorkerInstanceId,
};
use axum::{Json, Router, extract::State, routing::post};
use tokio::sync::Notify;
use tower::ServiceExt;

const CLUSTER_DOMAIN_ID: &str = "mac-cluster-ha";
const CLUSTER_MODEL_ID: &str = "llama-405b-int4-pp2";
const CLUSTER_LOGICAL_MODEL: &str = "llama/405b";
const CLUSTER_POOL_ID: &str = "mac-cluster";
const CLUSTER_TRUST_DOMAIN: &str = "private-lab";

#[derive(Default)]
struct BlockingWorker {
    calls: AtomicUsize,
    first_started: Notify,
    release_first: Notify,
}

async fn cluster_completion(
    State(state): State<Arc<BlockingWorker>>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let call = state.calls.fetch_add(1, Ordering::SeqCst);
    if call == 0 {
        state.first_started.notify_waiters();
        state.release_first.notified().await;
    }
    Json(serde_json::json!({
        "id": "cluster-response",
        "object": "chat.completion",
        "model": body["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    }))
}

async fn spawn_blocking_worker() -> Option<(SocketAddr, Arc<BlockingWorker>)> {
    let state = Arc::new(BlockingWorker::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(cluster_completion))
        .with_state(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let address = listener.local_addr().ok()?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Some((address, state))
}

fn manifest_digest() -> CompatibilityManifestDigest {
    CompatibilityManifestDigest::new(format!("sha256:{}", "a".repeat(64))).unwrap()
}

fn capabilities() -> BTreeSet<ProtocolCapability> {
    [
        ProtocolCapability::CONTROL_EXECUTION_DOMAIN,
        ProtocolCapability::CONTROL_MAC_CLUSTER,
        ProtocolCapability::DISPATCH_TYPED_ADMISSION,
        ProtocolCapability::TELEMETRY_CAPACITY,
        ProtocolCapability::TELEMETRY_DOMAIN_CAPACITY,
    ]
    .into_iter()
    .map(|value| ProtocolCapability::new(value).unwrap())
    .collect()
}

fn cluster_registration(address: SocketAddr, generation: u64) -> RegisterWorkerRequest {
    let pool = PoolId::new(CLUSTER_POOL_ID).unwrap();
    let trust_domain = TrustDomainId::new(CLUSTER_TRUST_DOMAIN).unwrap();
    let domain = DomainId::new(CLUSTER_DOMAIN_ID).unwrap();
    let model = RuntimeModelDescriptor {
        runtime_model_id: RuntimeModelId::new(CLUSTER_MODEL_ID).unwrap(),
        identity: ax_serving_protocol::DeploymentIdentity {
            runtime_kind: "ax_engine".into(),
            runtime_version: Some("6.12.0".into()),
            revision: Some("llama-405b-int4-rev1".into()),
            artifact_digest: Some(
                ax_serving_protocol::Digest::new(format!("sha256:{}", "b".repeat(64))).unwrap(),
            ),
            tokenizer_digest: Some(
                ax_serving_protocol::Digest::new(format!("sha256:{}", "c".repeat(64))).unwrap(),
            ),
            template_digest: Some(
                ax_serving_protocol::Digest::new(format!("sha256:{}", "d".repeat(64))).unwrap(),
            ),
            quantization: Some("int4".into()),
        },
        operations: BTreeSet::from([Operation::chat_completions()]),
        capabilities: BTreeSet::new(),
        max_context_tokens: Some(8_192),
        max_output_tokens: Some(2_048),
    };
    RegisterWorkerRequest {
        protocol: ProtocolDescriptor::current(capabilities()),
        agent: AgentDescriptor {
            name: "ax-mac-cluster-adapter".into(),
            version: "3.0.0".into(),
            build_sha: None,
        },
        worker: WorkerDescriptor {
            id: WorkerId::new("mac-cluster-adapter-ha").unwrap(),
            instance_id: WorkerInstanceId::new(),
            advertise_url: format!("http://{address}"),
            pool_id: pool.clone(),
            trust_domain: trust_domain.clone(),
            labels: BTreeMap::new(),
        },
        runtime: RuntimeDescriptor {
            kind: "ax_engine".into(),
            version: "6.12.0".into(),
            api: "openai-http".into(),
            endpoint: None,
        },
        hardware: HardwareDescriptor {
            platform: "macos".into(),
            accelerator: "apple-silicon-cluster".into(),
            device_count: 2,
            memory_bytes: Some(256 * 1024 * 1024 * 1024),
            hardware_class: Some("apple-silicon-cluster".into()),
        },
        domain: Some(ExecutionDomainDescriptor {
            id: domain.clone(),
            kind: ExecutionDomainKind::MacAxEngineCluster,
            endpoint_scope: EndpointScope::Domain,
            execution_owner: "ax_engine".into(),
            qualification: QualificationState::Experimental,
            pool_id: pool,
            trust_domain,
            hardware_class: "apple-silicon-cluster".into(),
            architecture: "arm64".into(),
            compatibility_manifest: Some(manifest_digest()),
            labels: BTreeMap::new(),
        }),
        domain_observation: Some(DomainObservation {
            observed_at: time::OffsetDateTime::now_utc(),
            generation,
            ready: true,
            state: RuntimeState::Ready,
            reason_code: None,
            frontend_instances_ready: Some(1),
            aggregate_capacity: Some(CapacityObservation {
                active_requests: Some(0),
                max_concurrent_requests: Some(1),
                waiting_requests: Some(0),
                ..Default::default()
            }),
            manifest_digest: Some(manifest_digest()),
            models: vec![model.clone()],
        }),
        observation: RuntimeObservation {
            observed_at: time::OffsetDateTime::now_utc(),
            runtime: RuntimeStatus::ready(),
            inventory_generation: generation,
            models: vec![model],
            capacity: Some(CapacityObservation {
                active_requests: Some(0),
                max_concurrent_requests: Some(1),
                waiting_requests: Some(0),
                ..Default::default()
            }),
        },
    }
}

fn cluster_config() -> OrchestratorConfig {
    let pool = PoolId::new(CLUSTER_POOL_ID).unwrap();
    let domain = DomainId::new(CLUSTER_DOMAIN_ID).unwrap();
    OrchestratorConfig {
        deployment_mode: "explicit".into(),
        pools: vec![PoolSpec {
            id: pool.clone(),
            runtime_kind: "ax_engine".into(),
            hardware_class: Some("apple-silicon-cluster".into()),
            trust_domain: TrustDomainId::new(CLUSTER_TRUST_DOMAIN).unwrap(),
            selector: BTreeMap::new(),
        }],
        domains: vec![DomainSpec {
            id: domain.clone(),
            kind: ExecutionDomainKind::MacAxEngineCluster,
            pool: pool.clone(),
            trust_domain: TrustDomainId::new(CLUSTER_TRUST_DOMAIN).unwrap(),
            hardware_class: "apple-silicon-cluster".into(),
            required_qualification: QualificationState::Experimental,
            selector: BTreeMap::new(),
            enabled: true,
        }],
        deployments: vec![DeploymentSpec {
            id: DeploymentId::new("llama-mac-cluster").unwrap(),
            logical_model: LogicalModelId::new(CLUSTER_LOGICAL_MODEL).unwrap(),
            pool,
            domain: Some(domain),
            runtime_model_id: RuntimeModelId::new(CLUSTER_MODEL_ID).unwrap(),
            equivalence_class: None,
            expected_identity: None,
            required_identity: IdentityPolicy {
                required_matching_fields: BTreeSet::new(),
            },
            required_capabilities: BTreeSet::new(),
            enabled: true,
        }],
        worker_ttl_ms: 5_000,
        ..OrchestratorConfig::default()
    }
}

fn gateway(store: Arc<dyn FleetStateStore>) -> Arc<OrchestratorLayer> {
    Arc::new(
        OrchestratorLayer::new_with_fleet_store(
            cluster_config(),
            ProjectPolicyConfig::default(),
            store,
        )
        .unwrap(),
    )
}

fn register_cluster(layer: &OrchestratorLayer, address: SocketAddr, generation: u64) {
    let registration = cluster_registration(address, generation);
    let negotiated = NegotiatedProtocol {
        version: CURRENT_PROTOCOL,
        capabilities: capabilities(),
    };
    layer
        .registry
        .register_protocol(
            registration,
            WorkerEndpoint::parse(&format!("http://{address}")).unwrap(),
            negotiated,
            1_000,
            5_000,
        )
        .unwrap();
}

fn request() -> axum::http::Request<axum::body::Body> {
    axum::http::Request::builder()
        .method(axum::http::Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(format!(
            r#"{{"model":"{CLUSTER_LOGICAL_MODEL}","messages":[{{"role":"user","content":"hello"}}]}}"#
        )))
        .unwrap()
}

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
        counterfactual_domain: None,
        rolled_back: false,
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

#[tokio::test]
async fn two_gateways_share_cluster_capacity_and_fence_generations() {
    let Ok(url) = std::env::var("AXS_TEST_REDIS_URL") else {
        eprintln!("AXS_TEST_REDIS_URL is unset; skipping two-gateway cluster HA test");
        return;
    };
    let Some((address, worker)) = spawn_blocking_worker().await else {
        eprintln!("loopback socket bind is unavailable; skipping two-gateway cluster HA test");
        return;
    };
    let prefix = format!("axs:test:mac-cluster-ha:{}", uuid::Uuid::new_v4().simple());
    let store_a: Arc<dyn FleetStateStore> =
        Arc::new(RedisFleetStateStore::new(&url, &prefix).unwrap());
    let store_b: Arc<dyn FleetStateStore> =
        Arc::new(RedisFleetStateStore::new(&url, &prefix).unwrap());
    let store_new_generation: Arc<dyn FleetStateStore> =
        Arc::new(RedisFleetStateStore::new(&url, &prefix).unwrap());
    let gateway_a = gateway(Arc::clone(&store_a));
    let gateway_b = gateway(Arc::clone(&store_b));
    let gateway_new_generation = gateway(store_new_generation);

    register_cluster(&gateway_a, address, 7);
    let worker_id = WorkerId::new("mac-cluster-adapter-ha").unwrap();
    let record = gateway_a
        .registry
        .export_protocol_record(&worker_id)
        .unwrap();
    store_a.put(&record).await.unwrap();
    gateway_b.reconcile_fleet_state().await.unwrap();
    register_cluster(&gateway_new_generation, address, 8);

    let first_router = proxy_router(Arc::clone(&gateway_a));
    let first_request = tokio::spawn(async move { first_router.oneshot(request()).await.unwrap() });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while worker.calls.load(Ordering::SeqCst) == 0 {
            worker.first_started.notified().await;
        }
    })
    .await
    .unwrap();

    let saturated = proxy_router(Arc::clone(&gateway_b))
        .oneshot(request())
        .await
        .unwrap();
    assert_eq!(
        saturated.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(worker.calls.load(Ordering::SeqCst), 1);

    let fenced = proxy_router(Arc::clone(&gateway_new_generation))
        .oneshot(request())
        .await
        .unwrap();
    assert_eq!(fenced.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(worker.calls.load(Ordering::SeqCst), 1);

    worker.release_first.notify_waiters();
    assert_eq!(
        first_request.await.unwrap().status(),
        axum::http::StatusCode::OK
    );

    let resumed = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let response = proxy_router(Arc::clone(&gateway_b))
                .oneshot(request())
                .await
                .unwrap();
            if response.status() == axum::http::StatusCode::OK {
                break response;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(resumed.status(), axum::http::StatusCode::OK);
    assert_eq!(worker.calls.load(Ordering::SeqCst), 2);

    let decisions = store_b.list_decisions(10).await.unwrap();
    assert_eq!(decisions.len(), 2);
    assert!(
        decisions
            .iter()
            .all(|decision| decision.selected_domain.as_str() == CLUSTER_DOMAIN_ID)
    );
    assert!(
        decisions
            .iter()
            .all(|decision| decision.observation_generations
                [&DomainId::new(CLUSTER_DOMAIN_ID).unwrap()]
                == 7)
    );
}
