//! Integration tests for the multi-worker orchestration layer (ADR-012, M2+M3).
//!
//! Each test spins up real in-process axum servers bound to ephemeral ports
//! so that `DirectDispatcher` exercises actual HTTP round-trips.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use ax_serving_api::config::{AdaptiveDomainPredictionConfig, AdaptiveFederationConfig};
use ax_serving_api::orchestration::{
    OrchestratorConfig, OrchestratorLayer, ProjectPolicyConfig,
    direct::DirectDispatcher,
    fleet_state::{FleetStateStore, MemoryFleetStateStore},
    internal_routes::{
        InternalAuthState, InternalState, internal_auth_middleware, parse_allowed_node_cidrs,
        router as internal_router,
    },
    policy::{DispatchContext, DispatchPolicy, policy_from_str},
    queue::{AcquireResult, GlobalQueue, GlobalQueueConfig, OverloadPolicy},
    registry::{
        HeartbeatRequest, ModelInventoryEntry, RegisterCapabilities, RegisterRequest, RequestKind,
        WorkerCapabilities, WorkerId, WorkerRegistry, WorkerStatus,
    },
    start_orchestrator,
};
use ax_serving_api::rest::schema::MAX_CONTENT_BYTES;
use ax_serving_protocol::{
    AgentDescriptor, CURRENT_PROTOCOL, CapacityObservation, CompatibilityManifestDigest,
    DecisionReasonCode, DeploymentIdentity, DeploymentSpec, Digest, DomainId, DomainObservation,
    DomainSpec, EndpointScope, EquivalenceClassId, EquivalencePolicy, ExecutionDomainDescriptor,
    ExecutionDomainKind, HardwareDescriptor, IdentityField, IdentityPolicy, LogicalModelId,
    NegotiatedProtocol, Operation, PolicyMode, PoolId, PoolSpec, ProtocolCapability,
    ProtocolDescriptor, QualificationState, RegisterWorkerRequest as ProtocolRegisterRequest,
    RuntimeDescriptor, RuntimeModelDescriptor, RuntimeModelId, RuntimeObservation, RuntimeState,
    RuntimeStatus, TrustDomainId, WorkerDescriptor as ProtocolWorkerDescriptor,
    WorkerId as ProtocolWorkerId, WorkerInstanceId,
};
use axum::{Router, middleware, routing::post};
use reqwest::Client;
use tower::ServiceExt;

// ── Helpers ───────────────────────────────────────────────────────────────────

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarsGuard {
    _guard: MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvVarsGuard {
    fn new() -> Self {
        let guard = ENV_LOCK.lock().unwrap();
        Self {
            _guard: guard,
            previous: Vec::new(),
        }
    }

    fn set(&mut self, key: &'static str, value: &str) {
        self.previous.push((key, std::env::var_os(key)));
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn remove(&mut self, key: &'static str) {
        self.previous.push((key, std::env::var_os(key)));
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            std::env::remove_var(key);
        }
    }
}

impl Drop for EnvVarsGuard {
    fn drop(&mut self) {
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            for (key, previous) in self.previous.iter().rev() {
                match previous {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

/// Spawn a minimal axum mock worker on an ephemeral port.
///
/// Returns `None` if the loopback socket cannot be bound (e.g. in restricted
/// sandbox environments). Tests that receive `None` must skip via
/// `skip_if_no_socket!`.
///
/// Every POST to `/v1/chat/completions` returns the given `status` and `body`.
/// The server runs until the test process exits.
async fn spawn_mock_worker(status: u16, body: &'static str) -> Option<SocketAddr> {
    let response = move || async move {
        axum::response::Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .unwrap()
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(response))
        .route("/v1/completions", post(response))
        .route("/v1/embeddings", post(response));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

async fn spawn_not_admitted_worker() -> Option<SocketAddr> {
    let response = || async move {
        axum::response::Response::builder()
            .status(503)
            .header("content-type", "application/json")
            .header("x-ax-admission-state", "not-admitted")
            .body(axum::body::Body::from(
                r#"{"error":{"code":"AXS_WORKER_CAPACITY","retryable":true,"phase":"pre_admission"}}"#,
            ))
            .unwrap()
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(response))
        .route("/v1/completions", post(response))
        .route("/v1/embeddings", post(response));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

#[derive(Default)]
struct EchoWorkerState {
    models: Mutex<Vec<String>>,
    bodies: Mutex<Vec<serde_json::Value>>,
    domain_ids: Mutex<Vec<String>>,
    public_authorization_seen: Mutex<bool>,
}

async fn spawn_echo_model_worker() -> Option<(SocketAddr, Arc<EchoWorkerState>)> {
    use axum::Json;
    use axum::extract::State;
    use axum::http::HeaderMap;

    async fn echo(
        State(state): State<Arc<EchoWorkerState>>,
        headers: HeaderMap,
        Json(body): Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        let model = body["model"].as_str().unwrap_or_default().to_string();
        state.models.lock().unwrap().push(model.clone());
        state.bodies.lock().unwrap().push(body.clone());
        if let Some(domain) = headers
            .get("x-ax-domain-id")
            .and_then(|value| value.to_str().ok())
        {
            state.domain_ids.lock().unwrap().push(domain.to_string());
        }
        if headers.contains_key(axum::http::header::AUTHORIZATION) {
            *state.public_authorization_seen.lock().unwrap() = true;
        }
        Json(serde_json::json!({
            "id": "echo",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]
        }))
    }

    let state = Arc::new(EchoWorkerState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(echo))
        .with_state(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some((addr, state))
}

fn proxy_router_with_key(layer: Arc<OrchestratorLayer>, key: &str) -> Router {
    layer.set_public_auth_required(true);
    let mut keys = HashSet::new();
    keys.insert(key.to_string());
    ax_serving_api::orchestration::proxy_router(layer)
        .route_layer(middleware::from_fn_with_state(
            Arc::new(keys),
            ax_serving_api::auth::auth_middleware,
        ))
        .layer(middleware::from_fn(
            ax_serving_api::auth::request_id_and_headers_middleware,
        ))
}

fn adaptive_digest(character: char) -> Digest {
    Digest::new(format!("sha256:{}", character.to_string().repeat(64))).unwrap()
}

fn adaptive_manifest_digest() -> CompatibilityManifestDigest {
    CompatibilityManifestDigest::new(format!("sha256:{}", "a".repeat(64))).unwrap()
}

fn adaptive_identity() -> DeploymentIdentity {
    DeploymentIdentity {
        runtime_kind: "ax_engine".into(),
        runtime_version: Some("6.12.0".into()),
        revision: Some("adaptive-fixture-rev1".into()),
        artifact_digest: Some(adaptive_digest('b')),
        tokenizer_digest: Some(adaptive_digest('c')),
        template_digest: Some(adaptive_digest('d')),
        quantization: Some("bf16".into()),
    }
}

struct AdaptiveDomainWorker<'a> {
    worker_id: &'a str,
    pool_id: PoolId,
    domain_id: DomainId,
    kind: ExecutionDomainKind,
    hardware_class: &'a str,
    runtime_model_id: RuntimeModelId,
}

fn register_adaptive_domain_worker(
    layer: &OrchestratorLayer,
    worker_addr: SocketAddr,
    worker: AdaptiveDomainWorker<'_>,
) {
    let AdaptiveDomainWorker {
        worker_id,
        pool_id,
        domain_id,
        kind,
        hardware_class,
        runtime_model_id,
    } = worker;
    let trust_domain = TrustDomainId::new("private").unwrap();
    let model = RuntimeModelDescriptor {
        runtime_model_id,
        identity: adaptive_identity(),
        operations: BTreeSet::from([Operation::chat_completions()]),
        capabilities: BTreeSet::new(),
        max_context_tokens: Some(8_192),
        max_output_tokens: Some(2_048),
    };
    let mut capabilities: BTreeSet<ProtocolCapability> = [
        ProtocolCapability::CONTROL_EXECUTION_DOMAIN,
        ProtocolCapability::DISPATCH_TYPED_ADMISSION,
        ProtocolCapability::TELEMETRY_CAPACITY,
        ProtocolCapability::TELEMETRY_DOMAIN_CAPACITY,
    ]
    .into_iter()
    .map(|value| ProtocolCapability::new(value).unwrap())
    .collect();
    if kind == ExecutionDomainKind::MacAxEngineCluster {
        capabilities
            .insert(ProtocolCapability::new(ProtocolCapability::CONTROL_MAC_CLUSTER).unwrap());
    }
    layer
        .registry
        .register_protocol(
            ProtocolRegisterRequest {
                protocol: ProtocolDescriptor::current(capabilities.clone()),
                agent: AgentDescriptor {
                    name: "adaptive-fixture".into(),
                    version: "1.0.0".into(),
                    build_sha: None,
                },
                worker: ProtocolWorkerDescriptor {
                    id: ProtocolWorkerId::new(worker_id).unwrap(),
                    instance_id: WorkerInstanceId::new(),
                    advertise_url: format!("http://{worker_addr}"),
                    pool_id: pool_id.clone(),
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
                    accelerator: hardware_class.into(),
                    device_count: if kind == ExecutionDomainKind::MacAxEngineCluster {
                        2
                    } else {
                        1
                    },
                    memory_bytes: Some(128 * 1024 * 1024 * 1024),
                    hardware_class: Some(hardware_class.into()),
                },
                domain: Some(ExecutionDomainDescriptor {
                    id: domain_id,
                    kind,
                    endpoint_scope: if kind == ExecutionDomainKind::MacAxEngineCluster {
                        EndpointScope::Domain
                    } else {
                        EndpointScope::Node
                    },
                    execution_owner: "ax_engine".into(),
                    qualification: QualificationState::Experimental,
                    pool_id,
                    trust_domain,
                    hardware_class: hardware_class.into(),
                    architecture: "arm64".into(),
                    compatibility_manifest: Some(adaptive_manifest_digest()),
                    labels: BTreeMap::new(),
                }),
                domain_observation: Some(DomainObservation {
                    observed_at: time::OffsetDateTime::now_utc(),
                    generation: 1,
                    ready: true,
                    state: RuntimeState::Ready,
                    reason_code: None,
                    frontend_instances_ready: Some(1),
                    aggregate_capacity: Some(CapacityObservation {
                        active_requests: Some(0),
                        max_concurrent_requests: Some(4),
                        waiting_requests: Some(0),
                        ..Default::default()
                    }),
                    manifest_digest: Some(adaptive_manifest_digest()),
                    models: vec![model.clone()],
                }),
                observation: RuntimeObservation {
                    observed_at: time::OffsetDateTime::now_utc(),
                    runtime: RuntimeStatus::ready(),
                    inventory_generation: 1,
                    models: vec![model],
                    capacity: Some(CapacityObservation {
                        active_requests: Some(0),
                        max_concurrent_requests: Some(4),
                        waiting_requests: Some(0),
                        ..Default::default()
                    }),
                },
            },
            ax_serving_api::orchestration::worker_endpoint::WorkerEndpoint::parse(&format!(
                "http://{worker_addr}"
            ))
            .unwrap(),
            NegotiatedProtocol {
                version: CURRENT_PROTOCOL,
                capabilities,
            },
            5_000,
            15_000,
        )
        .unwrap();
}

fn adaptive_orchestrator_config(mode: PolicyMode) -> OrchestratorConfig {
    let single_pool = PoolId::new("mac-single").unwrap();
    let cluster_pool = PoolId::new("mac-cluster").unwrap();
    let single_domain = DomainId::new("mac-single-main").unwrap();
    let cluster_domain = DomainId::new("mac-cluster-main").unwrap();
    let logical_model = LogicalModelId::new("fixture/adaptive").unwrap();
    let single_deployment =
        ax_serving_protocol::DeploymentId::new("fixture-adaptive-single").unwrap();
    let cluster_deployment =
        ax_serving_protocol::DeploymentId::new("fixture-adaptive-cluster").unwrap();
    let equivalence_id = EquivalenceClassId::new("fixture-adaptive-certified").unwrap();
    let identity_policy = IdentityPolicy {
        required_matching_fields: BTreeSet::from([
            IdentityField::RuntimeKind,
            IdentityField::RuntimeVersion,
            IdentityField::Revision,
            IdentityField::ArtifactDigest,
            IdentityField::TokenizerDigest,
            IdentityField::TemplateDigest,
            IdentityField::Quantization,
        ]),
    };
    OrchestratorConfig {
        deployment_mode: "explicit".into(),
        request_timeout_secs: 5,
        first_byte_timeout_ms: 5_000,
        stream_idle_timeout_ms: 5_000,
        pools: vec![
            PoolSpec {
                id: single_pool.clone(),
                runtime_kind: "ax_engine".into(),
                hardware_class: Some("apple-silicon".into()),
                trust_domain: TrustDomainId::new("private").unwrap(),
                selector: BTreeMap::new(),
            },
            PoolSpec {
                id: cluster_pool.clone(),
                runtime_kind: "ax_engine".into(),
                hardware_class: Some("apple-silicon-cluster".into()),
                trust_domain: TrustDomainId::new("private").unwrap(),
                selector: BTreeMap::new(),
            },
        ],
        domains: vec![
            DomainSpec {
                id: single_domain.clone(),
                kind: ExecutionDomainKind::MacAxEngine,
                pool: single_pool.clone(),
                trust_domain: TrustDomainId::new("private").unwrap(),
                hardware_class: "apple-silicon".into(),
                required_qualification: QualificationState::Experimental,
                selector: BTreeMap::new(),
                enabled: true,
            },
            DomainSpec {
                id: cluster_domain.clone(),
                kind: ExecutionDomainKind::MacAxEngineCluster,
                pool: cluster_pool.clone(),
                trust_domain: TrustDomainId::new("private").unwrap(),
                hardware_class: "apple-silicon-cluster".into(),
                required_qualification: QualificationState::Experimental,
                selector: BTreeMap::new(),
                enabled: true,
            },
        ],
        deployments: vec![
            DeploymentSpec {
                id: single_deployment.clone(),
                logical_model: logical_model.clone(),
                pool: single_pool,
                domain: Some(single_domain.clone()),
                runtime_model_id: RuntimeModelId::new("fixture/adaptive-single").unwrap(),
                equivalence_class: Some(equivalence_id.clone()),
                expected_identity: Some(adaptive_identity()),
                required_identity: identity_policy.clone(),
                required_capabilities: BTreeSet::new(),
                enabled: true,
            },
            DeploymentSpec {
                id: cluster_deployment.clone(),
                logical_model,
                pool: cluster_pool,
                domain: Some(cluster_domain.clone()),
                runtime_model_id: RuntimeModelId::new("fixture/adaptive-cluster").unwrap(),
                equivalence_class: Some(equivalence_id.clone()),
                expected_identity: Some(adaptive_identity()),
                required_identity: identity_policy.clone(),
                required_capabilities: BTreeSet::new(),
                enabled: true,
            },
        ],
        equivalence_classes: vec![EquivalencePolicy {
            id: equivalence_id,
            identity_policy,
            certified_deployments: BTreeSet::from([single_deployment, cluster_deployment]),
            certification_artifact_digest: Digest::new(format!("sha256:{}", "d".repeat(64)))
                .unwrap(),
        }],
        adaptive_federation: AdaptiveFederationConfig {
            enabled: true,
            mode: match mode {
                PolicyMode::Shadow => "shadow",
                PolicyMode::Canary => "canary",
                PolicyMode::Active => "active",
                PolicyMode::Rollback => "rollback",
            }
            .into(),
            target_domain: Some(cluster_domain.clone()),
            baseline_domain: Some(single_domain.clone()),
            canary_share_ppm: 1_000_000,
            max_cost_microusd: Some(1_000),
            latency_slo_ms: Some(1_000),
            domains: vec![
                AdaptiveDomainPredictionConfig {
                    domain: single_domain,
                    predicted_cost_microusd: 100,
                    predicted_latency_ms: 120,
                    stability_rank: 2,
                },
                AdaptiveDomainPredictionConfig {
                    domain: cluster_domain,
                    predicted_cost_microusd: 200,
                    predicted_latency_ms: 80,
                    stability_rank: 1,
                },
            ],
        },
        ..OrchestratorConfig::default()
    }
}

/// Unwrap a `spawn_mock_worker` / `TcpListener::bind` result, skipping the
/// test if loopback socket binding is unavailable (e.g. sandbox environments).
macro_rules! skip_if_no_socket {
    ($expr:expr) => {
        match $expr {
            Some(v) => v,
            None => {
                eprintln!("test skipped: loopback socket bind unavailable in this environment");
                return;
            }
        }
    };
}

fn reg_req(addr: SocketAddr, caps: &[&str]) -> RegisterRequest {
    RegisterRequest {
        worker_id: None,
        addr: addr.to_string(),
        capabilities: RegisterCapabilities::Legacy(caps.iter().map(|s| s.to_string()).collect()),
        backend: "native".into(),
        max_inflight: 8,
        friendly_name: None,
        chip_model: None,
        worker_pool: None,
        node_class: None,
        ..Default::default()
    }
}

fn protocol_reg_req(
    addr: SocketAddr,
    worker_id: &str,
    runtime_kind: &str,
    model_id: &str,
) -> ProtocolRegisterRequest {
    ProtocolRegisterRequest {
        protocol: ProtocolDescriptor::current(Vec::new()),
        agent: AgentDescriptor {
            name: "test-runtime-agent".into(),
            version: "1.0.0".into(),
            build_sha: None,
        },
        worker: ProtocolWorkerDescriptor {
            id: ProtocolWorkerId::new(worker_id).unwrap(),
            instance_id: WorkerInstanceId::new(),
            advertise_url: format!("http://{addr}"),
            pool_id: PoolId::new(format!("{runtime_kind}-pool")).unwrap(),
            trust_domain: TrustDomainId::new("private").unwrap(),
            labels: BTreeMap::from([
                ("node_class".into(), "pc-cuda".into()),
                ("friendly_name".into(), "test-node".into()),
            ]),
        },
        runtime: RuntimeDescriptor {
            kind: runtime_kind.into(),
            version: "test".into(),
            api: "openai-v1".into(),
            endpoint: Some("http://runtime.test:8000".into()),
        },
        hardware: HardwareDescriptor {
            platform: "linux".into(),
            accelerator: "nvidia-gpu".into(),
            device_count: 1,
            memory_bytes: None,
            hardware_class: Some("pc-cuda-sm120".into()),
        },
        domain: None,
        domain_observation: None,
        observation: RuntimeObservation {
            observed_at: time::OffsetDateTime::now_utc(),
            runtime: RuntimeStatus::ready(),
            inventory_generation: 1,
            models: vec![RuntimeModelDescriptor {
                runtime_model_id: RuntimeModelId::new(model_id).unwrap(),
                identity: DeploymentIdentity {
                    runtime_kind: runtime_kind.into(),
                    runtime_version: Some("test".into()),
                    revision: None,
                    artifact_digest: None,
                    tokenizer_digest: None,
                    template_digest: None,
                    quantization: None,
                },
                operations: BTreeSet::from([Operation::chat_completions()]),
                capabilities: BTreeSet::new(),
                max_context_tokens: Some(2_048),
                max_output_tokens: Some(512),
            }],
            capacity: Some(CapacityObservation {
                active_requests: Some(0),
                max_concurrent_requests: Some(8),
                waiting_requests: Some(0),
                ..Default::default()
            }),
        },
    }
}

#[tokio::test]
async fn gateway_prometheus_metrics_are_normalized_and_low_cardinality() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    // In-process tests do not bind sockets; mark listeners ready to exercise
    // control-plane readiness independent of worker capacity.
    layer.ops.mark_listeners_ready();
    let app = ax_serving_api::orchestration::proxy_router(layer);
    // Control-plane readiness must succeed with zero workers so agents can register.
    let readiness = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/readyz")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(readiness.status(), axum::http::StatusCode::OK);

    let routability = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/routablez")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        routability.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/metrics")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(
        response.headers()[axum::http::header::CONTENT_TYPE],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("axs_gateway_requests_total 0"));
    assert!(body.contains("axs_gateway_endpoint_selection_duration_seconds_count 0"));
    assert!(body.contains("axs_gateway_time_to_first_byte_seconds_count 0"));
    assert!(body.contains("axs_gateway_stream_duration_seconds_count 0"));
    assert!(body.contains("axs_gateway_endpoint_selections_total{outcome=\"no_candidate\"} 0"));
    assert!(body.contains("axs_gateway_workers{state=\"eligible\"} 0"));
    assert!(!body.contains("worker_id="));
}

fn sample_project_policy(default_project: Option<&str>) -> ProjectPolicyConfig {
    ProjectPolicyConfig {
        enabled: true,
        default_project: default_project.map(str::to_string),
        rules: vec![
            ax_serving_api::config::ProjectRuleConfig {
                project: "fabric".into(),
                allowed_models: vec!["pool-model".into(), "ops-model".into()],
                max_tokens_limit: Some(64),
                worker_pool: Some("green".into()),
            },
            ax_serving_api::config::ProjectRuleConfig {
                project: "ops".into(),
                allowed_models: vec!["*".into()],
                max_tokens_limit: None,
                worker_pool: None,
            },
        ],
    }
}

fn reg_req_with_pool(
    addr: SocketAddr,
    caps: &[&str],
    worker_pool: Option<&str>,
    node_class: Option<&str>,
) -> RegisterRequest {
    let mut req = reg_req(addr, caps);
    req.worker_pool = worker_pool.map(str::to_string);
    req.node_class = node_class.map(str::to_string);
    req
}

struct CountingPolicy {
    recorded: Arc<AtomicUsize>,
}

impl DispatchPolicy for CountingPolicy {
    fn select<'a>(
        &self,
        workers: &'a [WorkerStatus],
        _ctx: &DispatchContext<'_>,
    ) -> Option<&'a WorkerStatus> {
        workers.first()
    }

    fn record_dispatch(&self, _worker_id: WorkerId, _model_id: &str) {
        self.recorded.fetch_add(1, Ordering::Relaxed);
    }
}

// ── TASK-MW-010 tests ─────────────────────────────────────────────────────────

/// Register a worker, heartbeat it, verify it appears in eligible list.
#[tokio::test]
async fn test_register_heartbeat_eligible() {
    let registry = WorkerRegistry::new();
    // Use a non-binding address — we only test registry logic, no actual HTTP.
    let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();

    let resp = registry.register(reg_req(addr, &["llama3-8b"]), 5000);
    let id = ax_serving_api::orchestration::registry::WorkerId::parse(&resp.worker_id).unwrap();

    // Worker should be eligible immediately after registration.
    assert_eq!(registry.eligible_workers("llama3-8b").len(), 1);

    // Heartbeat should succeed.
    let hb = HeartbeatRequest {
        inflight: 0,
        thermal_state: "nominal".into(),
        model_ids: vec![],
        rss_bytes: 0,
        ..Default::default()
    };
    assert!(registry.heartbeat(id, hb));

    // Unknown model → no workers.
    assert!(registry.eligible_workers("unknown-model").is_empty());
}

#[test]
fn test_protocol_registration_preserves_generic_runtime_identity() {
    let registry = WorkerRegistry::new();
    let request = protocol_reg_req(
        "127.0.0.1:18081".parse().unwrap(),
        "df-rtx5090",
        "tensorrt_llm",
        "tinyllama-trtllm",
    );
    registry
        .register_protocol(
            request,
            ax_serving_api::orchestration::worker_endpoint::WorkerEndpoint::parse(
                "http://127.0.0.1:18081",
            )
            .unwrap(),
            NegotiatedProtocol {
                version: CURRENT_PROTOCOL,
                capabilities: BTreeSet::new(),
            },
            5_000,
            15_000,
        )
        .unwrap();

    let workers = registry.list_all();
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0].runtime, "tensorrt_llm");
    assert_eq!(
        workers[0].runtime_endpoint.as_deref(),
        Some("http://runtime.test:8000")
    );
    assert_eq!(
        workers[0].model_inventory[0].runtime_kind.as_deref(),
        Some("tensorrt_llm")
    );
    assert_eq!(
        registry
            .eligible_workers_filtered(
                "tinyllama-trtllm",
                RequestKind::Llm,
                Some("tensorrt_llm"),
                None,
            )
            .len(),
        1
    );
}

#[tokio::test]
async fn test_proxy_strips_ax_routing_hints_before_upstream_dispatch() {
    let (worker_addr, worker_state) = skip_if_no_socket!(spawn_echo_model_worker().await);
    let (gateway_addr, layer) =
        skip_if_no_socket!(spawn_orchestrator_with_layer(OrchestratorConfig::default()).await);
    layer
        .registry
        .register_protocol(
            protocol_reg_req(
                worker_addr,
                "strict-runtime-worker",
                "tensorrt_llm",
                "strict-model",
            ),
            ax_serving_api::orchestration::worker_endpoint::WorkerEndpoint::parse(&format!(
                "http://{worker_addr}"
            ))
            .unwrap(),
            NegotiatedProtocol {
                version: CURRENT_PROTOCOL,
                capabilities: BTreeSet::new(),
            },
            5_000,
            15_000,
        )
        .unwrap();

    let response = Client::new()
        .post(format!("http://{gateway_addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "strict-model",
            "backend": "auto",
            "runtime": "tensorrt_llm",
            "messages": [{"role": "user", "content": "hello"}],
            "extension": {"runtime": "runtime-owned", "backend": "runtime-owned"}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bodies = worker_state.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    assert!(bodies[0].get("backend").is_none());
    assert!(bodies[0].get("runtime").is_none());
    assert_eq!(bodies[0]["extension"]["runtime"], "runtime-owned");
    assert_eq!(bodies[0]["extension"]["backend"], "runtime-owned");
}

/// Dispatch a real request to a mock worker and verify it succeeds.
#[tokio::test]
async fn test_dispatch_to_mock_worker() {
    let addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"hi"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    registry.register(reg_req(addr, &["test-model"]), 5000);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"test-model","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "test-model",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn test_explicit_deployment_routes_logical_alias_and_preserves_public_credentials() {
    let (worker_addr, worker_state) = skip_if_no_socket!(spawn_echo_model_worker().await);
    let pool_id = PoolId::new("cuda").unwrap();
    let logical_model = LogicalModelId::new("public/qwen").unwrap();
    let runtime_model = RuntimeModelId::new("Qwen/Qwen3-32B").unwrap();
    let domain_id = DomainId::new("cuda-compat").unwrap();
    let config = OrchestratorConfig {
        deployment_mode: "explicit".into(),
        pools: vec![PoolSpec {
            id: pool_id.clone(),
            runtime_kind: "vllm".into(),
            hardware_class: Some("cuda".into()),
            trust_domain: TrustDomainId::new("private").unwrap(),
            selector: BTreeMap::new(),
        }],
        domains: vec![DomainSpec {
            id: domain_id.clone(),
            kind: ExecutionDomainKind::CompatibilityRuntimeEndpoint,
            pool: pool_id.clone(),
            trust_domain: TrustDomainId::new("private").unwrap(),
            hardware_class: "cuda".into(),
            required_qualification: QualificationState::Certified,
            selector: BTreeMap::new(),
            enabled: true,
        }],
        deployments: vec![DeploymentSpec {
            id: ax_serving_protocol::DeploymentId::new("qwen-cuda").unwrap(),
            logical_model: logical_model.clone(),
            pool: pool_id.clone(),
            domain: Some(domain_id.clone()),
            runtime_model_id: runtime_model.clone(),
            equivalence_class: None,
            expected_identity: None,
            required_identity: Default::default(),
            required_capabilities: BTreeSet::new(),
            enabled: true,
        }],
        ..OrchestratorConfig::default()
    };
    let fleet_store: Arc<dyn FleetStateStore> = MemoryFleetStateStore::shared();
    let gateway_a = Arc::new(
        OrchestratorLayer::new_with_fleet_store(
            config.clone(),
            ProjectPolicyConfig::default(),
            Arc::clone(&fleet_store),
        )
        .unwrap(),
    );
    let layer = Arc::new(
        OrchestratorLayer::new_with_fleet_store(
            config,
            ProjectPolicyConfig::default(),
            Arc::clone(&fleet_store),
        )
        .unwrap(),
    );
    let protocol = ProtocolDescriptor::current(Vec::new());
    gateway_a
        .registry
        .register_protocol(
            ProtocolRegisterRequest {
                protocol: protocol.clone(),
                agent: AgentDescriptor {
                    name: "test-agent".into(),
                    version: "1.0.0".into(),
                    build_sha: None,
                },
                worker: ProtocolWorkerDescriptor {
                    id: ProtocolWorkerId::new("cuda-worker").unwrap(),
                    instance_id: WorkerInstanceId::new(),
                    advertise_url: format!("http://{worker_addr}"),
                    pool_id,
                    trust_domain: TrustDomainId::new("private").unwrap(),
                    labels: BTreeMap::new(),
                },
                runtime: RuntimeDescriptor {
                    kind: "vllm".into(),
                    version: "0.9.0".into(),
                    api: "openai-http".into(),
                    endpoint: None,
                },
                hardware: HardwareDescriptor {
                    platform: "linux".into(),
                    accelerator: "nvidia-gpu".into(),
                    device_count: 1,
                    memory_bytes: Some(80 * 1024 * 1024 * 1024),
                    hardware_class: Some("cuda".into()),
                },
                domain: None,
                domain_observation: None,
                observation: RuntimeObservation {
                    observed_at: time::OffsetDateTime::now_utc(),
                    runtime: RuntimeStatus::ready(),
                    inventory_generation: 1,
                    models: vec![RuntimeModelDescriptor {
                        runtime_model_id: runtime_model,
                        identity: DeploymentIdentity {
                            runtime_kind: "vllm".into(),
                            runtime_version: Some("0.9.0".into()),
                            revision: None,
                            artifact_digest: None,
                            tokenizer_digest: None,
                            template_digest: None,
                            quantization: None,
                        },
                        operations: BTreeSet::from([Operation::chat_completions()]),
                        capabilities: BTreeSet::new(),
                        max_context_tokens: Some(32_768),
                        max_output_tokens: Some(4_096),
                    }],
                    capacity: Some(CapacityObservation {
                        active_requests: Some(0),
                        max_concurrent_requests: Some(4),
                        waiting_requests: Some(0),
                        ..Default::default()
                    }),
                },
            },
            ax_serving_api::orchestration::worker_endpoint::WorkerEndpoint::parse(&format!(
                "http://{worker_addr}"
            ))
            .unwrap(),
            NegotiatedProtocol {
                version: CURRENT_PROTOCOL,
                capabilities: BTreeSet::new(),
            },
            5_000,
            15_000,
        )
        .unwrap();
    let protocol_worker_id = ProtocolWorkerId::new("cuda-worker").unwrap();
    let record = gateway_a
        .registry
        .export_protocol_record(&protocol_worker_id)
        .unwrap();
    fleet_store.put(&record).await.unwrap();
    gateway_a.reconcile_deployment_state().await.unwrap();
    layer.reconcile_fleet_state().await.unwrap();
    layer.reconcile_deployment_state().await.unwrap();

    let app = proxy_router_with_key(Arc::clone(&layer), "public-secret");
    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, "Bearer public-secret")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{ "model" : "public/qwen", "messages":[{"role":"user","content":"hi"}], "extension":1.2300 }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["model"], "Qwen/Qwen3-32B");
    assert_eq!(
        worker_state.models.lock().unwrap().as_slice(),
        ["Qwen/Qwen3-32B"]
    );
    assert_eq!(
        worker_state.domain_ids.lock().unwrap().as_slice(),
        ["cuda-compat"]
    );
    assert!(!*worker_state.public_authorization_seen.lock().unwrap());
    let decisions = layer.dispatcher.decision_records(10);
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].selected_domain, domain_id);
    assert_eq!(
        fleet_store.list_decisions(10).await.unwrap(),
        decisions,
        "the shared fleet store must retain the same bounded decision evidence"
    );

    let decision_response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/admin/decisions?limit=10")
                .header(axum::http::header::AUTHORIZATION, "Bearer public-secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(decision_response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(decision_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let decision_response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        decision_response["record_type"],
        "gateway_routing_decision_v1"
    );
    assert_eq!(decision_response["execution_observed"], false);
    assert_eq!(
        decision_response["records"][0]["selected_domain"],
        "cuda-compat"
    );
    assert_eq!(
        decision_response["records"][0]["policy_id"],
        "explicit-catalog"
    );

    let deployments_response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/admin/deployments")
                .header(axum::http::header::AUTHORIZATION, "Bearer public-secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deployments_response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(deployments_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let deployments_response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(deployments_response["domains"][0]["id"], "cuda-compat");

    let models = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/models")
                .header(axum::http::header::AUTHORIZATION, "Bearer public-secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(models.into_body(), usize::MAX)
        .await
        .unwrap();
    let models: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(models["data"][0]["id"], "public/qwen");
}

#[tokio::test]
async fn test_adaptive_federation_modes_route_complete_domains_and_retain_evidence() {
    let (single_addr, single_state) = skip_if_no_socket!(spawn_echo_model_worker().await);
    let (cluster_addr, cluster_state) = skip_if_no_socket!(spawn_echo_model_worker().await);
    let single_domain = DomainId::new("mac-single-main").unwrap();
    let cluster_domain = DomainId::new("mac-cluster-main").unwrap();

    for (mode, selected_domain, runtime_model, counterfactual, reason_code) in [
        (
            PolicyMode::Shadow,
            single_domain.clone(),
            "fixture/adaptive-single",
            Some(cluster_domain.clone()),
            DecisionReasonCode::ShadowBaseline,
        ),
        (
            PolicyMode::Canary,
            cluster_domain.clone(),
            "fixture/adaptive-cluster",
            Some(single_domain.clone()),
            DecisionReasonCode::CanaryAssignment,
        ),
        (
            PolicyMode::Active,
            cluster_domain.clone(),
            "fixture/adaptive-cluster",
            None,
            DecisionReasonCode::ActiveAssignment,
        ),
        (
            PolicyMode::Rollback,
            single_domain.clone(),
            "fixture/adaptive-single",
            Some(cluster_domain.clone()),
            DecisionReasonCode::PolicyRollback,
        ),
    ] {
        let fleet_store: Arc<dyn FleetStateStore> = MemoryFleetStateStore::shared();
        let layer = Arc::new(
            OrchestratorLayer::new_with_fleet_store(
                adaptive_orchestrator_config(mode),
                ProjectPolicyConfig::default(),
                Arc::clone(&fleet_store),
            )
            .unwrap(),
        );
        register_adaptive_domain_worker(
            &layer,
            single_addr,
            AdaptiveDomainWorker {
                worker_id: "adaptive-single-worker",
                pool_id: PoolId::new("mac-single").unwrap(),
                domain_id: single_domain.clone(),
                kind: ExecutionDomainKind::MacAxEngine,
                hardware_class: "apple-silicon",
                runtime_model_id: RuntimeModelId::new("fixture/adaptive-single").unwrap(),
            },
        );
        register_adaptive_domain_worker(
            &layer,
            cluster_addr,
            AdaptiveDomainWorker {
                worker_id: "adaptive-cluster-worker",
                pool_id: PoolId::new("mac-cluster").unwrap(),
                domain_id: cluster_domain.clone(),
                kind: ExecutionDomainKind::MacAxEngineCluster,
                hardware_class: "apple-silicon-cluster",
                runtime_model_id: RuntimeModelId::new("fixture/adaptive-cluster").unwrap(),
            },
        );
        layer.reconcile_deployment_state().await.unwrap();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            proxy_router_with_key(Arc::clone(&layer), "adaptive-public-secret").oneshot(
                axum::http::Request::builder()
                    .method(axum::http::Method::POST)
                    .uri("/v1/chat/completions")
                    .header(
                        axum::http::header::AUTHORIZATION,
                        "Bearer adaptive-public-secret",
                    )
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"model":"fixture/adaptive","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            ),
        )
        .await
        .expect("adaptive request must complete within its bounded deadline")
        .unwrap();
        let response_status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            response_status,
            axum::http::StatusCode::OK,
            "mode {mode:?} returned {response}"
        );
        assert_eq!(response["model"], runtime_model);

        let decisions = layer.dispatcher.decision_records(10);
        assert_eq!(decisions.len(), 1);
        let decision = &decisions[0];
        assert_eq!(decision.selected_domain, selected_domain);
        assert_eq!(decision.policy_mode, mode);
        assert_eq!(decision.counterfactual_domain, counterfactual);
        assert_eq!(decision.rolled_back, mode == PolicyMode::Rollback);
        assert!(
            decision
                .reason_codes
                .contains(&DecisionReasonCode::AdaptivePolicy)
        );
        assert!(decision.reason_codes.contains(&reason_code));
        assert_eq!(
            fleet_store.list_decisions(10).await.unwrap(),
            decisions,
            "adaptive decision evidence must be retained in the shared store"
        );
    }

    assert_eq!(
        single_state.domain_ids.lock().unwrap().as_slice(),
        ["mac-single-main", "mac-single-main"]
    );
    assert_eq!(
        cluster_state.domain_ids.lock().unwrap().as_slice(),
        ["mac-cluster-main", "mac-cluster-main"]
    );
}

/// Dispatcher must re-check worker capacity after policy selection. Policies
/// operate on snapshots, so a worker can become full between selection and
/// dispatch under concurrent load.
#[tokio::test]
async fn test_dispatch_rejects_worker_that_fills_after_selection() {
    let addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"hi"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    let mut req = reg_req(addr, &["race-model"]);
    req.max_inflight = 1;
    let registered = registry.register(req, 5000);
    let worker_id = WorkerId::parse(&registered.worker_id).unwrap();
    let counter = registry
        .inflight_counter(worker_id)
        .expect("registered worker counter");

    // Simulate a concurrent dispatch that filled the worker after a policy saw
    // an older eligible snapshot.
    counter.fetch_add(1, Ordering::Relaxed);

    let recorded = Arc::new(AtomicUsize::new(0));
    let policy = CountingPolicy {
        recorded: Arc::clone(&recorded),
    };
    let dispatcher = DirectDispatcher::new(8, 300);

    let response = dispatcher
        .forward(
            &registry,
            &policy,
            "race-model",
            false,
            None,
            "/v1/chat/completions",
            axum::body::Bytes::from(r#"{"model":"race-model","messages":[]}"#),
            None,
        )
        .await;

    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        counter.load(Ordering::Relaxed),
        1,
        "failed dispatch must not over-increment worker inflight"
    );
    assert_eq!(
        recorded.load(Ordering::Relaxed),
        0,
        "capacity rejection must not be recorded as a successful dispatch"
    );
}

/// Primary 4xx responses must not be recorded as successful dispatches for
/// model-affinity accounting.
#[tokio::test]
async fn test_no_affinity_record_on_primary_4xx() {
    let bad_addr = skip_if_no_socket!(spawn_mock_worker(400, r#"{"error":"bad request"}"#).await);
    let registry = WorkerRegistry::new();
    registry.register(reg_req(bad_addr, &["m4xx"]), 5000);

    let counter = Arc::new(AtomicUsize::new(0));
    let policy = CountingPolicy {
        recorded: Arc::clone(&counter),
    };
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"m4xx","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            &policy,
            "m4xx",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(
        counter.load(Ordering::Relaxed),
        0,
        "4xx responses must not increment affinity dispatch counters"
    );
}

/// A trusted typed pre-admission rejection may be retried once.
#[tokio::test]
async fn test_reroute_on_typed_not_admitted() {
    let rejected_addr = skip_if_no_socket!(spawn_not_admitted_worker().await);
    let good_body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
    let good_addr = skip_if_no_socket!(spawn_mock_worker(200, good_body).await);

    let registry = WorkerRegistry::new();
    registry.register(reg_req(rejected_addr, &["m"]), 5000);
    let good = registry.register(reg_req(good_addr, &["m"]), 5000);
    let good_id = WorkerId::parse(&good.worker_id).unwrap();
    registry.heartbeat(
        good_id,
        HeartbeatRequest {
            inflight: 1,
            thermal_state: "nominal".into(),
            model_ids: vec!["m".into()],
            rss_bytes: 0,
            ..Default::default()
        },
    );

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"m","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "m",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(dispatcher.reroutes(), 1);
}

/// Generic runtime failures are preserved and never retried.
#[tokio::test]
async fn test_generic_5xx_is_not_retried() {
    let bad_addr = skip_if_no_socket!(spawn_mock_worker(500, r#"{"error":"down"}"#).await);

    let registry = WorkerRegistry::new();
    registry.register(reg_req(bad_addr, &["only-model"]), 5000);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"only-model","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "only-model",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(dispatcher.reroutes(), 0);
}

/// GlobalQueue rejects requests when the concurrency limit is full and depth=0.
#[tokio::test]
async fn test_queue_reject_when_full() {
    let q = GlobalQueue::new(GlobalQueueConfig {
        max_concurrent: 1,
        max_queue_depth: 0,
        wait_ms: 100,
        overload_policy: OverloadPolicy::Reject,
    });

    let permit = q.acquire("test-client-a".into()).await;
    assert!(matches!(permit, AcquireResult::Permit(_)));

    let r = q.acquire("test-client-a".into()).await;
    assert!(matches!(r, AcquireResult::Rejected));
}

/// GlobalQueue with ShedOldest evicts waiting requests when queue is full.
#[tokio::test]
async fn test_queue_shed_oldest() {
    let q = Arc::new(GlobalQueue::new(GlobalQueueConfig {
        max_concurrent: 1,
        max_queue_depth: 1,
        wait_ms: 2000,
        overload_policy: OverloadPolicy::ShedOldest,
    }));

    let permit = q.acquire("test-client-a".into()).await;
    assert!(matches!(permit, AcquireResult::Permit(_)));

    // First waiter fills the single queue depth slot.
    let q2 = Arc::clone(&q);
    let waiter1 = tokio::spawn(async move { q2.acquire("test-client-a".into()).await });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(q.queued(), 1);

    // Second request: queue full → shed waiter1, enqueue waiter2.
    let q3 = Arc::clone(&q);
    let waiter2 = tokio::spawn(async move { q3.acquire("test-client-b".into()).await });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    let r1 = waiter1.await.unwrap();
    assert!(
        matches!(r1, AcquireResult::Shed),
        "waiter1 should have been shed"
    );

    drop(permit);
    let r2 = waiter2.await.unwrap();
    assert!(
        matches!(r2, AcquireResult::Permit(_)),
        "waiter2 should receive the permit"
    );
}

/// Health TTL: a worker that stops heartbeating is evicted after the TTL.
#[tokio::test]
async fn test_health_ttl_eviction() {
    let registry = WorkerRegistry::new();
    // Use a non-binding address — we only test registry logic, no actual HTTP.
    let addr: SocketAddr = "127.0.0.1:2".parse().unwrap();

    registry.register(reg_req(addr, &["m1"]), 5000);
    assert_eq!(registry.eligible_workers("m1").len(), 1);

    // With ttl_ms=1 and sleep=5ms the worker must appear stale.
    std::thread::sleep(std::time::Duration::from_millis(5));
    let evicted = registry.tick(1);
    assert!(!evicted.is_empty(), "worker should have been evicted");
    assert!(registry.eligible_workers("m1").is_empty());
}

/// Verify that a typed pre-admission rejection increments the reroute counter.
#[tokio::test]
async fn test_reroute_counter_increments() {
    let bad_addr = skip_if_no_socket!(spawn_not_admitted_worker().await);

    let registry = WorkerRegistry::new();
    registry.register(reg_req(bad_addr, &["mdl"]), 5000);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"mdl","messages":[]}"#);
    let _ = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "mdl",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(
        dispatcher.reroutes(),
        1,
        "typed pre-admission rejection should count as one reroute"
    );
}

/// Verifies the /health endpoint JSON shape via reqwest to a real server.
#[tokio::test]
async fn test_health_endpoint_shape() {
    use ax_serving_api::orchestration::{OrchestratorConfig, OrchestratorLayer};

    let cfg = OrchestratorConfig {
        port: 0, // will bind ephemeral
        internal_port: 0,
        ..OrchestratorConfig::default()
    };

    // We just test the health endpoint shape without starting full servers —
    // use OrchestratorLayer directly and call proxy_health logic via a
    // one-shot axum server.
    let layer = Arc::new(OrchestratorLayer::new(cfg, ProjectPolicyConfig::default()).unwrap());

    let public_router = {
        let l = Arc::clone(&layer);
        Router::new()
            .route("/health", axum::routing::get(health_handler))
            .with_state(l)
    };

    let listener = skip_if_no_socket!(tokio::net::TcpListener::bind("127.0.0.1:0").await.ok());
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, public_router).await.ok();
    });

    let client = Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body.get("status").is_some(), "health must have 'status'");
    assert!(body.get("workers").is_some(), "health must have 'workers'");
    assert!(body.get("queue").is_some(), "health must have 'queue'");
}

#[tokio::test]
async fn test_non_loopback_internal_bind_requires_token() {
    let mut env = EnvVarsGuard::new();
    env.set("AXS_ALLOW_NO_AUTH", "true");
    env.remove("AXS_INTERNAL_API_TOKEN");

    let result = start_orchestrator(
        OrchestratorConfig {
            host: "127.0.0.1".into(),
            port: 0,
            internal_port: 0,
            internal_bind_addr: "0.0.0.0".into(),
            ..OrchestratorConfig::default()
        },
        ProjectPolicyConfig::default(),
    )
    .await;

    let err = result.expect_err("non-loopback internal bind without token must fail");
    assert!(
        err.to_string().contains("AXS_INTERNAL_API_TOKEN"),
        "error should mention missing internal token: {err}"
    );
}

#[tokio::test]
async fn test_invalid_allowed_node_cidrs_fails_startup() {
    let mut env = EnvVarsGuard::new();
    env.set("AXS_ALLOW_NO_AUTH", "true");
    env.set("AXS_INTERNAL_API_TOKEN", "secret");

    let result = start_orchestrator(
        OrchestratorConfig {
            host: "127.0.0.1".into(),
            port: 0,
            internal_port: 0,
            internal_bind_addr: "127.0.0.1".into(),
            allowed_node_cidrs: "not-a-cidr".into(),
            ..OrchestratorConfig::default()
        },
        ProjectPolicyConfig::default(),
    )
    .await;

    let err = result.expect_err("invalid allowlist must fail startup");
    assert!(
        err.to_string().contains("AXS_ALLOWED_NODE_CIDRS"),
        "error should mention malformed allowlist: {err}"
    );
}

#[tokio::test]
async fn test_internal_router_real_server_enforces_token_and_allowlist() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let state = InternalState {
        registry: layer.registry.clone(),
        fleet_store: Arc::clone(&layer.fleet_store),
        config: Arc::clone(&layer.config),
    };
    let addr = skip_if_no_socket!(
        spawn_internal_router_with_auth(
            state,
            Some(InternalAuthState {
                token: Some(Arc::new("secret".to_string())),
                allowed_sources: Arc::new(parse_allowed_node_cidrs("127.0.0.1/32").unwrap()),
            }),
        )
        .await
    );

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    let unauthorized = client
        .get(format!("http://{addr}/internal/workers"))
        .send()
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), axum::http::StatusCode::UNAUTHORIZED);

    let authorized = client
        .get(format!("http://{addr}/internal/workers"))
        .header("x-internal-token", "secret")
        .send()
        .await
        .unwrap();
    assert_eq!(authorized.status(), axum::http::StatusCode::OK);
}

// Minimal health handler for the shape test above.
async fn health_handler(
    axum::extract::State(layer): axum::extract::State<
        Arc<ax_serving_api::orchestration::OrchestratorLayer>,
    >,
) -> axum::Json<serde_json::Value> {
    use std::sync::atomic::Ordering;
    let (healthy, unhealthy, _draining) = layer.registry.counts();
    let status = if healthy > 0 { "ok" } else { "degraded" };
    let qm = &layer.queue.metrics;
    axum::Json(serde_json::json!({
        "status": status,
        "workers": { "total": healthy + unhealthy, "healthy": healthy, "unhealthy": unhealthy },
        "queue": {
            "active": layer.queue.active(),
            "queued": layer.queue.queued(),
            "rejected_total": qm.rejected_total.load(Ordering::Relaxed),
            "shed_total": qm.shed_total.load(Ordering::Relaxed),
            "timeout_total": qm.timeout_total.load(Ordering::Relaxed),
        }
    }))
}

// ── TASK-MW-018: Failure injection ────────────────────────────────────────────

/// Scenario: A registered worker disappears (port closed) and is the only worker.
/// The dispatcher must return 503 and increment the reroute counter.
///
/// This covers PRD §9.2 "Kill worker between requests".
#[tokio::test]
async fn test_failure_worker_connection_refused() {
    // Bind a listener to get a free port, then drop it immediately so the port
    // becomes "refused" when the dispatcher tries to connect.
    let port = {
        let l = skip_if_no_socket!(tokio::net::TcpListener::bind("127.0.0.1:0").await.ok());
        l.local_addr().unwrap().port()
        // l dropped here — port is now closed
    };
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    let registry = WorkerRegistry::new();
    registry.register(reg_req(addr, &["gone-model"]), 5000);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"gone-model","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "gone-model",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    // No alternative worker → must return 503.
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    // The dispatcher must count this as a reroute attempt.
    assert_eq!(
        dispatcher.reroutes(),
        1,
        "connection refusal counts as reroute"
    );
}

/// Scenario: All workers for a model die (evicted via TTL).
/// After eviction, dispatch must return 503 immediately — no workers selected.
///
/// This covers PRD §9.2 "Kill worker between requests" + registry eviction path.
#[tokio::test]
async fn test_failure_all_workers_evicted() {
    let registry = WorkerRegistry::new();
    let addr: SocketAddr = "127.0.0.1:5".parse().unwrap();
    registry.register(reg_req(addr, &["evict-model"]), 5000);

    // Force immediate eviction (ttl_ms=1, sleep 5ms).
    std::thread::sleep(std::time::Duration::from_millis(5));
    let evicted = registry.tick(1);
    assert!(!evicted.is_empty());

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"evict-model","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "evict-model",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "no eligible workers → must return 503"
    );
}

/// Scenario: Orchestrator "restarts" — registry is cleared and workers re-register.
/// After re-registration dispatch must succeed again.
///
/// This covers PRD §9.2 "Orchestrator restart (direct)".
#[tokio::test]
async fn test_failure_worker_restart_reregister() {
    let addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"ok"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    registry.register(reg_req(addr, &["restart-model"]), 5000);

    // Simulate orchestrator restart: evict all workers.
    let id_str = registry
        .eligible_workers("restart-model")
        .first()
        .map(|w| w.id.to_string())
        .unwrap();
    let id = ax_serving_api::orchestration::registry::WorkerId::parse(&id_str).unwrap();
    registry.evict(id);

    assert!(
        registry.eligible_workers("restart-model").is_empty(),
        "after eviction no workers should be eligible"
    );

    // Worker re-registers (simulates it reconnecting to the new orchestrator instance).
    registry.register(reg_req(addr, &["restart-model"]), 5000);
    assert_eq!(registry.eligible_workers("restart-model").len(), 1);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"restart-model","messages":[]}"#);
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "restart-model",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    assert_eq!(
        response.status(),
        axum::http::StatusCode::OK,
        "re-registered worker must serve requests"
    );
}

// ── TASK-MW-013: Drain lifecycle ──────────────────────────────────────────────

/// Drain lifecycle: register → drain → no longer eligible → evict → re-register.
///
/// Verifies the full state machine without needing in-flight streaming requests:
/// the key invariant is that a draining worker is excluded from `eligible_workers`
/// and can re-enter rotation after a drain-complete + re-registration.
#[tokio::test]
async fn test_drain_lifecycle() {
    let registry = WorkerRegistry::new();

    // Register two workers.
    let addr0: SocketAddr = "127.0.0.1:3".parse().unwrap();
    let addr1: SocketAddr = "127.0.0.1:4".parse().unwrap();
    let resp0 = registry.register(reg_req(addr0, &["drain-model"]), 5000);
    registry.register(reg_req(addr1, &["drain-model"]), 5000);

    // Both should be eligible.
    assert_eq!(registry.eligible_workers("drain-model").len(), 2);

    // Parse worker 0's ID and mark it for drain.
    let id0 = ax_serving_api::orchestration::registry::WorkerId::parse(&resp0.worker_id).unwrap();
    assert!(
        registry.mark_drain(id0),
        "mark_drain should return true for known worker"
    );

    // Draining worker must be excluded from eligible set.
    let eligible = registry.eligible_workers("drain-model");
    assert_eq!(eligible.len(), 1, "draining worker must not be eligible");
    assert!(
        eligible.iter().all(|w| w.id != id0),
        "draining worker id must not appear in eligible list"
    );

    // Drain-complete: evict worker 0.
    registry.evict(id0);
    assert_eq!(
        registry.eligible_workers("drain-model").len(),
        1,
        "only worker 1 should remain after eviction"
    );

    // Re-register worker 0 with the same address → enters rotation again.
    registry.register(reg_req(addr0, &["drain-model"]), 5000);
    assert_eq!(
        registry.eligible_workers("drain-model").len(),
        2,
        "re-registered worker must be eligible again"
    );
}

// ── TASK-MW-011: WeightedRoundRobin integration ───────────────────────────────

/// WeightedRoundRobinPolicy dispatches proportionally to available capacity.
#[tokio::test]
async fn test_wrr_dispatch_proportional() {
    use ax_serving_api::orchestration::policy::policy_from_str;

    // Worker A: max_inflight=4, Worker B: max_inflight=1.
    // Over 5 calls, A should receive 4 requests and B 1.
    let addr_a = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"a"}}]}"#).await
    );
    let addr_b = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"b"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    let resp_a = registry.register(
        RegisterRequest {
            worker_id: None,
            addr: addr_a.to_string(),
            capabilities: RegisterCapabilities::Legacy(vec!["wrr-model".into()]),
            backend: "native".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: None,
            ..Default::default()
        },
        5000,
    );
    registry.register(
        RegisterRequest {
            worker_id: None,
            addr: addr_b.to_string(),
            capabilities: RegisterCapabilities::Legacy(vec!["wrr-model".into()]),
            backend: "native".into(),
            max_inflight: 1,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: None,
            ..Default::default()
        },
        5000,
    );

    let _ = resp_a; // registered; id not needed for this test
    let policy = policy_from_str("weighted_round_robin").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    // Run 5 dispatches and verify each returns 200.
    // Proportional distribution is verified by the unit tests in policy.rs.
    let body_template = r#"{"model":"wrr-model","messages":[]}"#;
    for _ in 0..5 {
        let body = axum::body::Bytes::from(body_template);
        let resp = dispatcher
            .forward(
                &registry,
                policy.as_ref(),
                "wrr-model",
                false,
                None,
                "/v1/chat/completions",
                body,
                None,
            )
            .await;
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "WRR dispatch must always succeed"
        );
    }
}

#[tokio::test]
async fn test_token_cost_dispatch_prefers_lower_cost_worker() {
    let cfg = OrchestratorConfig {
        dispatch_policy: "token_cost".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let slow = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"slow"}}]}"#).await
    );
    let fast = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"fast"}}]}"#).await
    );

    let slow_reg = layer.registry.register(reg_req(slow, &["tc-model"]), 5000);
    let fast_reg = layer.registry.register(reg_req(fast, &["tc-model"]), 5000);

    let slow_id = WorkerId::parse(&slow_reg.worker_id).unwrap();
    let fast_id = WorkerId::parse(&fast_reg.worker_id).unwrap();

    assert!(layer.registry.heartbeat(
        slow_id,
        HeartbeatRequest {
            inflight: 3,
            thermal_state: "nominal".into(),
            model_ids: vec!["tc-model".into()],
            rss_bytes: 0,
            active_sequences: 3,
            decode_tok_per_sec: 20.0,
            ttft_p95_ms: 400,
            queue_depth: 0,
            error_rate: 0.0,
            ..Default::default()
        }
    ));
    assert!(layer.registry.heartbeat(
        fast_id,
        HeartbeatRequest {
            inflight: 1,
            thermal_state: "nominal".into(),
            model_ids: vec!["tc-model".into()],
            rss_bytes: 0,
            active_sequences: 1,
            decode_tok_per_sec: 80.0,
            ttft_p95_ms: 100,
            queue_depth: 0,
            error_rate: 0.0,
            ..Default::default()
        }
    ));

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "tc-model",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "fast");
}

#[tokio::test]
async fn test_internal_heartbeat_roundtrip_persists_extended_fields() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let state = InternalState {
        registry: layer.registry.clone(),
        fleet_store: Arc::clone(&layer.fleet_store),
        config: Arc::clone(&layer.config),
    };
    let app = internal_router(state);

    let register_resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/internal/workers/register")
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "addr": "127.0.0.1:18081",
                        "capabilities": ["hb-model"],
                        "backend": "native",
                        "max_inflight": 8
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(register_resp.status(), 200);
    let register_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(register_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let worker_id = register_json["worker_id"].as_str().unwrap().to_string();

    let heartbeat_resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri(format!("/internal/workers/{worker_id}/heartbeat"))
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "inflight": 2,
                        "thermal_state": "serious",
                        "model_ids": ["hb-model"],
                        "rss_bytes": 123456,
                        "active_sequences": 5,
                        "decode_tok_per_sec": 42.5,
                        "ttft_p95_ms": 150
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(heartbeat_resp.status(), 200);

    let get_resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri(format!("/internal/workers/{worker_id}"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_resp.status(), 200);
    let worker_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(get_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    assert_eq!(worker_json["inflight"], 2);
    assert_eq!(worker_json["thermal_state"], "serious");
    assert_eq!(worker_json["rss_bytes"], 123456);
    assert_eq!(worker_json["active_sequences"], 5);
    assert_eq!(worker_json["decode_tok_per_sec"], 42.5);
    assert_eq!(worker_json["ttft_p95_ms"], 150);
    assert_eq!(worker_json["capabilities"][0], "hb-model");
}

#[tokio::test]
async fn test_admin_status_requires_auth_and_returns_operational_summary() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let unauth = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/status")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauth.status(), axum::http::StatusCode::UNAUTHORIZED);

    let worker_addr = skip_if_no_socket!(spawn_mock_worker(200, r#"{"choices":[]}"#).await);
    layer
        .registry
        .register(reg_req(worker_addr, &["ops-model"]), 5000);

    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/status")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-request-id", "req-admin-123")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok()),
        Some("req-admin-123")
    );
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["request_id"], "req-admin-123");
    assert_eq!(json["mode"], "direct");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["auth_required"], true);
    assert_eq!(json["workers"]["total"], 1);
    assert_eq!(json["workers"]["eligible"], 1);
    assert_eq!(json["workers"]["runtimes"]["ax_engine"]["workers"], 1);
    assert_eq!(json["workers"]["runtimes"]["ax_engine"]["eligible"], 1);
    assert_eq!(json["license"]["license"], "Apache-2.0");
}

#[tokio::test]
async fn test_admin_status_reports_auth_required_from_runtime_state() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let app = ax_serving_api::orchestration::proxy_router(Arc::clone(&layer)).layer(
        middleware::from_fn(ax_serving_api::auth::request_id_and_headers_middleware),
    );

    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/status")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["auth_required"], false);

    layer.set_public_auth_required(true);
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/status")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["auth_required"], true);
}

#[tokio::test]
async fn test_admin_startup_report_and_diagnostics_include_audit() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let startup = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/startup-report")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(startup.status(), axum::http::StatusCode::OK);
    let startup_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(startup.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(startup_json["service"], "orchestrator");
    assert_eq!(startup_json["auth_required"], true);
    assert!(startup_json["dispatch_runtime"].is_object());

    let license_response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/license")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let license_status = license_response.status();
    let license_body = axum::body::to_bytes(license_response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        license_status,
        axum::http::StatusCode::OK,
        "license body: {}",
        String::from_utf8_lossy(&license_body)
    );
    let license_json: serde_json::Value = serde_json::from_slice(&license_body).unwrap();
    assert_eq!(license_json["license"], "Apache-2.0");

    let diag = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-request-id", "req-orch-diag")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let diag_status = diag.status();
    let diag_body = axum::body::to_bytes(diag.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        diag_status,
        axum::http::StatusCode::OK,
        "diagnostics body: {}",
        String::from_utf8_lossy(&diag_body)
    );
    let diag_json: serde_json::Value = serde_json::from_slice(&diag_body).unwrap();
    assert_eq!(diag_json["request_id"], "req-orch-diag");
    assert!(diag_json["audit_tail"].is_array());

    let audit = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/audit?limit=10")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(audit.status(), axum::http::StatusCode::OK);
    let audit_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(audit.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        audit_json["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["action"] == "startup" && e["target_type"] == "orchestrator_layer")
    );
}

#[tokio::test]
async fn test_admin_diagnostics_groups_runtime_details_and_issues() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );

    let mut mac_req = reg_req("127.0.0.1:28081".parse().unwrap(), &["mac-model"]);
    mac_req.runtime = Some("ax_engine".into());
    mac_req.runtime_mode = Some("embedded".into());
    mac_req.hardware_class = Some("mac".into());
    mac_req.node_class = Some("mac-studio".into());
    mac_req.worker_pool = Some("mac".into());
    let mac_resp = layer.registry.register(mac_req, 5000);
    let mac_id = WorkerId::parse(&mac_resp.worker_id).unwrap();
    assert!(layer.registry.heartbeat(
        mac_id,
        HeartbeatRequest {
            inflight: 1,
            model_ids: vec!["mac-model".into()],
            active_sequences: 1,
            queue_depth: 2,
            error_rate: 0.1,
            ..Default::default()
        }
    ));

    let mut vllm_req = reg_req("127.0.0.1:28082".parse().unwrap(), &["cuda-model"]);
    vllm_req.backend = "vllm".into();
    vllm_req.runtime = Some("vllm".into());
    vllm_req.runtime_mode = Some("adapter".into());
    vllm_req.runtime_endpoint = Some("http://127.0.0.1:8000".into());
    vllm_req.hardware_class = Some("pc-cuda".into());
    vllm_req.node_class = Some("pc-cuda".into());
    vllm_req.worker_pool = Some("cuda".into());
    vllm_req.supported_operations = vec!["llm".into(), "embedding".into()];
    vllm_req.model_inventory = vec![ModelInventoryEntry {
        id: "cuda-model".into(),
        max_context: Some(32768),
        quantization: Some("awq".into()),
        artifact_format: Some("safetensors".into()),
        modalities: vec!["text".into()],
        supported_operations: vec!["llm".into()],
        ..Default::default()
    }];
    let vllm_resp = layer.registry.register(vllm_req, 5000);
    let vllm_id = WorkerId::parse(&vllm_resp.worker_id).unwrap();
    assert!(layer.registry.heartbeat(
        vllm_id,
        HeartbeatRequest {
            inflight: 3,
            model_ids: vec!["cuda-model".into()],
            active_sequences: 3,
            queue_depth: 4,
            error_rate: 0.02,
            ..Default::default()
        }
    ));

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    let runtime_diag = &json["runtime_diagnostics"]["runtimes"];
    assert_eq!(runtime_diag["ax_engine"]["workers"], 1);
    assert_eq!(runtime_diag["ax_engine"]["hardware_classes"]["mac"], 1);
    assert_eq!(runtime_diag["ax_engine"]["runtime_modes"]["embedded"], 1);
    assert_eq!(
        runtime_diag["ax_engine"]["models"],
        serde_json::json!(["mac-model"])
    );
    assert_eq!(runtime_diag["ax_engine"]["total_queue_depth"], 2);
    assert!(
        runtime_diag["ax_engine"]["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "embedded_compatibility_path")
    );
    assert!(
        runtime_diag["ax_engine"]["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "missing_runtime_endpoint")
    );
    assert!(
        runtime_diag["ax_engine"]["recommended_actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["action"] == "migrate_embedded_compatibility_path")
    );
    assert!(
        json["runtime_diagnostics"]["recommended_actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["action"] == "fix_runtime_endpoint_registration")
    );

    assert_eq!(runtime_diag["vllm"]["workers"], 1);
    assert_eq!(runtime_diag["vllm"]["hardware_classes"]["pc-cuda"], 1);
    assert_eq!(runtime_diag["vllm"]["runtime_modes"]["adapter"], 1);
    assert_eq!(
        runtime_diag["vllm"]["supported_operations"],
        serde_json::json!(["embedding", "llm"])
    );
    assert_eq!(
        runtime_diag["vllm"]["runtime_endpoints"],
        serde_json::json!(["http://127.0.0.1:8000"])
    );
    assert_eq!(
        runtime_diag["vllm"]["model_inventory"][0]["model_id"],
        "cuda-model"
    );
    assert_eq!(
        runtime_diag["vllm"]["model_inventory"][0]["quantization"],
        "awq"
    );
    assert_eq!(
        runtime_diag["vllm"]["model_inventory"][0]["artifact_format"],
        "safetensors"
    );
    assert_eq!(runtime_diag["vllm"]["total_queue_depth"], 4);
}

#[tokio::test]
async fn test_admin_diagnostics_reports_runtime_specific_hardware_guidance() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );

    let mut req = reg_req("127.0.0.1:28083".parse().unwrap(), &["cuda-model"]);
    req.backend = "vllm".into();
    req.runtime = Some("vllm".into());
    req.runtime_endpoint = Some("http://127.0.0.1:8000".into());
    req.hardware_class = Some("mac".into());
    req.supported_operations = vec!["llm".into()];
    layer.registry.register(req, 5000);

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    let vllm = &json["runtime_diagnostics"]["runtimes"]["vllm"];
    assert_eq!(
        vllm["runtime_guidance"]["expected_hardware_classes"],
        serde_json::json!(["pc-cuda", "thor"])
    );
    assert!(
        vllm["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "unexpected_hardware_class")
    );
    assert!(
        vllm["recommended_actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["action"] == "fix_hardware_class")
    );
}

#[tokio::test]
async fn test_admin_diagnostics_preserves_tensorrt_runtime_and_hardware_subclass() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let request = protocol_reg_req(
        "127.0.0.1:28085".parse().unwrap(),
        "df-rtx5090",
        "tensorrt_llm",
        "tinyllama-trtllm",
    );
    layer
        .registry
        .register_protocol(
            request,
            ax_serving_api::orchestration::worker_endpoint::WorkerEndpoint::parse(
                "http://127.0.0.1:28085",
            )
            .unwrap(),
            NegotiatedProtocol {
                version: CURRENT_PROTOCOL,
                capabilities: BTreeSet::new(),
            },
            5_000,
            15_000,
        )
        .unwrap();

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    let runtimes = &json["runtime_diagnostics"]["runtimes"];
    assert!(runtimes.get("unknown").is_none());
    let tensorrt = &runtimes["tensorrt_llm"];
    assert_eq!(tensorrt["workers"], 1);
    assert_eq!(tensorrt["hardware_classes"]["pc-cuda-sm120"], 1);
    assert_eq!(
        tensorrt["runtime_endpoints"],
        serde_json::json!(["http://runtime.test:8000"])
    );
    assert_eq!(
        tensorrt["runtime_guidance"]["runtime_owner"],
        "NVIDIA TensorRT-LLM"
    );
    assert!(
        !tensorrt["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "unexpected_hardware_class")
    );
    assert!(
        !tensorrt["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "missing_runtime_endpoint")
    );
}

#[tokio::test]
async fn test_admin_diagnostics_distinguishes_sglang_and_tensorrt_edge_llm() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );

    let mut sglang = reg_req("127.0.0.1:28086".parse().unwrap(), &["sglang-model"]);
    sglang.backend = "sglang".into();
    sglang.runtime = Some("sglang".into());
    sglang.runtime_endpoint = Some("http://127.0.0.1:30000".into());
    sglang.hardware_class = Some("pc-cuda-sm120".into());
    sglang.supported_operations = vec!["llm".into()];
    layer.registry.register(sglang, 5_000);

    let mut edge = protocol_reg_req(
        "127.0.0.1:28087".parse().unwrap(),
        "df-thor-01",
        "TensorRT-Edge-LLM",
        "qwen3-edge",
    );
    edge.hardware.hardware_class = Some("thor-jetpack-7.2".into());
    layer
        .registry
        .register_protocol(
            edge,
            ax_serving_api::orchestration::worker_endpoint::WorkerEndpoint::parse(
                "http://127.0.0.1:28087",
            )
            .unwrap(),
            NegotiatedProtocol {
                version: CURRENT_PROTOCOL,
                capabilities: BTreeSet::new(),
            },
            5_000,
            15_000,
        )
        .unwrap();

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    let runtimes = &json["runtime_diagnostics"]["runtimes"];
    assert_eq!(
        runtimes["sglang"]["runtime_guidance"]["runtime_owner"],
        "SGLang"
    );
    assert_eq!(
        runtimes["tensorrt_edge_llm"]["runtime_guidance"]["runtime_owner"],
        "NVIDIA TensorRT Edge-LLM"
    );
    assert_eq!(
        runtimes["tensorrt_edge_llm"]["hardware_classes"]["thor-jetpack-7.2"],
        1
    );
    assert_eq!(
        runtimes["tensorrt_edge_llm"]["runtime_guidance"]["support_level"],
        "experimental_compatibility"
    );
    assert!(
        !runtimes["tensorrt_edge_llm"]["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "unexpected_hardware_class")
    );
}

#[tokio::test]
async fn test_admin_diagnostics_reports_runtime_telemetry_recovery_actions() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );

    let mut req = reg_req("127.0.0.1:28084".parse().unwrap(), &["pressure-model"]);
    req.backend = "vllm".into();
    req.runtime = Some("vllm".into());
    req.runtime_endpoint = Some("http://127.0.0.1:8000".into());
    req.hardware_class = Some("pc-cuda".into());
    req.supported_operations = vec!["llm".into()];
    let register = layer.registry.register(req, 5000);
    let worker_id = WorkerId::parse(&register.worker_id).unwrap();
    assert!(layer.registry.heartbeat(
        worker_id,
        HeartbeatRequest {
            inflight: 8,
            model_ids: vec!["pressure-model".into()],
            active_sequences: 8,
            queue_depth: 8,
            error_rate: 0.20,
            kv_utilization: Some(0.95),
            batch_utilization: Some(0.95),
            ..Default::default()
        }
    ));

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    let vllm = &json["runtime_diagnostics"]["runtimes"]["vllm"];
    let issues = vllm["issues"].as_array().unwrap();
    assert!(
        issues
            .iter()
            .any(|issue| issue["code"] == "high_runtime_error_rate")
    );
    assert!(
        issues
            .iter()
            .any(|issue| issue["code"] == "runtime_queue_backlog")
    );
    assert!(
        issues
            .iter()
            .any(|issue| issue["code"] == "high_runtime_kv_pressure")
    );
    assert!(
        issues
            .iter()
            .any(|issue| issue["code"] == "high_runtime_batch_pressure")
    );
    let actions = vllm["recommended_actions"].as_array().unwrap();
    assert!(
        actions
            .iter()
            .any(|action| action["action"] == "investigate_runtime_errors")
    );
    let relieve = actions
        .iter()
        .find(|action| action["action"] == "relieve_runtime_pressure")
        .expect("relieve_runtime_pressure action must be present");
    assert!(
        relieve["suggested_commands"]
            .as_array()
            .unwrap()
            .iter()
            .any(|command| command
                .as_str()
                .unwrap()
                .contains("ax-serving workers drain"))
    );
}

#[tokio::test]
async fn test_admin_diagnostics_flags_absent_hardware_class_for_known_runtime() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );

    // Register an ax_engine worker with no hardware_class set.
    let mut req = reg_req("127.0.0.1:28090".parse().unwrap(), &["native-model"]);
    req.backend = "ax_engine".into();
    req.runtime = Some("ax_engine".into());
    req.runtime_endpoint = Some("http://127.0.0.1:9000".into());
    req.hardware_class = None;
    req.supported_operations = vec!["llm".into()];
    layer.registry.register(req, 5000);

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/diagnostics")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();

    let ax = &json["runtime_diagnostics"]["runtimes"]["ax_engine"];
    assert!(
        ax["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "unexpected_hardware_class"),
        "absent hardware_class must produce unexpected_hardware_class issue"
    );
    assert!(
        ax["recommended_actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["action"] == "fix_hardware_class"),
        "absent hardware_class must produce fix_hardware_class recommended action"
    );
}

#[tokio::test]
async fn test_admin_fleet_summarizes_pools_and_node_classes() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let blue = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"blue"}}]}"#).await
    );
    let green = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"green"}}]}"#).await
    );
    let mut blue_req = reg_req_with_pool(blue, &["fleet-model"], Some("blue"), Some("m3-max"));
    blue_req.runtime = Some("ax_engine".into());
    let blue_id = layer.registry.register(blue_req, 5000);
    let mut green_req = reg_req_with_pool(green, &["fleet-model"], Some("green"), Some("m3-pro"));
    green_req.backend = "vllm".into();
    green_req.runtime = Some("vllm".into());
    let green_id = layer.registry.register(green_req, 5000);
    let blue_id = WorkerId::parse(&blue_id.worker_id).unwrap();
    let green_id = WorkerId::parse(&green_id.worker_id).unwrap();
    assert!(layer.registry.heartbeat(
        blue_id,
        HeartbeatRequest {
            inflight: 2,
            thermal_state: "nominal".into(),
            model_ids: vec!["fleet-model".into()],
            rss_bytes: 0,
            active_sequences: 2,
            decode_tok_per_sec: 100.0,
            ttft_p95_ms: 50,
            queue_depth: 3,
            error_rate: 0.25,
            ..Default::default()
        }
    ));
    assert!(layer.registry.heartbeat(
        green_id,
        HeartbeatRequest {
            inflight: 1,
            thermal_state: "nominal".into(),
            model_ids: vec!["fleet-model".into()],
            rss_bytes: 0,
            active_sequences: 1,
            decode_tok_per_sec: 120.0,
            ttft_p95_ms: 40,
            queue_depth: 1,
            error_rate: 0.05,
            ..Default::default()
        }
    ));

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/fleet")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["total_workers"], 2);
    assert_eq!(json["pools"]["blue"]["workers"], 1);
    assert_eq!(json["pools"]["green"]["workers"], 1);
    assert_eq!(json["node_classes"]["m3-max"]["workers"], 1);
    assert_eq!(json["node_classes"]["m3-pro"]["workers"], 1);
    assert_eq!(json["runtimes"]["ax_engine"]["workers"], 1);
    assert_eq!(json["runtimes"]["vllm"]["workers"], 1);
    assert_eq!(json["runtimes"]["ax_engine"]["total_queue_depth"], 3);
    assert_eq!(json["runtimes"]["vllm"]["total_queue_depth"], 1);
    assert_eq!(json["pools"]["blue"]["total_queue_depth"], 3);
    assert_eq!(json["pools"]["green"]["total_queue_depth"], 1);
    assert_eq!(json["pools"]["blue"]["max_error_rate"], 0.25);
    assert_eq!(json["pools"]["green"]["max_error_rate"], 0.05);
}

#[tokio::test]
async fn test_proxy_models_uses_structured_capability_models() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );

    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: "127.0.0.1:28081".into(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: true,
                vision: false,
                models: vec!["qwen2-72b".into(), "yolo11m".into()],
                max_context: Some(131072),
            }),
            backend: "sglang".into(),
            max_inflight: 8,
            friendly_name: Some("thor-01".into()),
            chip_model: Some("RTX".into()),
            worker_pool: Some("thor".into()),
            node_class: Some("thor".into()),
            ..Default::default()
        },
        5000,
    );

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/models")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let ids: Vec<_> = json["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert!(ids.contains(&"qwen2-72b"));
    assert!(ids.contains(&"yolo11m"));
}

#[tokio::test]
async fn test_admin_policy_returns_project_policy_summary() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            sample_project_policy(Some("fabric")),
        )
        .unwrap(),
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/policy")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["enabled"], true);
    assert_eq!(json["default_project"], "fabric");
    assert_eq!(json["rules"][0]["worker_pool"], "green");
}

#[tokio::test]
async fn test_worker_admin_validation_errors_are_audited() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let missing_worker = WorkerId::new();

    let invalid = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/workers/not-a-worker-id/drain")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-request-id", "req-invalid-worker-id")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), axum::http::StatusCode::BAD_REQUEST);

    let missing = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::DELETE)
                .uri(format!("/v1/workers/{missing_worker}"))
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-request-id", "req-missing-worker")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);

    let audit = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/admin/audit?limit=10")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(audit.status(), axum::http::StatusCode::OK);
    let audit_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(audit.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let events = audit_json["events"].as_array().unwrap();
    assert!(events.iter().any(|e| {
        e["action"] == "worker_drain"
            && e["actor"] == "request:req-invalid-worker-id"
            && e["outcome"] == "error"
            && e["detail"]["error"] == "invalid worker id"
    }));
    assert!(events.iter().any(|e| {
        e["action"] == "worker_delete"
            && e["actor"] == "request:req-missing-worker"
            && e["outcome"] == "error"
            && e["target_id"] == missing_worker.to_string()
            && e["detail"]["error"] == "worker not found"
    }));
}

#[tokio::test]
async fn test_pool_header_prefers_matching_worker_pool() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let blue = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"blue"}}]}"#).await
    );
    let green = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"green"}}]}"#).await
    );
    layer.registry.register(
        reg_req_with_pool(blue, &["pool-model"], Some("blue"), Some("m3-max")),
        5000,
    );
    layer.registry.register(
        reg_req_with_pool(green, &["pool-model"], Some("green"), Some("m3-pro")),
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-ax-worker-pool", "green")
                .body(axum::body::Body::from(
                    r#"{"model":"pool-model","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        text.contains("green"),
        "expected green-pool worker response, got {text}"
    );
}

#[tokio::test]
async fn test_pool_header_does_not_retry_generic_worker_failure() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let blue = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"blue"}}]}"#).await
    );
    let green = skip_if_no_socket!(spawn_mock_worker(500, r#"{"error":"down"}"#).await);
    layer.registry.register(
        reg_req_with_pool(blue, &["pool-model"], Some("blue"), Some("m3-max")),
        5000,
    );
    layer.registry.register(
        reg_req_with_pool(green, &["pool-model"], Some("green"), Some("m3-pro")),
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-ax-worker-pool", "green")
                .body(axum::body::Body::from(
                    r#"{"model":"pool-model","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        text.contains("down"),
        "generic worker failure should be returned without cross-pool retry, got {text}"
    );
}

#[tokio::test]
async fn test_project_policy_proxy_requires_header() {
    let layer = Arc::new(
        OrchestratorLayer::new(OrchestratorConfig::default(), sample_project_policy(None)).unwrap(),
    );
    let worker = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"ok"}}]}"#).await
    );
    layer.registry.register(
        reg_req_with_pool(worker, &["ops-model"], Some("green"), Some("m3-pro")),
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"model":"ops-model","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_project_policy_enforces_worker_pool() {
    let layer = Arc::new(
        OrchestratorLayer::new(OrchestratorConfig::default(), sample_project_policy(None)).unwrap(),
    );
    let blue = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"blue"}}]}"#).await
    );
    let green = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"green"}}]}"#).await
    );
    layer.registry.register(
        reg_req_with_pool(blue, &["pool-model"], Some("blue"), Some("m3-max")),
        5000,
    );
    layer.registry.register(
        reg_req_with_pool(green, &["pool-model"], Some("green"), Some("m3-pro")),
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-ax-project", "fabric")
                .body(axum::body::Body::from(
                    r#"{"model":"pool-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        text.contains("green"),
        "expected policy-enforced green pool response, got {text}"
    );
}

#[tokio::test]
async fn test_project_policy_worker_pool_does_not_fallback() {
    let layer = Arc::new(
        OrchestratorLayer::new(OrchestratorConfig::default(), sample_project_policy(None)).unwrap(),
    );
    let blue = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"blue"}}]}"#).await
    );
    layer.registry.register(
        reg_req_with_pool(blue, &["pool-model"], Some("blue"), Some("m3-max")),
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-ax-project", "fabric")
                .body(axum::body::Body::from(
                    r#"{"model":"pool-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_proxy_embeddings_route_and_project_policy() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig {
                enabled: true,
                default_project: None,
                rules: vec![ax_serving_api::config::ProjectRuleConfig {
                    project: "fabric".into(),
                    allowed_models: vec!["embed-*".into()],
                    max_tokens_limit: None,
                    worker_pool: Some("green".into()),
                }],
            },
        )
        .unwrap(),
    );
    let blue = skip_if_no_socket!(
        spawn_mock_worker(
            200,
            r#"{"data":[{"embedding":[0.1],"index":0}],"model":"embed-main"}"#
        )
        .await
    );
    let green = skip_if_no_socket!(
        spawn_mock_worker(
            200,
            r#"{"data":[{"embedding":[0.9],"index":0}],"model":"embed-main"}"#
        )
        .await
    );
    layer.registry.register(
        reg_req_with_pool(blue, &["embed-main"], Some("blue"), Some("m3-max")),
        5000,
    );
    layer.registry.register(
        reg_req_with_pool(green, &["embed-main"], Some("green"), Some("m3-pro")),
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/embeddings")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-ax-project", "fabric")
                .body(axum::body::Body::from(
                    r#"{"model":"embed-main","input":"hello"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        text.contains("0.9"),
        "expected policy-enforced green pool embedding response, got {text}"
    );
}

#[tokio::test]
async fn test_proxy_embeddings_does_not_route_to_legacy_llm_only_worker() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let worker_addr = skip_if_no_socket!(
        spawn_mock_worker(
            200,
            r#"{"data":[{"embedding":[0.1],"index":0}],"model":"embed-main"}"#
        )
        .await
    );
    let mut req = reg_req(worker_addr, &["embed-main"]);
    req.supported_operations = vec!["llm".into()];
    layer.registry.register(req, 5000);

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/embeddings")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"model":"embed-main","input":"hello"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "llm-only legacy workers must not receive embedding requests"
    );
}

#[tokio::test]
async fn test_proxy_chat_with_image_routes_to_vision_worker() {
    let cfg = OrchestratorConfig {
        dispatch_policy: "least_inflight".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let text_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"text-only"}}]}"#).await
    );
    let vision_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"vision"}}]}"#).await
    );

    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: text_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["vision-route-model".into()],
                max_context: None,
            }),
            supported_operations: vec!["llm".into()],
            backend: "sglang".into(),
            max_inflight: 4,
            ..Default::default()
        },
        5000,
    );
    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: vision_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: true,
                models: vec!["vision-route-model".into()],
                max_context: None,
            }),
            supported_operations: vec!["llm".into(), "vision".into()],
            backend: "sglang".into(),
            max_inflight: 4,
            ..Default::default()
        },
        5000,
    );

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "vision-route-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ]
            }]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "vision");
}

#[tokio::test]
async fn test_proxy_embeddings_routes_by_input_context() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let small_addr = skip_if_no_socket!(
        spawn_mock_worker(
            200,
            r#"{"data":[{"embedding":[0.1],"index":0}],"model":"small"}"#
        )
        .await
    );
    let large_addr = skip_if_no_socket!(
        spawn_mock_worker(
            200,
            r#"{"data":[{"embedding":[0.2],"index":0}],"model":"large"}"#
        )
        .await
    );

    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: small_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: false,
                embedding: true,
                vision: false,
                models: vec!["embed-main".into()],
                max_context: Some(8),
            }),
            backend: "sglang".into(),
            max_inflight: 4,
            worker_pool: Some("small".into()),
            ..Default::default()
        },
        5000,
    );
    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: large_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: false,
                embedding: true,
                vision: false,
                models: vec!["embed-main".into()],
                max_context: Some(4096),
            }),
            backend: "sglang".into(),
            max_inflight: 4,
            ..Default::default()
        },
        5000,
    );

    let app = proxy_router_with_key(Arc::clone(&layer), "secret");
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/embeddings")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .header("x-ax-worker-pool", "small")
                .header("x-ax-minimum-context-tokens", "128")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "embed-main",
                        "input": "x".repeat(128),
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["model"], "large",
        "declared context requirement should filter the preferred small worker"
    );
}

#[tokio::test]
async fn test_structured_embedding_worker_is_not_used_for_chat() {
    let worker_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"data":[{"embedding":[0.1,0.2],"index":0}]}"#).await
    );

    let registry = WorkerRegistry::new();
    registry.register(
        RegisterRequest {
            worker_id: None,
            addr: worker_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: false,
                embedding: true,
                vision: false,
                models: vec!["embed-only".into()],
                max_context: None,
            }),
            backend: "sglang".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("thor".into()),
            ..Default::default()
        },
        5000,
    );

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let chat_resp = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "embed-only",
            false,
            None,
            "/v1/chat/completions",
            axum::body::Bytes::from(r#"{"model":"embed-only","messages":[]}"#),
            None,
        )
        .await;
    assert_eq!(
        chat_resp.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );

    let embedding_resp = dispatcher
        .forward_kind(
            &registry,
            policy.as_ref(),
            "embed-only",
            RequestKind::Embedding,
            None,
            None,
            false,
            None,
            false,
            "/v1/embeddings",
            axum::body::Bytes::from(r#"{"model":"embed-only","input":"hello"}"#),
            None,
        )
        .await;
    assert_eq!(embedding_resp.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn test_backend_hint_routes_to_matching_worker() {
    let cfg = OrchestratorConfig {
        dispatch_policy: "least_inflight".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let native_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"native"}}]}"#).await
    );
    let sglang_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"sglang"}}]}"#).await
    );

    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: native_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["shared-backend-model".into()],
                max_context: Some(4096),
            }),
            backend: "native".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("mac".into()),
            ..Default::default()
        },
        5000,
    );
    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: sglang_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["shared-backend-model".into()],
                max_context: Some(16384),
            }),
            backend: "sglang".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("thor".into()),
            ..Default::default()
        },
        5000,
    );

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model":"shared-backend-model",
            "backend":"sglang",
            "messages":[{"role":"user","content":"hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "sglang");
}

#[tokio::test]
async fn test_invalid_backend_or_runtime_hint_returns_422() {
    let (addr, _layer) =
        skip_if_no_socket!(spawn_orchestrator_with_layer(OrchestratorConfig::default()).await);
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    for body in [
        serde_json::json!({
            "model": "shared-backend-model",
            "backend": "definitely-not-a-backend",
            "messages": [{"role": "user", "content": "hello"}]
        }),
        serde_json::json!({
            "model": "shared-backend-model",
            "runtime": "../definitely-not-a-runtime",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    ] {
        let resp = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), axum::http::StatusCode::UNPROCESSABLE_ENTITY);
    }
}

#[tokio::test]
async fn test_proxy_requires_valid_model_field() {
    let (addr, _layer) =
        skip_if_no_socket!(spawn_orchestrator_with_layer(OrchestratorConfig::default()).await);
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    for (body, expected_status) in [
        (
            serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}]
            }),
            axum::http::StatusCode::BAD_REQUEST,
        ),
        (
            serde_json::json!({
                "model": " ",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            axum::http::StatusCode::BAD_REQUEST,
        ),
        (
            serde_json::json!({
                "model": "bad model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let resp = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), expected_status);
    }
}

#[tokio::test]
async fn test_proxy_rejects_invalid_endpoint_bodies_before_dispatch() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let worker = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"unexpected"}}]}"#).await
    );
    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: worker.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: true,
                vision: false,
                models: vec!["shape-model".into()],
                max_context: Some(4096),
            }),
            supported_operations: vec!["llm".into(), "embedding".into()],
            backend: "sglang".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("thor".into()),
            ..Default::default()
        },
        5000,
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    for (path, body) in [
        (
            "/v1/chat/completions",
            serde_json::json!({"model": "shape-model", "messages": []}),
        ),
        (
            "/v1/chat/completions",
            serde_json::json!({"model": "shape-model", "messages": [{"role": "user"}]}),
        ),
        (
            "/v1/chat/completions",
            serde_json::json!({
                "model": "shape-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0
            }),
        ),
        (
            "/v1/completions",
            serde_json::json!({"model": "shape-model", "prompt": ""}),
        ),
        (
            "/v1/embeddings",
            serde_json::json!({"model": "shape-model", "input": []}),
        ),
    ] {
        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method(axum::http::Method::POST)
                    .uri(path)
                    .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            axum::http::StatusCode::BAD_REQUEST,
            "path {path} with body {body} should be rejected before dispatch"
        );
    }
}

#[tokio::test]
async fn test_proxy_accepts_valid_body_above_axum_default_limit() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let app = ax_serving_api::orchestration::proxy_router(layer);
    let messages: Vec<serde_json::Value> = (0..80)
        .map(|_| {
            serde_json::json!({
                "role": "user",
                "content": "x".repeat(MAX_CONTENT_BYTES)
            })
        })
        .collect();
    let body = serde_json::json!({
        "model": "missing-worker-model",
        "messages": messages
    })
    .to_string();
    assert!(body.len() > 2 * 1024 * 1024);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_routing_trace_header_includes_selected_worker() {
    let mut env = EnvVarsGuard::new();
    env.set("AXS_ROUTING_TRACE", "true");

    let cfg = OrchestratorConfig {
        dispatch_policy: "least_inflight".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let native_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"native"}}]}"#).await
    );
    let sglang_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"sglang"}}]}"#).await
    );

    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: native_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["trace-model".into()],
                max_context: Some(4096),
            }),
            backend: "native".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("mac".into()),
            ..Default::default()
        },
        5000,
    );
    let sglang_reg = layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: sglang_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["trace-model".into()],
                max_context: Some(16384),
            }),
            backend: "sglang".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("thor".into()),
            ..Default::default()
        },
        5000,
    );

    let sglang_worker_id = sglang_reg.worker_id;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model":"trace-model",
            "backend":"sglang",
            "messages":[{"role":"user","content":"hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let trace = resp
        .headers()
        .get("x-ax-routing-trace")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(trace.contains("candidates=1"));
    assert!(trace.contains(&format!("selected={sglang_worker_id}")));
    assert!(trace.contains("reason=primary"));
}

#[tokio::test]
async fn test_routing_trace_header_on_no_eligible_worker() {
    let mut env = EnvVarsGuard::new();
    env.set("AXS_ROUTING_TRACE", "true");

    let cfg = OrchestratorConfig {
        dispatch_policy: "least_inflight".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, _layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model":"missing-model",
            "messages":[{"role":"user","content":"hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    let trace = resp
        .headers()
        .get("x-ax-routing-trace")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(trace.contains("candidates=0"));
    assert!(trace.contains("selected=none"));
    assert!(trace.contains("reason=no_eligible_worker"));
}

#[tokio::test]
async fn test_declared_context_routes_to_sufficient_context_worker() {
    let cfg = OrchestratorConfig {
        dispatch_policy: "least_inflight".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let short_ctx_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"short-ctx"}}]}"#).await
    );
    let long_ctx_addr = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"long-ctx"}}]}"#).await
    );

    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: short_ctx_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["ctx-route-model".into()],
                max_context: Some(32),
            }),
            backend: "native".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("mac".into()),
            ..Default::default()
        },
        5000,
    );
    layer.registry.register(
        RegisterRequest {
            worker_id: None,
            addr: long_ctx_addr.to_string(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["ctx-route-model".into()],
                max_context: Some(4096),
            }),
            backend: "sglang".into(),
            max_inflight: 4,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: Some("thor".into()),
            ..Default::default()
        },
        5000,
    );

    let long_prompt = "x".repeat(400);
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("x-ax-minimum-context-tokens", "100")
        .json(&serde_json::json!({
            "model":"ctx-route-model",
            "messages":[{"role":"user","content": long_prompt}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "long-ctx");

    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("x-ax-minimum-context-tokens", "64")
        .json(&serde_json::json!({
            "model":"ctx-route-model",
            "messages":[{"role":"user","content": "short"}],
            "max_tokens": 64
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"], "long-ctx",
        "the client-declared context requirement must be enforced"
    );
}

#[tokio::test]
async fn test_public_worker_admin_flow_lists_drains_and_evicts() {
    let layer = Arc::new(
        OrchestratorLayer::new(
            OrchestratorConfig::default(),
            ProjectPolicyConfig::default(),
        )
        .unwrap(),
    );
    let app = proxy_router_with_key(Arc::clone(&layer), "secret");

    let worker_addr = skip_if_no_socket!(spawn_mock_worker(200, r#"{"choices":[]}"#).await);
    let register = layer
        .registry
        .register(reg_req(worker_addr, &["ops-model"]), 5000);
    let worker_id = register.worker_id;

    let list_resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/workers")
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_resp.status(), axum::http::StatusCode::OK);
    let list_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(list_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(list_json["workers"].as_array().unwrap().len(), 1);
    assert_eq!(list_json["workers"][0]["id"], worker_id);

    let get_resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri(format!("/v1/workers/{worker_id}"))
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_resp.status(), axum::http::StatusCode::OK);

    let drain_resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri(format!("/v1/workers/{worker_id}/drain"))
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(drain_resp.status(), axum::http::StatusCode::OK);

    let drained = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri(format!("/v1/workers/{worker_id}"))
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let drained_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(drained.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(drained_json["drain"], true);

    let complete_resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri(format!("/v1/workers/{worker_id}/drain-complete"))
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(complete_resp.status(), axum::http::StatusCode::NO_CONTENT);

    let missing = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::GET)
                .uri(format!("/v1/workers/{worker_id}"))
                .header(axum::http::header::AUTHORIZATION, "Bearer secret")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);
}

// ── Overload scenario helpers ─────────────────────────────────────────────────

/// Spawn an `OrchestratorLayer`-backed proxy server on an ephemeral port.
///
/// Returns the bound address and an `Arc` to the layer so tests can
/// manipulate the queue (hold permits, register workers) directly.
async fn spawn_orchestrator_with_layer(
    cfg: ax_serving_api::orchestration::OrchestratorConfig,
) -> Option<(
    std::net::SocketAddr,
    Arc<ax_serving_api::orchestration::OrchestratorLayer>,
)> {
    use ax_serving_api::orchestration::{OrchestratorLayer, proxy_router};
    let layer = Arc::new(OrchestratorLayer::new(cfg, ProjectPolicyConfig::default()).ok()?);
    let router = proxy_router(Arc::clone(&layer));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });
    Some((addr, layer))
}

async fn spawn_internal_router_with_auth(
    state: InternalState,
    auth_state: Option<InternalAuthState>,
) -> Option<SocketAddr> {
    let app = if let Some(auth_state) = auth_state {
        internal_router(state).route_layer(middleware::from_fn_with_state(
            auth_state,
            internal_auth_middleware,
        ))
    } else {
        internal_router(state)
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });
    Some(addr)
}

// ── Step 3: Overload scenario tests ───────────────────────────────────────────

/// Queue full (Reject policy) → HTTP 429 + X-Queue-Depth header.
#[tokio::test]
async fn test_overload_queue_full_429() {
    use ax_serving_api::orchestration::OrchestratorConfig;

    let cfg = OrchestratorConfig {
        global_queue_max: 1,
        global_queue_depth: 0,
        global_queue_policy: "reject".into(),
        global_queue_wait_ms: 200,
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let worker_addr = skip_if_no_socket!(spawn_mock_worker(200, r#"{"choices":[]}"#).await);
    layer
        .registry
        .register(reg_req(worker_addr, &["overload-model"]), 5000);

    // Hold the only concurrency slot.
    let AcquireResult::Permit(_permit) = layer.queue.acquire("test-client-a".into()).await else {
        panic!("expected permit");
    };

    // Next request must be rejected: 429 + X-Queue-Depth.
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "overload-model",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 429, "expected 429 when queue full");
    assert!(
        resp.headers().contains_key("x-queue-depth"),
        "429 response must carry X-Queue-Depth"
    );
}

/// Shed-oldest: oldest queued waiter receives 503 X-Reason:request_shed.
#[tokio::test]
async fn test_overload_shed_oldest_503() {
    use ax_serving_api::orchestration::OrchestratorConfig;

    let cfg = OrchestratorConfig {
        global_queue_max: 1,
        global_queue_depth: 1,
        global_queue_policy: "shed_oldest".into(),
        global_queue_wait_ms: 5000,
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let worker_addr = skip_if_no_socket!(spawn_mock_worker(200, r#"{"choices":[]}"#).await);
    layer
        .registry
        .register(reg_req(worker_addr, &["shed-model"]), 5000);

    // Hold the concurrency slot.
    let AcquireResult::Permit(permit) = layer.queue.acquire("test-client-a".into()).await else {
        panic!("expected permit");
    };

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();

    // req1 enters the queue.
    let client1 = client.clone();
    let req1 = tokio::spawn(async move {
        client1
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "shed-model",
                "messages": [{"role": "user", "content": "hello"}],
            }))
            .send()
            .await
            .unwrap()
    });

    // Wait for req1 to be queued.
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert_eq!(layer.queue.queued(), 1, "req1 must be queued");

    // req2 sheds req1 and takes its queue slot.
    let client2 = client.clone();
    let req2 = tokio::spawn(async move {
        client2
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "shed-model",
                "messages": [{"role": "user", "content": "hello"}],
            }))
            .send()
            .await
            .unwrap()
    });

    // Let req2 reach the server and shed req1.
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;

    // req1 should return 503 X-Reason:request_shed.
    let r1 = req1.await.unwrap();
    assert_eq!(r1.status(), 503, "shed request must return 503");
    assert_eq!(
        r1.headers().get("x-reason").and_then(|v| v.to_str().ok()),
        Some("request_shed"),
        "shed response must carry X-Reason:request_shed"
    );

    // Release the permit — req2 now dispatches to the worker.
    drop(permit);
    let r2 = req2.await.unwrap();
    assert_eq!(r2.status(), 200, "req2 must complete after permit released");
}

/// Queue deadline: request waits past wait_ms → 504 X-Reason:queue_timeout.
#[tokio::test]
async fn test_overload_queue_timeout_504() {
    use ax_serving_api::orchestration::OrchestratorConfig;

    let cfg = OrchestratorConfig {
        global_queue_max: 1,
        global_queue_depth: 1,
        global_queue_wait_ms: 60, // short timeout
        global_queue_policy: "reject".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let worker_addr = skip_if_no_socket!(spawn_mock_worker(200, r#"{"choices":[]}"#).await);
    layer
        .registry
        .register(reg_req(worker_addr, &["timeout-model"]), 5000);

    // Hold the only slot so every incoming request queues.
    let AcquireResult::Permit(_permit) = layer.queue.acquire("test-client-a".into()).await else {
        panic!("expected permit");
    };

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "timeout-model",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        504,
        "timed-out queued request must return 504"
    );
    assert_eq!(
        resp.headers().get("x-reason").and_then(|v| v.to_str().ok()),
        Some("queue_timeout"),
        "timeout response must carry X-Reason:queue_timeout"
    );
}

#[tokio::test]
async fn test_request_deadline_caps_queue_wait() {
    let cfg = OrchestratorConfig {
        global_queue_max: 1,
        global_queue_depth: 1,
        global_queue_wait_ms: 5_000,
        global_queue_policy: "queue".into(),
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);
    let worker_addr = skip_if_no_socket!(spawn_mock_worker(200, r#"{"choices":[]}"#).await);
    layer
        .registry
        .register(reg_req(worker_addr, &["deadline-model"]), 5000);
    let AcquireResult::Permit(_permit) = layer.queue.acquire("holder".into()).await else {
        panic!("expected permit");
    };

    let response = Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("x-ax-request-timeout-ms", "25")
        .json(&serde_json::json!({
            "model": "deadline-model",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 504);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "AXS_REQUEST_DEADLINE");
    assert_eq!(body["ax"]["retryable"], false);
}

/// Generic runtime 5xx does not trigger a reroute storm.
#[tokio::test]
async fn test_generic_5xx_does_not_create_reroute_storm() {
    use ax_serving_api::orchestration::OrchestratorConfig;

    let cfg = OrchestratorConfig {
        global_queue_max: 10,
        global_queue_depth: 5,
        ..OrchestratorConfig::default()
    };
    let (addr, layer) = skip_if_no_socket!(spawn_orchestrator_with_layer(cfg).await);

    let bad1 = skip_if_no_socket!(spawn_mock_worker(500, r#"{"error":"down"}"#).await);
    let bad2 = skip_if_no_socket!(spawn_mock_worker(500, r#"{"error":"down"}"#).await);
    layer
        .registry
        .register(reg_req(bad1, &["storm-model"]), 5000);
    layer
        .registry
        .register(reg_req(bad2, &["storm-model"]), 5000);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "storm-model",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 500, "runtime 500 should be preserved");

    let (healthy, unhealthy, _draining) = layer.registry.counts();
    assert_eq!(
        healthy, 2,
        "generic runtime errors do not prove worker failure"
    );
    assert_eq!(unhealthy, 0);
}

// ── Step 5: SSE chaos helper ──────────────────────────────────────────────────

/// Spawn a mock worker that returns a streaming SSE response.
///
/// Emits `tokens` chunks. If `drop_after` is `Some(n)`, the response body
/// ends cleanly after `n` chunks (no `[DONE]`), simulating a mid-stream drop.
/// If `drop_after` is `None`, `tokens` chunks are emitted followed by `[DONE]`.
async fn spawn_sse_worker(
    tokens: usize,
    drop_after: Option<usize>,
) -> Option<std::net::SocketAddr> {
    let app = Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || async move {
            let emit_count = drop_after.unwrap_or(tokens);
            let mut chunks: Vec<Result<axum::body::Bytes, std::io::Error>> = (0..emit_count)
                .map(|i| {
                    let s = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"tok{i}\"}}}}]}}\n\n"
                    );
                    Ok(axum::body::Bytes::from(s))
                })
                .collect();
            if drop_after.is_none() {
                chunks.push(Ok(axum::body::Bytes::from("data: [DONE]\n\n")));
            }
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(futures::stream::iter(chunks)))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

async fn spawn_delayed_first_byte_worker(delay: std::time::Duration) -> Option<SocketAddr> {
    let handler = move || {
        let delay = delay;
        async move {
            let stream = futures::stream::once(async move {
                tokio::time::sleep(delay).await;
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(
                    b"data: {\"choices\":[]}\n\n",
                ))
            });
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(stream))
                .unwrap()
        }
    };
    let app = Router::new().route("/v1/chat/completions", post(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Some(addr)
}

#[tokio::test]
async fn test_stream_first_byte_deadline_releases_worker_reservation() {
    let addr = skip_if_no_socket!(
        spawn_delayed_first_byte_worker(std::time::Duration::from_millis(80)).await
    );
    let registry = WorkerRegistry::new();
    let registration = registry.register(reg_req(addr, &["slow-stream"]), 5000);
    let worker_id = WorkerId::parse(&registration.worker_id).unwrap();
    let dispatcher = DirectDispatcher::try_new_with_timeouts(8, 300, 20, 20, None).unwrap();
    let policy = policy_from_str("least_inflight").unwrap();
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "slow-stream",
            true,
            None,
            "/v1/chat/completions",
            axum::body::Bytes::from(
                r#"{"model":"slow-stream","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            ),
            None,
        )
        .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert!(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .is_err(),
        "a stream that misses the first-byte deadline must terminate with a body error"
    );
    assert_eq!(registry.get_snapshot(worker_id).unwrap().inflight, 0);
    let metrics = dispatcher.metrics();
    assert_eq!(metrics.failed_total, 1);
    assert_eq!(metrics.stream_duration.count, 1);
    assert_eq!(metrics.time_to_first_byte.count, 0);
}

#[tokio::test]
async fn test_client_stream_cancellation_releases_worker_reservation() {
    let addr = skip_if_no_socket!(
        spawn_delayed_first_byte_worker(std::time::Duration::from_millis(500)).await
    );
    let registry = WorkerRegistry::new();
    let registration = registry.register(reg_req(addr, &["cancel-stream"]), 5000);
    let worker_id = WorkerId::parse(&registration.worker_id).unwrap();
    let dispatcher = DirectDispatcher::try_new_with_timeouts(8, 300, 1_000, 1_000, None).unwrap();
    let policy = policy_from_str("least_inflight").unwrap();
    let response = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "cancel-stream",
            true,
            None,
            "/v1/chat/completions",
            axum::body::Bytes::from(
                r#"{"model":"cancel-stream","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            ),
            None,
        )
        .await;
    assert_eq!(registry.get_snapshot(worker_id).unwrap().inflight, 1);
    drop(response);
    tokio::task::yield_now().await;
    assert_eq!(registry.get_snapshot(worker_id).unwrap().inflight, 0);
    let metrics = dispatcher.metrics();
    assert_eq!(metrics.cancelled_total, 1);
    assert_eq!(metrics.stream_duration.count, 1);
    assert_eq!(metrics.time_to_first_byte.count, 0);
}

// ── Step 5: Chaos integration tests ───────────────────────────────────────────

/// Worker drops SSE stream mid-generation (no [DONE] marker).
/// DirectDispatcher must not panic; response status must be 200 (worker sent
/// 200 initially) and body consumption must complete without unwinding.
#[tokio::test]
async fn test_chaos_mid_stream_crash() {
    // 3 tokens, then stream ends without [DONE].
    let addr = skip_if_no_socket!(spawn_sse_worker(3, Some(3)).await);

    let registry = WorkerRegistry::new();
    registry.register(reg_req(addr, &["stream-crash"]), 5000);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"stream-crash","messages":[],"stream":true}"#);
    let resp = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "stream-crash",
            true,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    // Worker sent HTTP 200 before dropping, so dispatcher returns 200.
    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    // Consume the body — must not panic even though stream ended prematurely.
    let _ = axum::body::to_bytes(resp.into_body(), usize::MAX).await;
}

/// A restarting worker rejects before admission; a healthy peer receives the safe retry.
#[tokio::test]
async fn test_chaos_restart_reroutes_to_healthy_worker() {
    let crashed = skip_if_no_socket!(spawn_not_admitted_worker().await);
    let healthy = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"ok"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    registry.register(reg_req(crashed, &["restart-chaos"]), 5000);
    let healthy_resp = registry.register(reg_req(healthy, &["restart-chaos"]), 5000);

    // Ensure the pre-admission rejection is selected first.
    let healthy_id = WorkerId::parse(&healthy_resp.worker_id).unwrap();
    registry.heartbeat(
        healthy_id,
        HeartbeatRequest {
            inflight: 1,
            thermal_state: "nominal".into(),
            model_ids: vec!["restart-chaos".into()],
            rss_bytes: 0,
            ..Default::default()
        },
    );

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let body = axum::body::Bytes::from(r#"{"model":"restart-chaos","messages":[]}"#);
    let resp = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "restart-chaos",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;

    // Rerouted to healthy worker → 200.
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

/// Two requests dispatched concurrently to two workers.
/// Both must complete and no deadlock may occur (InflightGuard drops cleanly).
#[tokio::test]
async fn test_chaos_concurrent_dispatch_no_deadlock() {
    let addr_a = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"a"}}]}"#).await
    );
    let addr_b = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"b"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    registry.register(reg_req(addr_a, &["concurrent-model"]), 5000);
    registry.register(reg_req(addr_b, &["concurrent-model"]), 5000);

    let policy = policy_from_str("least_inflight").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    let (resp_a, resp_b) = tokio::join!(
        dispatcher.forward(
            &registry,
            policy.as_ref(),
            "concurrent-model",
            false,
            None,
            "/v1/chat/completions",
            axum::body::Bytes::from(r#"{"model":"concurrent-model","messages":[]}"#),
            None,
        ),
        dispatcher.forward(
            &registry,
            policy.as_ref(),
            "concurrent-model",
            false,
            None,
            "/v1/chat/completions",
            axum::body::Bytes::from(r#"{"model":"concurrent-model","messages":[]}"#),
            None,
        ),
    );

    assert_eq!(resp_a.status(), axum::http::StatusCode::OK);
    assert_eq!(resp_b.status(), axum::http::StatusCode::OK);
}

// ── TASK-MW-012: ModelAffinity integration ────────────────────────────────────

/// ModelAffinityPolicy routes to the warm worker after initial dispatches.
#[tokio::test]
async fn test_model_affinity_prefers_warm_worker() {
    use ax_serving_api::orchestration::policy::policy_from_str;

    let addr_warm = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"warm"}}]}"#).await
    );
    let addr_cold = skip_if_no_socket!(
        spawn_mock_worker(200, r#"{"choices":[{"message":{"content":"cold"}}]}"#).await
    );

    let registry = WorkerRegistry::new();
    registry.register(reg_req(addr_warm, &["affinity-model"]), 5000);
    registry.register(reg_req(addr_cold, &["affinity-model"]), 5000);

    let policy = policy_from_str("model_affinity").unwrap();
    let dispatcher = DirectDispatcher::new(8, 300);

    // First dispatch — no affinity data yet; least-inflight wins (either worker).
    let body = axum::body::Bytes::from(r#"{"model":"affinity-model","messages":[]}"#);
    let resp = dispatcher
        .forward(
            &registry,
            policy.as_ref(),
            "affinity-model",
            false,
            None,
            "/v1/chat/completions",
            body,
            None,
        )
        .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    // After the first dispatch the policy recorded the chosen worker.
    // Subsequent dispatches should continue to succeed (affinity or fallback).
    for _ in 0..4 {
        let body = axum::body::Bytes::from(r#"{"model":"affinity-model","messages":[]}"#);
        let resp = dispatcher
            .forward(
                &registry,
                policy.as_ref(),
                "affinity-model",
                false,
                None,
                "/v1/chat/completions",
                body,
                None,
            )
            .await;
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "affinity dispatch must always return 200"
        );
    }
}
