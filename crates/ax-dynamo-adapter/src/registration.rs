//! Protocol-v1.1 registration, lease heartbeat, and drain for a Dynamo domain.

use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result};
use ax_serving_protocol::{
    AgentDescriptor, DrainDirective, EndpointScope, ExecutionDomainDescriptor, HardwareDescriptor,
    HeartbeatRequest, HeartbeatResponse, LeaseToken, ProtocolCapability, ProtocolDescriptor,
    RegisterWorkerRequest, RegisterWorkerResponse, RegistrationId, RuntimeDescriptor,
    WorkerDescriptor, WorkerInstanceId,
};
use tokio::sync::RwLock;

use crate::config::DynamoAdapterConfig;
use crate::domain_observer::{DomainState, ObservationSnapshot};
use crate::manifest::ValidatedManifest;

const CONTROL_PLANE_REQUEST_TIMEOUT_SECS: u64 = 10;

#[derive(Clone, Debug)]
pub struct AdapterSession {
    pub registration_id: RegistrationId,
    pub lease_token: LeaseToken,
    pub instance_id: WorkerInstanceId,
    pub heartbeat_interval_ms: u64,
    pub sequence: Arc<AtomicU64>,
}

pub type SharedSession = Arc<RwLock<Option<AdapterSession>>>;

pub async fn register(
    client: &reqwest::Client,
    config: &DynamoAdapterConfig,
    manifest: &ValidatedManifest,
    snapshot: &ObservationSnapshot,
    instance_id: WorkerInstanceId,
) -> Result<AdapterSession> {
    let body = registration_body(config, manifest, snapshot, instance_id)?;
    let request = with_control_token(
        client
            .post(format!(
                "{}/internal/workers/register",
                config.control_plane_url
            ))
            .timeout(control_timeout())
            .json(&body),
        config.worker_token.as_ref(),
    );
    let response = request
        .send()
        .await
        .context("Dynamo adapter registration request failed")?
        .error_for_status()
        .context("Dynamo adapter registration was rejected")?
        .json::<RegisterWorkerResponse>()
        .await
        .context("failed to parse Dynamo adapter registration response")?;
    Ok(AdapterSession {
        registration_id: response.registration_id,
        lease_token: response.lease_token,
        instance_id,
        heartbeat_interval_ms: response.heartbeat_interval_ms.clamp(1_000, 300_000),
        sequence: Arc::new(AtomicU64::new(0)),
    })
}

pub fn registration_body(
    config: &DynamoAdapterConfig,
    manifest: &ValidatedManifest,
    snapshot: &ObservationSnapshot,
    instance_id: WorkerInstanceId,
) -> Result<RegisterWorkerRequest> {
    let descriptor = ExecutionDomainDescriptor {
        id: config.domain_id.clone(),
        kind: config.domain_kind,
        endpoint_scope: EndpointScope::Domain,
        execution_owner: "dynamo".into(),
        qualification: config.qualification,
        pool_id: config.pool_id.clone(),
        trust_domain: config.trust_domain.clone(),
        hardware_class: config.hardware_class.clone(),
        architecture: normalize_architecture(&manifest.manifest.platform.arch).into(),
        compatibility_manifest: Some(manifest.digest.clone()),
        labels: BTreeMap::from([
            (
                "dynamo_release".into(),
                manifest.manifest.dynamo.tag.clone(),
            ),
            ("backend".into(), manifest.manifest.backend.kind.clone()),
        ]),
    };
    descriptor
        .validate()
        .context("invalid Dynamo execution-domain descriptor")?;

    let body = RegisterWorkerRequest {
        protocol: ProtocolDescriptor::current(protocol_capabilities()),
        agent: AgentDescriptor {
            name: "ax-dynamo-adapter".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_sha: option_env!("AXS_BUILD_SHA").map(ToOwned::to_owned),
        },
        worker: WorkerDescriptor {
            id: config.worker_id.clone(),
            instance_id,
            advertise_url: config.advertised_url.clone(),
            pool_id: config.pool_id.clone(),
            trust_domain: config.trust_domain.clone(),
            labels: BTreeMap::from([("endpoint_scope".into(), "domain".into())]),
        },
        runtime: RuntimeDescriptor {
            kind: "dynamo".into(),
            version: manifest.manifest.dynamo.tag.clone(),
            api: "openai-v1".into(),
        },
        hardware: HardwareDescriptor {
            platform: manifest.manifest.platform.os.clone(),
            accelerator: "nvidia-gpu-domain".into(),
            device_count: 0,
            memory_bytes: None,
            hardware_class: Some(config.hardware_class.clone()),
        },
        domain: Some(descriptor),
        domain_observation: Some(snapshot.domain.clone()),
        observation: snapshot.runtime.clone(),
    };
    body.observation
        .validate()
        .context("invalid initial Dynamo runtime observation")?;
    body.validate_domain_contract()
        .context("invalid initial Dynamo domain contract")?;
    Ok(body)
}

pub async fn heartbeat_loop(
    control_client: reqwest::Client,
    frontend_client: reqwest::Client,
    config: DynamoAdapterConfig,
    manifest: ValidatedManifest,
    state: DomainState,
    session: SharedSession,
    instance_id: WorkerInstanceId,
) {
    loop {
        let current = session.read().await.clone();
        let Some(current) = current else {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let snapshot = state.snapshot().await;
            match register(&control_client, &config, &manifest, &snapshot, instance_id).await {
                Ok(new_session) => *session.write().await = Some(new_session),
                Err(error) => tracing::warn!(%error, "Dynamo adapter re-registration failed"),
            }
            continue;
        };

        let snapshot = match state.observe(&frontend_client, &config, &manifest).await {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tracing::warn!(%error, "Dynamo observation could not be encoded");
                state.snapshot().await
            }
        };
        let sequence = current.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        let heartbeat = HeartbeatRequest {
            registration_id: current.registration_id,
            instance_id: current.instance_id,
            sequence,
            observed_at: snapshot.runtime.observed_at,
            runtime: snapshot.runtime.runtime.clone(),
            inventory_generation: snapshot.runtime.inventory_generation,
            models: Some(snapshot.runtime.models.clone()),
            capacity: snapshot.runtime.capacity.clone(),
            domain_observation: Some(snapshot.domain),
            deployment_jobs: Vec::new(),
        };
        let request = with_control_token(
            control_client
                .post(format!(
                    "{}/internal/workers/{}/heartbeat",
                    config.control_plane_url, config.worker_id
                ))
                .timeout(control_timeout())
                .header("x-ax-lease-token", current.lease_token.expose())
                .json(&heartbeat),
            config.worker_token.as_ref(),
        );
        match request.send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<HeartbeatResponse>().await {
                    Ok(directive) if directive.reregister => {
                        tracing::warn!("gateway requested Dynamo adapter re-registration");
                        *session.write().await = None;
                    }
                    Ok(directive) if directive.drain == DrainDirective::Begin => {
                        state.begin_drain();
                        tracing::info!("gateway requested Dynamo domain drain");
                        if state.inflight.load(Ordering::Acquire) == 0
                            && let Err(error) =
                                drain_complete(&control_client, &config, &current).await
                        {
                            tracing::warn!(%error, "Dynamo adapter drain completion failed");
                        }
                    }
                    Ok(directive) if directive.drain == DrainDirective::Complete => {
                        state.begin_drain();
                        *session.write().await = None;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(%error, "invalid Dynamo adapter heartbeat response");
                    }
                }
            }
            Ok(response) if matches!(response.status().as_u16(), 404 | 409 | 410) => {
                tracing::warn!(
                    status = %response.status(),
                    "Dynamo adapter lease was fenced; re-registering"
                );
                *session.write().await = None;
            }
            Ok(response) => {
                tracing::warn!(status = %response.status(), "Dynamo adapter heartbeat rejected");
            }
            Err(error) => {
                tracing::warn!(%error, "Dynamo adapter heartbeat failed");
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(
            current
                .heartbeat_interval_ms
                .min(config.probe_interval_ms.max(1_000)),
        ))
        .await;
    }
}

pub async fn begin_drain(
    client: &reqwest::Client,
    config: &DynamoAdapterConfig,
    session: &AdapterSession,
) -> Result<()> {
    let request = with_control_token(
        client
            .post(format!(
                "{}/internal/workers/{}/drain",
                config.control_plane_url, config.worker_id
            ))
            .timeout(control_timeout())
            .header("x-ax-lease-token", session.lease_token.expose()),
        config.worker_token.as_ref(),
    );
    request
        .send()
        .await
        .context("Dynamo adapter drain request failed")?
        .error_for_status()
        .context("Dynamo adapter drain request rejected")?;
    Ok(())
}

pub async fn drain_complete(
    client: &reqwest::Client,
    config: &DynamoAdapterConfig,
    session: &AdapterSession,
) -> Result<()> {
    let request = with_control_token(
        client
            .post(format!(
                "{}/internal/workers/{}/drain-complete",
                config.control_plane_url, config.worker_id
            ))
            .timeout(control_timeout())
            .header("x-ax-lease-token", session.lease_token.expose()),
        config.worker_token.as_ref(),
    );
    request
        .send()
        .await
        .context("Dynamo adapter drain-complete request failed")?
        .error_for_status()
        .context("Dynamo adapter drain-complete request rejected")?;
    Ok(())
}

fn protocol_capabilities() -> Vec<ProtocolCapability> {
    [
        ProtocolCapability::CONTROL_DRAIN,
        ProtocolCapability::CONTROL_EXECUTION_DOMAIN,
        ProtocolCapability::CONTROL_INVENTORY_DELTA,
        ProtocolCapability::DISPATCH_CANCEL,
        ProtocolCapability::DISPATCH_TYPED_ADMISSION,
        ProtocolCapability::TELEMETRY_CAPACITY,
        ProtocolCapability::TELEMETRY_DOMAIN_CAPACITY,
    ]
    .into_iter()
    .map(|value| ProtocolCapability::new(value).expect("static protocol capability"))
    .collect()
}

fn with_control_token(
    request: reqwest::RequestBuilder,
    token: Option<&String>,
) -> reqwest::RequestBuilder {
    match token {
        Some(token) => request.header("X-Internal-Token", token),
        None => request,
    }
}

fn control_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(CONTROL_PLANE_REQUEST_TIMEOUT_SECS)
}

fn normalize_architecture(raw: &str) -> &str {
    match raw {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize},
    };

    use ax_serving_protocol::{
        CompatibilityManifestDigest, Digest, DomainId, ExecutionDomainKind, PoolId,
        QualificationState, TrustDomainId, WorkerId, WorkerInstanceId,
    };
    use time::OffsetDateTime;

    use crate::config::DynamoAdapterConfig;
    use crate::domain_observer::{DomainState, ObservationSnapshot};
    use crate::manifest::{
        BackendRelease, DynamoCompatibilityManifest, DynamoRelease, PlatformRelease,
        ValidatedManifest,
    };

    use super::registration_body;

    fn fixtures() -> (DynamoAdapterConfig, ValidatedManifest, ObservationSnapshot) {
        let config = DynamoAdapterConfig {
            control_plane_url: "http://127.0.0.1:19090".into(),
            worker_token: None,
            dispatch_token: None,
            frontend_url: "http://127.0.0.1:8000".into(),
            dynamo_api_key: None,
            manifest_path: PathBuf::from("manifest.json"),
            domain_id: DomainId::new("nvidia-pc-main").unwrap(),
            domain_kind: ExecutionDomainKind::NvidiaDynamoPc,
            worker_id: WorkerId::new("dynamo-nvidia-pc-main").unwrap(),
            pool_id: PoolId::new("nvidia-pc").unwrap(),
            trust_domain: TrustDomainId::new("private-dc").unwrap(),
            qualification: QualificationState::Experimental,
            hardware_class: "nvidia-pc-cuda".into(),
            listen_addr: "127.0.0.1:18082".parse().unwrap(),
            advertised_url: "http://127.0.0.1:18082".into(),
            probe_interval_ms: 5_000,
            drain_timeout_secs: 300,
            max_inflight: 64,
            tls_profile: "loopback_dev".into(),
            allow_no_auth: true,
        };
        let raw_digest =
            |value: char| Digest::new(format!("sha256:{}", value.to_string().repeat(64))).unwrap();
        let manifest = ValidatedManifest {
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
                graph_config_digest: raw_digest('b'),
                model_certifications: vec![raw_digest('c')],
                issued_at: OffsetDateTime::UNIX_EPOCH,
                evidence: "sha256:evidence".into(),
            },
            digest: CompatibilityManifestDigest::new(format!("sha256:{}", "d".repeat(64))).unwrap(),
        };
        let state = DomainState::new(&manifest, 64);
        let snapshot = futures_lite_snapshot(&state);
        (config, manifest, snapshot)
    }

    fn futures_lite_snapshot(state: &DomainState) -> ObservationSnapshot {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(state.snapshot())
    }

    #[test]
    fn registration_represents_one_domain_not_a_gpu_worker() {
        let (config, manifest, snapshot) = fixtures();
        let body = registration_body(&config, &manifest, &snapshot, WorkerInstanceId::new())
            .expect("registration contract");
        let domain = body.domain.unwrap();
        assert_eq!(domain.kind, ExecutionDomainKind::NvidiaDynamoPc);
        assert_eq!(domain.endpoint_scope.as_str(), "domain");
        assert_eq!(domain.execution_owner, "dynamo");
        assert_eq!(body.hardware.device_count, 0);
    }

    #[test]
    fn domain_state_uses_shared_admission_counters() {
        let (_, manifest, _) = fixtures();
        let state = DomainState::new(&manifest, 64);
        let _: Arc<AtomicUsize> = state.inflight;
        let _: Arc<AtomicBool> = state.draining;
    }
}
