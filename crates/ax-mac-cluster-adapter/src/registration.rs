//! Protocol-v1.2 registration, heartbeat, and drain for one Mac cluster.

use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result};
use ax_serving_protocol::{
    AgentDescriptor, DrainDirective, EndpointScope, ExecutionDomainDescriptor, ExecutionDomainKind,
    HardwareDescriptor, HeartbeatRequest, HeartbeatResponse, LeaseToken, ProtocolCapability,
    ProtocolDescriptor, RegisterWorkerRequest, RegisterWorkerResponse, RegistrationId,
    RuntimeDescriptor, WorkerDescriptor, WorkerInstanceId,
};
use tokio::sync::RwLock;

use crate::config::MacClusterConfig;
use crate::coordinator::{ClusterCoordinator, ObservationSnapshot};
use crate::manifest::ValidatedManifest;

const CONTROL_TIMEOUT_SECS: u64 = 10;

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
    config: &MacClusterConfig,
    manifest: &ValidatedManifest,
    snapshot: &ObservationSnapshot,
    instance_id: WorkerInstanceId,
) -> Result<AdapterSession> {
    let body = registration_body(config, manifest, snapshot, instance_id)?;
    let response = with_control_token(
        client
            .post(format!(
                "{}/internal/workers/register",
                config.control_plane_url
            ))
            .timeout(control_timeout())
            .json(&body),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("Mac cluster registration request failed")?
    .error_for_status()
    .context("Mac cluster registration was rejected")?
    .json::<RegisterWorkerResponse>()
    .await
    .context("failed to parse Mac cluster registration response")?;
    Ok(AdapterSession {
        registration_id: response.registration_id,
        lease_token: response.lease_token,
        instance_id,
        heartbeat_interval_ms: response.heartbeat_interval_ms.clamp(1_000, 300_000),
        sequence: Arc::new(AtomicU64::new(0)),
    })
}

pub fn registration_body(
    config: &MacClusterConfig,
    manifest: &ValidatedManifest,
    snapshot: &ObservationSnapshot,
    instance_id: WorkerInstanceId,
) -> Result<RegisterWorkerRequest> {
    let descriptor = ExecutionDomainDescriptor {
        id: config.domain_id.clone(),
        kind: ExecutionDomainKind::MacAxEngineCluster,
        endpoint_scope: EndpointScope::Domain,
        execution_owner: "ax_engine".into(),
        qualification: config.qualification,
        pool_id: config.pool_id.clone(),
        trust_domain: config.trust_domain.clone(),
        hardware_class: config.hardware_class.clone(),
        architecture: "arm64".into(),
        compatibility_manifest: Some(manifest.digest.clone()),
        labels: BTreeMap::from([
            (
                "parallelism".into(),
                format!("{:?}", manifest.manifest.parallelism.kind).to_ascii_lowercase(),
            ),
            ("transport".into(), manifest.manifest.transport.kind.clone()),
        ]),
    };
    descriptor
        .validate()
        .context("invalid Mac cluster execution-domain descriptor")?;

    let memory_bytes = manifest
        .manifest
        .ranks
        .iter()
        .try_fold(0_u64, |total, rank| {
            total.checked_add(rank.memory.certified_usable_bytes)
        });
    let body = RegisterWorkerRequest {
        protocol: ProtocolDescriptor::current(protocol_capabilities()),
        agent: AgentDescriptor {
            name: "ax-mac-cluster-adapter".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_sha: option_env!("AXS_BUILD_SHA").map(ToOwned::to_owned),
        },
        worker: WorkerDescriptor {
            id: config.worker_id.clone(),
            instance_id,
            advertise_url: config.advertised_url.clone(),
            pool_id: config.pool_id.clone(),
            trust_domain: config.trust_domain.clone(),
            labels: BTreeMap::from([
                ("endpoint_scope".into(), "domain".into()),
                (
                    "cluster_generation".into(),
                    manifest.manifest.generation.to_string(),
                ),
            ]),
        },
        runtime: RuntimeDescriptor {
            kind: "ax_engine".into(),
            version: manifest.manifest.runtime.ax_engine_version.clone(),
            api: "openai-v1".into(),
        },
        hardware: HardwareDescriptor {
            platform: "macos".into(),
            accelerator: "apple-silicon-cluster".into(),
            device_count: u32::try_from(manifest.manifest.ranks.len()).unwrap_or(u32::MAX),
            memory_bytes,
            hardware_class: Some(config.hardware_class.clone()),
        },
        domain: Some(descriptor),
        domain_observation: Some(snapshot.domain.clone()),
        observation: snapshot.runtime.clone(),
    };
    body.observation
        .validate()
        .context("invalid Mac cluster runtime observation")?;
    body.validate_domain_contract()
        .context("invalid Mac cluster domain contract")?;
    Ok(body)
}

pub async fn heartbeat_loop(
    control_client: reqwest::Client,
    config: MacClusterConfig,
    manifest: ValidatedManifest,
    coordinator: ClusterCoordinator,
    session: SharedSession,
    instance_id: WorkerInstanceId,
) {
    loop {
        let current = session.read().await.clone();
        let Some(current) = current else {
            let snapshot = coordinator.snapshot().await;
            match register(&control_client, &config, &manifest, &snapshot, instance_id).await {
                Ok(new_session) => *session.write().await = Some(new_session),
                Err(error) => tracing::warn!(%error, "Mac cluster re-registration failed"),
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            continue;
        };

        let snapshot = coordinator.snapshot().await;
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
        let response = with_control_token(
            control_client
                .post(format!(
                    "{}/internal/workers/{}/heartbeat",
                    config.control_plane_url, config.worker_id
                ))
                .timeout(control_timeout())
                .header("x-ax-lease-token", current.lease_token.expose())
                .json(&heartbeat),
            config.worker_token.as_ref(),
        )
        .send()
        .await;
        match response {
            Ok(response) if response.status().is_success() => {
                match response.json::<HeartbeatResponse>().await {
                    Ok(directive) if directive.reregister => {
                        *session.write().await = None;
                    }
                    Ok(directive) if directive.drain != DrainDirective::None => {
                        coordinator.begin_drain();
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(%error, "invalid Mac cluster heartbeat response");
                    }
                }
            }
            Ok(response) if matches!(response.status().as_u16(), 404 | 409 | 410) => {
                tracing::warn!(
                    status = %response.status(),
                    "Mac cluster adapter lease was fenced"
                );
                *session.write().await = None;
            }
            Ok(response) => {
                tracing::warn!(status = %response.status(), "Mac cluster heartbeat rejected");
            }
            Err(error) => tracing::warn!(%error, "Mac cluster heartbeat failed"),
        }
        tokio::time::sleep(std::time::Duration::from_millis(
            current.heartbeat_interval_ms,
        ))
        .await;
    }
}

pub async fn begin_drain(
    client: &reqwest::Client,
    config: &MacClusterConfig,
    session: &AdapterSession,
) -> Result<()> {
    with_control_token(
        client
            .post(format!(
                "{}/internal/workers/{}/drain",
                config.control_plane_url, config.worker_id
            ))
            .timeout(control_timeout())
            .header("x-ax-lease-token", session.lease_token.expose()),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("Mac cluster drain request failed")?
    .error_for_status()
    .context("Mac cluster drain was rejected")?;
    Ok(())
}

pub async fn drain_complete(
    client: &reqwest::Client,
    config: &MacClusterConfig,
    session: &AdapterSession,
) -> Result<()> {
    with_control_token(
        client
            .post(format!(
                "{}/internal/workers/{}/drain-complete",
                config.control_plane_url, config.worker_id
            ))
            .timeout(control_timeout())
            .header("x-ax-lease-token", session.lease_token.expose()),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("Mac cluster drain completion request failed")?
    .error_for_status()
    .context("Mac cluster drain completion was rejected")?;
    Ok(())
}

fn protocol_capabilities() -> Vec<ProtocolCapability> {
    [
        ProtocolCapability::CONTROL_DRAIN,
        ProtocolCapability::CONTROL_EXECUTION_DOMAIN,
        ProtocolCapability::CONTROL_MAC_CLUSTER,
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
        Some(token) => request.header("x-internal-token", token),
        None => request,
    }
}

fn control_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(CONTROL_TIMEOUT_SECS)
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::path::PathBuf;

    use ax_serving_protocol::{
        DomainId, ExecutionDomainKind, PoolId, QualificationState, TrustDomainId, WorkerId,
        WorkerInstanceId,
    };

    use super::*;

    fn fixture_manifest() -> ValidatedManifest {
        ValidatedManifest::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../config/mac-cluster-manifest.example.json"),
        )
        .unwrap()
    }

    fn config() -> MacClusterConfig {
        MacClusterConfig {
            control_plane_url: "http://127.0.0.1:19090".into(),
            worker_token: None,
            dispatch_token: Some("dispatch-token-value".into()),
            rank_control_token: "rank-control-token-value".into(),
            rank0_url: "http://127.0.0.1:18100".into(),
            manifest_path: PathBuf::from("unused"),
            domain_id: DomainId::new("mac-cluster-main").unwrap(),
            worker_id: WorkerId::new("mac-cluster-adapter").unwrap(),
            pool_id: PoolId::new("mac-cluster").unwrap(),
            trust_domain: TrustDomainId::new("private-lab").unwrap(),
            qualification: QualificationState::Experimental,
            hardware_class: "apple-silicon-cluster".into(),
            listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 18083),
            advertised_url: "http://127.0.0.1:18083".into(),
            max_inflight: 2,
            rank_stale_ms: 15_000,
            drain_timeout_secs: 300,
        }
    }

    #[tokio::test]
    async fn registration_is_one_v1_2_cluster_domain() {
        let manifest = fixture_manifest();
        let coordinator =
            ClusterCoordinator::new(manifest.clone(), 2, std::time::Duration::from_secs(30));
        let snapshot = coordinator.snapshot().await;
        let body =
            registration_body(&config(), &manifest, &snapshot, WorkerInstanceId::new()).unwrap();

        assert_eq!(
            body.domain.as_ref().unwrap().kind,
            ExecutionDomainKind::MacAxEngineCluster
        );
        assert!(
            body.protocol.capabilities.iter().any(|capability| {
                capability.as_str() == ProtocolCapability::CONTROL_MAC_CLUSTER
            })
        );
        body.validate_domain_contract().unwrap();
    }
}
