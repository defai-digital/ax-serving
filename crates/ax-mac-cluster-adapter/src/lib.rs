//! Runtime-SDK-free adapter and gang coordinator for one Mac AX Engine cluster.

pub mod config;
pub mod coordinator;
pub mod evidence;
pub mod manifest;
pub mod planner;
pub mod reconcile;
pub mod registration;
pub mod replicas;

use std::sync::{Arc, atomic::Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use ax_serving_adapter_core::proxy::{self, ProxyConfig};
use ax_serving_protocol::WorkerInstanceId;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

use config::MacClusterConfig;
use coordinator::ClusterCoordinator;
use manifest::ValidatedManifest;
use registration::SharedSession;

pub async fn run_from_env() -> Result<()> {
    let _telemetry =
        ax_serving_observability::init("ax-mac-cluster-adapter", tracing::Level::INFO)?;
    let config = MacClusterConfig::from_env()?;
    let manifest = ValidatedManifest::load(&config.manifest_path)?;
    if manifest.manifest.cluster_id != config.domain_id {
        anyhow::bail!("manifest cluster_id does not match AXS_MAC_CLUSTER_DOMAIN_ID");
    }
    tracing::info!(
        domain_id = %config.domain_id,
        generation = manifest.manifest.generation,
        manifest_digest = %manifest.digest,
        ranks = manifest.manifest.ranks.len(),
        parallelism = ?manifest.manifest.parallelism.kind,
        "validated Mac cluster parallelism manifest"
    );

    let coordinator = ClusterCoordinator::new(
        manifest.clone(),
        config.max_inflight,
        Duration::from_millis(config.rank_stale_ms),
    );
    let control_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .build()
        .context("failed to construct AX control-plane client")?;
    let mut rank0_headers = HeaderMap::new();
    let mut rank0_authorization = HeaderValue::from_str(&format!("Bearer {}", config.rank0_token))
        .context("AXS_MAC_CLUSTER_RANK0_TOKEN is not a valid Authorization credential")?;
    rank0_authorization.set_sensitive(true);
    rank0_headers.insert(AUTHORIZATION, rank0_authorization);
    let proxy_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .default_headers(rank0_headers)
        .build()
        .context("failed to construct rank-0 proxy client")?;

    let snapshot = coordinator.snapshot().await;
    let instance_id = WorkerInstanceId::new();
    let initial =
        registration::register(&control_client, &config, &manifest, &snapshot, instance_id).await?;
    let session: SharedSession = Arc::new(tokio::sync::RwLock::new(Some(initial)));
    let heartbeat = tokio::spawn(registration::heartbeat_loop(
        control_client.clone(),
        config.clone(),
        manifest,
        coordinator.clone(),
        Arc::clone(&session),
        instance_id,
    ));
    let heartbeat_abort = heartbeat.abort_handle();

    let proxy_router = proxy::router(
        ProxyConfig {
            upstream_url: config.rank0_url.clone(),
            upstream_health_path: "/health".into(),
            dispatch_token: config.dispatch_token.clone(),
            max_inflight: config.max_inflight,
            expected_domain_id: Some(config.domain_id.clone()),
            require_dispatch_identity: true,
        },
        proxy_client,
        Arc::clone(&coordinator.inflight),
        Arc::clone(&coordinator.draining),
        Some(Arc::clone(&coordinator.ready)),
    );
    let app = proxy_router.merge(coordinator::router(
        coordinator.clone(),
        config.rank_control_token.clone(),
    ));
    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| {
            format!(
                "failed to bind Mac cluster adapter at {}",
                config.listen_addr
            )
        })?;
    tracing::info!(
        domain_id = %config.domain_id,
        addr = %config.listen_addr,
        "Mac cluster adapter listening"
    );

    let shutdown_coordinator = coordinator.clone();
    let shutdown_control = control_client.clone();
    let shutdown_config = config.clone();
    let shutdown_session = Arc::clone(&session);
    let shutdown = async move {
        wait_for_shutdown_signal().await;
        heartbeat_abort.abort();
        shutdown_coordinator.begin_drain();
        if let Some(current) = shutdown_session.read().await.clone()
            && let Err(error) =
                registration::begin_drain(&shutdown_control, &shutdown_config, &current).await
        {
            tracing::warn!(%error, "failed to announce Mac cluster drain");
        }
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("Mac cluster adapter server failed")?;

    heartbeat.abort();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(config.drain_timeout_secs);
    while coordinator.inflight.load(Ordering::Acquire) > 0 && tokio::time::Instant::now() < deadline
    {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    if let Some(current) = session.read().await.clone()
        && let Err(error) = registration::drain_complete(&control_client, &config, &current).await
    {
        tracing::warn!(%error, "failed to announce Mac cluster drain completion");
    }
    Ok(())
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = signal.recv() => {}
                }
            }
            Err(error) => {
                tracing::warn!(%error, "failed to install SIGTERM handler");
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
