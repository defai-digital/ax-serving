//! Runtime-SDK-free adapter between one AX execution domain and one pinned
//! NVIDIA Dynamo frontend.

pub mod config;
pub mod domain_observer;
pub mod inventory;
pub mod manifest;
pub mod registration;

use std::sync::{Arc, atomic::Ordering};

use anyhow::{Context, Result};
use ax_serving_adapter_core::proxy::{self, ProxyConfig};
use ax_serving_protocol::WorkerInstanceId;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

use config::DynamoAdapterConfig;
use domain_observer::DomainState;
use manifest::ValidatedManifest;
use registration::SharedSession;

pub async fn run_from_env() -> Result<()> {
    let _telemetry = ax_serving_observability::init("ax-dynamo-adapter", tracing::Level::INFO)?;
    let config = DynamoAdapterConfig::from_env()?;
    let manifest = ValidatedManifest::load(&config.manifest_path, config.domain_kind)?;
    tracing::info!(
        domain_id = %config.domain_id,
        domain_kind = config.domain_kind.as_str(),
        manifest_digest = %manifest.digest,
        dynamo_release = %manifest.manifest.dynamo.tag,
        backend = %manifest.manifest.backend.kind,
        "validated Dynamo compatibility manifest"
    );

    let control_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to construct AX control-plane client")?;
    let frontend_headers = frontend_headers(&config)?;
    let frontend_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(10))
        .default_headers(frontend_headers.clone())
        .build()
        .context("failed to construct Dynamo observation client")?;
    // No global timeout: a valid generation stream can exceed the observation
    // timeout. Client disconnect drops the reqwest body and propagates
    // cancellation to the Dynamo frontend.
    let proxy_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .default_headers(frontend_headers)
        .build()
        .context("failed to construct Dynamo proxy client")?;

    let state = DomainState::new(&manifest, config.max_inflight);
    let initial = match state.observe(&frontend_client, &config, &manifest).await {
        Ok(snapshot) => snapshot,
        Err(error) => {
            tracing::warn!(%error, "initial Dynamo observation remained starting");
            state.snapshot().await
        }
    };
    let instance_id = WorkerInstanceId::new();
    let adapter_session =
        registration::register(&control_client, &config, &manifest, &initial, instance_id).await?;
    let session: SharedSession = Arc::new(tokio::sync::RwLock::new(Some(adapter_session)));

    let heartbeat = tokio::spawn(registration::heartbeat_loop(
        control_client.clone(),
        frontend_client,
        config.clone(),
        manifest,
        state.clone(),
        Arc::clone(&session),
        instance_id,
    ));
    let heartbeat_abort = heartbeat.abort_handle();

    let app = proxy::router(
        ProxyConfig {
            upstream_url: config.frontend_url.clone(),
            upstream_health_path: "/v1/models".into(),
            dispatch_token: config.dispatch_token.clone(),
            max_inflight: config.max_inflight,
            expected_domain_id: Some(config.domain_id.clone()),
            require_dispatch_identity: true,
        },
        proxy_client,
        Arc::clone(&state.inflight),
        Arc::clone(&state.draining),
        Some(Arc::clone(&state.ready)),
    );
    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| format!("failed to bind Dynamo adapter at {}", config.listen_addr))?;
    tracing::info!(
        domain_id = %config.domain_id,
        addr = %config.listen_addr,
        "Dynamo domain adapter listening"
    );

    let shutdown_state = state.clone();
    let shutdown_config = config.clone();
    let shutdown_control = control_client.clone();
    let shutdown_session = Arc::clone(&session);
    let shutdown = async move {
        wait_for_shutdown_signal().await;
        heartbeat_abort.abort();
        shutdown_state.begin_drain();
        if let Some(current) = shutdown_session.read().await.clone()
            && let Err(error) =
                registration::begin_drain(&shutdown_control, &shutdown_config, &current).await
        {
            tracing::warn!(%error, "failed to announce Dynamo adapter drain");
        }
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("Dynamo adapter server failed")?;

    heartbeat.abort();
    let drain_deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(config.drain_timeout_secs);
    while state.inflight.load(Ordering::Acquire) > 0 && tokio::time::Instant::now() < drain_deadline
    {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    if let Some(current) = session.read().await.clone()
        && let Err(error) = registration::drain_complete(&control_client, &config, &current).await
    {
        tracing::warn!(%error, "failed to announce Dynamo adapter drain completion");
    }
    Ok(())
}

fn frontend_headers(config: &DynamoAdapterConfig) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    if let Some(api_key) = &config.dynamo_api_key {
        let mut value = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .context("AXS_DYNAMO_API_KEY cannot be represented as an HTTP credential")?;
        value.set_sensitive(true);
        headers.insert(AUTHORIZATION, value);
    }
    Ok(headers)
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
