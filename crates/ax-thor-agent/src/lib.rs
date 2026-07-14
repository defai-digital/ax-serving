pub mod agent;
pub mod config;
pub mod proxy;
pub mod sglang;

use anyhow::Result;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

use agent::SharedRuntime;
use config::ThorConfig;

fn runtime_default_headers(config: &ThorConfig) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    if let Some(api_key) = &config.runtime_api_key {
        let mut value = HeaderValue::from_str(&format!("Bearer {api_key}"))?;
        value.set_sensitive(true);
        headers.insert(AUTHORIZATION, value);
    }
    Ok(headers)
}

async fn begin_shutdown_drain(
    heartbeat_abort: tokio::task::AbortHandle,
    cp_client: reqwest::Client,
    config: ThorConfig,
    runtime: SharedRuntime,
) {
    // Stop heartbeats before drain so the control plane does not re-admit this
    // runtime node while it is shutting down.
    heartbeat_abort.abort();
    if let Err(e) = agent::drain(&cp_client, &config, &runtime).await {
        tracing::warn!(%e, "drain request failed");
    }
}

pub async fn run_from_env() -> Result<()> {
    let _telemetry = ax_serving_observability::init("ax-runtime-agent", tracing::Level::INFO)?;

    let config = ThorConfig::from_env()?;

    // BUG-055: use separate clients so streaming proxy connections are never
    // killed by the global 300s timeout that is appropriate for control-plane calls.
    let control_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let runtime_headers = runtime_default_headers(&config)?;
    let runtime_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(10))
        .default_headers(runtime_headers.clone())
        .build()?;
    // Proxy client has no global timeout because generated streams can be long.
    // It carries only the runtime credential, never the worker-control token.
    let proxy_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .default_headers(runtime_headers)
        .build()?;

    sglang::wait_for_runtime(&runtime_client, &config.runtime_url).await?;

    let runtime = SharedRuntime::new();
    let registration = agent::register(
        &control_client,
        &runtime_client,
        &config,
        runtime.instance_id,
    )
    .await?;
    {
        *runtime.models.write().await = registration.models;
        *runtime.session.write().await = Some(registration.session);
    }

    let heartbeat_runtime = runtime.clone();
    let heartbeat_control_client = control_client.clone();
    let heartbeat_runtime_client = runtime_client.clone();
    let heartbeat_config = config.clone();
    let heartbeat_task = tokio::spawn(async move {
        agent::heartbeat_loop(
            heartbeat_control_client,
            heartbeat_runtime_client,
            heartbeat_config,
            heartbeat_runtime,
        )
        .await;
    });
    let heartbeat_abort = heartbeat_task.abort_handle();

    let app = proxy::router(
        &config,
        proxy_client,
        runtime.inflight.clone(),
        runtime.draining.clone(),
    );
    let listener = tokio::net::TcpListener::bind(config.listen_addr).await?;
    tracing::info!(addr = %config.listen_addr, "runtime-node agent listening");

    let server_shutdown_secs = config.shutdown_timeout_secs.unwrap_or(30).max(1);
    let shutdown_client = control_client.clone();
    let shutdown_config = config.clone();
    let shutdown_runtime = runtime.clone();
    let shutdown = async move {
        #[cfg(unix)]
        {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut sigterm) => {
                    tokio::select! {
                        _ = tokio::signal::ctrl_c() => {}
                        _ = sigterm.recv() => {}
                    }
                }
                Err(e) => {
                    tracing::warn!(%e, "failed to register SIGTERM handler; using Ctrl-C only");
                    let _ = tokio::signal::ctrl_c().await;
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        tracing::info!(
            "shutdown signal received, draining connections (timeout {server_shutdown_secs}s)"
        );
        begin_shutdown_drain(
            heartbeat_abort,
            shutdown_client,
            shutdown_config,
            shutdown_runtime,
        )
        .await;
    };

    // Wrap graceful shutdown with a hard deadline so stuck streams don't hang forever (BUG-054).
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown);
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(server_shutdown_secs + 5),
        server,
    )
    .await;

    // If the server exits without the shutdown signal path completing, make
    // sure the heartbeat task cannot outlive this runtime-node process.
    heartbeat_task.abort();
    let shutdown_deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(config.shutdown_timeout_secs.unwrap_or(30).max(1));
    while runtime.inflight.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        if tokio::time::Instant::now() > shutdown_deadline {
            tracing::warn!("shutdown timeout exceeded with inflight requests; forcing exit");
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    if let Err(e) = agent::drain_complete(&control_client, &config, &runtime).await {
        tracing::warn!(%e, "drain-complete request failed");
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Serialize process-global environment mutation across all unit tests in
    /// this crate. Rust 2024 marks env mutation unsafe because concurrent
    /// readers and writers can race across threads.
    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use anyhow::Result;
    use ax_serving_protocol::{
        LeaseToken, RegistrationId, WorkerId as ProtocolWorkerId, WorkerInstanceId,
    };
    use axum::{
        Router,
        extract::{Path, State},
        routing::post,
    };
    use tokio::sync::Mutex;

    use super::{begin_shutdown_drain, runtime_default_headers};
    use crate::agent::{SharedRuntime, WorkerSession};
    use crate::config::ThorConfig;

    fn test_config(control_plane_url: String) -> ThorConfig {
        ThorConfig {
            control_plane_url,
            worker_token: None,
            runtime_url: "http://127.0.0.1:8000".into(),
            runtime_api_key: None,
            dispatch_token: None,
            tls_profile: "loopback_dev".into(),
            runtime: "vllm".into(),
            runtime_version: "test".into(),
            worker_id: "worker-test".into(),
            trust_domain: "test".into(),
            listen_addr: "127.0.0.1:18081".parse().unwrap(),
            advertised_url: "http://127.0.0.1:18081".into(),
            max_inflight: 8,
            worker_pool: None,
            node_class: "thor".into(),
            hardware_class: "thor".into(),
            friendly_name: None,
            chip_model: None,
            shutdown_timeout_secs: None,
            max_context: None,
            embedding: None,
            vision: None,
            model_identity: Default::default(),
        }
    }

    #[test]
    fn runtime_credentials_are_scoped_to_runtime_clients() {
        let mut config = test_config("http://127.0.0.1:19090".into());
        config.runtime_api_key = Some("runtime-secret".into());
        let headers = runtime_default_headers(&config).unwrap();
        assert_eq!(
            headers[reqwest::header::AUTHORIZATION],
            "Bearer runtime-secret"
        );
        assert!(headers[reqwest::header::AUTHORIZATION].is_sensitive());
    }

    #[tokio::test]
    async fn shutdown_drain_aborts_heartbeat_before_control_plane_drain() -> Result<()> {
        async fn handle_drain(
            State(drains): State<Arc<Mutex<Vec<String>>>>,
            Path(worker_id): Path<String>,
        ) {
            drains.lock().await.push(worker_id);
        }

        let drains = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/internal/workers/{id}/drain", post(handle_drain))
            .with_state(Arc::clone(&drains));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let runtime = SharedRuntime::new();
        *runtime.session.write().await = Some(WorkerSession {
            worker_id: ProtocolWorkerId::new("worker-1").unwrap(),
            instance_id: WorkerInstanceId::new(),
            registration_id: RegistrationId::new(),
            lease_token: LeaseToken::new("0123456789abcdef").unwrap(),
            heartbeat_interval_ms: 5_000,
            sequence: Arc::new(AtomicU64::new(0)),
        });
        let heartbeat_task = tokio::spawn(async {
            std::future::pending::<()>().await;
        });
        let heartbeat_abort = heartbeat_task.abort_handle();

        begin_shutdown_drain(
            heartbeat_abort,
            reqwest::Client::new(),
            test_config(format!("http://{addr}")),
            runtime,
        )
        .await;

        assert!(heartbeat_task.await.unwrap_err().is_cancelled());
        assert_eq!(drains.lock().await.as_slice(), ["worker-1"]);
        server.abort();

        Ok(())
    }
}
