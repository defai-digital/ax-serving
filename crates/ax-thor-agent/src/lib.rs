pub mod agent;
pub mod config;
pub mod proxy;
pub mod sglang;

use anyhow::Result;

use agent::SharedRuntime;
use config::ThorConfig;

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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("AXS_LOG")
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let config = ThorConfig::from_env()?;

    // BUG-055: use separate clients so streaming proxy connections are never
    // killed by the global 300s timeout that is appropriate for control-plane calls.
    let cp_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    // Proxy client has no global timeout; per-request timeouts are set explicitly
    // by control-plane helpers in agent.rs.
    let proxy_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()?;

    sglang::wait_for_runtime(&cp_client, &config.runtime_url).await?;

    let runtime = SharedRuntime::new();
    let registration = agent::register(&cp_client, &config).await?;
    {
        *runtime.models.write().await = registration.models;
        *runtime.session.write().await = Some(registration.session);
    }

    let heartbeat_runtime = runtime.clone();
    let heartbeat_client = cp_client.clone();
    let heartbeat_config = config.clone();
    let heartbeat_task = tokio::spawn(async move {
        agent::heartbeat_loop(heartbeat_client, heartbeat_config, heartbeat_runtime).await;
    });
    let heartbeat_abort = heartbeat_task.abort_handle();

    let app = proxy::router(&config, proxy_client, runtime.inflight.clone());
    let listener = tokio::net::TcpListener::bind(config.listen_addr).await?;
    tracing::info!(addr = %config.listen_addr, "runtime-node agent listening");

    let server_shutdown_secs = config.shutdown_timeout_secs.unwrap_or(30).max(1);
    let shutdown_client = cp_client.clone();
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
    if let Err(e) = agent::drain_complete(&cp_client, &config, &runtime).await {
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

    use anyhow::Result;
    use axum::{
        Router,
        extract::{Path, State},
        routing::post,
    };
    use tokio::sync::Mutex;

    use super::begin_shutdown_drain;
    use crate::agent::{SharedRuntime, WorkerSession};
    use crate::config::ThorConfig;

    fn test_config(control_plane_url: String) -> ThorConfig {
        ThorConfig {
            control_plane_url,
            worker_token: None,
            runtime_url: "http://127.0.0.1:8000".into(),
            runtime: "vllm".into(),
            listen_addr: "127.0.0.1:18081".parse().unwrap(),
            advertised_addr: "127.0.0.1:18081".parse().unwrap(),
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
        }
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
            worker_id: "worker-1".into(),
            heartbeat_interval_ms: 5_000,
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
