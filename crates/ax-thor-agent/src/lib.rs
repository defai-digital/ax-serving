pub mod agent;
pub mod config;
pub mod proxy;
pub mod sglang;

use anyhow::Result;

use agent::SharedRuntime;
use config::ThorConfig;

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

    let app = proxy::router(&config, proxy_client, runtime.inflight.clone());
    let listener = tokio::net::TcpListener::bind(config.listen_addr).await?;
    tracing::info!(addr = %config.listen_addr, "runtime-node agent listening");

    let server_shutdown_secs = config.shutdown_timeout_secs.unwrap_or(30).max(1);
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
    };

    // Wrap graceful shutdown with a hard deadline so stuck streams don't hang forever (BUG-054).
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown);
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(server_shutdown_secs + 5),
        server,
    )
    .await;

    // Abort heartbeat BEFORE drain to prevent re-registration race (BUG-023).
    heartbeat_task.abort();
    if let Err(e) = agent::drain(&cp_client, &config, &runtime).await {
        tracing::warn!(%e, "drain request failed");
    }
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
