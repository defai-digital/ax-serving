pub mod agent;
pub mod config;
pub mod proxy;
pub mod sglang;

use std::future::{Future, IntoFuture};
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use tokio::sync::oneshot;

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
    // Reject direct dispatch immediately, then stop heartbeats before asking
    // the control plane to drain. This closes the window where shutdown has
    // begun but a request can still enter through the agent listener.
    runtime.draining.store(true, Ordering::Release);
    heartbeat_abort.abort();
    if let Err(e) = agent::drain(&cp_client, &config, &runtime).await {
        tracing::warn!(%e, "drain request failed");
    }
}

async fn wait_for_shutdown_signal() {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerExit {
    ListenerClosed,
    GracefulShutdown,
    ForcedShutdown,
}

/// Serve until an explicit shutdown signal or an HTTP listener failure.
///
/// The graceful-shutdown deadline intentionally starts only after
/// `shutdown_signal` resolves. Applying it to the whole server future would
/// turn an operator drain bound into a process lifetime limit.
async fn serve_until_shutdown<S, D>(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    shutdown_signal: S,
    shutdown_drain: D,
    shutdown_timeout: Duration,
) -> Result<ServerExit>
where
    S: Future<Output = ()> + Send,
    D: Future<Output = ()> + Send,
{
    let (graceful_tx, graceful_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = graceful_rx.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::pin!(shutdown_signal);

    tokio::select! {
        result = &mut server => {
            result.context("runtime-node HTTP server failed")?;
            return Ok(ServerExit::ListenerClosed);
        }
        _ = &mut shutdown_signal => {}
    }

    let shutdown = async {
        shutdown_drain.await;
        let _ = graceful_tx.send(());
        (&mut server).await
    };
    match tokio::time::timeout(shutdown_timeout, shutdown).await {
        Ok(result) => {
            result.context("runtime-node HTTP server failed during graceful shutdown")?;
            Ok(ServerExit::GracefulShutdown)
        }
        Err(_) => {
            tracing::warn!(
                timeout_secs = shutdown_timeout.as_secs(),
                "graceful shutdown deadline exceeded; forcing runtime-node exit"
            );
            Ok(ServerExit::ForcedShutdown)
        }
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

    sglang::wait_for_runtime(
        &runtime_client,
        &config.runtime_url,
        &config.runtime_health_path,
    )
    .await?;

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
    let app = proxy::router(
        &config,
        proxy_client,
        runtime.inflight.clone(),
        runtime.draining.clone(),
    );
    let listener = tokio::net::TcpListener::bind(config.listen_addr).await?;
    tracing::info!(addr = %config.listen_addr, "runtime-node agent listening");

    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout_secs.unwrap_or(30).max(1));
    let shutdown_heartbeat_abort = heartbeat_task.abort_handle();
    let shutdown_client = control_client.clone();
    let shutdown_config = config.clone();
    let shutdown_runtime = runtime.clone();
    let shutdown_drain = async move {
        tracing::info!(
            timeout_secs = shutdown_timeout.as_secs(),
            "shutdown signal received, draining connections"
        );
        begin_shutdown_drain(
            shutdown_heartbeat_abort,
            shutdown_client,
            shutdown_config,
            shutdown_runtime,
        )
        .await;
    };

    let serve_result = serve_until_shutdown(
        listener,
        app,
        wait_for_shutdown_signal(),
        shutdown_drain,
        shutdown_timeout,
    )
    .await;

    // A listener failure bypasses the signal path. It still must stop
    // heartbeats, reject new dispatch, and remove the worker registration.
    heartbeat_task.abort();
    if !runtime.draining.load(Ordering::Acquire) {
        begin_shutdown_drain(
            heartbeat_task.abort_handle(),
            control_client.clone(),
            config.clone(),
            runtime.clone(),
        )
        .await;
    }
    if let Err(e) = agent::drain_complete(&control_client, &config, &runtime).await {
        tracing::warn!(%e, "drain-complete request failed");
    }

    serve_result?;
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
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Duration;

    use anyhow::Result;
    use ax_serving_protocol::{
        LeaseToken, RegistrationId, WorkerId as ProtocolWorkerId, WorkerInstanceId,
    };
    use axum::{
        Router,
        extract::{Path, State},
        routing::{get, post},
    };
    use tokio::sync::{Mutex, Notify, oneshot};

    use super::{ServerExit, begin_shutdown_drain, runtime_default_headers, serve_until_shutdown};
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
            execution_domain: None,
            friendly_name: None,
            chip_model: None,
            shutdown_timeout_secs: None,
            max_context: None,
            embedding: None,
            vision: None,
            runtime_health_path: "/health".into(),
            telemetry_metrics: Default::default(),
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
    async fn server_lifetime_is_not_limited_by_shutdown_timeout() -> Result<()> {
        let app = Router::new().route("/health", get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let drain_called = Arc::new(AtomicBool::new(false));
        let drain_observation = Arc::clone(&drain_called);

        let server = tokio::spawn(serve_until_shutdown(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
            },
            async move {
                drain_observation.store(true, Ordering::Release);
            },
            Duration::from_millis(50),
        ));

        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !server.is_finished(),
            "shutdown timeout must not act as a process lifetime"
        );
        let response = reqwest::get(format!("http://{addr}/health")).await?;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(!drain_called.load(Ordering::Acquire));

        shutdown_tx
            .send(())
            .expect("server should still be running");
        let exit = tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("server did not shut down")??;
        assert_eq!(exit, ServerExit::GracefulShutdown);
        assert!(drain_called.load(Ordering::Acquire));

        Ok(())
    }

    #[tokio::test]
    async fn shutdown_timeout_forces_a_stuck_request_after_signal() -> Result<()> {
        async fn hang(State(started): State<Arc<Notify>>) -> Result<String, Infallible> {
            started.notify_one();
            std::future::pending().await
        }

        let request_started = Arc::new(Notify::new());
        let app = Router::new()
            .route("/hang", get(hang))
            .with_state(Arc::clone(&request_started));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let server = tokio::spawn(serve_until_shutdown(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
            },
            async {},
            Duration::from_millis(50),
        ));
        let request =
            tokio::spawn(async move { reqwest::get(format!("http://{addr}/hang")).await });
        tokio::time::timeout(Duration::from_secs(1), request_started.notified())
            .await
            .expect("request did not reach the server");

        shutdown_tx
            .send(())
            .expect("server should still be running");
        let exit = tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("forced shutdown exceeded its deadline")??;
        assert_eq!(exit, ServerExit::ForcedShutdown);
        request.abort();

        Ok(())
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
            runtime.clone(),
        )
        .await;

        assert!(heartbeat_task.await.unwrap_err().is_cancelled());
        assert!(runtime.draining.load(Ordering::Acquire));
        assert_eq!(drains.lock().await.as_slice(), ["worker-1"]);
        server.abort();

        Ok(())
    }
}
