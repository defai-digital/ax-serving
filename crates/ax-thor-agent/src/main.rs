use anyhow::Result;

use ax_thor_agent::agent::{self, SharedRuntime};
use ax_thor_agent::config::ThorConfig;
use ax_thor_agent::{proxy, sglang};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("AXS_LOG")
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let config = ThorConfig::from_env()?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    sglang::wait_for_sglang(&client, &config.sglang_url).await?;

    let runtime = SharedRuntime::new();
    let registration = agent::register(&client, &config).await?;
    {
        *runtime.models.write().await = registration.models;
        *runtime.session.write().await = Some(registration.session);
    }

    let heartbeat_runtime = runtime.clone();
    let heartbeat_client = client.clone();
    let heartbeat_config = config.clone();
    let heartbeat_task = tokio::spawn(async move {
        agent::heartbeat_loop(heartbeat_client, heartbeat_config, heartbeat_runtime).await;
    });

    let app = proxy::router(&config, client.clone(), runtime.inflight.clone());
    let listener = tokio::net::TcpListener::bind(config.listen_addr).await?;
    tracing::info!(addr = %config.listen_addr, "ax-thor-agent listening");

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    let _ = agent::drain(&client, &config, &runtime).await;
    while runtime.inflight.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    heartbeat_task.abort();
    let _ = agent::drain_complete(&client, &config, &runtime).await;

    Ok(())
}
