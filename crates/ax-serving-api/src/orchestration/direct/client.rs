//! Reqwest client construction, pool knobs, and worker URL helpers.

use axum::http::HeaderValue;
use reqwest::Client;
use tracing::warn;

/// TCP connect timeout for the dispatcher's reqwest client.
/// Short enough to fail fast on unreachable workers without blocking the queue.
pub(super) const DISPATCHER_CONNECT_TIMEOUT_SECS: u64 = 5;
/// Default pool size and request timeout matching serving.example.yaml defaults.
pub(super) const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
pub(super) const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

/// Build a connection-pooled reqwest client, falling back to `Client::new` on error.
pub(super) fn build_dispatcher_client(
    pool_max_idle_per_host: usize,
    request_timeout_secs: u64,
) -> Client {
    match Client::builder()
        .pool_max_idle_per_host(pool_max_idle_per_host)
        .connect_timeout(std::time::Duration::from_secs(
            DISPATCHER_CONNECT_TIMEOUT_SECS,
        ))
        .timeout(std::time::Duration::from_secs(request_timeout_secs))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            warn!(
                %err,
                pool_max_idle_per_host,
                request_timeout_secs,
                "failed to build tuned reqwest client; falling back to default client"
            );
            Client::new()
        }
    }
}

pub(super) fn parse_dispatch_token(token: Option<&str>) -> anyhow::Result<Option<HeaderValue>> {
    token
        .map(|token| {
            let mut value = HeaderValue::from_str(token)
                .map_err(|_| anyhow::anyhow!("AXS_DISPATCH_TOKEN is not a valid HTTP header"))?;
            value.set_sensitive(true);
            Ok::<_, anyhow::Error>(value)
        })
        .transpose()
}

pub(super) fn worker_url(
    addr: &crate::orchestration::worker_endpoint::WorkerEndpoint,
    path: &str,
) -> String {
    addr.join_path(path)
}
