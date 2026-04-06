use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Context, Result};
use serde_json::json;
use tokio::sync::RwLock;

fn current_rss_bytes() -> u64 {
    // Read RSS from /proc/self/status on macOS via sysctl (same approach as
    // ax-serving-engine metrics).  Returns 0 on failure rather than panicking.
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn getpid() -> i32;
        }
        // SAFETY: libc call with correct argument structure.
        let pid = unsafe { getpid() };
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output();
        if let Ok(out) = output
            && let Ok(s) = std::str::from_utf8(&out.stdout)
            && let Ok(kb) = s.trim().parse::<u64>()
        {
            return kb * 1024;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

use crate::config::ThorConfig;
use crate::sglang;

fn with_internal_token(
    req: reqwest::RequestBuilder,
    token: Option<&String>,
) -> reqwest::RequestBuilder {
    match token {
        Some(t) => req.header("X-Internal-Token", t),
        None => req,
    }
}

#[derive(Debug, Clone)]
pub struct WorkerSession {
    pub worker_id: String,
    pub heartbeat_interval_ms: u64,
}

#[derive(Debug, Clone)]
pub struct RegistrationState {
    pub session: WorkerSession,
    pub models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SharedRuntime {
    pub inflight: Arc<AtomicUsize>,
    pub session: Arc<RwLock<Option<WorkerSession>>>,
    pub models: Arc<RwLock<Vec<String>>>,
}

impl SharedRuntime {
    pub fn new() -> Self {
        Self {
            inflight: Arc::new(AtomicUsize::new(0)),
            session: Arc::new(RwLock::new(None)),
            models: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl Default for SharedRuntime {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn register(client: &reqwest::Client, config: &ThorConfig) -> Result<RegistrationState> {
    let models = sglang::get_loaded_models(client, &config.sglang_url).await?;
    let body = json!({
        "addr": config.advertised_addr.to_string(),
        "capabilities": {
            "llm": true,
            "embedding": true,
            "vision": false,
            "models": models,
            "max_context": serde_json::Value::Null
        },
        "backend": "sglang",
        "max_inflight": config.max_inflight,
        "friendly_name": config.friendly_name,
        "chip_model": config.chip_model,
        "worker_pool": config.worker_pool,
        "node_class": config.node_class,
    });

    let req = with_internal_token(
        client
            .post(format!(
                "{}/internal/workers/register",
                config.control_plane_url
            ))
            .timeout(std::time::Duration::from_secs(10))
            .json(&body),
        config.worker_token.as_ref(),
    );

    let response: serde_json::Value = req
        .send()
        .await
        .context("thor agent registration request failed")?
        .error_for_status()
        .context("thor agent registration rejected")?
        .json()
        .await
        .context("failed to parse thor agent registration response")?;

    let worker_id = response["worker_id"]
        .as_str()
        .context("registration response missing worker_id")?
        .to_string();
    let heartbeat_interval_ms = response["heartbeat_interval_ms"]
        .as_u64()
        .unwrap_or(5_000)
        .clamp(1_000, 300_000);
    Ok(RegistrationState {
        session: WorkerSession {
            worker_id,
            heartbeat_interval_ms,
        },
        models,
    })
}

pub async fn heartbeat_loop(client: reqwest::Client, config: ThorConfig, runtime: SharedRuntime) {
    loop {
        let session = {
            let guard = runtime.session.read().await;
            guard.clone()
        };

        let Some(session) = session else {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            continue;
        };

        let models = match sglang::get_loaded_models(&client, &config.sglang_url).await {
            Ok(models) => {
                *runtime.models.write().await = models.clone();
                models
            }
            Err(err) => {
                tracing::warn!(%err, "failed to refresh sglang model list for heartbeat");
                runtime.models.read().await.clone()
            }
        };

        let current_inflight = runtime.inflight.load(Ordering::Relaxed);
        // BUG-073: use real RSS instead of hardcoded 0.
        let rss_bytes = current_rss_bytes();
        let body = json!({
            "inflight": current_inflight,
            "thermal_state": "nominal",
            "model_ids": models,
            "rss_bytes": rss_bytes,
            "active_sequences": current_inflight,
            "decode_tok_per_sec": 0.0_f64,
            "ttft_p95_ms": 0_u64,
            "queue_depth": 0_usize,
            "error_rate": 0.0_f64,
        });

        // BUG-096: use a short per-request timeout for control-plane calls so a
        // slow/unresponsive orchestrator doesn't stall the heartbeat loop for 300s.
        let req = with_internal_token(
            client
                .post(format!(
                    "{}/internal/workers/{}/heartbeat",
                    config.control_plane_url, session.worker_id
                ))
                .timeout(std::time::Duration::from_secs(10))
                .json(&body),
            config.worker_token.as_ref(),
        );

        match req.send().await {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) if matches!(resp.status().as_u16(), 404 | 410) => {
                tracing::warn!(status = %resp.status(), "thor agent evicted, re-registering");
                match register(&client, &config).await {
                    Ok(registration) => {
                        *runtime.models.write().await = registration.models;
                        *runtime.session.write().await = Some(registration.session);
                    }
                    Err(err) => {
                        tracing::warn!(%err, "thor agent re-registration failed, clearing stale session");
                        *runtime.session.write().await = None;
                    }
                }
            }
            Ok(resp) => tracing::warn!(status = %resp.status(), "thor heartbeat rejected"),
            Err(err) => tracing::warn!(%err, "thor heartbeat failed"),
        }

        tokio::time::sleep(std::time::Duration::from_millis(
            session.heartbeat_interval_ms,
        ))
        .await;
    }
}

pub async fn drain(
    client: &reqwest::Client,
    config: &ThorConfig,
    runtime: &SharedRuntime,
) -> Result<()> {
    let session = runtime
        .session
        .read()
        .await
        .clone()
        .context("thor agent has no active worker session")?;
    with_internal_token(
        client.post(format!(
            "{}/internal/workers/{}/drain",
            config.control_plane_url, session.worker_id
        )),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("thor drain request failed")?
    .error_for_status()
    .context("thor drain request rejected")?;
    Ok(())
}

pub async fn drain_complete(
    client: &reqwest::Client,
    config: &ThorConfig,
    runtime: &SharedRuntime,
) -> Result<()> {
    let session = runtime
        .session
        .read()
        .await
        .clone()
        .context("thor agent has no active worker session")?;
    with_internal_token(
        client.post(format!(
            "{}/internal/workers/{}/drain-complete",
            config.control_plane_url, session.worker_id
        )),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("thor drain-complete request failed")?
    .error_for_status()
    .context("thor drain-complete request rejected")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SharedRuntime;

    #[tokio::test]
    async fn shared_runtime_starts_with_empty_model_cache() {
        let runtime = SharedRuntime::new();
        assert!(runtime.models.read().await.is_empty());
    }
}
