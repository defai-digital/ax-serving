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
    let model_info = sglang::get_model_info(client, &config.runtime_url).await?;
    let models: Vec<String> = model_info.iter().map(|m| m.id.clone()).collect();
    let model_inventory = model_info
        .iter()
        .map(model_inventory_entry)
        .collect::<Vec<_>>();

    // BUG-114: derive capabilities from config overrides or runtime metadata
    // instead of hardcoding them.
    let max_context: serde_json::Value = config
        .max_context
        .or_else(|| {
            // Use the max context from any loaded model as a best-effort default.
            model_info.iter().filter_map(|m| m.max_model_len).max()
        })
        .map(serde_json::Value::from)
        .unwrap_or(serde_json::Value::Null);

    let embedding = config.embedding.unwrap_or(false);
    let vision = config.vision.unwrap_or(false);
    let mut supported_operations = vec!["llm"];
    if embedding {
        supported_operations.push("embedding");
    }
    if vision {
        supported_operations.push("vision");
    }

    let body = json!({
        "addr": config.advertised_addr.to_string(),
        "capabilities": {
            "llm": true,
            "embedding": embedding,
            "vision": vision,
            "models": models,
            "max_context": max_context
        },
        "model_inventory": model_inventory,
        "backend": config.runtime.as_str(),
        "runtime": config.runtime.as_str(),
        "runtime_mode": "adapter",
        "hardware_class": config.hardware_class.as_str(),
        "runtime_endpoint": config.runtime_url.as_str(),
        "supported_operations": supported_operations,
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
        .context("runtime-node agent registration request failed")?
        .error_for_status()
        .context("runtime-node agent registration rejected")?
        .json()
        .await
        .context("failed to parse runtime-node agent registration response")?;

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

        let (models, model_inventory) =
            match sglang::get_model_info(&client, &config.runtime_url).await {
                Ok(model_info) => {
                    let models = model_info.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
                    let inventory = model_info
                        .iter()
                        .map(model_inventory_entry)
                        .collect::<Vec<_>>();
                    *runtime.models.write().await = models.clone();
                    (models, inventory)
                }
                Err(err) => {
                    tracing::warn!(%err, "failed to refresh runtime model list for heartbeat");
                    (runtime.models.read().await.clone(), Vec::new())
                }
            };

        let current_inflight = runtime.inflight.load(Ordering::Relaxed);
        // BUG-073: use real RSS instead of hardcoded 0.
        let rss_bytes = current_rss_bytes();
        let telemetry = match sglang::get_runtime_telemetry(&client, &config.runtime_url).await {
            Ok(telemetry) => telemetry,
            Err(err) => {
                tracing::debug!(%err, "runtime metrics unavailable; using heartbeat defaults");
                sglang::RuntimeTelemetry::default()
            }
        };
        let body = json!({
            "inflight": current_inflight,
            "thermal_state": "nominal",
            "model_ids": models,
            "model_inventory": model_inventory,
            "rss_bytes": rss_bytes,
            "active_sequences": telemetry.active_sequences.unwrap_or(current_inflight),
            "decode_tok_per_sec": telemetry.decode_tok_per_sec.unwrap_or(0.0_f64),
            "ttft_p95_ms": telemetry.ttft_p95_ms.unwrap_or(0_u64),
            "queue_depth": telemetry.queue_depth.unwrap_or(0_usize),
            "error_rate": telemetry.error_rate.unwrap_or(0.0_f64),
            "kv_pages_used": telemetry.kv_pages_used.unwrap_or(0_u64),
            "kv_pages_total": telemetry.kv_pages_total.unwrap_or(0_u64),
            "kv_utilization": telemetry.kv_utilization,
            "prefix_reusable_tokens": telemetry.prefix_reusable_tokens.unwrap_or(0_u64),
            "active_batch_size": telemetry.active_batch_size.unwrap_or(0_u32),
            "max_batch_size": telemetry.max_batch_size.unwrap_or(0_u32),
            "batch_utilization": telemetry.batch_utilization,
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
                tracing::warn!(status = %resp.status(), "runtime-node agent evicted, re-registering");
                match register(&client, &config).await {
                    Ok(registration) => {
                        *runtime.models.write().await = registration.models;
                        *runtime.session.write().await = Some(registration.session);
                    }
                    Err(err) => {
                        tracing::warn!(%err, "runtime-node agent re-registration failed, clearing stale session");
                        *runtime.session.write().await = None;
                    }
                }
            }
            Ok(resp) => tracing::warn!(status = %resp.status(), "runtime-node heartbeat rejected"),
            Err(err) => tracing::warn!(%err, "runtime-node heartbeat failed"),
        }

        tokio::time::sleep(std::time::Duration::from_millis(
            session.heartbeat_interval_ms,
        ))
        .await;
    }
}

fn model_inventory_entry(model: &sglang::ModelInfo) -> serde_json::Value {
    serde_json::json!({
        "id": model.id.as_str(),
        "max_context": model.max_model_len,
        "quantization": model.quantization.as_deref(),
        "artifact_format": model.artifact_format.as_deref(),
        "modalities": &model.modalities,
        "supported_operations": &model.supported_operations,
    })
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
        .context("runtime-node agent has no active worker session")?;
    with_internal_token(
        client.post(format!(
            "{}/internal/workers/{}/drain",
            config.control_plane_url, session.worker_id
        )),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("runtime-node drain request failed")?
    .error_for_status()
    .context("runtime-node drain request rejected")?;
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
        .context("runtime-node agent has no active worker session")?;
    with_internal_token(
        client.post(format!(
            "{}/internal/workers/{}/drain-complete",
            config.control_plane_url, session.worker_id
        )),
        config.worker_token.as_ref(),
    )
    .send()
    .await
    .context("runtime-node drain-complete request failed")?
    .error_for_status()
    .context("runtime-node drain-complete request rejected")?;
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
