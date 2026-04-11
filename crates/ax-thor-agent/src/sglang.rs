use anyhow::{Context, Result};

pub async fn wait_for_sglang(client: &reqwest::Client, base_url: &str) -> Result<()> {
    const DEFAULT_TIMEOUT_SECS: u64 = 120;
    let timeout_secs = std::env::var("AXS_THOR_STARTUP_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let url = format!("{base_url}/health");
    loop {
        // BUG-115: check deadline BEFORE attempting the probe so the configured
        // timeout is a hard upper bound even if the connect timeout is longer.
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "sglang runtime at {base_url} did not become healthy within {timeout_secs}s"
            );
        }
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            Ok(resp) => tracing::warn!(status = %resp.status(), "sglang health not ready"),
            Err(err) => tracing::warn!(%err, "sglang health probe failed"),
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

/// Model information returned from the runtime.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub max_model_len: Option<u32>,
}

pub async fn get_loaded_models(client: &reqwest::Client, base_url: &str) -> Result<Vec<String>> {
    let info = get_model_info(client, base_url).await?;
    Ok(info.into_iter().map(|m| m.id).collect())
}

/// Query the runtime for loaded models and available metadata.
///
/// Parses optional `max_model_len` from the response (sglang and vLLM both
/// include this field when available).
pub async fn get_model_info(client: &reqwest::Client, base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{base_url}/v1/models");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("failed to fetch sglang model list")?
        .error_for_status()
        .context("sglang returned error status for /v1/models")?;
    let raw: serde_json::Value = resp
        .json()
        .await
        .context("failed to parse sglang /v1/models response")?;
    let entries = raw["data"].as_array().cloned().unwrap_or_default();
    let mut models = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(id) = entry["id"].as_str() {
            let max_model_len = entry["max_model_len"]
                .as_u64()
                .or_else(|| entry["context_length"].as_u64())
                .map(|v| v as u32);
            models.push(ModelInfo {
                id: id.to_string(),
                max_model_len,
            });
        } else {
            tracing::warn!("sglang /v1/models entry missing 'id' field, skipping: {entry}");
        }
    }
    Ok(models)
}
