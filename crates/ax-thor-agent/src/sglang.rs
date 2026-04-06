use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

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

pub async fn get_loaded_models(client: &reqwest::Client, base_url: &str) -> Result<Vec<String>> {
    let url = format!("{base_url}/v1/models");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("failed to fetch sglang model list")?
        .error_for_status()
        .context("sglang returned error status for /v1/models")?;
    let models: ModelList = resp
        .json()
        .await
        .context("failed to parse sglang /v1/models response")?;
    Ok(models.data.into_iter().map(|m| m.id).collect())
}
