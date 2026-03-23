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

fn backend_name(raw_backend: &str) -> &str {
    if raw_backend.eq_ignore_ascii_case("vllm") {
        "vllm"
    } else if raw_backend.eq_ignore_ascii_case("trtllm") {
        "trtllm"
    } else {
        "sglang"
    }
}

pub async fn wait_for_runtime(
    client: &reqwest::Client,
    base_url: &str,
    backend: &str,
) -> Result<()> {
    let backend = backend_name(backend);
    loop {
        match probe_runtime_ready(client, base_url, backend).await {
            Ok(()) => return Ok(()),
            Err(err) => tracing::warn!(backend, %err, "runtime not ready"),
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

pub async fn probe_runtime_ready(
    client: &reqwest::Client,
    base_url: &str,
    backend: &str,
) -> Result<()> {
    let backend = backend_name(backend);
    let base_url = base_url.trim_end_matches('/');
    let health_url = format!("{base_url}/health");
    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => return Ok(()),
        Ok(resp) => {
            tracing::warn!(backend, status = %resp.status(), "runtime /health not ready");
        }
        Err(err) => tracing::warn!(backend, %err, "runtime /health probe failed"),
    }

    get_loaded_models(client, base_url, backend)
        .await
        .map(|_| ())
        .with_context(|| format!("{backend} readiness fallback via /v1/models failed"))
}

pub async fn get_loaded_models(
    client: &reqwest::Client,
    base_url: &str,
    backend: &str,
) -> Result<Vec<String>> {
    let backend = backend_name(backend);
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to fetch {backend} model list"))?
        .error_for_status()
        .with_context(|| format!("{backend} returned error status for /v1/models"))?;
    let models: ModelList = resp
        .json()
        .await
        .with_context(|| format!("failed to parse {backend} /v1/models response"))?;
    Ok(models.data.into_iter().map(|m| m.id).collect())
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use axum::{Json, Router, http::StatusCode, routing::get};
    use serde_json::json;

    use super::probe_runtime_ready;

    #[tokio::test]
    async fn runtime_ready_uses_v1_models_when_health_is_missing() -> Result<()> {
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::NOT_FOUND }))
            .route(
                "/v1/models",
                get(|| async { Json(json!({ "data": [{ "id": "qwen2-72b" }] })) }),
            );
        let (base_url, _task) = spawn_server(app).await?;
        let client = reqwest::Client::new();

        probe_runtime_ready(&client, &base_url, "sglang").await?;
        probe_runtime_ready(&client, &base_url, "vllm").await?;
        probe_runtime_ready(&client, &base_url, "trtllm").await?;
        Ok(())
    }

    #[tokio::test]
    async fn runtime_ready_fails_with_backend_specific_context() -> Result<()> {
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::SERVICE_UNAVAILABLE }))
            .route("/v1/models", get(|| async { StatusCode::BAD_GATEWAY }));
        let (base_url, _task) = spawn_server(app).await?;
        let client = reqwest::Client::new();

        let err = probe_runtime_ready(&client, &base_url, "vllm")
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("vllm readiness fallback via /v1/models failed")
        );
        Ok(())
    }

    #[tokio::test]
    async fn runtime_ready_uses_trtllm_backend_context() -> Result<()> {
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::SERVICE_UNAVAILABLE }))
            .route("/v1/models", get(|| async { StatusCode::BAD_GATEWAY }));
        let (base_url, _task) = spawn_server(app).await?;
        let client = reqwest::Client::new();

        let err = probe_runtime_ready(&client, &base_url, "trtllm")
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("trtllm readiness fallback via /v1/models failed")
        );
        Ok(())
    }

    async fn spawn_server(app: Router) -> Result<(String, tokio::task::JoinHandle<()>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("failed to bind test listener")?;
        let addr = listener.local_addr().context("missing listener addr")?;
        let handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test server failed");
        });
        Ok((format!("http://{}", addr), handle))
    }
}
