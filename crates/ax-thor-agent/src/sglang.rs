use anyhow::{Context, Result};
use std::collections::BTreeMap;

pub async fn wait_for_runtime(client: &reqwest::Client, base_url: &str) -> Result<()> {
    const DEFAULT_TIMEOUT_SECS: u64 = 120;
    let timeout_secs = std::env::var("AXS_NODE_STARTUP_TIMEOUT_SECS")
        .or_else(|_| std::env::var("AXS_THOR_STARTUP_TIMEOUT_SECS"))
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let url = format!("{base_url}/health");
    loop {
        // BUG-115: check deadline BEFORE attempting the probe so the configured
        // timeout is a hard upper bound even if the connect timeout is longer.
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("runtime at {base_url} did not become healthy within {timeout_secs}s");
        }
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            Ok(resp) => tracing::warn!(status = %resp.status(), "runtime health not ready"),
            Err(err) => tracing::warn!(%err, "runtime health probe failed"),
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

pub async fn wait_for_sglang(client: &reqwest::Client, base_url: &str) -> Result<()> {
    wait_for_runtime(client, base_url).await
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
        .context("failed to fetch runtime model list")?
        .error_for_status()
        .context("runtime returned error status for /v1/models")?;
    let raw: serde_json::Value = resp
        .json()
        .await
        .context("failed to parse runtime /v1/models response")?;
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
            tracing::warn!("runtime /v1/models entry missing 'id' field, skipping: {entry}");
        }
    }
    Ok(models)
}

/// Best-effort runtime telemetry translated into the AX Serving heartbeat
/// contract. Runtimes that do not expose `/metrics` keep using safe defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeTelemetry {
    pub active_sequences: Option<usize>,
    pub decode_tok_per_sec: Option<f64>,
    pub ttft_p95_ms: Option<u64>,
    pub queue_depth: Option<usize>,
    pub error_rate: Option<f64>,
    pub kv_pages_used: Option<u64>,
    pub kv_pages_total: Option<u64>,
    pub prefix_reusable_tokens: Option<u64>,
    pub active_batch_size: Option<u32>,
    pub max_batch_size: Option<u32>,
}

pub async fn get_runtime_telemetry(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<RuntimeTelemetry> {
    let url = format!("{base_url}/metrics");
    let metrics = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .context("failed to fetch runtime metrics")?
        .error_for_status()
        .context("runtime returned error status for /metrics")?
        .text()
        .await
        .context("failed to read runtime /metrics response")?;
    Ok(parse_prometheus_telemetry(&metrics))
}

pub fn parse_prometheus_telemetry(metrics: &str) -> RuntimeTelemetry {
    let samples = collect_prometheus_samples(metrics);
    RuntimeTelemetry {
        active_sequences: sum_usize(
            &samples,
            &[
                "ax_runtime_active_sequences",
                "vllm:num_requests_running",
                "vllm_num_requests_running",
                "sglang:num_running_reqs",
                "sglang_num_running_reqs",
            ],
        ),
        decode_tok_per_sec: sum_f64(
            &samples,
            &[
                "ax_runtime_decode_tok_per_sec",
                "vllm:avg_generation_throughput_toks_per_s",
                "vllm_avg_generation_throughput_toks_per_s",
                "sglang:decode_throughput_toks_per_s",
                "sglang_decode_throughput_toks_per_s",
            ],
        ),
        ttft_p95_ms: max_u64(
            &samples,
            &[
                "ax_runtime_ttft_p95_ms",
                "vllm:time_to_first_token_p95_ms",
                "vllm_time_to_first_token_p95_ms",
                "sglang:time_to_first_token_p95_ms",
                "sglang_time_to_first_token_p95_ms",
            ],
        ),
        queue_depth: sum_usize(
            &samples,
            &[
                "ax_runtime_queue_depth",
                "vllm:num_requests_waiting",
                "vllm_num_requests_waiting",
                "sglang:num_queue_reqs",
                "sglang_num_queue_reqs",
            ],
        ),
        error_rate: max_f64(&samples, &["ax_runtime_error_rate"]),
        kv_pages_used: sum_u64(&samples, &["ax_runtime_kv_pages_used"]),
        kv_pages_total: sum_u64(&samples, &["ax_runtime_kv_pages_total"]),
        prefix_reusable_tokens: sum_u64(&samples, &["ax_runtime_prefix_reusable_tokens"]),
        active_batch_size: sum_u32(&samples, &["ax_runtime_active_batch_size"]),
        max_batch_size: sum_u32(&samples, &["ax_runtime_max_batch_size"]),
    }
}

fn collect_prometheus_samples(metrics: &str) -> BTreeMap<String, Vec<f64>> {
    let mut samples = BTreeMap::<String, Vec<f64>>::new();
    for raw_line in metrics.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = parse_prometheus_sample(line) else {
            continue;
        };
        samples.entry(name).or_default().push(value);
    }
    samples
}

fn parse_prometheus_sample(line: &str) -> Option<(String, f64)> {
    let mut parts = line.split_whitespace();
    let metric = parts.next()?;
    let value = parts.next()?.parse::<f64>().ok()?;
    let name = metric
        .split_once('{')
        .map(|(name, _)| name)
        .unwrap_or(metric)
        .trim();
    if name.is_empty() {
        return None;
    }
    Some((name.to_string(), value))
}

fn values_for<'a>(samples: &'a BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Vec<&'a f64> {
    aliases
        .iter()
        .filter_map(|alias| samples.get(*alias))
        .flat_map(|values| values.iter())
        .collect()
}

fn sum_f64(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<f64> {
    let values = values_for(samples, aliases);
    if values.is_empty() {
        None
    } else {
        Some(values.into_iter().sum())
    }
}

fn max_f64(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<f64> {
    values_for(samples, aliases)
        .into_iter()
        .copied()
        .reduce(f64::max)
}

fn sum_u64(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<u64> {
    sum_f64(samples, aliases).map(|v| v.max(0.0).round() as u64)
}

fn max_u64(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<u64> {
    max_f64(samples, aliases).map(|v| v.max(0.0).round() as u64)
}

fn sum_usize(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<usize> {
    sum_u64(samples, aliases).map(|v| v as usize)
}

fn sum_u32(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<u32> {
    sum_u64(samples, aliases).map(|v| v.min(u32::MAX as u64) as u32)
}

#[cfg(test)]
mod tests {
    use super::{RuntimeTelemetry, parse_prometheus_telemetry};

    #[test]
    fn parses_common_runtime_prometheus_metrics() {
        let telemetry = parse_prometheus_telemetry(
            r#"
# HELP ax_runtime_active_sequences Active decode sequences.
ax_runtime_active_sequences 4
ax_runtime_queue_depth{pool="default"} 3
ax_runtime_decode_tok_per_sec 42.5
ax_runtime_ttft_p95_ms 118
ax_runtime_error_rate 0.025
ax_runtime_kv_pages_used 12
ax_runtime_kv_pages_total 128
ax_runtime_prefix_reusable_tokens 256
ax_runtime_active_batch_size 2
ax_runtime_max_batch_size 16
"#,
        );

        assert_eq!(
            telemetry,
            RuntimeTelemetry {
                active_sequences: Some(4),
                decode_tok_per_sec: Some(42.5),
                ttft_p95_ms: Some(118),
                queue_depth: Some(3),
                error_rate: Some(0.025),
                kv_pages_used: Some(12),
                kv_pages_total: Some(128),
                prefix_reusable_tokens: Some(256),
                active_batch_size: Some(2),
                max_batch_size: Some(16),
            }
        );
    }

    #[test]
    fn parses_vllm_alias_metrics() {
        let telemetry = parse_prometheus_telemetry(
            r#"
vllm:num_requests_running 2
vllm:num_requests_waiting 5
vllm:avg_generation_throughput_toks_per_s 91.5
"#,
        );

        assert_eq!(telemetry.active_sequences, Some(2));
        assert_eq!(telemetry.queue_depth, Some(5));
        assert_eq!(telemetry.decode_tok_per_sec, Some(91.5));
    }
}
