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
    pub quantization: Option<String>,
    pub artifact_format: Option<String>,
    pub modalities: Vec<String>,
    pub supported_operations: Vec<String>,
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
    Ok(parse_model_info_response(&raw))
}

fn parse_model_info_response(raw: &serde_json::Value) -> Vec<ModelInfo> {
    let entries = raw["data"].as_array().cloned().unwrap_or_default();
    let mut models = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(id) = entry["id"].as_str() {
            let max_model_len = entry["max_model_len"]
                .as_u64()
                .or_else(|| entry["context_length"].as_u64())
                .or_else(|| entry["max_context"].as_u64())
                .map(|v| v as u32);
            let quantization = string_alias(
                &entry,
                &["quantization", "quantization_format", "quantization_config"],
            );
            let artifact_format =
                string_alias(&entry, &["artifact_format", "model_format", "format"]);
            let modalities = string_array_alias(&entry, &["modalities", "model_modalities"]);
            let supported_operations =
                operations_from_model_entry(&entry, modalities.iter().map(String::as_str));
            models.push(ModelInfo {
                id: id.to_string(),
                max_model_len,
                quantization,
                artifact_format,
                modalities,
                supported_operations,
            });
        } else {
            tracing::warn!("runtime /v1/models entry missing 'id' field, skipping: {entry}");
        }
    }
    models
}

fn string_alias(entry: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        entry.get(*key).and_then(|value| {
            value.as_str().map(str::to_string).or_else(|| {
                value
                    .as_object()
                    .and_then(|obj| obj.get("type").or_else(|| obj.get("name")))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
        })
    })
}

fn string_array_alias(entry: &serde_json::Value, keys: &[&str]) -> Vec<String> {
    let mut values = keys
        .iter()
        .find_map(|key| entry.get(*key))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn operations_from_model_entry<'a>(
    entry: &serde_json::Value,
    modalities: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let mut operations = string_array_alias(
        entry,
        &[
            "supported_operations",
            "operations",
            "tasks",
            "capabilities",
        ],
    );
    if entry
        .get("embedding")
        .or_else(|| entry.get("supports_embeddings"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        operations.push("embedding".to_string());
    }
    if entry
        .get("vision")
        .or_else(|| entry.get("supports_vision"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        operations.push("vision".to_string());
    }
    for modality in modalities {
        match modality {
            "embedding" | "embeddings" => operations.push("embedding".to_string()),
            "vision" | "image" | "multimodal" => operations.push("vision".to_string()),
            "text" | "llm" | "chat" | "completion" => operations.push("llm".to_string()),
            _ => {}
        }
    }
    if operations.is_empty() {
        operations.push("llm".to_string());
    }
    operations.sort();
    operations.dedup();
    operations
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
    pub kv_utilization: Option<f64>,
    pub prefix_reusable_tokens: Option<u64>,
    pub active_batch_size: Option<u32>,
    pub max_batch_size: Option<u32>,
    pub batch_utilization: Option<f64>,
}

pub async fn get_runtime_telemetry(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<RuntimeTelemetry> {
    let prometheus = get_prometheus_runtime_telemetry(client, base_url).await;
    match prometheus {
        // Prometheus gave us the critical routing fields — use it as-is.
        Ok(telemetry)
            if telemetry.active_sequences.is_some() || telemetry.queue_depth.is_some() =>
        {
            Ok(telemetry)
        }
        // Prometheus succeeded but is missing critical fields; try JSON to fill them in.
        Ok(telemetry) => match get_json_runtime_telemetry(client, base_url).await {
            Ok(json_telemetry) => Ok(json_telemetry),
            Err(_) => Ok(telemetry),
        },
        Err(prometheus_err) => get_json_runtime_telemetry(client, base_url)
            .await
            .with_context(|| {
                format!("runtime metrics unavailable; /metrics error: {prometheus_err}")
            }),
    }
}

async fn get_prometheus_runtime_telemetry(
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

async fn get_json_runtime_telemetry(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<RuntimeTelemetry> {
    let url = format!("{base_url}/v1/metrics");
    let metrics = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .context("failed to fetch runtime JSON metrics")?
        .error_for_status()
        .context("runtime returned error status for /v1/metrics")?
        .json::<serde_json::Value>()
        .await
        .context("failed to parse runtime /v1/metrics response")?;
    Ok(parse_json_telemetry(&metrics))
}

pub fn parse_prometheus_telemetry(metrics: &str) -> RuntimeTelemetry {
    let samples = collect_prometheus_samples(metrics);
    let ttft_bucket_p95_ms = prometheus_histogram_quantile_seconds(
        metrics,
        &[
            "vllm:time_to_first_token_seconds_bucket",
            "vllm_time_to_first_token_seconds_bucket",
        ],
        0.95,
    )
    .map(seconds_to_ms);
    RuntimeTelemetry {
        active_sequences: sum_usize(
            &samples,
            &[
                "ax_runtime_active_sequences",
                "axs_scheduler_decode_sequences_active",
                "axs_scheduler_inflight_count",
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
                "axs_recent_decode_tok_per_sec",
                "axs_decode_tok_per_sec",
                "vllm:avg_generation_throughput_toks_per_s",
                "vllm_avg_generation_throughput_toks_per_s",
                "sglang:decode_throughput_toks_per_s",
                "sglang_decode_throughput_toks_per_s",
            ],
        ),
        ttft_p95_ms: max_duration_ms(
            &samples,
            &["axs_ttft_p95_us"],
            &[
                "ax_runtime_ttft_p95_ms",
                "vllm:time_to_first_token_p95_ms",
                "vllm_time_to_first_token_p95_ms",
                "sglang:time_to_first_token_p95_ms",
                "sglang_time_to_first_token_p95_ms",
            ],
        )
        .or(ttft_bucket_p95_ms),
        queue_depth: sum_usize(
            &samples,
            &[
                "ax_runtime_queue_depth",
                "axs_scheduler_queue_depth",
                "vllm:num_requests_waiting",
                "vllm_num_requests_waiting",
                "sglang:num_queue_reqs",
                "sglang_num_queue_reqs",
            ],
        ),
        error_rate: max_f64(&samples, &["ax_runtime_error_rate", "axs_slo_error_rate"]),
        kv_pages_used: sum_u64(&samples, &["ax_runtime_kv_pages_used"]),
        kv_pages_total: sum_u64(&samples, &["ax_runtime_kv_pages_total"]),
        kv_utilization: max_ratio(
            &samples,
            &[
                "ax_runtime_kv_utilization",
                "ax_runtime_kv_cache_utilization",
                "axs_kv_utilization",
                "axs_kv_cache_utilization",
                "vllm:gpu_cache_usage_perc",
                "vllm_gpu_cache_usage_perc",
                "vllm:kv_cache_usage_perc",
                "vllm_kv_cache_usage_perc",
            ],
        ),
        prefix_reusable_tokens: sum_u64(
            &samples,
            &[
                "ax_runtime_prefix_reusable_tokens",
                "vllm:prefix_cache_hits",
                "vllm_prefix_cache_hits",
            ],
        ),
        active_batch_size: max_u32(
            &samples,
            &[
                "ax_runtime_active_batch_size",
                "vllm:num_requests_running",
                "vllm_num_requests_running",
            ],
        ),
        max_batch_size: max_u32(
            &samples,
            &["ax_runtime_max_batch_size", "axs_scheduler_max_inflight"],
        ),
        batch_utilization: max_ratio(
            &samples,
            &[
                "ax_runtime_batch_utilization",
                "ax_runtime_batch_pressure",
                "axs_batch_utilization",
                "axs_batch_pressure",
                "vllm:batch_utilization",
                "vllm_batch_utilization",
            ],
        ),
    }
}

pub fn parse_json_telemetry(metrics: &serde_json::Value) -> RuntimeTelemetry {
    RuntimeTelemetry {
        active_sequences: json_usize_any(
            metrics,
            &[
                "/active_sequences",
                "/scheduler/decode_sequences_active",
                "/scheduler/inflight_count",
                "/inflight_count",
            ],
        ),
        decode_tok_per_sec: json_f64_any(
            metrics,
            &[
                "/decode_tok_per_sec",
                "/recent_decode_tok_per_sec",
                "/metrics/decode_tok_per_sec",
                "/metrics/recent_decode_tok_per_sec",
            ],
        ),
        ttft_p95_ms: json_duration_ms_any(
            metrics,
            &["/scheduler/ttft_p95_us", "/ttft_p95_us"],
            &[
                "/ttft_p95_ms",
                "/scheduler/ttft_p95_ms",
                "/metrics/ttft_p95_ms",
            ],
        ),
        queue_depth: json_usize_any(metrics, &["/queue_depth", "/scheduler/queue_depth"]),
        error_rate: json_f64_any(metrics, &["/error_rate", "/metrics/error_rate"]),
        kv_pages_used: json_u64_any(
            metrics,
            &[
                "/kv_pages_used",
                "/kv_cache/pages_used",
                "/cache/kv_pages_used",
            ],
        ),
        kv_pages_total: json_u64_any(
            metrics,
            &[
                "/kv_pages_total",
                "/kv_cache/pages_total",
                "/cache/kv_pages_total",
            ],
        ),
        kv_utilization: json_ratio_any(
            metrics,
            &[
                "/kv_utilization",
                "/kv_cache_utilization",
                "/kv_cache/utilization",
                "/cache/kv_utilization",
            ],
        ),
        prefix_reusable_tokens: json_u64_any(
            metrics,
            &[
                "/prefix_reusable_tokens",
                "/prefix_cache/reusable_tokens",
                "/cache/prefix_reusable_tokens",
            ],
        ),
        active_batch_size: json_u32_any(
            metrics,
            &[
                "/active_batch_size",
                "/batch/active_size",
                "/scheduler/inflight_count",
            ],
        ),
        max_batch_size: json_u32_any(
            metrics,
            &[
                "/max_batch_size",
                "/batch/max_size",
                "/scheduler/max_inflight",
                "/scheduler/effective_inflight_limit",
            ],
        ),
        batch_utilization: json_ratio_any(
            metrics,
            &[
                "/batch_utilization",
                "/batch_pressure",
                "/batch/utilization",
                "/batch/pressure",
            ],
        ),
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

fn parse_bucket_upper_bound(line: &str) -> Option<f64> {
    let mut parts = line.split_whitespace();
    let metric = parts.next()?;
    parts.next()?.parse::<f64>().ok()?;
    let labels = metric.split_once('{')?.1.trim_end_matches('}');
    for label in labels.split(',') {
        let Some((key, value)) = label.split_once('=') else {
            continue;
        };
        if key.trim() != "le" {
            continue;
        }
        let bound = value.trim().trim_matches('"');
        return if bound == "+Inf" || bound == "Inf" {
            Some(f64::INFINITY)
        } else {
            bound.parse::<f64>().ok()
        };
    }
    None
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

fn max_ratio(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<f64> {
    max_f64(samples, aliases).map(|value| {
        let normalized = if value > 1.0 { value / 100.0 } else { value };
        normalized.clamp(0.0, 1.0)
    })
}

fn max_duration_ms(
    samples: &BTreeMap<String, Vec<f64>>,
    us_aliases: &[&str],
    ms_aliases: &[&str],
) -> Option<u64> {
    max_u64(samples, us_aliases)
        .map(us_to_ms)
        .or_else(|| max_u64(samples, ms_aliases))
}

fn prometheus_histogram_quantile_seconds(
    metrics: &str,
    bucket_aliases: &[&str],
    quantile: f64,
) -> Option<f64> {
    let mut buckets: Vec<(f64, f64)> = Vec::new();
    for raw_line in metrics.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, count)) = parse_prometheus_sample(line) else {
            continue;
        };
        if !bucket_aliases.iter().any(|alias| *alias == name) {
            continue;
        }
        let Some(bound) = parse_bucket_upper_bound(line) else {
            continue;
        };
        if count.is_finite() && count >= 0.0 {
            buckets.push((bound, count));
        }
    }

    if buckets.is_empty() {
        return None;
    }

    buckets.sort_by(|(left, _), (right, _)| {
        left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (bound, count) in buckets {
        if let Some((last_bound, last_count)) = merged.last_mut()
            && ((*last_bound - bound).abs() < f64::EPSILON
                || (last_bound.is_infinite() && bound.is_infinite()))
        {
            *last_count += count;
            continue;
        }
        merged.push((bound, count));
    }

    let total = merged
        .iter()
        .rev()
        .find_map(|(_, count)| (*count > 0.0).then_some(*count))?;
    if total <= 0.0 {
        return Some(0.0);
    }
    let target = total * quantile.clamp(0.0, 1.0);
    let mut previous_count = 0.0;
    let mut previous_bound = 0.0;
    for (bound, count) in merged {
        if count >= target {
            if bound.is_infinite() {
                return Some(previous_bound);
            }
            if count <= previous_count {
                return Some(bound);
            }
            let position = (target - previous_count) / (count - previous_count);
            return Some(previous_bound + (bound - previous_bound) * position);
        }
        previous_count = count;
        previous_bound = bound;
    }
    Some(previous_bound)
}

fn sum_u64(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<u64> {
    sum_f64(samples, aliases).map(|v| v.max(0.0).round() as u64)
}

fn max_u64(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<u64> {
    max_f64(samples, aliases).map(|v| v.max(0.0).round() as u64)
}

/// Sums per-label samples within each alias (label stripping collapses per-model values
/// into one Vec), then takes the max across aliases (different metric names for the same
/// concept must not be summed together — they'd double-count).
fn sum_per_alias_max_across(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<f64> {
    aliases
        .iter()
        .filter_map(|alias| {
            let values = samples.get(*alias)?;
            if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>())
            }
        })
        .reduce(f64::max)
}

fn sum_usize(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<usize> {
    sum_per_alias_max_across(samples, aliases).map(|v| v.max(0.0).round() as usize)
}

fn max_u32(samples: &BTreeMap<String, Vec<f64>>, aliases: &[&str]) -> Option<u32> {
    max_u64(samples, aliases).map(|v| v.min(u32::MAX as u64) as u32)
}

fn us_to_ms(value: u64) -> u64 {
    value.div_ceil(1_000)
}

fn seconds_to_ms(value: f64) -> u64 {
    (value.max(0.0) * 1_000.0).round() as u64
}

fn json_f64_any(body: &serde_json::Value, pointers: &[&str]) -> Option<f64> {
    pointers.iter().find_map(|pointer| {
        body.pointer(pointer).and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        })
    })
}

fn json_u64_any(body: &serde_json::Value, pointers: &[&str]) -> Option<u64> {
    json_f64_any(body, pointers).map(|value| value.max(0.0).round() as u64)
}

fn json_usize_any(body: &serde_json::Value, pointers: &[&str]) -> Option<usize> {
    json_u64_any(body, pointers).map(|value| value as usize)
}

fn json_u32_any(body: &serde_json::Value, pointers: &[&str]) -> Option<u32> {
    json_u64_any(body, pointers).map(|value| value.min(u32::MAX as u64) as u32)
}

fn json_ratio_any(body: &serde_json::Value, pointers: &[&str]) -> Option<f64> {
    json_f64_any(body, pointers).map(|value| {
        let normalized = if value > 1.0 { value / 100.0 } else { value };
        normalized.clamp(0.0, 1.0)
    })
}

fn json_duration_ms_any(
    body: &serde_json::Value,
    us_pointers: &[&str],
    ms_pointers: &[&str],
) -> Option<u64> {
    json_u64_any(body, us_pointers)
        .map(us_to_ms)
        .or_else(|| json_u64_any(body, ms_pointers))
}

#[cfg(test)]
mod tests {
    use super::{
        RuntimeTelemetry, parse_json_telemetry, parse_model_info_response,
        parse_prometheus_telemetry,
    };

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
ax_runtime_kv_utilization 0.75
ax_runtime_prefix_reusable_tokens 256
ax_runtime_active_batch_size 2
ax_runtime_max_batch_size 16
ax_runtime_batch_utilization 0.125
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
                kv_utilization: Some(0.75),
                prefix_reusable_tokens: Some(256),
                active_batch_size: Some(2),
                max_batch_size: Some(16),
                batch_utilization: Some(0.125),
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
vllm:gpu_cache_usage_perc 87
vllm:batch_utilization 0.5
"#,
        );

        assert_eq!(telemetry.active_sequences, Some(2));
        assert_eq!(telemetry.queue_depth, Some(5));
        assert_eq!(telemetry.decode_tok_per_sec, Some(91.5));
        assert_eq!(telemetry.kv_utilization, Some(0.87));
        assert_eq!(telemetry.batch_utilization, Some(0.5));
    }

    #[test]
    fn parses_ax_serving_prometheus_alias_metrics() {
        let telemetry = parse_prometheus_telemetry(
            r#"
axs_scheduler_inflight_count 3
axs_scheduler_decode_sequences_active 2
axs_scheduler_queue_depth 4
axs_ttft_p95_us 123456
axs_scheduler_max_inflight 16
axs_kv_cache_utilization 65
axs_batch_pressure 0.25
"#,
        );

        assert_eq!(telemetry.active_sequences, Some(3));
        assert_eq!(telemetry.queue_depth, Some(4));
        assert_eq!(telemetry.ttft_p95_ms, Some(124));
        assert_eq!(telemetry.max_batch_size, Some(16));
        assert_eq!(telemetry.kv_utilization, Some(0.65));
        assert_eq!(telemetry.batch_utilization, Some(0.25));
    }

    #[test]
    fn parses_vllm_ttft_histogram_bucket_metrics() {
        let telemetry = parse_prometheus_telemetry(
            r#"
vllm:time_to_first_token_seconds_bucket{le="0.1"} 10
vllm:time_to_first_token_seconds_bucket{le="0.2"} 90
vllm:time_to_first_token_seconds_bucket{le="0.4"} 100
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100
"#,
        );

        assert_eq!(telemetry.ttft_p95_ms, Some(300));
    }

    #[test]
    fn parses_ax_engine_json_metrics_profile() {
        let telemetry = parse_json_telemetry(&serde_json::json!({
            "scheduler": {
                "queue_depth": 7,
                "inflight_count": 3,
                "ttft_p95_us": 42_500,
                "max_inflight": 12
            },
            "kv_cache": {
                "utilization": 0.5,
                "pages_used": 64,
                "pages_total": 128
            },
            "batch": {
                "utilization": 75,
                "active_size": 4,
                "max_size": 16
            },
            "prefix_cache": {
                "reusable_tokens": 512
            }
        }));

        assert_eq!(telemetry.active_sequences, Some(3));
        assert_eq!(telemetry.queue_depth, Some(7));
        assert_eq!(telemetry.ttft_p95_ms, Some(43));
        assert_eq!(telemetry.kv_pages_used, Some(64));
        assert_eq!(telemetry.kv_pages_total, Some(128));
        assert_eq!(telemetry.kv_utilization, Some(0.5));
        assert_eq!(telemetry.prefix_reusable_tokens, Some(512));
        assert_eq!(telemetry.active_batch_size, Some(4));
        assert_eq!(telemetry.max_batch_size, Some(16));
        assert_eq!(telemetry.batch_utilization, Some(0.75));
    }

    #[test]
    fn parses_runtime_model_inventory_metadata() {
        let models = parse_model_info_response(&serde_json::json!({
            "data": [
                {
                    "id": "qwen3-32b",
                    "max_model_len": 32768,
                    "quantization": "awq",
                    "model_format": "safetensors",
                    "modalities": ["text", "vision"],
                    "supported_operations": ["llm", "vision"]
                },
                {
                    "id": "embed",
                    "context_length": 8192,
                    "quantization_config": {"type": "int8"},
                    "artifact_format": "gguf",
                    "supports_embeddings": true
                }
            ]
        }));

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "qwen3-32b");
        assert_eq!(models[0].max_model_len, Some(32768));
        assert_eq!(models[0].quantization.as_deref(), Some("awq"));
        assert_eq!(models[0].artifact_format.as_deref(), Some("safetensors"));
        assert_eq!(models[0].modalities, vec!["text", "vision"]);
        assert_eq!(models[0].supported_operations, vec!["llm", "vision"]);
        assert_eq!(models[1].quantization.as_deref(), Some("int8"));
        assert_eq!(models[1].artifact_format.as_deref(), Some("gguf"));
        assert_eq!(models[1].supported_operations, vec!["embedding"]);
    }
}
