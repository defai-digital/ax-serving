//! Shared helpers and re-exports for REST route handlers.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::ServingLayer;
use crate::project_policy;
use crate::scheduler::SchedulerError;
use crate::utils::request_meta::default_audit_limit;

// ── Re-exports so that rest/mod.rs router paths continue to work ─────────────

pub use super::admin::{
    admin_audit, admin_diagnostics, admin_policy, admin_status, admin_startup_report, health,
    metrics, prometheus_metrics,
};
pub use super::inference::{chat_completions, embeddings, text_completions};
pub use super::license::{dashboard, get_license, set_license};
pub use super::models::{list_models, rest_load_model, rest_reload_model, rest_unload_model};

// ── Shared helpers used by multiple modules ──────────────────────────────────

/// Map a scheduler error to the correct HTTP status code.
///
/// - [`SchedulerError::QueueFull`] → 429 Too Many Requests (client should retry with back-off)
/// - All other variants → 503 Service Unavailable (server-side overload / shutdown)
pub(crate) fn scheduler_error_status(e: &anyhow::Error) -> StatusCode {
    match e.downcast_ref::<SchedulerError>() {
        Some(SchedulerError::QueueFull { .. }) => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Inject `X-Ax-Stage-Timing: queue_wait_us=<N>` into any response.
///
/// Called on both the success path and all error paths so callers can always
/// correlate observed latency with time spent waiting in the admission queue,
/// regardless of whether the request eventually succeeded or failed.
#[inline]
pub(crate) fn with_timing(mut resp: Response, queue_wait_us: u64) -> Response {
    if let Ok(val) = HeaderValue::from_str(&format!("queue_wait_us={queue_wait_us}")) {
        resp.headers_mut().insert("x-ax-stage-timing", val);
    }
    resp
}

pub(crate) fn slo_pass_gauges(
    total_requests: u64,
    rejected_requests: u64,
    e2e_p99_us: u64,
    queue_p99_us: u64,
    slo_e2e_p99_ms: u64,
    slo_queue_p99_ms: u64,
    slo_max_error_rate: f64,
) -> (u8, u8, u8) {
    let error_rate = if total_requests > 0 {
        rejected_requests as f64 / total_requests as f64
    } else {
        0.0
    };

    // Require at least one completed request before reporting pass.
    // Without this guard all three gauges would be 1 at startup / idle,
    // which is a false positive that masks misconfigured alerting rules.
    let have_data = total_requests > 0;
    let e2e_pass = u8::from(have_data && e2e_p99_us <= slo_e2e_p99_ms * 1_000);
    let queue_pass = u8::from(have_data && queue_p99_us <= slo_queue_p99_ms * 1_000);
    let error_pass = u8::from(have_data && error_rate <= slo_max_error_rate);
    (e2e_pass, queue_pass, error_pass)
}

// ── Shared cache helpers ─────────────────────────────────────────────────────

pub(crate) fn record_cache_error(
    metrics: &crate::cache::CacheMetrics,
    msg: impl std::fmt::Display,
) {
    metrics.errors.fetch_add(1, Ordering::Relaxed);
    tracing::warn!("cache error: {msg}");
}

pub(crate) fn cache_hit_response(hit_json: String) -> Response {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        hit_json,
    )
        .into_response()
}

pub(crate) async fn write_cache_and_record<T: Serialize>(
    cache: &crate::cache::ResponseCache,
    key: &str,
    value: &T,
    ttl: std::time::Duration,
    cache_metrics: &crate::cache::CacheMetrics,
    serving_metrics: &crate::metrics::MetricsStore,
) {
    if let Err(e) = cache.set(key, value, ttl).await {
        record_cache_error(cache_metrics, format_args!("write: {e}"));
    } else {
        serving_metrics.record_cache_fill();
    }
}

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(serde::Deserialize)]
pub struct AuditQuery {
    #[serde(default = "default_audit_limit")]
    pub limit: usize,
}

pub(crate) fn serving_startup_report_value(layer: &Arc<ServingLayer>) -> serde_json::Value {
    let config_validation = layer.config.validate().err().map(|e| e.to_string());
    let allowed_model_dirs = std::env::var("AXS_MODEL_ALLOWED_DIRS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    serde_json::json!({
        "service": "serving",
        "status": if config_validation.is_none() { "ok" } else { "degraded" },
        "config_valid": config_validation.is_none(),
        "config_error": config_validation,
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "license": layer.license.to_json(),
        "runtime": {
            "rest_addr": layer.config.rest_addr,
            "grpc_socket": layer.config.grpc_socket,
            "grpc_host": layer.config.grpc_host,
            "grpc_port": layer.config.grpc_port,
            "cache_enabled": layer.config.cache.enabled,
            "split_scheduler": layer.config.split_scheduler,
            "default_max_tokens": layer.config.default_max_tokens,
            "idle_timeout_secs": layer.config.idle_timeout_secs,
            "thermal_poll_secs": layer.config.thermal_poll_secs,
            "sched_max_inflight": layer.config.sched_max_inflight,
            "sched_max_queue": layer.config.sched_max_queue,
        },
        "scheduler": {
            "effective_inflight_limit": layer.scheduler.effective_inflight_limit(),
            "split_scheduler_enabled": layer.scheduler.split_enabled,
        },
        "cache": {
            "enabled": layer.cache.is_some(),
            "mode": if layer.cache.is_some() { "exact_response" } else { "disabled" },
            "kv_prefix_cache": false,
        },
        "trust": {
            "allowed_model_dirs": allowed_model_dirs,
        },
        "project_policy": project_policy::summary_json(&layer.config.project_policy),
        "governance": {
            "project_policy_enabled": layer.config.project_policy.enabled,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::inference::build_cache_key;
    use super::super::models::is_valid_pooling_type;
    use super::super::schema::*;
    use super::super::validation::{
        validate_max_tokens, validate_response_format, validate_sampling_params,
    };
    use crate::cache::CachePreference;

    fn mk_req(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "default".into(),
            messages: vec![InputMessage {
                role: "user".into(),
                content: MessageContent::Text(content.into()),
                name: None,
            }],
            stream: false,
            temperature: 0.0,
            max_tokens: Some(16),
            top_p: 1.0,
            min_p: None,
            top_k: Some(1),
            seed: None,
            repeat_penalty: 1.1,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            grammar: None,
            response_format: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tools: None,
            tool_choice: None,
            cache: Some(CachePreference::Enable),
            cache_ttl: Some("1h".into()),
            logprobs: None,
            top_logprobs: None,
        }
    }

    #[test]
    fn cache_key_changes_for_different_resolved_models() {
        let req = mk_req("same prompt");
        let k1 = build_cache_key(&req, "models/Qwen3-8B-Q4_K_M.gguf", "qwen3", Some(16)).unwrap();
        let k2 = build_cache_key(
            &req,
            "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            "llama",
            Some(16),
        )
        .unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_stable_for_same_inputs() {
        let req = mk_req("same prompt");
        let k1 = build_cache_key(&req, "models/Qwen3-8B-Q4_K_M.gguf", "qwen3", Some(16)).unwrap();
        let k2 = build_cache_key(&req, "models/Qwen3-8B-Q4_K_M.gguf", "qwen3", Some(16)).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_normalizes_message_whitespace() {
        let r1 = mk_req("Hello world");
        let r2 = mk_req("  Hello world  ");
        let k1 = build_cache_key(&r1, "model.gguf", "llama", Some(16)).unwrap();
        let k2 = build_cache_key(&r2, "model.gguf", "llama", Some(16)).unwrap();
        assert_eq!(
            k1, k2,
            "leading/trailing whitespace must not affect cache key"
        );
    }

    #[test]
    fn cache_key_normalizes_role_case() {
        let mut r1 = mk_req("Hello");
        r1.messages[0].role = "User".into();
        let mut r2 = mk_req("Hello");
        r2.messages[0].role = "user".into();
        let k1 = build_cache_key(&r1, "model.gguf", "llama", Some(16)).unwrap();
        let k2 = build_cache_key(&r2, "model.gguf", "llama", Some(16)).unwrap();
        assert_eq!(k1, k2, "role case must not affect cache key");
    }

    #[test]
    fn cache_key_normalizes_float_precision_noise() {
        // f32 can represent 0.7 as ~0.6999999762; both must hash identically.
        let mut r1 = mk_req("Hi");
        r1.temperature = 0.7_f32;
        let mut r2 = mk_req("Hi");
        // Next representable f32 above 0.7 — rounds to "0.7000" at 4dp.
        r2.temperature = 0.700_001_f32;
        let k1 = build_cache_key(&r1, "model.gguf", "llama", Some(16)).unwrap();
        let k2 = build_cache_key(&r2, "model.gguf", "llama", Some(16)).unwrap();
        assert_eq!(
            k1, k2,
            "sub-4-decimal float noise must not affect cache key"
        );
    }

    #[test]
    fn cache_key_same_for_no_max_tokens_and_default() {
        // Verifies that req.max_tokens is NOT used in the cache key — only
        // effective_max_tokens (the server-resolved value) matters.  A request
        // with max_tokens=Some(16) and one with max_tokens=None both produce
        // the same cache entry when effective_max_tokens is the same.
        let req_explicit = mk_req("prompt");
        let req_none = ChatCompletionRequest {
            max_tokens: None,
            ..mk_req("prompt")
        };
        let k_explicit = build_cache_key(&req_explicit, "m.gguf", "llama", Some(16)).unwrap();
        let k_none = build_cache_key(&req_none, "m.gguf", "llama", Some(16)).unwrap();
        assert_eq!(
            k_explicit, k_none,
            "explicit max_tokens matching default must share cache key"
        );
    }

    #[test]
    fn sampling_params_valid_boundaries_accepted() {
        assert!(
            validate_sampling_params(
                0.0,
                1.0,
                None,
                Some(1),
                0.1,
                None,
                None,
                Some(true),
                Some(20),
                Some(2)
            )
            .is_none()
        );
        assert!(
            validate_sampling_params(
                2.0,
                0.01,
                None,
                None,
                10.0,
                Some(-2.0),
                Some(2.0),
                None,
                None,
                Some(0),
            )
            .is_none()
        );
    }

    #[test]
    fn sampling_params_temperature_out_of_range() {
        assert!(
            validate_sampling_params(-0.1, 1.0, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
        assert!(
            validate_sampling_params(2.01, 1.0, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
    }

    #[test]
    fn sampling_params_top_p_out_of_range() {
        assert!(
            validate_sampling_params(1.0, 0.0, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
        assert!(
            validate_sampling_params(1.0, 1.01, None, None, 1.1, None, None, None, None, None)
                .is_some()
        );
    }

    #[test]
    fn sampling_params_top_k_zero_rejected() {
        assert!(
            validate_sampling_params(1.0, 1.0, None, Some(0), 1.1, None, None, None, None, None)
                .is_some()
        );
        assert!(
            validate_sampling_params(1.0, 1.0, None, Some(1), 1.1, None, None, None, None, None)
                .is_none()
        );
    }

    #[test]
    fn sampling_params_penalties_out_of_range() {
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                Some(2.01),
                None,
                None,
                None,
                None,
            )
            .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                Some(-2.01),
                None,
                None,
                None,
            )
            .is_some()
        );
    }

    #[test]
    fn sampling_params_top_logprobs_over_limit() {
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(true),
                Some(21),
                None,
            )
            .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(true),
                Some(20),
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn sampling_params_top_logprobs_requires_logprobs() {
        assert!(
            validate_sampling_params(1.0, 1.0, None, None, 1.1, None, None, None, Some(1), None)
                .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(false),
                Some(1),
                None,
            )
            .is_some()
        );
        assert!(
            validate_sampling_params(
                1.0,
                1.0,
                None,
                None,
                1.1,
                None,
                None,
                Some(true),
                Some(1),
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn sampling_params_mirostat_invalid() {
        assert!(
            validate_sampling_params(1.0, 1.0, None, None, 1.1, None, None, None, None, Some(3))
                .is_some()
        );
        assert!(
            validate_sampling_params(1.0, 1.0, None, None, 1.1, None, None, None, None, Some(2))
                .is_none()
        );
    }

    #[test]
    fn response_format_validation_rejects_unknown_values() {
        let invalid = ResponseFormat {
            format_type: "xml".into(),
        };
        assert!(validate_response_format(Some(&invalid)).is_some());
        let text = ResponseFormat {
            format_type: "text".into(),
        };
        assert!(validate_response_format(Some(&text)).is_none());
        let json = ResponseFormat {
            format_type: "json_object".into(),
        };
        assert!(validate_response_format(Some(&json)).is_none());
    }

    #[test]
    fn max_tokens_validation_rejects_zero_and_schema_limit() {
        assert!(validate_max_tokens(Some(0)).is_some());
        assert!(validate_max_tokens(Some(MAX_MAX_TOKENS)).is_none());
        assert!(validate_max_tokens(Some(MAX_MAX_TOKENS + 1)).is_some());
    }

    #[test]
    fn pooling_type_validation_accepts_allowed_values() {
        for v in ["none", "mean", "cls", "last", "rank", "MEAN"] {
            assert!(is_valid_pooling_type(v), "expected valid pooling_type: {v}");
        }
    }

    #[test]
    fn pooling_type_validation_rejects_unknown_values() {
        for v in ["", "avg", "median", "foo"] {
            assert!(
                !is_valid_pooling_type(v),
                "expected invalid pooling_type: {v}"
            );
        }
    }

    #[test]
    fn scheduler_error_status_queue_full_is_429() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::QueueFull {
            waiting: 64,
            max: 64,
        }
        .into();
        assert_eq!(scheduler_error_status(&e), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn scheduler_error_status_timeout_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::Timeout { wait_ms: 250 }.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_throttled_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::Throttled { cap: 8 }.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_shed_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::Shed.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_shutting_down_is_503() {
        let e: anyhow::Error = crate::scheduler::SchedulerError::ShuttingDown.into();
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn scheduler_error_status_generic_error_is_503() {
        let e: anyhow::Error = anyhow::anyhow!("some unexpected error");
        assert_eq!(scheduler_error_status(&e), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn slo_pass_gauges_fail_closed_without_data() {
        let (e2e_pass, queue_pass, error_pass) =
            slo_pass_gauges(0, 0, 100, 100, 1_000, 1_000, 0.05);
        assert_eq!((e2e_pass, queue_pass, error_pass), (0, 0, 0));
    }

    #[test]
    fn slo_pass_gauges_apply_thresholds() {
        let (e2e_pass, queue_pass, error_pass) =
            slo_pass_gauges(100, 4, 900_000, 800_000, 1_000, 1_000, 0.05);
        assert_eq!((e2e_pass, queue_pass, error_pass), (1, 1, 1));

        let (e2e_pass, queue_pass, error_pass) =
            slo_pass_gauges(100, 6, 1_100_000, 1_200_000, 1_000, 1_000, 0.05);
        assert_eq!((e2e_pass, queue_pass, error_pass), (0, 0, 0));
    }
}
