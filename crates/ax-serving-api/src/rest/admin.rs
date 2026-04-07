//! Admin and observability route handlers.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use ax_serving_engine::current_rss_bytes;
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;

use super::routes::{slo_pass_gauges, unix_now, AuditQuery, serving_startup_report_value};
use super::schema::*;
use crate::ServingLayer;
use crate::auth::RequestId;
use axum::extract::Extension;

/// Emit a Prometheus metric (HELP + TYPE + value) into a buffer.
macro_rules! prom {
    ($buf:expr, $name:expr, $ty:expr, $help:expr, $val:expr) => {{
        $buf.push_str(concat!("# HELP ", $name, " "));
        $buf.push_str($help);
        $buf.push_str(concat!("\n# TYPE ", $name, " ", $ty, "\n"));
        $buf.push_str(&format!(concat!($name, " {}\n"), $val));
    }};
}

/// GET /health
pub async fn health(State(layer): State<Arc<ServingLayer>>) -> Json<HealthResponse> {
    let thermal = layer.backend.thermal_state();
    let loaded_models = layer.registry.list_ids();
    let model_available = !loaded_models.is_empty();
    let ready = !matches!(thermal, ax_serving_engine::ThermalState::Critical);
    let reason = match (ready, model_available) {
        (false, false) => Some("thermal_critical_no_models"),
        (false, true) => Some("thermal_critical"),
        (true, false) => Some("no_models_loaded"),
        (true, true) => None,
    };
    let status = if ready && model_available {
        "ok"
    } else {
        "degraded"
    };

    Json(HealthResponse {
        status,
        ready,
        model_available,
        reason,
        thermal: thermal.as_str().to_string(),
        loaded_models: loaded_models.clone(),
        loaded_model_count: loaded_models.len(),
        uptime_secs: layer.metrics.uptime_secs(),
    })
}

/// `GET /v1/admin/startup-report` — authenticated runtime and config summary.
pub async fn admin_startup_report(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(serving_startup_report_value(&layer))
}

/// `GET /v1/admin/status` — authenticated operational summary for the serving runtime.
pub async fn admin_status(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
) -> impl IntoResponse {
    let thermal = layer.backend.thermal_state();
    let loaded_models = layer.registry.list_ids();
    let status = if thermal.as_str() == "Critical" || loaded_models.is_empty() {
        "degraded"
    } else {
        "ok"
    };
    let scheduler = &layer.scheduler.metrics;
    let metrics = &layer.metrics;

    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "service": "serving",
        "status": status,
        "auth_required": layer.public_auth_required.load(Ordering::Relaxed),
        "license": layer.license.to_json(),
        "runtime": {
            "rest_addr": layer.config.rest_addr,
            "grpc_socket": layer.config.grpc_socket,
            "cache_enabled": layer.cache.is_some(),
            "split_scheduler": layer.config.split_scheduler,
        },
        "models": {
            "loaded_model_count": loaded_models.len(),
            "loaded_models": loaded_models,
        },
        "scheduler": {
            "queue_depth": scheduler.queue_depth.load(Ordering::Relaxed),
            "inflight_count": scheduler.inflight_count.load(Ordering::Relaxed),
            "rejected_requests": scheduler.rejected_requests.load(Ordering::Relaxed),
            "effective_inflight_limit": layer.scheduler.effective_inflight_limit(),
        },
        "system": {
            "thermal": thermal.as_str(),
            "rss_bytes": current_rss_bytes(),
            "uptime_secs": metrics.uptime_secs(),
        }
    }))
}

/// GET /v1/metrics — scheduler and serving metrics (JSON).
pub async fn metrics(State(layer): State<Arc<ServingLayer>>) -> Json<serde_json::Value> {
    let m = &layer.scheduler.metrics;
    let cfg = layer.scheduler.config();

    // Per-model KV estimates — no last_accessed_ms update.
    let models_meta = layer.registry.loaded_models_with_meta();
    let model_kv: serde_json::Value = models_meta
        .iter()
        .map(|(id, meta)| {
            (
                id.clone(),
                serde_json::json!({ "estimated_kv_bytes": meta.estimated_kv_bytes() }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();

    // Single-snapshot percentiles: compute each histogram's tuple once so that
    // p50 ≤ p95 ≤ p99 is always maintained in the output. Three independent
    // calls to snapshot() can read different shard subsets under contention
    // (try_lock failures), making p50 > p99 possible.
    let (qw_p50, qw_p95, qw_p99) = m.queue_wait_percentiles_us();
    let (e2e_p50, e2e_p95, e2e_p99) = m.e2e_percentiles_us();
    let (ttft_p50, ttft_p95, ttft_p99) = m.ttft_percentiles_us();

    Json(serde_json::json!({
        "scheduler": {
            "queue_depth": m.queue_depth.load(Ordering::Relaxed),
            "inflight_count": m.inflight_count.load(Ordering::Relaxed),
            "total_requests": m.total_requests.load(Ordering::Relaxed),
            "rejected_requests": m.rejected_requests.load(Ordering::Relaxed),
            "queued_requests": m.queued_requests.load(Ordering::Relaxed),
            "avg_queue_wait_us": m.avg_queue_wait_us(),
            "queue_wait_p50_us": qw_p50,
            "queue_wait_p95_us": qw_p95,
            "queue_wait_p99_us": qw_p99,
            "e2e_p50_us": e2e_p50,
            "e2e_p95_us": e2e_p95,
            "e2e_p99_us": e2e_p99,
            "cache_follower_waiting": m.cache_follower_waiting.load(Ordering::Relaxed),
            "prefill_tokens_active": m.prefill_tokens_active.load(Ordering::Relaxed),
            "decode_sequences_active": m.decode_sequences_active.load(Ordering::Relaxed),
            "ttft_p50_us": ttft_p50,
            "ttft_p95_us": ttft_p95,
            "ttft_p99_us": ttft_p99,
            "effective_inflight_limit": layer.scheduler.effective_inflight_limit(),
            "adaptive_target_p99_ms": layer.scheduler.adaptive_target_p99_ms(),
            "split_scheduler_enabled": layer.scheduler.split_enabled,
            "max_inflight": cfg.max_inflight,
            "max_queue": cfg.max_queue,
            "max_wait_ms": cfg.max_wait_ms,
        },
        "uptime_secs": layer.metrics.uptime_secs(),
        "loaded_models": layer.registry.list_ids(),
        "thermal": layer.backend.thermal_state().as_str(),
        "rss_bytes": current_rss_bytes(),
        "models": model_kv,
        "cache": {
            "enabled": layer.cache.is_some(),
            "hits": layer.cache_metrics.hits.load(Ordering::Relaxed),
            "misses": layer.cache_metrics.misses.load(Ordering::Relaxed),
            "writes": layer.cache_metrics.writes.load(Ordering::Relaxed),
            "errors": layer.cache_metrics.errors.load(Ordering::Relaxed),
        },
        "request_classes": {
            "cold_requests_total": layer.metrics.cold_requests_total(),
            "exact_cache_hits_total": layer.metrics.exact_cache_hits_total(),
            "cache_follower_hits_total": layer.metrics.cache_follower_hits_total(),
            "cache_fills_total": layer.metrics.cache_fills_total(),
        }
    }))
}

/// `GET /v1/admin/diagnostics` — authenticated diagnostics bundle.
pub async fn admin_diagnostics(
    State(layer): State<Arc<ServingLayer>>,
    req_id: Option<Extension<RequestId>>,
) -> impl IntoResponse {
    let health_resp = health(State(Arc::clone(&layer))).await;
    let metrics_resp = metrics(State(Arc::clone(&layer))).await;
    let models_resp = super::models::list_models(State(Arc::clone(&layer))).await;
    Json(serde_json::json!({
        "request_id": req_id.map(|v| v.0.0).unwrap_or_default(),
        "generated_at": unix_now(),
        "startup_report": serving_startup_report_value(&layer),
        "health": health_resp.0,
        "metrics": metrics_resp.0,
        "models": models_resp.0,
        "audit_tail": layer.audit.tail(50),
    }))
}

/// `GET /v1/admin/policy` — authenticated project-policy summary.
pub async fn admin_policy(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    Json(crate::project_policy::summary_json(&layer.config.project_policy))
}

/// `GET /v1/admin/audit` — authenticated recent audit events.
pub async fn admin_audit(
    State(layer): State<Arc<ServingLayer>>,
    Query(query): Query<AuditQuery>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "events": layer.audit.tail(query.limit.clamp(1, 200)),
    }))
}

/// GET /metrics — Prometheus scrape endpoint.
///
/// Emits all serving metrics in Prometheus text format (version 0.0.4).
/// Content-Type: `text/plain; version=0.0.4`
pub async fn prometheus_metrics(State(layer): State<Arc<ServingLayer>>) -> impl IntoResponse {
    let m = &layer.scheduler.metrics;
    // Single-snapshot percentile tuples — prevents p50 > p99 under shard contention.
    let (qw_p50, qw_p95, qw_p99) = m.queue_wait_percentiles_us();
    let (e2e_p50, e2e_p95, e2e_p99) = m.e2e_percentiles_us();
    let (ttft_p50, ttft_p95, ttft_p99) = m.ttft_percentiles_us();
    let thermal_val = layer.backend.thermal_state() as u64;
    let models_meta = layer.registry.loaded_models_with_meta();
    let loaded_count = models_meta.len();

    // Pre-allocate enough for all metric lines; avoids re-allocations on the hot path.
    const PROMETHEUS_BUF_CAPACITY: usize = 2048;
    let mut buf = String::with_capacity(PROMETHEUS_BUF_CAPACITY);

    // ── Scheduler ──────────────────────────────────────────────────────────
    prom!(
        buf,
        "axs_scheduler_queue_depth",
        "gauge",
        "Current request queue depth",
        m.queue_depth.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_inflight_count",
        "gauge",
        "Active inference requests",
        m.inflight_count.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_total_requests_total",
        "counter",
        "Total requests received",
        m.total_requests.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_rejected_requests_total",
        "counter",
        "Total requests rejected (queue full)",
        m.rejected_requests.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_queued_requests_total",
        "counter",
        "Requests that entered the slow-path wait queue",
        m.queued_requests.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_avg_queue_wait_us",
        "gauge",
        "Average queue wait time in microseconds",
        m.avg_queue_wait_us()
    );
    prom!(
        buf,
        "axs_scheduler_queue_wait_p50_us",
        "gauge",
        "Rolling P50 queue wait in microseconds (slow-path only)",
        qw_p50
    );
    prom!(
        buf,
        "axs_scheduler_queue_wait_p95_us",
        "gauge",
        "Rolling P95 queue wait in microseconds (slow-path only)",
        qw_p95
    );
    prom!(
        buf,
        "axs_scheduler_queue_wait_p99_us",
        "gauge",
        "Rolling P99 queue wait in microseconds (slow-path only)",
        qw_p99
    );
    prom!(
        buf,
        "axs_scheduler_e2e_p50_us",
        "gauge",
        "Rolling P50 end-to-end latency in microseconds",
        e2e_p50
    );
    prom!(
        buf,
        "axs_scheduler_e2e_p95_us",
        "gauge",
        "Rolling P95 end-to-end latency in microseconds",
        e2e_p95
    );
    prom!(
        buf,
        "axs_scheduler_e2e_p99_us",
        "gauge",
        "Rolling P99 end-to-end latency in microseconds",
        e2e_p99
    );
    prom!(
        buf,
        "axs_cache_follower_waiting",
        "gauge",
        "Cache followers currently waiting pre-permit (WS3)",
        m.cache_follower_waiting.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_prefill_tokens_active",
        "gauge",
        "Estimated prompt tokens currently in prefill",
        m.prefill_tokens_active.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_scheduler_decode_sequences_active",
        "gauge",
        "Active sequences currently in decode",
        m.decode_sequences_active.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_ttft_p50_us",
        "gauge",
        "Rolling P50 time-to-first-token in microseconds (streaming only)",
        ttft_p50
    );
    prom!(
        buf,
        "axs_ttft_p95_us",
        "gauge",
        "Rolling P95 time-to-first-token in microseconds (streaming only)",
        ttft_p95
    );
    prom!(
        buf,
        "axs_ttft_p99_us",
        "gauge",
        "Rolling P99 time-to-first-token in microseconds (streaming only)",
        ttft_p99
    );
    prom!(
        buf,
        "axs_adaptive_inflight_limit",
        "gauge",
        "Effective inflight limit (adaptive or static)",
        layer.scheduler.effective_inflight_limit()
    );
    prom!(
        buf,
        "axs_adaptive_target_p99_ms",
        "gauge",
        "AIMD target P99 latency in milliseconds (0 = disabled)",
        layer.scheduler.adaptive_target_p99_ms().unwrap_or(0)
    );
    prom!(
        buf,
        "axs_request_class_cold_requests_total",
        "counter",
        "Requests that executed inference without a cache result",
        layer.metrics.cold_requests_total()
    );
    prom!(
        buf,
        "axs_request_class_exact_cache_hits_total",
        "counter",
        "Requests served immediately from exact response-cache hits",
        layer.metrics.exact_cache_hits_total()
    );
    prom!(
        buf,
        "axs_request_class_cache_follower_hits_total",
        "counter",
        "Requests served from follower waits after a leader cache fill",
        layer.metrics.cache_follower_hits_total()
    );
    prom!(
        buf,
        "axs_request_class_cache_fills_total",
        "counter",
        "Successful exact response-cache writes after inference",
        layer.metrics.cache_fills_total()
    );

    // ── Thermal ────────────────────────────────────────────────────────────
    prom!(
        buf,
        "axs_thermal_state",
        "gauge",
        "Thermal pressure state (0=Nominal 1=Fair 2=Serious 3=Critical)",
        thermal_val
    );

    // ── Cache ──────────────────────────────────────────────────────────────
    prom!(
        buf,
        "axs_cache_hits_total",
        "counter",
        "Response cache hits",
        layer.cache_metrics.hits.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_cache_misses_total",
        "counter",
        "Response cache misses",
        layer.cache_metrics.misses.load(Ordering::Relaxed)
    );
    prom!(
        buf,
        "axs_cache_writes_total",
        "counter",
        "Response cache writes",
        layer.cache_metrics.writes.load(Ordering::Relaxed)
    );

    {
        let hits = layer.cache_metrics.hits.load(Ordering::Relaxed);
        let misses = layer.cache_metrics.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let ratio = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        buf.push_str(
            "# HELP axs_cache_hit_ratio Response cache hit ratio since startup (0.0–1.0)\n",
        );
        buf.push_str("# TYPE axs_cache_hit_ratio gauge\n");
        buf.push_str(&format!("axs_cache_hit_ratio {ratio:.4}\n"));
    }

    // ── System ────────────────────────────────────────────────────────────
    prom!(
        buf,
        "axs_uptime_seconds",
        "counter",
        "Seconds since server start",
        layer.metrics.uptime_secs()
    );
    prom!(
        buf,
        "axs_rss_bytes",
        "gauge",
        "Process resident set size in bytes",
        current_rss_bytes()
    );

    // ── Models ────────────────────────────────────────────────────────────
    prom!(
        buf,
        "axs_loaded_models_total",
        "gauge",
        "Number of currently loaded models",
        loaded_count
    );

    buf.push_str("# HELP axs_model_kv_bytes_estimated Estimated KV cache bytes for full context\n");
    buf.push_str("# TYPE axs_model_kv_bytes_estimated gauge\n");
    for (id, meta) in &models_meta {
        buf.push_str(&format!(
            "axs_model_kv_bytes_estimated{{model=\"{id}\"}} {}\n",
            meta.estimated_kv_bytes()
        ));
    }

    // ── SLO alerting gauges ────────────────────────────────────────────────
    // 1 = SLO met (pass), 0 = SLO violated (fail).
    // Thresholds are read from env at scrape time so they can be tuned without
    // restarting the server (only takes effect on the next scrape).
    let slo_e2e_p99_ms = std::env::var("AXS_SLO_E2E_P99_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5_000);
    let slo_queue_p99_ms = std::env::var("AXS_SLO_QUEUE_P99_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1_000);
    let slo_max_error_rate = std::env::var("AXS_SLO_MAX_ERROR_RATE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.05);

    let total = m.total_requests.load(Ordering::Relaxed);
    let rejected = m.rejected_requests.load(Ordering::Relaxed);
    let (e2e_pass, queue_pass, error_pass) = slo_pass_gauges(
        total,
        rejected,
        e2e_p99,
        qw_p99,
        slo_e2e_p99_ms,
        slo_queue_p99_ms,
        slo_max_error_rate,
    );

    prom!(
        buf,
        "axs_slo_e2e_p99_pass",
        "gauge",
        "1 if e2e P99 latency is within SLO, 0 otherwise",
        e2e_pass
    );
    prom!(
        buf,
        "axs_slo_queue_p99_pass",
        "gauge",
        "1 if queue-wait P99 is within SLO, 0 otherwise",
        queue_pass
    );
    prom!(
        buf,
        "axs_slo_error_rate_pass",
        "gauge",
        "1 if rejection rate is within SLO, 0 otherwise",
        error_pass
    );

    // ── Burn-rate alerting ────────────────────────────────────────────────────
    // Multi-window SLO burn rate.  error_budget = 0.001 (99.9% availability).
    // Fast burn: 1h window > 14.4× → 2% budget in 1h   (Google SRE chapter 5).
    // Slow burn: 6h window >  6.0× → 5% budget in 6h.
    let burn_1h = match layer.metrics.burn_1h.lock() {
        Ok(metric) => metric.burn_rate(0.001),
        Err(err) => {
            tracing::warn!(%err, "burn-1h metric lock poisoned; continuing with poisoned state");
            err.into_inner().burn_rate(0.001)
        }
    };
    let burn_6h = match layer.metrics.burn_6h.lock() {
        Ok(metric) => metric.burn_rate(0.001),
        Err(err) => {
            tracing::warn!(%err, "burn-6h metric lock poisoned; continuing with poisoned state");
            err.into_inner().burn_rate(0.001)
        }
    };
    let burn_alert = u8::from((burn_1h > 14.4) || (burn_6h > 6.0));

    prom!(
        buf,
        "axs_slo_burn_rate_1h",
        "gauge",
        "SLO error burn rate over 1-hour sliding window",
        burn_1h
    );
    prom!(
        buf,
        "axs_slo_burn_rate_6h",
        "gauge",
        "SLO error burn rate over 6-hour sliding window",
        burn_6h
    );
    prom!(
        buf,
        "axs_slo_burn_rate_alert",
        "gauge",
        "1 if multi-window burn-rate alert is firing, 0 otherwise",
        burn_alert
    );

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        buf,
    )
}
