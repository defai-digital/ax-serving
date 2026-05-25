//! AX Code support contracts: config validation, status, and smoke tests.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::blocking::{Client, RequestBuilder};
use serde::Serialize;
use serde_json::Value;

use crate::output::{emit_json_or_human, exit_if};
use ax_serving_api::config::ServeConfig;

const DEFAULT_TIMEOUT_SECS: u64 = 10;
const COMMAND_CONFIG_VALIDATE: &str = "ax-serving config validate";
const COMMAND_STATUS: &str = "ax-serving status";
const COMMAND_SMOKE_TEST: &str = "ax-serving smoke-test";
const STATUS_OK: &str = "ok";
const STATUS_FAIL: &str = "fail";
const STATUS_DEGRADED: &str = "degraded";
const STATUS_UNREACHABLE: &str = "unreachable";

#[derive(Debug, Serialize)]
struct ConfigValidateReport {
    command: &'static str,
    status: &'static str,
    source: String,
    valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<ConfigSummary>,
}

#[derive(Debug, Serialize)]
struct ConfigSummary {
    rest_addr: String,
    grpc_socket: String,
    grpc_port: Option<u16>,
    sched_max_inflight: usize,
    sched_max_queue: usize,
    sched_max_wait_ms: u64,
    sched_per_model_max_inflight: usize,
    default_max_tokens: u32,
    split_scheduler: bool,
    cache_enabled: bool,
    orchestrator_public: String,
    orchestrator_internal: String,
    dispatch_policy: String,
    project_policy_enabled: bool,
}

#[derive(Debug, Serialize)]
struct EndpointReport {
    name: &'static str,
    url: String,
    ok: bool,
    status_code: Option<u16>,
    latency_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct StatusReport {
    command: &'static str,
    base_url: String,
    status: &'static str,
    reachable: bool,
    endpoints: Vec<EndpointReport>,
}

#[derive(Debug, Serialize)]
struct SmokeTestReport {
    command: &'static str,
    base_url: String,
    model: String,
    status: &'static str,
    ok: bool,
    status_code: Option<u16>,
    latency_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub fn run_config_validate(config: Option<PathBuf>, json: bool) -> Result<()> {
    let report = build_config_validate_report(config);

    emit_json_or_human(json, &report, print_config_validate_human)?;
    exit_if(!report.valid);
    Ok(())
}

fn build_config_validate_report(config: Option<PathBuf>) -> ConfigValidateReport {
    let (source, loaded) = load_config_for_validation(config);

    let (cfg, load_error) = match loaded {
        Ok(cfg) => (Some(cfg), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let validation_error = load_error.or_else(|| {
        cfg.as_ref()
            .and_then(|cfg| cfg.validate().err().map(|e| e.to_string()))
    });
    let valid = validation_error.is_none();
    ConfigValidateReport {
        command: COMMAND_CONFIG_VALIDATE,
        status: if valid { STATUS_OK } else { STATUS_FAIL },
        source,
        valid,
        error: validation_error,
        summary: cfg.as_ref().map(config_summary),
    }
}

fn load_config_for_validation(config: Option<PathBuf>) -> (String, Result<ServeConfig>) {
    if let Some(path) = config {
        return (path.display().to_string(), ServeConfig::from_file(&path));
    }

    for path in default_config_candidates() {
        if path.exists() {
            return (path.display().to_string(), ServeConfig::from_file(&path));
        }
    }

    (
        "environment/defaults".to_string(),
        Ok(ServeConfig::from_env()),
    )
}

pub fn run_status(url: String, api_key: Option<String>, json: bool) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let endpoints = vec![
        get_json_endpoint(&client, &base_url, "/health", token.as_deref(), "health"),
        get_json_endpoint(&client, &base_url, "/v1/models", token.as_deref(), "models"),
        get_json_endpoint(
            &client,
            &base_url,
            "/v1/metrics",
            token.as_deref(),
            "metrics",
        ),
    ];
    let (reachable, status) = status_from_endpoints(&endpoints);
    let report = StatusReport {
        command: COMMAND_STATUS,
        base_url,
        status,
        reachable,
        endpoints,
    };

    emit_json_or_human(json, &report, print_status_human)?;
    exit_if(!report.reachable);
    Ok(())
}

pub fn run_smoke_test(
    url: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    api_key: Option<String>,
    json: bool,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let endpoint = format!("{base_url}/v1/chat/completions");
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": max_tokens,
    });

    let response = send_json_request(client.post(&endpoint).json(&body), token.as_deref());
    let report = SmokeTestReport {
        command: COMMAND_SMOKE_TEST,
        base_url,
        model,
        status: if response.ok {
            STATUS_OK
        } else if response.status_code.is_some() {
            STATUS_FAIL
        } else {
            STATUS_UNREACHABLE
        },
        ok: response.ok,
        status_code: response.status_code,
        latency_ms: response.latency_ms,
        response: response.body,
        error: response.error,
    };

    emit_json_or_human(json, &report, print_smoke_test_human)?;
    exit_if(!report.ok);
    Ok(())
}

fn support_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
        .context("failed to build support HTTP client")
}

fn get_json_endpoint(
    client: &Client,
    base_url: &str,
    path: &'static str,
    token: Option<&str>,
    name: &'static str,
) -> EndpointReport {
    let url = format!("{base_url}{path}");
    let response = send_json_request(client.get(&url), token);

    EndpointReport {
        name,
        url,
        ok: response.ok,
        status_code: response.status_code,
        latency_ms: response.latency_ms,
        body: response.body,
        error: response.error,
    }
}

struct JsonHttpResponse {
    ok: bool,
    status_code: Option<u16>,
    latency_ms: u128,
    body: Option<Value>,
    error: Option<String>,
}

fn send_json_request(request: RequestBuilder, token: Option<&str>) -> JsonHttpResponse {
    let started = Instant::now();
    let result = with_optional_bearer(request, token).send();
    let latency_ms = started.elapsed().as_millis();

    match result {
        Ok(resp) => JsonHttpResponse {
            ok: resp.status().is_success(),
            status_code: Some(resp.status().as_u16()),
            latency_ms,
            body: resp.json::<Value>().ok(),
            error: None,
        },
        Err(e) => JsonHttpResponse {
            ok: false,
            status_code: None,
            latency_ms,
            body: None,
            error: Some(e.to_string()),
        },
    }
}

fn status_from_endpoints(endpoints: &[EndpointReport]) -> (bool, &'static str) {
    let reachable = endpoints.iter().any(|e| e.status_code.is_some());
    if !reachable {
        return (false, STATUS_UNREACHABLE);
    }
    if endpoints.iter().any(|e| !e.ok) || health_endpoint_degraded(endpoints) {
        return (true, STATUS_DEGRADED);
    }
    (true, STATUS_OK)
}

fn health_endpoint_degraded(endpoints: &[EndpointReport]) -> bool {
    let Some(health) = endpoints.iter().find(|e| e.name == "health") else {
        return false;
    };
    let Some(body) = health.body.as_ref() else {
        return false;
    };
    body.get("status")
        .and_then(Value::as_str)
        .is_some_and(|s| s != STATUS_OK)
        || body.get("ready").and_then(Value::as_bool) == Some(false)
}

fn with_optional_bearer(request: RequestBuilder, token: Option<&str>) -> RequestBuilder {
    match token {
        Some(token) if !token.trim().is_empty() => request.bearer_auth(token.trim()),
        _ => request,
    }
}

fn config_summary(cfg: &ServeConfig) -> ConfigSummary {
    ConfigSummary {
        rest_addr: cfg.rest_addr.clone(),
        grpc_socket: cfg.grpc_socket.clone(),
        grpc_port: cfg.grpc_port,
        sched_max_inflight: cfg.sched_max_inflight,
        sched_max_queue: cfg.sched_max_queue,
        sched_max_wait_ms: cfg.sched_max_wait_ms,
        sched_per_model_max_inflight: cfg.sched_per_model_max_inflight,
        default_max_tokens: cfg.default_max_tokens,
        split_scheduler: cfg.split_scheduler,
        cache_enabled: cfg.cache.enabled,
        orchestrator_public: format!("{}:{}", cfg.orchestrator.host, cfg.orchestrator.port),
        orchestrator_internal: format!(
            "{}:{}",
            cfg.orchestrator.internal_bind_addr, cfg.orchestrator.internal_port
        ),
        dispatch_policy: cfg.orchestrator.dispatch_policy.clone(),
        project_policy_enabled: cfg.project_policy.enabled,
    }
}

fn print_config_validate_human(report: &ConfigValidateReport) {
    eprintln!("AX Serving Config Validate\n");
    eprintln!("  Source: {}", report.source);
    eprintln!("  Status: {}", report.status);
    if let Some(error) = &report.error {
        eprintln!("  Error:  {error}");
    }
    if let Some(summary) = &report.summary {
        eprintln!("  REST:   {}", summary.rest_addr);
        eprintln!(
            "  Queue:  inflight={} queue={} wait_ms={}",
            summary.sched_max_inflight, summary.sched_max_queue, summary.sched_max_wait_ms
        );
    }
}

fn print_status_human(report: &StatusReport) {
    eprintln!("AX Serving Status\n");
    eprintln!("  Base URL: {}", report.base_url);
    eprintln!("  Status:   {}", report.status);
    for endpoint in &report.endpoints {
        let status = endpoint
            .status_code
            .map(|code| code.to_string())
            .unwrap_or_else(|| "no response".to_string());
        eprintln!(
            "  [{}] {}: {} ({} ms)",
            if endpoint.ok { "OK" } else { "FAIL" },
            endpoint.name,
            status,
            endpoint.latency_ms,
        );
        if let Some(error) = &endpoint.error {
            eprintln!("         {error}");
        }
    }
}

fn print_smoke_test_human(report: &SmokeTestReport) {
    eprintln!("AX Serving Smoke Test\n");
    eprintln!("  Base URL: {}", report.base_url);
    eprintln!("  Model:    {}", report.model);
    eprintln!("  Status:   {}", report.status);
    eprintln!("  Latency:  {} ms", report.latency_ms);
    if let Some(status_code) = report.status_code {
        eprintln!("  HTTP:     {status_code}");
    }
    if let Some(error) = &report.error {
        eprintln!("  Error:    {error}");
    }
}

fn normalize_base_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

fn effective_api_key(api_key: Option<String>) -> Option<String> {
    api_key.and_then(trimmed_non_empty).or_else(|| {
        std::env::var("AXS_API_KEY").ok().and_then(|value| {
            value
                .split(',')
                .find_map(|part| trimmed_non_empty(part.to_string()))
        })
    })
}

fn trimmed_non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn default_config_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("AXS_CONFIG") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("config/serving.yaml"));
    candidates.push(PathBuf::from("serving.yaml"));
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".config/ax-serving/serving.yaml"));
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_base_url() {
        assert_eq!(
            normalize_base_url("127.0.0.1:18080/"),
            "http://127.0.0.1:18080"
        );
        assert_eq!(
            normalize_base_url("https://example.test/base/"),
            "https://example.test/base"
        );
    }

    #[test]
    fn config_summary_preserves_key_fields() {
        let cfg = ServeConfig::default();
        let summary = config_summary(&cfg);

        assert_eq!(summary.rest_addr, "127.0.0.1:18080");
        assert_eq!(summary.dispatch_policy, "least_inflight");
        assert_eq!(summary.sched_max_inflight, 16);
    }

    #[test]
    fn malformed_explicit_config_does_not_emit_fallback_summary() {
        let path =
            std::env::temp_dir().join(format!("ax-serving-bad-config-{}.toml", std::process::id()));
        std::fs::write(&path, "=").unwrap();

        let report = build_config_validate_report(Some(path.clone()));
        let _ = std::fs::remove_file(path);

        assert!(!report.valid);
        assert_eq!(report.status, STATUS_FAIL);
        assert!(report.error.is_some());
        assert!(report.summary.is_none());
    }

    #[test]
    fn explicit_api_key_wins_without_env_lookup() {
        assert_eq!(
            effective_api_key(Some("  token-a  ".into())).as_deref(),
            Some("token-a")
        );
    }

    #[test]
    fn status_reflects_degraded_health_body() {
        let endpoints = vec![EndpointReport {
            name: "health",
            url: "http://127.0.0.1:18080/health".into(),
            ok: true,
            status_code: Some(200),
            latency_ms: 1,
            body: Some(serde_json::json!({
                "status": "degraded",
                "ready": true,
                "reason": "no_models_loaded"
            })),
            error: None,
        }];

        let (reachable, status) = status_from_endpoints(&endpoints);

        assert!(reachable);
        assert_eq!(status, STATUS_DEGRADED);
    }

    #[test]
    fn status_reflects_unreachable_endpoints() {
        let endpoints = vec![EndpointReport {
            name: "health",
            url: "http://127.0.0.1:9/health".into(),
            ok: false,
            status_code: None,
            latency_ms: 1,
            body: None,
            error: Some("connection refused".into()),
        }];

        let (reachable, status) = status_from_endpoints(&endpoints);

        assert!(!reachable);
        assert_eq!(status, STATUS_UNREACHABLE);
    }
}
