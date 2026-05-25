//! AX Code support contracts: config validation, status, and smoke tests.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use reqwest::blocking::{Client, RequestBuilder};
use serde::Serialize;
use serde_json::Value;

use crate::output::{emit_json, emit_json_or_human, exit_if};
use ax_serving_api::config::ServeConfig;

const DEFAULT_TIMEOUT_SECS: u64 = 10;
const COMMAND_CONFIG_VALIDATE: &str = "ax-serving config validate";
const COMMAND_STATUS: &str = "ax-serving status";
const COMMAND_SMOKE_TEST: &str = "ax-serving smoke-test";
const COMMAND_SUPPORT_BUNDLE: &str = "ax-serving support-bundle";
const COMMAND_FABRIC_VALIDATE: &str = "ax-serving fabric validate";
const COMMAND_MIGRATION_EMBEDDED_READINESS: &str = "ax-serving migration embedded-readiness";
const COMMAND_WORKERS: &str = "ax-serving workers";
const ENDPOINT_DIAGNOSTICS: &str = "diagnostics";
const STATUS_OK: &str = "ok";
const STATUS_FAIL: &str = "fail";
const STATUS_DEGRADED: &str = "degraded";
const STATUS_UNREACHABLE: &str = "unreachable";
const FABRIC_PROFILE_GATEWAY: &str = "gateway";
const FABRIC_PROFILE_SINGLE_RUNTIME: &str = "single_runtime";
const FABRIC_PROFILE_UNKNOWN: &str = "unknown";

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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    recommended_actions: Vec<StatusRecommendedAction>,
}

#[derive(Debug, PartialEq, Serialize)]
struct StatusRecommendedAction {
    action: String,
    runtime: Option<String>,
    reason: Option<String>,
    operator_hint: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    suggested_commands: Vec<String>,
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

#[derive(Debug, Serialize)]
struct WorkerLifecycleReport {
    command: &'static str,
    base_url: String,
    operation: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker_id: Option<String>,
    status: &'static str,
    ok: bool,
    steps: Vec<EndpointReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct FabricValidateReport {
    command: &'static str,
    base_url: String,
    status: &'static str,
    ready: bool,
    profile: &'static str,
    endpoints: Vec<EndpointReport>,
    checks: Vec<FabricCheck>,
}

#[derive(Debug, Serialize)]
struct FabricCheck {
    name: &'static str,
    ok: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct MigrationEmbeddedReadinessReport {
    command: &'static str,
    base_url: String,
    status: &'static str,
    ready_to_deny: bool,
    recommended_policy: &'static str,
    diagnostics: EndpointReport,
    totals: MigrationReadinessTotals,
    runtimes: Vec<MigrationRuntimeSummary>,
    blockers: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
struct MigrationReadinessTotals {
    workers: usize,
    adapter_workers: usize,
    embedded_workers: usize,
    unknown_mode_workers: usize,
    eligible_workers: usize,
}

#[derive(Debug, Serialize)]
struct MigrationRuntimeSummary {
    runtime: String,
    workers: usize,
    eligible: usize,
    adapter_workers: usize,
    embedded_workers: usize,
    unknown_mode_workers: usize,
    models: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SupportBundleReport {
    command: &'static str,
    base_url: String,
    status: &'static str,
    reachable: bool,
    generated_at_unix_ms: u128,
    redaction: &'static str,
    endpoints: Vec<EndpointReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
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

pub fn run_status(
    url: String,
    api_key: Option<String>,
    diagnostics: bool,
    json: bool,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let mut endpoints = vec![
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
    if diagnostics {
        endpoints.push(get_json_endpoint(
            &client,
            &base_url,
            "/v1/admin/diagnostics",
            token.as_deref(),
            ENDPOINT_DIAGNOSTICS,
        ));
    }
    let (reachable, status) = status_from_endpoints(&endpoints);
    let recommended_actions = diagnostics_recommended_actions(&endpoints);
    let report = StatusReport {
        command: COMMAND_STATUS,
        base_url,
        status,
        reachable,
        endpoints,
        recommended_actions,
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

pub fn run_support_bundle(
    url: String,
    api_key: Option<String>,
    output: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let mut endpoints = support_bundle_endpoints()
        .into_iter()
        .map(|(name, path)| get_json_endpoint(&client, &base_url, path, token.as_deref(), name))
        .collect::<Vec<_>>();
    for endpoint in &mut endpoints {
        if let Some(body) = &mut endpoint.body {
            redact_sensitive_value(body);
        }
    }
    let (reachable, status) = status_from_endpoints(&endpoints);
    let output_label = output.as_ref().map(|path| path.display().to_string());
    let report = SupportBundleReport {
        command: COMMAND_SUPPORT_BUNDLE,
        base_url,
        status,
        reachable,
        generated_at_unix_ms: current_unix_ms(),
        redaction: "recursive sensitive-key redaction applied",
        endpoints,
        output: output_label.clone(),
    };

    if let Some(path) = output {
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("failed to write support bundle {}", path.display()))?;
    }
    if json {
        emit_json(&report)?;
    } else {
        print_support_bundle_human(&report);
    }
    exit_if(!report.reachable);
    Ok(())
}

pub fn run_fabric_validate(url: String, api_key: Option<String>, json: bool) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let endpoints = fabric_validate_endpoints()
        .into_iter()
        .map(|(name, path)| get_json_endpoint(&client, &base_url, path, token.as_deref(), name))
        .collect::<Vec<_>>();
    let report = build_fabric_validate_report(base_url, endpoints);

    emit_json_or_human(json, &report, print_fabric_validate_human)?;
    exit_if(!report.ready);
    Ok(())
}

pub fn run_migration_embedded_readiness(
    url: String,
    api_key: Option<String>,
    json: bool,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let diagnostics = get_json_endpoint(
        &client,
        &base_url,
        "/v1/admin/diagnostics",
        token.as_deref(),
        ENDPOINT_DIAGNOSTICS,
    );
    let report = build_migration_embedded_readiness_report(base_url, diagnostics);

    emit_json_or_human(json, &report, print_migration_embedded_readiness_human)?;
    exit_if(!report.ready_to_deny);
    Ok(())
}

pub fn run_workers_list(url: String, api_key: Option<String>, json: bool) -> Result<()> {
    run_worker_get_like(url, None, api_key, "list", "/v1/workers".to_string(), json)
}

pub fn run_worker_get(
    url: String,
    worker_id: String,
    api_key: Option<String>,
    json: bool,
) -> Result<()> {
    run_worker_get_like(
        url,
        Some(worker_id.clone()),
        api_key,
        "get",
        worker_path(&worker_id, ""),
        json,
    )
}

pub fn run_worker_drain(
    url: String,
    worker_id: String,
    api_key: Option<String>,
    complete_when_idle: bool,
    idle_timeout_secs: u64,
    poll_interval_ms: u64,
    json: bool,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let mut steps = Vec::new();

    steps.push(send_worker_step(
        &client,
        &base_url,
        &worker_path(&worker_id, "/drain"),
        token.as_deref(),
        "drain",
        WorkerHttpMethod::Post,
    ));

    let mut workflow_error = None;
    if steps.last().is_some_and(|step| step.ok) && complete_when_idle {
        workflow_error = wait_for_worker_idle_and_complete(
            &client,
            &base_url,
            &worker_id,
            token.as_deref(),
            idle_timeout_secs,
            poll_interval_ms,
            &mut steps,
        );
    }

    let report = worker_lifecycle_report(base_url, "drain", Some(worker_id), steps, workflow_error);
    emit_json_or_human(json, &report, print_worker_lifecycle_human)?;
    exit_if(!report.ok);
    Ok(())
}

pub fn run_worker_drain_complete(
    url: String,
    worker_id: String,
    api_key: Option<String>,
    json: bool,
) -> Result<()> {
    run_worker_mutation(
        url,
        worker_id,
        api_key,
        json,
        "drain-complete",
        "/drain-complete",
        WorkerHttpMethod::Post,
    )
}

pub fn run_worker_remove(
    url: String,
    worker_id: String,
    api_key: Option<String>,
    json: bool,
) -> Result<()> {
    run_worker_mutation(
        url,
        worker_id,
        api_key,
        json,
        "remove",
        "",
        WorkerHttpMethod::Delete,
    )
}

fn support_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
        .context("failed to build support HTTP client")
}

fn run_worker_get_like(
    url: String,
    worker_id: Option<String>,
    api_key: Option<String>,
    operation: &'static str,
    path: String,
    json: bool,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let steps = vec![send_worker_step(
        &client,
        &base_url,
        &path,
        token.as_deref(),
        operation,
        WorkerHttpMethod::Get,
    )];
    let report = worker_lifecycle_report(base_url, operation, worker_id, steps, None);
    emit_json_or_human(json, &report, print_worker_lifecycle_human)?;
    exit_if(!report.ok);
    Ok(())
}

fn run_worker_mutation(
    url: String,
    worker_id: String,
    api_key: Option<String>,
    json: bool,
    operation: &'static str,
    suffix: &str,
    method: WorkerHttpMethod,
) -> Result<()> {
    let base_url = normalize_base_url(&url);
    let client = support_client()?;
    let token = effective_api_key(api_key);
    let steps = vec![send_worker_step(
        &client,
        &base_url,
        &worker_path(&worker_id, suffix),
        token.as_deref(),
        operation,
        method,
    )];
    let report = worker_lifecycle_report(base_url, operation, Some(worker_id), steps, None);
    emit_json_or_human(json, &report, print_worker_lifecycle_human)?;
    exit_if(!report.ok);
    Ok(())
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

fn support_bundle_endpoints() -> Vec<(&'static str, &'static str)> {
    vec![
        ("health", "/health"),
        ("models", "/v1/models"),
        ("metrics", "/v1/metrics"),
        ("admin_status", "/v1/admin/status"),
        ("diagnostics", "/v1/admin/diagnostics"),
        ("fleet", "/v1/admin/fleet"),
        ("workers", "/v1/workers"),
        ("audit", "/v1/admin/audit?limit=50"),
    ]
}

fn fabric_validate_endpoints() -> Vec<(&'static str, &'static str)> {
    vec![
        ("health", "/health"),
        ("models", "/v1/models"),
        ("metrics", "/v1/metrics"),
    ]
}

#[derive(Clone, Copy)]
enum WorkerHttpMethod {
    Get,
    Post,
    Delete,
}

fn send_worker_step(
    client: &Client,
    base_url: &str,
    path: &str,
    token: Option<&str>,
    name: &'static str,
    method: WorkerHttpMethod,
) -> EndpointReport {
    let url = format!("{base_url}{path}");
    let request = match method {
        WorkerHttpMethod::Get => client.get(&url),
        WorkerHttpMethod::Post => client.post(&url),
        WorkerHttpMethod::Delete => client.delete(&url),
    };
    let response = send_json_request(request, token);

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

fn wait_for_worker_idle_and_complete(
    client: &Client,
    base_url: &str,
    worker_id: &str,
    token: Option<&str>,
    idle_timeout_secs: u64,
    poll_interval_ms: u64,
    steps: &mut Vec<EndpointReport>,
) -> Option<String> {
    let deadline = Instant::now() + Duration::from_secs(idle_timeout_secs);
    let poll_interval = Duration::from_millis(poll_interval_ms.max(1));
    loop {
        let inspect = send_worker_step(
            client,
            base_url,
            &worker_path(worker_id, ""),
            token,
            "wait-idle",
            WorkerHttpMethod::Get,
        );
        let inflight = worker_inflight(inspect.body.as_ref());
        let inspect_ok = inspect.ok;
        steps.push(inspect);

        if !inspect_ok {
            return Some("failed to inspect worker while waiting for idle".to_string());
        }
        if inflight == Some(0) {
            let step = send_worker_step(
                client,
                base_url,
                &worker_path(worker_id, "/drain-complete"),
                token,
                "drain-complete",
                WorkerHttpMethod::Post,
            );
            let drain_ok = step.ok;
            let status_code = step.status_code;
            steps.push(step);
            return if drain_ok {
                None
            } else {
                Some(format!(
                    "drain-complete failed (status {})",
                    status_code
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "no response".to_string())
                ))
            };
        }
        if Instant::now() >= deadline {
            return Some(format!(
                "worker did not become idle within {idle_timeout_secs}s"
            ));
        }
        std::thread::sleep(poll_interval);
    }
}

fn worker_lifecycle_report(
    base_url: String,
    operation: &'static str,
    worker_id: Option<String>,
    steps: Vec<EndpointReport>,
    error: Option<String>,
) -> WorkerLifecycleReport {
    let ok = error.is_none() && steps.iter().all(|step| step.ok);
    let reachable = steps.iter().any(|step| step.status_code.is_some());
    let status = if ok {
        STATUS_OK
    } else if reachable {
        STATUS_FAIL
    } else {
        STATUS_UNREACHABLE
    };
    WorkerLifecycleReport {
        command: COMMAND_WORKERS,
        base_url,
        operation,
        worker_id,
        status,
        ok,
        steps,
        error,
    }
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

fn build_fabric_validate_report(
    base_url: String,
    endpoints: Vec<EndpointReport>,
) -> FabricValidateReport {
    let profile = fabric_profile(&endpoints);
    let checks = fabric_validate_checks(&endpoints, profile);
    let reachable = endpoints.iter().any(|e| e.status_code.is_some());
    let ready = reachable && checks.iter().all(|check| check.ok);
    let status = if ready {
        STATUS_OK
    } else if reachable {
        STATUS_FAIL
    } else {
        STATUS_UNREACHABLE
    };

    FabricValidateReport {
        command: COMMAND_FABRIC_VALIDATE,
        base_url,
        status,
        ready,
        profile,
        endpoints,
        checks,
    }
}

fn fabric_validate_checks(endpoints: &[EndpointReport], profile: &'static str) -> Vec<FabricCheck> {
    let health = endpoints.iter().find(|e| e.name == "health");
    let models = endpoints.iter().find(|e| e.name == "models");
    let metrics = endpoints.iter().find(|e| e.name == "metrics");
    let mut checks = Vec::new();

    checks.push(FabricCheck {
        name: "health_http_200",
        ok: health.is_some_and(|e| e.status_code == Some(200)),
        detail: endpoint_check_detail(health, "GET /health returned HTTP 200"),
    });
    checks.push(FabricCheck {
        name: "health_status_known",
        ok: health
            .and_then(|e| e.body.as_ref())
            .and_then(|body| body.get("status"))
            .and_then(Value::as_str)
            .is_some_and(|status| status == STATUS_OK || status == STATUS_DEGRADED),
        detail: "GET /health status is ok or degraded".to_string(),
    });
    checks.push(FabricCheck {
        name: "health_readiness_signal",
        ok: health
            .and_then(|e| e.body.as_ref())
            .is_some_and(has_fabric_readiness_signal),
        detail: "GET /health exposes ready or workers.eligible readiness signal".to_string(),
    });
    checks.push(FabricCheck {
        name: "models_http_200",
        ok: models.is_some_and(|e| e.status_code == Some(200)),
        detail: endpoint_check_detail(models, "GET /v1/models returned HTTP 200"),
    });
    checks.push(FabricCheck {
        name: "models_data_array",
        ok: models
            .and_then(|e| e.body.as_ref())
            .and_then(|body| body.get("data"))
            .and_then(Value::as_array)
            .is_some(),
        detail: "GET /v1/models exposes OpenAI-compatible data array".to_string(),
    });
    checks.push(FabricCheck {
        name: "metrics_http_200",
        ok: metrics.is_some_and(|e| e.status_code == Some(200)),
        detail: endpoint_check_detail(metrics, "GET /v1/metrics returned HTTP 200"),
    });
    checks.push(FabricCheck {
        name: "metrics_contract_profile",
        ok: profile != FABRIC_PROFILE_UNKNOWN,
        detail: format!("GET /v1/metrics profile detected as {profile}"),
    });

    if let Some(metrics_body) = metrics.and_then(|e| e.body.as_ref()) {
        match profile {
            FABRIC_PROFILE_SINGLE_RUNTIME => checks.extend(
                missing_fabric_single_runtime_metric_keys(metrics_body)
                    .into_iter()
                    .map(|key| FabricCheck {
                        name: "metrics_single_runtime_key",
                        ok: false,
                        detail: format!("GET /v1/metrics missing {key}"),
                    }),
            ),
            FABRIC_PROFILE_GATEWAY => checks.extend(
                missing_fabric_gateway_metric_keys(metrics_body)
                    .into_iter()
                    .map(|key| FabricCheck {
                        name: "metrics_gateway_key",
                        ok: false,
                        detail: format!("GET /v1/metrics missing {key}"),
                    }),
            ),
            _ => {}
        }
    }

    checks
}

fn fabric_profile(endpoints: &[EndpointReport]) -> &'static str {
    let Some(metrics_body) = endpoints
        .iter()
        .find(|e| e.name == "metrics")
        .and_then(|e| e.body.as_ref())
    else {
        return FABRIC_PROFILE_UNKNOWN;
    };

    if missing_fabric_single_runtime_metric_keys(metrics_body).is_empty() {
        return FABRIC_PROFILE_SINGLE_RUNTIME;
    }
    if missing_fabric_gateway_metric_keys(metrics_body).is_empty() {
        return FABRIC_PROFILE_GATEWAY;
    }
    FABRIC_PROFILE_UNKNOWN
}

fn missing_fabric_single_runtime_metric_keys(body: &Value) -> Vec<&'static str> {
    [
        "scheduler.queue_depth",
        "scheduler.inflight_count",
        "scheduler.cache_follower_waiting",
        "scheduler.ttft_p50_us",
        "scheduler.ttft_p95_us",
        "scheduler.ttft_p99_us",
        "scheduler.prefill_tokens_active",
        "scheduler.decode_sequences_active",
        "scheduler.split_scheduler_enabled",
        "loaded_models",
        "thermal",
    ]
    .into_iter()
    .filter(|key| !json_key_exists(body, key))
    .collect()
}

fn missing_fabric_gateway_metric_keys(body: &Value) -> Vec<&'static str> {
    [
        "mode",
        "policy",
        "workers.healthy",
        "workers.unhealthy",
        "workers.draining",
        "total_inflight",
        "reroute_total",
        "queue.active",
        "queue.queued",
        "queue.rejected_total",
        "queue.shed_total",
        "queue.timeout_total",
        "worker_detail",
    ]
    .into_iter()
    .filter(|key| !json_key_exists(body, key))
    .collect()
}

fn has_fabric_readiness_signal(body: &Value) -> bool {
    body.get("ready").and_then(Value::as_bool).is_some()
        || body
            .pointer("/workers/eligible")
            .and_then(Value::as_u64)
            .is_some()
}

fn endpoint_check_detail(endpoint: Option<&EndpointReport>, success_detail: &str) -> String {
    match endpoint.and_then(|e| e.status_code) {
        Some(200) => success_detail.to_string(),
        Some(code) => format!("endpoint returned HTTP {code}"),
        None => "endpoint did not return an HTTP response".to_string(),
    }
}

fn json_key_exists(body: &Value, key: &str) -> bool {
    if body.get(key).is_some() {
        return true;
    }

    let mut cursor = body;
    for segment in key.split('.') {
        let Some(next) = cursor.get(segment) else {
            return false;
        };
        cursor = next;
    }
    true
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

fn build_migration_embedded_readiness_report(
    base_url: String,
    diagnostics: EndpointReport,
) -> MigrationEmbeddedReadinessReport {
    let mut blockers = Vec::new();
    let mut warnings = Vec::new();
    if !diagnostics.ok {
        blockers.push(endpoint_check_detail(
            Some(&diagnostics),
            "GET /v1/admin/diagnostics returned HTTP 200",
        ));
    }

    let body = diagnostics.body.as_ref();
    let mut runtimes = Vec::new();
    let mut totals = MigrationReadinessTotals::default();
    if let Some(runtime_map) = body
        .and_then(|body| body.pointer("/runtime_diagnostics/runtimes"))
        .and_then(Value::as_object)
    {
        let mut sorted = runtime_map.iter().collect::<Vec<_>>();
        sorted.sort_by_key(|(key, _)| *key);
        for (runtime, value) in sorted {
            let summary = migration_runtime_summary(runtime, value);
            totals.workers += summary.workers;
            totals.adapter_workers += summary.adapter_workers;
            totals.embedded_workers += summary.embedded_workers;
            totals.unknown_mode_workers += summary.unknown_mode_workers;
            totals.eligible_workers += summary.eligible;
            runtimes.push(summary);
        }
    }

    if diagnostics.ok && runtimes.is_empty() {
        blockers.push("no runtime workers registered".to_string());
    }
    if totals.adapter_workers == 0 {
        blockers.push("no runtime-node adapter workers registered".to_string());
    }
    if totals.embedded_workers > 0 {
        blockers.push(format!(
            "{} embedded compatibility worker(s) still registered",
            totals.embedded_workers
        ));
    }
    if totals.unknown_mode_workers > 0 {
        blockers.push(format!(
            "{} worker(s) have no runtime_mode; refresh adapters before denying embedded paths",
            totals.unknown_mode_workers
        ));
    }
    if totals.eligible_workers == 0 {
        blockers.push("no healthy non-draining workers are eligible for routing".to_string());
    }

    if let Some(issues) = body
        .and_then(|body| body.pointer("/runtime_diagnostics/issues"))
        .and_then(Value::as_array)
    {
        if issues.iter().any(|issue| {
            issue
                .get("code")
                .and_then(Value::as_str)
                .is_some_and(|code| code == "no_workers_registered")
        }) {
            blockers.push("gateway diagnostics reports no registered workers".to_string());
        }
        for issue in issues {
            if let Some(code) = issue.get("code").and_then(Value::as_str)
                && !matches!(
                    code,
                    "embedded_compatibility_path" | "no_workers_registered"
                )
            {
                warnings.push(format!("diagnostics issue remains: {code}"));
            }
        }
    }

    let mut deduped_blockers = BTreeSet::new();
    blockers.retain(|blocker| deduped_blockers.insert(blocker.clone()));
    let mut deduped_warnings = BTreeSet::new();
    warnings.retain(|warning| deduped_warnings.insert(warning.clone()));

    let ready_to_deny = diagnostics.ok && blockers.is_empty();
    MigrationEmbeddedReadinessReport {
        command: COMMAND_MIGRATION_EMBEDDED_READINESS,
        base_url,
        status: if ready_to_deny {
            STATUS_OK
        } else if diagnostics.status_code.is_some() {
            STATUS_FAIL
        } else {
            STATUS_UNREACHABLE
        },
        ready_to_deny,
        recommended_policy: if ready_to_deny { "deny" } else { "warn" },
        diagnostics,
        totals,
        runtimes,
        blockers,
        warnings,
    }
}

fn migration_runtime_summary(runtime: &str, value: &Value) -> MigrationRuntimeSummary {
    let workers = value.get("workers").and_then(Value::as_u64).unwrap_or(0) as usize;
    let eligible = value.get("eligible").and_then(Value::as_u64).unwrap_or(0) as usize;
    let runtime_modes = value
        .get("runtime_modes")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let adapter_workers = count_runtime_mode(&runtime_modes, "adapter");
    let embedded_from_mode = count_runtime_mode(&runtime_modes, "embedded");
    let embedded_from_issues = embedded_workers_from_issues(value);
    let embedded_workers = embedded_from_mode.max(embedded_from_issues);
    // Workers promoted to embedded via issue heuristics were previously counted
    // as unknown-mode; reduce unknown_mode_workers to avoid double-counting.
    let extra_from_issues = embedded_workers.saturating_sub(embedded_from_mode);
    let known_mode_workers = runtime_modes
        .values()
        .filter_map(Value::as_u64)
        .map(|value| value as usize)
        .sum::<usize>();
    let unknown_mode_workers = workers
        .saturating_sub(known_mode_workers)
        .saturating_sub(extra_from_issues);
    let models = value
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect();

    MigrationRuntimeSummary {
        runtime: runtime.to_string(),
        workers,
        eligible,
        adapter_workers,
        embedded_workers,
        unknown_mode_workers,
        models,
    }
}

fn count_runtime_mode(map: &serde_json::Map<String, Value>, mode: &str) -> usize {
    map.get(mode).and_then(Value::as_u64).unwrap_or(0) as usize
}

fn embedded_workers_from_issues(value: &Value) -> usize {
    value
        .get("issues")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|issue| {
            issue
                .get("code")
                .and_then(Value::as_str)
                .is_some_and(|code| code == "embedded_compatibility_path")
        })
        .filter_map(|issue| issue.get("workers").and_then(Value::as_array))
        .map(|workers| workers.len())
        .sum()
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
    if !report.recommended_actions.is_empty() {
        eprintln!("\n  Recommended actions:");
        for action in &report.recommended_actions {
            let runtime = action
                .runtime
                .as_deref()
                .map(|runtime| format!(" runtime={runtime}"))
                .unwrap_or_default();
            let reason = action
                .reason
                .as_deref()
                .map(|reason| format!(" reason={reason}"))
                .unwrap_or_default();
            eprintln!("  - {}{}{}", action.action, runtime, reason);
            if let Some(hint) = &action.operator_hint {
                eprintln!("    {hint}");
            }
            for command in &action.suggested_commands {
                eprintln!("    $ {command}");
            }
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

fn print_worker_lifecycle_human(report: &WorkerLifecycleReport) {
    eprintln!("AX Serving Workers\n");
    eprintln!("  Base URL:   {}", report.base_url);
    eprintln!("  Operation:  {}", report.operation);
    eprintln!("  Status:     {}", report.status);
    if let Some(worker_id) = &report.worker_id {
        eprintln!("  Worker ID:  {worker_id}");
    }
    for step in &report.steps {
        let status = step
            .status_code
            .map(|code| code.to_string())
            .unwrap_or_else(|| "no response".to_string());
        eprintln!(
            "  [{}] {}: {} ({} ms)",
            if step.ok { "OK" } else { "FAIL" },
            step.name,
            status,
            step.latency_ms,
        );
        if let Some(error) = &step.error {
            eprintln!("         {error}");
        }
        if step.name == "wait-idle"
            && let Some(inflight) = worker_inflight(step.body.as_ref())
        {
            eprintln!("         inflight={inflight}");
        }
    }
    if let Some(error) = &report.error {
        eprintln!("  Error:      {error}");
    }
}

fn print_fabric_validate_human(report: &FabricValidateReport) {
    eprintln!("AX Serving Fabric Validate\n");
    eprintln!("  Base URL: {}", report.base_url);
    eprintln!("  Status:   {}", report.status);
    eprintln!("  Profile:  {}", report.profile);
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
    eprintln!("\n  Contract checks:");
    for check in &report.checks {
        eprintln!(
            "  [{}] {}: {}",
            if check.ok { "OK" } else { "FAIL" },
            check.name,
            check.detail,
        );
    }
}

fn print_migration_embedded_readiness_human(report: &MigrationEmbeddedReadinessReport) {
    eprintln!("AX Serving Embedded Migration Readiness\n");
    eprintln!("  Base URL:       {}", report.base_url);
    eprintln!("  Status:         {}", report.status);
    eprintln!("  Ready to deny:  {}", report.ready_to_deny);
    eprintln!(
        "  Policy target:  AXS_EMBEDDED_RUNTIME_POLICY={}",
        report.recommended_policy
    );
    eprintln!(
        "  Workers:        total={} adapter={} embedded={} unknown_mode={} eligible={}",
        report.totals.workers,
        report.totals.adapter_workers,
        report.totals.embedded_workers,
        report.totals.unknown_mode_workers,
        report.totals.eligible_workers,
    );
    for runtime in &report.runtimes {
        eprintln!(
            "  Runtime {}: workers={} adapter={} embedded={} unknown_mode={} eligible={} models={}",
            runtime.runtime,
            runtime.workers,
            runtime.adapter_workers,
            runtime.embedded_workers,
            runtime.unknown_mode_workers,
            runtime.eligible,
            runtime.models.len(),
        );
    }
    if !report.blockers.is_empty() {
        eprintln!("\n  Blockers:");
        for blocker in &report.blockers {
            eprintln!("  - {blocker}");
        }
    }
    if !report.warnings.is_empty() {
        eprintln!("\n  Warnings:");
        for warning in &report.warnings {
            eprintln!("  - {warning}");
        }
    }
}

fn print_support_bundle_human(report: &SupportBundleReport) {
    eprintln!("AX Serving Support Bundle\n");
    eprintln!("  Base URL: {}", report.base_url);
    eprintln!("  Status:   {}", report.status);
    eprintln!("  Redaction: {}", report.redaction);
    if let Some(output) = &report.output {
        eprintln!("  Output:   {output}");
    }
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

fn diagnostics_recommended_actions(endpoints: &[EndpointReport]) -> Vec<StatusRecommendedAction> {
    let Some(diagnostics) = endpoints.iter().find(|e| e.name == ENDPOINT_DIAGNOSTICS) else {
        return Vec::new();
    };
    let Some(body) = diagnostics.body.as_ref() else {
        return Vec::new();
    };
    body.pointer("/runtime_diagnostics/recommended_actions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(recommended_action_from_value)
        .collect()
}

fn recommended_action_from_value(value: &Value) -> Option<StatusRecommendedAction> {
    Some(StatusRecommendedAction {
        action: value.get("action")?.as_str()?.to_string(),
        runtime: optional_string_field(value, "runtime"),
        reason: optional_string_field(value, "reason"),
        operator_hint: optional_string_field(value, "operator_hint"),
        suggested_commands: value
            .get("suggested_commands")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect(),
    })
}

fn optional_string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn redact_sensitive_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if is_sensitive_key(key) {
                    *child = Value::String("<redacted>".to_string());
                } else {
                    redact_sensitive_value(child);
                }
            }
        }
        Value::Array(values) => {
            for child in values {
                redact_sensitive_value(child);
            }
        }
        _ => {}
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    normalized.contains("api_key")
        || normalized.contains("apikey")
        || normalized.contains("authorization")
        || normalized.contains("bearer")
        || normalized.contains("license_key")
        || normalized.contains("password")
        || normalized.contains("secret")
        || normalized.contains("token")
}

fn current_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn worker_path(worker_id: &str, suffix: &str) -> String {
    format!("/v1/workers/{worker_id}{suffix}")
}

fn worker_inflight(body: Option<&Value>) -> Option<usize> {
    body.and_then(|body| body.get("inflight"))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
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

    #[test]
    fn fabric_validate_accepts_single_runtime_contract_payloads() {
        let report = build_fabric_validate_report(
            "http://127.0.0.1:18080".into(),
            vec![
                ok_endpoint(
                    "health",
                    "/health",
                    serde_json::json!({
                        "status": "degraded",
                        "ready": true,
                        "model_available": false,
                    }),
                ),
                ok_endpoint(
                    "models",
                    "/v1/models",
                    serde_json::json!({
                        "object": "list",
                        "data": [],
                    }),
                ),
                ok_endpoint(
                    "metrics",
                    "/v1/metrics",
                    serde_json::json!({
                        "scheduler": {
                            "queue_depth": 0,
                            "inflight_count": 0,
                            "cache_follower_waiting": 0,
                            "ttft_p50_us": 0,
                            "ttft_p95_us": 0,
                            "ttft_p99_us": 0,
                            "prefill_tokens_active": 0,
                            "decode_sequences_active": 0,
                            "split_scheduler_enabled": true
                        },
                        "loaded_models": [],
                        "thermal": "nominal"
                    }),
                ),
            ],
        );

        assert!(report.ready);
        assert_eq!(report.profile, FABRIC_PROFILE_SINGLE_RUNTIME);
        assert!(report.checks.iter().all(|check| check.ok));
    }

    #[test]
    fn fabric_validate_accepts_gateway_contract_payloads() {
        let report = build_fabric_validate_report(
            "http://127.0.0.1:18080".into(),
            vec![
                ok_endpoint(
                    "health",
                    "/health",
                    serde_json::json!({
                        "status": "ok",
                        "workers": {
                            "eligible": 1
                        }
                    }),
                ),
                ok_endpoint(
                    "models",
                    "/v1/models",
                    serde_json::json!({
                        "object": "list",
                        "data": [{"id": "model-a", "object": "model"}],
                    }),
                ),
                ok_endpoint(
                    "metrics",
                    "/v1/metrics",
                    serde_json::json!({
                        "mode": "direct",
                        "policy": "least_inflight",
                        "workers": {
                            "healthy": 1,
                            "unhealthy": 0,
                            "draining": 0
                        },
                        "total_inflight": 0,
                        "reroute_total": 0,
                        "queue": {
                            "active": 0,
                            "queued": 0,
                            "rejected_total": 0,
                            "shed_total": 0,
                            "timeout_total": 0
                        },
                        "worker_detail": []
                    }),
                ),
            ],
        );

        assert!(report.ready);
        assert_eq!(report.profile, FABRIC_PROFILE_GATEWAY);
        assert!(report.checks.iter().all(|check| check.ok));
    }

    #[test]
    fn fabric_validate_reports_missing_metrics_contract() {
        let report = build_fabric_validate_report(
            "http://127.0.0.1:18080".into(),
            vec![
                ok_endpoint(
                    "health",
                    "/health",
                    serde_json::json!({
                        "status": "ok",
                        "ready": true
                    }),
                ),
                ok_endpoint(
                    "models",
                    "/v1/models",
                    serde_json::json!({
                        "object": "list",
                        "data": []
                    }),
                ),
                ok_endpoint(
                    "metrics",
                    "/v1/metrics",
                    serde_json::json!({"scheduler": {}}),
                ),
            ],
        );

        assert!(!report.ready);
        assert_eq!(report.profile, FABRIC_PROFILE_UNKNOWN);
        assert!(report.checks.iter().any(|check| {
            !check.ok
                && check.name == "metrics_contract_profile"
                && check.detail.contains(FABRIC_PROFILE_UNKNOWN)
        }));
    }

    #[test]
    fn migration_embedded_readiness_accepts_adapter_only_fleet() {
        let report = build_migration_embedded_readiness_report(
            "http://127.0.0.1:18080".into(),
            ok_endpoint(
                ENDPOINT_DIAGNOSTICS,
                "/v1/admin/diagnostics",
                serde_json::json!({
                    "runtime_diagnostics": {
                        "runtimes": {
                            "ax_engine": {
                                "workers": 1,
                                "eligible": 1,
                                "runtime_modes": {"adapter": 1},
                                "models": ["mac-model"],
                                "issues": []
                            },
                            "vllm": {
                                "workers": 2,
                                "eligible": 2,
                                "runtime_modes": {"adapter": 2},
                                "models": ["cuda-model"],
                                "issues": []
                            }
                        },
                        "issues": []
                    }
                }),
            ),
        );

        assert!(report.ready_to_deny);
        assert_eq!(report.recommended_policy, "deny");
        assert_eq!(report.totals.adapter_workers, 3);
        assert!(report.blockers.is_empty());
    }

    #[test]
    fn migration_embedded_readiness_blocks_embedded_and_unknown_mode_workers() {
        let report = build_migration_embedded_readiness_report(
            "http://127.0.0.1:18080".into(),
            ok_endpoint(
                ENDPOINT_DIAGNOSTICS,
                "/v1/admin/diagnostics",
                serde_json::json!({
                    "runtime_diagnostics": {
                        "runtimes": {
                            "ax_engine": {
                                "workers": 2,
                                "eligible": 2,
                                "runtime_modes": {"embedded": 1},
                                "models": ["mac-model"],
                                "issues": [
                                    {
                                        "code": "embedded_compatibility_path",
                                        "workers": ["worker-embedded"]
                                    }
                                ]
                            }
                        },
                        "issues": [
                            {
                                "runtime": "ax_engine",
                                "code": "embedded_compatibility_path"
                            }
                        ]
                    }
                }),
            ),
        );

        assert!(!report.ready_to_deny);
        assert_eq!(report.recommended_policy, "warn");
        assert_eq!(report.totals.embedded_workers, 1);
        assert_eq!(report.totals.unknown_mode_workers, 1);
        assert!(
            report
                .blockers
                .iter()
                .any(|blocker| { blocker.contains("embedded compatibility worker") })
        );
        assert!(
            report
                .blockers
                .iter()
                .any(|blocker| { blocker.contains("no runtime_mode") })
        );
    }

    #[test]
    fn status_extracts_diagnostics_recommended_actions() {
        let endpoints = vec![EndpointReport {
            name: ENDPOINT_DIAGNOSTICS,
            url: "http://127.0.0.1:18080/v1/admin/diagnostics".into(),
            ok: true,
            status_code: Some(200),
            latency_ms: 1,
            body: Some(serde_json::json!({
                "runtime_diagnostics": {
                    "recommended_actions": [
                        {
                            "action": "restore_runtime_capacity",
                            "runtime": "vllm",
                            "reason": "runtime has no eligible workers",
                            "operator_hint": "Start or recover at least one healthy non-draining runtime node."
                        }
                    ]
                }
            })),
            error: None,
        }];

        let actions = diagnostics_recommended_actions(&endpoints);

        assert_eq!(
            actions,
            vec![StatusRecommendedAction {
                action: "restore_runtime_capacity".into(),
                runtime: Some("vllm".into()),
                reason: Some("runtime has no eligible workers".into()),
                operator_hint: Some(
                    "Start or recover at least one healthy non-draining runtime node.".into()
                ),
                suggested_commands: Vec::new(),
            }]
        );
    }

    #[test]
    fn worker_paths_target_public_worker_api() {
        assert_eq!(worker_path("worker-1", ""), "/v1/workers/worker-1");
        assert_eq!(
            worker_path("worker-1", "/drain-complete"),
            "/v1/workers/worker-1/drain-complete"
        );
    }

    #[test]
    fn worker_lifecycle_report_reflects_step_failure() {
        let report = worker_lifecycle_report(
            "http://127.0.0.1:18080".into(),
            "drain",
            Some("worker-1".into()),
            vec![EndpointReport {
                name: "drain",
                url: "http://127.0.0.1:18080/v1/workers/worker-1/drain".into(),
                ok: false,
                status_code: Some(404),
                latency_ms: 1,
                body: None,
                error: None,
            }],
            None,
        );

        assert!(!report.ok);
        assert_eq!(report.status, STATUS_FAIL);
    }

    #[test]
    fn worker_lifecycle_report_surfaces_workflow_error_in_error_field() {
        let report = worker_lifecycle_report(
            "http://127.0.0.1:18080".into(),
            "drain",
            Some("worker-1".into()),
            vec![
                EndpointReport {
                    name: "drain",
                    url: "http://127.0.0.1:18080/v1/workers/worker-1/drain".into(),
                    ok: true,
                    status_code: Some(200),
                    latency_ms: 1,
                    body: None,
                    error: None,
                },
                EndpointReport {
                    name: "drain-complete",
                    url: "http://127.0.0.1:18080/v1/workers/worker-1/drain-complete".into(),
                    ok: false,
                    status_code: Some(404),
                    latency_ms: 1,
                    body: None,
                    error: None,
                },
            ],
            Some("drain-complete failed (status 404)".into()),
        );

        assert!(!report.ok);
        assert_eq!(report.status, STATUS_FAIL);
        assert_eq!(
            report.error.as_deref(),
            Some("drain-complete failed (status 404)")
        );
    }

    #[test]
    fn worker_lifecycle_report_ok_requires_all_steps_pass_and_no_workflow_error() {
        let step = |name: &'static str, ok: bool| EndpointReport {
            name,
            url: format!("http://127.0.0.1:18080/v1/workers/worker-1/{name}"),
            ok,
            status_code: Some(if ok { 200 } else { 500 }),
            latency_ms: 1,
            body: None,
            error: None,
        };

        let all_pass = worker_lifecycle_report(
            "http://x".into(),
            "drain",
            None,
            vec![step("drain", true)],
            None,
        );
        assert!(all_pass.ok);
        assert_eq!(all_pass.status, STATUS_OK);
        assert!(all_pass.error.is_none());

        let step_fails = worker_lifecycle_report(
            "http://x".into(),
            "drain",
            None,
            vec![step("drain", false)],
            None,
        );
        assert!(!step_fails.ok);
        assert!(step_fails.error.is_none());

        let workflow_error = worker_lifecycle_report(
            "http://x".into(),
            "drain",
            None,
            vec![step("drain", true)],
            Some("workflow error".into()),
        );
        assert!(!workflow_error.ok);
        assert_eq!(workflow_error.error.as_deref(), Some("workflow error"));
    }

    #[test]
    fn worker_inflight_reads_worker_snapshot() {
        let body = serde_json::json!({
            "id": "worker-1",
            "inflight": 3
        });

        assert_eq!(worker_inflight(Some(&body)), Some(3));
        assert_eq!(worker_inflight(None), None);
    }

    #[test]
    fn support_bundle_endpoint_set_covers_operator_escalation_surfaces() {
        let endpoints = support_bundle_endpoints();

        assert!(endpoints.contains(&("health", "/health")));
        assert!(endpoints.contains(&("diagnostics", "/v1/admin/diagnostics")));
        assert!(endpoints.contains(&("fleet", "/v1/admin/fleet")));
        assert!(endpoints.contains(&("audit", "/v1/admin/audit?limit=50")));
    }

    #[test]
    fn support_bundle_redacts_sensitive_keys_recursively() {
        let mut body = serde_json::json!({
            "api_key": "top-secret",
            "nested": {
                "worker_token": "node-secret",
                "safe": "visible"
            },
            "events": [
                {"authorization": "Bearer abc"},
                {"detail": {"license_key": "license-secret"}}
            ]
        });

        redact_sensitive_value(&mut body);

        assert_eq!(body["api_key"], "<redacted>");
        assert_eq!(body["nested"]["worker_token"], "<redacted>");
        assert_eq!(body["nested"]["safe"], "visible");
        assert_eq!(body["events"][0]["authorization"], "<redacted>");
        assert_eq!(body["events"][1]["detail"]["license_key"], "<redacted>");
    }

    fn ok_endpoint(name: &'static str, path: &str, body: Value) -> EndpointReport {
        EndpointReport {
            name,
            url: format!("http://127.0.0.1:18080{path}"),
            ok: true,
            status_code: Some(200),
            latency_ms: 1,
            body: Some(body),
            error: None,
        }
    }
}
