//! `ax-serving doctor` — validate serving configuration and environment.

use std::path::Path;
use std::process::Command;

use anyhow::Result;
use serde::Serialize;

use crate::output::{emit_json_or_human, exit_if};
use crate::tune::HardwareProfile;

#[derive(Debug, Serialize)]
pub struct CheckResult {
    pub name: &'static str,
    pub status: CheckStatus,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remediation: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckStatus {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Serialize)]
struct DoctorSummary {
    pass: usize,
    warn: usize,
    fail: usize,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    command: &'static str,
    status: CheckStatus,
    summary: DoctorSummary,
    checks: Vec<CheckResult>,
}

pub fn run_doctor(json: bool) -> Result<()> {
    let results = run_checks();
    let report = build_report(results);

    emit_json_or_human(json, &report, print_human_report)?;
    exit_if(report.status == CheckStatus::Fail);
    Ok(())
}

fn run_checks() -> Vec<CheckResult> {
    vec![
        check_platform(),
        check_hardware(),
        check_llama_server(),
        check_runtime_boundary(),
        check_api_key(),
        check_config_file(),
        check_thermal(),
    ]
}

fn build_report(results: Vec<CheckResult>) -> DoctorReport {
    let pass_count = results
        .iter()
        .filter(|r| r.status == CheckStatus::Pass)
        .count();
    let warn_count = results
        .iter()
        .filter(|r| r.status == CheckStatus::Warn)
        .count();
    let fail_count = results
        .iter()
        .filter(|r| r.status == CheckStatus::Fail)
        .count();
    let status = if fail_count > 0 {
        CheckStatus::Fail
    } else if warn_count > 0 {
        CheckStatus::Warn
    } else {
        CheckStatus::Pass
    };

    DoctorReport {
        command: "ax-serving doctor",
        status,
        summary: DoctorSummary {
            pass: pass_count,
            warn: warn_count,
            fail: fail_count,
        },
        checks: results,
    }
}

fn print_human_report(report: &DoctorReport) {
    eprintln!("AX Serving Doctor\n");

    for r in &report.checks {
        let symbol = r.status.symbol();
        eprintln!("  [{symbol}] {}: {}", r.name, r.detail);
        if let Some(remediation) = r.remediation {
            eprintln!("         remediation: {remediation}");
        }
    }

    eprintln!();
    eprintln!(
        "{} passed, {} warnings, {} failures",
        report.summary.pass, report.summary.warn, report.summary.fail,
    );
}

fn check_platform() -> CheckResult {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    if arch == "aarch64" && os == "macos" {
        CheckResult {
            name: "Platform",
            status: CheckStatus::Pass,
            detail: format!("{os}/{arch} (Apple Silicon)"),
            remediation: None,
        }
    } else {
        CheckResult {
            name: "Platform",
            status: CheckStatus::Fail,
            detail: format!("{os}/{arch} — ax-serving requires aarch64-apple-darwin"),
            remediation: Some("Run AX Serving on Apple Silicon macOS."),
        }
    }
}

fn check_hardware() -> CheckResult {
    match HardwareProfile::detect() {
        Ok(profile) => CheckResult {
            name: "Hardware",
            status: CheckStatus::Pass,
            detail: format!(
                "{}, {} GB RAM, SKU class: {}",
                profile.chip_model,
                profile.total_memory_gb,
                profile.sku_class().as_str(),
            ),
            remediation: None,
        },
        Err(e) => CheckResult {
            name: "Hardware",
            status: CheckStatus::Warn,
            detail: format!("could not detect hardware: {e}"),
            remediation: Some("Verify system_profiler and sysctl are available on this host."),
        },
    }
}

fn check_llama_server() -> CheckResult {
    let worker_runtime = runtime_node_runtime_from_env();
    let embedded_runtime_policy = std::env::var("AXS_EMBEDDED_RUNTIME_POLICY").ok();
    if !llama_server_required(
        worker_runtime.as_deref(),
        embedded_runtime_policy.as_deref(),
    ) {
        return CheckResult {
            name: "llama-server",
            status: CheckStatus::Pass,
            detail: "not required for runtime-node deployment".into(),
            remediation: None,
        };
    }

    // Check AXS_LLAMA_CPP_BIN first, then PATH
    let bin = std::env::var("AXS_LLAMA_CPP_BIN").unwrap_or_else(|_| "llama-server".into());

    let result = Command::new("which").arg(&bin).output();

    match result {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            CheckResult {
                name: "llama-server",
                status: CheckStatus::Pass,
                detail: format!("found at {path}"),
                remediation: None,
            }
        }
        _ => CheckResult {
            name: "llama-server",
            status: CheckStatus::Warn,
            detail: format!(
                "'{bin}' not found on PATH. Set AXS_LLAMA_CPP_BIN or install llama.cpp."
            ),
            remediation: Some(
                "Install llama.cpp or set AXS_LLAMA_CPP_BIN to the llama-server path.",
            ),
        },
    }
}

fn llama_server_required(
    worker_runtime: Option<&str>,
    embedded_runtime_policy: Option<&str>,
) -> bool {
    if embedded_runtime_policy
        .map(str::trim)
        .is_some_and(|v| v.eq_ignore_ascii_case("deny"))
    {
        return false;
    }

    let runtime = worker_runtime.map(str::trim).filter(|v| !v.is_empty());
    !matches!(runtime, Some(v) if v.eq_ignore_ascii_case("ax_engine") || v.eq_ignore_ascii_case("vllm"))
}

fn check_runtime_boundary() -> CheckResult {
    runtime_boundary_result(
        runtime_node_control_plane_from_env().as_deref(),
        runtime_node_runtime_from_env().as_deref(),
        std::env::var("AXS_ROUTING_CONFIG").ok().as_deref(),
        std::env::var("AXS_LLAMA_CPP_BIN").ok().as_deref(),
        std::env::var("AXS_EMBEDDED_RUNTIME_POLICY").ok().as_deref(),
    )
}

fn runtime_node_control_plane_from_env() -> Option<String> {
    first_non_empty_env(&["AXS_ORCHESTRATOR_ADDR", "AXS_CONTROL_PLANE_URL"])
}

fn runtime_node_runtime_from_env() -> Option<String> {
    first_non_empty_env(&[
        "AXS_WORKER_RUNTIME",
        "AXS_NODE_RUNTIME",
        "AXS_THOR_RUNTIME",
        "AXS_THOR_BACKEND",
    ])
}

fn first_non_empty_env(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn runtime_boundary_result(
    orchestrator_addr: Option<&str>,
    worker_runtime: Option<&str>,
    routing_config: Option<&str>,
    llama_cpp_bin: Option<&str>,
    embedded_runtime_policy: Option<&str>,
) -> CheckResult {
    let orchestrator_configured = orchestrator_addr.is_some_and(|v| !v.trim().is_empty());
    let runtime = worker_runtime
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_ascii_lowercase());
    let embedded_hints = routing_config.is_some_and(|v| !v.trim().is_empty())
        || llama_cpp_bin.is_some_and(|v| !v.trim().is_empty());
    let embedded_policy = embedded_runtime_policy
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_ascii_lowercase());

    match (
        orchestrator_configured,
        runtime.as_deref(),
        embedded_hints,
        embedded_policy.as_deref(),
    ) {
        (true, Some("ax_engine" | "vllm"), _, _) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Pass,
            detail: format!(
                "worker registration is configured for runtime-node mode ({})",
                runtime.unwrap()
            ),
            remediation: None,
        },
        (_, Some(other), _, _) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Warn,
            detail: format!("worker runtime '{other}' is not a PRD target runtime"),
            remediation: Some("Use AXS_WORKER_RUNTIME=ax_engine or AXS_WORKER_RUNTIME=vllm."),
        },
        (_, _, _, Some("deny")) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Pass,
            detail: "embedded runtime compatibility paths are disabled by policy".into(),
            remediation: None,
        },
        (_, _, _, Some("allow")) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Warn,
            detail: "embedded runtime compatibility paths are explicitly allowed".into(),
            remediation: Some(
                "Use AXS_EMBEDDED_RUNTIME_POLICY=deny in production gateway-only deployments.",
            ),
        },
        (true, None, _, Some("warn") | None) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Warn,
            detail: "orchestrated worker will default to ax_engine/mac compatibility metadata"
                .into(),
            remediation: Some(
                "Set AXS_WORKER_RUNTIME and prefer dedicated ax-engine or vLLM node adapters.",
            ),
        },
        (false, None, true, Some("warn") | None) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Warn,
            detail: "embedded local runtime configuration detected; this is a compatibility path"
                .into(),
            remediation: Some(
                "Move inference execution to ax-engine or vLLM runtime nodes and keep AX Serving as the gateway.",
            ),
        },
        (false, None, false, Some("warn") | None) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Warn,
            detail: "standalone embedded inference mode is available only as a compatibility path"
                .into(),
            remediation: Some(
                "Run ax-serving-api as the gateway and register ax-engine or vLLM runtime nodes.",
            ),
        },
        (_, _, _, Some(other)) => CheckResult {
            name: "Runtime boundary",
            status: CheckStatus::Warn,
            detail: format!("unknown AXS_EMBEDDED_RUNTIME_POLICY value '{other}'"),
            remediation: Some("Use AXS_EMBEDDED_RUNTIME_POLICY=allow, warn, or deny."),
        },
    }
}

fn check_api_key() -> CheckResult {
    let key = std::env::var("AXS_API_KEY").unwrap_or_default();
    let allow_no_auth = std::env::var("AXS_ALLOW_NO_AUTH")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);

    if !key.is_empty() {
        CheckResult {
            name: "Auth",
            status: CheckStatus::Pass,
            detail: "AXS_API_KEY is set".into(),
            remediation: None,
        }
    } else if allow_no_auth {
        CheckResult {
            name: "Auth",
            status: CheckStatus::Warn,
            detail: "AXS_API_KEY is empty but AXS_ALLOW_NO_AUTH=true (dev mode)".into(),
            remediation: Some("Set AXS_API_KEY before production use."),
        }
    } else {
        CheckResult {
            name: "Auth",
            status: CheckStatus::Fail,
            detail:
                "AXS_API_KEY is empty and AXS_ALLOW_NO_AUTH is not set. Server will refuse to start."
                    .into(),
            remediation: Some("Set AXS_API_KEY or set AXS_ALLOW_NO_AUTH=true for local development."),
        }
    }
}

fn check_config_file() -> CheckResult {
    // Check --config flag (we can't read CLI args here, so check env + default paths)
    let config_path = std::env::var("AXS_CONFIG").ok();
    let paths_to_check: Vec<&str> = if let Some(ref p) = config_path {
        vec![p.as_str()]
    } else {
        vec!["./serving.yaml", "./serving.toml", "./config/serving.yaml"]
    };

    for path in &paths_to_check {
        if Path::new(path).exists() {
            return CheckResult {
                name: "Config file",
                status: CheckStatus::Pass,
                detail: format!("found {path}"),
                remediation: None,
            };
        }
    }

    // Check XDG default
    if let Ok(home) = std::env::var("HOME") {
        let xdg = format!("{home}/.config/ax-serving/serving.yaml");
        if Path::new(&xdg).exists() {
            return CheckResult {
                name: "Config file",
                status: CheckStatus::Pass,
                detail: format!("found {xdg}"),
                remediation: None,
            };
        }
    }

    CheckResult {
        name: "Config file",
        status: CheckStatus::Warn,
        detail:
            "no serving.toml found; using built-in defaults. Run `ax-serving tune` to generate one."
                .into(),
        remediation: Some("Run `ax-serving tune` to generate a serving.toml tuned for this host."),
    }
}

fn check_thermal() -> CheckResult {
    // Read thermal state via pmset
    let output = Command::new("pmset").args(["-g", "therm"]).output();

    match output {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            if text.contains("CPU_Speed_Limit") {
                // Parse the speed limit value
                for line in text.lines() {
                    if let Some(val) = line.trim().strip_prefix("CPU_Speed_Limit") {
                        let val = val
                            .trim()
                            .trim_start_matches('=')
                            .trim()
                            .trim_end_matches(|c: char| !c.is_ascii_digit());
                        if let Ok(limit) = val.parse::<u32>() {
                            return if limit >= 100 {
                                CheckResult {
                                    name: "Thermal",
                                    status: CheckStatus::Pass,
                                    detail: format!("CPU speed limit: {limit}% (nominal)"),
                                    remediation: None,
                                }
                            } else if limit >= 70 {
                                CheckResult {
                                    name: "Thermal",
                                    status: CheckStatus::Warn,
                                    detail: format!(
                                        "CPU speed limit: {limit}% (throttled). Performance may be reduced."
                                    ),
                                    remediation: Some(
                                        "Reduce load or let the machine cool before benchmarking.",
                                    ),
                                }
                            } else {
                                CheckResult {
                                    name: "Thermal",
                                    status: CheckStatus::Fail,
                                    detail: format!(
                                        "CPU speed limit: {limit}% (severe throttling). Cool down before serving."
                                    ),
                                    remediation: Some(
                                        "Cool down the machine before starting production serving.",
                                    ),
                                }
                            };
                        }
                    }
                }
            }
            CheckResult {
                name: "Thermal",
                status: CheckStatus::Pass,
                detail: "no thermal throttling detected".into(),
                remediation: None,
            }
        }
        _ => CheckResult {
            name: "Thermal",
            status: CheckStatus::Warn,
            detail: "could not read thermal state via pmset".into(),
            remediation: Some(
                "Run `pmset -g therm` manually if thermal state matters for this run.",
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn check_platform_on_current_host() {
        let r = check_platform();
        // We're compiling for aarch64-apple-darwin, so this should pass.
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn check_status_symbols() {
        assert_eq!(CheckStatus::Pass.symbol(), "PASS");
        assert_eq!(CheckStatus::Warn.symbol(), "WARN");
        assert_eq!(CheckStatus::Fail.symbol(), "FAIL");
    }

    #[test]
    fn report_summary_reflects_worst_status() {
        let report = build_report(vec![
            CheckResult {
                name: "ok",
                status: CheckStatus::Pass,
                detail: "ready".into(),
                remediation: None,
            },
            CheckResult {
                name: "missing",
                status: CheckStatus::Fail,
                detail: "not ready".into(),
                remediation: Some("fix it"),
            },
        ]);

        assert_eq!(report.status, CheckStatus::Fail);
        assert_eq!(report.summary.pass, 1);
        assert_eq!(report.summary.warn, 0);
        assert_eq!(report.summary.fail, 1);
    }

    #[test]
    fn runtime_boundary_passes_for_explicit_target_runtime_node() {
        let result = runtime_boundary_result(
            Some("http://127.0.0.1:19090"),
            Some("vllm"),
            None,
            None,
            None,
        );

        assert_eq!(result.status, CheckStatus::Pass);
        assert!(result.detail.contains("vllm"));
    }

    #[test]
    fn runtime_node_env_helpers_accept_generic_runtime_agent_names() {
        let _lock = env_lock().lock().expect("env lock");
        let _clear_orchestrator = EnvGuard::unset("AXS_ORCHESTRATOR_ADDR");
        let _clear_worker_runtime = EnvGuard::unset("AXS_WORKER_RUNTIME");
        let _clear_thor_runtime = EnvGuard::unset("AXS_THOR_RUNTIME");
        let _clear_thor_backend = EnvGuard::unset("AXS_THOR_BACKEND");
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", " http://127.0.0.1:19090/ ");
        let _runtime = EnvGuard::set("AXS_NODE_RUNTIME", " ax_engine ");

        assert_eq!(
            runtime_node_control_plane_from_env().as_deref(),
            Some("http://127.0.0.1:19090/")
        );
        assert_eq!(
            runtime_node_runtime_from_env().as_deref(),
            Some("ax_engine")
        );
    }

    #[test]
    fn llama_server_not_required_for_runtime_node_modes() {
        assert!(!llama_server_required(Some("ax_engine"), None));
        assert!(!llama_server_required(Some(" VLLM "), None));
    }

    #[test]
    fn llama_server_not_required_when_embedded_policy_denies_paths() {
        assert!(!llama_server_required(None, Some("deny")));
        assert!(!llama_server_required(
            Some("custom_runtime"),
            Some(" DENY ")
        ));
    }

    #[test]
    fn llama_server_required_for_embedded_compatibility_paths() {
        assert!(llama_server_required(None, None));
        assert!(llama_server_required(Some("custom_runtime"), None));
        assert!(llama_server_required(None, Some("allow")));
    }

    #[test]
    fn runtime_boundary_warns_for_embedded_compatibility_mode() {
        let result = runtime_boundary_result(None, None, Some("./backends.yaml"), None, None);

        assert_eq!(result.status, CheckStatus::Warn);
        assert!(result.detail.contains("compatibility path"));
    }

    #[test]
    fn runtime_boundary_warns_for_unknown_runtime() {
        let result = runtime_boundary_result(
            Some("http://127.0.0.1:19090"),
            Some("custom_runtime"),
            None,
            None,
            None,
        );

        assert_eq!(result.status, CheckStatus::Warn);
        assert!(result.detail.contains("not a PRD target runtime"));
    }

    #[test]
    fn runtime_boundary_passes_when_embedded_runtime_policy_denies_compatibility_paths() {
        let result =
            runtime_boundary_result(None, None, Some("./backends.yaml"), None, Some("deny"));

        assert_eq!(result.status, CheckStatus::Pass);
        assert!(result.detail.contains("disabled by policy"));
    }

    #[test]
    fn runtime_boundary_warns_when_embedded_runtime_policy_explicitly_allows_paths() {
        let result = runtime_boundary_result(None, None, None, None, Some("allow"));

        assert_eq!(result.status, CheckStatus::Warn);
        assert!(result.detail.contains("explicitly allowed"));
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }
}
