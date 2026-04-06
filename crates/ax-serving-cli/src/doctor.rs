//! `ax-serving doctor` — validate serving configuration and environment.

use std::path::Path;
use std::process::Command;

use anyhow::Result;

use crate::tune::HardwareProfile;

#[derive(Debug)]
pub struct CheckResult {
    pub name: &'static str,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

pub fn run_doctor() -> Result<()> {
    let results = vec![
        check_platform(),
        check_hardware(),
        check_llama_server(),
        check_api_key(),
        check_config_file(),
        check_thermal(),
    ];

    eprintln!("AX Serving Doctor\n");

    let mut has_fail = false;
    for r in &results {
        let symbol = r.status.symbol();
        eprintln!("  [{symbol}] {}: {}", r.name, r.detail);
        if r.status == CheckStatus::Fail {
            has_fail = true;
        }
    }

    eprintln!();
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
    eprintln!("{pass_count} passed, {warn_count} warnings, {fail_count} failures");

    if has_fail {
        std::process::exit(1);
    }

    Ok(())
}

fn check_platform() -> CheckResult {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    if arch == "aarch64" && os == "macos" {
        CheckResult {
            name: "Platform",
            status: CheckStatus::Pass,
            detail: format!("{os}/{arch} (Apple Silicon)"),
        }
    } else {
        CheckResult {
            name: "Platform",
            status: CheckStatus::Fail,
            detail: format!("{os}/{arch} — ax-serving requires aarch64-apple-darwin"),
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
        },
        Err(e) => CheckResult {
            name: "Hardware",
            status: CheckStatus::Warn,
            detail: format!("could not detect hardware: {e}"),
        },
    }
}

fn check_llama_server() -> CheckResult {
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
            }
        }
        _ => CheckResult {
            name: "llama-server",
            status: CheckStatus::Warn,
            detail: format!(
                "'{bin}' not found on PATH. Set AXS_LLAMA_CPP_BIN or install llama.cpp."
            ),
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
        }
    } else if allow_no_auth {
        CheckResult {
            name: "Auth",
            status: CheckStatus::Warn,
            detail: "AXS_API_KEY is empty but AXS_ALLOW_NO_AUTH=true (dev mode)".into(),
        }
    } else {
        CheckResult {
            name: "Auth",
            status: CheckStatus::Fail,
            detail:
                "AXS_API_KEY is empty and AXS_ALLOW_NO_AUTH is not set. Server will refuse to start."
                    .into(),
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
            };
        }
    }

    CheckResult {
        name: "Config file",
        status: CheckStatus::Warn,
        detail:
            "no serving.toml found; using built-in defaults. Run `ax-serving tune` to generate one."
                .into(),
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
                                }
                            } else if limit >= 70 {
                                CheckResult {
                                    name: "Thermal",
                                    status: CheckStatus::Warn,
                                    detail: format!(
                                        "CPU speed limit: {limit}% (throttled). Performance may be reduced."
                                    ),
                                }
                            } else {
                                CheckResult {
                                    name: "Thermal",
                                    status: CheckStatus::Fail,
                                    detail: format!(
                                        "CPU speed limit: {limit}% (severe throttling). Cool down before serving."
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
            }
        }
        _ => CheckResult {
            name: "Thermal",
            status: CheckStatus::Warn,
            detail: "could not read thermal state via pmset".into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
