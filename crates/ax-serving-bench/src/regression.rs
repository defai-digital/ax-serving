//! Regression-check subcommand: compares a benchmark results JSON against a
//! stored baseline and exits non-zero if any metric exceeds the allowed drift.
//!
//! # Baseline format
//!
//! Both `--results` and `--baseline` use the same JSON schema:
//!
//! ```json
//! {"short_p99": 500.0, "medium_p99": 2000.0, "long_p99": 8000.0, "overall_p99": 10000.0}
//! ```
//!
//! A `null` value in the baseline means "no baseline for this metric yet" and
//! the corresponding check is silently skipped.  This lets baselines be seeded
//! incrementally without failing CI on the first run.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Metrics {
    short_p99: Option<f64>,
    medium_p99: Option<f64>,
    long_p99: Option<f64>,
    overall_p99: Option<f64>,
}

/// Compare `results` against `baseline`, exiting non-zero if any metric
/// exceeds `baseline * (1 + tolerance_pct / 100)`.
///
/// Null baseline values → skip (no baseline established yet).
pub fn check(results: PathBuf, baseline: PathBuf, tolerance_pct: f64) -> Result<()> {
    let results_str = std::fs::read_to_string(&results)
        .with_context(|| format!("failed to read results file: {}", results.display()))?;
    let baseline_str = std::fs::read_to_string(&baseline)
        .with_context(|| format!("failed to read baseline file: {}", baseline.display()))?;

    let res: Metrics = serde_json::from_str(&results_str)
        .with_context(|| format!("failed to parse results file: {}", results.display()))?;
    let base: Metrics = serde_json::from_str(&baseline_str)
        .with_context(|| format!("failed to parse baseline file: {}", baseline.display()))?;

    let multiplier = 1.0 + tolerance_pct / 100.0;

    let checks: &[(&str, Option<f64>, Option<f64>)] = &[
        ("short_p99", res.short_p99, base.short_p99),
        ("medium_p99", res.medium_p99, base.medium_p99),
        ("long_p99", res.long_p99, base.long_p99),
        ("overall_p99", res.overall_p99, base.overall_p99),
    ];

    println!(
        "\nRegression check (tolerance: {tolerance_pct:.1}%)\n{}",
        "=".repeat(50)
    );
    println!(
        "{:<14}  {:>12}  {:>12}  {:>10}  Status",
        "Metric", "Result (ms)", "Baseline (ms)", "Threshold"
    );
    println!("{}", "-".repeat(62));

    let mut any_failed = false;

    for (name, result_val, baseline_val) in checks {
        match (result_val, baseline_val) {
            (_, None) => {
                println!(
                    "{:<14}  {:>12}  {:>12}  {:>10}  no baseline",
                    name,
                    result_val.map_or("N/A".to_string(), |v| format!("{v:.0}")),
                    "null",
                    "-",
                );
            }
            (None, Some(bv)) => {
                println!(
                    "{:<14}  {:>12}  {:>12}  {:>10}  no results",
                    name,
                    "N/A",
                    format!("{bv:.0}"),
                    format!("{:.0}", bv * multiplier),
                );
            }
            (Some(rv), Some(bv)) => {
                let threshold = bv * multiplier;
                let pass = *rv <= threshold;
                if !pass {
                    any_failed = true;
                }
                println!(
                    "{:<14}  {:>12}  {:>12}  {:>10}  {}",
                    name,
                    format!("{rv:.0}"),
                    format!("{bv:.0}"),
                    format!("{threshold:.0}"),
                    if pass { "PASS" } else { "FAIL ← regression" },
                );
            }
        }
    }

    println!();

    if any_failed {
        anyhow::bail!(
            "regression check failed: one or more metrics exceeded baseline + {tolerance_pct:.1}%"
        );
    }

    println!("All checks passed.");
    Ok(())
}
