//! Logging/tracing initialization utilities.

/// Initialize tracing subscriber with environment-based configuration.
///
/// # Arguments
/// * `verbose` - If true, sets default log level to DEBUG; otherwise WARN.
///
/// # Environment Variables
/// * `AXS_LOG` - Log filter directive (e.g., "debug", "ax_serving_api=trace")
/// * `AXS_LOG_FORMAT` - Output format: "text" (default) or "json"
pub fn init_logging(verbose: bool) {
    let default_level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::WARN
    };

    let log_filter =
        tracing_subscriber::EnvFilter::from_env("AXS_LOG").add_directive(default_level.into());

    let log_format = std::env::var("AXS_LOG_FORMAT").unwrap_or_else(|_| "text".into());

    if log_format == "json" {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(log_filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(log_filter).init();
    }
}
