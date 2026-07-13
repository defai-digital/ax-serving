//! Logging and optional OTLP trace initialization.

pub fn init_logging(verbose: bool) -> anyhow::Result<ax_serving_observability::TelemetryGuard> {
    ax_serving_observability::init(
        "ax-serving",
        if verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::WARN
        },
    )
}
