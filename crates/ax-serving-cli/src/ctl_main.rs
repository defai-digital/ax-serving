//! Portable operator CLI for an AX Serving gateway.

mod output;
mod support;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "ax-servingctl",
    version,
    about = "Portable operator client for AX Serving gateways",
    long_about = "Inspect, validate, and operate a remote AX Serving gateway without linking an inference runtime SDK."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Validate local gateway configuration and environment overrides.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Query gateway health, public models, metrics, and optional diagnostics.
    Status {
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        /// Public bearer token. Defaults to the first AXS_API_KEY token.
        #[arg(long)]
        api_key: Option<String>,
        /// Admin bearer token. Defaults to the first AXS_ADMIN_API_KEY token.
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        diagnostics: bool,
        #[arg(long)]
        json: bool,
    },
    /// Collect a recursively redacted diagnostics bundle.
    SupportBundle {
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Validate stable AX Fabric-facing gateway contracts.
    Fabric {
        #[command(subcommand)]
        command: FabricCommand,
    },
    /// Inspect compatibility-path migration readiness.
    Migration {
        #[command(subcommand)]
        command: MigrationCommand,
    },
    /// Inspect, drain, or remove registered workers.
    Workers {
        #[command(subcommand)]
        command: WorkersCommand,
    },
    /// Run a minimal OpenAI-compatible chat request.
    SmokeTest {
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long, default_value = "default")]
        model: String,
        #[arg(long, default_value = "Reply with ok.")]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_tokens: u32,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCommand {
    /// Validate the resolved serving config.
    Validate {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum FabricCommand {
    Validate {
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum MigrationCommand {
    EmbeddedReadiness {
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum WorkersCommand {
    List {
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Get {
        id: String,
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Drain {
        id: String,
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        complete_when_idle: bool,
        #[arg(long, default_value_t = 30)]
        idle_timeout_secs: u64,
        #[arg(long, default_value_t = 1000)]
        poll_interval_ms: u64,
        #[arg(long)]
        json: bool,
    },
    DrainComplete {
        id: String,
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Remove {
        id: String,
        #[arg(long, default_value = "http://127.0.0.1:18080")]
        url: String,
        #[arg(long)]
        admin_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Config { command } => match command {
            ConfigCommand::Validate { config, json } => support::run_config_validate(config, json),
        },
        Command::Status {
            url,
            api_key,
            admin_key,
            diagnostics,
            json,
        } => support::run_status(url, api_key, admin_key, diagnostics, json),
        Command::SupportBundle {
            url,
            api_key,
            admin_key,
            output,
            json,
        } => support::run_support_bundle(url, api_key, admin_key, output, json),
        Command::Fabric { command } => match command {
            FabricCommand::Validate {
                url,
                api_key,
                admin_key,
                json,
            } => support::run_fabric_validate(url, api_key, admin_key, json),
        },
        Command::Migration { command } => match command {
            MigrationCommand::EmbeddedReadiness {
                url,
                admin_key,
                json,
            } => support::run_migration_embedded_readiness(url, admin_key, json),
        },
        Command::Workers { command } => match command {
            WorkersCommand::List {
                url,
                admin_key,
                json,
            } => support::run_workers_list(url, admin_key, json),
            WorkersCommand::Get {
                id,
                url,
                admin_key,
                json,
            } => support::run_worker_get(url, id, admin_key, json),
            WorkersCommand::Drain {
                id,
                url,
                admin_key,
                complete_when_idle,
                idle_timeout_secs,
                poll_interval_ms,
                json,
            } => support::run_worker_drain(
                url,
                id,
                admin_key,
                complete_when_idle,
                idle_timeout_secs,
                poll_interval_ms,
                json,
            ),
            WorkersCommand::DrainComplete {
                id,
                url,
                admin_key,
                json,
            } => support::run_worker_drain_complete(url, id, admin_key, json),
            WorkersCommand::Remove {
                id,
                url,
                admin_key,
                json,
            } => support::run_worker_remove(url, id, admin_key, json),
        },
        Command::SmokeTest {
            url,
            model,
            prompt,
            max_tokens,
            api_key,
            json,
        } => support::run_smoke_test(url, model, prompt, max_tokens, api_key, json),
    }
}
