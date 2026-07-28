use std::path::PathBuf;

use ax_dynamo_adapter::manifest::ValidatedManifest;
use ax_serving_protocol::ExecutionDomainKind;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "ax-dynamo-adapter", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Validate and identify an immutable Dynamo compatibility manifest.
    CheckManifest {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, value_enum)]
        domain_kind: DomainKindArg,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum DomainKindArg {
    NvidiaDynamoPc,
    NvidiaDynamoThor,
}

impl From<DomainKindArg> for ExecutionDomainKind {
    fn from(value: DomainKindArg) -> Self {
        match value {
            DomainKindArg::NvidiaDynamoPc => Self::NvidiaDynamoPc,
            DomainKindArg::NvidiaDynamoThor => Self::NvidiaDynamoThor,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        None => ax_dynamo_adapter::run_from_env().await,
        Some(Command::CheckManifest {
            manifest,
            domain_kind,
        }) => {
            let validated = ValidatedManifest::load(&manifest, domain_kind.into())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "valid": true,
                    "digest": validated.digest.as_str(),
                    "domain_kind": validated.manifest.domain_kind.as_str(),
                    "dynamo_tag": validated.manifest.dynamo.tag,
                    "dynamo_commit": validated.manifest.dynamo.commit,
                    "backend": validated.manifest.backend,
                    "platform": validated.manifest.platform,
                }))?
            );
            Ok(())
        }
    }
}
