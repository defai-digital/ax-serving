use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "ax-runtime-agent",
    version,
    about = "Runtime-neutral AX Serving worker agent"
)]
struct Args {}

#[tokio::main]
async fn main() -> Result<()> {
    let _args = Args::parse();
    ax_thor_agent::run_from_env().await
}
