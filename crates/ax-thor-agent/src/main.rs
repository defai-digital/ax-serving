use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "ax-thor-agent",
    version,
    about = "Legacy alias for the AX Serving runtime agent"
)]
struct Args {}

#[tokio::main]
async fn main() -> Result<()> {
    let _args = Args::parse();
    ax_thor_agent::run_from_env().await
}
