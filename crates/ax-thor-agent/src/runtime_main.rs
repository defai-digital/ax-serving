use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    ax_thor_agent::run_from_env().await
}
