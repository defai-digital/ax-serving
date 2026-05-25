//! Small output helpers shared by CLI subcommands.

use anyhow::Result;
use serde::Serialize;

pub(crate) fn emit_json<T: Serialize>(report: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

pub(crate) fn emit_json_or_human<T: Serialize>(
    json: bool,
    report: &T,
    print_human: impl FnOnce(&T),
) -> Result<()> {
    if json {
        emit_json(report)?;
    } else {
        print_human(report);
    }
    Ok(())
}

pub(crate) fn exit_if(failed: bool) {
    if failed {
        std::process::exit(1);
    }
}
