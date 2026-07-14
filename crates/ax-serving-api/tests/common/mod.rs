//! Shared fixtures for `ax-serving-api` orchestration integration tests.
//!
//! Wired from test binaries via:
//! ```ignore
//! #[path = "common/mod.rs"]
//! mod common;
//! use common::*;
//! ```

mod env;
mod mock_workers;
mod orchestrator;
mod registry;

pub use env::*;
pub use mock_workers::*;
pub use orchestrator::*;
pub use registry::*;

/// Unwrap a `spawn_mock_worker` / `TcpListener::bind` result, skipping the
/// test if loopback socket binding is unavailable (e.g. sandbox environments).
macro_rules! skip_if_no_socket {
    ($expr:expr) => {
        match $expr {
            Some(v) => v,
            None => {
                eprintln!("test skipped: loopback socket bind unavailable in this environment");
                return;
            }
        }
    };
}
pub(crate) use skip_if_no_socket;
