//! Shared, runtime-SDK-free helpers for AX Serving execution-domain adapters.

pub mod openai_runtime;
pub mod proxy;

#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }
}
