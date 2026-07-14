//! Environment-variable locking helpers for integration tests.
//!
//! Process-global env mutation is serialized by `ENV_LOCK` so concurrent
//! tests cannot race on `set_var` / `remove_var`.

use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

pub struct TestConfigHome {
    _guard: MutexGuard<'static, ()>,
    _dir: tempfile::TempDir,
    previous_xdg: Option<std::ffi::OsString>,
    previous_home: Option<std::ffi::OsString>,
}

impl TestConfigHome {
    pub fn new() -> Self {
        let guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let previous_xdg = std::env::var_os("XDG_CONFIG_HOME");
        let previous_home = std::env::var_os("HOME");
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", dir.path());
            std::env::set_var("HOME", dir.path());
        }
        Self {
            _guard: guard,
            _dir: dir,
            previous_xdg,
            previous_home,
        }
    }
}

impl Drop for TestConfigHome {
    fn drop(&mut self) {
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            match &self.previous_xdg {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            match &self.previous_home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }
    }
}

pub struct EnvVarsGuard {
    _guard: MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvVarsGuard {
    pub fn new() -> Self {
        let guard = ENV_LOCK.lock().unwrap();
        Self {
            _guard: guard,
            previous: Vec::new(),
        }
    }

    pub fn set(&mut self, key: &'static str, value: &str) {
        self.previous.push((key, std::env::var_os(key)));
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    pub fn remove(&mut self, key: &'static str) {
        self.previous.push((key, std::env::var_os(key)));
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            std::env::remove_var(key);
        }
    }
}

impl Drop for EnvVarsGuard {
    fn drop(&mut self) {
        // SAFETY: test-only env mutation is serialized by ENV_LOCK.
        unsafe {
            for (key, previous) in self.previous.iter().rev() {
                match previous {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}
