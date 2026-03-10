//! Soft license reminder system.
//!
//! Trust-based model: no enforcement, no feature gating.
//! A reminder is emitted (once) when multi-machine usage is detected.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::info;

use crate::config::LicenseConfig;

/// Shared license state — cheap to clone (Arc inside).
#[derive(Debug)]
pub struct LicenseState {
    key: Mutex<Option<String>>,
    remote_worker_seen: AtomicBool,
    buy_link: String,
    config_dir: String,
    key_file: String,
    pub dashboard_poll_ms: u64,
}

impl LicenseState {
    /// Create a new `LicenseState` from config, seeding the key from
    /// `AXS_LICENSE_KEY` env var first, then `~/.config/<config_dir>/<key_file>`.
    pub fn new(config: &LicenseConfig) -> Arc<Self> {
        let buy_link = config.buy_link.clone();
        let config_dir = config.config_dir.clone();
        let key_file = config.key_file.clone();
        let dashboard_poll_ms = config.dashboard_poll_ms;
        let key = Self::read_key_from_env()
            .or_else(|| Self::read_key_from_file_path(&config_dir, &key_file));
        Arc::new(Self {
            key: Mutex::new(key),
            remote_worker_seen: AtomicBool::new(false),
            buy_link,
            config_dir,
            key_file,
            dashboard_poll_ms,
        })
    }

    /// `"oss"` or `"business"`.
    pub fn edition(&self) -> &'static str {
        if self.has_key() { "business" } else { "oss" }
    }

    /// Whether a license key is currently loaded.
    pub fn has_key(&self) -> bool {
        self.key.lock().unwrap().is_some()
    }

    /// Persist a new license key to `~/.config/<config_dir>/<key_file>`
    /// and update the in-memory state.
    pub fn set_key(&self, key: String) -> anyhow::Result<()> {
        if let Some(path) = self.license_file_path() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &key)?;
        }
        *self.key.lock().unwrap() = Some(key);
        Ok(())
    }

    /// Mark that at least one non-loopback worker has registered.
    ///
    /// Idempotent (compare-exchange).  Emits a one-time INFO log if this is
    /// the first remote worker seen and no license key is present.
    pub fn mark_remote_worker_seen(&self) {
        let was_seen = self
            .remote_worker_seen
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err();
        if was_seen {
            return; // already logged
        }
        if !self.has_key() {
            let buy_link = &self.buy_link;
            info!(
                "\n[ax-serving] A remote worker registered (multi-machine deployment detected).\n\
                 \x20            This is a Business Edition feature.\n\
                 \x20            If your annual revenue exceeds USD 2M, purchase a license at:\n\
                 \x20            {buy_link}"
            );
        }
    }

    /// `true` when no key is present but a remote worker has been seen.
    pub fn needs_reminder(&self) -> bool {
        !self.has_key() && self.remote_worker_seen.load(Ordering::Relaxed)
    }

    /// Serialize for API responses.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "edition": self.edition(),
            "has_key": self.has_key(),
            "needs_reminder": self.needs_reminder(),
            "buy_link": self.buy_link,
            "dashboard_poll_ms": self.dashboard_poll_ms,
        })
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn read_key_from_env() -> Option<String> {
        std::env::var("AXS_LICENSE_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn read_key_from_file_path(config_dir: &str, key_file: &str) -> Option<String> {
        let path = Self::file_path(config_dir, key_file)?;
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn license_file_path(&self) -> Option<std::path::PathBuf> {
        Self::file_path(&self.config_dir, &self.key_file)
    }

    fn file_path(config_dir: &str, key_file: &str) -> Option<std::path::PathBuf> {
        // Use $XDG_CONFIG_HOME or fall back to ~/.config on all platforms.
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config"))
            })?;
        Some(base.join(config_dir).join(key_file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state() -> LicenseState {
        LicenseState {
            key: Mutex::new(None),
            remote_worker_seen: AtomicBool::new(false),
            buy_link: "https://license.automatosx.com".into(),
            config_dir: "ax-serving".into(),
            key_file: "license.key".into(),
            dashboard_poll_ms: 2000,
        }
    }

    #[test]
    fn edition_oss_by_default() {
        let ls = default_state();
        assert_eq!(ls.edition(), "oss");
        assert!(!ls.has_key());
    }

    #[test]
    fn edition_business_when_key_set() {
        let ls = LicenseState {
            key: Mutex::new(Some("test-key".into())),
            ..default_state()
        };
        assert_eq!(ls.edition(), "business");
        assert!(ls.has_key());
    }

    #[test]
    fn needs_reminder_only_when_no_key_and_remote_seen() {
        let ls = default_state();
        assert!(!ls.needs_reminder());
        ls.remote_worker_seen.store(true, Ordering::Relaxed);
        assert!(ls.needs_reminder());
        // With key: no reminder even if remote seen.
        *ls.key.lock().unwrap() = Some("k".into());
        assert!(!ls.needs_reminder());
    }

    #[test]
    fn mark_remote_worker_seen_is_idempotent() {
        let ls = Arc::new(default_state());
        ls.mark_remote_worker_seen();
        ls.mark_remote_worker_seen(); // should not panic
        assert!(ls.remote_worker_seen.load(Ordering::Relaxed));
    }

    #[test]
    fn to_json_fields_present() {
        let ls = default_state();
        let j = ls.to_json();
        assert_eq!(j["edition"], "oss");
        assert_eq!(j["has_key"], false);
        assert_eq!(j["needs_reminder"], false);
        assert!(j["buy_link"].as_str().unwrap().starts_with("https://"));
        assert!(j["dashboard_poll_ms"].as_u64().is_some());
    }

    #[test]
    fn new_uses_config_values() {
        let config = LicenseConfig {
            buy_link: "https://example.com/buy".into(),
            config_dir: "my-app".into(),
            key_file: "my.key".into(),
            dashboard_poll_ms: 5000,
        };
        let ls = LicenseState::new(&config);
        assert_eq!(ls.buy_link, "https://example.com/buy");
        assert_eq!(ls.dashboard_poll_ms, 5000);
        let j = ls.to_json();
        assert_eq!(j["buy_link"], "https://example.com/buy");
        assert_eq!(j["dashboard_poll_ms"], 5000u64);
    }
}
