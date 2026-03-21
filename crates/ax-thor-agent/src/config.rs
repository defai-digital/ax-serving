use std::net::SocketAddr;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct ThorConfig {
    pub control_plane_url: String,
    pub worker_token: Option<String>,
    pub sglang_url: String,
    pub listen_addr: SocketAddr,
    pub advertised_addr: SocketAddr,
    pub max_inflight: usize,
    pub worker_pool: Option<String>,
    pub node_class: String,
    pub friendly_name: Option<String>,
    pub chip_model: Option<String>,
}

impl ThorConfig {
    pub fn from_env() -> Result<Self> {
        let control_plane_url = std::env::var("AXS_CONTROL_PLANE_URL")
            .context("AXS_CONTROL_PLANE_URL is required")?;
        let worker_token = std::env::var("AXS_WORKER_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let sglang_url =
            std::env::var("AXS_SGLANG_URL").unwrap_or_else(|_| "http://127.0.0.1:30000".into());
        let listen_addr: SocketAddr = std::env::var("AXS_THOR_LISTEN_ADDR")
            .unwrap_or_else(|_| "0.0.0.0:18081".into())
            .parse()
            .context("invalid AXS_THOR_LISTEN_ADDR")?;
        let advertised_addr: SocketAddr = std::env::var("AXS_THOR_ADVERTISED_ADDR")
            .unwrap_or_else(|_| listen_addr.to_string())
            .parse()
            .context("invalid AXS_THOR_ADVERTISED_ADDR")?;
        let max_inflight = std::env::var("AXS_THOR_MAX_INFLIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .max(1);
        let worker_pool = std::env::var("AXS_THOR_WORKER_POOL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let node_class = std::env::var("AXS_THOR_NODE_CLASS").unwrap_or_else(|_| "thor".into());
        let friendly_name = std::env::var("AXS_THOR_FRIENDLY_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let chip_model = std::env::var("AXS_THOR_CHIP_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        Ok(Self {
            control_plane_url: control_plane_url.trim_end_matches('/').to_string(),
            worker_token,
            sglang_url: sglang_url.trim_end_matches('/').to_string(),
            listen_addr,
            advertised_addr,
            max_inflight,
            worker_pool,
            node_class,
            friendly_name,
            chip_model,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ThorConfig;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn from_env_defaults_advertised_addr_to_listen_addr() {
        let _lock = env_lock().lock().unwrap();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_THOR_ADVERTISED_ADDR", "127.0.0.1:18081");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.listen_addr.to_string(), "0.0.0.0:18081");
        assert_eq!(config.advertised_addr.to_string(), "127.0.0.1:18081");
    }

    #[test]
    fn from_env_rejects_invalid_advertised_addr() {
        let _lock = env_lock().lock().unwrap();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_THOR_ADVERTISED_ADDR", "thor-node.local:18081");

        let err = ThorConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("AXS_THOR_ADVERTISED_ADDR"));
    }
}
