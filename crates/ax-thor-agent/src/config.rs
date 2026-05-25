use std::net::SocketAddr;

use anyhow::{Context, Result};

const DEFAULT_RUNTIME_URL: &str = "http://127.0.0.1:8000";
const DEFAULT_THOR_LISTEN_ADDR: &str = "0.0.0.0:18081";
const DEFAULT_MAX_INFLIGHT: usize = 8;
const DEFAULT_NODE_CLASS: &str = "thor";
const DEFAULT_RUNTIME: &str = "vllm";

fn load_first_optional_string_env(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}

fn parse_first_env<T>(keys: &[&str]) -> Option<T>
where
    T: std::str::FromStr,
{
    load_first_optional_string_env(keys).and_then(|v| v.parse::<T>().ok())
}

fn parse_first_bool_env(keys: &[&str]) -> Option<bool> {
    load_first_optional_string_env(keys).and_then(|v| match v.to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    })
}

fn load_optional_string_env(key: &str) -> Option<String> {
    load_first_optional_string_env(&[key])
}

fn load_control_plane_url() -> Result<String> {
    std::env::var("AXS_CONTROL_PLANE_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .context("AXS_CONTROL_PLANE_URL is required")
}

#[derive(Debug, Clone)]
pub struct ThorConfig {
    pub control_plane_url: String,
    pub worker_token: Option<String>,
    pub runtime_url: String,
    pub runtime: String,
    pub listen_addr: SocketAddr,
    pub advertised_addr: SocketAddr,
    pub max_inflight: usize,
    pub worker_pool: Option<String>,
    pub node_class: String,
    pub hardware_class: String,
    pub friendly_name: Option<String>,
    pub chip_model: Option<String>,
    /// env: `AXS_THOR_SHUTDOWN_TIMEOUT_SECS` (default 30)
    pub shutdown_timeout_secs: Option<u64>,
    /// env: `AXS_THOR_MAX_CONTEXT` — max context window advertised to control
    /// plane. If unset, the agent tries to derive it from the runtime.
    pub max_context: Option<u32>,
    /// env: `AXS_THOR_EMBEDDING` — override embedding capability (true/false).
    /// If unset, defaults to false (most LLM models are not embedding models).
    pub embedding: Option<bool>,
    /// env: `AXS_THOR_VISION` — override vision capability (true/false).
    /// If unset, defaults to false.
    pub vision: Option<bool>,
}

impl ThorConfig {
    pub fn from_env() -> Result<Self> {
        let control_plane_url = load_control_plane_url()?;
        let worker_token = load_optional_string_env("AXS_WORKER_TOKEN");
        let runtime_url = load_first_optional_string_env(&[
            "AXS_NODE_RUNTIME_URL",
            "AXS_THOR_RUNTIME_URL",
            "AXS_SGLANG_URL",
        ])
        .unwrap_or_else(|| DEFAULT_RUNTIME_URL.into());
        let runtime = load_first_optional_string_env(&[
            "AXS_NODE_RUNTIME",
            "AXS_THOR_RUNTIME",
            "AXS_THOR_BACKEND",
        ])
        .unwrap_or_else(|| DEFAULT_RUNTIME.into());
        let listen_addr: SocketAddr =
            load_first_optional_string_env(&["AXS_NODE_LISTEN_ADDR", "AXS_THOR_LISTEN_ADDR"])
                .unwrap_or_else(|| DEFAULT_THOR_LISTEN_ADDR.into())
                .parse()
                .context("invalid AXS_NODE_LISTEN_ADDR or AXS_THOR_LISTEN_ADDR")?;
        let advertised_addr: SocketAddr = load_first_optional_string_env(&[
            "AXS_NODE_ADVERTISED_ADDR",
            "AXS_THOR_ADVERTISED_ADDR",
        ])
        .unwrap_or_else(|| listen_addr.to_string())
        .parse()
        .context("invalid AXS_NODE_ADVERTISED_ADDR or AXS_THOR_ADVERTISED_ADDR")?;
        if advertised_addr.ip().is_unspecified() {
            tracing::warn!(
                %advertised_addr,
                "advertised address is a wildcard; the control plane will not be \
                 able to route traffic to this worker; set AXS_NODE_ADVERTISED_ADDR \
                 or AXS_THOR_ADVERTISED_ADDR to a routable IP"
            );
        }
        let max_inflight =
            parse_first_env::<usize>(&["AXS_NODE_MAX_INFLIGHT", "AXS_THOR_MAX_INFLIGHT"])
                .unwrap_or(DEFAULT_MAX_INFLIGHT)
                .max(1);
        let worker_pool =
            load_first_optional_string_env(&["AXS_NODE_WORKER_POOL", "AXS_THOR_WORKER_POOL"]);
        let node_class = load_first_optional_string_env(&["AXS_NODE_CLASS", "AXS_THOR_NODE_CLASS"])
            .unwrap_or_else(|| DEFAULT_NODE_CLASS.into());
        let hardware_class =
            load_first_optional_string_env(&["AXS_NODE_HARDWARE_CLASS", "AXS_THOR_HARDWARE_CLASS"])
                .unwrap_or_else(|| node_class.clone());
        let friendly_name =
            load_first_optional_string_env(&["AXS_NODE_FRIENDLY_NAME", "AXS_THOR_FRIENDLY_NAME"]);
        let chip_model =
            load_first_optional_string_env(&["AXS_NODE_CHIP_MODEL", "AXS_THOR_CHIP_MODEL"]);
        let shutdown_timeout_secs = parse_first_env::<u64>(&[
            "AXS_NODE_SHUTDOWN_TIMEOUT_SECS",
            "AXS_THOR_SHUTDOWN_TIMEOUT_SECS",
        ]);
        let max_context = parse_first_env::<u32>(&["AXS_NODE_MAX_CONTEXT", "AXS_THOR_MAX_CONTEXT"]);
        let embedding = parse_first_bool_env(&["AXS_NODE_EMBEDDING", "AXS_THOR_EMBEDDING"]);
        let vision = parse_first_bool_env(&["AXS_NODE_VISION", "AXS_THOR_VISION"]);

        Ok(Self {
            control_plane_url: control_plane_url.trim_end_matches('/').to_string(),
            worker_token,
            runtime_url: runtime_url.trim_end_matches('/').to_string(),
            runtime,
            listen_addr,
            advertised_addr,
            max_inflight,
            worker_pool,
            node_class,
            hardware_class,
            friendly_name,
            chip_model,
            shutdown_timeout_secs,
            max_context,
            embedding,
            vision,
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

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
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
        let _node_runtime = EnvGuard::remove("AXS_NODE_RUNTIME");
        let _node_runtime_url = EnvGuard::remove("AXS_NODE_RUNTIME_URL");
        let _node_listen = EnvGuard::remove("AXS_NODE_LISTEN_ADDR");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");
        let _node_hardware = EnvGuard::remove("AXS_NODE_HARDWARE_CLASS");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.listen_addr.to_string(), "0.0.0.0:18081");
        assert_eq!(config.advertised_addr.to_string(), "127.0.0.1:18081");
        assert_eq!(config.runtime, "vllm");
        assert_eq!(config.hardware_class, "thor");
    }

    #[test]
    fn from_env_accepts_generic_runtime_node_aliases() {
        let _lock = env_lock().lock().unwrap();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _runtime_url = EnvGuard::set("AXS_NODE_RUNTIME_URL", "http://127.0.0.1:9000");
        let _runtime = EnvGuard::set("AXS_NODE_RUNTIME", "ax_engine");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "127.0.0.1:18091");
        let _advertised = EnvGuard::set("AXS_NODE_ADVERTISED_ADDR", "127.0.0.1:18092");
        let _max_inflight = EnvGuard::set("AXS_NODE_MAX_INFLIGHT", "12");
        let _pool = EnvGuard::set("AXS_NODE_WORKER_POOL", "mac");
        let _node_class = EnvGuard::set("AXS_NODE_CLASS", "mac-studio");
        let _hardware_class = EnvGuard::set("AXS_NODE_HARDWARE_CLASS", "mac");
        let _embedding = EnvGuard::set("AXS_NODE_EMBEDDING", "true");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.runtime_url, "http://127.0.0.1:9000");
        assert_eq!(config.runtime, "ax_engine");
        assert_eq!(config.listen_addr.to_string(), "127.0.0.1:18091");
        assert_eq!(config.advertised_addr.to_string(), "127.0.0.1:18092");
        assert_eq!(config.max_inflight, 12);
        assert_eq!(config.worker_pool.as_deref(), Some("mac"));
        assert_eq!(config.node_class, "mac-studio");
        assert_eq!(config.hardware_class, "mac");
        assert_eq!(config.embedding, Some(true));
    }

    #[test]
    fn from_env_rejects_invalid_advertised_addr() {
        let _lock = env_lock().lock().unwrap();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_THOR_ADVERTISED_ADDR", "thor-node.local:18081");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");

        let err = ThorConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("AXS_THOR_ADVERTISED_ADDR"));
    }
}
