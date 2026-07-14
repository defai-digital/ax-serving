use std::net::SocketAddr;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{Digest, ProtocolCapability};

const DEFAULT_RUNTIME_URL: &str = "http://127.0.0.1:8000";
const DEFAULT_THOR_LISTEN_ADDR: &str = "0.0.0.0:18081";
const DEFAULT_MAX_INFLIGHT: usize = 8;
const DEFAULT_NODE_CLASS: &str = "thor";
const DEFAULT_RUNTIME: &str = "vllm";
const DEFAULT_TRUST_DOMAIN: &str = "local";
const DEFAULT_TLS_PROFILE: &str = "loopback_dev";

fn load_first_optional_string_env(keys: &[&str]) -> Option<String> {
    load_first_optional_string_env_with_key(keys).map(|(_, value)| value)
}

fn load_first_optional_string_env_with_key<'a>(keys: &'a [&str]) -> Option<(&'a str, String)> {
    keys.iter().find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(|value| (*key, value))
    })
}

fn parse_first_env<T>(keys: &[&str]) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    load_first_optional_string_env_with_key(keys)
        .map(|(key, value)| {
            value
                .parse::<T>()
                .map_err(|err| anyhow::anyhow!("invalid {key}: {value}: {err}"))
        })
        .transpose()
}

fn parse_first_bool_env(keys: &[&str]) -> Result<Option<bool>> {
    load_first_optional_string_env_with_key(keys)
        .map(|(key, value)| match value.to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => bail!("invalid {key}: {value}; expected true, false, 1, or 0"),
        })
        .transpose()
}

fn load_optional_string_env(key: &str) -> Option<String> {
    load_first_optional_string_env(&[key])
}

#[derive(Clone, Default)]
pub struct ModelIdentityConfig {
    pub revision: Option<String>,
    pub artifact_digest: Option<Digest>,
    pub tokenizer_digest: Option<Digest>,
    pub template_digest: Option<Digest>,
    pub quantization: Option<String>,
    pub max_output_tokens: Option<u64>,
    pub capabilities: Vec<ProtocolCapability>,
}

impl std::fmt::Debug for ModelIdentityConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelIdentityConfig")
            .field("revision", &self.revision)
            .field("artifact_digest", &self.artifact_digest)
            .field("tokenizer_digest", &self.tokenizer_digest)
            .field("template_digest", &self.template_digest)
            .field("quantization", &self.quantization)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

fn load_digest_env(key: &str) -> Result<Option<Digest>> {
    load_optional_string_env(key)
        .map(Digest::new)
        .transpose()
        .with_context(|| format!("invalid {key}"))
}

fn load_model_capabilities() -> Result<Vec<ProtocolCapability>> {
    let Some(raw) = load_optional_string_env("AXS_MODEL_CAPABILITIES") else {
        return Ok(Vec::new());
    };
    let mut capabilities = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            ProtocolCapability::new(value.to_string())
                .with_context(|| format!("invalid AXS_MODEL_CAPABILITIES entry {value:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    capabilities.sort();
    capabilities.dedup();
    Ok(capabilities)
}

fn load_control_plane_url() -> Result<String> {
    let raw = std::env::var("AXS_CONTROL_PLANE_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .context("AXS_CONTROL_PLANE_URL is required")?;
    normalize_http_base_url(&raw, "AXS_CONTROL_PLANE_URL")
}

fn normalize_http_base_url(raw: &str, field: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("{field} URL is empty");
    }
    let trimmed = trimmed.trim_end_matches('/');
    if trimmed.is_empty() {
        bail!("{field} URL is empty after trimming trailing slashes");
    }

    let mut rest = trimmed;
    let has_scheme = if let Some(scheme_end) = trimmed.find("://") {
        let scheme = &trimmed[..scheme_end];
        if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
            rest = &trimmed[scheme_end + 3..];
            true
        } else {
            bail!("{field} has unsupported URL scheme: {trimmed}");
        }
    } else {
        false
    };

    if rest.is_empty() {
        bail!("{field} URL is incomplete: {trimmed}");
    }
    if rest.contains('/') {
        bail!("{field} URL must not include a path: {trimmed}");
    }
    if rest.contains('?') || rest.contains('#') {
        bail!("{field} URL must not include query params or fragments: {trimmed}");
    }

    if has_scheme {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("http://{trimmed}"))
    }
}

fn default_advertised_url(listen_addr: SocketAddr) -> String {
    let addr = if listen_addr.ip().is_unspecified() {
        match listen_addr {
            SocketAddr::V4(addr) => SocketAddr::from(([127, 0, 0, 1], addr.port())),
            SocketAddr::V6(addr) => SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], addr.port())),
        }
    } else {
        listen_addr
    };
    format!("http://{addr}")
}

/// Parse `AXS_NODE_ADVERTISED_URL` or legacy `AXS_NODE_ADVERTISED_ADDR`.
fn parse_advertised_url(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("advertised URL is empty");
    }
    let candidate = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let url = reqwest::Url::parse(&candidate).context("invalid advertised URL")?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("advertised URL must use http or https");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("advertised URL must not contain credentials");
    }
    if !matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some() {
        bail!("advertised URL must not contain a path, query, or fragment");
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("advertised URL is missing a host"))?;
    if let Ok(ip) = host.parse::<std::net::IpAddr>()
        && ip.is_unspecified()
    {
        bail!(
            "advertised address {candidate} is a wildcard; set \
             AXS_NODE_ADVERTISED_URL or AXS_NODE_ADVERTISED_ADDR to a routable host"
        );
    }
    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("advertised URL is missing a port"))?;
    let host_for_url = if let Ok(ip) = host.parse::<std::net::Ipv6Addr>() {
        format!("[{ip}]")
    } else {
        host.to_string()
    };
    Ok(format!("{}://{}:{}", url.scheme(), host_for_url, port))
}

#[derive(Clone)]
pub struct ThorConfig {
    pub control_plane_url: String,
    pub worker_token: Option<String>,
    pub runtime_url: String,
    /// Credential used only for agent-to-runtime requests.
    pub runtime_api_key: Option<String>,
    /// Credential accepted only from the gateway on inference routes.
    pub dispatch_token: Option<String>,
    /// Transport security profile. Remote listeners require `trusted_mesh`.
    pub tls_profile: String,
    pub runtime: String,
    pub runtime_version: String,
    pub worker_id: String,
    pub trust_domain: String,
    pub listen_addr: SocketAddr,
    /// Canonical advertise base URL (`http(s)://host:port`), possibly DNS.
    pub advertised_url: String,
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
    /// If unset, the agent derives the capability from runtime model metadata.
    pub embedding: Option<bool>,
    /// env: `AXS_THOR_VISION` — override vision capability (true/false).
    /// If unset, the agent derives the capability from runtime model metadata.
    pub vision: Option<bool>,
    /// Operator-supplied semantic model identity. Runtime discovery remains
    /// authoritative for loaded model IDs, while these values provide the
    /// certification metadata most OpenAI-compatible `/v1/models` endpoints omit.
    pub model_identity: ModelIdentityConfig,
}

impl std::fmt::Debug for ThorConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ThorConfig")
            .field("control_plane_url", &self.control_plane_url)
            .field("worker_token_configured", &self.worker_token.is_some())
            .field("runtime_url", &self.runtime_url)
            .field(
                "runtime_api_key_configured",
                &self.runtime_api_key.is_some(),
            )
            .field("dispatch_token_configured", &self.dispatch_token.is_some())
            .field("tls_profile", &self.tls_profile)
            .field("runtime", &self.runtime)
            .field("runtime_version", &self.runtime_version)
            .field("worker_id", &self.worker_id)
            .field("trust_domain", &self.trust_domain)
            .field("listen_addr", &self.listen_addr)
            .field("advertised_url", &self.advertised_url)
            .field("max_inflight", &self.max_inflight)
            .field("worker_pool", &self.worker_pool)
            .field("node_class", &self.node_class)
            .field("hardware_class", &self.hardware_class)
            .field("friendly_name", &self.friendly_name)
            .field("chip_model", &self.chip_model)
            .field("shutdown_timeout_secs", &self.shutdown_timeout_secs)
            .field("max_context", &self.max_context)
            .field("embedding", &self.embedding)
            .field("vision", &self.vision)
            .field("model_identity", &self.model_identity)
            .finish()
    }
}

impl ThorConfig {
    pub fn from_env() -> Result<Self> {
        let control_plane_url = load_control_plane_url()?;
        let worker_token =
            load_first_optional_string_env(&["AXS_WORKER_TOKEN", "AXS_INTERNAL_API_TOKEN"]);
        let runtime_api_key = load_optional_string_env("AXS_RUNTIME_API_KEY");
        let dispatch_token = load_optional_string_env("AXS_DISPATCH_TOKEN");
        let tls_profile = load_optional_string_env("AXS_TLS_PROFILE")
            .unwrap_or_else(|| DEFAULT_TLS_PROFILE.into())
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_");
        if !matches!(tls_profile.as_str(), "loopback_dev" | "trusted_mesh") {
            bail!("invalid AXS_TLS_PROFILE: expected loopback_dev or trusted_mesh");
        }
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
        let runtime_version =
            load_optional_string_env("AXS_RUNTIME_VERSION").unwrap_or_else(|| "unknown".into());
        let listen_addr: SocketAddr =
            load_first_optional_string_env(&["AXS_NODE_LISTEN_ADDR", "AXS_THOR_LISTEN_ADDR"])
                .unwrap_or_else(|| DEFAULT_THOR_LISTEN_ADDR.into())
                .parse()
                .context("invalid AXS_NODE_LISTEN_ADDR or AXS_THOR_LISTEN_ADDR")?;
        let advertised_raw = load_first_optional_string_env(&[
            "AXS_NODE_ADVERTISED_URL",
            "AXS_NODE_ADVERTISED_ADDR",
            "AXS_THOR_ADVERTISED_ADDR",
        ]);
        let advertised_url = match advertised_raw {
            Some(raw) => parse_advertised_url(&raw).context(
                "invalid AXS_NODE_ADVERTISED_URL / AXS_NODE_ADVERTISED_ADDR / AXS_THOR_ADVERTISED_ADDR",
            )?,
            None => default_advertised_url(listen_addr),
        };
        let default_worker_id = advertised_url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .replace([':', '/', '[', ']'], "-");
        let default_worker_id = format!("node-{default_worker_id}");
        let worker_id = load_optional_string_env("AXS_NODE_ID").unwrap_or(default_worker_id);
        ax_serving_protocol::WorkerId::new(worker_id.clone()).context("invalid AXS_NODE_ID")?;
        let trust_domain = load_optional_string_env("AXS_TRUST_DOMAIN")
            .unwrap_or_else(|| DEFAULT_TRUST_DOMAIN.into());
        ax_serving_protocol::TrustDomainId::new(trust_domain.clone())
            .context("invalid AXS_TRUST_DOMAIN")?;
        let max_inflight =
            parse_first_env::<usize>(&["AXS_NODE_MAX_INFLIGHT", "AXS_THOR_MAX_INFLIGHT"])?
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
        ])?;
        let max_context =
            parse_first_env::<u32>(&["AXS_NODE_MAX_CONTEXT", "AXS_THOR_MAX_CONTEXT"])?;
        let embedding = parse_first_bool_env(&["AXS_NODE_EMBEDDING", "AXS_THOR_EMBEDDING"])?;
        let vision = parse_first_bool_env(&["AXS_NODE_VISION", "AXS_THOR_VISION"])?;
        let model_identity = ModelIdentityConfig {
            revision: load_optional_string_env("AXS_MODEL_REVISION"),
            artifact_digest: load_digest_env("AXS_MODEL_ARTIFACT_DIGEST")?,
            tokenizer_digest: load_digest_env("AXS_MODEL_TOKENIZER_DIGEST")?,
            template_digest: load_digest_env("AXS_MODEL_TEMPLATE_DIGEST")?,
            quantization: load_optional_string_env("AXS_MODEL_QUANTIZATION"),
            max_output_tokens: parse_first_env::<u64>(&["AXS_MODEL_MAX_OUTPUT_TOKENS"])?
                .filter(|value| *value > 0),
            capabilities: load_model_capabilities()?,
        };
        if !listen_addr.ip().is_loopback() && dispatch_token.is_none() {
            bail!(
                "AXS_DISPATCH_TOKEN is required when the agent listens on non-loopback address {listen_addr}"
            );
        }
        if !listen_addr.ip().is_loopback() && tls_profile != "trusted_mesh" {
            bail!(
                "AXS_TLS_PROFILE=trusted_mesh is required when the agent listens on non-loopback address {listen_addr}"
            );
        }

        Ok(Self {
            control_plane_url,
            worker_token,
            runtime_url: normalize_http_base_url(&runtime_url, "runtime URL")?,
            runtime_api_key,
            dispatch_token,
            tls_profile,
            runtime,
            runtime_version,
            worker_id,
            trust_domain,
            listen_addr,
            advertised_url,
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
            model_identity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ThorConfig, default_advertised_url, normalize_http_base_url, parse_advertised_url};
    use std::ffi::OsString;

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
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_THOR_ADVERTISED_ADDR", "127.0.0.1:18081");
        let _dispatch_token = EnvGuard::set("AXS_DISPATCH_TOKEN", "test-dispatch-token");
        let _tls_profile = EnvGuard::set("AXS_TLS_PROFILE", "trusted_mesh");
        let _node_runtime = EnvGuard::remove("AXS_NODE_RUNTIME");
        let _node_runtime_url = EnvGuard::remove("AXS_NODE_RUNTIME_URL");
        let _node_listen = EnvGuard::remove("AXS_NODE_LISTEN_ADDR");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");
        let _node_hardware = EnvGuard::remove("AXS_NODE_HARDWARE_CLASS");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.listen_addr.to_string(), "0.0.0.0:18081");
        assert_eq!(config.advertised_url, "http://127.0.0.1:18081");
        assert_eq!(config.runtime, "vllm");
        assert_eq!(config.hardware_class, "thor");
    }

    #[test]
    fn from_env_defaults_wildcard_listen_to_loopback_advertised_addr() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::remove("AXS_THOR_ADVERTISED_ADDR");
        let _dispatch_token = EnvGuard::set("AXS_DISPATCH_TOKEN", "test-dispatch-token");
        let _tls_profile = EnvGuard::set("AXS_TLS_PROFILE", "trusted_mesh");
        let _node_listen = EnvGuard::remove("AXS_NODE_LISTEN_ADDR");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.listen_addr.to_string(), "0.0.0.0:18081");
        assert_eq!(config.advertised_url, "http://127.0.0.1:18081");
    }

    #[test]
    fn from_env_requires_dispatch_token_for_non_loopback_listener() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_NODE_ADVERTISED_ADDR", "127.0.0.1:18081");
        let _dispatch_token = EnvGuard::remove("AXS_DISPATCH_TOKEN");
        let _tls_profile = EnvGuard::remove("AXS_TLS_PROFILE");

        let error = ThorConfig::from_env().unwrap_err().to_string();
        assert!(error.contains("AXS_DISPATCH_TOKEN"));
    }

    #[test]
    fn default_advertised_url_preserves_routable_listen_addr() {
        let listen = "10.0.0.7:18081".parse().unwrap();
        assert_eq!(default_advertised_url(listen), "http://10.0.0.7:18081");
    }

    #[test]
    fn parse_advertised_url_accepts_dns_host() {
        let url = parse_advertised_url("https://agent.runtime.svc.cluster.local:18443").unwrap();
        assert_eq!(url, "https://agent.runtime.svc.cluster.local:18443");
    }

    #[test]
    fn normalize_http_base_url_adds_http_scheme_if_missing() {
        let normalized =
            normalize_http_base_url("127.0.0.1:19090", "AXS_CONTROL_PLANE_URL").unwrap();
        assert_eq!(normalized, "http://127.0.0.1:19090");
    }

    #[test]
    fn normalize_http_base_url_trims_trailing_slashes() {
        let normalized =
            normalize_http_base_url(" https://127.0.0.1:19090// ", "AXS_CONTROL_PLANE_URL")
                .unwrap();
        assert_eq!(normalized, "https://127.0.0.1:19090");
    }

    #[test]
    fn normalize_http_base_url_rejects_path_query_and_fragment() {
        let path = normalize_http_base_url("http://127.0.0.1:19090/api", "runtime URL")
            .unwrap_err()
            .to_string();
        assert!(path.contains("must not include a path"), "got: {path}");

        let query = normalize_http_base_url("http://127.0.0.1:19090?x=1", "runtime URL")
            .unwrap_err()
            .to_string();
        assert!(
            query.contains("must not include query params or fragments"),
            "got: {query}"
        );

        let fragment = normalize_http_base_url("http://127.0.0.1:19090#runtime", "runtime URL")
            .unwrap_err()
            .to_string();
        assert!(
            fragment.contains("must not include query params or fragments"),
            "got: {fragment}"
        );
    }

    #[test]
    fn normalize_http_base_url_rejects_unsupported_scheme() {
        let err = normalize_http_base_url("ftp://127.0.0.1:19090", "runtime URL")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported URL scheme"), "got: {err}");
    }

    #[test]
    fn from_env_accepts_generic_runtime_node_aliases() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", " 127.0.0.1:8080// ");
        let _runtime_url = EnvGuard::set("AXS_NODE_RUNTIME_URL", " 127.0.0.1:9000// ");
        let _runtime = EnvGuard::set("AXS_NODE_RUNTIME", "ax_engine");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "127.0.0.1:18091");
        let _advertised = EnvGuard::set("AXS_NODE_ADVERTISED_ADDR", "127.0.0.1:18092");
        let _max_inflight = EnvGuard::set("AXS_NODE_MAX_INFLIGHT", "12");
        let _pool = EnvGuard::set("AXS_NODE_WORKER_POOL", "mac");
        let _node_class = EnvGuard::set("AXS_NODE_CLASS", "mac-studio");
        let _hardware_class = EnvGuard::set("AXS_NODE_HARDWARE_CLASS", "mac");
        let _embedding = EnvGuard::set("AXS_NODE_EMBEDDING", "true");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.control_plane_url, "http://127.0.0.1:8080");
        assert_eq!(config.runtime_url, "http://127.0.0.1:9000");
        assert_eq!(config.runtime, "ax_engine");
        assert_eq!(config.listen_addr.to_string(), "127.0.0.1:18091");
        assert_eq!(config.advertised_url, "http://127.0.0.1:18092");
        assert_eq!(config.max_inflight, 12);
        assert_eq!(config.worker_pool.as_deref(), Some("mac"));
        assert_eq!(config.node_class, "mac-studio");
        assert_eq!(config.hardware_class, "mac");
        assert_eq!(config.embedding, Some(true));
    }

    #[test]
    fn from_env_accepts_dns_advertised_hostname() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_THOR_ADVERTISED_ADDR", "thor-node.local:18081");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");
        let _node_url = EnvGuard::remove("AXS_NODE_ADVERTISED_URL");
        let _dispatch_token = EnvGuard::set("AXS_DISPATCH_TOKEN", "test-dispatch-token");
        let _tls_profile = EnvGuard::set("AXS_TLS_PROFILE", "trusted_mesh");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(config.advertised_url, "http://thor-node.local:18081");
    }

    #[test]
    fn from_env_accepts_advertised_url_env() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _url = EnvGuard::set(
            "AXS_NODE_ADVERTISED_URL",
            "https://ax-runtime-agent.runtime.svc.cluster.local:18443",
        );
        let _legacy = EnvGuard::remove("AXS_THOR_ADVERTISED_ADDR");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");
        let _dispatch_token = EnvGuard::set("AXS_DISPATCH_TOKEN", "test-dispatch-token");
        let _tls_profile = EnvGuard::set("AXS_TLS_PROFILE", "trusted_mesh");

        let config = ThorConfig::from_env().unwrap();
        assert_eq!(
            config.advertised_url,
            "https://ax-runtime-agent.runtime.svc.cluster.local:18443"
        );
    }

    #[test]
    fn from_env_rejects_explicit_wildcard_advertised_addr() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_THOR_LISTEN_ADDR", "0.0.0.0:18081");
        let _advertised = EnvGuard::set("AXS_THOR_ADVERTISED_ADDR", "0.0.0.0:18081");
        let _node_advertised = EnvGuard::remove("AXS_NODE_ADVERTISED_ADDR");
        let _node_url = EnvGuard::remove("AXS_NODE_ADVERTISED_URL");
        let _dispatch_token = EnvGuard::set("AXS_DISPATCH_TOKEN", "test-dispatch-token");
        let _tls_profile = EnvGuard::set("AXS_TLS_PROFILE", "trusted_mesh");

        let err = ThorConfig::from_env().unwrap_err();
        let err = format!("{err:#}");
        assert!(
            err.to_lowercase().contains("wildcard") || err.contains("0.0.0.0"),
            "got: {err}"
        );
    }

    #[test]
    fn from_env_rejects_invalid_numeric_overrides() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "127.0.0.1:18081");
        let _advertised = EnvGuard::set("AXS_NODE_ADVERTISED_ADDR", "127.0.0.1:18081");
        let _max_inflight = EnvGuard::set("AXS_NODE_MAX_INFLIGHT", "many");

        let err = ThorConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("AXS_NODE_MAX_INFLIGHT"));
    }

    #[test]
    fn from_env_rejects_invalid_bool_overrides() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "127.0.0.1:18081");
        let _advertised = EnvGuard::set("AXS_NODE_ADVERTISED_ADDR", "127.0.0.1:18081");
        let _embedding = EnvGuard::set("AXS_NODE_EMBEDDING", "sure");

        let err = ThorConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("AXS_NODE_EMBEDDING"));
    }

    #[test]
    fn debug_output_redacts_worker_and_runtime_credentials() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "127.0.0.1:18081");
        let _worker_token = EnvGuard::set("AXS_WORKER_TOKEN", "worker-secret-value");
        let _runtime_key = EnvGuard::set("AXS_RUNTIME_API_KEY", "runtime-secret-value");
        let _dispatch_token = EnvGuard::set("AXS_DISPATCH_TOKEN", "dispatch-secret-value");

        let config = ThorConfig::from_env().unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains("worker-secret-value"));
        assert!(!debug.contains("runtime-secret-value"));
        assert!(!debug.contains("dispatch-secret-value"));
        assert!(debug.contains("worker_token_configured: true"));
        assert!(debug.contains("runtime_api_key_configured: true"));
        assert!(debug.contains("dispatch_token_configured: true"));
    }
}
