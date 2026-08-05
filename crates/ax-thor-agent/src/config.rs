use std::net::SocketAddr;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{
    CompatibilityManifestDigest, Digest, DomainId, ProtocolCapability, QualificationState,
};

const DEFAULT_RUNTIME_URL: &str = "http://127.0.0.1:8000";
const DEFAULT_NODE_LISTEN_ADDR: &str = "0.0.0.0:18081";
const DEFAULT_MAX_INFLIGHT: usize = 8;
const DEFAULT_NODE_CLASS: &str = "mac";
const DEFAULT_RUNTIME: &str = "ax_engine";
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

/// Optional protocol-v1.1 node-domain declaration.
///
/// The runtime agent derives the domain kind from its runtime: AX Engine is a
/// `mac_ax_engine` node and every other direct runtime is compatibility-only.
#[derive(Clone, Debug)]
pub struct ExecutionDomainConfig {
    pub id: DomainId,
    pub qualification: QualificationState,
    pub compatibility_manifest: Option<CompatibilityManifestDigest>,
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
    let explicit = std::env::var("AXS_CONTROL_PLANE_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    let lan_enabled = ax_serving_discovery::discover_lan_enabled();
    let mut candidates = Vec::new();
    if explicit.is_none() && lan_enabled {
        let filter = ax_serving_discovery::filter_from_env_for(
            ax_serving_discovery::DiscoveredKind::AxServingGateway,
        );
        let timeout = ax_serving_discovery::discover_timeout_from_env();
        tracing::info!(
            ?timeout,
            cluster = ?filter.cluster,
            "AXS_DISCOVER_LAN enabled; browsing for _ax-serving-gateway._tcp"
        );
        candidates = ax_serving_discovery::browse_gateways(timeout, &filter)
            .context("LAN discovery browse for AX Serving gateway failed")?;
    }

    let filter = ax_serving_discovery::filter_from_env_for(
        ax_serving_discovery::DiscoveredKind::AxServingGateway,
    );
    let resolved = ax_serving_discovery::resolve_base_url(
        explicit.as_deref(),
        lan_enabled && explicit.is_none(),
        &candidates,
        &filter,
        None,
        ax_serving_discovery::ResolveRole::Gateway,
    )
    .context("AXS_CONTROL_PLANE_URL is required (or enable AXS_DISCOVER_LAN with a gateway advertising on the LAN)")?;
    normalize_http_base_url(&resolved, "AXS_CONTROL_PLANE_URL")
}

/// Resolve the upstream runtime HTTP base URL.
///
/// Priority:
/// 1. Explicit `AXS_NODE_RUNTIME_URL` / legacy aliases
/// 2. When `AXS_DISCOVER_LAN=1` and runtime is `ax_engine`, mDNS browse
/// 3. Loopback default (legacy local agent bring-up)
fn resolve_runtime_url(runtime_kind: &str) -> Result<String> {
    let explicit = load_first_optional_string_env(&[
        "AXS_NODE_RUNTIME_URL",
        "AXS_THOR_RUNTIME_URL",
        "AXS_SGLANG_URL",
    ]);

    let kind = runtime_kind.trim().to_ascii_lowercase().replace('-', "_");
    let can_discover = matches!(kind.as_str(), "ax_engine" | "axengine" | "native");
    let lan_enabled = ax_serving_discovery::discover_lan_enabled() && can_discover;

    let mut candidates = Vec::new();
    if explicit.is_none() && lan_enabled {
        let filter = ax_serving_discovery::filter_from_env();
        let timeout = ax_serving_discovery::discover_timeout_from_env();
        tracing::info!(
            ?timeout,
            cluster = ?filter.cluster,
            "AXS_DISCOVER_LAN enabled; browsing for _ax-engine._tcp"
        );
        candidates = ax_serving_discovery::browse_engines(timeout, &filter)
            .context("LAN discovery browse for AX Engine failed")?;
        if let Ok(selected) = ax_serving_discovery::select_unique_engine(&candidates, &filter) {
            tracing::info!(
                base_url = %selected.base_url,
                instance = %selected.instance_name,
                model = ?selected.model_id,
                "resolved AX Engine runtime via LAN discovery"
            );
        }
    }

    let filter = ax_serving_discovery::filter_from_env();
    let resolved = ax_serving_discovery::resolve_base_url(
        explicit.as_deref(),
        lan_enabled && explicit.is_none(),
        &candidates,
        &filter,
        Some(DEFAULT_RUNTIME_URL),
        ax_serving_discovery::ResolveRole::Engine,
    )?;
    Ok(resolved)
}

fn normalize_http_base_url(raw: &str, field: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("{field} URL is empty");
    }

    // Detect the scheme before trimming trailing slashes, so inputs like
    // "http://" cannot collapse into a slash-less string that evades the
    // empty-host check and gets re-prefixed into a garbage URL.
    let (scheme, rest) = if let Some(scheme_end) = trimmed.find("://") {
        let scheme = &trimmed[..scheme_end];
        if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
            (Some(scheme), &trimmed[scheme_end + 3..])
        } else {
            bail!("{field} has unsupported URL scheme: {trimmed}");
        }
    } else {
        (None, trimmed)
    };

    let rest = rest.trim_end_matches('/');
    if rest.is_empty() {
        bail!("{field} URL is missing a host: {trimmed}");
    }
    if rest.contains('/') {
        bail!("{field} URL must not include a path: {trimmed}");
    }
    if rest.contains('?') || rest.contains('#') {
        bail!("{field} URL must not include query params or fragments: {trimmed}");
    }

    match scheme {
        Some(scheme) => Ok(format!("{scheme}://{rest}")),
        None => Ok(format!("http://{rest}")),
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
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        let link_local = match ip {
            std::net::IpAddr::V4(v4) => v4.is_link_local(),
            std::net::IpAddr::V6(v6) => v6.is_unicast_link_local(),
        };
        if ip.is_unspecified() || ip.is_multicast() || link_local {
            bail!(
                "advertised address {candidate} is not routable (wildcard, multicast, or link-local); set \
                 AXS_NODE_ADVERTISED_URL or AXS_NODE_ADVERTISED_ADDR to a routable host"
            );
        }
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
    /// Optional explicit execution-domain identity advertised through protocol v1.1.
    pub execution_domain: Option<ExecutionDomainConfig>,
    pub friendly_name: Option<String>,
    pub chip_model: Option<String>,
    /// env: `AXS_NODE_SHUTDOWN_TIMEOUT_SECS` (legacy `AXS_THOR_*` aliases remain accepted).
    pub shutdown_timeout_secs: Option<u64>,
    /// env: `AXS_NODE_MAX_CONTEXT` — max context window advertised to control
    /// plane. If unset, the agent tries to derive it from the runtime.
    pub max_context: Option<u32>,
    /// env: `AXS_NODE_EMBEDDING` — override embedding capability (true/false).
    /// If unset, the agent derives the capability from runtime model metadata.
    pub embedding: Option<bool>,
    /// env: `AXS_NODE_VISION` — override vision capability (true/false).
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
            .field("execution_domain", &self.execution_domain)
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
        let runtime = load_first_optional_string_env(&[
            "AXS_NODE_RUNTIME",
            "AXS_THOR_RUNTIME",
            "AXS_THOR_BACKEND",
        ])
        .unwrap_or_else(|| DEFAULT_RUNTIME.into());
        let runtime_url = resolve_runtime_url(&runtime)?;
        let runtime_version =
            load_optional_string_env("AXS_RUNTIME_VERSION").unwrap_or_else(|| "unknown".into());
        let listen_addr: SocketAddr =
            load_first_optional_string_env(&["AXS_NODE_LISTEN_ADDR", "AXS_THOR_LISTEN_ADDR"])
                .unwrap_or_else(|| DEFAULT_NODE_LISTEN_ADDR.into())
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
        let domain_id = load_optional_string_env("AXS_NODE_DOMAIN_ID")
            .map(DomainId::new)
            .transpose()
            .context("invalid AXS_NODE_DOMAIN_ID")?;
        let domain_qualification = load_optional_string_env("AXS_NODE_DOMAIN_QUALIFICATION");
        let domain_manifest = load_optional_string_env("AXS_NODE_DOMAIN_COMPATIBILITY_MANIFEST")
            .map(CompatibilityManifestDigest::new)
            .transpose()
            .context("invalid AXS_NODE_DOMAIN_COMPATIBILITY_MANIFEST")?;
        if domain_id.is_none() && (domain_qualification.is_some() || domain_manifest.is_some()) {
            bail!(
                "AXS_NODE_DOMAIN_ID is required when domain qualification or compatibility manifest is configured"
            );
        }
        let execution_domain = domain_id
            .map(|id| {
                Ok::<_, anyhow::Error>(ExecutionDomainConfig {
                    id,
                    qualification: parse_domain_qualification(domain_qualification.as_deref())?,
                    compatibility_manifest: domain_manifest,
                })
            })
            .transpose()?;
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
            execution_domain,
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

fn parse_domain_qualification(value: Option<&str>) -> Result<QualificationState> {
    match value
        .unwrap_or("unverified")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "unverified" => Ok(QualificationState::Unverified),
        "experimental" => Ok(QualificationState::Experimental),
        "certified" => Ok(QualificationState::Certified),
        "suspended" => Ok(QualificationState::Suspended),
        other => bail!(
            "invalid AXS_NODE_DOMAIN_QUALIFICATION {other:?}; expected unverified, experimental, certified, or suspended"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QualificationState, ThorConfig, default_advertised_url, normalize_http_base_url,
        parse_advertised_url,
    };
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
        assert_eq!(config.runtime, "ax_engine");
        assert_eq!(config.hardware_class, "mac");
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
    fn parse_advertised_url_rejects_link_local_and_multicast() {
        let err = parse_advertised_url("http://169.254.10.1:18081")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not routable"), "got: {err}");
        let err = parse_advertised_url("http://224.0.0.1:18081")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not routable"), "got: {err}");
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
    fn normalize_http_base_url_rejects_scheme_without_host() {
        for raw in ["http://", "https://", " http://// "] {
            let err = normalize_http_base_url(raw, "AXS_CONTROL_PLANE_URL")
                .unwrap_err()
                .to_string();
            assert!(err.contains("missing a host"), "input {raw:?}: got: {err}");
        }
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
        let _domain_id = EnvGuard::set("AXS_NODE_DOMAIN_ID", "mac-studio-1");
        let _domain_qualification = EnvGuard::set("AXS_NODE_DOMAIN_QUALIFICATION", "certified");
        let _domain_manifest = EnvGuard::remove("AXS_NODE_DOMAIN_COMPATIBILITY_MANIFEST");
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
        assert_eq!(
            config
                .execution_domain
                .as_ref()
                .map(|domain| domain.id.as_str()),
            Some("mac-studio-1")
        );
        assert_eq!(
            config
                .execution_domain
                .as_ref()
                .map(|domain| domain.qualification),
            Some(QualificationState::Certified)
        );
        assert_eq!(config.embedding, Some(true));
    }

    #[test]
    fn from_env_rejects_domain_metadata_without_domain_id() {
        let _lock = crate::test_env::lock();
        let _control = EnvGuard::set("AXS_CONTROL_PLANE_URL", "http://127.0.0.1:8080");
        let _runtime_url = EnvGuard::set("AXS_NODE_RUNTIME_URL", "http://127.0.0.1:9000");
        let _listen = EnvGuard::set("AXS_NODE_LISTEN_ADDR", "127.0.0.1:18091");
        let _domain_id = EnvGuard::remove("AXS_NODE_DOMAIN_ID");
        let _domain_qualification = EnvGuard::set("AXS_NODE_DOMAIN_QUALIFICATION", "certified");
        let _domain_manifest = EnvGuard::remove("AXS_NODE_DOMAIN_COMPATIBILITY_MANIFEST");

        let error = ThorConfig::from_env().unwrap_err().to_string();
        assert!(error.contains("AXS_NODE_DOMAIN_ID"));
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
