//! Environment configuration for one Mac cluster domain adapter.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{DomainId, PoolId, QualificationState, TrustDomainId, WorkerId};

const DEFAULT_CONTROL_PLANE_URL: &str = "http://127.0.0.1:19090";
const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:18083";
const DEFAULT_MAX_INFLIGHT: usize = 2;
const DEFAULT_RANK_STALE_MS: u64 = 15_000;
const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 300;

#[derive(Clone)]
pub struct MacClusterConfig {
    pub control_plane_url: String,
    pub worker_token: Option<String>,
    pub dispatch_token: Option<String>,
    pub rank_control_token: String,
    pub rank0_url: String,
    pub rank0_token: String,
    pub manifest_path: PathBuf,
    pub domain_id: DomainId,
    pub worker_id: WorkerId,
    pub pool_id: PoolId,
    pub trust_domain: TrustDomainId,
    pub qualification: QualificationState,
    pub hardware_class: String,
    pub listen_addr: SocketAddr,
    pub advertised_url: String,
    pub max_inflight: usize,
    pub rank_stale_ms: u64,
    pub drain_timeout_secs: u64,
}

impl std::fmt::Debug for MacClusterConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MacClusterConfig")
            .field("control_plane_url", &self.control_plane_url)
            .field(
                "worker_token",
                &self.worker_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "dispatch_token",
                &self.dispatch_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("rank_control_token", &"[REDACTED]")
            .field("rank0_url", &self.rank0_url)
            .field("rank0_token", &"[REDACTED]")
            .field("manifest_path", &self.manifest_path)
            .field("domain_id", &self.domain_id)
            .field("worker_id", &self.worker_id)
            .field("pool_id", &self.pool_id)
            .field("trust_domain", &self.trust_domain)
            .field("qualification", &self.qualification)
            .field("hardware_class", &self.hardware_class)
            .field("listen_addr", &self.listen_addr)
            .field("advertised_url", &self.advertised_url)
            .field("max_inflight", &self.max_inflight)
            .field("rank_stale_ms", &self.rank_stale_ms)
            .field("drain_timeout_secs", &self.drain_timeout_secs)
            .finish()
    }
}

impl MacClusterConfig {
    pub fn from_env() -> Result<Self> {
        let tls_profile = optional_env("AXS_TLS_PROFILE")
            .unwrap_or_else(|| "loopback_dev".into())
            .to_ascii_lowercase();
        if !matches!(tls_profile.as_str(), "loopback_dev" | "trusted_mesh") {
            bail!("AXS_TLS_PROFILE must be loopback_dev or trusted_mesh");
        }
        let trusted_mesh = tls_profile == "trusted_mesh";
        let allow_no_auth = optional_env("AXS_ALLOW_NO_AUTH")
            .map(|value| parse_bool("AXS_ALLOW_NO_AUTH", &value))
            .transpose()?
            .unwrap_or(false);
        let domain_id = DomainId::new(required_env("AXS_MAC_CLUSTER_DOMAIN_ID")?)
            .context("invalid AXS_MAC_CLUSTER_DOMAIN_ID")?;
        let listen_addr = optional_env("AXS_MAC_CLUSTER_LISTEN_ADDR")
            .unwrap_or_else(|| DEFAULT_LISTEN_ADDR.into())
            .parse::<SocketAddr>()
            .context("invalid AXS_MAC_CLUSTER_LISTEN_ADDR")?;
        let advertised_url = match optional_env("AXS_MAC_CLUSTER_ADVERTISED_URL") {
            Some(value) => {
                normalize_service_url(&value, "AXS_MAC_CLUSTER_ADVERTISED_URL", trusted_mesh)?
            }
            None if listen_addr.ip().is_unspecified() => {
                bail!("AXS_MAC_CLUSTER_ADVERTISED_URL is required for a wildcard listen address")
            }
            None => format!("http://{listen_addr}"),
        };
        let control_plane_url = normalize_service_url(
            optional_env("AXS_CONTROL_PLANE_URL")
                .as_deref()
                .unwrap_or(DEFAULT_CONTROL_PLANE_URL),
            "AXS_CONTROL_PLANE_URL",
            trusted_mesh,
        )?;
        let rank0_url = normalize_service_url(
            &required_env("AXS_MAC_CLUSTER_RANK0_URL")?,
            "AXS_MAC_CLUSTER_RANK0_URL",
            trusted_mesh,
        )?;
        let rank_control_token = required_env("AXS_MAC_CLUSTER_CONTROL_TOKEN")?;
        validate_secret("AXS_MAC_CLUSTER_CONTROL_TOKEN", &rank_control_token)?;
        let rank0_token = required_env("AXS_MAC_CLUSTER_RANK0_TOKEN")?;
        validate_secret("AXS_MAC_CLUSTER_RANK0_TOKEN", &rank0_token)?;
        let dispatch_token = optional_env("AXS_DISPATCH_TOKEN");
        if dispatch_token.is_none()
            && !permits_unauthenticated_dispatch(&tls_profile, allow_no_auth, listen_addr)
        {
            bail!(
                "AXS_DISPATCH_TOKEN is required unless a loopback listener explicitly enables loopback_dev AXS_ALLOW_NO_AUTH"
            );
        }

        let max_inflight = optional_env("AXS_MAC_CLUSTER_MAX_INFLIGHT")
            .map(|value| parse_usize("AXS_MAC_CLUSTER_MAX_INFLIGHT", &value))
            .transpose()?
            .unwrap_or(DEFAULT_MAX_INFLIGHT);
        if max_inflight == 0 {
            bail!("AXS_MAC_CLUSTER_MAX_INFLIGHT must be greater than zero");
        }

        Ok(Self {
            control_plane_url,
            worker_token: optional_env("AXS_WORKER_TOKEN"),
            dispatch_token,
            rank_control_token,
            rank0_url,
            rank0_token,
            manifest_path: PathBuf::from(required_env("AXS_MAC_CLUSTER_MANIFEST_PATH")?),
            worker_id: WorkerId::new(
                optional_env("AXS_MAC_CLUSTER_ADAPTER_ID")
                    .unwrap_or_else(|| format!("mac-cluster-{}", domain_id.as_str())),
            )
            .context("invalid AXS_MAC_CLUSTER_ADAPTER_ID")?,
            pool_id: PoolId::new(
                optional_env("AXS_MAC_CLUSTER_POOL_ID")
                    .unwrap_or_else(|| format!("{}-pool", domain_id.as_str())),
            )
            .context("invalid AXS_MAC_CLUSTER_POOL_ID")?,
            trust_domain: TrustDomainId::new(
                optional_env("AXS_MAC_CLUSTER_TRUST_DOMAIN").unwrap_or_else(|| "local".into()),
            )
            .context("invalid AXS_MAC_CLUSTER_TRUST_DOMAIN")?,
            qualification: parse_qualification(
                optional_env("AXS_MAC_CLUSTER_QUALIFICATION")
                    .as_deref()
                    .unwrap_or("experimental"),
            )?,
            hardware_class: optional_env("AXS_MAC_CLUSTER_HARDWARE_CLASS")
                .unwrap_or_else(|| "apple-silicon-cluster".into()),
            listen_addr,
            advertised_url,
            max_inflight,
            rank_stale_ms: optional_env("AXS_MAC_CLUSTER_RANK_STALE_MS")
                .map(|value| parse_u64("AXS_MAC_CLUSTER_RANK_STALE_MS", &value))
                .transpose()?
                .unwrap_or(DEFAULT_RANK_STALE_MS)
                .clamp(1_000, 300_000),
            drain_timeout_secs: optional_env("AXS_MAC_CLUSTER_DRAIN_TIMEOUT_SECS")
                .map(|value| parse_u64("AXS_MAC_CLUSTER_DRAIN_TIMEOUT_SECS", &value))
                .transpose()?
                .unwrap_or(DEFAULT_DRAIN_TIMEOUT_SECS)
                .clamp(1, 3_600),
            domain_id,
        })
    }
}

fn required_env(key: &str) -> Result<String> {
    optional_env(key).with_context(|| format!("{key} is required"))
}

fn optional_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_u64(field: &'static str, value: &str) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("{field} must be an unsigned integer"))
}

fn parse_usize(field: &'static str, value: &str) -> Result<usize> {
    value
        .parse()
        .with_context(|| format!("{field} must be a positive integer"))
}

fn parse_qualification(value: &str) -> Result<QualificationState> {
    match value.trim().to_ascii_lowercase().as_str() {
        "unverified" => Ok(QualificationState::Unverified),
        "experimental" => Ok(QualificationState::Experimental),
        "certified" => Ok(QualificationState::Certified),
        _ => bail!("AXS_MAC_CLUSTER_QUALIFICATION must be unverified, experimental, or certified"),
    }
}

fn normalize_service_url(raw: &str, field: &'static str, trusted_mesh: bool) -> Result<String> {
    let mut url = reqwest::Url::parse(raw).with_context(|| format!("invalid {field} URL"))?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        bail!("{field} must be an HTTP(S) origin without credentials, path, query, or fragment");
    }
    let host = url
        .host_str()
        .with_context(|| format!("{field} has no host"))?;
    reject_unsafe_host(field, host)?;
    if url.scheme() == "http" && !host_is_loopback(host) && !trusted_mesh {
        bail!("{field} requires HTTPS outside loopback unless AXS_TLS_PROFILE=trusted_mesh");
    }
    url.set_path("");
    Ok(url.as_str().trim_end_matches('/').to_string())
}

fn reject_unsafe_host(field: &'static str, host: &str) -> Result<()> {
    if matches!(host, "0.0.0.0" | "::" | "[::]") {
        bail!("{field} must not use a wildcard host");
    }
    if let Ok(ip) = host.parse::<IpAddr>()
        && (ip.is_unspecified() || ip.is_multicast() || is_link_local(ip))
    {
        bail!("{field} must not use an unspecified, multicast, or link-local address");
    }
    Ok(())
}

fn host_is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn is_link_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_link_local(),
        IpAddr::V6(ip) => ip.is_unicast_link_local(),
    }
}

fn parse_bool(key: &'static str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => bail!("{key} must be true, false, 1, or 0"),
    }
}

fn permits_unauthenticated_dispatch(
    tls_profile: &str,
    allow_no_auth: bool,
    listen_addr: SocketAddr,
) -> bool {
    tls_profile == "loopback_dev" && allow_no_auth && listen_addr.ip().is_loopback()
}

fn validate_secret(field: &'static str, value: &str) -> Result<()> {
    if value.len() < 16 || value.len() > 4096 || value.chars().any(char::is_control) {
        bail!("{field} must contain 16 to 4096 non-control characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{normalize_service_url, permits_unauthenticated_dispatch};

    #[test]
    fn service_url_rejects_credentials_paths_and_link_local() {
        assert!(
            normalize_service_url("http://user:pass@localhost:8000", "TEST_URL", false).is_err()
        );
        assert!(normalize_service_url("http://localhost:8000/v1", "TEST_URL", false).is_err());
        assert!(normalize_service_url("http://169.254.1.2:8000", "TEST_URL", true).is_err());
    }

    #[test]
    fn plaintext_remote_requires_trusted_mesh() {
        assert!(normalize_service_url("http://mac.internal:8000", "TEST_URL", false).is_err());
        assert_eq!(
            normalize_service_url("http://mac.internal:8000", "TEST_URL", true).unwrap(),
            "http://mac.internal:8000"
        );
    }

    #[test]
    fn unauthenticated_dispatch_is_loopback_only() {
        assert!(permits_unauthenticated_dispatch(
            "loopback_dev",
            true,
            "127.0.0.1:18083".parse().unwrap()
        ));
        assert!(!permits_unauthenticated_dispatch(
            "loopback_dev",
            true,
            "0.0.0.0:18083".parse().unwrap()
        ));
        assert!(!permits_unauthenticated_dispatch(
            "trusted_mesh",
            true,
            "127.0.0.1:18083".parse().unwrap()
        ));
    }
}
