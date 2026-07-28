//! Environment configuration for one Dynamo execution-domain adapter.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use ax_serving_protocol::{
    DomainId, ExecutionDomainKind, PoolId, QualificationState, TrustDomainId, WorkerId,
};

const DEFAULT_CONTROL_PLANE_URL: &str = "http://127.0.0.1:19090";
const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:18082";
const DEFAULT_PROBE_INTERVAL_MS: u64 = 5_000;
const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 300;
const DEFAULT_MAX_INFLIGHT: usize = 64;

#[derive(Clone)]
pub struct DynamoAdapterConfig {
    pub control_plane_url: String,
    pub worker_token: Option<String>,
    pub dispatch_token: Option<String>,
    pub frontend_url: String,
    pub dynamo_api_key: Option<String>,
    pub manifest_path: PathBuf,
    pub domain_id: DomainId,
    pub domain_kind: ExecutionDomainKind,
    pub worker_id: WorkerId,
    pub pool_id: PoolId,
    pub trust_domain: TrustDomainId,
    pub qualification: QualificationState,
    pub hardware_class: String,
    pub listen_addr: SocketAddr,
    pub advertised_url: String,
    pub probe_interval_ms: u64,
    pub drain_timeout_secs: u64,
    pub max_inflight: usize,
    pub tls_profile: String,
    pub allow_no_auth: bool,
}

impl std::fmt::Debug for DynamoAdapterConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamoAdapterConfig")
            .field("control_plane_url", &self.control_plane_url)
            .field(
                "worker_token",
                &self.worker_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "dispatch_token",
                &self.dispatch_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("frontend_url", &self.frontend_url)
            .field(
                "dynamo_api_key",
                &self.dynamo_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field("manifest_path", &self.manifest_path)
            .field("domain_id", &self.domain_id)
            .field("domain_kind", &self.domain_kind)
            .field("worker_id", &self.worker_id)
            .field("pool_id", &self.pool_id)
            .field("trust_domain", &self.trust_domain)
            .field("qualification", &self.qualification)
            .field("hardware_class", &self.hardware_class)
            .field("listen_addr", &self.listen_addr)
            .field("advertised_url", &self.advertised_url)
            .field("probe_interval_ms", &self.probe_interval_ms)
            .field("drain_timeout_secs", &self.drain_timeout_secs)
            .field("max_inflight", &self.max_inflight)
            .field("tls_profile", &self.tls_profile)
            .field("allow_no_auth", &self.allow_no_auth)
            .finish()
    }
}

impl DynamoAdapterConfig {
    pub fn from_env() -> Result<Self> {
        let tls_profile = optional_env("AXS_TLS_PROFILE")
            .unwrap_or_else(|| "loopback_dev".to_string())
            .to_ascii_lowercase();
        if !matches!(tls_profile.as_str(), "loopback_dev" | "trusted_mesh") {
            bail!("AXS_TLS_PROFILE must be loopback_dev or trusted_mesh");
        }
        let allow_no_auth = optional_env("AXS_ALLOW_NO_AUTH")
            .map(|value| parse_bool("AXS_ALLOW_NO_AUTH", &value))
            .transpose()?
            .unwrap_or(false);

        let domain_id = DomainId::new(required_env("AXS_DYNAMO_DOMAIN_ID")?)
            .context("invalid AXS_DYNAMO_DOMAIN_ID")?;
        let domain_kind = parse_domain_kind(&required_env("AXS_DYNAMO_DOMAIN_KIND")?)?;
        let frontend_url = normalize_service_url(
            &required_env("AXS_DYNAMO_FRONTEND_URL")?,
            "AXS_DYNAMO_FRONTEND_URL",
            &tls_profile,
        )?;
        let control_plane_url = normalize_service_url(
            optional_env("AXS_CONTROL_PLANE_URL")
                .as_deref()
                .unwrap_or(DEFAULT_CONTROL_PLANE_URL),
            "AXS_CONTROL_PLANE_URL",
            &tls_profile,
        )?;
        let manifest_path = PathBuf::from(required_env("AXS_DYNAMO_MANIFEST_PATH")?);

        let listen_addr = optional_env("AXS_DYNAMO_LISTEN_ADDR")
            .unwrap_or_else(|| DEFAULT_LISTEN_ADDR.to_string())
            .parse::<SocketAddr>()
            .context("invalid AXS_DYNAMO_LISTEN_ADDR")?;
        let advertised_url = match optional_env("AXS_DYNAMO_ADVERTISED_URL") {
            Some(value) => {
                normalize_service_url(&value, "AXS_DYNAMO_ADVERTISED_URL", &tls_profile)?
            }
            None if listen_addr.ip().is_unspecified() => {
                bail!(
                    "AXS_DYNAMO_ADVERTISED_URL is required when AXS_DYNAMO_LISTEN_ADDR uses a wildcard address"
                )
            }
            None => format!("http://{listen_addr}"),
        };

        let worker_id = WorkerId::new(
            optional_env("AXS_DYNAMO_ADAPTER_ID")
                .unwrap_or_else(|| format!("dynamo-{}", domain_id.as_str())),
        )
        .context("invalid AXS_DYNAMO_ADAPTER_ID")?;
        let pool_id = PoolId::new(
            optional_env("AXS_DYNAMO_POOL_ID")
                .unwrap_or_else(|| format!("{}-pool", domain_id.as_str())),
        )
        .context("invalid AXS_DYNAMO_POOL_ID")?;
        let trust_domain = TrustDomainId::new(
            optional_env("AXS_DYNAMO_TRUST_DOMAIN").unwrap_or_else(|| "local".into()),
        )
        .context("invalid AXS_DYNAMO_TRUST_DOMAIN")?;
        let qualification = parse_qualification(
            optional_env("AXS_DYNAMO_QUALIFICATION")
                .as_deref()
                .unwrap_or("experimental"),
        )?;
        let hardware_class = optional_env("AXS_DYNAMO_HARDWARE_CLASS")
            .unwrap_or_else(|| default_hardware_class(domain_kind).into());
        validate_metadata_token("AXS_DYNAMO_HARDWARE_CLASS", &hardware_class)?;

        let probe_interval_ms = optional_env("AXS_DYNAMO_PROBE_INTERVAL_MS")
            .map(|value| parse_u64("AXS_DYNAMO_PROBE_INTERVAL_MS", &value))
            .transpose()?
            .unwrap_or(DEFAULT_PROBE_INTERVAL_MS)
            .clamp(1_000, 300_000);
        let drain_timeout_secs = optional_env("AXS_DYNAMO_DRAIN_TIMEOUT_SECS")
            .map(|value| parse_u64("AXS_DYNAMO_DRAIN_TIMEOUT_SECS", &value))
            .transpose()?
            .unwrap_or(DEFAULT_DRAIN_TIMEOUT_SECS)
            .clamp(1, 3_600);
        let max_inflight = optional_env("AXS_DYNAMO_MAX_INFLIGHT")
            .map(|value| parse_usize("AXS_DYNAMO_MAX_INFLIGHT", &value))
            .transpose()?
            .unwrap_or(DEFAULT_MAX_INFLIGHT);
        if max_inflight == 0 {
            bail!("AXS_DYNAMO_MAX_INFLIGHT must be greater than zero");
        }

        let worker_token = optional_env("AXS_WORKER_TOKEN");
        let dispatch_token = optional_env("AXS_DISPATCH_TOKEN");
        if dispatch_token.is_none()
            && !permits_unauthenticated_dispatch(&tls_profile, allow_no_auth, listen_addr)
        {
            bail!(
                "AXS_DISPATCH_TOKEN is required unless a loopback listener explicitly enables loopback_dev AXS_ALLOW_NO_AUTH"
            );
        }

        Ok(Self {
            control_plane_url,
            worker_token,
            dispatch_token,
            frontend_url,
            dynamo_api_key: optional_env("AXS_DYNAMO_API_KEY"),
            manifest_path,
            domain_id,
            domain_kind,
            worker_id,
            pool_id,
            trust_domain,
            qualification,
            hardware_class,
            listen_addr,
            advertised_url,
            probe_interval_ms,
            drain_timeout_secs,
            max_inflight,
            tls_profile,
            allow_no_auth,
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

fn parse_domain_kind(value: &str) -> Result<ExecutionDomainKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "nvidia_dynamo_pc" => Ok(ExecutionDomainKind::NvidiaDynamoPc),
        "nvidia_dynamo_thor" => Ok(ExecutionDomainKind::NvidiaDynamoThor),
        _ => bail!("AXS_DYNAMO_DOMAIN_KIND must be nvidia_dynamo_pc or nvidia_dynamo_thor"),
    }
}

fn parse_qualification(value: &str) -> Result<QualificationState> {
    match value.trim().to_ascii_lowercase().as_str() {
        "unverified" => Ok(QualificationState::Unverified),
        "experimental" => Ok(QualificationState::Experimental),
        "certified" => Ok(QualificationState::Certified),
        _ => bail!("AXS_DYNAMO_QUALIFICATION must be unverified, experimental, or certified"),
    }
}

fn default_hardware_class(kind: ExecutionDomainKind) -> &'static str {
    match kind {
        ExecutionDomainKind::NvidiaDynamoPc => "nvidia-pc-cuda",
        ExecutionDomainKind::NvidiaDynamoThor => "nvidia-thor",
        _ => "invalid",
    }
}

fn normalize_service_url(raw: &str, field: &'static str, tls_profile: &str) -> Result<String> {
    let with_scheme = if raw.contains("://") {
        raw.trim().to_string()
    } else {
        format!("http://{}", raw.trim())
    };
    let mut url =
        reqwest::Url::parse(&with_scheme).with_context(|| format!("invalid {field} URL"))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("{field} must use http or https");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("{field} must not contain embedded credentials");
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("{field} must not contain a query or fragment");
    }
    if !matches!(url.path(), "" | "/") {
        bail!("{field} must be an origin URL without a path");
    }
    let host = url
        .host_str()
        .with_context(|| format!("{field} has no host"))?;
    reject_unsafe_host(field, host)?;
    if url.scheme() == "http" && !host_is_loopback(host) && tls_profile != "trusted_mesh" {
        bail!("{field} requires https outside loopback unless AXS_TLS_PROFILE=trusted_mesh");
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

fn validate_metadata_token(field: &'static str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value.trim() != value
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'+' | b'-')
        })
    {
        bail!("{field} contains invalid metadata");
    }
    Ok(())
}

fn parse_bool(key: &'static str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => bail!("{key} must be true, false, 1, or 0"),
    }
}

fn parse_u64(key: &'static str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .with_context(|| format!("{key} must be an unsigned integer"))
}

fn parse_usize(key: &'static str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("{key} must be an unsigned integer"))
}

fn permits_unauthenticated_dispatch(
    tls_profile: &str,
    allow_no_auth: bool,
    listen_addr: SocketAddr,
) -> bool {
    tls_profile == "loopback_dev" && allow_no_auth && listen_addr.ip().is_loopback()
}

#[cfg(test)]
mod tests {
    use super::{normalize_service_url, parse_domain_kind, permits_unauthenticated_dispatch};
    use ax_serving_protocol::ExecutionDomainKind;

    #[test]
    fn accepts_only_dynamo_domain_kinds() {
        assert_eq!(
            parse_domain_kind("nvidia_dynamo_pc").unwrap(),
            ExecutionDomainKind::NvidiaDynamoPc
        );
        assert!(parse_domain_kind("compatibility_runtime_endpoint").is_err());
    }

    #[test]
    fn service_url_rejects_credentials_paths_and_link_local() {
        assert!(
            normalize_service_url(
                "http://user:pass@localhost:8000",
                "TEST_URL",
                "loopback_dev"
            )
            .is_err()
        );
        assert!(
            normalize_service_url("http://localhost:8000/v1", "TEST_URL", "loopback_dev").is_err()
        );
        assert!(
            normalize_service_url("http://169.254.1.2:8000", "TEST_URL", "trusted_mesh").is_err()
        );
    }

    #[test]
    fn plaintext_remote_requires_trusted_mesh() {
        assert!(
            normalize_service_url("http://dynamo.internal:8000", "TEST_URL", "loopback_dev")
                .is_err()
        );
        assert_eq!(
            normalize_service_url("http://dynamo.internal:8000", "TEST_URL", "trusted_mesh")
                .unwrap(),
            "http://dynamo.internal:8000"
        );
    }

    #[test]
    fn unauthenticated_dispatch_is_loopback_only() {
        assert!(permits_unauthenticated_dispatch(
            "loopback_dev",
            true,
            "127.0.0.1:18082".parse().unwrap()
        ));
        assert!(!permits_unauthenticated_dispatch(
            "loopback_dev",
            true,
            "0.0.0.0:18082".parse().unwrap()
        ));
        assert!(!permits_unauthenticated_dispatch(
            "trusted_mesh",
            true,
            "127.0.0.1:18082".parse().unwrap()
        ));
    }
}
