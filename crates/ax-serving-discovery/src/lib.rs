//! LAN discovery for AX Engine runtimes and AX Serving gateways.
//!
//! Uses DNS-SD / mDNS service types shared with `ax-engine-server`
//! (`docs/designs/ax-engine-integration-and-lan-discovery-2026-07-14.md`).
//!
//! Discovery is unauthenticated and must be followed by HTTP verification and
//! normal worker tokens / API keys before fleet join.

mod advertise;

pub use advertise::{
    AdvertiseConfig, LanAdvertiser, env_truthy, is_advertisable_v4, pick_advertise_ipv4,
    sanitize_instance_name,
};

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use serde::{Deserialize, Serialize};

/// Engine HTTP runtime (OpenAI-compatible `/v1`).
pub const ENGINE_SERVICE_TYPE: &str = "_ax-engine._tcp.local.";
/// Portable AX Serving gateway control/public plane.
pub const GATEWAY_SERVICE_TYPE: &str = "_ax-serving-gateway._tcp.local.";

pub const DISCOVERY_PROTO: &str = "1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveredKind {
    AxEngine,
    AxServingGateway,
    Unknown,
}

impl DiscoveredKind {
    pub fn from_txt(kind: Option<&str>) -> Self {
        match kind.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("ax_engine") | Some("ax-engine") | Some("engine") => Self::AxEngine,
            Some("ax_serving_gateway") | Some("ax-serving-gateway") | Some("gateway") => {
                Self::AxServingGateway
            }
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredService {
    pub kind: DiscoveredKind,
    pub instance_name: String,
    pub host: String,
    pub port: u16,
    pub base_url: String,
    pub version: Option<String>,
    pub model_id: Option<String>,
    pub auth: Option<String>,
    pub cluster: Option<String>,
    pub instance_id: Option<String>,
    pub platform: Option<String>,
    pub txt: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct BrowseFilter {
    pub cluster: Option<String>,
    pub instance_name: Option<String>,
    pub instance_id: Option<String>,
    pub kind: Option<DiscoveredKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineDiscoveryDocument {
    pub schema: String,
    pub service: String,
    pub version: String,
    pub model_id: String,
    pub auth_required: bool,
    #[serde(default)]
    pub openai_base_path: Option<String>,
    #[serde(default)]
    pub operations: Vec<String>,
    #[serde(default)]
    pub cluster: Option<String>,
    #[serde(default)]
    pub instance_id: Option<String>,
}

/// Browse the LAN for DNS-SD services of `service_type` until `timeout`.
pub fn browse_services(
    service_type: &str,
    timeout: Duration,
    filter: &BrowseFilter,
) -> Result<Vec<DiscoveredService>> {
    let daemon = ServiceDaemon::new().context("failed to start mDNS daemon")?;
    let receiver = daemon
        .browse(service_type)
        .with_context(|| format!("failed to browse {service_type}"))?;

    let deadline = Instant::now() + timeout;
    let mut found: BTreeMap<String, DiscoveredService> = BTreeMap::new();

    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let wait = remaining.min(Duration::from_millis(200));
        match receiver.recv_timeout(wait) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                if let Some(service) = service_from_resolved(&info)
                    && matches_filter(&service, filter)
                {
                    found.insert(service.base_url.clone(), service);
                }
            }
            Ok(ServiceEvent::SearchStopped(_)) => break,
            Ok(_) => {}
            Err(_) => {
                // flume timeout or disconnect — keep looping until deadline.
            }
        }
    }

    let _ = daemon.stop_browse(service_type);
    let _ = daemon.shutdown();

    let mut services: Vec<_> = found.into_values().collect();
    services.sort_by(|a, b| a.instance_name.cmp(&b.instance_name));
    Ok(services)
}

pub fn browse_engines(timeout: Duration, filter: &BrowseFilter) -> Result<Vec<DiscoveredService>> {
    let mut filter = filter.clone();
    if filter.kind.is_none() {
        filter.kind = Some(DiscoveredKind::AxEngine);
    }
    browse_services(ENGINE_SERVICE_TYPE, timeout, &filter)
}

pub fn browse_gateways(timeout: Duration, filter: &BrowseFilter) -> Result<Vec<DiscoveredService>> {
    let mut filter = filter.clone();
    if filter.kind.is_none() {
        filter.kind = Some(DiscoveredKind::AxServingGateway);
    }
    browse_services(GATEWAY_SERVICE_TYPE, timeout, &filter)
}

/// Pick a single engine from browse results, erroring on ambiguity.
pub fn select_unique_engine(
    candidates: &[DiscoveredService],
    filter: &BrowseFilter,
) -> Result<DiscoveredService> {
    select_unique(
        candidates,
        filter,
        "AX Engine",
        "start ax-engine-server with --advertise-lan or set AXS_NODE_RUNTIME_URL",
        "AXS_DISCOVER_LAN_INSTANCE / AXS_DISCOVER_LAN_INSTANCE_ID",
    )
}

/// Pick a single gateway from browse results, erroring on ambiguity.
pub fn select_unique_gateway(
    candidates: &[DiscoveredService],
    filter: &BrowseFilter,
) -> Result<DiscoveredService> {
    select_unique(
        candidates,
        filter,
        "AX Serving gateway",
        "start ax-serving-api with --advertise-lan or set AXS_CONTROL_PLANE_URL",
        "AXS_DISCOVER_LAN_GATEWAY_INSTANCE / AXS_DISCOVER_LAN_INSTANCE_ID",
    )
}

fn select_unique(
    candidates: &[DiscoveredService],
    filter: &BrowseFilter,
    role: &str,
    empty_hint: &str,
    disambiguate_hint: &str,
) -> Result<DiscoveredService> {
    let mut matches: Vec<_> = candidates
        .iter()
        .filter(|s| matches_filter(s, filter))
        .cloned()
        .collect();
    matches.sort_by(|a, b| a.instance_name.cmp(&b.instance_name));
    match matches.len() {
        0 => bail!("no {role} services found on the LAN; {empty_hint}"),
        1 => Ok(matches.remove(0)),
        n => {
            let names: Vec<_> = matches
                .iter()
                .map(|s| {
                    format!(
                        "{} ({})",
                        s.instance_name,
                        s.instance_id.as_deref().unwrap_or("?")
                    )
                })
                .collect();
            bail!(
                "found {n} {role} services; disambiguate with {disambiguate_hint}. candidates: {}",
                names.join(", ")
            )
        }
    }
}

/// Pure URL resolution priority used by the runtime agent (unit-testable).
///
/// 1. explicit non-empty URL
/// 2. when `lan_enabled`, unique candidate `base_url`
/// 3. optional default (loopback bring-up)
pub fn resolve_base_url(
    explicit: Option<&str>,
    lan_enabled: bool,
    lan_candidates: &[DiscoveredService],
    filter: &BrowseFilter,
    default_url: Option<&str>,
    role: ResolveRole,
) -> Result<String> {
    if let Some(raw) = explicit.map(str::trim).filter(|v| !v.is_empty()) {
        return Ok(raw.to_string());
    }
    if lan_enabled {
        let selected = match role {
            ResolveRole::Engine => select_unique_engine(lan_candidates, filter)?,
            ResolveRole::Gateway => select_unique_gateway(lan_candidates, filter)?,
        };
        return Ok(selected.base_url);
    }
    if let Some(default_url) = default_url.map(str::trim).filter(|v| !v.is_empty()) {
        return Ok(default_url.to_string());
    }
    bail!("no URL configured for {role:?} and LAN discovery is disabled")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveRole {
    Engine,
    Gateway,
}

/// Whether an IP is acceptable for AX Serving worker advertise / runtime base URL.
pub fn is_fleet_safe_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            !v4.is_unspecified()
                && !v4.is_loopback()
                && !v4.is_multicast()
                && !v4.is_broadcast()
                && !v4.is_link_local()
                && (v4.is_private() || is_shared_v4(v4))
        }
        IpAddr::V6(v6) => {
            !v6.is_unspecified()
                && !v6.is_loopback()
                && !v6.is_multicast()
                && !v6.is_unicast_link_local()
        }
    }
}

/// Prefer private IPv4 when selecting an address from a resolved service.
pub fn prefer_private_v4(addresses: impl IntoIterator<Item = IpAddr>) -> Option<IpAddr> {
    let mut addrs: Vec<IpAddr> = addresses.into_iter().collect();
    addrs.sort_by_key(|ip| match ip {
        IpAddr::V4(v4) if v4.is_private() => 0u8,
        IpAddr::V4(v4) if is_shared_v4(*v4) => 1,
        IpAddr::V4(_) => 2,
        IpAddr::V6(_) => 3,
    });
    addrs.into_iter().find(|ip| match ip {
        IpAddr::V4(v4) => {
            !v4.is_unspecified()
                && !v4.is_multicast()
                && !v4.is_link_local()
                && (v4.is_private() || is_shared_v4(*v4) || v4.is_loopback())
        }
        IpAddr::V6(v6) => !v6.is_unspecified() && !v6.is_multicast() && !v6.is_unicast_link_local(),
    })
}

fn is_shared_v4(ip: Ipv4Addr) -> bool {
    // CGNAT 100.64.0.0/10 — `Ipv4Addr::is_shared` on recent Rust.
    let octets = ip.octets();
    octets[0] == 100 && (octets[1] & 0xc0) == 64
}

fn service_from_resolved(info: &mdns_sd::ServiceInfo) -> Option<DiscoveredService> {
    let port = info.get_port();
    let addresses: Vec<IpAddr> = info.get_addresses().iter().copied().collect();
    let ip = prefer_private_v4(addresses)?;
    // Accept private/shared unicast and loopback (same-host lab). Reject
    // link-local / multicast — fleet worker advertise_url forbids link-local.
    let acceptable = match ip {
        IpAddr::V4(v4) if v4.is_loopback() => true,
        other => is_fleet_safe_ip(other),
    };
    if !acceptable {
        return None;
    }

    let txt_props = info.get_properties();
    let mut txt = BTreeMap::new();
    for prop in txt_props.iter() {
        let key = prop.key().to_string();
        txt.insert(key, prop.val_str().to_string());
    }

    let scheme = txt
        .get("scheme")
        .map(String::as_str)
        .unwrap_or("http")
        .to_string();
    let host_for_url = match ip {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    };
    let base_url = format!("{scheme}://{host_for_url}:{port}");
    let kind = DiscoveredKind::from_txt(txt.get("kind").map(String::as_str));

    // Fullname looks like `Name._ax-engine._tcp.local.`; use the instance label.
    let fullname = info.get_fullname().to_string();
    let instance_name = fullname
        .split('.')
        .next()
        .unwrap_or(info.get_hostname())
        .trim_end_matches('.')
        .to_string();

    Some(DiscoveredService {
        kind,
        instance_name,
        host: host_for_url,
        port,
        base_url,
        version: txt.get("version").cloned(),
        model_id: txt.get("model").cloned(),
        auth: txt.get("auth").cloned(),
        cluster: txt.get("cluster").cloned(),
        instance_id: txt.get("instance").cloned(),
        platform: txt.get("platform").cloned(),
        txt,
    })
}

/// Public for unit tests that build synthetic candidate lists.
pub fn matches_filter(service: &DiscoveredService, filter: &BrowseFilter) -> bool {
    if let Some(kind) = filter.kind
        && service.kind != DiscoveredKind::Unknown
        && service.kind != kind
    {
        return false;
    }
    if let Some(cluster) = filter.cluster.as_deref() {
        match service.cluster.as_deref() {
            Some(value) if value == cluster => {}
            _ => return false,
        }
    }
    if let Some(name) = filter.instance_name.as_deref() {
        let hay = service.instance_name.to_ascii_lowercase();
        if !hay.contains(&name.to_ascii_lowercase()) {
            return false;
        }
    }
    if let Some(id) = filter.instance_id.as_deref() {
        match service.instance_id.as_deref() {
            Some(value) if value == id => {}
            _ => return false,
        }
    }
    true
}

/// Build a filter from AX Serving agent environment variables (engine runtime).
pub fn filter_from_env() -> BrowseFilter {
    filter_from_env_for(DiscoveredKind::AxEngine)
}

/// Gateway-specific instance name env: `AXS_DISCOVER_LAN_GATEWAY_INSTANCE` falls
/// back to the shared instance name keys.
pub fn filter_from_env_for(kind: DiscoveredKind) -> BrowseFilter {
    let instance_name = match kind {
        DiscoveredKind::AxServingGateway => std::env::var("AXS_DISCOVER_LAN_GATEWAY_INSTANCE")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .or_else(|| {
                std::env::var("AXS_DISCOVER_LAN_INSTANCE")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            }),
        _ => std::env::var("AXS_DISCOVER_LAN_INSTANCE")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
    };
    BrowseFilter {
        cluster: std::env::var("AXS_DISCOVER_LAN_CLUSTER")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
        instance_name,
        instance_id: std::env::var("AXS_DISCOVER_LAN_INSTANCE_ID")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
        kind: Some(kind),
    }
}

pub fn discover_lan_enabled() -> bool {
    matches!(
        std::env::var("AXS_DISCOVER_LAN")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

pub fn discover_timeout_from_env() -> Duration {
    std::env::var("AXS_DISCOVER_LAN_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(3))
        .max(Duration::from_millis(200))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(kind: DiscoveredKind, name: &str, url: &str, id: &str) -> DiscoveredService {
        DiscoveredService {
            kind,
            instance_name: name.into(),
            host: "10.0.0.1".into(),
            port: 8080,
            base_url: url.into(),
            version: Some("1.0".into()),
            model_id: None,
            auth: Some("required".into()),
            cluster: Some("lab".into()),
            instance_id: Some(id.into()),
            platform: None,
            txt: BTreeMap::new(),
        }
    }

    #[test]
    fn kind_parse() {
        assert_eq!(
            DiscoveredKind::from_txt(Some("ax_engine")),
            DiscoveredKind::AxEngine
        );
        assert_eq!(
            DiscoveredKind::from_txt(Some("ax-serving-gateway")),
            DiscoveredKind::AxServingGateway
        );
    }

    #[test]
    fn select_unique_errors_on_many() {
        let a = sample(DiscoveredKind::AxEngine, "a", "http://10.0.0.1:8080", "1");
        let b = sample(DiscoveredKind::AxEngine, "b", "http://10.0.0.2:8080", "2");
        let err = select_unique_engine(&[a, b], &BrowseFilter::default()).unwrap_err();
        assert!(err.to_string().contains("disambiguate"));
    }

    #[test]
    fn select_unique_gateway_errors_when_empty() {
        let err = select_unique_gateway(&[], &BrowseFilter::default()).unwrap_err();
        assert!(err.to_string().contains("AX Serving gateway"));
    }

    #[test]
    fn resolve_base_url_prefers_explicit() {
        let candidates = [sample(
            DiscoveredKind::AxEngine,
            "a",
            "http://10.0.0.1:8080",
            "1",
        )];
        let url = resolve_base_url(
            Some("http://explicit:9"),
            true,
            &candidates,
            &BrowseFilter::default(),
            Some("http://127.0.0.1:1"),
            ResolveRole::Engine,
        )
        .unwrap();
        assert_eq!(url, "http://explicit:9");
    }

    #[test]
    fn resolve_base_url_uses_lan_when_enabled() {
        let candidates = [sample(
            DiscoveredKind::AxEngine,
            "a",
            "http://10.0.0.1:8080",
            "1",
        )];
        let url = resolve_base_url(
            None,
            true,
            &candidates,
            &BrowseFilter::default(),
            Some("http://127.0.0.1:1"),
            ResolveRole::Engine,
        )
        .unwrap();
        assert_eq!(url, "http://10.0.0.1:8080");
    }

    #[test]
    fn resolve_base_url_falls_back_to_default() {
        let url = resolve_base_url(
            None,
            false,
            &[],
            &BrowseFilter::default(),
            Some("http://127.0.0.1:8000"),
            ResolveRole::Engine,
        )
        .unwrap();
        assert_eq!(url, "http://127.0.0.1:8000");
    }

    #[test]
    fn resolve_base_url_lan_multi_candidate_fail_closed() {
        let candidates = [
            sample(
                DiscoveredKind::AxServingGateway,
                "g1",
                "http://10.0.0.1:19090",
                "1",
            ),
            sample(
                DiscoveredKind::AxServingGateway,
                "g2",
                "http://10.0.0.2:19090",
                "2",
            ),
        ];
        let err = resolve_base_url(
            None,
            true,
            &candidates,
            &BrowseFilter::default(),
            None,
            ResolveRole::Gateway,
        )
        .unwrap_err();
        assert!(err.to_string().contains("disambiguate"));
    }

    #[test]
    fn cluster_filter_excludes_mismatch() {
        let svc = sample(DiscoveredKind::AxEngine, "a", "http://10.0.0.1:8080", "1");
        let filter = BrowseFilter {
            cluster: Some("other".into()),
            ..Default::default()
        };
        assert!(!matches_filter(&svc, &filter));
    }

    #[test]
    fn prefer_private_v4_orders() {
        let ip = prefer_private_v4([
            IpAddr::V4(Ipv4Addr::new(169, 254, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)),
        ]);
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5))));
    }
}
