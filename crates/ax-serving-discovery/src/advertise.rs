//! Opt-in mDNS / DNS-SD advertisement helpers for AX Serving gateway (and tests).

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};

use anyhow::{Context, Result, bail};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use tracing::{info, warn};

use crate::{DISCOVERY_PROTO, GATEWAY_SERVICE_TYPE};

#[derive(Debug, Clone)]
pub struct AdvertiseConfig {
    pub service_type: String,
    pub kind: String,
    pub instance_name: String,
    pub port: u16,
    pub advertise_ip: Ipv4Addr,
    pub version: String,
    pub auth_required: bool,
    pub cluster: Option<String>,
    pub instance_id: String,
    /// Optional model / role label (engine uses model id; gateway may omit).
    pub model: Option<String>,
}

pub struct LanAdvertiser {
    daemon: ServiceDaemon,
    fullname: String,
}

impl LanAdvertiser {
    pub fn start(config: AdvertiseConfig) -> Result<Self> {
        let daemon = ServiceDaemon::new().context("failed to start mDNS daemon")?;

        let mut properties: HashMap<String, String> = HashMap::new();
        properties.insert("proto".into(), DISCOVERY_PROTO.into());
        properties.insert("kind".into(), config.kind);
        properties.insert("version".into(), config.version);
        properties.insert(
            "auth".into(),
            if config.auth_required {
                "required".into()
            } else {
                "open".into()
            },
        );
        properties.insert("scheme".into(), "http".into());
        properties.insert("path".into(), "/v1".into());
        properties.insert("instance".into(), config.instance_id.clone());
        properties.insert("platform".into(), current_platform());
        if let Some(model) = config.model.filter(|m| !m.is_empty()) {
            properties.insert("model".into(), model);
        }
        if let Some(cluster) = config.cluster.filter(|c| !c.is_empty()) {
            properties.insert("cluster".into(), cluster);
        }

        let sanitized = sanitize_instance_name(&config.instance_name);
        let host_name = format!("{sanitized}.local.");
        let service = ServiceInfo::new(
            &config.service_type,
            &sanitized,
            &host_name,
            IpAddr::V4(config.advertise_ip),
            config.port,
            Some(properties),
        )
        .context("failed to build mDNS ServiceInfo")?;

        let fullname = service.get_fullname().to_string();
        daemon
            .register(service)
            .context("failed to register mDNS service")?;

        info!(
            service = %fullname,
            advertise_ip = %config.advertise_ip,
            port = config.port,
            "LAN mDNS advertisement registered"
        );

        Ok(Self { daemon, fullname })
    }

    pub fn start_gateway(
        instance_name: &str,
        port: u16,
        advertise_ip: Ipv4Addr,
        version: &str,
        auth_required: bool,
        cluster: Option<String>,
        instance_id: String,
    ) -> Result<Self> {
        Self::start(AdvertiseConfig {
            service_type: GATEWAY_SERVICE_TYPE.into(),
            kind: "ax_serving_gateway".into(),
            instance_name: instance_name.into(),
            port,
            advertise_ip,
            version: version.into(),
            auth_required,
            cluster,
            instance_id,
            model: None,
        })
    }
}

impl Drop for LanAdvertiser {
    fn drop(&mut self) {
        if let Err(err) = self.daemon.unregister(&self.fullname) {
            warn!(error = %err, fullname = %self.fullname, "failed to unregister mDNS service");
        }
        let _ = self.daemon.shutdown();
    }
}

pub fn sanitize_instance_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '.' {
            out.push('-');
        }
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "ax-service".into()
    } else {
        out.chars().take(63).collect()
    }
}

pub fn current_platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

pub fn is_advertisable_v4(ip: Ipv4Addr) -> bool {
    !ip.is_unspecified()
        && !ip.is_loopback()
        && !ip.is_multicast()
        && !ip.is_broadcast()
        && !ip.is_link_local()
        && (ip.is_private() || is_cgnat_v4(ip))
}

fn is_cgnat_v4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 100 && (octets[1] & 0xc0) == 64
}

/// Prefer a stable private IPv4 for advertisement.
pub fn pick_advertise_ipv4(explicit: Option<&str>, bind_host: &str) -> Result<Ipv4Addr> {
    if let Some(raw) = explicit {
        let ip: IpAddr = raw
            .parse()
            .with_context(|| format!("invalid lan advertise host {raw}"))?;
        match ip {
            IpAddr::V4(v4) if is_advertisable_v4(v4) => return Ok(v4),
            IpAddr::V4(v4) => bail!(
                "lan advertise host {v4} is not a private unicast IPv4 suitable for fleet join"
            ),
            IpAddr::V6(_) => bail!("lan advertise host must be IPv4 in phase 1"),
        }
    }

    if let Ok(ip) = bind_host.parse::<Ipv4Addr>()
        && is_advertisable_v4(ip)
    {
        return Ok(ip);
    }

    if let Some(ip) = guess_private_ipv4() {
        return Ok(ip);
    }

    bail!("could not determine a private IPv4 for LAN advertise; set --lan-advertise-host")
}

fn guess_private_ipv4() -> Option<Ipv4Addr> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("1.1.1.1:80").ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(v4) if is_advertisable_v4(v4) => Some(v4),
        _ => None,
    }
}

pub fn env_truthy(key: &str) -> bool {
    matches!(
        std::env::var(key)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_collapses_junk() {
        assert_eq!(sanitize_instance_name("GW #1 lab"), "GW-1-lab");
        assert_eq!(sanitize_instance_name("@@@"), "ax-service");
    }

    #[test]
    fn private_ipv4_accepted() {
        assert!(is_advertisable_v4(Ipv4Addr::new(192, 168, 1, 10)));
        assert!(!is_advertisable_v4(Ipv4Addr::LOCALHOST));
        assert!(!is_advertisable_v4(Ipv4Addr::new(169, 254, 1, 1)));
    }

    #[test]
    fn explicit_advertise_host_parses() {
        let ip = pick_advertise_ipv4(Some("10.0.0.5"), "127.0.0.1").unwrap();
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 5));
    }
}
