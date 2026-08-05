//! Validated worker advertise endpoints (HTTP/HTTPS base URLs).
//!
//! Dispatch preserves the scheme and host (IP or DNS). Active TCP probes are
//! only performed when the host is a concrete IP address.

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Canonical base URL for a runtime agent: `http(s)://host:port` with no path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorkerEndpoint {
    base_url: String,
}

impl WorkerEndpoint {
    /// Parse a full advertise URL or a legacy `host:port` / `ip:port` value.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("advertise endpoint is empty".into());
        }
        let candidate = if trimmed.contains("://") {
            trimmed.to_string()
        } else {
            format!("http://{trimmed}")
        };
        let url = reqwest::Url::parse(&candidate)
            .map_err(|error| format!("invalid advertise endpoint: {error}"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err("advertise endpoint must use http or https".into());
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err("advertise endpoint must not contain credentials".into());
        }
        if !matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some() {
            return Err("advertise endpoint must not contain a path, query, or fragment".into());
        }
        let host = url
            .host_str()
            .ok_or_else(|| "advertise endpoint is missing a host".to_string())?;
        if host.is_empty() || host == "*" {
            return Err("advertise endpoint host is invalid".into());
        }
        // `Url::host_str()` keeps the brackets on IPv6 literals ("[::1]").
        // Strip them so the literal validates as an IP address below instead
        // of falling into the DNS-name check and being rejected; the brackets
        // are re-added when the base URL is built.
        let host = host
            .strip_prefix('[')
            .and_then(|inner| inner.strip_suffix(']'))
            .unwrap_or(host);
        if host.is_empty() {
            return Err("advertise endpoint host is invalid".into());
        }
        // Reject wildcard IPs when the host is an IP literal.
        if let Ok(ip) = host.parse::<IpAddr>() {
            if ip.is_unspecified() || ip.is_multicast() || is_link_local(ip) {
                return Err("advertise endpoint uses a disallowed destination address".into());
            }
        } else {
            // DNS hostnames only (IPv6 literals are handled above via IpAddr).
            // Labels: alphanumeric and hyphen; dots separate labels; no underscores.
            if host.contains(' ')
                || host.starts_with('.')
                || host.ends_with('.')
                || host.contains("..")
                || !host
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
            {
                return Err("advertise endpoint host is not a valid DNS name or IP".into());
            }
        }
        let port = url
            .port_or_known_default()
            .ok_or_else(|| "advertise endpoint is missing a port".to_string())?;
        // Url may omit default ports; always store an explicit port for stable dispatch.
        // IPv6 literals must be bracketed to keep the stored `base_url` a valid URL.
        let host_for_url = if let Ok(ip) = host.parse::<std::net::Ipv6Addr>() {
            format!("[{ip}]")
        } else {
            host.to_string()
        };
        let base_url = format!("{}://{}:{}", url.scheme(), host_for_url, port);
        Ok(Self { base_url })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn join_path(&self, path: &str) -> String {
        if path.starts_with('/') {
            format!("{}{path}", self.base_url)
        } else {
            format!("{}/{path}", self.base_url)
        }
    }

    /// TCP probe address when the host is an IP literal; DNS hosts skip TCP probes.
    pub fn tcp_probe_addr(&self) -> Option<SocketAddr> {
        let url = reqwest::Url::parse(&self.base_url).ok()?;
        let host = url.host_str()?;
        // `Url::host_str()` keeps brackets on IPv6 literals — strip before parsing.
        let host = host
            .strip_prefix('[')
            .and_then(|inner| inner.strip_suffix(']'))
            .unwrap_or(host);
        let ip = host.parse::<IpAddr>().ok()?;
        let port = url.port_or_known_default()?;
        Some(SocketAddr::new(ip, port))
    }

    pub fn is_loopback(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.base_url) else {
            return false;
        };
        let Some(host) = url.host_str() else {
            return false;
        };
        // `Url::host_str()` keeps brackets on IPv6 literals — strip before parsing.
        let host = host
            .strip_prefix('[')
            .and_then(|inner| inner.strip_suffix(']'))
            .unwrap_or(host);
        match host.parse::<IpAddr>() {
            Ok(ip) => ip.is_loopback(),
            Err(_) => host.eq_ignore_ascii_case("localhost"),
        }
    }

    pub fn host_port_display(&self) -> String {
        let Ok(url) = reqwest::Url::parse(&self.base_url) else {
            return self.base_url.clone();
        };
        let host = url.host_str().unwrap_or("?");
        let port = url.port_or_known_default().unwrap_or(0);
        if host.contains(':') && !host.starts_with('[') {
            format!("[{host}]:{port}")
        } else if let Ok(ip) = host.parse::<std::net::Ipv6Addr>() {
            format!("[{ip}]:{port}")
        } else {
            format!("{host}:{port}")
        }
    }
}

impl fmt::Display for WorkerEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.base_url)
    }
}

impl FromStr for WorkerEndpoint {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for WorkerEndpoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.base_url)
    }
}

impl<'de> Deserialize<'de> for WorkerEndpoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

fn is_link_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_link_local(),
        IpAddr::V6(ip) => ip.is_unicast_link_local(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dns_https_url() {
        let ep = WorkerEndpoint::parse("https://ax-runtime-agent.runtime.svc.cluster.local:18443")
            .unwrap();
        assert_eq!(
            ep.base_url(),
            "https://ax-runtime-agent.runtime.svc.cluster.local:18443"
        );
        assert!(ep.tcp_probe_addr().is_none());
        assert_eq!(
            ep.join_path("/v1/chat/completions"),
            "https://ax-runtime-agent.runtime.svc.cluster.local:18443/v1/chat/completions"
        );
    }

    #[test]
    fn parses_legacy_socket_addr() {
        let ep = WorkerEndpoint::parse("10.20.30.40:18081").unwrap();
        assert_eq!(ep.base_url(), "http://10.20.30.40:18081");
        assert_eq!(
            ep.tcp_probe_addr(),
            Some("10.20.30.40:18081".parse().unwrap())
        );
    }

    #[test]
    fn ipv6_host_is_rebracketed_in_base_url() {
        // Regression: `Url::host_str()` strips brackets from IPv6 literals, so
        // the stored base_url became `http://::1:18081` — an invalid URL that
        // broke dispatch, TCP probing, and serde roundtrips for IPv6 workers.
        let ep = WorkerEndpoint::parse("http://[2001:db8::5]:18081").unwrap();
        assert_eq!(ep.base_url(), "http://[2001:db8::5]:18081");
        assert_eq!(
            ep.tcp_probe_addr(),
            Some("[2001:db8::5]:18081".parse().unwrap())
        );
        assert_eq!(
            ep.join_path("/v1/chat/completions"),
            "http://[2001:db8::5]:18081/v1/chat/completions"
        );

        // Legacy bracketed host:port form.
        let ep = WorkerEndpoint::parse("[::1]:18081").unwrap();
        assert_eq!(ep.base_url(), "http://[::1]:18081");
        assert_eq!(ep.tcp_probe_addr(), Some("[::1]:18081".parse().unwrap()));
        assert_eq!(ep.host_port_display(), "[::1]:18081");

        // Serde roundtrip re-parses the stored base_url — must succeed.
        let json = serde_json::to_string(&ep).unwrap();
        let back: WorkerEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ep);

        // Wildcard IPv6 is still rejected after bracket stripping.
        assert!(WorkerEndpoint::parse("http://[::]:18081").is_err());
    }

    #[test]
    fn rejects_wildcard_and_path() {
        assert!(WorkerEndpoint::parse("http://0.0.0.0:18081").is_err());
        assert!(WorkerEndpoint::parse("http://10.0.0.1:18081/v1").is_err());
        assert!(WorkerEndpoint::parse("http://user:pass@10.0.0.1:18081").is_err());
        assert!(WorkerEndpoint::parse("http://169.254.1.1:18081").is_err());
        assert!(WorkerEndpoint::parse("http://not a host:18081").is_err());
        assert!(WorkerEndpoint::parse("http://bad_host:18081").is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let ep = WorkerEndpoint::parse("http://agent.example:18081").unwrap();
        let json = serde_json::to_string(&ep).unwrap();
        assert_eq!(json, "\"http://agent.example:18081\"");
        let back: WorkerEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ep);
    }
}
