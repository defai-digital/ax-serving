# BUG-030: IPv6 URL Formatting Missing Brackets in `local_probe_base`

**Severity:** Medium
**File:** `crates/ax-serving-cli/src/thor.rs`
**Lines:** 455–462
**Status:** ✅ FIXED (2026-03-29)

## Description

When the Thor agent listen address is an IPv6 address, `local_probe_base` formats it as `http://::1:18081` (no brackets). Per RFC 2732, IPv6 addresses in URLs must be bracketed: `http://[::1]:18081`:

```rust
fn local_probe_base(listen_addr: SocketAddr) -> String {
    let ip = match listen_addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    format!("http://{}:{}", ip, listen_addr.port()) // no brackets for IPv6
}
```

## Impact

`reqwest` uses the `url` crate internally, which will fail to parse an unbracketed IPv6 literal. Any `thor status`, `thor drain`, or `thor wait-ready` command that needs to probe a Thor agent listening on an IPv6 address will fail with a URL parse error.

## Fix

```rust
match ip {
    IpAddr::V6(_) => format!("http://[{ip}]:{}", listen_addr.port()),
    IpAddr::V4(_) => format!("http://{ip}:{}", listen_addr.port()),
}
```

## Fix Applied
Replaced `format!("http://{}:{}", ip, port)` with a match on `IpAddr`: v6 uses `[{v6}]` brackets, v4 formats normally.
