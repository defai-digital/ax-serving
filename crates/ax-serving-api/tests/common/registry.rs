//! Worker registration request builders for orchestration integration tests.

use std::net::SocketAddr;

use ax_serving_api::orchestration::registry::{RegisterCapabilities, RegisterRequest};

pub fn reg_req(addr: SocketAddr, caps: &[&str]) -> RegisterRequest {
    RegisterRequest {
        worker_id: None,
        addr: addr.to_string(),
        capabilities: RegisterCapabilities::Legacy(caps.iter().map(|s| s.to_string()).collect()),
        backend: "native".into(),
        max_inflight: 8,
        friendly_name: None,
        chip_model: None,
        worker_pool: None,
        node_class: None,
        ..Default::default()
    }
}

pub fn reg_req_with_pool(
    addr: SocketAddr,
    caps: &[&str],
    worker_pool: Option<&str>,
    node_class: Option<&str>,
) -> RegisterRequest {
    let mut req = reg_req(addr, caps);
    req.worker_pool = worker_pool.map(str::to_string);
    req.node_class = node_class.map(str::to_string);
    req
}
