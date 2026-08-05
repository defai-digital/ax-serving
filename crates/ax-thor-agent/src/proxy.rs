//! Compatibility wrapper around the shared adapter proxy.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize},
};

use ax_serving_adapter_core::proxy::{self, ProxyConfig};

use crate::config::ThorConfig;

pub use ax_serving_adapter_core::proxy::ProxyState;

pub fn router(
    config: &ThorConfig,
    client: reqwest::Client,
    inflight: Arc<AtomicUsize>,
    draining: Arc<AtomicBool>,
) -> axum::Router {
    proxy::router(
        ProxyConfig {
            upstream_url: config.runtime_url.clone(),
            upstream_health_path: config.runtime_health_path.clone(),
            dispatch_token: config.dispatch_token.clone(),
            max_inflight: config.max_inflight,
            expected_domain_id: None,
            require_dispatch_identity: config.dispatch_token.is_some(),
        },
        client,
        inflight,
        draining,
        None,
    )
}
