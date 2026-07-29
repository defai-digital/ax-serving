//! Build-time license identity exposed by diagnostics and the dashboard.

use std::sync::Arc;

/// SPDX identifier for AX Serving.
pub const SPDX_LICENSE: &str = "Apache-2.0";

/// Immutable license metadata shared by serving modes.
#[derive(Debug, Default)]
pub struct LicenseState;

impl LicenseState {
    /// Create immutable Apache-2.0 license metadata.
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }

    /// Serialize the public license identity.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "license": SPDX_LICENSE,
            "name": "Apache License 2.0",
            "notice": "NOTICE",
            "source": env!("CARGO_PKG_REPOSITORY"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_apache_license_identity() {
        let json = LicenseState::new().to_json();
        assert_eq!(json["license"], "Apache-2.0");
        assert_eq!(json["name"], "Apache License 2.0");
        assert_eq!(json["notice"], "NOTICE");
    }
}
