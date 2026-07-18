use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::ProtocolCapability;

pub const CURRENT_PROTOCOL: ProtocolVersion = ProtocolVersion { major: 1, minor: 1 };

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolDescriptor {
    pub version: ProtocolVersion,
    #[serde(default)]
    pub capabilities: BTreeSet<ProtocolCapability>,
}

impl ProtocolDescriptor {
    pub fn current(capabilities: impl IntoIterator<Item = ProtocolCapability>) -> Self {
        Self {
            version: CURRENT_PROTOCOL,
            capabilities: capabilities.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiatedProtocol {
    pub version: ProtocolVersion,
    pub capabilities: BTreeSet<ProtocolCapability>,
}

impl From<NegotiatedProtocol> for ProtocolDescriptor {
    fn from(value: NegotiatedProtocol) -> Self {
        Self {
            version: value.version,
            capabilities: value.capabilities,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    #[error("unsupported protocol major {offered}; supported major is {supported}")]
    IncompatibleMajor { offered: u16, supported: u16 },
    #[error("protocol minor {offered} predates required minor {minimum}")]
    MinorTooOld { offered: u16, minimum: u16 },
}

pub fn negotiate_protocol(
    peer: &ProtocolDescriptor,
    supported_major: u16,
    minimum_minor: u16,
    maximum_minor: u16,
    local_capabilities: &BTreeSet<ProtocolCapability>,
) -> Result<NegotiatedProtocol, ProtocolError> {
    if peer.version.major != supported_major {
        return Err(ProtocolError::IncompatibleMajor {
            offered: peer.version.major,
            supported: supported_major,
        });
    }
    if peer.version.minor < minimum_minor {
        return Err(ProtocolError::MinorTooOld {
            offered: peer.version.minor,
            minimum: minimum_minor,
        });
    }

    Ok(NegotiatedProtocol {
        version: ProtocolVersion {
            major: supported_major,
            minor: peer.version.minor.min(maximum_minor),
        },
        capabilities: peer
            .capabilities
            .intersection(local_capabilities)
            .cloned()
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{ProtocolDescriptor, ProtocolError, ProtocolVersion, negotiate_protocol};
    use crate::ProtocolCapability;

    fn capability(value: &str) -> ProtocolCapability {
        ProtocolCapability::new(value).unwrap()
    }

    #[test]
    fn negotiation_selects_lower_minor_and_capability_intersection() {
        let peer = ProtocolDescriptor {
            version: ProtocolVersion { major: 1, minor: 3 },
            capabilities: BTreeSet::from([
                capability("control.drain"),
                capability("future.feature"),
            ]),
        };
        let local = BTreeSet::from([
            capability("control.drain"),
            capability("telemetry.capacity"),
        ]);

        let negotiated = negotiate_protocol(&peer, 1, 0, 1, &local).unwrap();
        assert_eq!(negotiated.version.minor, 1);
        assert_eq!(
            negotiated.capabilities,
            BTreeSet::from([capability("control.drain")])
        );
    }

    #[test]
    fn negotiation_rejects_incompatible_major() {
        let peer = ProtocolDescriptor {
            version: ProtocolVersion { major: 2, minor: 0 },
            capabilities: BTreeSet::new(),
        };
        assert!(matches!(
            negotiate_protocol(&peer, 1, 0, 0, &BTreeSet::new()),
            Err(ProtocolError::IncompatibleMajor { .. })
        ));
    }
}
