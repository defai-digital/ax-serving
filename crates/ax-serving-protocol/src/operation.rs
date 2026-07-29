use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MAX_WIRE_NAME_BYTES: usize = 96;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireNameError {
    #[error("wire name must not be empty")]
    Empty,
    #[error("wire name exceeds {MAX_WIRE_NAME_BYTES} bytes")]
    TooLong,
    #[error("wire name must use lowercase ASCII letters, digits, '.', '_' or '-'")]
    InvalidCharacter,
}

fn validate_wire_name(value: &str) -> Result<(), WireNameError> {
    if value.is_empty() {
        return Err(WireNameError::Empty);
    }
    if value.len() > MAX_WIRE_NAME_BYTES {
        return Err(WireNameError::TooLong);
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
    }) {
        return Err(WireNameError::InvalidCharacter);
    }
    Ok(())
}

macro_rules! wire_name {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, WireNameError> {
                let value = value.into();
                validate_wire_name(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = WireNameError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

wire_name!(Operation);
wire_name!(ProtocolCapability);

impl Operation {
    pub const CHAT_COMPLETIONS: &'static str = "chat_completions";
    pub const TEXT_COMPLETIONS: &'static str = "text_completions";
    pub const EMBEDDINGS: &'static str = "embeddings";
    pub const RESPONSES: &'static str = "responses";

    pub fn chat_completions() -> Self {
        Self(Self::CHAT_COMPLETIONS.to_string())
    }

    pub fn text_completions() -> Self {
        Self(Self::TEXT_COMPLETIONS.to_string())
    }

    pub fn embeddings() -> Self {
        Self(Self::EMBEDDINGS.to_string())
    }
}

impl ProtocolCapability {
    pub const CONTROL_DRAIN: &'static str = "control.drain";
    pub const CONTROL_DEPLOYMENT_JOBS: &'static str = "control.deployment-jobs";
    pub const CONTROL_EXECUTION_DOMAIN: &'static str = "control.execution-domain.v1";
    pub const CONTROL_MAC_CLUSTER: &'static str = "control.mac-cluster.v1";
    pub const CONTROL_INVENTORY_DELTA: &'static str = "control.inventory-delta";
    pub const DISPATCH_CANCEL: &'static str = "dispatch.cancel";
    pub const DISPATCH_TYPED_ADMISSION: &'static str = "dispatch.typed-admission";
    pub const TELEMETRY_CAPACITY: &'static str = "telemetry.capacity";
    pub const TELEMETRY_DOMAIN_CAPACITY: &'static str = "telemetry.domain-capacity.v1";
    pub const TELEMETRY_KV_CACHE: &'static str = "telemetry.kv-cache";
    pub const TELEMETRY_PREFIX_CACHE: &'static str = "telemetry.prefix-cache";
}

#[cfg(test)]
mod tests {
    use super::{Operation, ProtocolCapability};

    #[test]
    fn unknown_future_names_round_trip() {
        let capability = ProtocolCapability::new("inference.future-mode").unwrap();
        let encoded = serde_json::to_string(&capability).unwrap();
        assert_eq!(
            serde_json::from_str::<ProtocolCapability>(&encoded).unwrap(),
            capability
        );
    }

    #[test]
    fn wire_names_reject_unstable_casing() {
        assert!(Operation::new("ChatCompletions").is_err());
        assert!(ProtocolCapability::new("control/drain").is_err());
    }
}
