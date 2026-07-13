use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

const MAX_STRING_ID_BYTES: usize = 128;
const MAX_MODEL_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("identifier must not be empty")]
    Empty,
    #[error("identifier exceeds {max} bytes")]
    TooLong { max: usize },
    #[error("identifier contains an invalid character at byte {index}")]
    InvalidCharacter { index: usize },
    #[error("invalid UUID: {0}")]
    InvalidUuid(#[from] uuid::Error),
}

fn validate_string_id(value: &str) -> Result<(), IdError> {
    if value.is_empty() {
        return Err(IdError::Empty);
    }
    if value.len() > MAX_STRING_ID_BYTES {
        return Err(IdError::TooLong {
            max: MAX_STRING_ID_BYTES,
        });
    }
    for (index, byte) in value.bytes().enumerate() {
        if !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')) {
            return Err(IdError::InvalidCharacter { index });
        }
    }
    Ok(())
}

fn validate_model_id(value: &str) -> Result<(), IdError> {
    if value.is_empty() {
        return Err(IdError::Empty);
    }
    if value.len() > MAX_MODEL_ID_BYTES {
        return Err(IdError::TooLong {
            max: MAX_MODEL_ID_BYTES,
        });
    }
    for (index, byte) in value.bytes().enumerate() {
        if !(byte.is_ascii_alphanumeric()
            || matches!(byte, b'.' | b'_' | b':' | b'-' | b'/' | b'@' | b'+'))
        {
            return Err(IdError::InvalidCharacter { index });
        }
    }
    if value.starts_with('/')
        || value.ends_with('/')
        || value
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(IdError::InvalidCharacter {
            index: value.find('/').unwrap_or(0),
        });
    }
    Ok(())
}

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, IdError> {
                let value = value.into();
                validate_string_id(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdError;

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

macro_rules! model_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, IdError> {
                let value = value.into();
                validate_model_id(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdError;

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

string_id!(WorkerId);
string_id!(PoolId);
model_id!(RuntimeModelId);
model_id!(LogicalModelId);
string_id!(DeploymentId);
string_id!(EquivalenceClassId);
string_id!(TrustDomainId);
string_id!(TenantId);

macro_rules! uuid_id {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            pub const fn from_uuid(value: Uuid) -> Self {
                Self(value)
            }

            pub const fn as_uuid(self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl FromStr for $name {
            type Err = IdError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Ok(Self(Uuid::parse_str(value)?))
            }
        }
    };
}

uuid_id!(WorkerInstanceId);
uuid_id!(RegistrationId);
uuid_id!(RequestId);
uuid_id!(AttemptId);
uuid_id!(JobId);

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{IdError, RequestId, RuntimeModelId, WorkerId};

    #[test]
    fn string_ids_accept_contract_alphabet() {
        let id = WorkerId::new("mac.pool_1:worker-07").unwrap();
        assert_eq!(id.as_str(), "mac.pool_1:worker-07");
    }

    #[test]
    fn infrastructure_ids_reject_whitespace_and_paths() {
        assert!(matches!(
            WorkerId::new("worker 1"),
            Err(IdError::InvalidCharacter { .. })
        ));
        assert!(WorkerId::new("../../worker").is_err());
    }

    #[test]
    fn model_ids_accept_repository_names_but_reject_traversal() {
        let id = RuntimeModelId::new("Qwen/Qwen3-32B@main").unwrap();
        assert_eq!(id.as_str(), "Qwen/Qwen3-32B@main");
        assert!(RuntimeModelId::new("../Qwen3").is_err());
        assert!(RuntimeModelId::new("Qwen//Qwen3").is_err());
        assert!(RuntimeModelId::new("/Qwen3").is_err());
    }

    #[test]
    fn deserialization_validates_string_ids() {
        let err = serde_json::from_str::<WorkerId>(r#""bad/id""#).unwrap_err();
        assert!(err.to_string().contains("invalid character"));
    }

    #[test]
    fn uuid_ids_round_trip() {
        let id = RequestId::new();
        let parsed = RequestId::from_str(&id.to_string()).unwrap();
        assert_eq!(parsed, id);
    }
}
