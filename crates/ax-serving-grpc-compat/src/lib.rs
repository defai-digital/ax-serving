//! Generated gRPC compatibility protocol for the embedded AX Engine mode.
//!
//! This crate deliberately owns the `prost` and `tonic-build` dependency boundary so the
//! portable gateway does not compile protobuf tooling or the legacy embedded service.

/// Generated `ax.serving.v1` protobuf types and server traits.
pub mod proto {
    tonic::include_proto!("ax.serving.v1");
}
