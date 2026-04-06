//! gRPC server: tonic AxServingService over Unix Domain Socket (and optionally TCP).
//!
//! Both a UDS listener at `AXS_GRPC_SOCKET` (default `/tmp/ax-serving.sock`)
//! and an optional TCP listener at `AXS_GRPC_PORT` (e.g. `50051`) run concurrently
//! in the same tonic server.
//!
//! # Auth
//!
//! The UDS path is protected by filesystem permissions — no token check needed.
//! The TCP path requires `Authorization: Bearer <key>` when `AXS_API_KEY` is set,
//! matching the REST layer's auth policy.

pub mod proto {
    tonic::include_proto!("ax.serving.v1");
}

pub mod service;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use proto::ax_serving_service_server::AxServingServiceServer;
use service::AxServingService;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

use crate::ServingLayer;

/// Start the gRPC server on the given UDS socket path, and optionally on TCP.
///
/// The UDS listener has no token check (filesystem permissions are the gate).
/// The TCP listener enforces `Authorization: Bearer <key>` when `keys` is non-empty,
/// consistent with the REST API's auth policy.
///
/// Removes any stale socket file before binding.
pub async fn serve(
    layer: Arc<ServingLayer>,
    socket_path: String,
    tcp_host: String,
    tcp_port: Option<u16>,
    keys: Arc<HashSet<String>>,
) -> Result<()> {
    // Remove stale socket file so bind succeeds on restart.
    let _ = tokio::fs::remove_file(&socket_path).await;

    let uds = tokio::net::UnixListener::bind(&socket_path)?;
    let uds_stream = UnixListenerStream::new(uds);

    // UDS: no auth — protected by filesystem permissions.
    let uds_svc = AxServingServiceServer::new(AxServingService::new(Arc::clone(&layer)));

    tracing::info!("gRPC server listening on unix:{socket_path}");

    if let Some(port) = tcp_port {
        let addr: std::net::SocketAddr = format!("{tcp_host}:{port}").parse()?;
        tracing::info!("gRPC server also listening on TCP {addr}");

        // TCP: enforce Bearer token auth when keys are configured.
        // tonic::Status is inherently large (176 B); boxing it in an interceptor
        // return type adds overhead without benefit — suppress the lint here.
        #[allow(clippy::result_large_err)]
        let tcp_svc = AxServingServiceServer::with_interceptor(
            AxServingService::new(layer),
            move |req: tonic::Request<()>| {
                if keys.is_empty() {
                    return Ok(req);
                }
                let authorized = req
                    .metadata()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer "))
                    .map(|key| crate::auth::has_valid_api_key(key.trim(), &keys))
                    .unwrap_or(false);
                if authorized {
                    Ok(req)
                } else {
                    Err(tonic::Status::unauthenticated("missing or invalid API key"))
                }
            },
        );

        let uds_fut = Server::builder()
            .add_service(uds_svc)
            .serve_with_incoming(uds_stream);

        let tcp_fut = Server::builder().add_service(tcp_svc).serve(addr);

        tokio::try_join!(uds_fut, tcp_fut)?;
    } else {
        Server::builder()
            .add_service(uds_svc)
            .serve_with_incoming(uds_stream)
            .await?;
    }

    Ok(())
}
