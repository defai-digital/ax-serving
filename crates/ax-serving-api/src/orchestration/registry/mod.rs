//! Worker registry: identity, health state machine, eligible-worker query.
//!
//! # Health state machine
//!
//! Worker health is derived from heartbeat age relative to `ttl_ms`:
//!
//! ```text
//! age ≤ ttl/3           → Healthy
//! ttl/3 < age ≤ 2*ttl/3 → Unhealthy { missed: 1 }
//! 2*ttl/3 < age ≤ ttl   → Unhealthy { missed: 2 }
//! age > ttl              → Dead  (evicted from registry)
//! ```
//!
//! With the defaults `heartbeat_ms = 5000`, `ttl_ms = 15000`, a worker
//! must heartbeat at least once every 5 s or it transitions through
//! Unhealthy within 10 s and is evicted at 15 s.

mod eligibility;
mod health_tick;
mod legacy_register;
mod normalize;
mod protocol_session;
mod snapshots;
mod types;

pub use types::{
    BackendKind, HeartbeatRequest, ModelInventoryEntry, ProtocolRegistryError,
    RegisterCapabilities, RegisterRequest, RegisterResponse, RequestKind, RuntimeKind,
    WorkerCapabilities, WorkerEntry, WorkerHealth, WorkerId, WorkerModelEndpoint, WorkerSnapshot,
    WorkerStatus,
};

use std::sync::Arc;

use dashmap::DashMap;

use ax_serving_protocol::WorkerId as ProtocolWorkerId;

#[cfg(test)]
use ax_serving_protocol::ProtocolVersion;
#[cfg(test)]
use std::collections::BTreeSet;
#[cfg(test)]
use std::sync::atomic::Ordering;

use self::types::ProtocolSession;

const MAX_WORKER_INFLIGHT: usize = 1_000_000;

// ── WorkerRegistry ────────────────────────────────────────────────────────────

/// Thread-safe worker registry.  All orchestration components share one
/// instance via `Clone` (backed by an `Arc`).
///
/// # Concurrency model
///
/// The registry uses a [`DashMap`] (sharded `RwLock<HashMap>`) instead of a
/// single `RwLock<HashMap>`.  Heartbeats from N concurrent workers each lock
/// only one shard, so they proceed in parallel rather than serialising on a
/// global write lock.  Read operations (`eligible_workers`, `list_all`, etc.)
/// are also sharded and do not block each other or mutation on other shards.
///
/// The only operation that must touch all entries is `tick()` (health eviction);
/// it iterates every shard with `iter_mut()`, collecting dead IDs, then removes
/// them in a second pass.
#[derive(Clone)]
pub struct WorkerRegistry {
    inner: Arc<DashMap<WorkerId, WorkerEntry>>,
    protocol_sessions: Arc<DashMap<ProtocolWorkerId, ProtocolSession>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
            protocol_sessions: Arc::new(DashMap::new()),
        }
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn reg_req(addr: &str, caps: &[&str], max: usize) -> RegisterRequest {
        RegisterRequest {
            worker_id: None,
            addr: addr.into(),
            capabilities: RegisterCapabilities::Legacy(
                caps.iter().map(|s| s.to_string()).collect(),
            ),
            backend: "native".into(),
            max_inflight: max,
            friendly_name: None,
            chip_model: None,
            worker_pool: None,
            node_class: None,
            ..Default::default()
        }
    }

    #[test]
    fn register_and_eligible() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["llama3-8b"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let workers = r.eligible_workers("llama3-8b");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, id);

        // Unknown model → empty
        assert!(r.eligible_workers("unknown-model").is_empty());
    }

    #[test]
    fn register_caps_max_inflight() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], usize::MAX), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.max_inflight, MAX_WORKER_INFLIGHT);
    }

    #[test]
    fn dispatch_workers_prefer_matching_pool_when_available() {
        let r = WorkerRegistry::new();
        let blue = r.register(
            RegisterRequest {
                worker_pool: Some("blue".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_pool: Some("green".into()),
                ..reg_req("127.0.0.1:8082", &["m1"], 4)
            },
            5000,
        );

        let workers =
            r.dispatch_workers_filtered("m1", RequestKind::Llm, None, None, Some("blue"), None);

        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, WorkerId::parse(&blue.worker_id).unwrap());
    }

    #[test]
    fn dispatch_workers_fall_back_when_preferred_pool_missing() {
        let r = WorkerRegistry::new();
        let blue = r.register(
            RegisterRequest {
                worker_pool: Some("blue".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        let green = r.register(
            RegisterRequest {
                worker_pool: Some("green".into()),
                ..reg_req("127.0.0.1:8082", &["m1"], 4)
            },
            5000,
        );

        let workers =
            r.dispatch_workers_filtered("m1", RequestKind::Llm, None, None, Some("red"), None);

        assert_eq!(workers.len(), 2);
        assert!(
            workers
                .iter()
                .any(|worker| worker.id == WorkerId::parse(&blue.worker_id).unwrap())
        );
        assert!(
            workers
                .iter()
                .any(|worker| worker.id == WorkerId::parse(&green.worker_id).unwrap())
        );
    }

    #[test]
    fn dispatch_workers_fall_back_when_soft_preferred_pool_is_excluded() {
        let r = WorkerRegistry::new();
        let blue = r.register(
            RegisterRequest {
                worker_pool: Some("blue".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        let green = r.register(
            RegisterRequest {
                worker_pool: Some("green".into()),
                ..reg_req("127.0.0.1:8082", &["m1"], 4)
            },
            5000,
        );

        let workers = r.dispatch_workers_filtered(
            "m1",
            RequestKind::Llm,
            None,
            None,
            Some("blue"),
            Some(WorkerId::parse(&blue.worker_id).unwrap()),
        );

        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, WorkerId::parse(&green.worker_id).unwrap());
    }

    #[test]
    fn parse_sglang_backend() {
        assert_eq!(BackendKind::parse("sglang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg_lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg-lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::SgLang.as_str(), "sglang");
    }

    #[test]
    fn parse_vllm_backend() {
        assert_eq!(BackendKind::parse("vllm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v_llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v-llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::Vllm.as_str(), "vllm");
        assert_eq!(
            RuntimeKind::from_backend(&BackendKind::Vllm),
            RuntimeKind::Vllm
        );
    }

    #[test]
    fn register_trims_backend_and_runtime_names() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                backend: " SGLANG ".into(),
                runtime: Some(" VLLM ".into()),
                ..reg_req("127.0.0.1:8081", &["m1"], 4)
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snap = r.get_snapshot(id).unwrap();

        assert_eq!(snap.backend, BackendKind::SgLang.as_str());
        assert_eq!(snap.runtime, RuntimeKind::Vllm.as_str());
    }

    #[test]
    fn structured_capabilities_registration_is_preserved() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["embed-1".into()],
                    max_context: Some(8192),
                }),
                backend: "sglang".into(),
                max_inflight: 8,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("thor".into()),
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.backend, "sglang");
        assert_eq!(snapshot.runtime, "sglang");
        assert_eq!(snapshot.capabilities, vec!["embed-1".to_string()]);
        assert_eq!(
            snapshot.supported_operations,
            vec!["llm".to_string(), "embedding".to_string()]
        );
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(snapshot.capability_descriptor.max_context, Some(8192));
    }

    #[test]
    fn structured_embedding_only_worker_is_not_llm_eligible() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: true,
                    vision: false,
                    models: vec!["embed-1".into()],
                    max_context: None,
                }),
                backend: "sglang".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("thor".into()),
                ..Default::default()
            },
            5000,
        );

        assert!(r.eligible_workers("embed-1").is_empty());
        assert_eq!(
            r.eligible_workers_for("embed-1", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn model_inventory_operations_constrain_mixed_runtime_models() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["chat-model".into(), "embed-model".into()],
                    max_context: None,
                }),
                model_inventory: vec![
                    ModelInventoryEntry {
                        id: "chat-model".into(),
                        supported_operations: vec!["llm".into()],
                        ..Default::default()
                    },
                    ModelInventoryEntry {
                        id: "embed-model".into(),
                        supported_operations: vec!["embedding".into()],
                        ..Default::default()
                    },
                ],
                supported_operations: vec!["llm".into(), "embedding".into()],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(r.eligible_workers("chat-model").len(), 1);
        assert!(
            r.eligible_workers_for("chat-model", RequestKind::Embedding)
                .is_empty(),
            "chat-only model inventory must reject embedding requests"
        );
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "embedding-only model inventory must reject llm requests"
        );
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn vision_requests_require_vision_operation() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["vision-model".into()],
                    max_context: None,
                }),
                supported_operations: vec!["llm".into()],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: true,
                    models: vec!["vision-model".into()],
                    max_context: None,
                }),
                supported_operations: vec!["llm".into(), "vision".into()],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_for("vision-model", RequestKind::Llm)
                .len(),
            2
        );
        let vision_workers = r.eligible_workers_for("vision-model", RequestKind::Vision);
        assert_eq!(vision_workers.len(), 1);
        assert_eq!(vision_workers[0].addr.to_string(), "http://127.0.0.1:8082");
    }

    #[test]
    fn registration_routes_models_from_inventory_when_capability_models_absent() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "inventory-model".into(),
                    max_context: Some(32768),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["inventory-model".to_string()]);
        assert_eq!(snapshot.model_inventory[0].id, "inventory-model");
        assert_eq!(r.eligible_workers("inventory-model").len(), 1);
    }

    #[test]
    fn registration_refreshes_structured_operations_from_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: None,
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    max_context: Some(8192),
                    supported_operations: vec!["embedding".into()],
                    ..Default::default()
                }],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["embed-model".to_string()]);
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(snapshot.capability_descriptor.max_context, Some(8192));
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "inventory-only embedding registration must not be llm eligible"
        );
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn registration_preserves_explicit_max_context_when_inventory_omits_it() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: Some(4096),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "chat-model".into(),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capability_descriptor.max_context, Some(4096));
        assert_eq!(
            r.eligible_workers_filtered("chat-model", RequestKind::Llm, None, Some(4096))
                .len(),
            1
        );
        assert!(
            r.eligible_workers_filtered("chat-model", RequestKind::Llm, None, Some(4097))
                .is_empty()
        );
    }

    #[test]
    fn registration_treats_model_inventory_as_additive() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["capability-model".into()],
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "inventory-model".into(),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(r.eligible_workers("capability-model").len(), 1);
        assert_eq!(r.eligible_workers("inventory-model").len(), 1);
    }

    #[test]
    fn legacy_worker_explicit_llm_only_operations_are_not_embedding_eligible() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                supported_operations: vec!["llm".into()],
                ..reg_req("127.0.0.1:8081", &["shared-model"], 4)
            },
            5000,
        );

        assert_eq!(r.eligible_workers("shared-model").len(), 1);
        assert!(
            r.eligible_workers_for("shared-model", RequestKind::Embedding)
                .is_empty(),
            "explicit supported_operations must constrain legacy worker routing"
        );
    }

    #[test]
    fn legacy_worker_without_explicit_operations_keeps_model_id_compatibility() {
        let r = WorkerRegistry::new();
        r.register(reg_req("127.0.0.1:8081", &["shared-model"], 4), 5000);

        assert_eq!(
            r.eligible_workers_for("shared-model", RequestKind::Embedding)
                .len(),
            1,
            "legacy model-id-only registrations remain backward compatible"
        );
    }

    #[test]
    fn explicit_operations_are_normalized_and_deduplicated() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["shared-model".into()],
                    max_context: None,
                }),
                supported_operations: vec![
                    " LLM ".into(),
                    "Embeddings".into(),
                    "llm".into(),
                    "completion".into(),
                    "text-generation".into(),
                    "text_generation".into(),
                ],
                backend: "native".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(
            snapshot.supported_operations,
            vec![
                "llm".to_string(),
                "embedding".to_string(),
                "text_generation".to_string(),
            ]
        );
    }

    #[test]
    fn explicit_operations_refresh_structured_capability_routing() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: false,
                    vision: false,
                    models: vec!["embed-model".into()],
                    max_context: None,
                }),
                supported_operations: vec!["Embeddings".into()],
                backend: "vllm".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "explicit embedding-only operations must not leave stale llm routing enabled"
        );
    }

    #[test]
    fn inventory_operations_are_normalized_to_routing_operations() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: false,
                    vision: false,
                    models: Vec::new(),
                    max_context: None,
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    supported_operations: vec!["Embeddings".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn backend_hint_filters_workers() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["shared-model".into()],
                    max_context: Some(4096),
                }),
                backend: "native".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("mac".into()),
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["shared-model".into()],
                    max_context: Some(16384),
                }),
                backend: "sglang".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: Some("thor".into()),
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("shared-model", RequestKind::Llm, Some("sglang"), None)
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("shared-model", RequestKind::Llm, Some("native"), None)
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("shared-model", RequestKind::Llm, Some("unknown"), None)
                .len(),
            2
        );
    }

    #[test]
    fn runtime_hint_filters_workers() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["runtime-model".into()],
                    max_context: Some(4096),
                }),
                backend: "auto".into(),
                runtime: Some("ax_engine".into()),
                max_inflight: 4,
                node_class: Some("mac".into()),
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["runtime-model".into()],
                    max_context: Some(16384),
                }),
                backend: "vllm".into(),
                runtime: Some("vllm".into()),
                max_inflight: 4,
                node_class: Some("pc-cuda".into()),
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("runtime-model", RequestKind::Llm, Some("ax_engine"), None)
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("runtime-model", RequestKind::Llm, Some("vllm"), None)
                .len(),
            1
        );
    }

    #[test]
    fn vllm_worker_exposes_runtime_metadata() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: true,
                    models: vec!["qwen3-32b".into()],
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "qwen3-32b".into(),
                    max_context: Some(32768),
                    quantization: Some("awq".into()),
                    artifact_format: Some("safetensors".into()),
                    modalities: vec!["text".into()],
                    supported_operations: vec!["llm".into(), "vision".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                runtime_mode: Some("adapter".into()),
                max_inflight: 16,
                friendly_name: None,
                chip_model: None,
                worker_pool: Some("cuda".into()),
                node_class: Some("pc-cuda".into()),
                runtime: Some("vllm".into()),
                runtime_version: Some("0.13.0".into()),
                hardware_class: Some("pc-cuda".into()),
                runtime_endpoint: Some("http://127.0.0.1:8000".into()),
                supported_operations: vec!["llm".into(), "vision".into()],
            },
            5000,
        );

        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.backend, "vllm");
        assert_eq!(snapshot.runtime, "vllm");
        assert_eq!(snapshot.runtime_mode.as_deref(), Some("adapter"));
        assert_eq!(snapshot.runtime_version.as_deref(), Some("0.13.0"));
        assert_eq!(snapshot.hardware_class.as_deref(), Some("pc-cuda"));
        assert_eq!(
            snapshot.runtime_endpoint.as_deref(),
            Some("http://127.0.0.1:8000")
        );
        assert_eq!(
            snapshot.supported_operations,
            vec!["llm".to_string(), "vision".to_string()]
        );
        assert_eq!(snapshot.model_inventory.len(), 1);
        assert_eq!(
            snapshot.model_inventory[0].quantization.as_deref(),
            Some("awq")
        );
        assert_eq!(
            snapshot.model_inventory[0].artifact_format.as_deref(),
            Some("safetensors")
        );
        assert_eq!(
            r.eligible_workers_filtered("qwen3-32b", RequestKind::Llm, Some("vllm"), None)
                .len(),
            1
        );
    }

    #[test]
    fn min_context_filters_workers() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["ctx-model".into()],
                    max_context: Some(4096),
                }),
                backend: "native".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["ctx-model".into()],
                    max_context: Some(16384),
                }),
                backend: "sglang".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("ctx-model", RequestKind::Llm, None, Some(8000))
                .len(),
            1
        );
        assert_eq!(
            r.eligible_workers_filtered("ctx-model", RequestKind::Llm, None, Some(20000))
                .len(),
            0
        );
    }

    #[test]
    fn min_context_respects_model_inventory_context_limit() {
        let r = WorkerRegistry::new();
        r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["short-model".into()],
                    max_context: Some(32768),
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "short-model".into(),
                    max_context: Some(4096),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );

        assert_eq!(
            r.eligible_workers_filtered("short-model", RequestKind::Llm, None, Some(4096))
                .len(),
            1
        );
        assert!(
            r.eligible_workers_filtered("short-model", RequestKind::Llm, None, Some(8000))
                .is_empty(),
            "per-model context limit must override broader worker context"
        );
    }

    #[test]
    fn explicit_endpoint_filters_fail_closed_on_unknown_declared_limits() {
        let registry = WorkerRegistry::new();
        registry.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["repo/model".into()],
                    max_context: None,
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "repo/model".into(),
                    modalities: vec!["text".into()],
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                backend: "vllm".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5_000,
        );

        assert_eq!(
            registry
                .eligible_model_endpoints(
                    "repo/model",
                    RequestKind::Llm,
                    None,
                    None,
                    None,
                    &BTreeSet::from(["text".to_string()]),
                    &BTreeSet::new(),
                    None,
                )
                .len(),
            1
        );
        assert!(
            registry
                .eligible_model_endpoints(
                    "repo/model",
                    RequestKind::Llm,
                    None,
                    Some(8_192),
                    None,
                    &BTreeSet::from(["text".to_string()]),
                    &BTreeSet::new(),
                    None,
                )
                .is_empty()
        );
        assert!(
            registry
                .eligible_model_endpoints(
                    "repo/model",
                    RequestKind::Llm,
                    None,
                    None,
                    Some(1_024),
                    &BTreeSet::from(["text".to_string()]),
                    &BTreeSet::new(),
                    None,
                )
                .is_empty()
        );
    }

    #[test]
    fn legacy_retry_guard_never_crosses_runtime_pool_or_identity() {
        let registry = WorkerRegistry::new();
        let register = |addr: &str, runtime: &str, pool: &str, revision: &str| RegisterRequest {
            addr: addr.into(),
            capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                llm: true,
                embedding: false,
                vision: false,
                models: vec!["shared-model".into()],
                max_context: Some(8_192),
            }),
            model_inventory: vec![ModelInventoryEntry {
                id: "shared-model".into(),
                revision: Some(revision.into()),
                modalities: vec!["text".into()],
                supported_operations: vec!["llm".into()],
                ..Default::default()
            }],
            backend: runtime.into(),
            runtime: Some(runtime.into()),
            worker_pool: Some(pool.into()),
            max_inflight: 4,
            ..Default::default()
        };
        let source = registry.register(register("127.0.0.1:8081", "vllm", "cuda", "rev-1"), 5_000);
        let same = registry.register(register("127.0.0.1:8082", "vllm", "cuda", "rev-1"), 5_000);
        let other_runtime =
            registry.register(register("127.0.0.1:8083", "native", "mac", "rev-1"), 5_000);
        let other_identity =
            registry.register(register("127.0.0.1:8084", "vllm", "cuda", "rev-2"), 5_000);
        let source = WorkerId::parse(&source.worker_id).unwrap();
        let same = WorkerId::parse(&same.worker_id).unwrap();
        let other_runtime = WorkerId::parse(&other_runtime.worker_id).unwrap();
        let other_identity = WorkerId::parse(&other_identity.worker_id).unwrap();

        assert!(registry.legacy_retry_compatible(source, same, "shared-model"));
        assert!(!registry.legacy_retry_compatible(source, other_runtime, "shared-model"));
        assert!(!registry.legacy_retry_compatible(source, other_identity, "shared-model"));
    }

    #[test]
    fn reregister_is_idempotent() {
        let r = WorkerRegistry::new();
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = resp1.worker_id.clone();

        // Re-register with same id — updates capabilities
        let mut req2 = reg_req("127.0.0.1:8081", &["m1", "m2"], 8);
        req2.worker_id = Some(id1.clone());
        let resp2 = r.register(req2, 5000);

        assert_eq!(resp2.worker_id, id1);
        assert_eq!(r.eligible_workers("m2").len(), 1);
        assert_eq!(r.list_all().len(), 1); // still one entry
    }

    #[test]
    fn reregister_without_inventory_preserves_matching_model_metadata() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    models: vec!["m1".into()],
                    ..Default::default()
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    quantization: Some("Q4_K_M".into()),
                    artifact_format: Some("gguf".into()),
                    ..Default::default()
                }],
                backend: "native".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let mut req2 = reg_req("127.0.0.1:8081", &["m1"], 8);
        req2.worker_id = Some(resp.worker_id);
        r.register(req2, 5000);

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.model_inventory.len(), 1);
        assert_eq!(snapshot.model_inventory[0].id, "m1");
        assert_eq!(
            snapshot.model_inventory[0].quantization.as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            snapshot.model_inventory[0].artifact_format.as_deref(),
            Some("gguf")
        );
    }

    #[test]
    fn reregister_without_inventory_preserves_retained_operation_routing() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: false,
                    vision: false,
                    models: vec!["embed-model".into()],
                    max_context: None,
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    supported_operations: vec!["embedding".into()],
                    ..Default::default()
                }],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.register(
            RegisterRequest {
                worker_id: Some(resp.worker_id),
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: false,
                    vision: false,
                    models: vec!["embed-model".into()],
                    max_context: None,
                }),
                backend: "sglang".into(),
                max_inflight: 8,
                ..Default::default()
            },
            5000,
        );

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert!(snapshot.capability_descriptor.embedding);
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "retained embedding-only inventory must continue to reject llm routing"
        );
    }

    #[test]
    fn reregister_without_inventory_prefers_retained_operations_over_derived_capabilities() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: false,
                    embedding: true,
                    vision: false,
                    models: vec!["embed-model".into()],
                    max_context: None,
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    supported_operations: vec!["embedding".into()],
                    ..Default::default()
                }],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.register(
            RegisterRequest {
                worker_id: Some(resp.worker_id),
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["embed-model".into()],
                    max_context: None,
                }),
                backend: "sglang".into(),
                max_inflight: 8,
                ..Default::default()
            },
            5000,
        );

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert!(snapshot.capability_descriptor.embedding);
        assert!(!snapshot.capability_descriptor.llm);
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "retained embedding-only inventory must override derived llm capabilities"
        );
    }

    #[test]
    fn reregister_without_models_clears_stale_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    models: vec!["m1".into()],
                    ..Default::default()
                }),
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    quantization: Some("Q4_K_M".into()),
                    ..Default::default()
                }],
                backend: "native".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let mut req2 = reg_req("127.0.0.1:8081", &[], 8);
        req2.worker_id = Some(resp.worker_id);
        r.register(req2, 5000);

        let snapshot = r.get_snapshot(id).unwrap();
        assert!(snapshot.model_inventory.is_empty());
        assert!(snapshot.capability_descriptor.models.is_empty());
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn drain_removes_from_eligible() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert_eq!(r.eligible_workers("m1").len(), 1);
        r.mark_drain(id);
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn eligible_healthy_count_excludes_draining() {
        let r = WorkerRegistry::new();

        // Register two workers.
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = WorkerId::parse(&resp1.worker_id).unwrap();
        let resp2 = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);
        let id2 = WorkerId::parse(&resp2.worker_id).unwrap();

        // Both healthy, neither draining → eligible = 2.
        assert_eq!(r.eligible_healthy_count(), 2);

        // Drain worker 1 — still healthy but not eligible.
        r.mark_drain(id1);
        assert_eq!(r.eligible_healthy_count(), 1);

        // Mark worker 2 unhealthy — now eligible = 0.
        r.mark_unhealthy(id2);
        assert_eq!(r.eligible_healthy_count(), 0);

        // counts() returns healthy=1 (worker 1 is Healthy but draining).
        // eligible_healthy_count() returns 0 because draining workers are
        // excluded from dispatch even if their health state is Healthy.
        let (healthy, _unhealthy, _draining) = r.counts();
        assert_eq!(healthy, 1); // worker 1 is still Healthy (just draining)
        // eligible_healthy_count correctly reports 0 despite healthy=1.
        assert_eq!(r.eligible_healthy_count(), 0);
    }

    #[test]
    fn unhealthy_removed_from_eligible_until_heartbeat() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert_eq!(r.eligible_workers("m1").len(), 1);
        r.mark_unhealthy(id);
        assert!(
            r.eligible_workers("m1").is_empty(),
            "unhealthy workers must be excluded from dispatch eligibility"
        );

        // A fresh heartbeat restores the worker to healthy/eligible.
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));
        assert_eq!(r.eligible_workers("m1").len(), 1);
    }

    #[test]
    fn tick_preserves_recent_unhealthy_until_heartbeat() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.mark_unhealthy(id);
        r.tick(9_000);

        assert_eq!(
            r.inner.get(&id).unwrap().health,
            WorkerHealth::Unhealthy { missed: 1 },
            "tick must not erase a dispatch failure before a heartbeat restores health"
        );
        assert!(r.eligible_workers("m1").is_empty());
        assert!(
            r.list_unhealthy_addrs()
                .iter()
                .any(|(candidate_id, _)| *candidate_id == id),
            "health ticker must still see the failed worker as a probe candidate"
        );

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        assert_eq!(
            r.inner.get(&id).unwrap().health,
            WorkerHealth::Healthy,
            "heartbeat is the signal that restores a dispatch-failed worker"
        );
        assert_eq!(r.eligible_workers("m1").len(), 1);
    }

    #[test]
    fn evict_removes_entry() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.evict(id);
        assert!(r.get_snapshot(id).is_none());
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn tick_evicts_stale_workers() {
        let r = WorkerRegistry::new();

        // Register with a past last_heartbeat by manipulating via a fake entry.
        // We can't set last_heartbeat directly from outside, so we test tick
        // with a very small ttl_ms (1 ms) so any entry looks stale immediately.
        r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);

        // With ttl=1ms, any worker will appear stale after the first tick.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let evicted = r.tick(1);
        assert!(!evicted.is_empty());
        assert!(r.eligible_workers("m1").is_empty());
    }

    #[test]
    fn heartbeat_updates_capabilities_from_model_ids() {
        let r = WorkerRegistry::new();
        // Register with no initial capabilities.
        let resp = r.register(reg_req("127.0.0.1:8081", &[], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        // Worker not eligible for any model yet.
        assert!(r.eligible_workers("m1").is_empty());

        // Heartbeat carries model_ids — orchestrator now knows worker has m1 loaded.
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 256 * 1024 * 1024,
                ..Default::default()
            }
        ));
        assert_eq!(r.eligible_workers("m1").len(), 1);

        // Subsequent heartbeat with empty model_ids clears capabilities.
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec![],
                rss_bytes: 0,
                ..Default::default()
            }
        ));
        assert!(
            r.eligible_workers("m1").is_empty(),
            "empty model_ids heartbeat must clear stale capabilities"
        );
    }

    #[test]
    fn model_ids_are_trimmed_for_registration_and_heartbeat_routing() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &[" m1 "], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert_eq!(r.eligible_workers("m1").len(), 1);
        assert_eq!(
            r.get_snapshot(id).unwrap().model_inventory[0].id.as_str(),
            "m1"
        );

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec![" m2 ".to_string()],
                rss_bytes: 256 * 1024 * 1024,
                ..Default::default()
            }
        ));

        assert!(r.eligible_workers("m1").is_empty());
        assert_eq!(r.eligible_workers("m2").len(), 1);
        assert_eq!(
            r.get_snapshot(id).unwrap().model_inventory[0].id.as_str(),
            "m2"
        );
    }

    #[test]
    fn heartbeat_retains_inventory_when_model_ids_need_trimming() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    max_context: Some(8192),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                ..reg_req("127.0.0.1:8081", &[], 4)
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec![" m1 ".to_string()],
                ..Default::default()
            }
        ));

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.model_inventory[0].id, "m1");
        assert_eq!(snapshot.model_inventory[0].max_context, Some(8192));
    }

    #[test]
    fn heartbeat_treats_model_inventory_as_additive() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1", "m2"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                model_ids: vec!["m1".into(), "m2".into()],
                model_inventory: vec![ModelInventoryEntry {
                    id: "m1".into(),
                    quantization: Some("q4".into()),
                    supported_operations: vec!["llm".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }
        ));

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(
            snapshot.capabilities,
            vec!["m1".to_string(), "m2".to_string()]
        );
        assert_eq!(snapshot.model_inventory.len(), 2);
        assert_eq!(r.eligible_workers("m1").len(), 1);
        assert_eq!(r.eligible_workers("m2").len(), 1);
    }

    #[test]
    fn heartbeat_refreshes_structured_operations_from_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: false,
                    vision: false,
                    models: vec!["chat-model".into()],
                    max_context: Some(2048),
                }),
                supported_operations: vec!["llm".into()],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                model_ids: vec!["embed-model".into()],
                model_inventory: vec![ModelInventoryEntry {
                    id: "embed-model".into(),
                    max_context: Some(8192),
                    supported_operations: vec!["embedding".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }
        ));

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["embed-model".to_string()]);
        assert_eq!(snapshot.supported_operations, vec!["embedding".to_string()]);
        assert_eq!(snapshot.capability_descriptor.max_context, Some(8192));
        assert!(
            r.eligible_workers("embed-model").is_empty(),
            "heartbeat inventory must remove stale llm eligibility"
        );
        assert_eq!(
            r.eligible_workers_for("embed-model", RequestKind::Embedding)
                .len(),
            1
        );
    }

    #[test]
    fn heartbeat_model_ids_refresh_operations_from_retained_inventory() {
        let r = WorkerRegistry::new();
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::Structured(WorkerCapabilities {
                    llm: true,
                    embedding: true,
                    vision: false,
                    models: vec!["chat-model".into(), "embed-model".into()],
                    max_context: None,
                }),
                model_inventory: vec![
                    ModelInventoryEntry {
                        id: "chat-model".into(),
                        supported_operations: vec!["llm".into()],
                        ..Default::default()
                    },
                    ModelInventoryEntry {
                        id: "embed-model".into(),
                        supported_operations: vec!["embedding".into()],
                        ..Default::default()
                    },
                ],
                backend: "sglang".into(),
                max_inflight: 4,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                model_ids: vec!["chat-model".into()],
                ..Default::default()
            }
        ));

        let snapshot = r.get_snapshot(id).unwrap();
        assert_eq!(snapshot.capabilities, vec!["chat-model".to_string()]);
        assert_eq!(snapshot.supported_operations, vec!["llm".to_string()]);
        assert!(snapshot.capability_descriptor.llm);
        assert!(!snapshot.capability_descriptor.embedding);
    }

    #[test]
    fn heartbeat_stores_thermal_and_rss() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 3,
                thermal_state: "serious".into(),
                model_ids: vec![],
                rss_bytes: 1_073_741_824, // 1 GiB
                ..Default::default()
            },
        );

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.thermal_state, "serious");
        assert_eq!(snap.rss_bytes, 1_073_741_824);
        assert_eq!(snap.inflight, 3);
    }

    #[test]
    fn heartbeat_does_not_overwrite_dispatcher_inflight_counter() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let counter = r.inflight_counter(id).expect("registered worker counter");

        counter.fetch_add(1, Ordering::Relaxed);
        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 0,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(
            snap.inflight, 1,
            "heartbeat must not erase dispatcher-owned in-flight accounting"
        );
        assert_eq!(r.eligible_workers("m1")[0].inflight, 1);

        counter.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(r.get_snapshot(id).unwrap().inflight, 0);
    }

    #[test]
    fn heartbeat_reported_inflight_is_used_when_dispatch_counter_is_lower() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 2,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        assert_eq!(r.get_snapshot(id).unwrap().inflight, 2);
        assert_eq!(r.eligible_workers("m1")[0].inflight, 2);
    }

    #[test]
    fn heartbeat_clamps_runtime_load_telemetry() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        assert!(r.heartbeat(
            id,
            HeartbeatRequest {
                inflight: usize::MAX,
                thermal_state: "nominal".into(),
                model_ids: vec!["m1".to_string()],
                active_sequences: usize::MAX,
                decode_tok_per_sec: f64::INFINITY,
                queue_depth: usize::MAX,
                error_rate: 2.5,
                kv_utilization: Some(f64::NAN),
                batch_utilization: Some(f64::INFINITY),
                ..Default::default()
            }
        ));

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.inflight, MAX_WORKER_INFLIGHT);
        assert_eq!(snap.active_sequences, MAX_WORKER_INFLIGHT);
        assert_eq!(snap.decode_tok_per_sec, 0.0);
        assert_eq!(snap.queue_depth, MAX_WORKER_INFLIGHT);
        assert_eq!(snap.error_rate, 1.0);
        assert_eq!(snap.kv_utilization, Some(0.0));
        assert_eq!(snap.batch_utilization, Some(0.0));

        let worker = r.eligible_workers("m1").remove(0);
        assert_eq!(worker.active_sequences, MAX_WORKER_INFLIGHT);
        assert_eq!(worker.kv_utilization, Some(0.0));
    }

    #[test]
    fn register_stores_identity_fields() {
        let r = WorkerRegistry::new();
        let req = RegisterRequest {
            worker_id: None,
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec![]),
            backend: "auto".into(),
            max_inflight: 4,
            friendly_name: Some("Aki's MacBook Pro".to_string()),
            chip_model: Some("Apple M3 Pro".to_string()),
            worker_pool: Some("blue".to_string()),
            node_class: Some("m3-pro".to_string()),
            ..Default::default()
        };
        let resp = r.register(req, 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.friendly_name.as_deref(), Some("Aki's MacBook Pro"));
        assert_eq!(snap.chip_model.as_deref(), Some("Apple M3 Pro"));
        assert_eq!(snap.worker_pool.as_deref(), Some("blue"));
        assert_eq!(snap.node_class.as_deref(), Some("m3-pro"));
    }

    #[test]
    fn register_normalizes_optional_metadata_fields() {
        let r = WorkerRegistry::new();
        let req = RegisterRequest {
            worker_id: None,
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
            backend: "auto".into(),
            runtime_mode: Some(" Adapter ".into()),
            runtime_version: Some(" 0.13.0 ".into()),
            hardware_class: Some(" pc-cuda ".into()),
            runtime_endpoint: Some(" http://127.0.0.1:8000 ".into()),
            max_inflight: 4,
            friendly_name: Some(" node-a ".into()),
            chip_model: Some(" NVIDIA L40S ".into()),
            worker_pool: Some(" blue ".into()),
            node_class: Some(" pc-cuda ".into()),
            ..Default::default()
        };
        let resp = r.register(req, 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.runtime_mode.as_deref(), Some("adapter"));
        assert_eq!(snap.runtime_version.as_deref(), Some("0.13.0"));
        assert_eq!(snap.hardware_class.as_deref(), Some("pc-cuda"));
        assert_eq!(
            snap.runtime_endpoint.as_deref(),
            Some("http://127.0.0.1:8000")
        );
        assert_eq!(snap.friendly_name.as_deref(), Some("node-a"));
        assert_eq!(snap.chip_model.as_deref(), Some("NVIDIA L40S"));
        assert_eq!(snap.worker_pool.as_deref(), Some("blue"));
        assert_eq!(snap.node_class.as_deref(), Some("pc-cuda"));
        assert_eq!(
            r.dispatch_workers_filtered_with_pool_mode(
                "m1",
                RequestKind::Llm,
                None,
                None,
                Some("blue"),
                true,
                None,
            )
            .len(),
            1
        );
    }

    #[test]
    fn reregister_clears_stale_identity_and_routing_fields() {
        let r = WorkerRegistry::new();
        let req = RegisterRequest {
            worker_id: None,
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
            backend: "vllm".into(),
            runtime: Some("vllm".into()),
            runtime_mode: Some("adapter".into()),
            runtime_version: Some("0.13.0".into()),
            hardware_class: Some("pc-cuda".into()),
            runtime_endpoint: Some("http://127.0.0.1:8000".into()),
            max_inflight: 4,
            friendly_name: Some("node-a".to_string()),
            chip_model: Some("NVIDIA L40S".to_string()),
            worker_pool: Some("blue".to_string()),
            node_class: Some("pc-cuda".to_string()),
            ..Default::default()
        };
        let resp = r.register(req, 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        let req = RegisterRequest {
            worker_id: Some(resp.worker_id),
            addr: "127.0.0.1:8081".into(),
            capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
            backend: "vllm".into(),
            runtime: Some("vllm".into()),
            runtime_mode: Some("adapter".into()),
            max_inflight: 4,
            ..Default::default()
        };
        r.register(req, 5000);

        let snap = r.get_snapshot(id).unwrap();
        assert_eq!(snap.runtime_version, None);
        assert_eq!(snap.hardware_class, None);
        assert_eq!(snap.runtime_endpoint, None);
        assert_eq!(snap.friendly_name, None);
        assert_eq!(snap.chip_model, None);
        assert_eq!(snap.worker_pool, None);
        assert_eq!(snap.node_class, None);
        assert!(
            r.dispatch_workers_filtered_with_pool_mode(
                "m1",
                RequestKind::Llm,
                None,
                None,
                Some("blue"),
                true,
                None,
            )
            .is_empty()
        );
    }

    #[test]
    fn counts_basic_breakdown() {
        let r = WorkerRegistry::new();

        // Two healthy, one to be drained, one to be marked unhealthy.
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = WorkerId::parse(&resp1.worker_id).unwrap();
        let _resp2 = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);
        let resp3 = r.register(reg_req("127.0.0.1:8083", &["m1"], 4), 5000);
        let id3 = WorkerId::parse(&resp3.worker_id).unwrap();
        let resp4 = r.register(reg_req("127.0.0.1:8084", &["m1"], 4), 5000);
        let id4 = WorkerId::parse(&resp4.worker_id).unwrap();

        // Initially all healthy, none draining.
        let (h, u, d) = r.counts();
        assert_eq!(h, 4);
        assert_eq!(u, 0);
        assert_eq!(d, 0);

        r.mark_drain(id1);
        r.mark_unhealthy(id3);

        // id1: Healthy+drain; id3: Unhealthy; id4: Healthy (for eviction below)
        let _ = id4;

        let (h, u, d) = r.counts();
        assert_eq!(
            h, 3,
            "id1 is still Healthy (just draining), id2+id4 healthy"
        );
        assert_eq!(u, 1, "id3 is unhealthy");
        assert_eq!(d, 1, "id1 is draining");
    }

    #[test]
    fn list_unhealthy_addrs_returns_unhealthy() {
        let r = WorkerRegistry::new();
        let resp1 = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id1 = WorkerId::parse(&resp1.worker_id).unwrap();
        let _resp2 = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);

        // Initially no unhealthy workers.
        assert!(r.list_unhealthy_addrs().is_empty());

        r.mark_unhealthy(id1);
        let unhealthy = r.list_unhealthy_addrs();
        assert_eq!(unhealthy.len(), 1);
        assert_eq!(unhealthy[0].0, id1);
        assert_eq!(unhealthy[0].1.to_string(), "http://127.0.0.1:8081");
    }

    #[test]
    fn backend_kind_parse_variants() {
        assert_eq!(BackendKind::parse("llama_cpp"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("llamacpp"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("llama-cpp"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("LLAMA_CPP"), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse(" llama-cpp "), BackendKind::LlamaCpp);
        assert_eq!(BackendKind::parse("sglang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg_lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("sg-lang"), BackendKind::SgLang);
        assert_eq!(BackendKind::parse("vllm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v_llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("v-llm"), BackendKind::Vllm);
        assert_eq!(BackendKind::parse("native"), BackendKind::Native);
        assert_eq!(BackendKind::parse("NATIVE"), BackendKind::Native);
        assert_eq!(BackendKind::parse("auto"), BackendKind::Auto);
        assert_eq!(BackendKind::parse("unknown"), BackendKind::Auto);
        assert_eq!(BackendKind::parse(""), BackendKind::Auto);
    }

    #[test]
    fn runtime_kind_parse_trims_operator_values() {
        assert_eq!(RuntimeKind::parse(" ax-engine "), RuntimeKind::AxEngine);
        assert_eq!(RuntimeKind::parse(" LLAMA_CPP "), RuntimeKind::LlamaCpp);
        assert_eq!(RuntimeKind::parse(" sglang "), RuntimeKind::SgLang);
        assert_eq!(RuntimeKind::parse(" V-LLM "), RuntimeKind::Vllm);
    }

    #[test]
    fn worker_id_display_and_parse_roundtrip() {
        let id = WorkerId::new();
        let s = id.to_string();
        let parsed = WorkerId::parse(&s).expect("must parse valid UUID string");
        assert_eq!(id, parsed);
        assert!(WorkerId::parse("not-a-uuid").is_none());
    }

    // ── register: invalid address falls back gracefully ───────────────────────

    #[test]
    fn register_invalid_addr_falls_back_to_loopback_sentinel() {
        let r = WorkerRegistry::new();
        // A bad address must not poison the registry — the worker is registered
        // with the "127.0.0.1:1" sentinel so it never receives real traffic.
        let resp = r.register(
            RegisterRequest {
                worker_id: None,
                addr: "not-a-valid:addr:at:all".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".to_string()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        let snap = r.get_snapshot(id).unwrap();
        // The sentinel address is "127.0.0.1:1".
        assert_eq!(snap.addr, "http://127.0.0.1:1");
        // Other fields should still be set correctly.
        assert_eq!(snap.max_inflight, 4);
        // The registry should still contain this entry (not poisoned/absent).
        assert_eq!(r.list_all().len(), 1);
    }

    // ── tick: full health state-machine transition matrix ─────────────────────

    #[test]
    fn tick_health_state_transitions_all_four_stages() {
        let r = WorkerRegistry::new();
        // ttl = 9000 ms → boundaries at ttl/3 = 3000 ms, 2*ttl/3 = 6000 ms.
        let ttl_ms = 9_000u64;

        // Helper: register a worker then backdates its last_heartbeat.
        let make_aged = |age_ms: u64| -> WorkerId {
            let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
            let id = WorkerId::parse(&resp.worker_id).unwrap();
            let past = std::time::Instant::now()
                .checked_sub(std::time::Duration::from_millis(age_ms))
                .expect("test machine must have been running for at least 10 s");
            r.inner.get_mut(&id).unwrap().last_heartbeat = past;
            id
        };

        let id_healthy = make_aged(1_000); // 1 s → Healthy   (≤ 3 s)
        let id_miss1 = make_aged(4_000); // 4 s → Unhealthy{1} (3 s < age ≤ 6 s)
        let id_miss2 = make_aged(7_000); // 7 s → Unhealthy{2} (6 s < age ≤ 9 s)
        let id_dead = make_aged(10_000); // 10 s → Dead         (> 9 s)

        let evicted = r.tick(ttl_ms);

        assert_eq!(evicted.len(), 1, "only the dead worker should be evicted");
        assert!(evicted.contains(&id_dead));
        assert!(
            r.inner.get(&id_dead).is_none(),
            "dead worker must be removed"
        );

        assert_eq!(
            r.inner.get(&id_healthy).unwrap().health,
            WorkerHealth::Healthy
        );
        assert_eq!(
            r.inner.get(&id_miss1).unwrap().health,
            WorkerHealth::Unhealthy { missed: 1 }
        );
        assert_eq!(
            r.inner.get(&id_miss2).unwrap().health,
            WorkerHealth::Unhealthy { missed: 2 }
        );
    }

    // ── tick: draining workers ─────────────────────────────────────────────────

    #[test]
    fn tick_draining_worker_evicted_only_after_ttl() {
        let r = WorkerRegistry::new();
        let ttl_ms = 9_000u64;

        // Register two draining workers: one fresh, one stale.
        let resp_fresh = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id_fresh = WorkerId::parse(&resp_fresh.worker_id).unwrap();
        r.mark_drain(id_fresh);

        let resp_stale = r.register(reg_req("127.0.0.1:8082", &["m1"], 4), 5000);
        let id_stale = WorkerId::parse(&resp_stale.worker_id).unwrap();
        r.mark_drain(id_stale);
        // Backdate the stale worker past the TTL.
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(ttl_ms + 1_000))
            .unwrap();
        r.inner.get_mut(&id_stale).unwrap().last_heartbeat = past;

        let evicted = r.tick(ttl_ms);

        assert!(
            evicted.contains(&id_stale),
            "stale draining worker must be evicted"
        );
        assert!(
            !evicted.contains(&id_fresh),
            "fresh draining worker must not be evicted yet"
        );
        assert!(r.inner.get(&id_fresh).is_some());
    }

    // ── mark_unhealthy: idempotent — already-unhealthy stays at missed:1 ───────

    #[test]
    fn mark_unhealthy_is_idempotent_does_not_escalate() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        r.mark_unhealthy(id);
        // Second call on an already-unhealthy worker must not escalate missed count.
        r.mark_unhealthy(id);

        assert_eq!(
            r.inner.get(&id).unwrap().health,
            WorkerHealth::Unhealthy { missed: 1 },
            "mark_unhealthy must not escalate beyond missed:1"
        );
    }

    // ── eligible_workers: exact model match (no substring) ───────────────────

    #[test]
    fn eligible_workers_requires_exact_model_id_match() {
        let r = WorkerRegistry::new();
        // Worker has "llama3-8b" — must NOT be returned for "llama3" or "llama3-8b-v2".
        r.register(reg_req("127.0.0.1:8081", &["llama3-8b"], 4), 5000);

        assert_eq!(
            r.eligible_workers("llama3-8b").len(),
            1,
            "exact match must work"
        );
        assert!(
            r.eligible_workers("llama3").is_empty(),
            "substring 'llama3' must not match 'llama3-8b'"
        );
        assert!(
            r.eligible_workers("llama3-8b-v2").is_empty(),
            "extended name must not match shorter capability"
        );
    }

    // ── mark_drain / mark_unhealthy on unknown worker ────────────────────────

    #[test]
    fn mark_drain_returns_false_for_unknown_worker() {
        let r = WorkerRegistry::new();
        assert!(
            !r.mark_drain(WorkerId::new()),
            "mark_drain must return false for an unregistered worker"
        );
    }

    #[test]
    fn mark_unhealthy_noop_for_unknown_worker() {
        // Should not panic — no entry exists, so the call is silently ignored.
        let r = WorkerRegistry::new();
        r.mark_unhealthy(WorkerId::new()); // must not panic
        assert!(r.list_all().is_empty());
    }

    #[test]
    fn heartbeat_resets_health() {
        let r = WorkerRegistry::new();
        let resp = r.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5000);
        let id = WorkerId::parse(&resp.worker_id).unwrap();

        // Heartbeat returns true for a known worker and resets last_heartbeat.
        let hb = HeartbeatRequest {
            inflight: 2,
            thermal_state: "nominal".into(),
            model_ids: vec!["m1".to_string()],
            rss_bytes: 1024 * 1024 * 512,
            ..Default::default()
        };
        assert!(r.heartbeat(id, hb));

        // Heartbeat returns false for an unknown worker.
        assert!(!r.heartbeat(
            WorkerId::new(),
            HeartbeatRequest {
                inflight: 0,
                thermal_state: String::new(),
                model_ids: vec![],
                rss_bytes: 0,
                ..Default::default()
            }
        ));

        // After a fresh heartbeat, a tick with a large TTL must not evict the worker.
        let evicted = r.tick(60_000);
        assert!(evicted.is_empty());
        assert_eq!(r.eligible_workers("m1").len(), 1);
    }

    #[test]
    fn heartbeat_stores_cache_telemetry() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8081".into(),
                capabilities: RegisterCapabilities::default(),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 2,
                kv_pages_used: 100,
                kv_pages_total: 256,
                active_batch_size: 3,
                max_batch_size: 8,
                ..Default::default()
            },
        );
        let workers = reg.list_all();
        let snap = workers.iter().find(|w| w.id == id).unwrap();
        assert_eq!(snap.kv_pages_used, 100);
        assert_eq!(snap.kv_pages_total, 256);
        assert_eq!(snap.active_batch_size, 3);
        assert_eq!(snap.max_batch_size, 8);
    }

    #[test]
    fn worker_status_computes_kv_utilization_and_batch_headroom() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 1,
                kv_pages_used: 200,
                kv_pages_total: 400,
                active_batch_size: 2,
                max_batch_size: 8,
                model_ids: vec!["m1".into()],
                ..Default::default()
            },
        );
        let eligible = reg.eligible_workers("m1");
        assert_eq!(eligible.len(), 1);
        let ws = &eligible[0];
        assert!((ws.kv_utilization.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!((ws.batch_headroom.unwrap() - 0.75).abs() < f64::EPSILON);

        let all = reg.list_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].kv_pages_used, 200);
        assert_eq!(all[0].kv_pages_total, 400);
        assert_eq!(all[0].kv_utilization, Some(0.5));
        assert_eq!(all[0].batch_utilization, Some(0.25));
    }

    #[test]
    fn worker_status_uses_ratio_telemetry_when_counters_are_absent() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8084".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 1,
                kv_utilization: Some(0.6),
                batch_utilization: Some(0.25),
                model_ids: vec!["m1".into()],
                ..Default::default()
            },
        );

        let eligible = reg.eligible_workers("m1");
        assert_eq!(eligible[0].kv_utilization, Some(0.6));
        assert_eq!(eligible[0].batch_headroom, Some(0.75));

        let all = reg.list_all();
        assert_eq!(all[0].kv_utilization, Some(0.6));
        assert_eq!(all[0].batch_utilization, Some(0.25));
    }

    #[test]
    fn worker_status_clamps_kv_utilization_to_one() {
        let reg = WorkerRegistry::new();
        let resp = reg.register(
            RegisterRequest {
                worker_id: None,
                addr: "127.0.0.1:8082".into(),
                capabilities: RegisterCapabilities::Legacy(vec!["m1".into()]),
                backend: "auto".into(),
                max_inflight: 4,
                friendly_name: None,
                chip_model: None,
                worker_pool: None,
                node_class: None,
                ..Default::default()
            },
            5000,
        );
        let id = WorkerId::parse(&resp.worker_id).unwrap();
        reg.heartbeat(
            id,
            HeartbeatRequest {
                inflight: 1,
                kv_pages_used: 500,
                kv_pages_total: 400,
                model_ids: vec!["m1".into()],
                ..Default::default()
            },
        );
        let eligible = reg.eligible_workers("m1");
        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].kv_utilization, Some(1.0));
    }

    #[test]
    fn authoritative_runtime_readiness_controls_eligibility() {
        let registry = WorkerRegistry::new();
        let response = registry.register(reg_req("127.0.0.1:8081", &["m1"], 4), 5_000);
        let id = WorkerId::parse(&response.worker_id).unwrap();

        assert!(registry.heartbeat(
            id,
            HeartbeatRequest {
                model_ids: vec!["m1".into()],
                runtime_ready: Some(false),
                runtime_state: Some("unavailable".into()),
                runtime_status_reason: Some("runtime_health_failed".into()),
                protocol_version: Some(ProtocolVersion { major: 1, minor: 0 }),
                agent_version: Some("3.0.0".into()),
                ..Default::default()
            },
        ));
        assert!(registry.eligible_workers("m1").is_empty());
        let unavailable = registry.get_snapshot(id).unwrap();
        assert_eq!(unavailable.runtime_ready, Some(false));
        assert_eq!(unavailable.health, "unhealthy");

        assert!(registry.heartbeat(
            id,
            HeartbeatRequest {
                model_ids: vec!["m1".into()],
                runtime_ready: Some(true),
                runtime_state: Some("ready".into()),
                protocol_version: Some(ProtocolVersion { major: 1, minor: 0 }),
                agent_version: Some("3.0.0".into()),
                ..Default::default()
            },
        ));
        assert_eq!(registry.eligible_workers("m1").len(), 1);
        assert_eq!(registry.get_snapshot(id).unwrap().runtime_ready, Some(true));
    }
}
