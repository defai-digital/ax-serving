//! Validated control-plane deployment catalog.
//!
//! Worker inventory is observed state. This catalog is desired state: it maps
//! public logical model aliases to homogeneous runtime pools and records the
//! operator certification required for cross-pool routing.

use std::collections::BTreeMap;

use ax_serving_protocol::{
    DeploymentControlRecord, DeploymentDesiredState, DeploymentId, DeploymentIdentity,
    DeploymentSpec, Digest, EquivalenceClassId, EquivalencePolicy, LogicalModelId, PoolId,
    PoolSpec,
};
use serde::Serialize;

use crate::config::OrchestratorConfig;

use super::WorkerRegistry;
use super::registry::{WorkerId, WorkerModelEndpoint};
use super::request_profile::RequestProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentMode {
    LegacyCompat,
    Explicit,
}

impl DeploymentMode {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "legacy_compat" | "legacy-compat" => Ok(Self::LegacyCompat),
            "explicit" => Ok(Self::Explicit),
            other => anyhow::bail!(
                "unknown deployment mode {other:?}; expected legacy_compat or explicit"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModelResolution<'a> {
    Legacy {
        logical_model: LogicalModelId,
    },
    Explicit {
        logical_model: &'a LogicalModelId,
        deployments: Vec<&'a DeploymentSpec>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct LogicalModelSummary {
    pub id: LogicalModelId,
    pub deployments: Vec<DeploymentId>,
}

#[derive(Clone)]
pub struct RouteCandidate {
    pub endpoint: WorkerModelEndpoint,
    pub deployment: DeploymentSpec,
    pub pool: PoolSpec,
    pub observed_identity: DeploymentIdentity,
}

#[derive(Debug, Clone)]
pub struct DeploymentCatalog {
    mode: DeploymentMode,
    pools: BTreeMap<PoolId, PoolSpec>,
    deployments: BTreeMap<DeploymentId, DeploymentSpec>,
    deployments_by_model: BTreeMap<LogicalModelId, Vec<DeploymentId>>,
    equivalence_classes: BTreeMap<EquivalenceClassId, EquivalencePolicy>,
}

/// Atomically replaceable desired-state catalog. Request routing always takes
/// an immutable snapshot before any network await.
pub struct DeploymentCatalogStore {
    baseline: DeploymentCatalog,
    inner: std::sync::RwLock<DeploymentCatalog>,
}

impl DeploymentCatalogStore {
    pub fn new(catalog: DeploymentCatalog) -> Self {
        Self {
            baseline: catalog.clone(),
            inner: std::sync::RwLock::new(catalog),
        }
    }

    pub fn snapshot(&self) -> DeploymentCatalog {
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn replace(&self, catalog: DeploymentCatalog) {
        *self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = catalog;
    }

    pub fn apply_control_records(&self, records: &[DeploymentControlRecord]) -> anyhow::Result<()> {
        self.replace(DeploymentCatalog::from_control_records(
            &self.baseline,
            records,
        )?);
        Ok(())
    }

    pub fn catalog_for_records(
        &self,
        records: &[DeploymentControlRecord],
    ) -> anyhow::Result<DeploymentCatalog> {
        DeploymentCatalog::from_control_records(&self.baseline, records)
    }
}

impl DeploymentCatalog {
    pub fn from_config(config: &OrchestratorConfig) -> anyhow::Result<Self> {
        Self::new(
            DeploymentMode::parse(&config.deployment_mode)?,
            config.pools.clone(),
            config.deployments.clone(),
            config.equivalence_classes.clone(),
        )
    }

    pub fn new(
        mode: DeploymentMode,
        pools: Vec<PoolSpec>,
        deployments: Vec<DeploymentSpec>,
        equivalence_classes: Vec<EquivalencePolicy>,
    ) -> anyhow::Result<Self> {
        if mode == DeploymentMode::LegacyCompat
            && (!pools.is_empty() || !deployments.is_empty() || !equivalence_classes.is_empty())
        {
            anyhow::bail!("deployment declarations require orchestrator.deployment_mode=explicit");
        }
        // Explicit mode may start empty so a gateway can install and become
        // ready before operators create deployments via admin APIs or config.


        let mut pool_map = BTreeMap::new();
        for mut pool in pools {
            normalize_pool(&mut pool)?;
            let id = pool.id.clone();
            if pool_map.insert(id.clone(), pool).is_some() {
                anyhow::bail!("duplicate pool id '{id}'");
            }
        }

        let mut deployment_map = BTreeMap::new();
        let mut deployments_by_model = BTreeMap::<LogicalModelId, Vec<DeploymentId>>::new();
        for deployment in deployments {
            let pool = pool_map.get(&deployment.pool).ok_or_else(|| {
                anyhow::anyhow!(
                    "deployment '{}' references unknown pool '{}'",
                    deployment.id,
                    deployment.pool
                )
            })?;
            validate_deployment(&deployment, pool)?;
            let id = deployment.id.clone();
            if deployment_map
                .insert(id.clone(), deployment.clone())
                .is_some()
            {
                anyhow::bail!("duplicate deployment id '{id}'");
            }
            deployments_by_model
                .entry(deployment.logical_model.clone())
                .or_default()
                .push(id);
        }

        for ids in deployments_by_model.values_mut() {
            ids.sort();
        }

        let mut equivalence_map = BTreeMap::new();
        for policy in equivalence_classes {
            validate_equivalence_policy(&policy, &deployment_map)?;
            let id = policy.id.clone();
            if equivalence_map.insert(id.clone(), policy).is_some() {
                anyhow::bail!("duplicate equivalence class id '{id}'");
            }
        }

        validate_logical_model_groups(&deployments_by_model, &deployment_map, &equivalence_map)?;

        Ok(Self {
            mode,
            pools: pool_map,
            deployments: deployment_map,
            deployments_by_model,
            equivalence_classes: equivalence_map,
        })
    }

    pub fn from_control_records(
        base: &Self,
        records: &[DeploymentControlRecord],
    ) -> anyhow::Result<Self> {
        let mut deployments = records
            .iter()
            .filter(|record| record.desired_state != DeploymentDesiredState::Absent)
            .map(|record| {
                let mut deployment = record.deployment.clone();
                deployment.enabled = record.desired_state == DeploymentDesiredState::Enabled;
                deployment
            })
            .collect::<Vec<_>>();
        deployments.sort_by(|left, right| left.id.cmp(&right.id));
        let retained = deployments
            .iter()
            .map(|deployment| deployment.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        if deployments.is_empty() {
            return Ok(Self {
                mode: base.mode,
                pools: base.pools.clone(),
                deployments: BTreeMap::new(),
                deployments_by_model: BTreeMap::new(),
                equivalence_classes: BTreeMap::new(),
            });
        }
        let equivalence_classes = base
            .equivalence_classes
            .values()
            .filter_map(|policy| {
                let mut policy = policy.clone();
                policy
                    .certified_deployments
                    .retain(|deployment| retained.contains(deployment));
                (!policy.certified_deployments.is_empty()).then_some(policy)
            })
            .collect();
        Self::new(
            base.mode,
            base.pools.values().cloned().collect(),
            deployments,
            equivalence_classes,
        )
    }

    pub const fn mode(&self) -> DeploymentMode {
        self.mode
    }

    pub fn resolve(&self, model: &str) -> anyhow::Result<ModelResolution<'_>> {
        let logical_model = LogicalModelId::new(model.to_string())?;
        if self.mode == DeploymentMode::LegacyCompat {
            return Ok(ModelResolution::Legacy { logical_model });
        }

        let deployments = self
            .deployments_by_model
            .get(&logical_model)
            .into_iter()
            .flatten()
            .filter_map(|id| self.deployments.get(id))
            .filter(|deployment| deployment.enabled)
            .collect::<Vec<_>>();
        if deployments.is_empty() {
            anyhow::bail!("unknown logical model '{logical_model}'");
        }
        Ok(ModelResolution::Explicit {
            logical_model: self
                .deployments_by_model
                .get_key_value(&logical_model)
                .map(|(id, _)| id)
                .expect("logical model key exists after deployment lookup"),
            deployments,
        })
    }

    pub fn pool(&self, id: &PoolId) -> Option<&PoolSpec> {
        self.pools.get(id)
    }

    pub fn deployment(&self, id: &DeploymentId) -> Option<&DeploymentSpec> {
        self.deployments.get(id)
    }

    pub fn observed_endpoint_summary(
        &self,
        registry: &WorkerRegistry,
        deployment: &DeploymentSpec,
    ) -> (usize, usize) {
        let Some(pool) = self.pool(&deployment.pool) else {
            return (0, 0);
        };
        let mut endpoints = std::collections::HashMap::<WorkerId, WorkerModelEndpoint>::new();
        for request_kind in [
            super::registry::RequestKind::Llm,
            super::registry::RequestKind::Embedding,
            super::registry::RequestKind::Vision,
        ] {
            for endpoint in registry.eligible_model_endpoints(
                deployment.runtime_model_id.as_str(),
                request_kind,
                Some(&pool.runtime_kind),
                None,
                None,
                &std::collections::BTreeSet::new(),
                &deployment.required_capabilities,
                None,
            ) {
                if !endpoint_matches_pool(&endpoint, pool) {
                    continue;
                }
                let Ok(identity) = observed_identity(&endpoint) else {
                    continue;
                };
                if observed_identity_matches(self, deployment, &identity) {
                    endpoints.entry(endpoint.worker.id).or_insert(endpoint);
                }
            }
        }
        (
            endpoints.len(),
            endpoints
                .values()
                .map(|endpoint| endpoint.worker.inflight)
                .sum(),
        )
    }

    pub fn equivalence_policy(&self, id: &EquivalenceClassId) -> Option<&EquivalencePolicy> {
        self.equivalence_classes.get(id)
    }

    pub fn permits_failover(&self, source: &DeploymentSpec, target: &DeploymentSpec) -> bool {
        if source.id == target.id {
            return true;
        }
        let Some(class_id) = source.equivalence_class.as_ref() else {
            return false;
        };
        if target.equivalence_class.as_ref() != Some(class_id) {
            return false;
        }
        let Some(policy) = self.equivalence_classes.get(class_id) else {
            return false;
        };
        let (Some(source_identity), Some(target_identity)) = (
            source.expected_identity.as_ref(),
            target.expected_identity.as_ref(),
        ) else {
            return false;
        };
        policy.permits_failover(&source.id, source_identity, &target.id, target_identity)
    }

    pub fn route_candidates(
        &self,
        registry: &WorkerRegistry,
        profile: &RequestProfile,
        excluded_id: Option<WorkerId>,
        retry_source: Option<&DeploymentSpec>,
    ) -> anyhow::Result<Vec<RouteCandidate>> {
        let ModelResolution::Explicit { deployments, .. } =
            self.resolve(profile.logical_model.as_str())?
        else {
            anyhow::bail!("explicit route candidates requested in legacy compatibility mode");
        };

        let mut candidates = Vec::new();
        for deployment in deployments {
            if retry_source.is_some_and(|source| !self.permits_failover(source, deployment)) {
                continue;
            }
            if profile
                .required_pool
                .as_ref()
                .is_some_and(|required| required != &deployment.pool)
            {
                continue;
            }
            let Some(pool) = self.pool(&deployment.pool) else {
                continue;
            };
            let mut required_capabilities = profile.required_capabilities.clone();
            required_capabilities.extend(deployment.required_capabilities.iter().cloned());
            for endpoint in registry.eligible_model_endpoints(
                deployment.runtime_model_id.as_str(),
                profile.request_kind(),
                profile.runtime_hint.as_deref(),
                profile.minimum_context_tokens,
                profile.max_output_tokens,
                &profile.modalities,
                &required_capabilities,
                excluded_id,
            ) {
                // Explicit production routing requires a fenced protocol-v1
                // lease. Legacy registrations remain available only through
                // compatibility mode and cannot participate in HA admission.
                if endpoint.protocol_worker_id.is_none()
                    || endpoint.worker_instance_id.is_none()
                    || endpoint.registration_id.is_none()
                {
                    continue;
                }
                if !endpoint_matches_pool(&endpoint, pool) {
                    continue;
                }
                let Ok(observed_identity) = observed_identity(&endpoint) else {
                    continue;
                };
                if !observed_identity_matches(self, deployment, &observed_identity) {
                    continue;
                }
                candidates.push(RouteCandidate {
                    endpoint,
                    deployment: deployment.clone(),
                    pool: pool.clone(),
                    observed_identity,
                });
            }
        }

        if let Some(preferred_pool) = profile.preferred_pool.as_ref()
            && candidates
                .iter()
                .any(|candidate| &candidate.deployment.pool == preferred_pool)
        {
            candidates.retain(|candidate| &candidate.deployment.pool == preferred_pool);
        }
        Ok(candidates)
    }

    pub fn logical_models(&self) -> Vec<LogicalModelSummary> {
        self.deployments_by_model
            .iter()
            .filter_map(|(model, deployment_ids)| {
                let deployments = deployment_ids
                    .iter()
                    .filter(|id| self.deployments.get(*id).is_some_and(|value| value.enabled))
                    .cloned()
                    .collect::<Vec<_>>();
                (!deployments.is_empty()).then(|| LogicalModelSummary {
                    id: model.clone(),
                    deployments,
                })
            })
            .collect()
    }

    pub fn pools(&self) -> impl Iterator<Item = &PoolSpec> {
        self.pools.values()
    }

    pub fn deployments(&self) -> impl Iterator<Item = &DeploymentSpec> {
        self.deployments.values()
    }

    pub fn equivalence_classes(&self) -> impl Iterator<Item = &EquivalencePolicy> {
        self.equivalence_classes.values()
    }
}

fn endpoint_matches_pool(endpoint: &WorkerModelEndpoint, pool: &PoolSpec) -> bool {
    if !endpoint
        .runtime_kind
        .eq_ignore_ascii_case(&pool.runtime_kind)
        || endpoint.trust_domain.as_deref() != Some(pool.trust_domain.as_str())
        || pool
            .hardware_class
            .as_ref()
            .is_some_and(|required| endpoint.hardware_class.as_deref() != Some(required.as_str()))
    {
        return false;
    }

    for (key, expected) in &pool.selector {
        let observed = match key.as_str() {
            "worker_pool" => endpoint.worker_pool.as_deref(),
            "node_class" => endpoint.node_class.as_deref(),
            "hardware_class" => endpoint.hardware_class.as_deref(),
            "runtime_kind" => Some(endpoint.runtime_kind.as_str()),
            "trust_domain" => endpoint.trust_domain.as_deref(),
            _ => return false,
        };
        if observed != Some(expected.as_str()) {
            return false;
        }
    }

    pool.selector.contains_key("worker_pool")
        || endpoint.worker_pool.as_deref() == Some(pool.id.as_str())
}

fn observed_identity(endpoint: &WorkerModelEndpoint) -> anyhow::Result<DeploymentIdentity> {
    Ok(DeploymentIdentity {
        runtime_kind: endpoint
            .model
            .runtime_kind
            .clone()
            .unwrap_or_else(|| endpoint.runtime_kind.clone()),
        runtime_version: endpoint
            .model
            .runtime_version
            .clone()
            .or_else(|| endpoint.runtime_version.clone()),
        revision: endpoint.model.revision.clone(),
        artifact_digest: parse_optional_digest(endpoint.model.artifact_digest.as_deref())?,
        tokenizer_digest: parse_optional_digest(endpoint.model.tokenizer_digest.as_deref())?,
        template_digest: parse_optional_digest(endpoint.model.template_digest.as_deref())?,
        quantization: endpoint.model.quantization.clone(),
    })
}

fn parse_optional_digest(value: Option<&str>) -> anyhow::Result<Option<Digest>> {
    value
        .map(|value| Digest::new(value.to_string()).map_err(anyhow::Error::from))
        .transpose()
}

fn observed_identity_matches(
    catalog: &DeploymentCatalog,
    deployment: &DeploymentSpec,
    observed: &DeploymentIdentity,
) -> bool {
    let Some(expected) = deployment.expected_identity.as_ref() else {
        return deployment.equivalence_class.is_none();
    };
    if !deployment
        .required_identity
        .identities_match(expected, observed)
    {
        return false;
    }
    deployment
        .equivalence_class
        .as_ref()
        .and_then(|id| catalog.equivalence_policy(id))
        .is_none_or(|policy| policy.identity_policy.identities_match(expected, observed))
}

fn normalize_pool(pool: &mut PoolSpec) -> anyhow::Result<()> {
    pool.runtime_kind = pool.runtime_kind.trim().to_ascii_lowercase();
    if pool.runtime_kind.is_empty() {
        anyhow::bail!("pool '{}' runtime_kind must not be empty", pool.id);
    }
    if pool
        .runtime_kind
        .bytes()
        .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')))
    {
        anyhow::bail!(
            "pool '{}' runtime_kind contains invalid characters",
            pool.id
        );
    }
    pool.hardware_class = pool
        .hardware_class
        .take()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    for key in pool.selector.keys() {
        if !matches!(
            key.as_str(),
            "worker_pool" | "node_class" | "hardware_class" | "runtime_kind" | "trust_domain"
        ) {
            anyhow::bail!("pool '{}' has unsupported selector key '{key}'", pool.id);
        }
    }
    if pool.selector.values().any(|value| value.trim().is_empty()) {
        anyhow::bail!("pool '{}' selector values must not be empty", pool.id);
    }
    Ok(())
}

fn validate_deployment(deployment: &DeploymentSpec, pool: &PoolSpec) -> anyhow::Result<()> {
    if let Some(identity) = deployment.expected_identity.as_ref() {
        if identity.runtime_kind.trim().to_ascii_lowercase() != pool.runtime_kind {
            anyhow::bail!(
                "deployment '{}' expected runtime '{}' does not match pool '{}' runtime '{}'",
                deployment.id,
                identity.runtime_kind,
                pool.id,
                pool.runtime_kind
            );
        }
        if !deployment.required_identity.identity_is_complete(identity) {
            anyhow::bail!(
                "deployment '{}' expected identity is missing a required identity field",
                deployment.id
            );
        }
    }
    Ok(())
}

fn validate_equivalence_policy(
    policy: &EquivalencePolicy,
    deployments: &BTreeMap<DeploymentId, DeploymentSpec>,
) -> anyhow::Result<()> {
    if policy.certification_artifact.trim().is_empty() {
        anyhow::bail!(
            "equivalence class '{}' requires a certification_artifact",
            policy.id
        );
    }
    if policy.certified_deployments.is_empty() {
        anyhow::bail!(
            "equivalence class '{}' must certify at least one deployment",
            policy.id
        );
    }
    for deployment_id in &policy.certified_deployments {
        let deployment = deployments.get(deployment_id).ok_or_else(|| {
            anyhow::anyhow!(
                "equivalence class '{}' references unknown deployment '{}'",
                policy.id,
                deployment_id
            )
        })?;
        if deployment.equivalence_class.as_ref() != Some(&policy.id) {
            anyhow::bail!(
                "deployment '{}' must reference equivalence class '{}'",
                deployment.id,
                policy.id
            );
        }
    }
    Ok(())
}

fn validate_logical_model_groups(
    groups: &BTreeMap<LogicalModelId, Vec<DeploymentId>>,
    deployments: &BTreeMap<DeploymentId, DeploymentSpec>,
    equivalence_classes: &BTreeMap<EquivalenceClassId, EquivalencePolicy>,
) -> anyhow::Result<()> {
    for (logical_model, ids) in groups {
        let enabled = ids
            .iter()
            .filter_map(|id| deployments.get(id))
            .filter(|deployment| deployment.enabled)
            .collect::<Vec<_>>();
        if enabled.len() <= 1 {
            continue;
        }
        let Some(class_id) = enabled[0].equivalence_class.as_ref() else {
            anyhow::bail!(
                "logical model '{}' has multiple enabled deployments but no equivalence class",
                logical_model
            );
        };
        let policy = equivalence_classes.get(class_id).ok_or_else(|| {
            anyhow::anyhow!(
                "logical model '{}' references undeclared equivalence class '{}'",
                logical_model,
                class_id
            )
        })?;
        let certified = &policy.certified_deployments;
        for deployment in &enabled {
            if deployment.equivalence_class.as_ref() != Some(class_id)
                || !certified.contains(&deployment.id)
            {
                anyhow::bail!(
                    "all enabled deployments for logical model '{}' must be certified by '{}'",
                    logical_model,
                    class_id
                );
            }
            let Some(identity) = deployment.expected_identity.as_ref() else {
                anyhow::bail!(
                    "multi-deployment logical model '{}' requires expected identity for '{}'",
                    logical_model,
                    deployment.id
                );
            };
            if !policy.identity_policy.identity_is_complete(identity) {
                anyhow::bail!(
                    "deployment '{}' lacks identity required by equivalence class '{}'",
                    deployment.id,
                    class_id
                );
            }
        }
        let baseline = enabled[0];
        for deployment in enabled.iter().skip(1) {
            if !policy.permits_failover(
                &baseline.id,
                baseline
                    .expected_identity
                    .as_ref()
                    .expect("validated expected identity"),
                &deployment.id,
                deployment
                    .expected_identity
                    .as_ref()
                    .expect("validated expected identity"),
            ) {
                anyhow::bail!(
                    "deployments '{}' and '{}' do not satisfy equivalence class '{}'",
                    baseline.id,
                    deployment.id,
                    class_id
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ax_serving_protocol::{
        DeploymentIdentity, DeploymentSpec, Digest, EquivalenceClassId, EquivalencePolicy,
        IdentityField, IdentityPolicy, LogicalModelId, PoolId, PoolSpec, RuntimeModelId,
        TrustDomainId,
    };

    use super::{DeploymentCatalog, DeploymentMode, ModelResolution};

    fn digest(value: char) -> Digest {
        Digest::new(format!("sha256:{}", value.to_string().repeat(64))).unwrap()
    }

    fn pool(id: &str, runtime: &str) -> PoolSpec {
        PoolSpec {
            id: PoolId::new(id).unwrap(),
            runtime_kind: runtime.into(),
            hardware_class: None,
            trust_domain: TrustDomainId::new("private").unwrap(),
            selector: BTreeMap::new(),
        }
    }

    fn identity(runtime: &str) -> DeploymentIdentity {
        DeploymentIdentity {
            runtime_kind: runtime.into(),
            runtime_version: Some("1.0".into()),
            revision: Some("rev-1".into()),
            artifact_digest: None,
            tokenizer_digest: Some(digest('b')),
            template_digest: Some(digest('c')),
            quantization: Some("int4".into()),
        }
    }

    fn deployment(id: &str, pool: &str, runtime: &str) -> DeploymentSpec {
        DeploymentSpec {
            id: ax_serving_protocol::DeploymentId::new(id).unwrap(),
            logical_model: LogicalModelId::new("qwen/code").unwrap(),
            pool: PoolId::new(pool).unwrap(),
            runtime_model_id: RuntimeModelId::new(format!("{runtime}/qwen-code")).unwrap(),
            equivalence_class: Some(EquivalenceClassId::new("qwen-certified").unwrap()),
            expected_identity: Some(identity(runtime)),
            required_identity: IdentityPolicy {
                required_matching_fields: BTreeSet::from([
                    IdentityField::Revision,
                    IdentityField::TokenizerDigest,
                    IdentityField::TemplateDigest,
                    IdentityField::Quantization,
                ]),
            },
            required_capabilities: BTreeSet::new(),
            enabled: true,
        }
    }

    fn equivalence(deployments: &[DeploymentSpec]) -> EquivalencePolicy {
        EquivalencePolicy {
            id: EquivalenceClassId::new("qwen-certified").unwrap(),
            identity_policy: deployments[0].required_identity.clone(),
            certified_deployments: deployments.iter().map(|item| item.id.clone()).collect(),
            certification_artifact: "cert/qwen-v1.json".into(),
        }
    }

    #[test]
    fn explicit_catalog_resolves_certified_cross_runtime_alias() {
        let deployments = vec![
            deployment("qwen-mlx", "mac", "ax_engine"),
            deployment("qwen-cuda", "cuda", "vllm"),
        ];
        let catalog = DeploymentCatalog::new(
            DeploymentMode::Explicit,
            vec![pool("mac", "ax_engine"), pool("cuda", "vllm")],
            deployments.clone(),
            vec![equivalence(&deployments)],
        )
        .unwrap();

        let ModelResolution::Explicit { deployments, .. } = catalog.resolve("qwen/code").unwrap()
        else {
            panic!("expected explicit model resolution");
        };
        assert_eq!(deployments.len(), 2);
        assert!(catalog.permits_failover(deployments[0], deployments[1]));
    }

    #[test]
    fn multiple_deployments_without_certification_fail_closed() {
        let deployments = vec![
            deployment("qwen-mlx", "mac", "ax_engine"),
            deployment("qwen-cuda", "cuda", "vllm"),
        ];
        let error = DeploymentCatalog::new(
            DeploymentMode::Explicit,
            vec![pool("mac", "ax_engine"), pool("cuda", "vllm")],
            deployments,
            Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("undeclared equivalence class"));
    }

    #[test]
    fn legacy_mode_rejects_silently_ignored_declarations() {
        let error = DeploymentCatalog::new(
            DeploymentMode::LegacyCompat,
            vec![pool("mac", "ax_engine")],
            Vec::new(),
            Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("deployment declarations require"));
    }
}
