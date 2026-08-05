//! Adaptive federation policy inputs for replay, shadow, canary, and rollback.
//!
//! These helpers score and select among complete execution domains using
//! bounded cost/SLO signals. They never inspect prompts, activations, KV, or
//! rank-local state. Active traffic mutation is gated by policy mode.

use serde::{Deserialize, Serialize};

use crate::{DecisionProfileV1, DecisionValidationError, DomainId, PolicyMode, PolicyVersion};

/// Prompt-free cost and latency prediction for one complete domain candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DomainCostSignal {
    pub domain: DomainId,
    pub predicted_cost_microusd: u64,
    pub predicted_latency_ms: u64,
    pub ready: bool,
    /// Higher is more preferred when scores otherwise tie.
    pub stability_rank: u32,
}

/// Versioned adaptive policy that can be replayed against retained decisions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveFederationPolicyV1 {
    pub policy_version: PolicyVersion,
    pub mode: PolicyMode,
    /// Canary share in parts-per-million of eligible traffic (0..=1_000_000).
    pub canary_share_ppm: u32,
    /// Domain promoted into canary or active selection when eligible.
    pub target_domain: DomainId,
    /// Previous active domain used for canary control and rollback target.
    pub baseline_domain: DomainId,
    pub max_cost_microusd: Option<u64>,
    pub latency_slo_ms: Option<u64>,
}

/// Outcome of applying an adaptive policy to a candidate set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveSelection {
    pub selected_domain: DomainId,
    pub mode: PolicyMode,
    /// Domain that would have been chosen under active policy (shadow/canary).
    pub counterfactual_domain: Option<DomainId>,
    pub predicted_cost_microusd: u64,
    pub predicted_latency_ms: u64,
    pub rolled_back: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AdaptivePolicyError {
    #[error("adaptive policy profile is invalid: {0}")]
    Profile(#[from] DecisionValidationError),
    #[error("canary share must be at most 1_000_000 ppm")]
    CanaryShareOutOfRange,
    #[error("adaptive target and baseline domains must differ")]
    SamePolicyDomain,
    #[error("no eligible domain candidate remains after cost/SLO filters")]
    NoEligibleCandidate,
    #[error("target or baseline domain is missing from the candidate set")]
    MissingPolicyDomain,
}

impl AdaptiveFederationPolicyV1 {
    pub fn validate(&self) -> Result<(), AdaptivePolicyError> {
        if self.canary_share_ppm > 1_000_000 {
            return Err(AdaptivePolicyError::CanaryShareOutOfRange);
        }
        if self.target_domain == self.baseline_domain {
            return Err(AdaptivePolicyError::SamePolicyDomain);
        }
        DecisionProfileV1 {
            max_cost_microusd: self.max_cost_microusd,
            latency_slo_ms: self.latency_slo_ms,
            ..DecisionProfileV1::default()
        }
        .validate()?;
        Ok(())
    }

    /// Filter candidates by hard cost/SLO bounds from the policy profile.
    pub fn eligible<'a>(&self, candidates: &'a [DomainCostSignal]) -> Vec<&'a DomainCostSignal> {
        candidates
            .iter()
            .filter(|candidate| candidate.ready)
            .filter(|candidate| {
                self.max_cost_microusd
                    .is_none_or(|limit| candidate.predicted_cost_microusd <= limit)
            })
            .filter(|candidate| {
                self.latency_slo_ms
                    .is_none_or(|limit| candidate.predicted_latency_ms <= limit)
            })
            .collect()
    }

    /// Select a domain for the given mode. Shadow never mutates active traffic.
    pub fn select(
        &self,
        candidates: &[DomainCostSignal],
        request_hash: u64,
    ) -> Result<AdaptiveSelection, AdaptivePolicyError> {
        self.validate()?;
        let eligible = self.eligible(candidates);
        if eligible.is_empty() {
            return Err(AdaptivePolicyError::NoEligibleCandidate);
        }

        let baseline = find_domain(&eligible, &self.baseline_domain);
        let target = find_domain(&eligible, &self.target_domain);

        match self.mode {
            PolicyMode::Shadow => {
                let active = baseline.ok_or(AdaptivePolicyError::MissingPolicyDomain)?;
                Ok(selection(active, target, self.mode, false))
            }
            PolicyMode::Canary => {
                let baseline = baseline.ok_or(AdaptivePolicyError::MissingPolicyDomain)?;
                let Some(target) = target else {
                    return Ok(selection(baseline, None, self.mode, false));
                };
                let use_target = request_hash % 1_000_000 < u64::from(self.canary_share_ppm);
                if use_target {
                    Ok(selection(target, Some(baseline), self.mode, false))
                } else {
                    Ok(selection(baseline, Some(target), self.mode, false))
                }
            }
            PolicyMode::Active => {
                let selected = target
                    .or(baseline)
                    .ok_or(AdaptivePolicyError::MissingPolicyDomain)?;
                Ok(selection(selected, None, self.mode, false))
            }
            PolicyMode::Rollback => {
                // Rollback always restores the baseline and records that fact.
                let selected = baseline.ok_or(AdaptivePolicyError::MissingPolicyDomain)?;
                Ok(selection(selected, target, self.mode, true))
            }
        }
    }

    /// Replay selection for retained request hashes (deterministic).
    pub fn replay(
        &self,
        candidates: &[DomainCostSignal],
        request_hashes: &[u64],
    ) -> Result<Vec<AdaptiveSelection>, AdaptivePolicyError> {
        request_hashes
            .iter()
            .map(|hash| self.select(candidates, *hash))
            .collect()
    }
}

fn find_domain<'a>(
    candidates: &[&'a DomainCostSignal],
    domain: &DomainId,
) -> Option<&'a DomainCostSignal> {
    candidates
        .iter()
        .copied()
        .find(|candidate| &candidate.domain == domain)
}

fn selection(
    selected: &DomainCostSignal,
    counterfactual: Option<&DomainCostSignal>,
    mode: PolicyMode,
    rolled_back: bool,
) -> AdaptiveSelection {
    AdaptiveSelection {
        selected_domain: selected.domain.clone(),
        mode,
        counterfactual_domain: counterfactual.map(|value| value.domain.clone()),
        predicted_cost_microusd: selected.predicted_cost_microusd,
        predicted_latency_ms: selected.predicted_latency_ms,
        rolled_back,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PolicyVersion;

    fn domain(name: &str) -> DomainId {
        DomainId::new(name).unwrap()
    }

    fn signal(name: &str, latency: u64, cost: u64, ready: bool) -> DomainCostSignal {
        DomainCostSignal {
            domain: domain(name),
            predicted_cost_microusd: cost,
            predicted_latency_ms: latency,
            ready,
            stability_rank: 1,
        }
    }

    fn policy(mode: PolicyMode) -> AdaptiveFederationPolicyV1 {
        AdaptiveFederationPolicyV1 {
            policy_version: PolicyVersion::new("1").unwrap(),
            mode,
            canary_share_ppm: 250_000,
            target_domain: domain("mac-cluster"),
            baseline_domain: domain("mac-single"),
            max_cost_microusd: Some(5_000),
            latency_slo_ms: Some(2_000),
        }
    }

    #[test]
    fn shadow_mode_never_changes_active_selection() {
        let candidates = vec![
            signal("mac-single", 100, 100, true),
            signal("mac-cluster", 80, 200, true),
        ];
        let decision = policy(PolicyMode::Shadow).select(&candidates, 1).unwrap();
        assert_eq!(decision.selected_domain, domain("mac-single"));
        assert_eq!(decision.counterfactual_domain, Some(domain("mac-cluster")));
        assert!(!decision.rolled_back);
    }

    #[test]
    fn canary_splits_deterministically_by_request_hash() {
        let candidates = vec![
            signal("mac-single", 100, 100, true),
            signal("mac-cluster", 80, 200, true),
        ];
        let policy = policy(PolicyMode::Canary);
        let mut target_hits = 0;
        // Spread hashes across the full ppm space so the 25% canary share is observable.
        for bucket in 0..1_000_u64 {
            let hash = bucket * 1_000;
            if policy.select(&candidates, hash).unwrap().selected_domain == domain("mac-cluster") {
                target_hits += 1;
            }
        }
        assert!(target_hits > 200 && target_hits < 300, "hits={target_hits}");
    }

    #[test]
    fn cost_and_slo_filters_fail_closed() {
        let candidates = vec![
            signal("mac-single", 100, 100, true),
            signal("mac-cluster", 9_000, 200, true),
        ];
        let decision = policy(PolicyMode::Active).select(&candidates, 42).unwrap();
        assert_eq!(decision.selected_domain, domain("mac-single"));
    }

    #[test]
    fn rollback_restores_baseline() {
        let candidates = vec![
            signal("mac-single", 100, 100, true),
            signal("mac-cluster", 80, 200, true),
        ];
        let decision = policy(PolicyMode::Rollback).select(&candidates, 7).unwrap();
        assert_eq!(decision.selected_domain, domain("mac-single"));
        assert!(decision.rolled_back);
    }

    #[test]
    fn shadow_and_rollback_fail_closed_without_the_baseline() {
        let candidates = vec![signal("mac-cluster", 80, 200, true)];
        for mode in [PolicyMode::Shadow, PolicyMode::Rollback] {
            assert_eq!(
                policy(mode).select(&candidates, 1),
                Err(AdaptivePolicyError::MissingPolicyDomain)
            );
        }
    }

    #[test]
    fn target_and_baseline_must_be_distinct() {
        let mut invalid = policy(PolicyMode::Active);
        invalid.target_domain = invalid.baseline_domain.clone();
        assert_eq!(invalid.validate(), Err(AdaptivePolicyError::SamePolicyDomain));
    }

    #[test]
    fn replay_is_deterministic_for_retained_hashes() {
        let candidates = vec![
            signal("mac-single", 100, 100, true),
            signal("mac-cluster", 80, 200, true),
        ];
        let policy = policy(PolicyMode::Canary);
        let hashes = [1_u64, 2, 3, 4, 5];
        let first = policy.replay(&candidates, &hashes).unwrap();
        let second = policy.replay(&candidates, &hashes).unwrap();
        assert_eq!(first, second);
    }
}
