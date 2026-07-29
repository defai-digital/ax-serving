//! Adaptive federation policy hooks for Mac cluster and other domains.
//!
//! Wraps runtime-neutral protocol adaptive selection so the gateway can run
//! replay, shadow, canary, and rollback evaluations without inspecting ranks,
//! activations, KV, or prompts.

use ax_serving_protocol::{
    AdaptiveFederationPolicyV1, AdaptivePolicyError, AdaptiveSelection, DomainCostSignal,
    DomainId, PolicyMode, PolicyVersion,
};

/// Build a validated adaptive policy from operator inputs.
pub fn build_policy(
    mode: PolicyMode,
    target_domain: DomainId,
    baseline_domain: DomainId,
    canary_share_ppm: u32,
    max_cost_microusd: Option<u64>,
    latency_slo_ms: Option<u64>,
) -> Result<AdaptiveFederationPolicyV1, AdaptivePolicyError> {
    let policy = AdaptiveFederationPolicyV1 {
        policy_version: PolicyVersion::new("1").expect("static adaptive policy version is valid"),
        mode,
        canary_share_ppm,
        target_domain,
        baseline_domain,
        max_cost_microusd,
        latency_slo_ms,
    };
    policy.validate()?;
    Ok(policy)
}

/// Select a domain under the configured adaptive mode.
pub fn select_domain(
    policy: &AdaptiveFederationPolicyV1,
    candidates: &[DomainCostSignal],
    request_hash: u64,
) -> Result<AdaptiveSelection, AdaptivePolicyError> {
    policy.select(candidates, request_hash)
}

/// Replay retained request hashes against a candidate set for shadow analysis.
pub fn replay_domain_policy(
    policy: &AdaptiveFederationPolicyV1,
    candidates: &[DomainCostSignal],
    request_hashes: &[u64],
) -> Result<Vec<AdaptiveSelection>, AdaptivePolicyError> {
    policy.replay(candidates, request_hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(name: &str) -> DomainId {
        DomainId::new(name).unwrap()
    }

    fn signal(name: &str, latency: u64, cost: u64) -> DomainCostSignal {
        DomainCostSignal {
            domain: domain(name),
            predicted_cost_microusd: cost,
            predicted_latency_ms: latency,
            ready: true,
            stability_rank: 1,
        }
    }

    #[test]
    fn shadow_records_counterfactual_without_routing_to_target() {
        let policy = build_policy(
            PolicyMode::Shadow,
            domain("mac-cluster"),
            domain("mac-single"),
            500_000,
            Some(10_000),
            Some(5_000),
        )
        .unwrap();
        let candidates = vec![
            signal("mac-single", 120, 100),
            signal("mac-cluster", 90, 200),
        ];
        let selected = select_domain(&policy, &candidates, 11).unwrap();
        assert_eq!(selected.selected_domain, domain("mac-single"));
        assert_eq!(
            selected.counterfactual_domain,
            Some(domain("mac-cluster"))
        );
    }

    #[test]
    fn rollback_hook_restores_baseline_domain() {
        let policy = build_policy(
            PolicyMode::Rollback,
            domain("mac-cluster"),
            domain("mac-single"),
            0,
            None,
            None,
        )
        .unwrap();
        let candidates = vec![
            signal("mac-single", 120, 100),
            signal("mac-cluster", 90, 200),
        ];
        let selected = select_domain(&policy, &candidates, 99).unwrap();
        assert!(selected.rolled_back);
        assert_eq!(selected.selected_domain, domain("mac-single"));
    }

    #[test]
    fn replay_hook_is_stable_for_canary_policy() {
        let policy = build_policy(
            PolicyMode::Canary,
            domain("mac-cluster"),
            domain("mac-single"),
            500_000,
            None,
            None,
        )
        .unwrap();
        let candidates = vec![
            signal("mac-single", 120, 100),
            signal("mac-cluster", 90, 200),
        ];
        // Hashes span both sides of the 50% canary boundary (hash % 1_000_000).
        let hashes = [100_u64, 600_000, 200, 900_000];
        let first = replay_domain_policy(&policy, &candidates, &hashes).unwrap();
        let second = replay_domain_policy(&policy, &candidates, &hashes).unwrap();
        assert_eq!(first, second);
        assert!(
            first
                .iter()
                .any(|item| item.selected_domain == domain("mac-cluster"))
        );
        assert!(
            first
                .iter()
                .any(|item| item.selected_domain == domain("mac-single"))
        );
    }
}
