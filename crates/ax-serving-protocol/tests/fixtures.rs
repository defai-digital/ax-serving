use ax_serving_protocol::{
    CURRENT_PROTOCOL, DomainContractError, ExecutionDomainKind, ProtocolVersion,
    RegisterWorkerRequest,
};

#[test]
fn canonical_v1_registration_fixture_decodes_and_validates() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-worker.json")).unwrap();

    assert_eq!(
        request.protocol.version,
        ProtocolVersion { major: 1, minor: 0 }
    );
    assert_eq!(request.worker.id.as_str(), "mac-mini-07");
    assert_eq!(request.runtime.kind, "ax_engine");
    assert!(request.observation.runtime.ready);
    request.observation.validate().unwrap();
    request.validate_domain_contract().unwrap();
}

#[test]
fn canonical_fixture_round_trip_preserves_unknown_capabilities() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-worker.json")).unwrap();
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: RegisterWorkerRequest = serde_json::from_slice(&encoded).unwrap();

    assert_eq!(decoded, request);
}

#[test]
fn canonical_v1_1_domain_registration_decodes_and_validates() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-domain.json")).unwrap();

    assert_eq!(
        request.protocol.version,
        ProtocolVersion { major: 1, minor: 1 }
    );
    assert_eq!(
        request.domain.as_ref().unwrap().kind,
        ExecutionDomainKind::NvidiaDynamoPc
    );
    request.observation.validate().unwrap();
    request.validate_domain_contract().unwrap();
}

#[test]
fn canonical_v1_2_mac_cluster_registration_decodes_and_validates() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-mac-cluster.json")).unwrap();

    assert_eq!(request.protocol.version, CURRENT_PROTOCOL);
    assert_eq!(
        request.domain.as_ref().unwrap().kind,
        ExecutionDomainKind::MacAxEngineCluster
    );
    request.observation.validate().unwrap();
    request.validate_domain_contract().unwrap();
}

#[test]
fn mac_cluster_registration_requires_the_v1_2_capability() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/v1/register-mac-cluster.json")).unwrap();
    value["protocol"]["capabilities"]
        .as_array_mut()
        .unwrap()
        .retain(|capability| capability != "control.mac-cluster.v1");
    let request: RegisterWorkerRequest = serde_json::from_value(value).unwrap();

    assert_eq!(
        request.validate_domain_contract(),
        Err(DomainContractError::MacClusterDescriptorWithoutCapability)
    );
}

#[test]
fn mac_cluster_registration_rejects_protocol_v1_1() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/v1/register-mac-cluster.json")).unwrap();
    value["protocol"]["version"]["minor"] = serde_json::json!(1);
    let request: RegisterWorkerRequest = serde_json::from_value(value).unwrap();

    assert_eq!(
        request.validate_domain_contract(),
        Err(DomainContractError::MacClusterProtocolMinorTooOld)
    );
}

#[test]
fn mac_cluster_capability_cannot_describe_a_node_domain() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/v1/register-mac-cluster.json")).unwrap();
    value["domain"]["kind"] = serde_json::json!("mac_ax_engine");
    value["domain"]["endpoint_scope"] = serde_json::json!("node");
    let request: RegisterWorkerRequest = serde_json::from_value(value).unwrap();

    assert_eq!(
        request.validate_domain_contract(),
        Err(DomainContractError::MacClusterCapabilityForOtherDomain)
    );
}

#[test]
fn v1_0_fixture_tolerates_absent_v1_1_domain_fields() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-worker.json")).unwrap();

    assert!(request.domain.is_none());
    assert!(request.domain_observation.is_none());
    request.validate_domain_contract().unwrap();
}

#[test]
fn domain_observation_cannot_introduce_an_unregistered_manifest() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/v1/register-domain.json")).unwrap();
    value["domain"]
        .as_object_mut()
        .unwrap()
        .remove("compatibility_manifest");
    let request: RegisterWorkerRequest = serde_json::from_value(value).unwrap();

    assert_eq!(
        request.validate_domain_contract(),
        Err(DomainContractError::ManifestMismatch)
    );
}
