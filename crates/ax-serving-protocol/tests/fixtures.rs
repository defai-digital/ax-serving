use ax_serving_protocol::{CURRENT_PROTOCOL, RegisterWorkerRequest};

#[test]
fn canonical_v1_registration_fixture_decodes_and_validates() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-worker.json")).unwrap();

    assert_eq!(request.protocol.version, CURRENT_PROTOCOL);
    assert_eq!(request.worker.id.as_str(), "mac-mini-07");
    assert_eq!(request.runtime.kind, "ax_engine");
    assert!(request.observation.runtime.ready);
    request.observation.validate().unwrap();
}

#[test]
fn canonical_fixture_round_trip_preserves_unknown_capabilities() {
    let request: RegisterWorkerRequest =
        serde_json::from_str(include_str!("fixtures/v1/register-worker.json")).unwrap();
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: RegisterWorkerRequest = serde_json::from_slice(&encoded).unwrap();

    assert_eq!(decoded, request);
}
