# Separate Product Repository Bootstrap

This runbook defines the minimum boundary between the Apache-2.0 AX Serving
project and separately distributed products or services such as AX Fabric and
AX Trust.

It must not be used to create a private fork of AX Serving, withhold AX Serving
runtime adapters, or reintroduce license-key enforcement into the open-source
gateway. Changes to AX Serving itself should be contributed to this repository.

## Repository boundaries

Separate repositories may contain product-specific:

- AX Fabric orchestration, workflow, and fleet-management services;
- AX Trust identity, policy, attestation, and governance services;
- managed-service operations and customer-specific integrations;
- private deployment overlays, support tooling, and delivery automation.

They should consume a tagged AX Serving release through its documented REST,
SSE, protocol, configuration, deployment, and artifact contracts. They must not
depend on unpublished workspace crates, hidden Cargo features, or private
in-process hooks.

## Repository controls

When a separate product repository is private:

1. enable branch protection and required review;
2. keep CI and customer secrets in the private repository or private runner;
3. publish private artifacts only to approved registries or delivery systems;
4. record the compatible AX Serving version and contract revisions;
5. keep proprietary source and customer data out of AX Serving issues, release
   archives, examples, and documentation;
6. preserve the Apache-2.0 license and notices when redistributing AX Serving.

Each repository should include:

- `README.md`;
- its applicable license or commercial notice;
- `SECURITY.md`;
- `CODEOWNERS`;
- CI definitions;
- compatibility metadata;
- release and upgrade notes.

## Dependency and release rules

Separate products may depend on released AX Serving packages, tagged source
baselines, and documented public contracts. They must not require direct edits
inside the AX Serving repository as a normal build or release step.

Before a customer delivery:

1. record the exact AX Serving release and source provenance;
2. run integration tests against that release;
3. validate compatibility metadata and deployment bundles;
4. prepare release and upgrade notes;
5. confirm that AX Serving license and notice files accompany any
   redistribution;
6. verify that private source and customer material are absent from public
   artifacts.
