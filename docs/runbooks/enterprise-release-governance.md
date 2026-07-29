# Separate Product Release Governance

This runbook governs releases of products and services that integrate with the
Apache-2.0 AX Serving project, including AX Fabric and AX Trust.

AX Serving remains an independently releasable open-source project. A separate
product may add value around it, but must not claim that private entitlement is
required to use, modify, redistribute, or operate AX Serving under Apache-2.0.

## Preconditions

A separate-product release may proceed only when:

1. the target AX Serving version is tagged or frozen;
2. the product version and applicable commercial terms are set;
3. compatibility metadata identifies the AX Serving and contract versions;
4. integration, release, and upgrade notes are ready;
5. third-party notices and AX Serving attribution are included where required.

## Validation gates

Run:

1. product build and test validation;
2. integration smoke tests against the recorded AX Serving release;
3. deployment-bundle validation;
4. compatibility-metadata validation;
5. source-provenance and notice review;
6. a check that no private source or customer data entered public artifacts.

Private products may be delivered through private registries, internal artifact
stores, managed services, or customer bundles. Do not publish their proprietary
source in AX Serving release archives or public documentation.

## Required release outputs

Every release should produce:

- versioned product artifacts;
- compatibility metadata;
- release notes and, when applicable, upgrade notes;
- provenance for the AX Serving baseline;
- the Apache-2.0 license and notices when AX Serving is redistributed.

Use
[enterprise-compatibility-metadata.example.yaml](../contracts/enterprise-compatibility-metadata.example.yaml)
as the machine-readable starting point. Contract changes should follow the
[contract change template](../contracts/ax-serving-contract-change-template.md).

## Operational principle

AX Serving provides the open inference-federation foundation. AX Fabric, AX
Trust, managed operations, support, certification, indemnification, and
customer-specific integration may be offered separately through stable public
boundaries. Separate product terms do not narrow the Apache-2.0 rights granted
for AX Serving.
