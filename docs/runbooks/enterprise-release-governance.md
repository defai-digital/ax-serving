# Enterprise Release Governance

This runbook defines how enterprise artifacts are built and delivered without
publishing enterprise source code in the public repository.

## Preconditions

An enterprise release may proceed only when:

1. the target public-core version is tagged or frozen
2. the enterprise repository version is set
3. compatibility metadata is prepared
4. release notes and upgrade notes are drafted

## Release Inputs

Every enterprise release should identify:

- target public-core version
- enterprise repository version
- supported previous enterprise release
- contract revisions used by the release
- private artifact destinations

## Validation Gates

Run these checks before publishing:

1. build validation for the enterprise repository
2. integration smoke tests against the target public-core version
3. deployment bundle validation where applicable
4. compatibility metadata validation
5. release note and upgrade note review

## Publication Rules

Enterprise artifacts may be published to:

- private container registries
- private package registries
- internal artifact stores
- customer delivery bundles

Enterprise artifacts must not be published to:

- the public GitHub repository
- public GitHub Releases as source bundles
- public package registries
- public docs that contain proprietary source snapshots

## Required Enterprise Release Outputs

Every enterprise release should produce:

- versioned private artifacts
- compatibility metadata
- release notes
- upgrade notes when applicable
- internal provenance record for the public-core baseline used

## Compatibility Metadata Rule

Use a machine-readable metadata file per enterprise release.

Minimum fields:

- enterprise product name
- enterprise version
- supported public-core range
- contract revision markers
- tested public-core versions
- artifact references

See:

- [enterprise-compatibility-metadata.example.yaml](/Users/akiralam/code/ax-serving/docs/contracts/enterprise-compatibility-metadata.example.yaml)

## Contract Change Rule

If a public contract changes in a way that affects enterprise repositories:

1. publish a contract change note
2. update compatibility metadata if required
3. update the affected enterprise runbook or release note

See:

- [ax-serving-contract-change-template.md](/Users/akiralam/code/ax-serving/docs/contracts/ax-serving-contract-change-template.md)

## Approval Rule

An enterprise release is not complete until:

1. the owning enterprise team approves the release
2. the public-core baseline is recorded
3. artifact destinations are confirmed private
4. customer-facing notes are prepared

## Operational Principle

The public repository remains the source of the open-source core.

Enterprise value is delivered by:

- private repositories
- private artifacts
- private release governance
- stable public contracts
