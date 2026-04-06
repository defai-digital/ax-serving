# Enterprise Private Repository Bootstrap

This runbook defines the minimum setup for any private AX Serving enterprise
repository.

It exists to keep enterprise source code private by default and to stop private
deliverables from leaking back into the public workspace.

## Target Repositories

Bootstrap these repositories as private repositories:

- `ax-serving-enterprise-control-plane`
- `ax-serving-enterprise-workers`
- `ax-serving-enterprise-deploy`
- `ax-serving-enterprise-connectors`

## Repository Rules

Every enterprise repository must start with these controls:

1. repository visibility is private
2. branch protection is enabled on the default branch
3. CODEOWNERS requires review from the owning enterprise team
4. CI secrets are stored only in the private repository or private runner
5. releases publish only to private registries or private artifact stores
6. public GitHub release notes must not include enterprise source bundles
7. public repository submodules or mirrors must not contain enterprise source

## Bootstrap Checklist

### 1. Repository Creation

- create the repository as private
- enable signed commits if required by commercial operations policy
- add maintainers and enterprise-team reviewers
- add branch protection and required status checks

### 2. Baseline Files

Each enterprise repository should start with:

- `README.md`
- `LICENSE.txt` or internal commercial notice
- `SECURITY.md`
- `CODEOWNERS`
- CI workflow definitions
- release checklist or release automation entrypoint

### 3. CI/CD Baseline

Each enterprise repository must define:

- build validation
- integration smoke tests against a tagged or frozen public-core release
- artifact publishing to a private registry
- compatibility metadata generation

### 4. Dependency Rules

Private repositories may depend on:

- released public crates if intentionally published
- tagged public-core source baselines under internal build policy
- documented public APIs and protocols

Private repositories must not depend on:

- unpublished local branches from the public repo
- hidden Cargo features in the public workspace
- direct edits inside the public repository as a normal release step

## Artifact And Registry Rules

Use private distribution channels only:

- private container registry
- private package registry
- internal artifact storage
- customer delivery portal or signed bundle exchange

Do not publish enterprise source archives to:

- the public GitHub repository
- public GitHub Releases
- public package registries
- public documentation examples that embed proprietary source

## Suggested Repository Boundaries

### `ax-serving-enterprise-control-plane`

- enterprise auth
- RBAC / tenant / org policy services
- governance workflows
- private management UI

### `ax-serving-enterprise-workers`

- NVIDIA / Thor-class worker binaries
- accelerator runtime adapters
- private worker packaging

### `ax-serving-enterprise-deploy`

- Helm / Terraform / operator assets
- air-gapped bundles
- installers and entitlement activation flows

### `ax-serving-enterprise-connectors`

- SIEM
- KMS / secret manager integration
- customer-specific but reusable adapters

## First Release Exit Criteria

Before the first customer delivery from a private repository:

1. compatibility metadata exists
2. release notes exist
3. upgrade notes exist if replacing a previous private release
4. source publication path is confirmed private only
5. integration tests have passed against a public-core baseline
