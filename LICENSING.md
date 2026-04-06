# Licensing

AX Serving is offered under a dual-license model.

AX Serving is the serving and orchestration layer in the AutomatosX product
family. It can be used as open-source software under AGPL, or licensed
commercially for proprietary internal deployments, embedded use, managed
inference fleets, and custom enterprise integrations.

The public repository is the open-source core of AX Serving. The default
open-source deployment focus is single-Mac serving and Mac-grid serving.
Commercial offerings may license the same core under non-AGPL terms and may
also include separate enterprise modules, deployment bundles, and supported
integrations for heterogeneous fleets such as NVIDIA / Thor-class workers.

## 1. Open-Source License

Unless you have a separate written commercial agreement with DEFAI Private
Limited / AutomatosX, use of this repository is governed by the GNU Affero
General Public License, version 3 or any later version.

- Full license text: [LICENSE](LICENSE)
- SPDX identifier: `AGPL-3.0-or-later`

Use this path when you want to operate AX Serving under open-source copyleft
terms.

## 2. Commercial License

A separate commercial license is available for organizations that want to use
AX Serving without AGPL obligations.

Typical commercial-license scenarios include:

- proprietary internal deployments
- private or air-gapped inference and orchestration backends
- embedding AX Serving into commercial software, appliances, or internal platforms
- distributing systems or managed fleets that include AX Serving
- operating heterogeneous fleets that include NVIDIA / Thor-class workers
- OEM redistribution, custom integrations, and commercial support agreements

Commercial offerings may also include separate proprietary services and
integration layers built around the open-source core, such as:

- enterprise auth, governance, and deployment tooling
- supported accelerator-worker integrations
- private fleet operations tooling and packaging
- private connectors and compliance-oriented extensions

The commercial license is provided by separate written agreement. A summary is
available in [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

- Contact: `enquiry@automatosx.com`

## 3. Packaging Metadata

Workspace and package metadata may advertise the open-source leg as
`AGPL-3.0-or-later` for compatibility with Cargo, npm, Python packaging, and
other ecosystem tooling. That metadata does not remove the availability of
separate commercial licensing from the copyright holder.

## 4. Public Contributions

This repository does not currently accept unsolicited public code
contributions.

Public issue reports are welcome. Code changes are handled by the maintainers
or by separately approved contributors under written terms.

See:

- [CONTRIBUTING.md](CONTRIBUTING.md)

## 5. Deployment Reminder

If you modify AX Serving and make it available for remote network interaction,
the AGPL has additional source-availability requirements. Review section 13 of
the AGPL carefully before deploying modified hosted versions.

## 6. Architectural Boundary Guidance

The recommended boundary between the open-source core and proprietary
enterprise components is a service or deployment boundary, not a private crate
inside the public workspace.

Recommended:

- separate private repositories
- REST / gRPC / worker protocol integration
- private deployment bundles, management planes, and connectors

Avoid when possible:

- private crates mixed into the public Rust workspace
- in-process enterprise plugins linked into the public binaries
- proprietary code that depends on undocumented internal Rust module contracts

Supporting execution documents:

- [docs/contracts/ax-serving-public-contract-inventory.md](docs/contracts/ax-serving-public-contract-inventory.md)
- [docs/contracts/ax-serving-contract-change-template.md](docs/contracts/ax-serving-contract-change-template.md)
- [docs/contracts/enterprise-compatibility-metadata.example.yaml](docs/contracts/enterprise-compatibility-metadata.example.yaml)
- [docs/runbooks/enterprise-private-repo-bootstrap.md](docs/runbooks/enterprise-private-repo-bootstrap.md)
- [docs/runbooks/enterprise-release-governance.md](docs/runbooks/enterprise-release-governance.md)
