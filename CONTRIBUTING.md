# Contributing

Thanks for your interest in AX Serving.

## What We Accept

We welcome:

- bug reports,
- reproducible issue reports,
- security reports through the appropriate private channel,
- feature requests,
- performance benchmark reports.

## Public Code Contributions

We do not currently accept unsolicited public code contributions, pull requests,
or patches.

If you want to propose a code change:

1. open an issue first,
2. describe the problem, reproduction, and expected outcome,
3. wait for explicit written approval from the maintainers before preparing a
   patch.

Without prior written approval, public pull requests may be closed without
review.

## Why

AX Serving uses a dual-license model. To keep licensing, product direction, and
commercial distribution rights clear, code changes are handled only by the core
maintainers or by separately approved contributors under written terms.

The public repository is also intended to remain the open-source serving core.
Changes that introduce private crates, hidden enterprise-only code paths, or
proprietary dependencies into the public workspace are out of bounds unless the
maintainers explicitly approve a documented repository-boundary change.

See:

- [LICENSING.md](LICENSING.md)
- [LICENSE](LICENSE)
- [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)

## Public Workspace Boundary

When proposing public-repository changes, treat these as default rules:

- keep the public Rust workspace self-contained and buildable
- prefer documented REST / gRPC / worker-protocol boundaries over internal
  proprietary module hooks
- do not add private or unpublished workspace members to `crates/*`
- do not rely on hidden Cargo features to carry enterprise-only logic

Enterprise products should normally integrate through separate private
repositories and service boundaries rather than through mixed public/private
workspace code.

Execution documents:

- [docs/contracts/ax-serving-public-contract-inventory.md](docs/contracts/ax-serving-public-contract-inventory.md)
- [docs/runbooks/enterprise-private-repo-bootstrap.md](docs/runbooks/enterprise-private-repo-bootstrap.md)
- [docs/runbooks/enterprise-release-governance.md](docs/runbooks/enterprise-release-governance.md)

## Issue Reports

Please include:

- AX Serving version or commit,
- macOS version and Apple Silicon chip,
- model name and quantization,
- exact command or request,
- expected behavior,
- actual behavior,
- logs or error output,
- minimal reproduction if possible.

For performance issues, include:

- prompt length,
- decode length,
- concurrency,
- backend used,
- release vs debug build,
- thermal context if known.

## Security

Do not post sensitive security details publicly in an issue if public disclosure
would create risk. Report privately through the project's security contact or
commercial support channel.

## Trademarks and Naming

This policy does not grant trademark rights in the AX Serving, AutomatosX, or
DEFAI names, logos, or branding.
