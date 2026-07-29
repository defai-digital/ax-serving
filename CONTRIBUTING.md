# Contributing

Thanks for helping improve AX Serving.

## Before starting

For bug fixes and small documentation improvements, open a focused pull
request. For new features, protocol changes, architectural changes, or
substantial refactors, open an issue first and wait for maintainer agreement
on scope.

Public contributions must:

- keep the Rust workspace and affected SDKs buildable;
- preserve the runtime-SDK-free portable gateway boundary;
- use mock backends unless hardware is essential to the test;
- include tests and documentation appropriate to the behavior change;
- avoid secrets, private infrastructure details, and sensitive vulnerability
  information;
- follow the repository's Conventional Commit and pull-request guidance.

## Developer Certificate of Origin

Every commit in a contribution must include a `Signed-off-by` trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Add it with `git commit -s`. By signing off, you certify that you have the
right to submit the contribution under the repository's license and the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).

Contributions accepted into AX Serving are licensed under Apache-2.0 in
accordance with section 5 of [LICENSE](LICENSE), unless a separate written
contributor agreement applies.

## Validation

Run the narrowest relevant tests, then the portable baseline where practical:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --tests -- -D warnings
cargo test --workspace --lib
AXS_ALLOW_NO_AUTH=true cargo test -p ax-serving-api --test orchestration
```

Hardware-dependent tests require their pinned environment. A skipped hardware
test is not support evidence.

JavaScript SDK changes:

```bash
cd sdk/javascript
npm ci
npm test
```

Python SDK changes:

```bash
python -m pip install -e "sdk/python[dev]"
pytest sdk/python/tests
```

## Pull requests

Pull requests should:

1. explain the problem and scope;
2. link the approved issue when one was required;
3. list validation commands and results;
4. include compatibility, rollout, or benchmark evidence when behavior or
   performance changes;
5. preserve third-party license and attribution notices.

Maintainers may request a contributor agreement for substantial, corporate, or
provenance-sensitive contributions.

## Public workspace boundary

Keep the public workspace self-contained. Prefer documented REST, protocol, and
artifact boundaries over private in-process hooks. Do not add unpublished
workspace dependencies, hidden enterprise-only features, or proprietary
runtime dependencies to the portable gateway.

Related contracts:

- [Public contract inventory](docs/contracts/ax-serving-public-contract-inventory.md)
- [Contract change template](docs/contracts/ax-serving-contract-change-template.md)
- [AX Fabric runtime contract](docs/contracts/ax-fabric-runtime-contract.md)

## Security

Do not open a public issue for a vulnerability when disclosure would create
risk. Use the private security channel described by the repository or contact
the maintainers directly.

## Trademarks

Contributing does not grant rights to AX Serving, AutomatosX, DEFAI, or related
names, logos, and marks. See [TRADEMARKS.md](TRADEMARKS.md).
