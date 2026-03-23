# Release Runbook

This repository uses SemVer tags and GitHub Releases.

## Release Types

- Stable: `v1.2.3`
- Prerelease / beta: `v2.0.0-beta.1`

GitHub does not have a separate "beta channel" object. The standard pattern is:

- publish SemVer prerelease tags
- mark those GitHub releases as prereleases
- keep stable install paths pointed at stable releases only

In this repo that means:

- prerelease tags become GitHub prereleases
- prereleases do not update the Homebrew tap
- prerelease npm publishes use the `beta` dist-tag instead of `latest`

## Version Rules

Before tagging any release:

1. Update the Rust workspace version in [Cargo.toml](/Users/akiralam/code/ax-serving/Cargo.toml).
2. Update the JS SDK version in [package.json](/Users/akiralam/code/ax-serving/sdk/javascript/package.json) if the SDK is being published.
3. Ensure the tag exactly matches those versions, without the leading `v`.

Examples:

- `Cargo.toml` version `2.0.0-beta.1` -> tag `v2.0.0-beta.1`
- `Cargo.toml` version `2.0.0` -> tag `v2.0.0`

The release workflow validates these version matches and fails fast on mismatch.

## Stable Release Process

1. Land the release commit on the intended branch.
2. Set versions to the stable version, for example `2.0.0`.
3. Run the normal validation gates.
4. Create and push the tag:

```bash
git tag v2.0.0
git push origin v2.0.0
```

Effects:

- GitHub Release is published as a normal release.
- Homebrew tap is updated.
- npm publish uses the default `latest` dist-tag.

## Beta Release Process

1. Land the beta candidate commit on the intended branch.
2. Set versions to the prerelease version, for example `2.0.0-beta.1`.
3. Run the normal validation gates.
4. Create and push the tag:

```bash
git tag v2.0.0-beta.1
git push origin v2.0.0-beta.1
```

Effects:

- GitHub Release is published as a prerelease.
- Homebrew tap is not updated.
- npm publish uses the `beta` dist-tag.

## Notes

- Do not use a bare `v2.0-beta1` style tag. Use valid SemVer prerelease formatting like `v2.0.0-beta.1`.
- Do not publish a beta by reusing the stable version number.
- If multiple betas are needed, increment the prerelease suffix: `beta.1`, `beta.2`, `beta.3`.
