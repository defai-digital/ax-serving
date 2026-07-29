# AX Serving v3.0.0: Apache-2.0 migration

AX Serving v3.0.0 is licensed under the Apache License, Version 2.0. The
license applies to this repository's source, SDKs, protocol definitions,
adapters, documentation, tests, configuration, and packaging unless a file or
subdirectory states otherwise.

This is a source-license and product-boundary change. It does not make AX
Fabric, AX Trust, private services, or separately distributed products part of
this repository or subject to its license.

## Operator changes

The v3 server no longer has a commercial-license activation state:

- `POST /v1/license` has been removed;
- `license:` configuration blocks have been removed;
- `AXS_LICENSE_KEY`, `AXS_LICENSE_FILE`, `AXS_LICENSE_SERVER`, and related
  activation/persistence settings are no longer read;
- startup and admin diagnostics no longer report an edition, entitlement,
  activation, grace period, or license expiry.

`GET /v1/license` remains available without authentication as immutable
build-license metadata. Its v3 response has this shape:

```json
{
  "license": "Apache-2.0",
  "name": "Apache License 2.0",
  "notice": "See NOTICE and LICENSE in the source distribution.",
  "source": "https://github.com/defai-digital/ax-serving"
}
```

Remove old license blocks and activation environment variables from deployment
secrets before upgrading. Clients that called `POST /v1/license` or parsed the
old entitlement response must stop doing so.

## Package versions

The Rust workspace, JavaScript SDK, and Python SDK are versioned `3.0.0`.
Release archives, container images, macOS packages, and SDK packages include
the Apache license and attribution notice.

## Previous releases

Previously distributed AX Serving releases remain governed by the license
terms under which each copy was received. The Apache-2.0 grant applies to the
v3.0.0 source and distributions in this release line.

See [LICENSING.md](../../LICENSING.md) for repository scope and commercial
product boundaries.
