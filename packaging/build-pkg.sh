#!/usr/bin/env bash
# packaging/build-pkg.sh — build a signed + notarized macOS .pkg installer
#
# Usage (local signing + notarization):
#   VERSION=2.3.0 \
#   DEVELOPER_ID_INSTALLER="Developer ID Installer: ACME Corp (TEAM1234567)" \
#   APPLE_ID="you@example.com" \
#   APPLE_TEAM_ID="TEAM1234567" \
#   APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
#   ./packaging/build-pkg.sh
#
# Unsigned local build (skip signing + notarization):
#   VERSION=2.3.0 ./packaging/build-pkg.sh
#
# Prerequisites: Xcode Command Line Tools, cargo, Developer ID Installer cert
# in Keychain (for signed builds).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$REPO_ROOT/Cargo.toml" | head -1)}"
IDENTIFIER="digital.defai.ax-serving"
INSTALL_ROOT="/"
PKG_NAME="ax-serving-v${VERSION}.pkg"

STAGING="${REPO_ROOT}/packaging/payload"
DIST_XML="${REPO_ROOT}/packaging/distribution.xml"
RESOURCES="${REPO_ROOT}/target/pkg-resources"

# ── 1. Build release binaries ───────────────────────────────────────────────
echo "==> Building release binaries…"
cd "$REPO_ROOT"
cargo build --release -p ax-serving-cli --features embedded-compat --bins
cargo build --release -p ax-thor-agent --bins
cargo build --release -p ax-dynamo-adapter --bin ax-dynamo-adapter
cargo build --release -p ax-mac-cluster-adapter --bin ax-mac-cluster-adapter

# ── 2. Stage payload ─────────────────────────────────────────────────────────
echo "==> Staging payload…"
rm -rf "$STAGING"
mkdir -p "$STAGING/usr/local/bin"
cp target/release/ax-serving     "$STAGING/usr/local/bin/"
cp target/release/ax-serving-api "$STAGING/usr/local/bin/"
cp target/release/ax-servingctl  "$STAGING/usr/local/bin/"
cp target/release/ax-runtime-agent "$STAGING/usr/local/bin/"
cp target/release/ax-thor-agent "$STAGING/usr/local/bin/"
cp target/release/ax-dynamo-adapter "$STAGING/usr/local/bin/"
cp target/release/ax-mac-cluster-adapter "$STAGING/usr/local/bin/"

# Copy default config to /etc/ax-serving (postinstall script can do this too)
mkdir -p "$STAGING/etc/ax-serving"
cp config/backends.yaml "$STAGING/etc/ax-serving/"
cp config/serving.yaml  "$STAGING/etc/ax-serving/"
cp config/dynamo-adapter.example.env "$STAGING/etc/ax-serving/"
cp config/mac-cluster-adapter.example.env "$STAGING/etc/ax-serving/"
cp config/mac-cluster-manifest.example.json "$STAGING/etc/ax-serving/"
cp config/serving.mac-cluster.example.yaml "$STAGING/etc/ax-serving/"
cp integrations/nvidia/compatibility-manifest.schema.json "$STAGING/etc/ax-serving/"
cp integrations/nvidia/compatibility-manifest.example.json "$STAGING/etc/ax-serving/"

# Copy release docs and license notices.
mkdir -p "$STAGING/usr/local/share/doc/ax-serving"
cp README.md "$STAGING/usr/local/share/doc/ax-serving/"
cp LICENSE "$STAGING/usr/local/share/doc/ax-serving/"
cp NOTICE "$STAGING/usr/local/share/doc/ax-serving/"
cp LICENSING.md "$STAGING/usr/local/share/doc/ax-serving/"
cp TRADEMARKS.md "$STAGING/usr/local/share/doc/ax-serving/"

# ── 3. Build component .pkg ──────────────────────────────────────────────────
echo "==> Running pkgbuild…"
pkgbuild \
  --root "$STAGING" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  --install-location "$INSTALL_ROOT" \
  "packaging/component.pkg"

# ── 4. Build distribution .pkg (adds installer metadata/license) ────────────
echo "==> Running productbuild…"
rm -rf "$RESOURCES"
mkdir -p "$RESOURCES"
cp "$REPO_ROOT/LICENSE" "$RESOURCES/LICENSE"

productbuild \
  --distribution "$DIST_XML" \
  --package-path "packaging" \
  --resources "$RESOURCES" \
  --version "$VERSION" \
  "$PKG_NAME"

rm -f packaging/component.pkg
rm -rf "$RESOURCES"

# ── 5. Sign (if DEVELOPER_ID_INSTALLER is set) ───────────────────────────────
if [[ -n "${DEVELOPER_ID_INSTALLER:-}" ]]; then
  echo "==> Signing with: $DEVELOPER_ID_INSTALLER"
  SIGNED_PKG="ax-serving-v${VERSION}-signed.pkg"
  productsign \
    --sign "$DEVELOPER_ID_INSTALLER" \
    "$PKG_NAME" \
    "$SIGNED_PKG"
  mv "$SIGNED_PKG" "$PKG_NAME"
  pkgutil --check-signature "$PKG_NAME"
fi

# ── 6. Notarize (if Apple credentials are set) ───────────────────────────────
if [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]]; then
  echo "==> Submitting for notarization…"
  xcrun notarytool submit "$PKG_NAME" \
    --apple-id "$APPLE_ID" \
    --team-id  "$APPLE_TEAM_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --wait

  echo "==> Stapling notarization ticket…"
  xcrun stapler staple "$PKG_NAME"
fi

echo "==> Done: $PKG_NAME"
