#!/usr/bin/env bash
# packaging/build-pkg.sh — build a signed + notarized macOS .pkg installer
#
# Usage (local signing + notarization):
#   VERSION=0.2.0 \
#   DEVELOPER_ID_INSTALLER="Developer ID Installer: ACME Corp (TEAM1234567)" \
#   APPLE_ID="you@example.com" \
#   APPLE_TEAM_ID="TEAM1234567" \
#   APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
#   ./packaging/build-pkg.sh
#
# Unsigned local build (skip signing + notarization):
#   VERSION=0.2.0 ./packaging/build-pkg.sh
#
# Prerequisites: Xcode Command Line Tools, cargo, Developer ID Installer cert
# in Keychain (for signed builds).

set -euo pipefail

VERSION="${VERSION:-0.1.0}"
IDENTIFIER="com.automatosx.ax-serving"
INSTALL_ROOT="/"
PKG_NAME="ax-serving-v${VERSION}.pkg"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="${REPO_ROOT}/packaging/payload"
DIST_XML="${REPO_ROOT}/packaging/distribution.xml"

# ── 1. Build release binaries ───────────────────────────────────────────────
echo "==> Building release binaries…"
cd "$REPO_ROOT"
cargo build --workspace --release

# ── 2. Stage payload ─────────────────────────────────────────────────────────
echo "==> Staging payload…"
rm -rf "$STAGING"
mkdir -p "$STAGING/usr/local/bin"
cp target/release/ax-serving     "$STAGING/usr/local/bin/"
cp target/release/ax-serving-api "$STAGING/usr/local/bin/"

# Copy default config to /etc/ax-serving (postinstall script can do this too)
mkdir -p "$STAGING/etc/ax-serving"
cp config/backends.yaml "$STAGING/etc/ax-serving/"
cp config/serving.yaml  "$STAGING/etc/ax-serving/"

# ── 3. Build component .pkg ──────────────────────────────────────────────────
echo "==> Running pkgbuild…"
pkgbuild \
  --root "$STAGING" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  --install-location "$INSTALL_ROOT" \
  "packaging/component.pkg"

# ── 4. Build distribution .pkg (adds welcome/readme screens) ────────────────
echo "==> Running productbuild…"
productbuild \
  --distribution "$DIST_XML" \
  --package-path "packaging" \
  --version "$VERSION" \
  "$PKG_NAME"

rm -f packaging/component.pkg

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
