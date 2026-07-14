#!/usr/bin/env bash
# Structural chart checks: multi-replica must not run on memory fleet store.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HELM="${HELM:-helm}"
failed=0

run_expect_fail() {
  local name="$1"
  shift
  if "$HELM" template test "$ROOT" "$@" >/dev/null 2>"${TMPDIR:-/tmp}/helm-err.$$"; then
    echo "FAIL: $name — expected helm template to fail"
    failed=1
  else
    if grep -Eq 'fleet_store|redis.existingSecret|replicaCount' "${TMPDIR:-/tmp}/helm-err.$$"; then
      echo "OK: $name"
    else
      echo "FAIL: $name — failed for unexpected reason:"
      cat "${TMPDIR:-/tmp}/helm-err.$$"
      failed=1
    fi
  fi
  rm -f "${TMPDIR:-/tmp}/helm-err.$$"
}

run_expect_ok() {
  local name="$1"
  shift
  if "$HELM" template test "$ROOT" "$@" >/dev/null 2>&1; then
    echo "OK: $name"
  else
    echo "FAIL: $name — expected success"
    "$HELM" template test "$ROOT" "$@" 2>&1 | head -20
    failed=1
  fi
}

# Visible inline config: memory store is never OK for multi-replica, even with redis secret.
run_expect_fail "replicas=2 + fleet_store=memory + redis.existingSecret" \
  --set gateway.replicaCount=2 \
  --set config.inline.orchestrator.fleet_store=memory \
  --set redis.existingSecret=fake-redis

run_expect_fail "replicas=2 + default memory (no redis secret)" \
  --set gateway.replicaCount=2

# Visible inline redis is OK (secret optional at chart layer; operator still needs URL).
run_expect_ok "replicas=2 + fleet_store=redis" \
  --set gateway.replicaCount=2 \
  --set config.inline.orchestrator.fleet_store=redis

# Opaque configMap: require redis.existingSecret.
run_expect_fail "replicas=2 + existingConfigMap without redis secret" \
  --set gateway.replicaCount=2 \
  --set config.existingConfigMap=external-config

run_expect_ok "replicas=2 + existingConfigMap + redis.existingSecret" \
  --set gateway.replicaCount=2 \
  --set config.existingConfigMap=external-config \
  --set redis.existingSecret=fake-redis

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "all multi-replica store checks passed"
