#!/usr/bin/env bash
# Contract smoke for one AX Serving -> Dynamo domain path.
#
# Required:
#   AXS_DYNAMO_MANIFEST_PATH=/immutable/manifest.json
#   AXS_DYNAMO_DOMAIN_KIND=nvidia_dynamo_pc|nvidia_dynamo_thor
#   MODEL_ID=<logical AX Serving model id>
#
# Optional:
#   AXS_URL=http://127.0.0.1:18080
#   AXS_API_KEY=<public gateway credential>
#   AX_DYNAMO_ADAPTER_BIN=target/debug/ax-dynamo-adapter
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

AXS_URL="${AXS_URL:-http://127.0.0.1:18080}"
MODEL_ID="${MODEL_ID:?set MODEL_ID to the logical AX Serving model id}"
MANIFEST="${AXS_DYNAMO_MANIFEST_PATH:?set AXS_DYNAMO_MANIFEST_PATH}"
DOMAIN_KIND="${AXS_DYNAMO_DOMAIN_KIND:?set AXS_DYNAMO_DOMAIN_KIND}"
ADAPTER_BIN="${AX_DYNAMO_ADAPTER_BIN:-target/debug/ax-dynamo-adapter}"

case "$DOMAIN_KIND" in
  nvidia_dynamo_pc) DOMAIN_KIND_ARG="nvidia-dynamo-pc" ;;
  nvidia_dynamo_thor) DOMAIN_KIND_ARG="nvidia-dynamo-thor" ;;
  *)
    echo "invalid AXS_DYNAMO_DOMAIN_KIND: $DOMAIN_KIND" >&2
    exit 2
    ;;
esac

if [[ ! -x "$ADAPTER_BIN" ]]; then
  cargo build -p ax-dynamo-adapter --quiet
fi

AUTH_ARGS=()
if [[ -n "${AXS_API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${AXS_API_KEY}")
fi

models_file="$(mktemp)"
chat_file="$(mktemp)"
stream_file="$(mktemp)"
cleanup() {
  rm -f "$models_file" "$chat_file" "$stream_file"
}
trap cleanup EXIT

echo "==> validate immutable Dynamo compatibility manifest"
"$ADAPTER_BIN" check-manifest \
  --manifest "$MANIFEST" \
  --domain-kind "$DOMAIN_KIND_ARG"

echo "==> verify logical model inventory"
curl --fail --silent --show-error --retry 0 \
  "${AUTH_ARGS[@]}" \
  "${AXS_URL%/}/v1/models" >"$models_file"
python3 - "$models_file" "$MODEL_ID" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = sys.argv[2]
ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
if expected not in ids:
    raise SystemExit(f"logical model {expected!r} not present: {sorted(ids)}")
PY

echo "==> non-streaming chat through the selected domain"
curl --fail --silent --show-error --retry 0 \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":false}" \
  "${AXS_URL%/}/v1/chat/completions" >"$chat_file"
python3 - "$chat_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choices = payload.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit("chat response has no choices")
PY

echo "==> SSE streaming through the same logical deployment"
curl --fail --silent --show-error --no-buffer --retry 0 \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}" \
  "${AXS_URL%/}/v1/chat/completions" >"$stream_file"
if ! grep -q 'data: \[DONE\]' "$stream_file"; then
  echo "stream did not terminate with [DONE]" >&2
  exit 1
fi

echo "DYNAMO_DOMAIN_SMOKE_OK"
