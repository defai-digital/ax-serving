#!/usr/bin/env bash
# Operate the allow-listed Jetson Thor Edge-LLM hosts by SSH alias.
set -euo pipefail

UV_VERSION="0.12.2"
UV_PYTHON="3.12"

usage() {
  cat <<'EOF'
Usage:
  manage_thor_edge_llm.sh ACTION [--target all|df-thor-01|df-thor-02] [OPTIONS]

Actions:
  system-deps | preflight | install | start | stop | status | qualify-direct

Options:
  --target NAME             Allow-listed SSH alias (default: all).
  --output-dir PATH         Qualification evidence directory.
  --stability-seconds N     Inventory observation window (default: 45).
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
installer="$repo_root/scripts/qualification/runtime/install_tensorrt_edge_llm_thor.sh"
smoke_runner="$repo_root/scripts/qualification/runtime/smoke_openai_runtime.py"

action="${1:-}"
[[ -n "$action" ]] || {
  usage >&2
  exit 2
}
if [[ "$action" == "-h" || "$action" == "--help" ]]; then
  usage
  exit 0
fi
shift

target=all
output_dir="$repo_root/target/runtime-qualification"
stability_seconds=45

while (($#)); do
  case "$1" in
    --target)
      (($# >= 2)) || die "--target requires a value"
      target="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --stability-seconds)
      (($# >= 2)) || die "--stability-seconds requires a value"
      stability_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option $1"
      ;;
  esac
done

case "$action" in
  system-deps|preflight|install|start|stop|status|qualify-direct) ;;
  *) die "unknown action $action" ;;
esac
[[ "$stability_seconds" =~ ^[0-9]+$ ]] ||
  die "--stability-seconds must be an integer"

case "$target" in
  all) targets=(df-thor-01 df-thor-02) ;;
  df-thor-01|df-thor-02) targets=("$target") ;;
  *) die "unsupported target $target; expected all, df-thor-01, or df-thor-02" ;;
esac

expected_hostname() {
  case "$1" in
    df-thor-01) printf 'df-thor-02\n' ;;
    df-thor-02) printf 'df-thor-01\n' ;;
  esac
}

run_remote_action() {
  local alias="$1"
  local expected
  expected="$(expected_hostname "$alias")"
  printf 'target=%s expected_hostname=%s action=%s\n' "$alias" "$expected" "$action"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$alias" \
    "bash -s -- '$action' --expected-hostname '$expected'" <"$installer"
}

qualify_direct() {
  local alias="$1"
  local expected observed remote_output
  expected="$(expected_hostname "$alias")"
  observed="$(
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$alias" hostname
  )"
  [[ "$observed" == "$expected" ]] ||
    die "hostname mismatch: expected $expected but connected to $observed"

  mkdir -p "$output_dir"
  remote_output="/tmp/ax-serving-$alias-edge-direct.json"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$alias" \
    "test \"\$(~/.local/bin/uv --version | awk '{print \$2}')\" = '$UV_VERSION' && \
    ~/.local/bin/uv run --no-project --python '$UV_PYTHON' - \
    --base-url http://127.0.0.1:8000 \
    --model qwen3-edge \
    --stability-seconds \"$stability_seconds\" \
    --requests 8 \
    --concurrency 1 \
    --output \"$remote_output\"" <"$smoke_runner"
  scp -q "$alias:$remote_output" "$output_dir/$alias-edge-direct.json"
}

for alias in "${targets[@]}"; do
  if [[ "$action" == "qualify-direct" ]]; then
    qualify_direct "$alias"
  else
    run_remote_action "$alias"
  fi
done
