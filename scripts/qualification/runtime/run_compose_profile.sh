#!/usr/bin/env bash
# Reproducible compatibility evaluation for one allow-listed NVIDIA runtime.
set -euo pipefail

UV_VERSION="0.12.2"
UV_PYTHON="3.12"

usage() {
  cat <<'EOF'
Usage:
  run_compose_profile.sh ACTION --runtime RUNTIME [OPTIONS]

Actions:
  plan       Validate pins and print a secret-free execution plan.
  validate   Validate pins and render the Compose configuration.
  up         Start the runtime, gateway, Redis, and runtime agent.
  qualify    Qualify already-running direct and gateway endpoints.
  test       Start, qualify both paths, and clean up.
  down       Stop the selected Compose project.

Runtimes:
  vllm | sglang | tensorrt_llm

Options:
  --env-file PATH          Operator-owned environment file.
  --output-dir PATH        Evidence directory (default: target/runtime-qualification).
  --stability-seconds N    Inventory observation window (default: 45).
  --wait-seconds N         Startup deadline (default: 900).
  --keep                   Keep the stack after ACTION=test.
  --allow-concurrent       Permit another labeled runtime profile to be active.
  --static-only            Skip Docker Compose rendering for ACTION=validate.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
base_compose="$repo_root/deploy/compose/compose.yaml"
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

runtime=""
env_file=""
output_dir="$repo_root/target/runtime-qualification"
stability_seconds=45
wait_seconds=900
keep=false
allow_concurrent=false
static_only=false
uv_bin=""

while (($#)); do
  case "$1" in
    --runtime)
      (($# >= 2)) || die "--runtime requires a value"
      runtime="$2"
      shift 2
      ;;
    --env-file)
      (($# >= 2)) || die "--env-file requires a value"
      env_file="$2"
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
    --wait-seconds)
      (($# >= 2)) || die "--wait-seconds requires a value"
      wait_seconds="$2"
      shift 2
      ;;
    --keep)
      keep=true
      shift
      ;;
    --allow-concurrent)
      allow_concurrent=true
      shift
      ;;
    --static-only)
      static_only=true
      shift
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
  plan|validate|up|qualify|test|down) ;;
  *) die "unknown action $action" ;;
esac

case "$runtime" in
  vllm)
    overlay="$repo_root/deploy/compose/vllm.compose.example.yaml"
    default_env="$repo_root/deploy/compose/vllm.env.example"
    compose_profile="vllm"
    runtime_service="vllm"
    image_key="VLLM_IMAGE"
    model_key="VLLM_SERVED_MODEL"
    revision_key="VLLM_MODEL_REVISION"
    port_key="VLLM_HOST_PORT"
    default_port=8001
    ;;
  sglang)
    overlay="$repo_root/deploy/compose/sglang.compose.example.yaml"
    default_env="$repo_root/deploy/compose/sglang.env.example"
    compose_profile="sglang"
    runtime_service="sglang"
    image_key="SGLANG_IMAGE"
    model_key="SGLANG_SERVED_MODEL"
    revision_key="SGLANG_MODEL_REVISION"
    port_key="SGLANG_HOST_PORT"
    default_port=8002
    ;;
  tensorrt_llm)
    overlay="$repo_root/deploy/compose/tensorrt-llm.compose.example.yaml"
    default_env="$repo_root/deploy/compose/tensorrt-llm.env.example"
    compose_profile="tensorrt-llm"
    runtime_service="tensorrt-llm"
    image_key="TRTLLM_IMAGE"
    model_key="TRTLLM_SERVED_MODEL"
    revision_key="TRTLLM_MODEL_REVISION"
    port_key="TRTLLM_HOST_PORT"
    default_port=8000
    ;;
  *)
    die "unsupported runtime ${runtime:-<empty>}; expected vllm, sglang, or tensorrt_llm"
    ;;
esac

env_file="${env_file:-$default_env}"
[[ -f "$env_file" ]] || die "environment file not found: $env_file"
[[ -r "$env_file" ]] || die "environment file is not readable: $env_file"
[[ -f "$overlay" ]] || die "Compose overlay not found: $overlay"

env_value() {
  local key="$1"
  awk -v wanted="$key" '
    /^[[:space:]]*(#|$)/ { next }
    {
      separator = index($0, "=")
      if (!separator) next
      key = substr($0, 1, separator - 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key == wanted) {
        value = substr($0, separator + 1)
        sub(/\r$/, "", value)
        print value
        exit
      }
    }
  ' "$env_file"
}

require_safe_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer"
}

validate_static() {
  local image model revision configured_runtime
  image="$(env_value "$image_key")"
  model="$(env_value "$model_key")"
  revision="$(env_value "$revision_key")"
  configured_runtime="$(env_value AXS_NODE_RUNTIME)"

  [[ "$image" =~ @sha256:[0-9a-f]{64}$ ]] ||
    die "$image_key must end in an immutable sha256 digest"
  [[ -n "$model" ]] || die "$model_key must not be empty"
  [[ "$revision" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] ||
    die "$revision_key must be an exact 40- or 64-character hexadecimal revision"
  [[ "$configured_runtime" == "$runtime" ]] ||
    die "AXS_NODE_RUNTIME must be $runtime"
  require_safe_integer "--stability-seconds" "$stability_seconds"
  require_safe_integer "--wait-seconds" "$wait_seconds"
}

compose=()

compose_version_is_supported() {
  local version="$1"
  local major
  version="${version##*$'\n'}"
  version="${version##* }"
  version="${version#v}"
  major="${version%%.*}"
  [[ "$major" =~ ^[0-9]+$ ]] && ((10#$major >= 2))
}

select_compose_command() {
  ((${#compose[@]} == 0)) || return

  local version=""
  if command -v docker >/dev/null 2>&1 &&
    version="$(docker compose version --short 2>/dev/null)" &&
    compose_version_is_supported "$version"; then
    compose=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1 &&
    version="$(docker-compose version --short 2>/dev/null)" &&
    compose_version_is_supported "$version"; then
    compose=(docker-compose)
  else
    die "Docker Compose v2 or newer is required (docker compose or docker-compose)"
  fi

  compose+=(
    --env-file "$env_file"
    -f "$base_compose"
    -f "$overlay"
    --profile agent
    --profile "$compose_profile"
  )
}

validate_compose() {
  select_compose_command
  "${compose[@]}" config --quiet
}

require_runtime_credentials() {
  local dispatch worker
  dispatch="$(env_value AXS_DISPATCH_TOKEN)"
  worker="$(env_value AXS_WORKER_TOKEN)"
  [[ -n "$dispatch" && "$dispatch" != replace-this-* ]] ||
    die "replace AXS_DISPATCH_TOKEN in an operator-owned environment file"
  [[ -n "$worker" && "$worker" != replace-this-* ]] ||
    die "replace AXS_WORKER_TOKEN in an operator-owned environment file"
}

check_profile_conflicts() {
  $allow_concurrent && return
  local observed
  while IFS= read -r observed; do
    [[ -z "$observed" || "$observed" == "$runtime" ]] && continue
    die "runtime profile $observed is already active; stop it or pass --allow-concurrent"
  done < <(
    docker ps \
      --filter label=com.automatosx.ax-serving.runtime-profile \
      --format '{{.Label "com.automatosx.ax-serving.runtime-profile"}}'
  )
}

check_port_conflicts() {
  local project host_port port observed
  project="$(env_value COMPOSE_PROJECT_NAME)"
  [[ -n "$project" ]] || die "COMPOSE_PROJECT_NAME must not be empty"
  host_port="$(env_value "$port_key")"
  host_port="${host_port:-$default_port}"
  require_safe_integer "$port_key" "$host_port"
  for port in 18080 19090 "$host_port"; do
    while IFS= read -r observed; do
      [[ -z "$observed" || "$observed" == "$project" ]] && continue
      die "host port $port is owned by Compose project $observed"
    done < <(
      docker ps \
        --filter "publish=$port" \
        --format '{{.Label "com.docker.compose.project"}}'
    )
  done
}

print_plan() {
  printf 'runtime=%s\n' "$runtime"
  printf 'compose_profile=%s\n' "$compose_profile"
  printf 'runtime_service=%s\n' "$runtime_service"
  printf 'environment_file=%s\n' "$env_file"
  printf 'image_pin=immutable-sha256\n'
  printf 'model_revision=immutable\n'
  printf 'qualification=direct-and-gateway\n'
  printf 'evidence_directory=%s\n' "$output_dir"
}

require_uv() {
  if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
  else
    die "uv is required (expected on PATH or at ~/.local/bin/uv)"
  fi
  [[ "$("$uv_bin" --version | awk '{print $2}')" == "$UV_VERSION" ]] ||
    die "uv $UV_VERSION is required"
}

wait_for_model() {
  local base_url="$1"
  local model="$2"
  require_uv
  "$uv_bin" run --no-project --python "$UV_PYTHON" python - \
    "$base_url" "$model" "$wait_seconds" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

base_url, expected, timeout_raw = sys.argv[1:]
deadline = time.monotonic() + int(timeout_raw)
url = f"{base_url.rstrip('/')}/v1/models"
last_error = "endpoint not ready"
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
        identities = {
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, dict)
        }
        if expected in identities:
            print(f"ready model={expected} endpoint={base_url}")
            raise SystemExit(0)
        last_error = f"advertised models did not include {expected!r}"
    except (OSError, ValueError, urllib.error.URLError) as error:
        last_error = str(error).replace("\n", " ")[:240]
    time.sleep(2)
raise SystemExit(f"model readiness timed out for {base_url}: {last_error}")
PY
}

start_stack() {
  require_runtime_credentials
  validate_compose
  check_profile_conflicts
  check_port_conflicts
  "${compose[@]}" up -d --build redis gateway "$runtime_service" runtime-agent
}

stop_stack() {
  validate_compose
  "${compose[@]}" down --remove-orphans
}

qualify_stack() {
  local model host_port direct_url gateway_url
  model="$(env_value "$model_key")"
  host_port="$(env_value "$port_key")"
  host_port="${host_port:-$default_port}"
  require_safe_integer "$port_key" "$host_port"
  direct_url="http://127.0.0.1:$host_port"
  gateway_url="http://127.0.0.1:18080"

  require_uv
  mkdir -p "$output_dir"
  wait_for_model "$direct_url" "$model"
  "$uv_bin" run --python "$UV_PYTHON" "$smoke_runner" \
    --base-url "$direct_url" \
    --model "$model" \
    --stability-seconds "$stability_seconds" \
    --requests 8 \
    --concurrency 4 \
    --output "$output_dir/$runtime-direct.json"

  wait_for_model "$gateway_url" "$model"
  "$uv_bin" run --python "$UV_PYTHON" "$smoke_runner" \
    --base-url "$gateway_url" \
    --model "$model" \
    --runtime "$runtime" \
    --stability-seconds "$stability_seconds" \
    --requests 8 \
    --concurrency 4 \
    --output "$output_dir/$runtime-gateway.json"
}

validate_static

case "$action" in
  plan)
    print_plan
    ;;
  validate)
    if ! $static_only; then
      validate_compose
    fi
    print_plan
    ;;
  up)
    start_stack
    ;;
  qualify)
    qualify_stack
    ;;
  down)
    stop_stack
    ;;
  test)
    cleanup=true
    cleanup_stack() {
      local status=$?
      if $cleanup && ((${#compose[@]} > 0)); then
        "${compose[@]}" down --remove-orphans >/dev/null 2>&1 || true
      fi
      exit "$status"
    }
    trap cleanup_stack EXIT INT TERM
    start_stack
    qualify_stack
    if $keep; then
      cleanup=false
      printf 'stack_kept=true\n'
    fi
    ;;
esac
