#!/usr/bin/env bash
# Install and operate NVIDIA TensorRT Edge-LLM on Jetson Thor.
set -euo pipefail

EDGE_TAG="v0.9.1"
EDGE_COMMIT="7f061f21f0a581ba234a1e233c9315b89d8e47d6"
UV_VERSION="0.12.2"
UV_PYTHON="3.12"
JETPACK_PACKAGE_VERSION="7.2-b187"
L4T_CORE_VERSION="39.2.0-20260601141651"
TENSORRT_PACKAGE_VERSION="10.16.2.10-1+cuda13.2"
DEFAULT_MODEL="Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"

usage() {
  cat <<'EOF'
Usage:
  install_tensorrt_edge_llm_thor.sh ACTION [OPTIONS]

Actions:
  system-deps  Install JetPack 7.2 and native build dependencies with sudo.
  preflight    Verify Jetson Thor, JetPack, CUDA, TensorRT, and build tools.
  install      Install pinned uv, clone/build Edge-LLM, and fetch the pinned model.
  start        Start the experimental OpenAI-compatible server.
  stop         Stop the server started by this script.
  status       Report the PID and /health + /v1/models state.

Options:
  --prefix PATH               Install root (default: ~/.local/share/ax-serving/tensorrt-edge-llm/v0.9.1).
  --expected-hostname NAME    Fail if the remote hostname differs.
  --model ID                  Hugging Face model (default: Qwen/Qwen3-0.6B).
  --model-revision SHA        Exact model revision.
  --served-model NAME         Stable /v1/models identity (default: qwen3-edge).
  --host ADDRESS              Server bind address (default: 127.0.0.1).
  --port PORT                 Server port (default: 8000).
  --wait-seconds N            Startup deadline (default: 1800).
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

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

prefix="$HOME/.local/share/ax-serving/tensorrt-edge-llm/$EDGE_TAG"
expected_hostname=""
model="$DEFAULT_MODEL"
model_revision="$DEFAULT_MODEL_REVISION"
served_model="qwen3-edge"
bind_host="127.0.0.1"
port=8000
wait_seconds=1800

while (($#)); do
  case "$1" in
    --prefix)
      (($# >= 2)) || die "--prefix requires a value"
      prefix="$2"
      shift 2
      ;;
    --expected-hostname)
      (($# >= 2)) || die "--expected-hostname requires a value"
      expected_hostname="$2"
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a value"
      model="$2"
      shift 2
      ;;
    --model-revision)
      (($# >= 2)) || die "--model-revision requires a value"
      model_revision="$2"
      shift 2
      ;;
    --served-model)
      (($# >= 2)) || die "--served-model requires a value"
      served_model="$2"
      shift 2
      ;;
    --host)
      (($# >= 2)) || die "--host requires a value"
      bind_host="$2"
      shift 2
      ;;
    --port)
      (($# >= 2)) || die "--port requires a value"
      port="$2"
      shift 2
      ;;
    --wait-seconds)
      (($# >= 2)) || die "--wait-seconds requires a value"
      wait_seconds="$2"
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
  system-deps|preflight|install|start|stop|status) ;;
  *) die "unknown action $action" ;;
esac

[[ "$prefix" == /* ]] || die "--prefix must be an absolute path"
[[ "$prefix" != "/" ]] || die "--prefix must not be the filesystem root"
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port <= 0 || port > 65535)); then
  die "--port must be between 1 and 65535"
fi
[[ "$wait_seconds" =~ ^[0-9]+$ ]] || die "--wait-seconds must be an integer"
[[ "$model" =~ ^[A-Za-z0-9._/-]+$ ]] || die "--model contains unsupported characters"
[[ "$model_revision" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] ||
  die "--model-revision must be an exact hexadecimal revision"
[[ "$served_model" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "--served-model contains unsupported characters"

source_dir="$prefix/source"
build_dir="$source_dir/build"
venv_dir="$prefix/.venv"
model_root="$prefix/models"
model_dir="$model_root/$served_model-$model_revision"
model_alias="$model_root/$served_model"
state_dir="$prefix/state"
log_dir="$prefix/log"
pid_file="$state_dir/server.pid"
server_log="$log_dir/server.log"
uv_install_dir="$HOME/.local/bin"
uv_bin="$uv_install_dir/uv"

verify_expected_hostname() {
  if [[ -n "$expected_hostname" && "$(hostname)" != "$expected_hostname" ]]; then
    die "hostname mismatch: expected $expected_hostname but connected to $(hostname)"
  fi
}

install_system_dependencies() {
  verify_expected_hostname
  command -v sudo >/dev/null 2>&1 || die "sudo is required for system-deps"
  sudo -n true || die "passwordless sudo is required for system-deps"
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "nvidia-jetpack=$JETPACK_PACKAGE_VERSION" \
    cmake \
    build-essential \
    git \
    python3-venv \
    python3-dev \
    pkg-config \
    ninja-build
}

preflight() {
  verify_expected_hostname
  [[ "$(uname -m)" == "aarch64" ]] || die "TensorRT Edge-LLM Thor requires aarch64"
  [[ -r /etc/nv_tegra_release ]] || die "/etc/nv_tegra_release is missing"
  grep -q '^# R39 .*REVISION: 2\.' /etc/nv_tegra_release ||
    die "JetPack 7.2 / Jetson Linux R39.2 is required"
  [[ "$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)" == \
    "$JETPACK_PACKAGE_VERSION" ]] ||
    die "nvidia-jetpack $JETPACK_PACKAGE_VERSION is required"
  [[ "$(dpkg-query -W -f='${Version}' nvidia-l4t-core 2>/dev/null)" == \
    "$L4T_CORE_VERSION" ]] ||
    die "nvidia-l4t-core $L4T_CORE_VERSION is required"
  [[ -x /usr/local/cuda-13.2/bin/nvcc ]] || die "CUDA Toolkit 13.2 is missing"
  /usr/local/cuda-13.2/bin/nvcc --version | grep -q 'release 13\.2' ||
    die "nvcc does not report CUDA 13.2"
  [[ "$(dpkg-query -W -f='${Version}' libnvinfer-dev 2>/dev/null)" == \
    "$TENSORRT_PACKAGE_VERSION" ]] ||
    die "libnvinfer-dev $TENSORRT_PACKAGE_VERSION is required"
  for command_name in cmake git ninja python3 curl; do
    command -v "$command_name" >/dev/null 2>&1 ||
      die "required command is missing: $command_name"
  done
  [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == \
    "$UV_PYTHON" ]] ||
    die "Python $UV_PYTHON is required"
  printf 'platform=jetson-thor\n'
  printf 'jetpack=%s\n' "$JETPACK_PACKAGE_VERSION"
  printf 'cuda=13.2\n'
  printf 'tensorrt=%s\n' "$TENSORRT_PACKAGE_VERSION"
  printf 'edge_llm_tag=%s\n' "$EDGE_TAG"
  printf 'edge_llm_commit=%s\n' "$EDGE_COMMIT"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1 &&
    [[ "$(uv --version | awk '{print $2}')" == "$UV_VERSION" ]]; then
    uv_bin="$(command -v uv)"
    return
  fi
  if [[ -x "$uv_bin" ]] &&
    [[ "$("$uv_bin" --version | awk '{print $2}')" == "$UV_VERSION" ]]; then
    return
  fi

  mkdir -p "$uv_install_dir"
  local installer
  installer="$(mktemp "${TMPDIR:-/tmp}/uv-installer.XXXXXX")"
  trap 'rm -f "$installer"' RETURN
  curl -fsSL "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$installer"
  UV_INSTALL_DIR="$uv_install_dir" UV_NO_MODIFY_PATH=1 sh "$installer"
  [[ -x "$uv_bin" ]] || die "uv installer did not create $uv_bin"
  [[ "$("$uv_bin" --version | awk '{print $2}')" == "$UV_VERSION" ]] ||
    die "installed uv version does not match $UV_VERSION"
  rm -f "$installer"
  trap - RETURN
}

clone_source() {
  mkdir -p "$prefix"
  if [[ -e "$source_dir" && ! -d "$source_dir/.git" ]]; then
    die "$source_dir exists but is not a Git checkout"
  fi
  if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 --branch "$EDGE_TAG" \
      https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$source_dir"
  fi
  [[ "$(git -C "$source_dir" rev-parse HEAD)" == "$EDGE_COMMIT" ]] ||
    die "$source_dir is not pinned to $EDGE_COMMIT"
  git -C "$source_dir" submodule update --init --recursive --depth 1
}

install_python_environment() {
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$uv_bin" venv --python "$UV_PYTHON" "$venv_dir"
  fi
  "$uv_bin" pip install \
    --python "$venv_dir/bin/python" \
    --editable "${source_dir}[server]"
}

build_runtime() {
  local pybind11_dir jobs
  pybind11_dir="$("$venv_dir/bin/python" -m pybind11 --cmakedir)"
  jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
  cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCUDA_CTK_VERSION=13.2 \
    -DCMAKE_TOOLCHAIN_FILE="$source_dir/cmake/aarch64_linux_toolchain.cmake" \
    -DEMBEDDED_TARGET=jetson-thor \
    -DENABLE_CUTE_DSL=ALL \
    -DCUTE_DSL_ARTIFACT_TAG=sm_110 \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$venv_dir/bin/python" \
    -Dpybind11_DIR="$pybind11_dir"
  cmake --build "$build_dir" --parallel "$jobs"
  compgen -G "$build_dir/pybind/*_edgellm_runtime*.so" >/dev/null ||
    die "Python runtime binding was not built"
  runtime_plugin_path >/dev/null
}

runtime_plugin_path() {
  local candidate
  local -a candidates=()
  while IFS= read -r -d '' candidate; do
    candidates+=("$candidate")
  done < <(
    find -L "$build_dir" -type f \
      -name 'libNvInfer_edgellm_plugin.so' -print0
  )
  ((${#candidates[@]} == 1)) ||
    die "expected one TensorRT Edge-LLM plugin, found ${#candidates[@]}"
  printf '%s\n' "${candidates[0]}"
}

fetch_model() {
  mkdir -p "$model_root"
  "$venv_dir/bin/python" - "$model" "$model_revision" "$model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, revision, local_dir = sys.argv[1:]
path = snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
print(f"model_snapshot={path}")
PY
  if [[ -L "$model_alias" ]]; then
    rm -f "$model_alias"
  elif [[ -e "$model_alias" ]]; then
    die "model alias exists and is not a symlink: $model_alias"
  fi
  ln -s "$model_dir" "$model_alias"
}

write_install_manifest() {
  "$venv_dir/bin/python" - "$prefix/installation.json" \
    "$EDGE_TAG" "$EDGE_COMMIT" "$UV_VERSION" "$JETPACK_PACKAGE_VERSION" \
    "$L4T_CORE_VERSION" "$TENSORRT_PACKAGE_VERSION" "$model" "$model_revision" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

(
    path,
    tag,
    commit,
    uv_version,
    jetpack_version,
    l4t_version,
    tensorrt_version,
    model,
    revision,
) = sys.argv[1:]
payload = {
    "schema": "com.automatosx.ax-serving.tensorrt-edge-llm-install.v1",
    "hostname": platform.node(),
    "architecture": platform.machine(),
    "edge_llm_tag": tag,
    "edge_llm_commit": commit,
    "uv_version": uv_version,
    "jetpack_package_version": jetpack_version,
    "l4t_core_version": l4t_version,
    "model": model,
    "model_revision": revision,
    "cuda_version": "13.2",
    "tensorrt_version": tensorrt_version,
    "support_level": "experimental_compatibility",
}
destination = Path(path)
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY
}

install_edge_llm() {
  preflight
  ensure_uv
  clone_source
  install_python_environment
  build_runtime
  fetch_model
  write_install_manifest
  printf 'install_root=%s\n' "$prefix"
}

server_pid() {
  [[ -r "$pid_file" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' <"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o command= | grep -Fq 'experimental.server' || return 1
  printf '%s\n' "$pid"
}

wait_for_server() {
  "$venv_dir/bin/python" - "$port" "$served_model" "$wait_seconds" <<'PY'
import json
import sys
import time
import urllib.request

port, expected, timeout_raw = sys.argv[1:]
deadline = time.monotonic() + int(timeout_raw)
health_url = f"http://127.0.0.1:{port}/health"
models_url = f"http://127.0.0.1:{port}/v1/models"
last_error = "server not ready"
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            health = json.load(response)
        with urllib.request.urlopen(models_url, timeout=5) as response:
            models = json.load(response)
        identities = {
            item.get("id")
            for item in models.get("data", [])
            if isinstance(item, dict)
        }
        if health.get("status") == "healthy" and expected in identities:
            print(f"ready model={expected}")
            raise SystemExit(0)
        last_error = f"health={health!r}, models={sorted(identities)!r}"
    except (OSError, ValueError) as error:
        last_error = str(error).replace("\n", " ")[:240]
    time.sleep(2)
raise SystemExit(f"server readiness timed out: {last_error}")
PY
}

start_server() {
  local plugin_path
  preflight >/dev/null
  [[ -x "$venv_dir/bin/python" ]] || die "run install before start"
  [[ -d "$source_dir/.git" ]] || die "run install before start"
  [[ -L "$model_alias" ]] || die "pinned model alias is missing; run install"
  plugin_path="$(runtime_plugin_path)"
  if pid="$(server_pid)"; then
    printf 'server_already_running_pid=%s\n' "$pid"
    wait_for_server
    return
  fi

  mkdir -p "$state_dir" "$log_dir"
  (
    cd "$model_root"
    export PYTHONPATH="$source_dir"
    export BUILD_DIR="$build_dir"
    export HF_HOME="$prefix/huggingface"
    export EDGELLM_PLUGIN_PATH="$plugin_path"
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    nohup "$venv_dir/bin/python" -m experimental.server \
      --model "$served_model" \
      --host "$bind_host" \
      --port "$port" \
      --max-input-len 2048 \
      --max-batch-size 1 \
      --max-kv-cache-capacity 8192 \
      >"$server_log" 2>&1 &
    printf '%s\n' "$!" >"$pid_file"
  )
  wait_for_server
  printf 'server_pid=%s\n' "$(server_pid)"
  printf 'server_log=%s\n' "$server_log"
}

stop_server() {
  local pid forced
  if ! pid="$(server_pid)"; then
    rm -f "$pid_file"
    printf 'server_running=false\n'
    return
  fi
  forced=false
  kill "$pid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    forced=true
    kill -KILL "$pid"
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    die "server PID $pid survived SIGKILL"
  fi
  rm -f "$pid_file"
  printf 'server_stopped_pid=%s forced=%s\n' "$pid" "$forced"
}

report_status() {
  local pid
  if pid="$(server_pid)"; then
    printf 'server_running=true\n'
    printf 'server_pid=%s\n' "$pid"
  else
    printf 'server_running=false\n'
    return 1
  fi
  "$venv_dir/bin/python" - "$port" <<'PY'
import json
import sys
import urllib.request

port = sys.argv[1]
for path in ("/health", "/v1/models"):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        print(f"{path}={json.dumps(json.load(response), sort_keys=True)}")
PY
}

case "$action" in
  system-deps) install_system_dependencies ;;
  preflight) preflight ;;
  install) install_edge_llm ;;
  start) start_server ;;
  stop) stop_server ;;
  status) report_status ;;
esac
