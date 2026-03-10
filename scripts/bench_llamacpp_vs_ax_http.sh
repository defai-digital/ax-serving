#!/usr/bin/env bash
# End-to-end apples-to-apples benchmark:
#   direct llama-server HTTP vs ax-serving HTTP (llama.cpp route)
#
# Outputs under --out-dir/<model-id>/:
#   direct_http.json
#   ax_http.json
#   comparison.json
#   summary.md
#
# Batch outputs under --out-dir:
#   aggregate.json
#   aggregate.md
#
# Notes:
# - This compares HTTP end-to-end behavior (request/response overhead included).
# - Prefill uses max_tokens=0 with calibrated prompt lengths.
# - Decode uses a decode-focused run at the first prompt length from --prompt-lengths.

set -euo pipefail

MODEL=""
MODEL_LIST=""
PROMPT_LENGTHS="39,509"
DECODE_TOKENS="128"
WARMUP="1"
ITERS="3"
OUT_DIR="automatosx/tmp/bench-compare-http-$(date +%Y%m%d-%H%M%S)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
AX_CLI_BIN="${AX_CLI_BIN:-./target/release/ax-llama}"
AXS_ROUTING_CONFIG_PATH="${AXS_ROUTING_CONFIG:-config/backends.yaml}"

LLAMA_PORT="${LLAMA_PORT:-18081}"
AX_PORT="${AX_PORT:-18082}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-120}"

usage() {
  cat <<'EOF'
Usage: bench_llamacpp_vs_ax_http.sh (--model <path> | --model-list <file>) [options]

Required (choose one):
  --model <path>                 One GGUF model path
  --model-list <file>            Text file: one GGUF path per line

Options:
  --prompt-lengths <csv>         Prompt token targets, default: 39,509
  --decode-tokens <n>            Decode tokens, default: 128
  --warmup <n>                   Warmup iterations, default: 1
  --iters <n>                    Measurement iterations, default: 3
  --out-dir <path>               Output directory
  --llama-server-bin <path>      llama-server binary path
  --ax-cli-bin <path>            ax-llama binary path
  --llama-port <n>               direct llama-server port (default: 18081, 0 = auto)
  --ax-port <n>                  ax-serving port (default: 18082, 0 = auto)
  --startup-timeout-sec <n>      server startup timeout seconds (default: 120)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-list) MODEL_LIST="$2"; shift 2 ;;
    --prompt-lengths) PROMPT_LENGTHS="$2"; shift 2 ;;
    --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --llama-server-bin) LLAMA_SERVER_BIN="$2"; shift 2 ;;
    --ax-cli-bin) AX_CLI_BIN="$2"; shift 2 ;;
    --llama-port) LLAMA_PORT="$2"; shift 2 ;;
    --ax-port) AX_PORT="$2"; shift 2 ;;
    --startup-timeout-sec) STARTUP_TIMEOUT_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" && -z "$MODEL_LIST" ]]; then
  echo "either --model or --model-list is required" >&2
  usage
  exit 1
fi
if [[ -n "$MODEL" && -n "$MODEL_LIST" ]]; then
  echo "choose only one: --model or --model-list" >&2
  exit 1
fi
if [[ -n "$MODEL" && ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi
if [[ -n "$MODEL_LIST" && ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi
if ! command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1; then
  echo "llama-server binary not found on PATH: $LLAMA_SERVER_BIN" >&2
  exit 1
fi
if [[ ! -x "$AX_CLI_BIN" ]]; then
  echo "ax-llama binary not executable: $AX_CLI_BIN" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not found" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

sanitize_model_id() {
  local p="$1"
  local b
  b="$(basename "$p")"
  b="${b%.gguf}"
  b="$(echo "$b" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')"
  echo "$b"
}

wait_for_http_200() {
  local url="$1"
  local timeout="$2"
  local i
  for ((i=0; i<timeout; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

pick_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

run_single_model() {
  local model_path="$1"
  local model_dir="$2"
  local direct_log="$model_dir/direct_llama_server.log"
  local ax_log="$model_dir/ax_serving.log"
  local direct_json="$model_dir/direct_http.json"
  local ax_json="$model_dir/ax_http.json"
  local comparison_json="$model_dir/comparison.json"
  local summary_md="$model_dir/summary.md"
  local llama_port_run="$LLAMA_PORT"
  local ax_port_run="$AX_PORT"

  if [[ "$llama_port_run" == "0" ]]; then
    llama_port_run="$(pick_free_port)"
  fi
  if [[ "$ax_port_run" == "0" ]]; then
    ax_port_run="$(pick_free_port)"
    if [[ "$ax_port_run" == "$llama_port_run" ]]; then
      ax_port_run="$(pick_free_port)"
    fi
  fi

  mkdir -p "$model_dir"
  echo "==> Running model: $model_path"
  echo "    output: $model_dir"
  echo "    ports: direct=$llama_port_run ax=$ax_port_run"

  (
    set -euo pipefail
    local llama_pid=""
    local ax_pid=""

    cleanup() {
      if [[ -n "${ax_pid:-}" ]]; then
        kill "$ax_pid" >/dev/null 2>&1 || true
        wait "$ax_pid" >/dev/null 2>&1 || true
      fi
      if [[ -n "${llama_pid:-}" ]]; then
        kill "$llama_pid" >/dev/null 2>&1 || true
        wait "$llama_pid" >/dev/null 2>&1 || true
      fi
    }
    trap cleanup EXIT

    echo "    [1/4] start direct llama-server"
    "$LLAMA_SERVER_BIN" \
      -m "$model_path" \
      --host 127.0.0.1 \
      --port "$llama_port_run" \
      -ngl 99 >"$direct_log" 2>&1 &
    llama_pid=$!
    if ! wait_for_http_200 "http://127.0.0.1:${llama_port_run}/health" "$STARTUP_TIMEOUT_SEC"; then
      echo "direct llama-server failed to become ready (see $direct_log)" >&2
      exit 1
    fi

    echo "    [2/4] start ax-serving (llama.cpp route)"
    AXS_ROUTING_CONFIG="$AXS_ROUTING_CONFIG_PATH" \
      "$AX_CLI_BIN" serve \
      -m "$model_path" \
      --host 127.0.0.1 \
      --port "$ax_port_run" >"$ax_log" 2>&1 &
    ax_pid=$!
    if ! wait_for_http_200 "http://127.0.0.1:${ax_port_run}/health" "$STARTUP_TIMEOUT_SEC"; then
      echo "ax-serving failed to become ready (see $ax_log)" >&2
      exit 1
    fi

    echo "    [3/4] run end-to-end HTTP benchmark"
    python3 - "$PROMPT_LENGTHS" "$DECODE_TOKENS" "$WARMUP" "$ITERS" \
      "$llama_port_run" "$ax_port_run" "$model_path" "$direct_json" "$ax_json" "$comparison_json" "$summary_md" <<'PY'
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

(
    prompt_csv,
    decode_tokens_s,
    warmup_s,
    iters_s,
    llama_port_s,
    ax_port_s,
    model_path,
    direct_json_path,
    ax_json_path,
    comparison_json_path,
    summary_md_path,
) = sys.argv[1:]

prompt_targets = [int(x.strip()) for x in prompt_csv.split(",") if x.strip()]
decode_tokens = int(decode_tokens_s)
warmup = int(warmup_s)
iters = int(iters_s)
llama_port = int(llama_port_s)
ax_port = int(ax_port_s)
decode_prompt_target = prompt_targets[0]

llama_url = f"http://127.0.0.1:{llama_port}/v1/completions"
ax_url = f"http://127.0.0.1:{ax_port}/v1/chat/completions"


def http_post_json(url: str, payload: dict, timeout_sec: int = 300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - start
            return elapsed, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} HTTP {e.code}: {body}") from e


def usage_from_response(resp: dict):
    usage = resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None:
        prompt_tokens = resp.get("tokens_evaluated")
    if completion_tokens is None:
        completion_tokens = resp.get("tokens_predicted")
    if prompt_tokens is None:
        prompt_tokens = 0
    if completion_tokens is None:
        completion_tokens = 0
    return int(prompt_tokens), int(completion_tokens)


def llama_payload(prompt_text: str, n_predict: int) -> dict:
    return {
        "prompt": prompt_text,
        "stream": False,
        "cache_prompt": False,
        "n_predict": int(n_predict),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }


def ax_payload(user_content: str, max_tokens: int) -> dict:
    return {
        "model": "default",
        "messages": [{"role": "user", "content": user_content}],
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }


def make_user_content(word_count: int) -> str:
    return ("hello " * max(1, word_count)).strip()


def to_direct_prompt(user_content: str) -> str:
    # Match ax-serving llama.cpp chat flattening:
    # "user: <content>\nassistant:"
    return f"user: {user_content}\nassistant:"


def measure_direct_once(user_content: str, n_predict: int):
    prompt = to_direct_prompt(user_content)
    elapsed, resp = http_post_json(llama_url, llama_payload(prompt, n_predict))
    prompt_tokens, completion_tokens = usage_from_response(resp)
    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def measure_ax_once(user_content: str, max_tokens: int):
    elapsed, resp = http_post_json(ax_url, ax_payload(user_content, max_tokens))
    prompt_tokens, completion_tokens = usage_from_response(resp)
    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def calibrate_prompt_for_target(target_tokens: int):
    low = 1
    high = max(8, target_tokens * 4)

    def count_for_words(words: int):
        m = measure_direct_once(make_user_content(words), 0)
        return m["prompt_tokens"]

    low_toks = count_for_words(low)
    high_toks = count_for_words(high)
    while high_toks < target_tokens and high < target_tokens * 64:
        high *= 2
        high_toks = count_for_words(high)

    best_words = low
    best_tokens = low_toks
    best_diff = abs(low_toks - target_tokens)

    lo = low
    hi = high
    while lo <= hi:
        mid = (lo + hi) // 2
        toks = count_for_words(mid)
        diff = abs(toks - target_tokens)
        if diff < best_diff:
            best_diff = diff
            best_words = mid
            best_tokens = toks
        if toks < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1

    # local refinement near best_words
    for w in range(max(1, best_words - 6), best_words + 7):
        toks = count_for_words(w)
        diff = abs(toks - target_tokens)
        if diff < best_diff:
            best_diff = diff
            best_words = w
            best_tokens = toks

    return {
        "target_prompt_tokens": target_tokens,
        "chosen_words": best_words,
        "direct_prompt_tokens": best_tokens,
        "token_error": best_tokens - target_tokens,
        "user_content": make_user_content(best_words),
    }


def median(values):
    if not values:
        return 0.0
    return statistics.median(values)


calibrations = [calibrate_prompt_for_target(t) for t in prompt_targets]
content_by_target = {c["target_prompt_tokens"]: c["user_content"] for c in calibrations}
calibrated_prompt_tokens_by_target = {
    c["target_prompt_tokens"]: c["direct_prompt_tokens"] for c in calibrations
}

direct_prefill_rows = []
ax_prefill_rows = []
prefill_comparison = []

for t in prompt_targets:
    content = content_by_target[t]

    for _ in range(warmup):
        measure_direct_once(content, 0)
        measure_ax_once(content, 0)

    direct_samples = []
    ax_samples = []

    for _ in range(iters):
        d = measure_direct_once(content, 0)
        a = measure_ax_once(content, 0)
        prompt_tokens_used = calibrated_prompt_tokens_by_target[t]
        d_tps = (prompt_tokens_used / d["elapsed_sec"]) if d["elapsed_sec"] > 0 else 0.0
        a_tps = (prompt_tokens_used / a["elapsed_sec"]) if a["elapsed_sec"] > 0 else 0.0
        direct_samples.append(d_tps)
        ax_samples.append(a_tps)

    direct_pp = float(median(direct_samples))
    ax_pp = float(median(ax_samples))
    delta = ((ax_pp / direct_pp - 1.0) * 100.0) if direct_pp > 0 else None

    direct_prefill_rows.append(
        {
            "prompt_length_target": t,
            "prefill_tok_per_sec": direct_pp,
        }
    )
    ax_prefill_rows.append(
        {
            "prompt_length_target": t,
            "prefill_tok_per_sec": ax_pp,
        }
    )
    prefill_comparison.append(
        {
            "prompt_length_target": t,
            "raw_llama_tok_per_sec": direct_pp,
            "ax_serving_tok_per_sec": ax_pp,
            "delta_percent": delta,
        }
    )

decode_content = content_by_target[decode_prompt_target]

for _ in range(warmup):
    measure_direct_once(decode_content, decode_tokens)
    measure_ax_once(decode_content, decode_tokens)

direct_decode_samples = []
ax_decode_samples = []

for _ in range(iters):
    d = measure_direct_once(decode_content, decode_tokens)
    a = measure_ax_once(decode_content, decode_tokens)
    d_tps = (decode_tokens / d["elapsed_sec"]) if d["elapsed_sec"] > 0 else 0.0
    a_tps = (decode_tokens / a["elapsed_sec"]) if a["elapsed_sec"] > 0 else 0.0
    direct_decode_samples.append(d_tps)
    ax_decode_samples.append(a_tps)

direct_decode = float(median(direct_decode_samples))
ax_decode = float(median(ax_decode_samples))
decode_delta = ((ax_decode / direct_decode - 1.0) * 100.0) if direct_decode > 0 else None

direct_out = {
    "tool": "llama-server-direct-http",
    "model_path": model_path,
    "prompt_calibration": calibrations,
    "prefill_results": direct_prefill_rows,
    "decode_result": {
        "prompt_length_target": decode_prompt_target,
        "decode_tokens": decode_tokens,
        "decode_tok_per_sec": direct_decode,
    },
}

ax_out = {
    "tool": "ax-serving-http-llama-cpp-route",
    "model_path": model_path,
    "prompt_calibration": calibrations,
    "prefill_results": ax_prefill_rows,
    "decode_result": {
        "prompt_length_target": decode_prompt_target,
        "decode_tokens": decode_tokens,
        "decode_tok_per_sec": ax_decode,
    },
}

comparison = {
    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "model_path": model_path,
    "method": "e2e_http_apples_to_apples",
    "notes": [
        "Both paths are measured via HTTP request wall-clock time.",
        "Prefill uses max_tokens=0 with prompt calibration by direct llama-server usage.prompt_tokens.",
        "Token accounting for throughput uses calibrated prompt tokens and requested decode tokens.",
        f"Decode uses prompt target {decode_prompt_target} and decode_tokens={decode_tokens}.",
    ],
    "calibration": [
        {
            "prompt_length_target": c["target_prompt_tokens"],
            "direct_prompt_tokens": c["direct_prompt_tokens"],
            "token_error": c["token_error"],
        }
        for c in calibrations
    ],
    "prefill": prefill_comparison,
    "decode": {
        "prompt_length_target": decode_prompt_target,
        "decode_tokens": decode_tokens,
        "raw_llama_tok_per_sec": direct_decode,
        "ax_serving_tok_per_sec": ax_decode,
        "delta_percent": decode_delta,
        "method": "completion_tokens / wall_time over non-streaming HTTP",
    },
}

with open(direct_json_path, "w", encoding="utf-8") as f:
    json.dump(direct_out, f, indent=2)
with open(ax_json_path, "w", encoding="utf-8") as f:
    json.dump(ax_out, f, indent=2)
with open(comparison_json_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


lines = []
lines.append("# direct llama-server vs ax-serving (apples-to-apples HTTP)")
lines.append("")
lines.append(f"- Generated: {comparison['generated_at']}")
lines.append(f"- Model path: `{model_path}`")
lines.append("- Method: end-to-end HTTP wall-clock (same request shape)")
lines.append("")
lines.append("## Prompt Calibration")
lines.append("")
lines.append("| target prompt tokens | direct prompt tokens | token error |")
lines.append("|---:|---:|---:|")
for c in comparison["calibration"]:
    lines.append(
        f"| {c['prompt_length_target']} | {c['direct_prompt_tokens']} | {c['token_error']} |"
    )
lines.append("")
lines.append("## Prefill")
lines.append("")
lines.append("| prompt_len target | direct llama-server | ax-serving | delta % |")
lines.append("|---:|---:|---:|---:|")
for row in comparison["prefill"]:
    lines.append(
        f"| {row['prompt_length_target']} | {row['raw_llama_tok_per_sec']:.2f} | "
        f"{row['ax_serving_tok_per_sec']:.2f} | {fmt(row['delta_percent'])}% |"
    )
lines.append("")
lines.append("## Decode")
lines.append("")
lines.append("| prompt_len target | decode tokens | direct llama-server | ax-serving | delta % |")
lines.append("|---:|---:|---:|---:|---:|")
d = comparison["decode"]
lines.append(
    f"| {d['prompt_length_target']} | {d['decode_tokens']} | {d['raw_llama_tok_per_sec']:.2f} | "
    f"{d['ax_serving_tok_per_sec']:.2f} | {fmt(d['delta_percent'])}% |"
)
lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append("- `direct_http.json`")
lines.append("- `ax_http.json`")
lines.append("- `comparison.json`")

with open(summary_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY

    echo "    [4/4] complete"
  )
}

build_aggregate() {
  local out_dir="$1"
  local aggregate_json="$out_dir/aggregate.json"
  local aggregate_md="$out_dir/aggregate.md"

  python3 - "$out_dir" "$aggregate_json" "$aggregate_md" <<'PY'
import json
import pathlib
import sys
from datetime import UTC, datetime

out_dir = pathlib.Path(sys.argv[1])
agg_json = pathlib.Path(sys.argv[2])
agg_md = pathlib.Path(sys.argv[3])

rows = []
for comp_path in sorted(out_dir.glob("*/comparison.json")):
    with comp_path.open("r", encoding="utf-8") as f:
        c = json.load(f)
    model = c.get("model_path")
    prefill = {int(r["prompt_length_target"]): r.get("delta_percent") for r in c.get("prefill", [])}
    decode = c.get("decode", {}).get("delta_percent")
    rows.append(
        {
            "model_path": model,
            "prefill_delta_percent": prefill,
            "decode_delta_percent": decode,
            "comparison_json": str(comp_path),
        }
    )

out = {
    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "method": "e2e_http_apples_to_apples",
    "models": rows,
}
with agg_json.open("w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

def fmt(v):
    return "n/a" if v is None else f"{v:.2f}%"

all_prompt_lens = sorted(
    {p for r in rows for p in r.get("prefill_delta_percent", {}).keys()}
)

lines = []
lines.append("# Aggregate: direct llama-server vs ax-serving (HTTP apples-to-apples)")
lines.append("")
lines.append(f"- Generated: {out['generated_at']}")
lines.append("")
if not rows:
    lines.append("No comparison files found.")
else:
    hdr = "| model | " + " | ".join([f"prefill p{p} delta" for p in all_prompt_lens]) + " | decode delta |"
    sep = "|---|" + "|".join(["---:"] * (len(all_prompt_lens) + 1)) + "|"
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        vals = [fmt(r["prefill_delta_percent"].get(p)) for p in all_prompt_lens]
        vals.append(fmt(r.get("decode_delta_percent")))
        lines.append(f"| `{r['model_path']}` | " + " | ".join(vals) + " |")

with agg_md.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY
}

declare -a MODELS
if [[ -n "$MODEL" ]]; then
  MODELS+=("$MODEL")
else
  while IFS= read -r line; do
    line="$(echo "$line" | sed 's/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    MODELS+=("$line")
  done < "$MODEL_LIST"
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models to run." >&2
  exit 1
fi

i=0
for m in "${MODELS[@]}"; do
  if [[ ! -f "$m" ]]; then
    echo "Model not found: $m" >&2
    exit 1
  fi
  i=$((i + 1))
  id="$(sanitize_model_id "$m")"
  model_dir="$OUT_DIR/${i}-${id}"
  run_single_model "$m" "$model_dir"
done

build_aggregate "$OUT_DIR"

echo ""
echo "Done."
echo "Aggregate:"
echo "  - $OUT_DIR/aggregate.md"
echo "  - $OUT_DIR/aggregate.json"
