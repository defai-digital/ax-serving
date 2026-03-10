#!/usr/bin/env bash
# Compare raw llama.cpp vs ax-serving(llama.cpp route) and output JSON + diff%.
#
# Single-model outputs under --out-dir/<model-id>/:
#   raw_llama.json       parsed llama-bench results
#   ax_prefill.json      ax-serving prefill-only benchmark JSON
#   ax_decode.json       ax-serving decode-focused benchmark JSON
#   comparison.json      merged metrics + percent deltas
#   summary.md           human-readable report
#
# Batch outputs under --out-dir:
#   aggregate.json       merged list of per-model comparisons
#   aggregate.md         one-page delta summary table
#
# Example:
#   AXS_ROUTING_CONFIG=config/backends.yaml \
#   ./scripts/bench_llamacpp_vs_ax.sh \
#     --model models/Qwen3-8B-Q4_K_M.gguf \
#     --prompt-lengths 39,509 \
#     --decode-tokens 128 \
#     --warmup 1 --iters 3

set -euo pipefail

MODEL=""
MODEL_LIST=""
PROMPT_LENGTHS="39,509"
DECODE_TOKENS="128"
WARMUP="1"
ITERS="3"
OUT_DIR="automatosx/tmp/bench-compare-$(date +%Y%m%d-%H%M%S)"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-/opt/homebrew/bin/llama-bench}"
AX_BENCH_BIN="${AX_BENCH_BIN:-./target/release/ax-serving-bench}"
AXS_ROUTING_CONFIG_PATH="${AXS_ROUTING_CONFIG:-config/backends.yaml}"

usage() {
  cat <<'EOF'
Usage: bench_llamacpp_vs_ax.sh (--model <path> | --model-list <file>) [options]

Required (choose one):
  --model <path>                 One GGUF model path
  --model-list <file>            Text file: one GGUF path per line

Options:
  --prompt-lengths <csv>         Prompt lengths, default: 39,509
  --decode-tokens <n>            Decode tokens, default: 128
  --warmup <n>                   Warmup iterations, default: 1
  --iters <n>                    Measurement iterations, default: 3
  --out-dir <path>               Output directory
  --llama-bench-bin <path>       llama-bench binary path
  --ax-bench-bin <path>          ax-serving-bench binary path
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
    --llama-bench-bin) LLAMA_BENCH_BIN="$2"; shift 2 ;;
    --ax-bench-bin) AX_BENCH_BIN="$2"; shift 2 ;;
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
if [[ ! -x "$LLAMA_BENCH_BIN" ]]; then
  echo "llama-bench binary not executable: $LLAMA_BENCH_BIN" >&2
  exit 1
fi
if [[ ! -x "$AX_BENCH_BIN" ]]; then
  echo "ax-serving-bench binary not executable: $AX_BENCH_BIN" >&2
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

run_single_model() {
  local model_path="$1"
  local model_dir="$2"
  local raw_txt="$model_dir/raw_llama.txt"
  local raw_json="$model_dir/raw_llama.json"
  local ax_prefill_json="$model_dir/ax_prefill.json"
  local ax_decode_json="$model_dir/ax_decode.json"
  local comparison_json="$model_dir/comparison.json"
  local summary_md="$model_dir/summary.md"

  mkdir -p "$model_dir"
  echo "==> Running model: $model_path"
  echo "    output: $model_dir"

  echo "    [1/5] raw llama.cpp"
  local pp_flags=""
  local pp
  IFS=',' read -r -a PP_ARR <<< "$PROMPT_LENGTHS"
  for pp in "${PP_ARR[@]}"; do
    pp_flags="$pp_flags -p $pp"
  done
  "$LLAMA_BENCH_BIN" -m "$model_path" $pp_flags -n "$DECODE_TOKENS" -ngl 99 -r "$ITERS" > "$raw_txt" 2>&1

  echo "    [2/5] parse raw llama.cpp JSON"
  python3 - "$raw_txt" "$raw_json" "$DECODE_TOKENS" <<'PY'
import json
import re
import sys

raw_path, out_path, decode_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
prefill = {}
decode = None
model = None

row_re = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")
test_pp = re.compile(r"pp(\d+)")
test_tg = re.compile(r"tg(\d+)")
ts_val = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = row_re.match(line.strip())
        if not m:
            continue
        cols = [c.strip() for c in m.groups()]
        if cols[0] == "model" or cols[0].startswith("---"):
            continue
        model = cols[0]
        test_col = cols[5]
        ts_col = cols[6]
        ts_m = ts_val.search(ts_col)
        if not ts_m:
            continue
        tps = float(ts_m.group(1))
        pp_m = test_pp.fullmatch(test_col)
        if pp_m:
            prefill[int(pp_m.group(1))] = tps
            continue
        tg_m = test_tg.fullmatch(test_col)
        if tg_m and int(tg_m.group(1)) == decode_tokens:
            decode = tps

out = {
    "tool": "llama.cpp",
    "model": model,
    "decode_tokens": decode_tokens,
    "prefill_tok_per_sec": prefill,
    "decode_tok_per_sec": decode,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
PY

  echo "    [3/5] ax-serving prefill-only"
  AXS_ROUTING_CONFIG="$AXS_ROUTING_CONFIG_PATH" "$AX_BENCH_BIN" bench \
    -m "$model_path" \
    --prompt-lengths "$PROMPT_LENGTHS" \
    --decode-tokens 0 \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --json "$ax_prefill_json" >/dev/null

  echo "    [4/5] ax-serving decode-focused"
  AXS_ROUTING_CONFIG="$AXS_ROUTING_CONFIG_PATH" "$AX_BENCH_BIN" bench \
    -m "$model_path" \
    --prompt-lengths 1 \
    --decode-tokens "$DECODE_TOKENS" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --json "$ax_decode_json" >/dev/null

  echo "    [5/5] compute comparison"
  python3 - "$raw_json" "$ax_prefill_json" "$ax_decode_json" "$comparison_json" "$summary_md" "$PROMPT_LENGTHS" "$model_path" <<'PY'
import json
import sys
from datetime import datetime, UTC

raw_path, ax_prefill_path, ax_decode_path, comp_path, md_path, prompt_csv, model_path = sys.argv[1:]
prompt_lengths = [int(x.strip()) for x in prompt_csv.split(",") if x.strip()]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

raw = load_json(raw_path)
ax_prefill = load_json(ax_prefill_path)
ax_decode = load_json(ax_decode_path)

raw_prefill = {int(k): float(v) for k, v in raw.get("prefill_tok_per_sec", {}).items()}
raw_decode = float(raw.get("decode_tok_per_sec") or 0.0)
ax_prefill_map = {int(r["prompt_length"]): float(r["prefill_tok_per_sec"]) for r in ax_prefill.get("results", [])}
ax_decode_map = {int(r["prompt_length"]): float(r["decode_tok_per_sec"]) for r in ax_decode.get("results", [])}
ax_decode_val = ax_decode_map.get(1, 0.0)

def pct(ax, rawv):
    if rawv == 0:
        return None
    return (ax / rawv - 1.0) * 100.0

prefill_rows = []
for p in prompt_lengths:
    r = raw_prefill.get(p, 0.0)
    a = ax_prefill_map.get(p, 0.0)
    prefill_rows.append({
        "prompt_length": p,
        "raw_llama_tok_per_sec": r,
        "ax_serving_tok_per_sec": a,
        "delta_percent": pct(a, r),
    })

decode_row = {
    "raw_llama_tok_per_sec": raw_decode,
    "ax_serving_tok_per_sec": ax_decode_val,
    "delta_percent": pct(ax_decode_val, raw_decode),
    "method": "ax-serving prompt_len=1, decode-focused",
}

comp = {
    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "model_path": model_path,
    "raw_model_label": raw.get("model"),
    "prefill": prefill_rows,
    "decode": decode_row,
    "notes": [
        "Prefill uses ax-serving run with decode_tokens=0.",
        "Decode uses ax-serving run with prompt_len=1 for decode-focused comparison.",
    ],
}

with open(comp_path, "w", encoding="utf-8") as f:
    json.dump(comp, f, indent=2)

def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"

lines = []
lines.append("# llama.cpp vs ax-serving (llama.cpp route)")
lines.append("")
lines.append(f"- Generated: {comp['generated_at']}")
lines.append(f"- Model path: `{model_path}`")
lines.append(f"- Raw model label: `{comp.get('raw_model_label')}`")
lines.append("")
lines.append("## Prefill")
lines.append("")
lines.append("| prompt_len | raw llama.cpp | ax-serving | delta % |")
lines.append("|---:|---:|---:|---:|")
for row in prefill_rows:
    lines.append(
        f"| {row['prompt_length']} | {row['raw_llama_tok_per_sec']:.2f} | "
        f"{row['ax_serving_tok_per_sec']:.2f} | {fmt(row['delta_percent'])}% |"
    )
lines.append("")
lines.append("## Decode")
lines.append("")
lines.append("| raw llama.cpp | ax-serving | delta % | note |")
lines.append("|---:|---:|---:|---|")
lines.append(
    f"| {raw_decode:.2f} | {ax_decode_val:.2f} | {fmt(decode_row['delta_percent'])}% | "
    f"{decode_row['method']} |"
)
lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append("- `raw_llama.json`")
lines.append("- `ax_prefill.json`")
lines.append("- `ax_decode.json`")
lines.append("- `comparison.json`")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY

  echo "    done: $comparison_json"
}

declare -a MODELS=()
if [[ -n "$MODEL" ]]; then
  MODELS+=("$MODEL")
else
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -z "$line" ]] && continue
    [[ ! -f "$line" ]] && { echo "Model not found in list: $line" >&2; exit 1; }
    MODELS+=("$line")
  done < "$MODEL_LIST"
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models to run." >&2
  exit 1
fi

declare -a COMPARISONS=()
idx=0
for m in "${MODELS[@]}"; do
  idx=$((idx + 1))
  model_id="$(sanitize_model_id "$m")"
  model_dir="$OUT_DIR/${idx}-${model_id}"
  run_single_model "$m" "$model_dir"
  COMPARISONS+=("$model_dir/comparison.json")
done

AGG_JSON="$OUT_DIR/aggregate.json"
AGG_MD="$OUT_DIR/aggregate.md"

python3 - "$AGG_JSON" "$AGG_MD" "$PROMPT_LENGTHS" "${COMPARISONS[@]}" <<'PY'
import json
import sys
from datetime import datetime, UTC

agg_json, agg_md, prompt_csv, *comparison_paths = sys.argv[1:]
prompt_lengths = [int(x.strip()) for x in prompt_csv.split(",") if x.strip()]

items = []
for p in comparison_paths:
    with open(p, "r", encoding="utf-8") as f:
        items.append(json.load(f))

out = {
    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "count": len(items),
    "items": items,
}
with open(agg_json, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

def fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.2f}%"

lines = []
lines.append("# Aggregate: llama.cpp vs ax-serving (llama.cpp route)")
lines.append("")
lines.append(f"- Generated: {out['generated_at']}")
lines.append(f"- Models: {out['count']}")
lines.append("")

header = ["model"]
for p in prompt_lengths:
    header.append(f"prefill p{p} delta")
header.append("decode delta")

lines.append("| " + " | ".join(header) + " |")
lines.append("|" + "|".join(["---"] * len(header)) + "|")

for item in items:
    prefill_map = {int(r["prompt_length"]): r.get("delta_percent") for r in item.get("prefill", [])}
    row = [item.get("model_path", item.get("raw_model_label", "unknown"))]
    for p in prompt_lengths:
        row.append(fmt(prefill_map.get(p)))
    row.append(fmt(item.get("decode", {}).get("delta_percent")))
    lines.append("| " + " | ".join(row) + " |")

with open(agg_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY

echo ""
echo "Done."
echo "  - per-model dirs: $OUT_DIR/<index>-<model-id>/"
echo "  - aggregate json: $AGG_JSON"
echo "  - aggregate md:   $AGG_MD"
