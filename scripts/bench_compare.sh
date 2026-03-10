#!/usr/bin/env bash
# bench_compare.sh — fair comparison: raw-mistralrs vs ax-serving vs llama.cpp
#
# Methodology (apples-to-apples):
#   - All tools use **release builds**
#   - All tools send **exact token-ID sequences** (no chat template overhead)
#   - All tools use **greedy decoding** (top-k=1, temperature=None)
#   - Same prompt lengths and decode token count for all three
#   - llama-bench is the ground truth reference (runs -r 3 warmup+measure)
#
# Usage:
#   TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.850.10 ./scripts/bench_compare.sh
set -euo pipefail

MODEL="models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
AX_BENCH="./target/release/ax-serving-bench"
RAW_BENCH="./target/release/raw-mistralrs-bench"
LLAMA_BENCH="llama-bench"

PP_SIZES="64,256,512"   # prompt token counts (exact, matches llama-bench -p N)
TG_TOKENS=128           # decode tokens
WARMUP=1
ITERS=3

echo "================================================================"
echo "  Benchmark: raw-mistralrs vs ax-serving vs llama.cpp"
echo "  Model:  $MODEL"
echo "  Config: greedy, pp={$PP_SIZES} tokens, tg=$TG_TOKENS tokens"
echo "  Runs:   warmup=$WARMUP, iters=$ITERS"
echo "  Date:   $(date '+%Y-%m-%d %H:%M')"
echo "================================================================"
echo ""

for bin in "$AX_BENCH" "$RAW_BENCH"; do
    if [ ! -f "$bin" ]; then
        echo "ERROR: release binary not found: $bin"
        echo "Run: TOOLCHAINS=... cargo build -p ax-serving-bench --release"
        exit 1
    fi
done

# ── raw mistralrs (GgufModelBuilder direct, no ax-serving wrapping) ────────
echo "── raw-mistralrs-bench (release, CompletionTokens, greedy) ─────"
TOOLCHAINS="${TOOLCHAINS:-}" "$RAW_BENCH" \
    -m "$MODEL" \
    --prompt-lengths "$PP_SIZES" \
    --decode-tokens "$TG_TOKENS" \
    --warmup "$WARMUP" \
    --iters "$ITERS"
echo ""

# ── ax-serving-bench (MistralrsBackend wrapper, CompletionTokens, greedy) ─
echo "── ax-serving-bench (release, CompletionTokens, greedy) ────────"
TOOLCHAINS="${TOOLCHAINS:-}" "$AX_BENCH" bench \
    -m "$MODEL" \
    --prompt-lengths "$PP_SIZES" \
    --decode-tokens "$TG_TOKENS" \
    --warmup "$WARMUP" \
    --iters "$ITERS"
echo ""

# ── llama-bench (reference, -p N exact tokens, greedy by default) ─────────
echo "── llama-bench (ngl=99, exact token sequences, greedy) ─────────"
IFS=',' read -ra PP_ARR <<< "$PP_SIZES"
PP_FLAGS=""
for pp in "${PP_ARR[@]}"; do PP_FLAGS="$PP_FLAGS -p $pp"; done
$LLAMA_BENCH -m "$MODEL" $PP_FLAGS -n "$TG_TOKENS" -ngl 99 -r "$ITERS" 2>&1 \
    | grep -E "^\|" | grep -v "model\|─"
echo ""

echo "================================================================"
echo "Notes:"
echo "  All tools: exact token IDs, greedy, release builds"
echo "  raw-mistralrs: GgufModelBuilder direct, is_streaming=false"
echo "  ax-serving:    MistralrsBackend wrapper, GenerateInput::Tokens"
echo "  llama.cpp:     ggml Metal Q8_0 on-the-fly dequant"
echo "================================================================"
