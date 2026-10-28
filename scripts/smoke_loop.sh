#!/usr/bin/env bash
# The R5 smoke loop: generate -> filter -> train -> eval on a smoke batch, end to end.
#
#   scripts/smoke_loop.sh [OUT_DIR]            # local: CPU only, a tiny base, every stage capped
#   scripts/smoke_loop.sh --kaggle [OUT_DIR]   # Kaggle GPU: 20 real LoRA steps on Qwen3-0.6B
#
# Generation (teacher) and the judge are mocks, so no keys are needed. Training is real.
#
# Local mode is the only model run allowed on the dev Mac (D-078): it trains a tiny random
# small-vocab Qwen3 (scripts/make_tiny_base.py) on the CPU at batch 1, and every stage runs under
# `timeout 120` (or `gtimeout 120`) with CUDA hidden and the MPS fallback off. macOS has no
# timeout command by default (`brew install coreutils` provides gtimeout); without one the script
# stops before any stage. --kaggle trains the real Qwen/Qwen3-0.6B with --device cuda
# (train/kaggle_train.ipynb runs it). Kaggle has no uv, and `uv run` would build a fresh .venv
# from uv.lock around Kaggle's torch, so --kaggle runs every stage with the notebook's `python`.
#
# eval.py loads models from the Hub, so the eval stage scores the mock compliant model; it proves
# the harness end of the loop, not the adapter.
# Outputs go to OUT_DIR (default: a new temp dir): distill/, dataset/, adapter/ and eval/, plus
# tiny_base/ in local mode.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE=local
if [ "${1:-}" = "--kaggle" ]; then
    MODE=kaggle
    shift
fi
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/smoke_loop.XXXXXX")}"

if [ "$MODE" = local ]; then
    if command -v timeout >/dev/null 2>&1; then
        CAP=(timeout 120)
    elif command -v gtimeout >/dev/null 2>&1; then
        CAP=(gtimeout 120)
    else
        echo "smoke_loop.sh: local mode needs a timeout command to cap every stage at 120 s;" \
            "install GNU coreutils (macOS: brew install coreutils) or run --kaggle" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES=''
    export PYTORCH_ENABLE_MPS_FALLBACK=0
    BASE="$OUT/tiny_base"
    DEVICE=cpu
    RUN=(uv run)
else
    CAP=()
    RUN=(python)
    BASE=Qwen/Qwen3-0.6B
    DEVICE=cuda
fi

# Run one stage's Python script, under the 120-second cap in local mode.
stage() {
    if [ "${#CAP[@]}" -gt 0 ]; then
        "${CAP[@]}" "${RUN[@]}" "$@"
    else
        "${RUN[@]}" "$@"
    fi
}

mkdir -p "$OUT"

if [ "$MODE" = local ]; then
    echo "==> base: a tiny random Qwen3 with a small vocab"
    stage scripts/make_tiny_base.py --out "$BASE"
fi

echo "==> generate: mock teacher, 2 candidates for each of 32 training scenarios"
stage scripts/distill.py --scenarios data/scenarios/train_pool.jsonl \
    --teacher mock:compliant --k 2 --limit 32 --out "$OUT/distill"

echo "==> filter: deterministic checks and the mock judge"
stage scripts/build_dataset.py --candidates "$OUT/distill/candidates.jsonl" --judge mock \
    --out "$OUT/dataset"

echo "==> train: peft LoRA, 20 steps on $BASE ($DEVICE, batch 1)"
stage train/train_qlora.py --dataset "$OUT/dataset/train.jsonl" --out "$OUT/adapter" \
    --seed 0 --base "$BASE" --backend peft --device "$DEVICE" --batch-size 1 --max-steps 20

echo "==> eval: mock model and mock judge on 32 own-eval scenarios"
stage eval.py --model mock:compliant --eval-set data/scenarios/eval_own.jsonl \
    --judge mock --no-base --limit 32 --allow-dirty --out "$OUT/eval"

echo "smoke loop done: $OUT"
