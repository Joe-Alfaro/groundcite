#!/usr/bin/env bash
# The R5 smoke loop: generate -> filter -> train -> eval on a smoke batch, end to end.
#
#   scripts/smoke_loop.sh [OUT_DIR]
#
# Generation (teacher) and the judge are mocks, so no keys are needed. Training is real: the peft
# backend runs 20 LoRA steps on SMOKE_BASE (default Qwen/Qwen3-0.6B, downloaded once into the HF
# cache) on CUDA, MPS or CPU. eval.py loads models from the Hub, so the eval stage scores the
# mock compliant model; it proves the harness end of the loop, not the adapter.
# Outputs go to OUT_DIR (default: a new temp dir): distill/, dataset/, adapter/ and eval/.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/smoke_loop.XXXXXX")}"
BASE="${SMOKE_BASE:-Qwen/Qwen3-0.6B}"
mkdir -p "$OUT"

echo "==> generate: mock teacher, 2 candidates for each of 32 training scenarios"
uv run scripts/distill.py --scenarios data/scenarios/train_pool.jsonl --teacher mock:compliant \
    --k 2 --limit 32 --out "$OUT/distill"

echo "==> filter: deterministic checks and the mock judge"
uv run scripts/build_dataset.py --candidates "$OUT/distill/candidates.jsonl" --judge mock \
    --out "$OUT/dataset"

echo "==> train: peft LoRA, 20 steps on $BASE"
uv run train/train_qlora.py --dataset "$OUT/dataset/train.jsonl" --out "$OUT/adapter" --seed 0 \
    --base "$BASE" --backend peft --max-steps 20

echo "==> eval: mock model and mock judge on 32 own-eval scenarios"
uv run eval.py --model mock:compliant --eval-set data/scenarios/eval_own.jsonl --judge mock \
    --no-base --limit 32 --allow-dirty --out "$OUT/eval"

echo "smoke loop done: $OUT"
