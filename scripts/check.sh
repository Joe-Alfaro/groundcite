#!/usr/bin/env bash
# Local quality gate. Runs each step in order and stops at the first failure.
# The scrub step reads terms from $SCRUB_TERMS_PATH (default: <repo>/_private/scrub_terms.txt).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff format --check"
uv run ruff format --check .
echo "==> ruff check"
uv run ruff check .
echo "==> mypy src"
uv run mypy src
echo "==> pytest"
uv run pytest
echo "==> leakage"
# Committed eval sets: eval_own (training families) must not leak into heldout. The report is
# deterministic, so it only changes when the data does.
uv run python scripts/check_leakage.py --train data/scenarios/eval_own.jsonl \
    --eval data/scenarios/heldout.jsonl --report data/scenarios/leakage_report.md
# Every committed training set against every eval set.
shopt -s nullglob
for train in data/datasets/*/train.jsonl; do
    uv run python scripts/check_leakage.py --train "$train" \
        --eval data/scenarios/ablation.jsonl data/scenarios/eval_own.jsonl data/scenarios/heldout.jsonl \
        --report "$(dirname "$train")/leakage_report.md"
done
shopt -u nullglob
echo "==> secret_scan"
uv run python scripts/secret_scan.py --repo .
echo "==> scrub_check"
uv run python scripts/scrub_check.py --repo .

echo "All checks passed."
