#!/usr/bin/env bash
# Release gate: run before a release, after the push and the re-pin
# (scripts/repin_notebooks.py). It runs only the release_gate tests, which scripts/check.sh
# deselects: the notebooks' REPO_COMMIT is current for every file they run, and the default
# DATASET exists at that commit.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> pytest --release-gate"
uv run pytest --release-gate "$@"

echo "Release gate passed."
