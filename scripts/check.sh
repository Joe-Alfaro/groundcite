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
echo "==> scrub_check"
uv run python scripts/scrub_check.py --repo .

echo "All checks passed."
