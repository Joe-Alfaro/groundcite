#!/usr/bin/env python3
"""Pin the Kaggle notebooks' ``REPO_COMMIT`` to one commit (D-079, D-084, D-089).

    uv run scripts/repin_notebooks.py --commit SHA [--notebook PATH ...]

A notebook clones the repo from the ``REPO_URL`` secret and checks out ``REPO_COMMIT``, so the pin
must be a commit that reaches the canonical host's ``main``. Merges to ``main`` are fast-forward
only, so the script accepts only a full 40-hex SHA that is reachable from this repo's ``HEAD``.
Pushing it is a separate step.

``--notebook`` (repeatable) defaults to both Kaggle notebooks. Only the ``REPO_COMMIT = "..."``
assignment changes; every other byte of each file stays the same. Every notebook is checked before
any is written: a bad SHA, an unreachable commit or a notebook without the assignment exits 2 and
changes nothing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = (
    REPO_ROOT / "train" / "kaggle_train.ipynb",
    REPO_ROOT / "train" / "kaggle_eval.ipynb",
)
SHA_RE = re.compile(r"[0-9a-f]{40}")
# The assignment as it sits in the notebook JSON: at the start of a source string (or after an
# escaped newline), with the value in escaped double quotes.
PIN_RE = re.compile(r'((?:"|\\n)REPO_COMMIT\s*=\s*\\")([^"\\]*)(\\")')


class RepinError(ValueError):
    pass


def _reachable(commit: str) -> bool:
    done = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def repin(text: str, commit: str) -> str:
    """``text`` with every ``REPO_COMMIT`` assignment set to ``commit``."""
    new, count = PIN_RE.subn(lambda m: f"{m.group(1)}{commit}{m.group(3)}", text)
    if count == 0:
        raise RepinError("no REPO_COMMIT assignment")
    return new


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", required=True, help="full 40-hex SHA reachable from HEAD")
    parser.add_argument(
        "--notebook", action="append", type=Path, help="notebook to re-pin (repeatable)"
    )
    args = parser.parse_args(argv)
    notebooks = args.notebook or list(DEFAULT_NOTEBOOKS)
    if not SHA_RE.fullmatch(args.commit):
        print(f"error: --commit must be a full 40-hex SHA, got {args.commit!r}", file=sys.stderr)
        return 2
    if not _reachable(args.commit):
        print(f"error: {args.commit} is not a commit reachable from HEAD", file=sys.stderr)
        return 2
    updated: dict[Path, str] = {}
    for path in notebooks:
        try:
            updated[path] = repin(path.read_text(encoding="utf-8"), args.commit)
        except (OSError, RepinError) as exc:
            print(f"error: {path}: {exc}; no notebook changed", file=sys.stderr)
            return 2
    for path, text in updated.items():
        path.write_text(text, encoding="utf-8")
        print(f"{path}: REPO_COMMIT = {args.commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
