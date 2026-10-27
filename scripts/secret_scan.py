#!/usr/bin/env python3
"""Fail if any tracked file holds a key-shaped string (R24).

Scans every file tracked by git, notebooks included, for OpenAI and Anthropic ``sk-`` keys,
Hugging Face ``hf_`` tokens and Google ``AIza`` keys. Untracked files, such as a git-excluded
``.env``, are out of scope. Hits are reported as ``path:line`` only; the match is never printed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The lookbehind stops a match inside a longer word ("risk-...", "task-...").
KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{32,}"),
    re.compile(r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{30,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}"),
)


def _tracked_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [rel for rel in result.stdout.split("\0") if rel]


def scan(repo: Path) -> list[str]:
    """Return ``path:line`` for every line of a tracked file that holds a key-shaped string."""
    hits = []
    for rel in _tracked_files(repo):
        path = repo / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue  # binary file
        text = data.decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in KEY_PATTERNS):
                hits.append(f"{rel}:{number}")
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="git repo to scan")
    args = parser.parse_args(argv)

    hits = scan(args.repo.resolve())
    if hits:
        print(f"secret_scan: {len(hits)} hit(s):", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print("secret_scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
