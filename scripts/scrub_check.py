#!/usr/bin/env python3
"""Fail if any scrub term appears in tracked files, the branch name or commit messages.

Terms are read from $SCRUB_TERMS_PATH, defaulting to ../_private/scrub_terms.txt relative
to this script. One term per line; blank lines and lines starting with '#' are ignored.
Matching is a case-insensitive substring match. Hits are reported by location only
(path:line, 'branch name' or 'commit <sha>'); terms are never printed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TERMS_PATH = Path(__file__).resolve().parent.parent / "_private" / "scrub_terms.txt"


def load_terms(path: Path) -> list[str]:
    """Return lower-cased terms from ``path``, skipping blanks and comments."""
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line.lower())
    return terms


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _contains_term(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def scan_files(repo: Path, terms: list[str]) -> list[str]:
    """Return ``path:line`` for every line of a tracked file that contains a term."""
    hits = []
    for rel in _git(repo, "ls-files", "-z").split("\0"):
        path = repo / rel
        if not rel or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue  # binary file
        text = data.decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if _contains_term(line, terms):
                hits.append(f"{rel}:{number}")
    return hits


def scan_branch(repo: Path, terms: list[str]) -> list[str]:
    """Return ``['branch name']`` if the current branch name contains a term."""
    result = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--short", "-q", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    return ["branch name"] if branch and _contains_term(branch, terms) else []


def scan_commits(repo: Path, terms: list[str]) -> list[str]:
    """Return ``commit <sha>`` for every commit reachable from HEAD whose message has a term."""
    has_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "-q", "--verify", "HEAD"],
        capture_output=True,
    )
    if has_head.returncode != 0:
        return []  # no commits yet
    hits = []
    for record in _git(repo, "log", "-z", "--format=%h%n%B", "HEAD").split("\0"):
        sha, _, message = record.partition("\n")
        if sha and _contains_term(message, terms):
            hits.append(f"commit {sha}")
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="git repo to scan")
    args = parser.parse_args(argv)

    terms_path = Path(os.environ.get("SCRUB_TERMS_PATH") or DEFAULT_TERMS_PATH)
    if not terms_path.is_file():
        print(f"scrub_check: terms file not found (missing): {terms_path}", file=sys.stderr)
        return 2
    terms = load_terms(terms_path)
    if not terms:
        print(f"scrub_check: terms file has no terms: {terms_path}", file=sys.stderr)
        return 2

    repo = args.repo.resolve()
    hits = scan_files(repo, terms) + scan_branch(repo, terms) + scan_commits(repo, terms)
    if hits:
        print(f"scrub_check: {len(hits)} hit(s):", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print("scrub_check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
