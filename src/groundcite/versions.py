"""Code and library versions recorded in every result (R13, GOTCHA G8, G17)."""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Distributions whose versions every results.json records; optional ones only when installed.
LIBRARIES = ("groundcite", "anthropic", "openai", "python-dotenv")
OPTIONAL_LIBRARIES = ("torch", "transformers", "peft")


def _git(repo: str | Path | None, *args: str) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=Path(repo) if repo is not None else REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def git_sha(repo: str | Path | None = None) -> str:
    """``git rev-parse HEAD`` of the eval code (default: this repo)."""
    return _git(repo, "rev-parse", "HEAD")


def git_dirty(repo: str | Path | None = None) -> bool:
    """True when tracked files have changes, staged or not. Untracked files don't count, so a new
    ``results/`` run doesn't dirty the tree for the next one."""
    return bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    for name in OPTIONAL_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions
