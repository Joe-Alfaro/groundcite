"""No tracked file holds a code-repo URL or names a code host (D-083, D-084, D-091).

The canonical repo's host name must never be committed, so the Kaggle notebooks read the clone
URL from the ``REPO_URL`` Kaggle secret and nothing tracked (notebooks, README, docs, tests,
code, data) names a repo URL or a git remote:

- every ``http(s)://``, ``ssh://`` or ``git://`` URL in a tracked file points at an allowed,
  non-repo host (package indexes, the HF Hub, API endpoints, reserved example domains);
- there is no scp-style git remote (a ``git`` user, ``@``, a host, ``:``, then an owner/repo
  path);
- the old owner path is gone. The HF Hub repo ``joe-alfaro/groundcite-qwen3-1.7b`` stays;
- no tracked file names a code-hosting service at all, in any case, as a bare word or inside an
  identifier: docs, comments and code say "the canonical host" instead (B2).

The patterns are written so this file does not match itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Hosts a tracked file may link to: none of them hosts a code repo.
ALLOWED_HOSTS = {
    "files.pythonhosted.org",
    "pypi.org",
    "huggingface.co",
    "docs.astral.sh",
    "api.openai.com",
    "api.anthropic.com",
    "example.com",
    "example.invalid",
}
URL = re.compile(r"\b(?:https?|ssh|git)://(?:[^/\s@\"'<>]+@)?([A-Za-z0-9.-]+)")
SCP_REMOTE = re.compile(r"\bgit@[A-Za-z0-9.-]+:[A-Za-z0-9_.~-]+/")
OLD_OWNER = re.compile("Joe" + "-" + "Alfaro")
CODE_HOSTS = re.compile(r"git(hub|lab)\.[a-z]{2,}", re.IGNORECASE)
# Code-hosting service names, split so this file does not match itself. A letter may not touch
# either end (so "digital" or "legit" pass), but "_" and digits may ("old_<name>_owner" fails).
CODE_HOST_NAMES = (
    "git" + "hub",
    "git" + "lab",
    "bit" + "bucket",
    "code" + "berg",
    "source" + "hut",
    "git" + "ea",
    "git" + "ee",
    "source" + "forge",
)
CODE_HOST_NAME = re.compile(
    r"(?<![a-z])(?:" + "|".join(CODE_HOST_NAMES) + r")(?![a-z])", re.IGNORECASE
)


def _tracked_texts() -> dict[str, str]:
    names = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    texts = {}
    for name in filter(None, names):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary
        texts[name] = data.decode("utf-8", errors="replace")
    return texts


def _hits(pattern: re.Pattern[str], allowed: set[str] | None = None) -> list[str]:
    hits = []
    for name, text in _tracked_texts().items():
        for number, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                host = match.group(1).lower().rstrip(".") if match.groups() else ""
                if allowed is None or host not in allowed:
                    hits.append(f"{name}:{number}")
    return hits


def test_every_tracked_url_points_at_an_allowed_non_repo_host() -> None:
    hits = _hits(URL, ALLOWED_HOSTS)
    assert not hits, f"a tracked file links to a host outside the allowlist: {hits}"


def test_no_tracked_file_holds_an_scp_style_git_remote() -> None:
    hits = _hits(SCP_REMOTE)
    assert not hits, f"a tracked file holds a git remote: {hits}"


def test_no_tracked_file_names_a_code_host_domain_or_the_old_owner() -> None:
    hits = _hits(OLD_OWNER) + _hits(CODE_HOSTS)
    assert not hits, f"a tracked file names a repo URL: {hits}"


def test_no_tracked_file_names_a_code_hosting_service() -> None:
    """B2: a bare service name (any case, in prose, comments or identifiers) is out too."""
    hits = _hits(CODE_HOST_NAME)
    assert not hits, f"a tracked file names a code-hosting service: {hits}"


@pytest.mark.parametrize(
    "line",
    ["see the HUB mirror", "a digital twin", "legit code", "the canonical host's main"],
)
def test_the_code_host_name_pattern_passes_ordinary_words(line: str) -> None:
    assert not CODE_HOST_NAME.search(line), line


@pytest.mark.parametrize("name", CODE_HOST_NAMES)
def test_the_code_host_name_pattern_catches_each_name_in_any_case(name: str) -> None:
    for line in (name, name.upper(), name.title(), f"old_{name}_owner", f"({name})", f"{name}.io"):
        assert CODE_HOST_NAME.search(line), line


def test_the_hub_repo_name_stays() -> None:
    """The private HF Hub repo is not a code-repo URL; the notebooks keep it."""
    texts = _tracked_texts()
    for name in ("train/kaggle_train.ipynb", "train/kaggle_eval.ipynb"):
        assert "joe-alfaro/groundcite-qwen3-1.7b" in texts[name], name
