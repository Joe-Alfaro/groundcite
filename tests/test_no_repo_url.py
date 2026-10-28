"""No tracked file holds a code-repo URL (D-083 GitLab, D-084 GitHub mirror, D-091).

The canonical repo's host name must never be committed, so the Kaggle notebooks read the clone
URL from the ``REPO_URL`` Kaggle secret and nothing tracked (notebooks, README, docs, tests,
code, data) names a repo URL or a git remote:

- every ``http(s)://``, ``ssh://`` or ``git://`` URL in a tracked file points at an allowed,
  non-repo host (package indexes, the HF Hub, API endpoints, reserved example domains);
- there is no scp-style git remote (a ``git`` user, ``@``, a host, ``:``, then an owner/repo
  path);
- the old GitHub owner path is gone. The HF Hub repo ``joe-alfaro/groundcite-qwen3-1.7b`` stays.

The patterns are written so this file does not match itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

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


def test_no_tracked_file_names_a_code_host_or_the_old_github_owner() -> None:
    hits = _hits(OLD_OWNER) + _hits(CODE_HOSTS)
    assert not hits, f"a tracked file names a repo URL: {hits}"


def test_the_hub_repo_name_stays() -> None:
    """The private HF Hub repo is not a code-repo URL; the notebooks keep it."""
    texts = _tracked_texts()
    for name in ("train/kaggle_train.ipynb", "train/kaggle_eval.ipynb"):
        assert "joe-alfaro/groundcite-qwen3-1.7b" in texts[name], name
