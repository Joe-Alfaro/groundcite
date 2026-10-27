"""Tests for scripts/secret_scan.py (R24), run against throwaway git repos.

CLI: secret_scan.py [--repo PATH]. It scans every tracked file, notebooks included, for
key-shaped strings (OpenAI and Anthropic ``sk-`` keys, Hugging Face ``hf_`` tokens, Google
``AIza`` keys). Exit 0 when clean, 1 on any hit. Hits are reported as ``path:line`` only; the
matched string is never printed.

Every fake key below is assembled at test time from split prefixes and a generated body, so this
file never holds a key-shaped literal and the scan stays clean on this repo.
"""

from __future__ import annotations

import json
import string
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "secret_scan.py"
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"

ALNUM = string.ascii_letters + string.digits


def _body(n: int, alphabet: str = ALNUM, step: int = 7) -> str:
    """A deterministic, non-repeating-looking fake key body."""
    return "".join(alphabet[(i * step + 3) % len(alphabet)] for i in range(n))


# label -> fake key. Prefixes are split so no literal here matches a key pattern.
FAKE_KEYS: dict[str, str] = {
    "openai": "s" + "k-" + _body(48),
    "openai-project": "s" + "k-" + "proj-" + _body(56, ALNUM + "_-"),
    "anthropic": "s" + "k-" + "ant-" + "api03-" + _body(86, ALNUM + "_-", step=5),
    "huggingface": "h" + "f_" + _body(34, string.ascii_letters, step=11),
    "google": "AI" + "za" + _body(35, ALNUM + "_-", step=13),
}

# Ordinary text that shares a prefix with a key shape but is not a key.
NOT_KEYS = [
    "token = os.environ['HF_TOKEN']  # read hf_token from the environment",
    "A risk-assessment-for-the-quarterly-maintenance-schedule is attached.",
    "Run the task-runner-configuration-loader before the build.",
    "The prefix is sk- followed by the key; AIza starts a Google key; hf_ starts a token.",
    "short: " + "s" + "k-" + "abc" + " and " + "h" + "f_" + "xyz",
]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _make_repo(root: Path, files: dict[str, str], untracked: dict[str, str] | None = None) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial commit")
    for rel, content in (untracked or {}).items():
        (repo / rel).write_text(content, encoding="utf-8")
    return repo


def _run(repo: Path) -> tuple[int, str]:
    # A missing script also exits non-zero; don't let that pass as a real failure.
    assert SCRIPT.is_file(), f"{SCRIPT.relative_to(REPO_ROOT)} is missing"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=60,
    )
    return result.returncode, result.stdout + result.stderr


def _assert_not_leaked(output: str, key: str) -> None:
    assert key not in output, "the matched key was printed"
    body = key[-16:]
    assert body not in output, "part of the matched key was printed"


def _notes(key: str) -> str:
    return f"Setup notes\n\nexport API_KEY={key}\nmore text\n"


def _notebook(key: str) -> str:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Training run\n"]},
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["import os\n", "print('loaded')\n"],
                "outputs": [{"output_type": "stream", "name": "stdout", "text": [f"{key}\n"]}],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"{SCRIPT.relative_to(REPO_ROOT)} is missing"


def test_fake_keys_are_key_shaped() -> None:
    assert FAKE_KEYS["openai"].startswith("sk-") and len(FAKE_KEYS["openai"]) > 40
    assert FAKE_KEYS["anthropic"].startswith("sk-ant-")
    assert FAKE_KEYS["huggingface"].startswith("hf_") and len(FAKE_KEYS["huggingface"]) == 37
    assert FAKE_KEYS["google"].startswith("AIza") and len(FAKE_KEYS["google"]) == 39


def test_clean_repo_exits_zero(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"README.md": "# Project\n", "src/app.py": "x = 1\n"})
    code, output = _run(repo)
    assert code == 0, output


@pytest.mark.parametrize("label", list(FAKE_KEYS))
def test_key_in_tracked_file_fails_with_path_and_line_only(tmp_path: Path, label: str) -> None:
    key = FAKE_KEYS[label]
    repo = _make_repo(tmp_path, {"README.md": "# Project\n", "docs/notes.txt": _notes(key)})
    code, output = _run(repo)
    assert code == 1, f"{label}: expected exit 1, got {code}:\n{output}"
    assert "docs/notes.txt:3" in output, output
    _assert_not_leaked(output, key)


@pytest.mark.parametrize("label", ["openai", "anthropic", "huggingface", "google"])
def test_key_in_notebook_fails(tmp_path: Path, label: str) -> None:
    key = FAKE_KEYS[label]
    repo = _make_repo(tmp_path, {"notebooks/train.ipynb": _notebook(key)})
    code, output = _run(repo)
    assert code == 1, f"{label}: expected exit 1, got {code}:\n{output}"
    lines = _notebook(key).splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if key in text)
    assert f"notebooks/train.ipynb:{line}" in output, output
    _assert_not_leaked(output, key)


def test_key_in_python_source_fails(tmp_path: Path) -> None:
    key = FAKE_KEYS["openai"]
    repo = _make_repo(tmp_path, {"src/config.py": f'API_KEY = "{key}"\n'})
    code, output = _run(repo)
    assert code == 1, output
    assert "src/config.py:1" in output
    _assert_not_leaked(output, key)


def test_every_hit_is_reported(tmp_path: Path) -> None:
    files = {
        "a.txt": _notes(FAKE_KEYS["huggingface"]),
        "b/c.md": f"line one\n{FAKE_KEYS['google']}\n",
    }
    code, output = _run(_make_repo(tmp_path, files))
    assert code == 1, output
    assert "a.txt:3" in output and "b/c.md:2" in output, output
    for key in FAKE_KEYS.values():
        _assert_not_leaked(output, key)


def test_untracked_files_are_not_scanned(tmp_path: Path) -> None:
    # Keys live in a git-excluded .env; only tracked files are in scope.
    repo = _make_repo(
        tmp_path,
        {"README.md": "# Project\n"},
        untracked={".env": f"OPENAI_API_KEY={FAKE_KEYS['openai']}\n"},
    )
    code, output = _run(repo)
    assert code == 0, output
    _assert_not_leaked(output, FAKE_KEYS["openai"])


def test_key_like_prose_is_not_flagged(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"notes.md": "\n".join(NOT_KEYS) + "\n"})
    code, output = _run(repo)
    assert code == 0, output


def test_this_repo_is_clean() -> None:
    # Also proves this test file builds its fixtures at runtime instead of holding literals.
    code, output = _run(REPO_ROOT)
    assert code == 0, output


def test_check_sh_runs_the_secret_scan() -> None:
    assert CHECK_SH.is_file(), f"{CHECK_SH} is missing"
    lines = [
        line.strip()
        for line in CHECK_SH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any("secret_scan" in line and "echo" not in line for line in lines), (
        "check.sh does not invoke scripts/secret_scan.py"
    )
