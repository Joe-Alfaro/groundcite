"""Tests for scripts/scrub_check.py, run against throwaway git repos.

Uses dummy terms only; the real terms file is never read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "scrub_check.py"

TERM = "zebracorn"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(
    root: Path,
    files: dict[str, str],
    message: str = "initial commit",
    branch: str = "main",
) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", branch)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return repo


def _write_terms(root: Path, content: str) -> Path:
    terms = root / "terms.txt"
    terms.write_text(content)
    return terms


def _run(repo: Path, terms_path: Path | None) -> subprocess.CompletedProcess[str]:
    # A missing script also exits non-zero; don't let that pass as a real failure.
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    env = dict(os.environ)
    env.pop("SCRUB_TERMS_PATH", None)
    if terms_path is not None:
        env["SCRUB_TERMS_PATH"] = str(terms_path)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo,
        timeout=60,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"


def test_missing_terms_file_fails_with_message(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "hello\n"})
    result = _run(repo, tmp_path / "does_not_exist.txt")
    assert result.returncode != 0
    assert "missing" in _output(result).lower() or "not found" in _output(result).lower()


@pytest.mark.parametrize(
    "content",
    ["", "\n\n   \n", "# only a comment\n\n# another\n"],
    ids=["empty", "blank-lines", "comments-only"],
)
def test_empty_terms_file_fails(tmp_path: Path, content: str) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "hello\n"})
    result = _run(repo, _write_terms(tmp_path, content))
    assert result.returncode != 0


def test_clean_repo_passes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "hello\n", "src/b.py": "x = 1\n"})
    terms = _write_terms(tmp_path, f"# comment\n\n{TERM}\nquuxwidget\n")
    result = _run(repo, terms)
    assert result.returncode == 0, _output(result)


def test_term_in_tracked_file_reports_path_line_without_term(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        {"a.txt": "hello\n", "docs/notes.md": "line one\nline two\nabout ZebraCorn here\n"},
    )
    terms = _write_terms(tmp_path, f"{TERM}\n")
    result = _run(repo, terms)
    out = _output(result)
    assert result.returncode != 0
    assert "docs/notes.md:3" in out
    assert TERM not in out.lower()


def test_term_match_is_case_insensitive_in_terms_file(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "a zebracorn appears\n"})
    terms = _write_terms(tmp_path, "ZEBRACORN\n")
    result = _run(repo, terms)
    out = _output(result)
    assert result.returncode != 0
    assert "a.txt:1" in out
    assert TERM not in out.lower()


def test_untracked_file_is_ignored(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "hello\n"})
    (repo / "untracked.txt").write_text(f"{TERM}\n")
    result = _run(repo, _write_terms(tmp_path, f"{TERM}\n"))
    assert result.returncode == 0, _output(result)


def test_term_in_branch_name_fails_without_printing_term(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "hello\n"}, branch=f"feat/{TERM}-work")
    result = _run(repo, _write_terms(tmp_path, f"{TERM}\n"))
    assert result.returncode != 0
    assert TERM not in _output(result).lower()


def test_term_in_commit_message_fails_without_printing_term(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"a.txt": "hello\n"}, message="initial commit")
    (repo / "b.txt").write_text("more\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add b\n\nmentions Zebracorn in body")
    (repo / "c.txt").write_text("even more\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add c")
    result = _run(repo, _write_terms(tmp_path, f"{TERM}\n"))
    assert result.returncode != 0
    assert TERM not in _output(result).lower()


def test_default_terms_path_is_private_dir_next_to_scripts() -> None:
    # Default must resolve as ../_private/scrub_terms.txt relative to scripts/.
    # Checked statically so the real terms file is never read.
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    source = SCRIPT.read_text()
    assert "SCRUB_TERMS_PATH" in source
    assert "_private" in source
    assert "scrub_terms.txt" in source
