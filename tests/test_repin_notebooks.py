"""The notebooks' REPO_COMMIT can be cloned, and one script re-pins it (B2, D-079, D-084).

A Kaggle notebook clones the public repo and checks out ``REPO_COMMIT``, so that commit must end
up on the canonical host's ``main``. Merges to ``main`` are fast-forward only, so a pin that is an
ancestor of (or equal to) the branch HEAD reaches ``main`` with the merge. The push itself is the
owner's step.

    uv run scripts/repin_notebooks.py --commit SHA [--notebook PATH ...]

``--commit`` takes a full 40-hex SHA that is a commit reachable from ``HEAD`` (``git merge-base
--is-ancestor SHA HEAD``) of the repo the script lives in. ``--notebook`` (repeatable) defaults to
``train/kaggle_train.ipynb`` and ``train/kaggle_eval.ipynb``. The script rewrites only the
``REPO_COMMIT = "..."`` assignment in each notebook's parameters cell: every other byte of the
file stays the same, so the notebook JSON keeps its layout. It exits 2 and changes no file when
the SHA is not 40 hex characters, is not reachable from HEAD, or a notebook has no
``REPO_COMMIT`` assignment. ``main(argv)`` returns the exit code.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from notebook_sandbox import parameters_cell, source
from train_data import NOTEBOOK, REPO_ROOT

EVAL_NOTEBOOK = REPO_ROOT / "train" / "kaggle_eval.ipynb"
NOTEBOOKS = (NOTEBOOK, EVAL_NOTEBOOK)
REPIN = REPO_ROOT / "scripts" / "repin_notebooks.py"
PIN_RE = re.compile(r"^REPO_COMMIT\s*=\s*['\"]([0-9a-f]{40})['\"]", re.MULTILINE)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )


def _pin(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    match = PIN_RE.search(source(parameters_cell(notebook)))
    assert match, f"{path.name}: no full-SHA REPO_COMMIT in the parameters cell"
    return match.group(1)


def _reachable_commits(count: int) -> list[str]:
    shas = _git("rev-list", f"--max-count={count}", "HEAD").stdout.split()
    if len(shas) < count:
        pytest.skip("needs a repo history of at least two commits")
    return shas


# ---------------------------------------------------------------------------------------------
# The committed pins
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_repo_commit_is_reachable_from_head(path: Path) -> None:
    """A pin off the branch history (a rebased or squashed-away commit) can't be checked out
    from a clone of main."""
    commit = _pin(path)
    assert _git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0, (
        f"{path.name}: REPO_COMMIT {commit} is not a commit in this repo"
    )
    assert _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0, (
        f"{path.name}: REPO_COMMIT {commit} is not an ancestor of HEAD; re-pin with "
        "scripts/repin_notebooks.py"
    )


def test_both_notebooks_pin_the_same_commit() -> None:
    assert _pin(NOTEBOOK) == _pin(EVAL_NOTEBOOK), "train and eval run the same code"


# ---------------------------------------------------------------------------------------------
# scripts/repin_notebooks.py
# ---------------------------------------------------------------------------------------------


def _load() -> ModuleType:
    assert REPIN.is_file(), f"{REPIN} is missing"
    spec = importlib.util.spec_from_file_location("repin_notebooks", REPIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main(argv: list[str]) -> int:
    try:
        return int(_load().main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


@pytest.fixture
def copies(tmp_path: Path) -> list[Path]:
    out = []
    for path in NOTEBOOKS:
        copy = tmp_path / path.name
        shutil.copy2(path, copy)
        out.append(copy)
    return out


def _args(commit: str, notebooks: list[Path]) -> list[str]:
    argv = ["--commit", commit]
    for path in notebooks:
        argv += ["--notebook", str(path)]
    return argv


def test_repin_sets_repo_commit_in_every_notebook(copies: list[Path]) -> None:
    target = next(c for c in _reachable_commits(2) if c != _pin(copies[0]))
    assert _main(_args(target, copies)) == 0
    for path in copies:
        assert _pin(path) == target


def test_repin_changes_only_the_repo_commit_line(copies: list[Path]) -> None:
    target = next(c for c in _reachable_commits(2) if c != _pin(copies[0]))
    before = {path: (path.read_text(encoding="utf-8"), _pin(path)) for path in copies}
    assert _main(_args(target, copies)) == 0
    for path, (text, old) in before.items():
        assert path.read_text(encoding="utf-8") == text.replace(old, target), (
            f"{path.name}: only REPO_COMMIT may change"
        )
        json.loads(path.read_text(encoding="utf-8"))


def test_repin_is_idempotent(copies: list[Path]) -> None:
    target = _reachable_commits(1)[0]
    assert _main(_args(target, copies)) == 0
    once = [p.read_text(encoding="utf-8") for p in copies]
    assert _main(_args(target, copies)) == 0
    assert [p.read_text(encoding="utf-8") for p in copies] == once


@pytest.mark.parametrize(
    "commit",
    ["abc1234", "HEAD", "g" * 40, "A" * 40, ""],
    ids=["short", "ref-name", "not-hex", "upper-case", "empty"],
)
def test_repin_refuses_a_commit_that_is_not_a_full_sha(copies: list[Path], commit: str) -> None:
    before = [p.read_text(encoding="utf-8") for p in copies]
    assert _main(_args(commit, copies)) == 2
    assert [p.read_text(encoding="utf-8") for p in copies] == before


def test_repin_refuses_a_commit_not_reachable_from_head(copies: list[Path]) -> None:
    before = [p.read_text(encoding="utf-8") for p in copies]
    assert _main(_args("0" * 40, copies)) == 2
    assert [p.read_text(encoding="utf-8") for p in copies] == before


def test_repin_refuses_a_notebook_without_repo_commit_and_changes_nothing(
    copies: list[Path], tmp_path: Path
) -> None:
    bare = tmp_path / "bare.ipynb"
    notebook = json.loads(copies[0].read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = source(cell)
        cell["source"] = re.sub(r"(?m)^REPO_COMMIT\s*=.*$", "", text)
    bare.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    before = [p.read_text(encoding="utf-8") for p in [*copies, bare]]
    target = _reachable_commits(1)[0]
    assert _main(_args(target, [*copies, bare])) == 2
    assert [p.read_text(encoding="utf-8") for p in [*copies, bare]] == before, (
        "check every notebook before writing any"
    )


def test_repin_defaults_to_both_kaggle_notebooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --notebook it targets train/kaggle_train.ipynb and train/kaggle_eval.ipynb; a
    commit that fails validation shows that without touching the real files."""
    before = [p.read_text(encoding="utf-8") for p in NOTEBOOKS]
    module = _load()
    defaults = getattr(module, "DEFAULT_NOTEBOOKS", None)
    assert defaults is not None, "expose the default list as DEFAULT_NOTEBOOKS"
    assert {Path(p).resolve() for p in defaults} == {p.resolve() for p in NOTEBOOKS}
    assert _main(["--commit", "0" * 40]) == 2
    assert [p.read_text(encoding="utf-8") for p in NOTEBOOKS] == before
