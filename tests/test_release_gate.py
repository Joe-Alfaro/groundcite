"""Release gate: the notebooks' ``REPO_COMMIT`` is never stale (B1) and its dataset exists (B3).

A Kaggle notebook clones the repo and checks out ``REPO_COMMIT``, so the code it runs is that
commit's, not HEAD's. These tests hold only once the pin has been moved to a pushed commit at or
after every change to that code, which happens after the push (``scripts/repin_notebooks.py``).
So they are marked ``release_gate``: collected but deselected by a plain ``pytest`` (and so by
``scripts/check.sh``), and run by ``scripts/release_check.sh`` before a release.

- **B1:** in every notebook (the two Kaggle notebooks and the inference demo) ``REPO_COMMIT`` is
  an ancestor of (or equal to) HEAD, and no file the notebooks run differs between ``REPO_COMMIT``
  and HEAD: ``train/*.py``, ``train/config.toml``, ``train/requirements-kaggle.txt``,
  ``scripts/smoke_loop.sh`` and every script it stages, ``eval.py``, ``pyproject.toml`` (the
  editable install), ``src/groundcite/**``, and the data files the notebooks' parameters name
  (``DATASET``, ``EVAL_SET``).
- **B3:** the training notebook's default ``DATASET`` is a file in the repo at ``REPO_COMMIT``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from notebook_sandbox import parameters_cell, source
from train_data import NOTEBOOK, REPO_ROOT

pytestmark = pytest.mark.release_gate

EVAL_NOTEBOOK = REPO_ROOT / "train" / "kaggle_eval.ipynb"
# The public demo installs the same pinned commit, so it goes stale the same way.
DEMO_NOTEBOOK = REPO_ROOT / "train" / "demo_inference.ipynb"
NOTEBOOKS = (NOTEBOOK, EVAL_NOTEBOOK, DEMO_NOTEBOOK)
SMOKE_LOOP = "scripts/smoke_loop.sh"
# Code a notebook runs from its clone, as git pathspecs.
RUN_PATHS = (
    "train/*.py",
    "train/config.toml",
    "train/requirements-kaggle.txt",
    SMOKE_LOOP,
    "eval.py",
    "pyproject.toml",
    "src/groundcite",
)
# Parameters that name a repo-relative data file the notebook reads.
DATA_PARAMS = ("DATASET", "EVAL_SET")
# `stage <script>` lines in the smoke loop.
STAGE_RE = re.compile(r"^\s*stage\s+(\S+\.py)\b", re.MULTILINE)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )


def _params(path: Path) -> dict[str, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = source(parameters_cell(notebook))
    return dict(re.findall(r"(?m)^([A-Z_]+)\s*=\s*['\"]([^'\"]*)['\"]", text))


def _pin(path: Path) -> str:
    commit = _params(path).get("REPO_COMMIT", "")
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"{path.name}: no full-SHA REPO_COMMIT"
    return commit


def _run_paths() -> list[str]:
    """The pathspecs above plus every script the smoke loop stages (at HEAD)."""
    staged = STAGE_RE.findall((REPO_ROOT / SMOKE_LOOP).read_text(encoding="utf-8"))
    data = [v for nb in NOTEBOOKS for k, v in _params(nb).items() if k in DATA_PARAMS and v]
    return sorted({*RUN_PATHS, *staged, *data})


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_repo_commit_is_head_or_an_ancestor_of_it(path: Path) -> None:
    commit = _pin(path)
    assert _git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0, (
        f"{path.name}: REPO_COMMIT {commit} is not a commit in this repo"
    )
    assert _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0, (
        f"{path.name}: REPO_COMMIT {commit} is not HEAD or an ancestor of it"
    )


def test_the_smoke_loop_stages_scripts_the_gate_tracks() -> None:
    """Guards the STAGE_RE parse: if it found nothing, staged scripts would go unchecked."""
    staged = STAGE_RE.findall((REPO_ROOT / SMOKE_LOOP).read_text(encoding="utf-8"))
    assert "train/train_qlora.py" in staged and "eval.py" in staged, staged


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_file_the_notebook_runs_changed_since_repo_commit(path: Path) -> None:
    """A pin that predates a fix runs the unfixed code on Kaggle (B1)."""
    commit = _pin(path)
    done = _git("diff", "--name-only", commit, "HEAD", "--", *_run_paths())
    assert done.returncode == 0, f"git diff failed: {done.stderr.strip()}"
    stale = done.stdout.split()
    assert not stale, (
        f"{path.name}: REPO_COMMIT {commit[:12]} is stale; these files differ at HEAD: {stale}. "
        "Push, then re-pin with scripts/repin_notebooks.py"
    )


def test_the_default_dataset_exists_at_repo_commit() -> None:
    """An unedited training run reads DATASET from the clone at REPO_COMMIT (B3)."""
    params = _params(NOTEBOOK)
    commit, dataset = _pin(NOTEBOOK), params.get("DATASET", "")
    assert dataset, "the training notebook names a default DATASET"
    assert _git("cat-file", "-e", f"{commit}:{dataset}").returncode == 0, (
        f"DATASET {dataset} is not in the repo at REPO_COMMIT {commit[:12]}"
    )
