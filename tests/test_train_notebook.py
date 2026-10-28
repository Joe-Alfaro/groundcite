"""The Kaggle training notebook (D-003, R6, R9, R24) and the R5 smoke loop.

``train/kaggle_train.ipynb`` is committed without outputs. A code cell tagged ``parameters``
pins ``REPO_URL`` and ``REPO_COMMIT`` (a full 40-hex SHA); the notebook clones the repo, checks
out that commit, installs from ``uv.lock``, reads ``HF_TOKEN`` from Kaggle Secrets into the
environment without printing it, and runs ``train/train_qlora.py``.

``scripts/smoke_loop.sh`` (R5) runs generate (mock teacher) -> filter (mock judge) -> train
(peft, <=20 steps) -> eval (mock judge) on a smoke batch of <=32 examples, and the README
documents it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest
from train_data import NOTEBOOK, REPO_ROOT, SMOKE_LOOP

from groundcite.versions import _git

SECRET_SCAN = REPO_ROOT / "scripts" / "secret_scan.py"


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    assert NOTEBOOK.is_file(), f"{NOTEBOOK} is missing"
    return dict(json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def _code(notebook: dict[str, Any]) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code")


def test_notebook_is_valid_nbformat_4(notebook: dict[str, Any]) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown", "raw"}
        assert isinstance(cell["source"], str | list)


def test_notebook_has_no_outputs(notebook: dict[str, Any]) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_parameter_cell_pins_the_repo_commit(notebook: dict[str, Any]) -> None:
    tagged = [
        c
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "parameters" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(tagged) == 1, "exactly one code cell tagged 'parameters'"
    params = _source(tagged[0])
    assert re.search(r"^REPO_URL\s*=\s*['\"].+['\"]", params, re.MULTILINE)
    commit = re.search(r"^REPO_COMMIT\s*=\s*['\"]([0-9a-f]{40})['\"]", params, re.MULTILINE)
    assert commit, "REPO_COMMIT must be a full 40-hex commit SHA"


def test_notebook_clones_and_checks_out_the_pinned_commit(notebook: dict[str, Any]) -> None:
    code = _code(notebook)
    assert re.search(r"git clone[^\n]*REPO_URL", code)
    assert re.search(r"git[^\n]*checkout[^\n]*REPO_COMMIT", code)


def test_notebook_installs_from_the_lockfile(notebook: dict[str, Any]) -> None:
    code = _code(notebook)
    assert re.search(r"uv (sync|export)[^\n]*--(frozen|locked)", code), (
        "install the pinned dependencies from uv.lock (uv sync --frozen, or uv export --frozen "
        "piped to pip)"
    )


def test_notebook_reads_hf_token_from_kaggle_secrets(notebook: dict[str, Any]) -> None:
    code = _code(notebook)
    assert "UserSecretsClient" in code
    assert re.search(r"get_secret\(\s*['\"]HF_TOKEN['\"]\s*\)", code)
    assert re.search(r"os\.environ\[\s*['\"]HF_TOKEN['\"]\s*\]\s*=", code)


def test_notebook_never_prints_the_token(notebook: dict[str, Any]) -> None:
    for line in _code(notebook).splitlines():
        if re.search(r"\b(print|display|echo)\b", line):
            assert not re.search(r"HF_TOKEN|get_secret|token", line, re.IGNORECASE), line
        assert "$HF_TOKEN" not in line, line


def test_notebook_runs_the_training_script(notebook: dict[str, Any]) -> None:
    code = _code(notebook)
    run = re.search(r"train/train_qlora\.py[^\n]*", code)
    assert run, "the notebook calls train/train_qlora.py"
    call = run.group(0)
    for flag in ("--dataset", "--out", "--seed", "--hub-repo"):
        assert flag in call or flag in code
    assert "--backend peft" not in code, "Kaggle trains with the default unsloth backend"


def test_notebook_holds_no_key_shaped_strings() -> None:
    """The R24 secret scan's patterns find nothing in the notebook."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("secret_scan", SECRET_SCAN)
    assert spec is not None and spec.loader is not None
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not [p.pattern for p in scan.KEY_PATTERNS if p.search(text)]


def test_notebook_and_training_files_are_tracked() -> None:
    """The secret and scrub scans only see tracked files."""
    tracked = set(_git(REPO_ROOT, "ls-files", "train").splitlines())
    expected = {"train/kaggle_train.ipynb", "train/train_qlora.py", "train/config.toml"}
    assert expected <= tracked


# ---------------------------------------------------------------------------------------------
# (7) The R5 smoke loop
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke() -> str:
    assert SMOKE_LOOP.is_file(), f"{SMOKE_LOOP} is missing"
    return SMOKE_LOOP.read_text(encoding="utf-8")


def test_smoke_loop_is_an_executable_strict_shell_script(smoke: str) -> None:
    assert smoke.startswith("#!")
    assert "set -euo pipefail" in smoke
    assert os.access(SMOKE_LOOP, os.X_OK)


def _stage(smoke: str, pattern: str) -> re.Match[str]:
    """The first command line (backslash continuations joined) matching ``pattern``."""
    joined = re.sub(r"\\\n\s*", " ", smoke)
    match = re.search(rf"^[^#\n]*{pattern}[^\n]*$", joined, re.MULTILINE)
    assert match, f"no command matching {pattern!r}"
    return match


def test_smoke_loop_runs_generate_filter_train_eval_in_order(smoke: str) -> None:
    generate = _stage(smoke, r"scripts/distill\.py")
    filter_ = _stage(smoke, r"scripts/build_dataset\.py")
    train = _stage(smoke, r"train/train_qlora\.py")
    evaluate = _stage(smoke, r"eval\.py")
    assert generate.start() < filter_.start() < train.start() < evaluate.start()


def test_smoke_loop_uses_mocks_peft_and_at_most_20_steps(smoke: str) -> None:
    generate = _stage(smoke, r"scripts/distill\.py").group(0)
    assert re.search(r"--teacher[ =]['\"]?mock", generate)
    limit = re.search(r"--limit[ =](\d+)", generate)
    assert limit and 1 <= int(limit.group(1)) <= 32, "a smoke batch of <=32 scenarios"

    assert re.search(r"--judge[ =]['\"]?mock", _stage(smoke, r"scripts/build_dataset\.py").group(0))

    train = _stage(smoke, r"train/train_qlora\.py").group(0)
    assert "--backend peft" in train
    steps = re.search(r"--max-steps[ =](\d+)", train)
    assert steps and 1 <= int(steps.group(1)) <= 20

    assert re.search(r"--judge[ =]['\"]?mock", _stage(smoke, r"eval\.py").group(0))


def test_readme_documents_the_smoke_loop() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/smoke_loop.sh" in readme


def test_readme_documents_the_training_rerun_command() -> None:
    """R14: one curve point reruns from a published subset file."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"uv run train/train_qlora\.py --dataset \S*subsets/n\S*\.jsonl", readme)
    assert "kaggle_train.ipynb" in readme
