"""The Kaggle notebooks run top to bottom (D-003, D-077, D-078, R6, R9, R13, R24).

Both ``train/kaggle_train.ipynb`` and ``train/kaggle_eval.ipynb``:

- keep Kaggle's preinstalled CUDA torch: the repo installs with ``pip install --no-deps``, plus
  the explicit pinned list ``train/requirements-kaggle.txt`` (``name==version`` lines; no torch,
  torchvision, torchaudio, triton or nvidia-* packages; every runtime dependency of the project
  except torch; versions equal to uv.lock except transformers, which Unsloth caps, and
  huggingface-hub, which that transformers caps; the list resolves on Kaggle's platform). Neither
  notebook runs ``uv export`` or ``uv sync``, and no pip command names torch, triton or nvidia-*;
- set ``CUDA_VISIBLE_DEVICES=0`` before the model run (Kaggle's accelerator is "GPU T4 x2");
- keep ``REPO_URL``, ``HUB_REPO`` and the data path (``DATASET`` and ``KAGGLE_DATASET`` in the
  training notebook, ``EVAL_SET`` in the eval notebook) in the ``parameters`` cell, each marked
  with an ``# OWNER:`` comment on its line or in the comment block right above it;
- fail before any shell command, with a message naming the variable, while a placeholder (a value
  containing ``OWNER``) is still set.

The training notebook trains on ``DATASET`` from the repo at ``REPO_COMMIT``, or on
``KAGGLE_DATASET`` (a ``/kaggle/input/...`` train.jsonl) when that is non-empty. The eval notebook
runs ``eval.py --generate-only`` for ``hf:HUB_REPO`` (so the base rows, zero-shot and the best
ablation prompt, come along) on ``EVAL_SET`` and writes under ``/kaggle/working``; it holds no API
keys, because judging runs locally with ``eval.py --judge-from``.

The notebooks are executed in-process by ``tests/notebook_sandbox.py`` with every side effect
faked.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
from notebook_sandbox import flag, parameters_cell, run_notebook, source
from train_data import NOTEBOOK, REPO_ROOT

from groundcite.versions import _git

EVAL_NOTEBOOK = REPO_ROOT / "train" / "kaggle_eval.ipynb"
REQUIREMENTS = REPO_ROOT / "train" / "requirements-kaggle.txt"
SECRET_SCAN = REPO_ROOT / "scripts" / "secret_scan.py"
NOTEBOOKS = {"train": NOTEBOOK, "eval": EVAL_NOTEBOOK}
# The parameters a user must replace before the first run.
PLACEHOLDERS = ("REPO_URL", "HUB_REPO")
OWNER_MARKED = {
    "train": ("REPO_URL", "HUB_REPO", "DATASET", "KAGGLE_DATASET"),
    "eval": ("REPO_URL", "HUB_REPO", "EVAL_SET"),
}
FILLED = {"REPO_URL": "https://example.com/someone/repo.git", "HUB_REPO": "someone/gc-test"}
# What each notebook's model run looks like.
MODEL_RUN = {"train": "train/train_qlora.py", "eval": "eval.py"}
FORBIDDEN = re.compile(r"^(torch|torchvision|torchaudio|triton|nvidia-.*)$")
# Kaggle pins that may differ from uv.lock: Unsloth caps transformers (D-077), and that
# transformers caps huggingface-hub below the uv.lock version (B1).
UV_LOCK_EXEMPT = {"transformers", "huggingface-hub"}
# Declared requirement ranges of pinned releases, from their PyPI metadata (requires_dist).
KNOWN_REQUIRES = {
    ("transformers", "5.17.0"): {"huggingface-hub": ">=1.5.0,<2.0"},
}
NO_SOLUTION = "No solution found"


def _load(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"{path} is missing"
    return dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(params=sorted(NOTEBOOKS), scope="module")
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def notebook(kind: str) -> dict[str, Any]:
    return _load(NOTEBOOKS[kind])


def _code(notebook: dict[str, Any]) -> str:
    return "\n".join(source(c) for c in notebook["cells"] if c["cell_type"] == "code")


def _filled(kind: str, **extra: str) -> dict[str, str]:
    values = dict(FILLED)
    if kind == "train":
        values["KAGGLE_DATASET"] = ""
    values.update(extra)
    return values


def _run_to_model(notebook: dict[str, Any], kind: str, **extra: str) -> Any:
    run = run_notebook(
        notebook, params=_filled(kind, **extra), stop_at=lambda cmd: MODEL_RUN[kind] in cmd
    )
    assert run.error is None, f"the filled-in notebook failed: {run.error!r}"
    assert run.stopped_at is not None, f"the notebook never runs {MODEL_RUN[kind]}"
    return run


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------------------------------------
# Parameters and fail-fast placeholders
# ---------------------------------------------------------------------------------------------


def test_eval_notebook_is_valid_nbformat_4_without_outputs() -> None:
    notebook = _load(EVAL_NOTEBOOK)
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [] and cell.get("execution_count") is None


def test_parameters_cell_pins_a_full_repo_commit(notebook: dict[str, Any]) -> None:
    params = source(parameters_cell(notebook))
    assert re.search(r"^REPO_COMMIT\s*=\s*['\"][0-9a-f]{40}['\"]", params, re.MULTILINE)


def test_owner_settings_are_marked_in_the_parameters_cell(
    kind: str, notebook: dict[str, Any]
) -> None:
    lines = source(parameters_cell(notebook)).splitlines()
    for name in OWNER_MARKED[kind]:
        index = next(
            (i for i, line in enumerate(lines) if re.match(rf"{name}\s*(:[^=]*)?=", line)), None
        )
        assert index is not None, f"{name} is not set in the parameters cell"
        block = [lines[index]]
        j = index - 1
        while j >= 0 and lines[j].lstrip().startswith("#"):
            block.append(lines[j])
            j -= 1
        assert any("OWNER:" in line for line in block), f"{name} has no '# OWNER:' marker"


def test_the_committed_placeholders_contain_owner(notebook: dict[str, Any]) -> None:
    params = source(parameters_cell(notebook))
    for name in PLACEHOLDERS:
        match = re.search(rf"^{name}\s*=\s*['\"]([^'\"]*)['\"]", params, re.MULTILINE)
        assert match and "OWNER" in match.group(1), f"{name} is a placeholder until set"


@pytest.mark.parametrize("unset", PLACEHOLDERS)
def test_an_unset_placeholder_fails_fast_naming_it(
    kind: str, notebook: dict[str, Any], unset: str
) -> None:
    values = {k: v for k, v in _filled(kind).items() if k != unset}
    run = run_notebook(notebook, params=values, stop_at=lambda cmd: True)
    assert run.error is not None, "the notebook must stop while a placeholder is set"
    assert unset in str(run.error), f"the error must name {unset}: {run.error!r}"
    assert run.commands == [], "fail before any shell command (no clone, no install)"


def test_the_filled_notebook_reaches_the_model_run(kind: str, notebook: dict[str, Any]) -> None:
    run = _run_to_model(notebook, kind)
    assert any("git clone" in c and FILLED["REPO_URL"] in c for c in run.commands)


def test_cuda_visible_devices_is_0_for_the_model_run(kind: str, notebook: dict[str, Any]) -> None:
    run = _run_to_model(notebook, kind)
    assert run.stop_env.get("CUDA_VISIBLE_DEVICES") == "0", "T4 x2: use one GPU"


def test_the_notebook_names_the_kaggle_accelerator(notebook: dict[str, Any]) -> None:
    text = "\n".join(source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert "T4 x2" in text


# ---------------------------------------------------------------------------------------------
# Install: Kaggle's torch stays
# ---------------------------------------------------------------------------------------------


def _pip_installs(commands: list[str]) -> list[list[str]]:
    installs = []
    for command in commands:
        for part in re.split(r"&&|;|\|", command):
            words = shlex.split(part)
            if "pip" in " ".join(words[:4]) and "install" in words:
                installs.append(words[words.index("install") + 1 :])
    return installs


def test_install_keeps_kaggles_torch(kind: str, notebook: dict[str, Any]) -> None:
    commands = _run_to_model(notebook, kind).commands
    assert not [c for c in commands if re.search(r"\buv (export|sync)\b", c)], (
        "uv export/sync would emit torch's nvidia-* and triton pins"
    )
    installs = _pip_installs(commands)
    assert installs
    for args in installs:
        skip = False
        for word in args:
            if skip or word.startswith("-"):
                skip = word in {"-r", "--requirement", "-c", "--constraint"}
                continue
            name = re.split(r"[<>=!~\[ ]", word, maxsplit=1)[0]
            assert not FORBIDDEN.match(_norm(name)), f"installs {word}: {args}"


def test_install_uses_no_deps_for_the_repo_and_the_pinned_list(
    kind: str, notebook: dict[str, Any]
) -> None:
    installs = _pip_installs(_run_to_model(notebook, kind).commands)
    assert any("requirements-kaggle.txt" in " ".join(a) for a in installs), (
        "install train/requirements-kaggle.txt"
    )
    repo_installs = [
        a
        for a in installs
        if "--no-deps" in a and any(w in {"-e", "--editable", "."} or "/" in w for w in a)
    ]
    assert repo_installs, "install the project itself with --no-deps"


def _requirements() -> dict[str, str]:
    assert REQUIREMENTS.is_file(), f"{REQUIREMENTS} is missing"
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!_-]+)", line)
        assert match, f"not a plain name==version pin: {line!r}"
        pins[_norm(match.group(1))] = match.group(2)
    return pins


def test_requirements_list_excludes_torch_and_the_cuda_stack() -> None:
    bad = [name for name in _requirements() if FORBIDDEN.match(name)]
    assert not bad, f"Kaggle's CUDA torch stack must stay: {bad}"


def test_requirements_list_covers_the_project_runtime_and_unsloth() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = {
        _norm(re.split(r"[<>=!~\[ ;]", dep, maxsplit=1)[0])
        for dep in pyproject["project"]["dependencies"]
    }
    pins = _requirements()
    missing = sorted(runtime - {"torch"} - set(pins))
    assert not missing, f"the project is installed with --no-deps; also pin {missing}"
    assert {"unsloth", "unsloth-zoo"} <= set(pins)


def test_requirements_list_matches_uv_lock_except_transformers_and_the_hub() -> None:
    """transformers is capped by Unsloth, and huggingface-hub by that transformers (B1)."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {_norm(p["name"]): p["version"] for p in lock["package"] if "version" in p}
    drift = {
        name: (version, locked[name])
        for name, version in _requirements().items()
        if name in locked and name not in UV_LOCK_EXEMPT and version != locked[name]
    }
    assert not drift, f"pins differ from uv.lock: {drift}"


# ---------------------------------------------------------------------------------------------
# The pinned list resolves (B1): pip on Kaggle must not hit ResolutionImpossible
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("package", "version"), sorted(KNOWN_REQUIRES))
def test_pins_satisfy_the_known_declared_ranges(package: str, version: str) -> None:
    """Ranges read from the published metadata (``requires_dist``) of the pinned releases.

    transformers 5.17.0 declares ``huggingface-hub>=1.5.0,<2.0``; pinning hub 2.x makes
    ``pip install -r`` fail in both notebooks."""
    from packaging.specifiers import SpecifierSet

    pins = _requirements()
    if pins.get(package) != version:
        pytest.skip(f"{package} is not pinned at {version}; the resolve test covers it")
    for dependency, spec in KNOWN_REQUIRES[(package, version)].items():
        assert dependency in pins, f"{package}=={version} needs {dependency}{spec}; pin it"
        assert pins[dependency] in SpecifierSet(spec), (
            f"{dependency}=={pins[dependency]} breaks {package}=={version} ({dependency}{spec})"
        )


def _compile(*extra: str) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    assert uv is not None
    return subprocess.run(
        [
            uv,
            "pip",
            "compile",
            str(REQUIREMENTS),
            "--python-version",
            "3.11",
            "--python-platform",
            "x86_64-manylinux_2_28",
            "--no-header",
            "--quiet",
            "-o",
            os.devnull,
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=REPO_ROOT,
    )


def _pypi_reachable() -> bool:
    try:
        socket.create_connection(("pypi.org", 443), timeout=3).close()
    except OSError:
        return False
    return True


def test_requirements_list_resolves_on_kaggles_platform() -> None:
    """A full resolve of every pin and its transitive ranges, for Kaggle's Python and platform.

    Uses uv's cache offline first; falls back to PyPI when the cache lacks metadata, and skips
    only when neither works."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    done = _compile("--offline")
    if done.returncode != 0 and NO_SOLUTION not in done.stderr:
        if not _pypi_reachable():
            pytest.skip("uv's cache lacks the metadata and PyPI is unreachable")
        done = _compile()
    assert done.returncode == 0, f"pip would fail on Kaggle:\n{done.stderr[-2000:]}"


# ---------------------------------------------------------------------------------------------
# The training notebook's data
# ---------------------------------------------------------------------------------------------


def _params(notebook: dict[str, Any]) -> dict[str, str]:
    values = {}
    for match in re.finditer(
        r"^([A-Z_]+)\s*=\s*['\"]([^'\"]*)['\"]", source(parameters_cell(notebook)), re.MULTILINE
    ):
        values[match.group(1)] = match.group(2)
    return values


def test_training_reads_the_dataset_from_the_pinned_repo_by_default() -> None:
    notebook = _load(NOTEBOOK)
    run = _run_to_model(notebook, "train")
    dataset = flag(run.stopped_at or "", "--dataset")
    assert dataset and dataset.endswith(_params(notebook)["DATASET"])
    assert not dataset.startswith("/kaggle/input")
    commit = _params(notebook)["REPO_COMMIT"]
    assert any("checkout" in c and commit in c for c in run.commands[:-1]), (
        "train from the repo checked out at REPO_COMMIT"
    )


def test_training_reads_a_kaggle_dataset_path_when_one_is_given() -> None:
    path = "/kaggle/input/groundcite-v1/train.jsonl"
    run = _run_to_model(_load(NOTEBOOK), "train", KAGGLE_DATASET=path)
    assert flag(run.stopped_at or "", "--dataset") == path


# ---------------------------------------------------------------------------------------------
# The eval notebook: generation on the GPU only
# ---------------------------------------------------------------------------------------------


def test_eval_notebook_generates_base_and_tuned_replies_only() -> None:
    notebook = _load(EVAL_NOTEBOOK)
    command = _run_to_model(notebook, "eval").stopped_at or ""
    assert "--generate-only" in command
    assert "--judge-from" not in command
    assert "--no-base" not in command, "the base rows (zero-shot and best prompt) come along"
    model = flag(command, "--model") or ""
    assert FILLED["HUB_REPO"] in model
    eval_set = flag(command, "--eval-set") or ""
    assert eval_set.endswith(_params(notebook)["EVAL_SET"])
    assert (flag(command, "--out") or "").startswith("/kaggle/working")


def test_eval_notebook_holds_no_api_keys() -> None:
    code = _code(_load(EVAL_NOTEBOOK))
    assert "OPENAI_API_KEY" not in code and "ANTHROPIC_API_KEY" not in code


def test_eval_notebook_holds_no_key_shaped_strings() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("secret_scan", SECRET_SCAN)
    assert spec is not None and spec.loader is not None
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    text = EVAL_NOTEBOOK.read_text(encoding="utf-8")
    assert not [p.pattern for p in scan.KEY_PATTERNS if p.search(text)]


def test_kaggle_files_are_tracked() -> None:
    tracked = set(_git(REPO_ROOT, "ls-files", "train").splitlines())
    assert {"train/kaggle_eval.ipynb", "train/requirements-kaggle.txt"} <= tracked


def test_readme_documents_the_split_eval() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "kaggle_eval.ipynb" in readme
    assert "--generate-only" in readme and "--judge-from" in readme
