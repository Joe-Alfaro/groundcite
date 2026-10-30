"""Training dataset v2 changes parameters only, never the config (R7, G21, D-145).

- ``train/train_qlora.py --expect-config-sha256 HEX`` exits 2, before any model work and before a
  dry run prints its plan, when the resolved config hashes to anything else. The flag is not part
  of the config hash.
- ``train/kaggle_train.ipynb`` holds ``CONFIG_SHA256`` in its parameters cell, equal to the v1
  run's ``config_sha256`` (``results/training/v1/train_log.json``), and passes it to every
  training run with ``--expect-config-sha256``.
- The v2 run is two parameter edits: ``DATASET = "data/datasets/v2/train.jsonl"`` and
  ``HUB_REPO = "<v1 repo>-v2"``. A ``data/datasets/vN/train.jsonl`` other than v1 must push to a
  repo ending in ``-vN``; anything else stops before the model run, so the public v1 adapter is
  never overwritten. The push stays private (``create_repo(private=True)``, see
  tests/test_kaggle_notebooks.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from notebook_sandbox import flag, parameters_cell, run_notebook, source
from train_data import (
    NOTEBOOK,
    REPO_ROOT,
    load_train_module,
    run_train,
    train_records,
    write_dataset,
)

V1_TRAIN_LOG = REPO_ROOT / "results" / "training" / "v1" / "train_log.json"
V1_HUB_REPO = "joe-alfaro/groundcite-qwen3-1.7b"
V2_DATASET = "data/datasets/v2/train.jsonl"
FAKE_REPO_URL = "https://example.invalid/someone/repo.git"
WRONG_SHA = "0" * 64


def _v1_config_sha() -> str:
    return str(json.loads(V1_TRAIN_LOG.read_text(encoding="utf-8"))["config_sha256"])


def _notebook() -> dict[str, Any]:
    return dict(json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _run(**params: Any) -> Any:
    return run_notebook(
        _notebook(),
        params={"KAGGLE_DATASET": "", **params},
        stop_at=lambda cmd: "train/train_qlora.py" in cmd,
        secrets={"REPO_URL": FAKE_REPO_URL},
    )


# ---------------------------------------------------------------------------------------------
# train_qlora.py --expect-config-sha256
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def env(isolated_env: Path, no_network: None, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    return isolated_env


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "data" / "train.jsonl", train_records(10))


def _argv(dataset: Path, out: Path, *extra: str) -> list[str]:
    return ["--dataset", str(dataset), "--out", str(out), "--seed", "0", *extra]


def test_the_default_config_matches_the_frozen_v1_hash(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_train(
        _argv(dataset, tmp_path / "out", "--dry-run", "--expect-config-sha256", _v1_config_sha())
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert json.loads(captured.out)["config_sha256"] == _v1_config_sha()


def test_the_expected_hash_is_not_part_of_the_config_hash(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_train(_argv(dataset, tmp_path / "a", "--dry-run"))
    plain = json.loads(capsys.readouterr().out)["config_sha256"]
    run_train(_argv(dataset, tmp_path / "b", "--dry-run", "--expect-config-sha256", plain))
    assert json.loads(capsys.readouterr().out)["config_sha256"] == plain


def test_a_dry_run_with_a_different_hash_exits_2_and_prints_no_plan(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_train(
        _argv(dataset, tmp_path / "out", "--dry-run", "--expect-config-sha256", WRONG_SHA)
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "config_sha256" in captured.err and "Traceback" not in captured.err


def test_an_override_that_changes_the_config_no_longer_matches(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--max-steps changes the resolved config, so the frozen hash no longer matches."""
    code = run_train(
        _argv(dataset, tmp_path / "out", "--dry-run", "--max-steps", "1")
        + ["--expect-config-sha256", _v1_config_sha()]
    )
    assert code == 2
    assert "config_sha256" in capsys.readouterr().err


def test_a_mismatch_stops_a_real_run_before_any_model_work(
    env: Path,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_train_module()

    def no_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("train() ran despite a config hash mismatch")

    monkeypatch.setattr(module, "train", no_training)
    out = tmp_path / "out"
    code = module.main(_argv(dataset, out, "--device", "cpu", "--expect-config-sha256", WRONG_SHA))
    assert code == 2
    assert "config_sha256" in capsys.readouterr().err
    assert not out.exists()


def test_a_malformed_expected_hash_exits_2(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_train(
        _argv(dataset, tmp_path / "out", "--dry-run", "--expect-config-sha256", "c981")
    )
    assert code == 2
    assert "expect-config-sha256" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# The training notebook
# ---------------------------------------------------------------------------------------------


def test_the_parameters_cell_holds_the_frozen_v1_config_hash() -> None:
    params = source(parameters_cell(_notebook()))
    assert f'CONFIG_SHA256 = "{_v1_config_sha()}"' in params


@pytest.mark.parametrize("curve_n", [0, 112])
def test_every_training_run_checks_the_frozen_config_hash(curve_n: int) -> None:
    run = _run(CURVE_N=curve_n)
    assert run.error is None, repr(run.error)
    assert flag(run.stopped_at or "", "--expect-config-sha256") == _v1_config_sha()


def test_v2_trains_with_only_two_parameter_edits() -> None:
    run = _run(DATASET=V2_DATASET, HUB_REPO=f"{V1_HUB_REPO}-v2")
    assert run.error is None, repr(run.error)
    command = run.stopped_at or ""
    assert flag(command, "--dataset") == V2_DATASET
    assert flag(command, "--hub-repo") == f"{V1_HUB_REPO}-v2"
    assert flag(command, "--expect-config-sha256") == _v1_config_sha()


@pytest.mark.parametrize(
    "hub_repo", [V1_HUB_REPO, f"{V1_HUB_REPO}-n112", f"{V1_HUB_REPO}-v3"], ids=str
)
def test_a_v2_dataset_never_pushes_outside_a_v2_repo(hub_repo: str) -> None:
    run = _run(DATASET=V2_DATASET, HUB_REPO=hub_repo)
    assert run.stopped_at is None, "no model run"
    assert isinstance(run.error, ValueError)
    assert "-v2" in str(run.error)


def test_the_committed_v1_run_is_unchanged() -> None:
    run = _run()
    assert run.error is None, repr(run.error)
    command = run.stopped_at or ""
    assert flag(command, "--dataset") == "data/datasets/v1/train.jsonl"
    assert flag(command, "--hub-repo") == V1_HUB_REPO
