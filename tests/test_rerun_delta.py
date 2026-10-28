"""Tests for scripts/rerun_delta.py (R28): Spec-adherence delta between two runs of the same pins.

CLI: rerun_delta.py RUN_A RUN_B [--out REPORT]

Each RUN is a run directory with ``results.json`` and ``transcripts.jsonl``. The runs must have
equal ``pins`` and ``scenario_ids``; ``run_id`` and the timestamp keys ``created_at``,
``started_at``, ``finished_at`` and ``timestamp`` are ignored. Otherwise exit 2, naming the pin
that differs.

Deltas are B minus A, computed from the transcripts:
- per cell (model, strategy): mean Spec-adherence in A and B and the delta;
- per row: every (scenario_id, model, strategy) whose Spec-adherence changed.

REPORT is JSON with ``tolerance`` (0.03), ``max_abs_delta``, ``cells`` (model, strategy, a, b,
delta) and ``changed_rows`` (scenario_id, model, strategy, a, b). The same numbers are printed.
Exit 1 when any cell's |delta| is above 0.03 (exactly 0.03 passes), else 0.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "rerun_delta.py"

MODEL = "hf:org/model"
STRATEGIES = ("zero_shot", "few_shot")
N = 100
SCENARIO_IDS = [f"eo-{i:04d}" for i in range(N)]


def _pins() -> dict[str, Any]:
    return {
        "git_sha": "1" * 40,
        "git_dirty": False,
        "judge_model": "openai:gpt-test-2026-01-01",
        "rubric_sha256": "a" * 64,
        "eval_set": {"path": "data/scenarios/eval_own.jsonl", "sha256": "e" * 64},
        "models": [{"model": MODEL, "model_revision": "c" * 40}],
        "decoding": {MODEL: {"do_sample": False}, "openai:gpt-test-2026-01-01": {}},
        "bootstrap_seed": 1027,
        "library_versions": {"python": "3.11.9", "torch": "2.5.0"},
    }


def _make_run(
    root: Path,
    run_id: str,
    passes: dict[str, int],
    *,
    pins: dict[str, Any] | None = None,
    scenario_ids: list[str] | None = None,
) -> Path:
    """A run whose first ``passes[strategy]`` scenarios pass in each cell."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    ids = scenario_ids if scenario_ids is not None else SCENARIO_IDS
    rows = [
        {
            "run_id": run_id,
            "scenario_id": sid,
            "model": MODEL,
            "strategy": strategy,
            "category": "clean",
            "score": {"spec_adherence": int(i < passes[strategy]), "robustness": 3},
            "rubric_sha256": "a" * 64,
        }
        for strategy in STRATEGIES
        for i, sid in enumerate(ids)
    ]
    (run_dir / "transcripts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    cells = [
        {"model": MODEL, "strategy": s, "n": len(ids), "spec_adherence": {"mean": passes[s] / N}}
        for s in STRATEGIES
    ]
    results = {
        "run_id": run_id,
        "pins": pins if pins is not None else _pins(),
        "scenario_ids": ids,
        "rows": cells,
    }
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return run_dir


def _run(run_a: Path, run_b: Path, report: Path) -> tuple[int, str, dict[str, Any] | None]:
    assert SCRIPT.is_file(), f"{SCRIPT.name} does not exist yet"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_a), str(run_b), "--out", str(report)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    data = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
    return done.returncode, done.stdout + done.stderr, data


def _cell(data: dict[str, Any], strategy: str) -> dict[str, Any]:
    [cell] = [c for c in data["cells"] if c["model"] == MODEL and c["strategy"] == strategy]
    return cell


def test_identical_reruns_have_zero_delta(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(tmp_path, "run-b", {"zero_shot": 50, "few_shot": 70})
    code, out, data = _run(a, b, tmp_path / "delta.json")
    assert code == 0, out
    assert data is not None
    assert data["tolerance"] == 0.03
    assert data["max_abs_delta"] == 0
    assert data["changed_rows"] == []
    for strategy in STRATEGIES:
        assert _cell(data, strategy)["delta"] == 0


def test_per_cell_and_per_row_deltas(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(tmp_path, "run-b", {"zero_shot": 52, "few_shot": 69})
    code, out, data = _run(a, b, tmp_path / "delta.json")
    assert code == 0, out
    assert data is not None
    zero, few = _cell(data, "zero_shot"), _cell(data, "few_shot")
    assert zero["a"] == pytest.approx(0.50) and zero["b"] == pytest.approx(0.52)
    assert zero["delta"] == pytest.approx(0.02)
    assert few["delta"] == pytest.approx(-0.01)
    assert data["max_abs_delta"] == pytest.approx(0.02)
    changed = {(r["scenario_id"], r["strategy"], r["a"], r["b"]) for r in data["changed_rows"]}
    assert changed == {
        ("eo-0050", "zero_shot", 0, 1),
        ("eo-0051", "zero_shot", 0, 1),
        ("eo-0069", "few_shot", 1, 0),
    }
    for strategy in STRATEGIES:
        assert strategy in out


def test_delta_of_exactly_three_points_is_within_tolerance(tmp_path: Path) -> None:
    """0.53 - 0.50 is 0.030000000000000027 in floating point; it must still pass."""
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(tmp_path, "run-b", {"zero_shot": 53, "few_shot": 67})
    code, out, _ = _run(a, b, tmp_path / "delta.json")
    assert code == 0, out


def test_any_cell_above_tolerance_exits_one(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(tmp_path, "run-b", {"zero_shot": 50, "few_shot": 66})
    code, out, data = _run(a, b, tmp_path / "delta.json")
    assert code == 1, out
    assert data is not None
    assert data["max_abs_delta"] == pytest.approx(0.04)
    assert _cell(data, "few_shot")["delta"] == pytest.approx(-0.04)


def test_run_id_and_timestamps_may_differ(tmp_path: Path) -> None:
    pins_a, pins_b = _pins(), _pins()
    pins_a["started_at"], pins_b["started_at"] = "2026-10-27T10:00:00Z", "2026-10-27T12:00:00Z"
    pins_a["timestamp"], pins_b["timestamp"] = "1", "2"
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70}, pins=pins_a)
    b = _make_run(tmp_path, "run-b", {"zero_shot": 50, "few_shot": 70}, pins=pins_b)
    code, out, _ = _run(a, b, tmp_path / "delta.json")
    assert code == 0, out


@pytest.mark.parametrize(
    ("pin", "path", "value"),
    [
        ("git_sha", ("git_sha",), "2" * 40),
        ("judge_model", ("judge_model",), "openai:gpt-other-2026-02-02"),
        ("rubric_sha256", ("rubric_sha256",), "b" * 64),
        ("eval_set", ("eval_set", "sha256"), "f" * 64),
        ("models", ("models", 0, "model_revision"), "d" * 40),
        ("decoding", ("decoding", MODEL, "do_sample"), True),
        ("library_versions", ("library_versions", "torch"), "2.6.0"),
        ("git_dirty", ("git_dirty",), True),
    ],
)
def test_runs_with_different_pins_are_refused(
    tmp_path: Path, pin: str, path: tuple[Any, ...], value: Any
) -> None:
    pins_b = copy.deepcopy(_pins())
    target: Any = pins_b
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(tmp_path, "run-b", {"zero_shot": 50, "few_shot": 70}, pins=pins_b)
    code, out, _ = _run(a, b, tmp_path / "delta.json")
    assert code == 2, out
    assert pin in out, f"refusal does not name the differing pin {pin!r}:\n{out}"


def test_runs_over_different_scenarios_are_refused(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(
        tmp_path,
        "run-b",
        {"zero_shot": 50, "few_shot": 70},
        scenario_ids=[*SCENARIO_IDS[:-1], "eo-9999"],
    )
    code, out, _ = _run(a, b, tmp_path / "delta.json")
    assert code == 2, out
    assert "scenario_ids" in out


def test_missing_results_json_exits_two(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "run-a", {"zero_shot": 50, "few_shot": 70})
    b = _make_run(tmp_path, "run-b", {"zero_shot": 50, "few_shot": 70})
    (b / "results.json").unlink()
    code, out, _ = _run(a, b, tmp_path / "delta.json")
    assert code == 2, out
