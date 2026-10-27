"""Tests for the prompt-ceiling ablation runner (R3, R11, R14, R20, GOTCHA G1, G2, G3, G6, G12).

`scripts/run_ablation.py` exposes `main(argv) -> int` and is loaded in-process so sockets stay
blocked and test backends can be registered. Pure helpers live in `groundcite.ablation`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from groundcite import backends
from groundcite.backends import BackendError, MockCompliant, MockNaive
from groundcite.config import load_models_config
from groundcite.evalset import load_eval_set
from groundcite.report import validate_transcript_row

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_ablation.py"
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
REQUIREMENTS = REPO_ROOT / "docs" / "requirements" / "small-model-finetune.md"
STRATEGIES = ("zero_shot", "few_shot", "structured")
CONFIG = load_models_config()


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ablation", SCRIPT)
    assert spec is not None and spec.loader is not None, f"{SCRIPT} is missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ablation(argv: list[str]) -> int:
    try:
        return int(_load_script().main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def run_dir(out: Path) -> Path:
    dirs = [p for p in out.iterdir() if p.is_dir()] if out.exists() else []
    assert len(dirs) == 1, f"expected one run dir in {out}, found {dirs}"
    return dirs[0]


def no_run_written(out: Path) -> bool:
    return not out.exists() or not any(out.iterdir())


def read_results(run: Path) -> dict[str, Any]:
    return json.loads((run / "results.json").read_text(encoding="utf-8"))


def read_transcripts(run: Path) -> list[dict[str, Any]]:
    lines = (run / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def mock_grid(out: Path, *extra: str, limit: int = 5) -> list[str]:
    return [
        "--model",
        "mock:compliant",
        "--model",
        "mock:naive",
        "--judge",
        "mock",
        "--eval-set",
        str(ABLATION),
        "--limit",
        str(limit),
        "--allow-dirty",
        "--out",
        str(out),
        *extra,
    ]


def _calls(stdout: str, kind: str) -> int:
    match = re.search(rf"(?mi)^\s*{kind} calls:\s*(\d+)\s*$", stdout)
    assert match, f"no '{kind} calls: N' line in:\n{stdout}"
    return int(match.group(1))


# ---------------------------------------------------------------------------------------------
# Pure helpers: reliability bar, plateau, deltas, judge family
# ---------------------------------------------------------------------------------------------


def _cell(model: str, strategy: str, spec: float, robustness: float) -> dict[str, Any]:
    return {
        "model": model,
        "strategy": strategy,
        "n": 30,
        "spec_adherence": {"mean": spec, "ci_low": spec - 0.1, "ci_high": min(spec + 0.1, 1.0)},
        "robustness_mean": robustness,
    }


def test_reliability_bar_matches_the_requirements() -> None:
    from groundcite.ablation import ROBUSTNESS_BAR, SPEC_ADHERENCE_BAR

    assert SPEC_ADHERENCE_BAR == 0.90
    assert ROBUSTNESS_BAR == 4.3
    text = REQUIREMENTS.read_text(encoding="utf-8")
    assert "**0.90**" in text and "**4.3 out of 5**" in text


def test_plateau_when_no_cell_meets_both_bars() -> None:
    from groundcite.ablation import plateau_verdict

    cells = [
        _cell("anthropic:a", "zero_shot", 0.70, 4.0),
        _cell("anthropic:a", "structured", 0.95, 4.2),  # spec ok, robustness short
        _cell("openai:b", "few_shot", 0.85, 4.6),  # robustness ok, spec short
    ]
    verdict = plateau_verdict(cells)
    assert verdict["plateau"] is True
    assert verdict["cells_meeting_bar"] == []
    assert verdict["bar"] == {"spec_adherence": 0.90, "robustness": 4.3}


def test_no_plateau_when_any_cell_meets_the_bar_inclusive() -> None:
    from groundcite.ablation import plateau_verdict

    cells = [
        _cell("anthropic:a", "zero_shot", 0.70, 4.0),
        _cell("openai:b", "structured", 0.90, 4.3),
    ]
    verdict = plateau_verdict(cells)
    assert verdict["plateau"] is False
    assert verdict["cells_meeting_bar"] == [{"model": "openai:b", "strategy": "structured"}]


def test_best_vs_second_best_delta_per_model() -> None:
    from groundcite.ablation import strategy_deltas

    cells = [
        _cell("anthropic:a", "zero_shot", 0.50, 3.0),
        _cell("anthropic:a", "few_shot", 0.70, 3.5),
        _cell("anthropic:a", "structured", 0.65, 3.9),
        _cell("openai:b", "zero_shot", 0.80, 4.0),
        _cell("openai:b", "few_shot", 0.60, 3.0),
        _cell("openai:b", "structured", 0.75, 4.1),
    ]
    deltas = strategy_deltas(cells)
    assert set(deltas) == {"anthropic:a", "openai:b"}
    a, b = deltas["anthropic:a"], deltas["openai:b"]
    assert (a["best"], a["second"]) == ("few_shot", "structured")
    assert a["delta"] == pytest.approx(0.05)
    assert (b["best"], b["second"]) == ("zero_shot", "structured")
    assert b["delta"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("spec", "family"),
    [
        ("anthropic:claude-opus-5-5", "anthropic"),
        ("openai:gpt-5.5-2026-04-23", "openai"),
        ("hf:Qwen/Qwen3-1.7B@abc", "hf"),
        ("Qwen/Qwen3-1.7B", "hf"),
        ("mock:compliant", "mock"),
        ("mock:judge", "mock"),
    ],
)
def test_model_family(spec: str, family: str) -> None:
    from groundcite.ablation import model_family

    assert model_family(spec) == family


def test_pinned_judge_shares_a_family_with_exactly_one_ablation_model() -> None:
    from groundcite.ablation import model_family

    judge_family = model_family(CONFIG["judge"]["model"])
    families = [model_family(m) for m in CONFIG["ablation"]["models"]]
    assert len(set(families)) >= 2, "ablation needs two vendor families"
    assert families.count(judge_family) == 1


# ---------------------------------------------------------------------------------------------
# Dry run (R20) and the R14 one-cell command
# ---------------------------------------------------------------------------------------------


def test_dry_run_default_is_the_full_grid_with_no_keys(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ablation"
    assert run_ablation(["--dry-run", "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    for model in CONFIG["ablation"]["models"]:
        assert model in stdout
    for strategy in STRATEGIES:
        assert strategy in stdout
    assert CONFIG["judge"]["model"] in stdout
    n = len(load_eval_set(ABLATION))
    assert _calls(stdout, "generation") == 2 * 3 * n
    assert _calls(stdout, "judge") == 2 * 3 * n
    assert no_run_written(out)


def test_dry_run_one_cell(env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ablation"
    argv = [
        "--model",
        "openai:gpt-5.5-2026-04-23",
        "--strategy",
        "few_shot",
        "--limit",
        "30",
        "--dry-run",
        "--out",
        str(out),
    ]
    assert run_ablation(argv) == 0
    stdout = capsys.readouterr().out
    assert _calls(stdout, "generation") == 30
    assert _calls(stdout, "judge") == 30
    assert "few_shot" in stdout
    assert no_run_written(out)


def test_r14_one_cell_command_runs_as_a_script(tmp_path: Path) -> None:
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN")
    }
    env["GROUNDCITE_ENV_FILE"] = str(empty_env)
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "anthropic:claude-opus-5-5",
            "--strategy",
            "few_shot",
            "--limit",
            "30",
            "--dry-run",
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert _calls(done.stdout, "generation") == 30


def test_unknown_strategy_is_a_usage_error(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "ablation"
    assert run_ablation(mock_grid(out, "--strategy", "chain_of_thought")) == 2
    assert no_run_written(out)


def test_missing_key_exits_2_naming_the_var(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ablation"
    argv = [
        "--model",
        "anthropic:claude-opus-5-5",
        "--strategy",
        "zero_shot",
        "--judge",
        "mock",
        "--limit",
        "2",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_ablation(argv) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert no_run_written(out)


# ---------------------------------------------------------------------------------------------
# A full mock grid: outputs, same scenario IDs, table contents
# ---------------------------------------------------------------------------------------------


def test_mock_grid_writes_six_cells_with_the_same_scenario_ids(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "ablation"
    assert run_ablation(mock_grid(out, limit=5)) == 0
    run = run_dir(out)
    for name in ("results.md", "results.json", "transcripts.jsonl"):
        assert (run / name).is_file(), name

    expected_ids = [s.id for s in load_eval_set(ABLATION)[:5]]
    rows = read_transcripts(run)
    assert len(rows) == 2 * 3 * 5
    by_cell: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        validate_transcript_row(row)
        by_cell.setdefault((row["model"], row["strategy"]), []).append(row["scenario_id"])
    assert set(by_cell) == {(m, s) for m in ("mock:compliant", "mock:naive") for s in STRATEGIES}
    for ids in by_cell.values():
        assert sorted(ids) == sorted(expected_ids)

    results = read_results(run)
    assert results["run_id"] == run.name
    assert results["scenario_ids"] == expected_ids
    assert len(results["cells"]) == 6
    for cell in results["cells"]:
        assert cell["n"] == 5
        assert set(cell["spec_adherence"]) == {"mean", "ci_low", "ci_high"}
        assert "robustness_mean" in cell
        assert "spec_adherence_by_category" in cell


def test_results_json_has_pins_deltas_plateau_and_judge_family_flags(
    env: Path, tmp_path: Path
) -> None:
    from groundcite.judge import rubric_sha256

    out = tmp_path / "ablation"
    assert run_ablation(mock_grid(out, limit=5)) == 0
    results = read_results(run_dir(out))

    pins = results["pins"]
    assert pins["judge_model"] == "mock:judge"
    assert pins["rubric_sha256"] == rubric_sha256()
    assert {"git_sha", "git_dirty", "eval_set", "decoding", "library_versions"} <= set(pins)

    assert set(results["deltas"]) == {"mock:compliant", "mock:naive"}
    for delta in results["deltas"].values():
        assert delta["best"] in STRATEGIES and delta["second"] in STRATEGIES
        assert delta["best"] != delta["second"]
        assert delta["delta"] >= 0

    plateau = results["plateau"]
    assert plateau["bar"] == {"spec_adherence": 0.90, "robustness": 4.3}
    assert plateau["plateau"] is True  # mocks never reach the bar on the ablation set

    # Mock subjects and the mock judge share the "mock" family.
    assert all(cell["shares_judge_family"] is True for cell in results["cells"])


def test_judge_family_flag_is_false_for_other_families(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeHF:
        def __init__(self, spec: str, transport: Any = None) -> None:
            self.name = spec

        def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
            return MockNaive().generate(messages, system, params)

    monkeypatch.setitem(backends.BACKENDS, "hf:", FakeHF)
    out = tmp_path / "ablation"
    argv = [
        "--model",
        "hf:org/some-model",
        "--model",
        "mock:compliant",
        "--strategy",
        "zero_shot",
        "--judge",
        "mock",
        "--limit",
        "2",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_ablation(argv) == 0
    flags = {c["model"]: c["shares_judge_family"] for c in read_results(run_dir(out))["cells"]}
    assert flags == {"hf:org/some-model": False, "mock:compliant": True}


def test_results_md_shows_the_table_categories_deltas_flag_and_verdict(
    env: Path, tmp_path: Path
) -> None:
    out = tmp_path / "ablation"
    assert run_ablation(mock_grid(out, limit=10)) == 0
    run = run_dir(out)
    md = (run / "results.md").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\|.*\|$", md), "results.md needs a markdown table"
    for model in ("mock:compliant", "mock:naive"):
        assert model in md
    for strategy in STRATEGIES:
        assert strategy in md
    assert "CI" in md
    assert "robustness" in md.lower()
    categories = {s.category for s in load_eval_set(ABLATION)[:10]}
    for category in categories:
        assert category is not None and category in md
    assert re.search(r"best.*second", md, re.IGNORECASE | re.DOTALL)
    assert "judge family" in md.lower()
    assert "plateau" in md.lower()
    assert "0.90" in md and "4.3" in md


# ---------------------------------------------------------------------------------------------
# Errors and refusals are scored as failures, never dropped (R3)
# ---------------------------------------------------------------------------------------------

FAIL_TEXTS = {s.messages[-1]["content"] for s in load_eval_set(ABLATION)[0:4:2]}


class Flaky:
    name = "mock:flaky"

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        if any(text in m["content"] for m in messages for text in FAIL_TEXTS):
            raise BackendError("mock:flaky: refused")
        return MockCompliant().generate(messages, system, params)


def test_errors_keep_every_cell_at_full_n(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(backends.MOCKS, "flaky", Flaky)
    out = tmp_path / "ablation"
    argv = [
        "--model",
        "mock:flaky",
        "--judge",
        "mock",
        "--limit",
        "4",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_ablation(argv) == 0
    run = run_dir(out)
    rows = read_transcripts(run)
    assert len(rows) == 3 * 4
    errored = [r for r in rows if r["generation_error"]]
    assert len(errored) == 3 * 2
    assert all(r["score"] == {"spec_adherence": 0, "robustness": 1} for r in errored)
    assert [c["n"] for c in read_results(run)["cells"]] == [4, 4, 4]


# ---------------------------------------------------------------------------------------------
# Resume (an interrupted run continues without re-paying)
# ---------------------------------------------------------------------------------------------


class Counting:
    name = "mock:counting"
    calls = 0

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        type(self).calls += 1
        return MockCompliant().generate(messages, system, params)


def _resume_argv(out: Path, *extra: str) -> list[str]:
    return [
        "--model",
        "mock:counting",
        "--strategy",
        "zero_shot",
        "--strategy",
        "structured",
        "--judge",
        "mock",
        "--limit",
        "4",
        "--allow-dirty",
        "--out",
        str(out),
        "--run-id",
        "r1",
        *extra,
    ]


def test_resume_skips_scenarios_already_transcribed(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(backends.MOCKS, "counting", Counting)
    out = tmp_path / "ablation"
    Counting.calls = 0
    assert run_ablation(_resume_argv(out)) == 0
    assert Counting.calls == 8
    run = out / "r1"
    assert run.is_dir()
    full = read_transcripts(run)

    # Simulate an interruption after three transcribed rows.
    lines = (run / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "transcripts.jsonl").write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
    (run / "results.json").unlink()
    (run / "results.md").unlink()

    Counting.calls = 0
    assert run_ablation(_resume_argv(out, "--resume")) == 0
    assert Counting.calls == 5

    resumed = read_transcripts(run)
    keys = [(r["model"], r["strategy"], r["scenario_id"]) for r in resumed]
    assert len(keys) == len(set(keys)) == 8
    assert sorted(keys) == sorted((r["model"], r["strategy"], r["scenario_id"]) for r in full)
    assert [c["n"] for c in read_results(run)["cells"]] == [4, 4]


def test_existing_run_without_resume_is_refused(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(backends.MOCKS, "counting", Counting)
    out = tmp_path / "ablation"
    assert run_ablation(_resume_argv(out)) == 0
    before = (out / "r1" / "transcripts.jsonl").read_text(encoding="utf-8")
    assert run_ablation(_resume_argv(out)) == 2
    assert (out / "r1" / "transcripts.jsonl").read_text(encoding="utf-8") == before


def test_resume_needs_a_run_id(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "ablation"
    assert run_ablation(mock_grid(out, "--resume")) == 2
