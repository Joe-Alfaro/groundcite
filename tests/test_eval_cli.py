"""Tests for the one-command eval (R10, R11, R13, R19, R20, R25, R26, GOTCHA G15, G16, G17, G39).

`eval.py` at the repo root is a thin wrapper around `groundcite.cli:main(argv) -> int`. Every test
here runs with sockets blocked and no keys; the Hub lookups (`groundcite.hub`) and the `hf:`
backend are replaced with fakes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from groundcite import backends
from groundcite.backends import BackendError, MockCompliant, MockNaive
from groundcite.config import load_models_config
from groundcite.evalset import load_eval_set
from groundcite.judge import rubric_sha256
from groundcite.report import validate_transcript_row

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "data" / "scenarios"
ABLATION = SCENARIOS / "ablation.jsonl"
EVAL_OWN = SCENARIOS / "eval_own.jsonl"
JUDGE_MODEL = load_models_config()["judge"]["model"]


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


def run_cli(argv: list[str]) -> int:
    from groundcite.cli import main

    try:
        return main(argv)
    except SystemExit as exc:  # argparse errors
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


def cells(results: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(r["model"], r["strategy"]) for r in results["rows"]]


def mock_run(out: Path, *extra: str, limit: int = 4) -> list[str]:
    return [
        "--model",
        "mock:compliant",
        "--model",
        "mock:naive",
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--no-base",
        "--limit",
        str(limit),
        "--allow-dirty",
        "--out",
        str(out),
        *extra,
    ]


# ---------------------------------------------------------------------------------------------
# eval.py is a thin entry point
# ---------------------------------------------------------------------------------------------


def test_eval_py_is_a_thin_wrapper_over_groundcite_cli() -> None:
    script = REPO_ROOT / "eval.py"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "groundcite.cli" in text
    code_lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(code_lines) <= 25, "eval.py should only call groundcite.cli:main"


def test_eval_py_help_lists_every_flag() -> None:
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    for flag in (
        "--model",
        "--eval-set",
        "--judge",
        "--base",
        "--no-base",
        "--strategy",
        "--limit",
        "--dry-run",
        "--allow-dirty",
        "--out",
    ):
        assert flag in done.stdout, flag


# ---------------------------------------------------------------------------------------------
# Outputs: results.md, results.json, transcripts.jsonl (R10, R11)
# ---------------------------------------------------------------------------------------------


def test_writes_the_three_result_files_under_one_run_dir(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out)) == 0
    run = run_dir(out)
    for name in ("results.md", "results.json", "transcripts.jsonl"):
        assert (run / name).is_file(), name
    results = read_results(run)
    assert results["run_id"] == run.name
    assert all(row["run_id"] == run.name for row in read_transcripts(run))


def test_transcripts_have_one_valid_row_per_scenario_and_system(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out, limit=5)) == 0
    rows = read_transcripts(run_dir(out))
    assert len(rows) == 5 * 2
    for row in rows:
        validate_transcript_row(row)
        assert row["judge_model"] == "mock:judge"
        assert row["rubric_sha256"] == rubric_sha256()
        assert "generation_error" in row
    first_five = [s.id for s in load_eval_set(ABLATION)[:5]]
    for model in ("mock:compliant", "mock:naive"):
        assert sorted(r["scenario_id"] for r in rows if r["model"] == model) == sorted(first_five)


def test_results_rows_are_aggregated_per_system(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out, limit=6)) == 0
    results = read_results(run_dir(out))
    assert cells(results) == [("mock:compliant", "zero_shot"), ("mock:naive", "zero_shot")]
    for row in results["rows"]:
        assert row["n"] == 6
        assert set(row["spec_adherence"]) == {"mean", "ci_low", "ci_high"}
        assert "robustness_mean" in row
        assert "invented_id_rate" in row
        assert "spec_adherence_by_category" in row


def test_results_md_has_a_table_with_every_system_and_the_pins(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out)) == 0
    run = run_dir(out)
    md = (run / "results.md").read_text(encoding="utf-8")
    pins = read_results(run)["pins"]
    assert re.search(r"(?m)^\|.*\|$", md), "results.md needs a markdown table"
    for model in ("mock:compliant", "mock:naive"):
        assert model in md
    assert pins["git_sha"] in md
    assert pins["rubric_sha256"] in md
    assert "mock:judge" in md


def test_limit_takes_the_first_n_scenarios(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out, limit=3)) == 0
    rows = read_transcripts(run_dir(out))
    expected = {s.id for s in load_eval_set(ABLATION)[:3]}
    assert {r["scenario_id"] for r in rows} == expected


def test_strategy_flag_sets_the_prompt_strategy(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:compliant",
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--no-base",
        "--strategy",
        "structured",
        "--limit",
        "2",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 0
    run = run_dir(out)
    assert cells(read_results(run)) == [("mock:compliant", "structured")]
    assert {r["strategy"] for r in read_transcripts(run)} == {"structured"}


def test_unknown_strategy_is_a_usage_error(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out, "--strategy", "chain_of_thought")) == 2
    assert no_run_written(out)


# ---------------------------------------------------------------------------------------------
# R13 pins in results.json
# ---------------------------------------------------------------------------------------------


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_results_json_carries_the_r13_pins(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert run_cli(mock_run(out)) == 0
    pins = read_results(run_dir(out))["pins"]
    assert pins["git_sha"] == _git_head()
    assert isinstance(pins["git_dirty"], bool)
    assert pins["judge_model"] == "mock:judge"
    assert pins["rubric_sha256"] == rubric_sha256()
    assert pins["eval_set"]["sha256"] == hashlib.sha256(ABLATION.read_bytes()).hexdigest()
    assert Path(pins["eval_set"]["path"]).name == ABLATION.name
    models = {m["model"]: m for m in pins["models"]}
    assert set(models) == {"mock:compliant", "mock:naive"}
    assert all("model_revision" in m for m in models.values())
    # Effective decoding params per system and for the judge (D-037).
    assert {"mock:compliant", "mock:naive", "mock:judge"} <= set(pins["decoding"])
    assert all(isinstance(v, dict) for v in pins["decoding"].values())
    versions = pins["library_versions"]
    assert {"python", "groundcite", "anthropic", "openai"} <= set(versions)
    assert all(isinstance(v, str) and v for v in versions.values())


def test_default_judge_is_the_pinned_config_judge(env: Path, tmp_path: Path) -> None:
    """Without --judge, the pinned (Opus) judge is used, so a missing key stops the run."""
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:compliant",
        "--eval-set",
        str(ABLATION),
        "--no-base",
        "--limit",
        "2",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert JUDGE_MODEL.startswith("anthropic:")
    assert run_cli(argv) == 2
    assert no_run_written(out)


# ---------------------------------------------------------------------------------------------
# Dirty tree (R13, GOTCHA G17)
# ---------------------------------------------------------------------------------------------


def _set_dirty(monkeypatch: pytest.MonkeyPatch, dirty: bool) -> None:
    from groundcite import versions

    monkeypatch.setattr(versions, "git_dirty", lambda *a, **k: dirty)


def test_refuses_a_dirty_tree_without_allow_dirty(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_dirty(monkeypatch, True)
    out = tmp_path / "results"
    argv = [a for a in mock_run(out) if a != "--allow-dirty"]
    assert run_cli(argv) == 2
    err = capsys.readouterr().err
    assert "dirty" in err.lower() and "--allow-dirty" in err
    assert no_run_written(out)


def test_allow_dirty_runs_and_records_it(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_dirty(monkeypatch, True)
    out = tmp_path / "results"
    assert run_cli(mock_run(out)) == 0
    assert read_results(run_dir(out))["pins"]["git_dirty"] is True


def test_clean_tree_runs_without_allow_dirty(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_dirty(monkeypatch, False)
    out = tmp_path / "results"
    argv = [a for a in mock_run(out) if a != "--allow-dirty"]
    assert run_cli(argv) == 0
    assert read_results(run_dir(out))["pins"]["git_dirty"] is False


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_versions_git_state_on_a_real_repo(tmp_path: Path) -> None:
    from groundcite.versions import git_dirty, git_sha

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")

    assert git_sha(repo) == _git(repo, "rev-parse", "HEAD")
    assert git_dirty(repo) is False
    # Untracked files (for example a new results/ run) don't make the tree dirty.
    (repo / "new.txt").write_text("x\n", encoding="utf-8")
    assert git_dirty(repo) is False
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    assert git_dirty(repo) is True
    _git(repo, "add", "a.txt")
    assert git_dirty(repo) is True


# ---------------------------------------------------------------------------------------------
# Fail fast: malformed eval set, missing key (R12, R25)
# ---------------------------------------------------------------------------------------------


def test_malformed_eval_set_exits_2_with_line_and_field(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "ok-1", "prompt": "Engine idles rough. [A-1] Check the hose."})
        + "\n"
        + json.dumps({"id": "", "prompt": "No id here."})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:compliant",
        "--eval-set",
        str(bad),
        "--judge",
        "mock",
        "--no-base",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 2
    err = capsys.readouterr().err
    assert "line 2" in err
    assert "id" in err
    assert no_run_written(out)


@pytest.mark.parametrize(
    ("model", "var"),
    [
        ("openai:gpt-5.5-2026-04-23", "OPENAI_API_KEY"),
        ("anthropic:claude-opus-5-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_missing_model_key_exits_2_naming_the_var(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], model: str, var: str
) -> None:
    out = tmp_path / "results"
    argv = [
        "--model",
        model,
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--limit",
        "2",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 2
    assert var in capsys.readouterr().err
    assert no_run_written(out)


def test_missing_judge_key_exits_2_naming_the_var(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:compliant",
        "--eval-set",
        str(ABLATION),
        "--judge",
        JUDGE_MODEL,
        "--no-base",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert no_run_written(out)


# ---------------------------------------------------------------------------------------------
# Dry run (R20)
# ---------------------------------------------------------------------------------------------


def test_dry_run_prints_the_plan_needs_no_keys_and_writes_nothing(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_dirty(monkeypatch, True)  # a plan is fine on a dirty tree
    out = tmp_path / "results"
    argv = [
        "--model",
        "openai:gpt-5.5-2026-04-23",
        "--model",
        "anthropic:claude-opus-5-5",
        "--eval-set",
        str(ABLATION),
        "--no-base",
        "--limit",
        "3",
        "--out",
        str(out),
        "--dry-run",
    ]
    assert run_cli(argv) == 0
    stdout = capsys.readouterr().out
    assert "openai:gpt-5.5-2026-04-23" in stdout
    assert "anthropic:claude-opus-5-5" in stdout
    assert JUDGE_MODEL in stdout
    # 3 scenarios x 2 systems
    assert re.search(r"(?mi)^\s*generation calls:\s*6\s*$", stdout)
    assert re.search(r"(?mi)^\s*judge calls:\s*6\s*$", stdout)
    assert no_run_written(out)


# ---------------------------------------------------------------------------------------------
# Subject-model errors and refusals are scored, never dropped (R3, R11)
# ---------------------------------------------------------------------------------------------


# The 2nd, 4th and 6th ablation scenarios fail; matched by their final user message so retries or
# call order can't change which rows error.
FAIL_TEXTS = {s.messages[-1]["content"] for s in load_eval_set(ABLATION)[1:6:2]}


class Flaky:
    """Always fails with a BackendError on the FAIL_TEXTS scenarios."""

    name = "mock:flaky"

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        if any(text in m["content"] for m in messages for text in FAIL_TEXTS):
            raise BackendError("mock:flaky: HTTP 529 overloaded")
        return MockCompliant().generate(messages, system, params)


class Refuser:
    """Returns an empty reply, the way a refused API response surfaces."""

    name = "mock:refuser"

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        return ""


def test_errors_and_refusals_are_scored_as_failures(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(backends.MOCKS, "flaky", Flaky)
    monkeypatch.setitem(backends.MOCKS, "refuser", Refuser)
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:flaky",
        "--model",
        "mock:refuser",
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--no-base",
        "--limit",
        "6",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 0
    run = run_dir(out)
    rows = read_transcripts(run)
    assert len(rows) == 6 * 2, "row count = scenarios x systems"
    for row in rows:
        validate_transcript_row(row)

    flaky = [r for r in rows if r["model"] == "mock:flaky"]
    errored = [r for r in flaky if r["generation_error"]]
    assert len(errored) == 3
    for row in errored:
        assert "529" in row["generation_error"]
        assert row["score"] == {"spec_adherence": 0, "robustness": 1}
        assert row["reasoning"].strip()
    assert all(r["generation_error"] is None for r in flaky if r not in errored)

    refused = [r for r in rows if r["model"] == "mock:refuser"]
    assert len(refused) == 6
    assert all(r["score"]["spec_adherence"] == 0 for r in refused)

    results = read_results(run)
    assert [row["n"] for row in results["rows"]] == [6, 6]


# ---------------------------------------------------------------------------------------------
# R26 / R19: mock compliant vs naive, end to end, sockets blocked
# ---------------------------------------------------------------------------------------------


def test_r26_mock_compliant_beats_naive_on_eval_own(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:compliant",
        "--model",
        "mock:naive",
        "--eval-set",
        str(EVAL_OWN),
        "--judge",
        "mock",
        "--no-base",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 0
    run = run_dir(out)
    results = read_results(run)
    assert len(results["rows"]) == 2
    by_model = {r["model"]: r for r in results["rows"]}
    compliant = by_model["mock:compliant"]["spec_adherence"]["mean"]
    naive = by_model["mock:naive"]["spec_adherence"]["mean"]
    assert compliant > naive
    n = len(load_eval_set(EVAL_OWN))
    assert len(read_transcripts(run)) == 2 * n


def test_single_mock_model_gets_no_automatic_base_rows(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    argv = [
        "--model",
        "mock:compliant",
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--limit",
        "2",
        "--allow-dirty",
        "--out",
        str(out),
    ]
    assert run_cli(argv) == 0
    assert cells(read_results(run_dir(out))) == [("mock:compliant", "zero_shot")]


# ---------------------------------------------------------------------------------------------
# Automatic base rows for one hf: model (R10, GOTCHA G16, G39)
# ---------------------------------------------------------------------------------------------

BASE_REPO = "Qwen/Qwen3-1.7B"
TUNED = "hf:org/groundcite-tuned@rev1"


def fake_sha(repo: str, revision: str | None) -> str:
    return hashlib.sha1(f"{repo}@{revision}".encode()).hexdigest()


class FakeHF:
    """Stands in for HFBackend: the tuned repo answers on-spec, anything else off-spec."""

    def __init__(self, spec: str, transport: Any = None) -> None:
        self.name = spec
        self._inner = MockCompliant() if "tuned" in spec else MockNaive()

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        return self._inner.generate(messages, system, params)


def _patch_hub(
    monkeypatch: pytest.MonkeyPatch,
    adapter_config: Callable[[str, str | None], dict[str, Any] | None],
) -> list[tuple[str, str | None]]:
    from groundcite import hub

    fetched: list[tuple[str, str | None]] = []

    def fetch(repo: str, revision: str | None = None) -> dict[str, Any] | None:
        fetched.append((repo, revision))
        return adapter_config(repo, revision)

    monkeypatch.setattr(hub, "fetch_adapter_config", fetch)
    monkeypatch.setattr(
        hub, "resolve_revision", lambda repo, revision=None: fake_sha(repo, revision)
    )
    monkeypatch.setitem(backends.BACKENDS, "hf:", FakeHF)
    return fetched


def _adapter(repo: str, revision: str | None) -> dict[str, Any] | None:
    return {"base_model_name_or_path": BASE_REPO, "peft_type": "LORA"}


def _hf_argv(out: Path, *extra: str) -> list[str]:
    return [
        "--model",
        TUNED,
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--limit",
        "3",
        "--allow-dirty",
        "--out",
        str(out),
        *extra,
    ]


def test_best_strategy_for_the_base_row_is_pinned_in_config() -> None:
    best = load_models_config()["eval"]["best_strategy"]
    assert best in ("few_shot", "structured")


def test_one_hf_model_gets_three_rows_with_the_base_from_adapter_config(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched = _patch_hub(monkeypatch, _adapter)
    out = tmp_path / "results"
    assert run_cli(_hf_argv(out)) == 0
    assert fetched and fetched[0] == ("org/groundcite-tuned", "rev1")
    run = run_dir(out)
    results = read_results(run)
    best = load_models_config()["eval"]["best_strategy"]
    base = f"hf:{BASE_REPO}"
    assert cells(results) == [(base, "zero_shot"), (base, best), (TUNED, "zero_shot")]
    assert len(read_transcripts(run)) == 3 * 3

    revisions = {m["model"]: m["model_revision"] for m in results["pins"]["models"]}
    assert revisions[TUNED] == fake_sha("org/groundcite-tuned", "rev1")
    assert revisions[base] == fake_sha(BASE_REPO, None)
    for row in read_transcripts(run):
        assert row["model_revision"] == revisions[row["model"]]


def test_base_flag_overrides_adapter_config(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_fetch(repo: str, revision: str | None) -> dict[str, Any] | None:
        raise AssertionError("--base given, adapter_config.json must not be fetched")

    _patch_hub(monkeypatch, no_fetch)
    out = tmp_path / "results"
    assert run_cli(_hf_argv(out, "--base", "hf:other/base-model")) == 0
    models = [m for m, _ in cells(read_results(run_dir(out)))]
    assert models == ["hf:other/base-model", "hf:other/base-model", TUNED]


def test_no_base_turns_the_base_rows_off(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hub(monkeypatch, _adapter)
    out = tmp_path / "results"
    assert run_cli(_hf_argv(out, "--no-base")) == 0
    assert cells(read_results(run_dir(out))) == [(TUNED, "zero_shot")]


def test_merged_model_without_adapter_config_needs_base(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_hub(monkeypatch, lambda repo, revision: None)
    out = tmp_path / "results"
    assert run_cli(_hf_argv(out)) == 2
    err = capsys.readouterr().err
    assert "--base" in err and "--no-base" in err
    assert no_run_written(out)


def test_dry_run_with_explicit_base_plans_three_systems_without_hub_calls(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from groundcite import hub

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dry run must not touch the Hub")

    monkeypatch.setattr(hub, "fetch_adapter_config", boom)
    monkeypatch.setattr(hub, "resolve_revision", boom)
    out = tmp_path / "results"
    argv = [
        "--model",
        TUNED,
        "--base",
        f"hf:{BASE_REPO}",
        "--eval-set",
        str(ABLATION),
        "--limit",
        "2",
        "--out",
        str(out),
        "--dry-run",
    ]
    assert run_cli(argv) == 0
    stdout = capsys.readouterr().out
    best = load_models_config()["eval"]["best_strategy"]
    assert f"hf:{BASE_REPO}" in stdout and TUNED in stdout and best in stdout
    # 2 scenarios x 3 systems
    assert re.search(r"(?mi)^\s*generation calls:\s*6\s*$", stdout)
    assert re.search(r"(?mi)^\s*judge calls:\s*6\s*$", stdout)
    assert no_run_written(out)
