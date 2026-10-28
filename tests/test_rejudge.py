"""Re-judge an existing run's stored replies with another judge (D-080).

    uv run scripts/rejudge.py RUN_DIR --judge SPEC --out NEW_DIR [--dry-run] [--resume]

RUN_DIR is a finished run (``results.json`` + ``transcripts.jsonl``) from ``eval.py``,
``scripts/run_ablation.py`` or ``scripts/run_rollouts.py``. The tool reloads the run's eval set
or conversation scripts from ``pins.eval_set`` (refusing, exit 2, if its SHA-256 changed), then
re-checks and re-judges every stored reply with ``--judge``. It makes no generation calls and no
Hub calls. NEW_DIR is the new run directory itself (its name is the new ``run_id``); it receives
``results.json``, ``results.md`` and ``transcripts.jsonl`` in the source run's shape:

- every transcript row keeps the source row's model, strategy, scenario/turn ids, reply,
  generation_error, decoding and prompt hash, in the source order, and carries the new
  ``judge_model`` and ``run_id``; a row whose generation failed stays a scored failure without
  a judge call;
- ``results.json`` has the source run's top-level keys, its cells (``cells``/``rows``) sorted by
  ``(model, strategy)`` whatever the order of the source transcript rows (which is completion
  order), so the output is deterministic; ``pins.judge_model`` = the new judge,
  the new judge's decoding in ``pins.decoding``, the source eval-set and model pins, the current
  rubric hash, and ``rejudged_from`` = {``run_id``, ``judge_model``} of the source run;
- per-conversation runs keep conversation-aware aggregation (``n_conversations``,
  ``conversation_pass_rate``, ``first_failure_turns``, ``spec_adherence_by_turn``).

``--dry-run`` prints ``judge calls: N`` (the replies that need a judge call) and writes nothing,
with no key needed. The tool is resumable: ``--resume`` continues an interrupted re-judge into
the same NEW_DIR, judging only the rows that are missing, and refuses a NEW_DIR started from
another source run or judge. Without ``--resume`` a non-empty NEW_DIR is refused (exit 2).

Every test runs with sockets blocked and no keys; the new judge is a fake registered under the
``rj:`` prefix, and the source run's models are made to fail if anything asks them to generate.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from groundcite import backends, hub
from groundcite.backends import MockCompliant, MockDrift, MockNaive, MockSteady
from groundcite.judge import rubric_sha256
from groundcite.report import validate_transcript_row

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
REJUDGE = SCRIPTS / "rejudge.py"
SCENARIOS = REPO_ROOT / "data" / "scenarios"
ABLATION = SCENARIOS / "ablation.jsonl"
ROLLOUT_PILOT = SCENARIOS / "rollout_pilot.jsonl"
NEW_JUDGE = "rj:strict-2026"
CONVERSATION_KEYS = {
    "n_conversations",
    "conversation_pass_rate",
    "first_failure_turns",
    "spec_adherence_by_turn",
}
# What a re-judge must carry over from the source row unchanged.
KEPT_FIELDS = (
    "model",
    "model_revision",
    "strategy",
    "scenario_id",
    "category",
    "gold",
    "reply",
    "prompt_sha256",
    "generation_error",
    "decoding",
)


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


# ---------------------------------------------------------------------------------------------
# The new judge: a fake that fails every reply and counts its calls
# ---------------------------------------------------------------------------------------------


class StrictJudge:
    """Scores every reply (0, 2), so a re-judged run is visibly different from the mock judge's."""

    calls = 0
    lock = threading.Lock()

    def __init__(self, spec: str, transport: Any = None) -> None:
        self.name = spec

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        with StrictJudge.lock:
            StrictJudge.calls += 1
        verdict = {
            "spec_adherence": 0,
            "robustness": 2,
            "failed_checks": [],
            "reasoning": "Strict re-judge: scored as a failure.",
        }
        return json.dumps(verdict)


@pytest.fixture
def strict(monkeypatch: pytest.MonkeyPatch) -> type[StrictJudge]:
    monkeypatch.setattr(StrictJudge, "calls", 0)
    monkeypatch.setitem(backends.BACKENDS, "rj:", StrictJudge)
    return StrictJudge


def _block_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the source run exists, nothing may generate a reply or touch the Hub."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("rejudge must not generate replies or call the Hub")

    for cls in (MockCompliant, MockNaive, MockSteady, MockDrift):
        monkeypatch.setattr(cls, "generate", refuse)
    monkeypatch.setattr(hub, "fetch_adapter_config", refuse)
    monkeypatch.setattr(hub, "resolve_revision", refuse)
    monkeypatch.setitem(backends.BACKENDS, "hf:", refuse)


# ---------------------------------------------------------------------------------------------
# Helpers: the tools, loaded in-process
# ---------------------------------------------------------------------------------------------


def _load(path: Path, name: str) -> ModuleType:
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} does not exist yet"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main(path: Path, name: str, argv: list[str]) -> int:
    try:
        return int(_load(path, name).main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def rejudge(*argv: str) -> int:
    return _main(REJUDGE, "rejudge", list(argv))


def _eval_cli(argv: list[str]) -> int:
    from groundcite.cli import main

    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def _only_dir(out: Path) -> Path:
    dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"expected one run dir in {out}, found {dirs}"
    return dirs[0]


def read_results(run: Path) -> dict[str, Any]:
    return dict(json.loads((run / "results.json").read_text(encoding="utf-8")))


def read_rows(run: Path) -> list[dict[str, Any]]:
    lines = (run / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _calls(stdout: str) -> int:
    match = re.search(r"(?mi)^\s*judge calls:\s*(\d+)\s*$", stdout)
    assert match, f"no 'judge calls: N' line in:\n{stdout}"
    return int(match.group(1))


def make_ablation_run(root: Path) -> Path:
    out = root / "ablation"
    argv = ["--model", "mock:compliant", "--model", "mock:naive", "--judge", "mock"]
    argv += ["--eval-set", str(ABLATION), "--limit", "3", "--allow-dirty", "--out", str(out)]
    assert _main(SCRIPTS / "run_ablation.py", "run_ablation", argv) == 0
    return _only_dir(out)


def make_eval_run(root: Path) -> Path:
    out = root / "eval"
    argv = ["--model", "mock:compliant", "--model", "mock:naive", "--eval-set", str(ABLATION)]
    argv += ["--judge", "mock", "--no-base", "--limit", "3", "--allow-dirty", "--out", str(out)]
    assert _eval_cli(argv) == 0
    return _only_dir(out)


def make_rollout_run(root: Path) -> Path:
    out = root / "rollouts"
    argv = ["--model", "mock:drift@2", "--strategy", "structured", "--scripts", str(ROLLOUT_PILOT)]
    argv += ["--limit", "3", "--judge", "mock", "--allow-dirty", "--out", str(out)]
    assert _main(SCRIPTS / "run_rollouts.py", "run_rollouts", argv) == 0
    return _only_dir(out)


MAKERS = {"ablation": make_ablation_run, "eval": make_eval_run, "rollout": make_rollout_run}


@pytest.fixture(params=sorted(MAKERS))
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def source(
    kind: str, env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: type
) -> Path:
    run = MAKERS[kind](tmp_path / "src")
    _block_generation(monkeypatch)
    return run


def _cells(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(results.get("cells") or results.get("rows") or [])


# ---------------------------------------------------------------------------------------------
# Output: same replies, new judge, new pins, a pointer to the source
# ---------------------------------------------------------------------------------------------


def test_rejudge_writes_the_three_files_with_the_new_judge(
    source: Path, tmp_path: Path, strict: type[StrictJudge]
) -> None:
    new = tmp_path / "rejudged" / "rj-1"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    for name in ("results.json", "results.md", "transcripts.jsonl"):
        assert (new / name).is_file(), f"{name} was not written"
    rows = read_rows(new)
    assert strict.calls == len(rows) > 0, "one judge call per stored reply"
    for row in rows:
        validate_transcript_row(row)
        assert row["judge_model"] == NEW_JUDGE
        assert row["run_id"] == "rj-1"
        assert row["rubric_sha256"] == rubric_sha256()
        assert row["score"]["spec_adherence"] == 0


def test_rejudge_reuses_every_stored_reply_in_order(source: Path, tmp_path: Path) -> None:
    new = tmp_path / "rj-2"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    old_rows, new_rows = read_rows(source), read_rows(new)
    assert len(new_rows) == len(old_rows)
    for old, fresh in zip(old_rows, new_rows, strict=True):
        for field in KEPT_FIELDS:
            assert fresh.get(field) == old.get(field), f"{field} changed for {old['scenario_id']}"
        for field in ("conversation_id", "turn_index"):
            assert fresh.get(field) == old.get(field), field


def test_results_json_pins_the_new_judge_and_points_at_the_source(
    source: Path, tmp_path: Path
) -> None:
    new = tmp_path / "rj-3"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    old, fresh = read_results(source), read_results(new)
    assert set(old) - {"run_id"} <= set(fresh), "keep the source run's results shape"
    assert fresh["run_id"] == "rj-3"
    pins = fresh["pins"]
    assert pins["judge_model"] == NEW_JUDGE
    assert NEW_JUDGE in pins["decoding"], "the new judge's effective decoding is pinned"
    assert pins["rubric_sha256"] == rubric_sha256() == old["pins"]["rubric_sha256"]
    assert pins["eval_set"] == old["pins"]["eval_set"]
    assert pins["models"] == old["pins"]["models"]
    assert fresh["rejudged_from"]["run_id"] == old["run_id"]
    assert fresh["rejudged_from"]["judge_model"] == old["pins"]["judge_model"] == "mock:judge"


def test_results_md_names_the_new_judge_and_the_source(source: Path, tmp_path: Path) -> None:
    new = tmp_path / "rj-4"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    md = (new / "results.md").read_text(encoding="utf-8")
    old = read_results(source)
    assert NEW_JUDGE in md
    assert old["run_id"] in md and "mock:judge" in md, "say what was re-judged and by whom"


def _by_cell(cells: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    keyed = {(c["model"], c["strategy"]): c for c in cells}
    assert len(keyed) == len(cells), "one cell per (model, strategy)"
    return keyed


def test_same_judge_rejudge_reproduces_the_source_aggregates(
    source: Path, tmp_path: Path, kind: str
) -> None:
    """With the source's own (deterministic) judge, every cell comes back identical, including
    the per-conversation metrics of a rollout run. Cells are compared by (model, strategy): the
    source run lists them in its own order."""
    new = tmp_path / "rj-same"
    assert rejudge(str(source), "--judge", "mock", "--out", str(new)) == 0
    assert _by_cell(_cells(read_results(new))) == _by_cell(_cells(read_results(source)))
    if kind == "rollout":
        for cell in _cells(read_results(new)):
            assert CONVERSATION_KEYS <= set(cell)


def _shuffled(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The source rows as written, reversed, and in several seeded random orders."""
    orders = [list(rows), rows[::-1]]
    for seed in range(4):
        order = list(rows)
        random.Random(seed).shuffle(order)
        orders.append(order)
    return orders


def test_rejudge_output_does_not_depend_on_the_source_row_order(
    source: Path, tmp_path: Path
) -> None:
    """The source transcripts are in completion order, which varies run to run. Re-judging the
    same rows in any order gives the same cells, sorted by (model, strategy), the same deltas
    and plateau verdict, and the same results.md."""
    rows = read_rows(source)
    seen: list[tuple[Any, ...]] = []
    for i, order in enumerate(_shuffled(rows)):
        (source / "transcripts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in order), encoding="utf-8"
        )
        new = tmp_path / f"rj-order-{i}"
        assert rejudge(str(source), "--judge", "mock", "--out", str(new)) == 0
        results = read_results(new)
        cells = _cells(results)
        keys = [(c["model"], c["strategy"]) for c in cells]
        assert keys == sorted(keys), f"cells must be sorted by (model, strategy): {keys}"
        md = (new / "results.md").read_text(encoding="utf-8").replace(new.name, "<run_id>")
        seen.append((cells, results.get("deltas"), results.get("plateau"), md))
    for other in seen[1:]:
        assert other == seen[0], "the re-judged output changed with the source row order"


def test_rollout_rejudge_stays_conversation_aware(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: type
) -> None:
    run = make_rollout_run(tmp_path / "src")
    _block_generation(monkeypatch)
    new = tmp_path / "rj-conv"
    assert rejudge(str(run), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    old, fresh = read_results(run), read_results(new)
    assert fresh["conversation_ids"] == old["conversation_ids"]
    [cell] = _cells(fresh)
    [old_cell] = _cells(old)
    assert CONVERSATION_KEYS <= set(cell)
    assert cell["n_conversations"] == old_cell["n_conversations"] == 3
    # The strict judge fails every turn: no conversation passes, each fails at turn 0.
    assert cell["conversation_pass_rate"] == 0
    assert cell["first_failure_turns"] == {"0": 3}
    assert cell["n"] == old_cell["n"], "per-turn rows are not collapsed or dropped"


def test_a_generation_failure_stays_a_failure_without_a_judge_call(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: type[StrictJudge]
) -> None:
    run = make_ablation_run(tmp_path / "src")
    _block_generation(monkeypatch)
    rows = read_rows(run)
    rows[0]["reply"] = ""
    rows[0]["generation_error"] = "mock: refused"
    rows[0]["decoding"] = None
    (run / "transcripts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    new = tmp_path / "rj-generr"
    assert rejudge(str(run), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    assert strict.calls == len(rows) - 1
    first = read_rows(new)[0]
    assert first["generation_error"] == "mock: refused"
    assert first["score"]["spec_adherence"] == 0
    assert first["judge_model"] == NEW_JUDGE


# ---------------------------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------------------------


def test_dry_run_prints_the_judge_call_count_and_writes_nothing(
    source: Path, tmp_path: Path, strict: type[StrictJudge], capsys: pytest.CaptureFixture[str]
) -> None:
    new = tmp_path / "rj-dry"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new), "--dry-run") == 0
    assert _calls(capsys.readouterr().out) == len(read_rows(source))
    assert strict.calls == 0
    assert not new.exists() or not any(new.iterdir())


def test_dry_run_needs_no_judge_key(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_ablation_run(tmp_path / "src")
    _block_generation(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    new = tmp_path / "rj-dry-key"
    code = rejudge(str(run), "--judge", "anthropic:claude-opus-5-5", "--out", str(new), "--dry-run")
    assert code == 0
    assert _calls(capsys.readouterr().out) == len(read_rows(run))
    assert not new.exists()


# ---------------------------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------------------------


def _interrupt(new: Path, keep: int) -> None:
    """Leave NEW_DIR as an interrupted re-judge: the first ``keep`` rows and no results yet."""
    lines = (new / "transcripts.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    (new / "transcripts.jsonl").write_text("".join(lines[:keep]), encoding="utf-8")
    for name in ("results.json", "results.md"):
        (new / name).unlink(missing_ok=True)


def test_resume_judges_only_the_missing_rows(
    source: Path, tmp_path: Path, strict: type[StrictJudge], capsys: pytest.CaptureFixture[str]
) -> None:
    full = tmp_path / "rj-full"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(full)) == 0
    total = strict.calls
    new = tmp_path / "rj-resume"
    shutil.copytree(full, new)
    _interrupt(new, keep=2)
    capsys.readouterr()

    args = (str(source), "--judge", NEW_JUDGE, "--out", str(new), "--resume")
    assert rejudge(*args, "--dry-run") == 0
    assert _calls(capsys.readouterr().out) == total - 2

    strict.calls = 0
    assert rejudge(*args) == 0
    assert strict.calls == total - 2
    key = ("model", "strategy", "scenario_id")
    resumed = [tuple(r[k] for k in key) for r in read_rows(new)]
    assert sorted(resumed) == sorted(tuple(r[k] for k in key) for r in read_rows(full))
    assert len(resumed) == len(set(resumed)), "no row is judged twice"
    assert _cells(read_results(new)) == _cells(read_results(full))


def test_existing_out_dir_without_resume_is_refused(
    source: Path, tmp_path: Path, strict: type[StrictJudge]
) -> None:
    new = tmp_path / "rj-exists"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    before = (new / "transcripts.jsonl").read_bytes()
    strict.calls = 0
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 2
    assert strict.calls == 0
    assert (new / "transcripts.jsonl").read_bytes() == before


def test_resume_with_another_judge_is_refused(
    source: Path, tmp_path: Path, strict: type[StrictJudge], capsys: pytest.CaptureFixture[str]
) -> None:
    new = tmp_path / "rj-other-judge"
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    _interrupt(new, keep=1)
    capsys.readouterr()
    assert rejudge(str(source), "--judge", "mock", "--out", str(new), "--resume") == 2
    assert "judge" in capsys.readouterr().err.lower()


def test_resume_from_another_source_run_is_refused(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: type
) -> None:
    first = make_ablation_run(tmp_path / "a")
    second = make_eval_run(tmp_path / "b")
    _block_generation(monkeypatch)
    new = tmp_path / "rj-other-src"
    assert rejudge(str(first), "--judge", NEW_JUDGE, "--out", str(new)) == 0
    _interrupt(new, keep=1)
    assert rejudge(str(second), "--judge", NEW_JUDGE, "--out", str(new), "--resume") == 2


# ---------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------


def test_a_changed_eval_set_is_refused(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: type[StrictJudge]
) -> None:
    run = make_ablation_run(tmp_path / "src")
    _block_generation(monkeypatch)
    results = read_results(run)
    results["pins"]["eval_set"]["sha256"] = "0" * 64
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
    new = tmp_path / "rj-changed"
    assert rejudge(str(run), "--judge", NEW_JUDGE, "--out", str(new)) == 2
    assert strict.calls == 0
    assert not new.exists() or not any(new.iterdir())


def test_a_run_without_results_json_is_refused(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: type[StrictJudge]
) -> None:
    run = make_ablation_run(tmp_path / "src")
    _block_generation(monkeypatch)
    (run / "results.json").unlink()
    new = tmp_path / "rj-partial"
    assert rejudge(str(run), "--judge", NEW_JUDGE, "--out", str(new)) == 2
    assert strict.calls == 0


def test_a_missing_judge_key_fails_before_any_work(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = make_ablation_run(tmp_path / "src")
    _block_generation(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    new = tmp_path / "rj-nokey"
    assert rejudge(str(run), "--judge", "anthropic:claude-opus-5-5", "--out", str(new)) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not new.exists() or not any(new.iterdir())


def test_the_rejudged_run_passes_check_rubric(source: Path, tmp_path: Path) -> None:
    """A re-judged run sits under results/ next to its source: one rubric hash, and each run
    is scored by a single judge."""
    results = tmp_path / "results"
    shutil.copytree(source, results / "source")
    assert rejudge(str(source), "--judge", NEW_JUDGE, "--out", str(results / "rejudged")) == 0
    check = _load(SCRIPTS / "check_rubric.py", "check_rubric")
    assert check.main([str(results)]) == 0
