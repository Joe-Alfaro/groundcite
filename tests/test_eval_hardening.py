"""Eval hardening before the real ablation run (D-043; R3, R10, R11, R13, R27, R28, R30).

1. A subject 401, 403 or other non-retryable 4xx stops the run with exit 2 and a redacted
   message; no failure rows are written. Refusals and empty replies are still scored failures.
2. Effective decoding params are persisted with the run and reported correctly after a resume
   that makes no calls.
3. Resume refuses a different eval-set sha256, scenario-ID list or per-strategy prompt hash. The
   per-strategy prompt hash covers the rendered system prompt, and every transcript row records
   it as ``prompt_sha256``; a cached row is reused only when its hash matches.
5. Error rows are left out of the abstain precision/recall predictions but count as Spec
   failures.
6. Every script that reports scores calls ``aggregate_scores``; no committed transcript under
   ``results/`` contains ``<think>``.
7. The README opens with the key env vars and the hardware, and states the rerun tolerance.

Sockets are blocked and keys are fake. API subjects use the real backend classes with an
injected fake transport.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from groundcite import backends
from groundcite.backends import AnthropicBackend, MockCompliant

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
EVAL_OWN = REPO_ROOT / "data" / "scenarios" / "eval_own.jsonl"
SCRIPT = REPO_ROOT / "scripts" / "run_ablation.py"
SUBJECT = "anthropic:claude-opus-5-5"
FAKE_KEY = "fake-anthropic-key-hardening-123"


@pytest.fixture
def env(isolated_env: Path, no_network: None, monkeypatch: pytest.MonkeyPatch) -> Path:
    isolated_env.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    # A fresh process: no parameter has been rejected yet.
    monkeypatch.setattr(backends, "_REJECTED", {})
    return isolated_env


def _load_ablation() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ablation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ablation(argv: list[str]) -> int:
    try:
        return int(_load_ablation().main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def run_eval(argv: list[str]) -> int:
    from groundcite.cli import main

    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def all_rows(out: Path) -> list[dict[str, Any]]:
    return [row for p in out.rglob("transcripts.jsonl") for row in read_rows(p)]


# ---------------------------------------------------------------------------------------------
# A fake Anthropic transport
# ---------------------------------------------------------------------------------------------


class FakeHTTPError(Exception):
    """An SDK-style HTTP error: ``status_code`` and, for 400s, the offending ``param``."""

    def __init__(self, status: int, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status
        self.param = param


def _anthropic_text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


class FakeAnthropic:
    """Answers like mock:compliant; ``fail`` maps a call to an exception or a raw response."""

    def __init__(
        self,
        *,
        status: int | None = None,
        reject_temperature: bool = False,
        responses: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        self.status = status
        self.reject_temperature = reject_temperature
        self.responses = dict(responses or {})
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append(dict(body))
        if self.status is not None:
            # Some HTTP errors echo the request headers, key included.
            raise FakeHTTPError(
                self.status,
                f"Error code: {self.status} - request rejected (x-api-key: {headers['x-api-key']})",
            )
        if self.reject_temperature and "temperature" in body:
            raise FakeHTTPError(400, "temperature is not supported for this model", "temperature")
        final = body["messages"][-1]["content"]
        for marker, response in self.responses.items():
            if marker in final:
                return response
        return _anthropic_text(MockCompliant().generate(body["messages"], None, {}))


def use_transport(monkeypatch: pytest.MonkeyPatch, transport: FakeAnthropic) -> None:
    monkeypatch.setitem(
        backends.BACKENDS,
        "anthropic:",
        lambda spec, _transport=None: AnthropicBackend(spec, transport),
    )


def _eval_argv(out: Path, *extra: str, limit: int = 3) -> list[str]:
    return [
        "--model",
        SUBJECT,
        "--eval-set",
        str(ABLATION),
        "--judge",
        "mock",
        "--no-base",
        "--limit",
        str(limit),
        "--concurrency",
        "1",
        "--allow-dirty",
        "--out",
        str(out),
        *extra,
    ]


def _ablation_argv(out: Path, *extra: str, limit: int = 3, run_id: str = "r1") -> list[str]:
    return [
        "--model",
        SUBJECT,
        "--strategy",
        "zero_shot",
        "--judge",
        "mock",
        "--limit",
        str(limit),
        "--concurrency",
        "1",
        "--allow-dirty",
        "--out",
        str(out),
        "--run-id",
        run_id,
        *extra,
    ]


# ---------------------------------------------------------------------------------------------
# 1. Auth and other non-retryable 4xx errors stop the run (D-043 item 1)
# ---------------------------------------------------------------------------------------------

STOP_STATUSES = [401, 403, 400]


def _assert_stopped(
    code: int, out: Path, transport: FakeAnthropic, stderr: str, status: int
) -> None:
    assert code == 2, f"a subject {status} must stop the run with exit 2, got {code}"
    assert len(transport.calls) == 1, "no further requests after a non-retryable 4xx"
    assert all_rows(out) == [], "no rows may be scored as failures for an auth/request error"
    assert str(status) in stderr
    assert FAKE_KEY not in stderr
    for path in out.rglob("*") if out.exists() else []:
        if path.is_file():
            assert FAKE_KEY not in path.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("status", STOP_STATUSES)
def test_eval_stops_on_a_subject_4xx(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    transport = FakeAnthropic(status=status)
    use_transport(monkeypatch, transport)
    out = tmp_path / "results"
    code = run_eval(_eval_argv(out))
    captured = capsys.readouterr()
    _assert_stopped(code, out, transport, captured.err + captured.out, status)


@pytest.mark.parametrize("status", STOP_STATUSES)
def test_ablation_stops_on_a_subject_4xx(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    transport = FakeAnthropic(status=status)
    use_transport(monkeypatch, transport)
    out = tmp_path / "ablation"
    code = run_ablation(_ablation_argv(out))
    captured = capsys.readouterr()
    _assert_stopped(code, out, transport, captured.err + captured.out, status)


def test_refusals_and_empty_replies_are_still_scored_as_failures(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from groundcite.evalset import load_eval_set

    scenarios = load_eval_set(ABLATION)[:3]
    refused, empty = (s.messages[-1]["content"] for s in scenarios[:2])
    transport = FakeAnthropic(
        responses={
            refused: {"content": [], "stop_reason": "refusal"},
            empty: {"content": [], "stop_reason": "end_turn"},
        }
    )
    use_transport(monkeypatch, transport)
    out = tmp_path / "results"
    assert run_eval(_eval_argv(out)) == 0
    rows = {r["scenario_id"]: r for r in all_rows(out)}
    assert len(rows) == 3
    for s in scenarios[:2]:
        assert rows[s.id]["generation_error"]
        assert rows[s.id]["score"] == {"spec_adherence": 0, "robustness": 1}


# ---------------------------------------------------------------------------------------------
# 2. Effective decoding params survive a resume (D-043 item 2, R13, R28)
# ---------------------------------------------------------------------------------------------


def test_effective_params_are_reported_after_a_zero_call_resume(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeAnthropic(reject_temperature=True)
    use_transport(monkeypatch, transport)
    out = tmp_path / "ablation"
    assert run_ablation(_ablation_argv(out, limit=2)) == 0
    run = out / "r1"
    first = json.loads((run / "results.json").read_text(encoding="utf-8"))["pins"]["decoding"]
    assert "temperature" not in first[SUBJECT], "the first run must pin what actually ran"
    assert first[SUBJECT], "the effective params are not empty"

    # A new process: nothing rejected in memory, every row already on disk.
    monkeypatch.setattr(backends, "_REJECTED", {})
    (run / "results.json").unlink()
    (run / "results.md").unlink()
    transport.calls.clear()
    assert run_ablation(_ablation_argv(out, "--resume", limit=2)) == 0
    assert transport.calls == [], "nothing was left to run"

    resumed = json.loads((run / "results.json").read_text(encoding="utf-8"))["pins"]["decoding"]
    assert resumed[SUBJECT] == first[SUBJECT]
    md = (run / "results.md").read_text(encoding="utf-8")
    line = next(ln for ln in md.splitlines() if "Decoding" in ln and SUBJECT in ln)
    assert "temperature" not in line


# ---------------------------------------------------------------------------------------------
# 3. Resume continues the same run or refuses (D-043 item 3, R11, R14)
# ---------------------------------------------------------------------------------------------


class Counting:
    name = "mock:counting"
    calls = 0

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        type(self).calls += 1
        return MockCompliant().generate(messages, system, params)


def _mock_argv(out: Path, eval_set: Path, *extra: str, limit: int = 4) -> list[str]:
    return [
        "--model",
        "mock:counting",
        "--strategy",
        "zero_shot",
        "--strategy",
        "structured",
        "--eval-set",
        str(eval_set),
        "--judge",
        "mock",
        "--limit",
        str(limit),
        "--allow-dirty",
        "--out",
        str(out),
        "--run-id",
        "r1",
        *extra,
    ]


@pytest.fixture
def first_run(env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setitem(backends.MOCKS, "counting", Counting)
    Counting.calls = 0
    eval_set = tmp_path / "set.jsonl"
    shutil.copyfile(ABLATION, eval_set)
    out = tmp_path / "ablation"
    assert run_ablation(_mock_argv(out, eval_set)) == 0
    Counting.calls = 0
    return {"out": out, "eval_set": eval_set, "transcripts": out / "r1" / "transcripts.jsonl"}


def _refused(first_run: dict[str, Path], argv: list[str]) -> None:
    before = first_run["transcripts"].read_text(encoding="utf-8")
    assert run_ablation(argv) == 2
    assert Counting.calls == 0
    assert first_run["transcripts"].read_text(encoding="utf-8") == before


def test_resume_refuses_a_different_eval_set(first_run: dict[str, Path]) -> None:
    _refused(first_run, _mock_argv(first_run["out"], EVAL_OWN, "--resume"))


def test_resume_refuses_a_different_scenario_list(first_run: dict[str, Path]) -> None:
    _refused(first_run, _mock_argv(first_run["out"], first_run["eval_set"], "--resume", limit=5))


def test_resume_refuses_a_changed_eval_set_file_with_the_same_ids(
    first_run: dict[str, Path],
) -> None:
    lines = first_run["eval_set"].read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])  # far outside --limit 4: only the file sha changes
    row["messages"][-1]["content"] += " It started last week."
    lines[-1] = json.dumps(row, ensure_ascii=False)
    first_run["eval_set"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    _refused(first_run, _mock_argv(first_run["out"], first_run["eval_set"], "--resume"))


def test_resume_refuses_a_changed_prompt(
    first_run: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from groundcite import prompts

    monkeypatch.setattr(prompts, "_ROLE", prompts._ROLE + " Be brief.")
    _refused(first_run, _mock_argv(first_run["out"], first_run["eval_set"], "--resume"))


def test_resume_with_the_same_inputs_still_works(first_run: dict[str, Path]) -> None:
    argv = _mock_argv(first_run["out"], first_run["eval_set"], "--resume")
    assert run_ablation(argv) == 0
    assert Counting.calls == 0


def test_rows_record_a_per_strategy_prompt_hash(first_run: dict[str, Path]) -> None:
    rows = read_rows(first_run["transcripts"])
    by_strategy: dict[str, set[str]] = {}
    for row in rows:
        value = row.get("prompt_sha256")
        assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), row.keys()
        by_strategy.setdefault(row["strategy"], set()).add(value)
    assert all(len(v) == 1 for v in by_strategy.values()), "one hash per strategy"
    assert by_strategy["zero_shot"] != by_strategy["structured"]


def test_a_cached_row_with_another_prompt_hash_is_never_reused(
    first_run: dict[str, Path],
) -> None:
    path = first_run["transcripts"]
    rows = read_rows(path)
    current = rows[0]["prompt_sha256"]
    rows[0]["prompt_sha256"] = "0" * 64
    rows[0]["reply"] = "STALE REPLY FROM AN OLD PROMPT"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    code = run_ablation(_mock_argv(first_run["out"], first_run["eval_set"], "--resume"))
    if code == 2:  # refusing the mixed run is acceptable
        return
    assert code == 0
    assert Counting.calls == 1, "only the row with the stale prompt hash is re-run"
    results = json.loads((path.parent / "results.json").read_text(encoding="utf-8"))
    assert [c["n"] for c in results["cells"]] == [4, 4]
    latest = {}
    for row in read_rows(path):
        latest[(row["model"], row["strategy"], row["scenario_id"], row["prompt_sha256"])] = row
    key = (rows[0]["model"], rows[0]["strategy"], rows[0]["scenario_id"], current)
    assert key in latest and latest[key]["reply"] != "STALE REPLY FROM AN OLD PROMPT"


# ---------------------------------------------------------------------------------------------
# 5. Error rows in the abstain metrics (D-043 item 5, R30)
# ---------------------------------------------------------------------------------------------

ABSTAIN = (
    "None of the provided references cover this complaint, so I can't point to a documented "
    "step yet. Which trouble code is stored?"
)


def _row(mode: str, reply: str, spec: int, error: str | None = None) -> dict[str, Any]:
    return {
        "model": "m",
        "strategy": "zero_shot",
        "category": "no_snippet" if mode == "abstain" else "clean",
        "gold": {"mode": mode},
        "reply": reply,
        "checks": {},
        "score": {"spec_adherence": spec, "robustness": 5 if spec else 1},
        "generation_error": error,
    }


def test_error_rows_are_excluded_from_abstain_predictions_but_fail_spec() -> None:
    from groundcite.report import aggregate_scores

    rows = [
        _row("abstain", ABSTAIN, 1),
        _row("cite", "Inspect the hose at the elbow [A-1].", 1),
        _row("cite", "", 0, error="anthropic: refused the request"),
        _row("cite", "", 0, error="anthropic: returned no text content"),
    ]
    [cell] = aggregate_scores(rows, seed=0)
    assert cell["n"] == 4
    assert cell["spec_adherence"]["mean"] == pytest.approx(0.5)
    assert cell["abstain_precision"] == pytest.approx(1.0)
    assert cell["abstain_recall"] == pytest.approx(1.0)


def test_an_all_error_cell_has_no_abstain_metrics() -> None:
    from groundcite.report import aggregate_scores

    rows = [
        _row("abstain", "", 0, error="refused"),
        _row("cite", "", 0, error="refused"),
        _row("cite", "", 0, error="refused"),
    ]
    [cell] = aggregate_scores(rows, seed=0)
    assert cell["spec_adherence"]["mean"] == 0
    assert cell["abstain_precision"] is None
    assert cell["abstain_recall"] is None


# ---------------------------------------------------------------------------------------------
# 6. One aggregation path; no thinking text in committed transcripts (D-043 item 6, R27)
# ---------------------------------------------------------------------------------------------

SCORE_MARKERS = ("results.json", "spec_adherence", "Spec-adherence")
# Scripts that read per-row Spec-adherence verdicts without reporting aggregate scores.
# judge_agreement.py compares one judge's verdicts with another's (agreement, kappa); it never
# aggregates subject-model scores or writes results.json (D-096).
NOT_SCORE_REPORTS = {"judge_agreement.py"}


def _calls(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == name) or (
                isinstance(func, ast.Attribute) and func.attr == name
            ):
                return True
    return False


def _delegate(tree: ast.Module) -> Path | None:
    """A thin entry script's implementation module (``from groundcite.x import main``)."""
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("groundcite")
        ):
            if any(alias.name == "main" for alias in node.names):
                parts = node.module.split(".")
                return REPO_ROOT / "src" / Path(*parts).with_suffix(".py")
    return None


def _entry_scripts() -> list[Path]:
    return sorted([*REPO_ROOT.glob("*.py"), *(REPO_ROOT / "scripts").glob("*.py")])


def test_every_script_that_reports_scores_calls_aggregate_scores() -> None:
    reporting = []
    for script in _entry_scripts():
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        target, target_source = script, source
        delegate = _delegate(tree)
        if delegate is not None:
            target, target_source = delegate, delegate.read_text(encoding="utf-8")
            tree = ast.parse(target_source)
        if script.name in NOT_SCORE_REPORTS:
            assert "results.json" not in target_source, f"{script.name} writes results.json"
            continue
        if any(marker in target_source for marker in SCORE_MARKERS):
            reporting.append(script.name)
            assert _calls(tree, "aggregate_scores"), (
                f"{script.name} ({target.name}) reports scores without calling aggregate_scores()"
            )
    assert {"eval.py", "run_ablation.py"} <= set(reporting)


@pytest.mark.parametrize("script", ["eval", "ablation"])
def test_each_script_aggregates_through_aggregate_scores_at_run_time(
    script: str,
    isolated_env: Path,
    no_network: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groundcite import cli, report

    seen: list[int] = []
    real = report.aggregate_scores

    def spy(rows: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = list(rows)
        seen.append(len(rows))
        return real(rows, **kwargs)

    monkeypatch.setattr(report, "aggregate_scores", spy)
    monkeypatch.setattr(cli, "aggregate_scores", spy, raising=False)
    out = tmp_path / "out"
    common = ["--model", "mock:compliant", "--judge", "mock", "--limit", "2", "--allow-dirty"]
    if script == "eval":
        argv = [*common, "--eval-set", str(ABLATION), "--no-base", "--out", str(out)]
        assert run_eval(argv) == 0
    else:
        argv = [*common, "--strategy", "zero_shot", "--out", str(out)]
        assert run_ablation(argv) == 0
    assert seen == [2]


def _committed_transcripts() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "results"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    names = [n for n in result.stdout.split("\0") if n]
    return [REPO_ROOT / n for n in names if n.endswith((".jsonl", ".json"))]


def test_no_committed_transcript_contains_thinking_text() -> None:
    offenders = []
    for path in _committed_transcripts():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "<think>" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders, f"thinking text in committed transcripts: {offenders}"


# ---------------------------------------------------------------------------------------------
# 7. The README opens with what a rerun needs (D-043 item 7, R10, R28)
# ---------------------------------------------------------------------------------------------

README_OPENING_LINES = 30


def _readme_opening() -> str:
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[:README_OPENING_LINES])


@pytest.mark.parametrize("var", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"])
def test_readme_opening_names_the_env_vars(var: str) -> None:
    assert var in _readme_opening()


def test_readme_opening_names_the_hardware() -> None:
    assert re.search(r"(?i)\b(hardware|gpu)\b", _readme_opening())


def test_readme_opening_states_the_rerun_tolerance() -> None:
    assert re.search(r"(±|\+/-|\+-)\s?0\.03\b", _readme_opening())
