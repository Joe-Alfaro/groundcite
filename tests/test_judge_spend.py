"""Judge spend safety: dry runs, recorded usage, a spend cap and --judge-from resume (R10, R11,
R13, R20, D-078, D-095).

``eval.py --judge-from GENERATIONS --dry-run`` used to make real judge calls (it branched before
the dry-run check), recorded no token usage and could not resume, so a split run's spend could be
neither computed nor capped. These tests pin the guarantees for every judging entry point:
``eval.py --judge-from``, the one-command ``eval.py`` and ``scripts/rejudge.py``.

- ``--dry-run`` makes zero judge calls, needs no key and writes nothing. It prints
  ``judge calls: N`` (the judgments the run would make) and ``estimated cost: $X.XX`` from the
  config ``[prices]`` table, then exits 0.
- A judging run records each judge call's usage in its transcript row: ``row["judge"]["usage"]``
  is a list with one ``{input_tokens, output_tokens, reasoning_tokens}`` entry per judge call
  made for that row (empty for a row that needed none). The run directory gets ``spend.json``:
  ``{judge_model, calls, input_tokens, output_tokens, reasoning_tokens, cost_usd}``, with
  ``cost_usd`` the actual spend from the recorded usage and ``[prices]`` (reasoning tokens bill
  at the output price), summed over every invocation of a resumed run and written even when the
  run stops early.
- ``--max-spend USD`` stops the run cleanly before a judge call that would push the recorded
  actual spend past the limit: the finished rows and ``spend.json`` stay on disk, no
  ``results.json`` is written, and the exit code is non-zero with a message naming
  ``--max-spend``. A resumed run counts the spend already recorded in ``spend.json``.
- ``eval.py --judge-from GENERATIONS --resume RUN_DIR`` continues an interrupted split run into
  RUN_DIR: rows already in its transcripts (matched by model, strategy and scenario_id) are kept
  without a call and only the missing rows are judged. Its final results equal an uninterrupted
  run's apart from run_id. It refuses (exit 2, no calls) when the generations, eval set, judge or
  rubric differ from what RUN_DIR was started with.

``config/models.toml`` gains a ``[prices]`` entry for the gpt-5.5 judge: $5 input and $30 output
per million tokens, marked "verify on the dashboard".

Every test runs with sockets blocked and a fake key. The judge is the real OpenAI backend with a
fake transport that answers like the mock judge, reports fixed usage and counts its calls.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from config_helpers import patched_config

from groundcite import backends, config
from groundcite.backends import MockJudge

REPO_ROOT = Path(__file__).resolve().parents[1]
REJUDGE = REPO_ROOT / "scripts" / "rejudge.py"
MODELS_TOML = REPO_ROOT / "config" / "models.toml"
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
JUDGE = "openai:gpt-5.5-2026-04-23"
SUBJECTS = ["--model", "mock:compliant", "--model", "mock:naive", "--no-base"]
LIMIT = 3
JUDGMENTS = 2 * LIMIT  # two systems x three scenarios, no generation failures

# The fake judge's usage per call, as OpenAI reports it: completion_tokens include reasoning.
PROMPT_TOKENS = 1000
COMPLETION_TOKENS = 500
REASONING_TOKENS = 300
CALL_USAGE = {
    "input_tokens": PROMPT_TOKENS,
    "output_tokens": COMPLETION_TOKENS - REASONING_TOKENS,
    "reasoning_tokens": REASONING_TOKENS,
}
# $5 input and $30 output (reasoning included) per million tokens.
CALL_COST = (PROMPT_TOKENS * 5 + COMPLETION_TOKENS * 30) / 1e6
SPEND_FIELDS = {
    "judge_model",
    "calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost_usd",
}


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


# ---------------------------------------------------------------------------------------------
# The fake judge API
# ---------------------------------------------------------------------------------------------


class FakeOpenAI:
    """An OpenAI chat-completions transport: answers like the mock judge, reports fixed usage and
    counts calls. ``fail_at`` makes the n-th call (1-based) and every later one raise."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        with self.lock:
            self.calls += 1
            n = self.calls
        if self.fail_at is not None and n >= self.fail_at:
            raise RuntimeError("the fake judge API went away")
        messages = list(body["messages"])
        system = None
        if messages and messages[0]["role"] == "developer":
            system = messages.pop(0)["content"]
        content = MockJudge().generate(messages, system, {})
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                "completion_tokens_details": {"reasoning_tokens": REASONING_TOKENS},
            },
        }


def use_judge(monkeypatch: pytest.MonkeyPatch, transport: FakeOpenAI) -> FakeOpenAI:
    monkeypatch.setitem(
        backends.BACKENDS,
        "openai:",
        lambda spec, _transport=None: backends.OpenAIBackend(spec, transport),
    )
    return transport


def with_key(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=sk-test-not-a-real-key\n", encoding="utf-8")


@pytest.fixture
def judge(env: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    """The fake judge with a fake key in place."""
    with_key(env)
    return use_judge(monkeypatch, FakeOpenAI())


# ---------------------------------------------------------------------------------------------
# The three judging entry points
# ---------------------------------------------------------------------------------------------


def _main(module_main: Callable[[list[str]], int], argv: list[str]) -> int:
    try:
        return int(module_main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def eval_cli(*argv: str) -> int:
    from groundcite.cli import main

    return _main(main, list(argv))


def _load_rejudge() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rejudge_spend", REJUDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejudge_cli(*argv: str) -> int:
    return _main(_load_rejudge().main, list(argv))


def only_dir(out: Path) -> Path | None:
    dirs = [p for p in out.iterdir() if p.is_dir()] if out.exists() else []
    assert len(dirs) <= 1, f"expected at most one run dir in {out}, found {dirs}"
    return dirs[0] if dirs else None


def generate(out: Path, eval_set: Path = ABLATION) -> Path:
    """A --generate-only run of the mock models (no judge, no key): its generations.jsonl."""
    argv = [*SUBJECTS, "--eval-set", str(eval_set), "--limit", str(LIMIT), "--allow-dirty"]
    assert eval_cli(*argv, "--generate-only", "--out", str(out)) == 0
    run = only_dir(out)
    assert run is not None
    return run / "generations.jsonl"


def mock_judged_run(out: Path) -> Path:
    """A finished one-command run judged by the mock judge: a source for rejudge.py."""
    argv = [*SUBJECTS, "--eval-set", str(ABLATION), "--limit", str(LIMIT), "--allow-dirty"]
    assert eval_cli(*argv, "--judge", "mock", "--out", str(out)) == 0
    run = only_dir(out)
    assert run is not None
    return run


class Tool:
    """One judging entry point: ``invoke(*extra)`` runs it with the gpt-5.5 judge into a fresh
    output, and ``run_dir()`` is the run directory it wrote (None if there is none)."""

    def __init__(self, kind: str, root: Path) -> None:
        self.kind = kind
        self.root = root
        self.out = root / "judged"
        if kind == "judge_from":
            self.generations = generate(root / "gen")
        elif kind == "rejudge":
            self.source = mock_judged_run(root / "src")

    def invoke(self, *extra: str) -> int:
        if self.kind == "judge_from":
            argv = ["--judge-from", str(self.generations), "--judge", JUDGE, "--allow-dirty"]
            return eval_cli(*argv, "--out", str(self.out), *extra)
        if self.kind == "one_shot":
            argv = [*SUBJECTS, "--eval-set", str(ABLATION), "--limit", str(LIMIT)]
            argv += ["--judge", JUDGE, "--allow-dirty", "--out", str(self.out)]
            return eval_cli(*argv, *extra)
        return rejudge_cli(str(self.source), "--judge", JUDGE, "--out", str(self.out), *extra)

    def run_dir(self) -> Path | None:
        if self.kind == "rejudge":
            return self.out if self.out.is_dir() else None
        return only_dir(self.out)


TOOLS = ("judge_from", "one_shot", "rejudge")


@pytest.fixture(params=TOOLS)
def tool(request: pytest.FixtureRequest, env: Path, tmp_path: Path) -> Tool:
    return Tool(str(request.param), tmp_path)


# ---------------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def read_spend(run: Path) -> dict[str, Any]:
    path = run / "spend.json"
    assert path.is_file(), f"{run.name} has no spend.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def judge_calls(stdout: str) -> int:
    match = re.search(r"(?mi)^\s*judge calls:\s*(\d+)\s*$", stdout)
    assert match, f"no 'judge calls: N' line in:\n{stdout}"
    return int(match.group(1))


def estimated_cost(stdout: str) -> float:
    match = re.search(r"(?mi)^\s*estimated cost:\s*\$(\d+(?:\.\d+)?)\s*$", stdout)
    assert match, f"no 'estimated cost: $X.XX' line in:\n{stdout}"
    return float(match.group(1))


def not_an_unknown_flag(err: str) -> None:
    """argparse's own rejection also exits 2 and echoes every flag: rule it out."""
    assert "unrecognized arguments" not in err, err


def is_empty(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


# ---------------------------------------------------------------------------------------------
# (2) The gpt-5.5 judge has a price
# ---------------------------------------------------------------------------------------------


def test_the_gpt55_judge_has_a_configured_price() -> None:
    assert config.price(JUDGE) == {"input_per_mtok": 5.0, "output_per_mtok": 30.0}


def test_the_gpt55_price_is_marked_verify_on_the_dashboard() -> None:
    lines = MODELS_TOML.read_text(encoding="utf-8").splitlines()
    at = [i for i, line in enumerate(lines) if line.strip().startswith(f'"{JUDGE}"')]
    assert at, f"no [prices] line for {JUDGE} in config/models.toml"
    # The entry's own line, or the comment lines directly above it.
    i = at[-1]
    context = [lines[i]]
    j = i - 1
    while j >= 0 and lines[j].strip().startswith("#"):
        context.append(lines[j])
        j -= 1
    assert "verify on the dashboard" in "\n".join(context).lower(), context


def test_a_gpt55_call_costs_input_plus_output_and_reasoning() -> None:
    assert config.cost_usd(JUDGE, [CALL_USAGE]) == pytest.approx(CALL_COST)


# ---------------------------------------------------------------------------------------------
# (1, 5) --dry-run: zero calls, the judgment count and an estimated cost
# ---------------------------------------------------------------------------------------------


def test_dry_run_makes_no_judge_calls_and_prints_count_and_cost(
    tool: Tool, judge: FakeOpenAI, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert tool.invoke("--dry-run") == 0
    out = capsys.readouterr().out
    assert judge.calls == 0, f"{tool.kind} --dry-run made {judge.calls} judge calls"
    assert judge_calls(out) == JUDGMENTS
    assert estimated_cost(out) > 0
    assert is_empty(tool.out), "a dry run writes nothing"


def test_dry_run_needs_no_key(
    tool: Tool, env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = use_judge(monkeypatch, FakeOpenAI())
    capsys.readouterr()
    assert tool.invoke("--dry-run") == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert transport.calls == 0
    assert judge_calls(out) == JUDGMENTS
    estimated_cost(out)


def test_dry_run_estimate_follows_the_configured_price(
    tool: Tool,
    judge: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capsys.readouterr()
    assert tool.invoke("--dry-run") == 0
    base = estimated_cost(capsys.readouterr().out)

    def triple(cfg: dict[str, Any]) -> None:
        price = cfg["prices"][JUDGE]
        price["input_per_mtok"] *= 3
        price["output_per_mtok"] *= 3

    patched_config(monkeypatch, tmp_path, triple)
    assert tool.invoke("--dry-run") == 0
    tripled = estimated_cost(capsys.readouterr().out)
    assert tripled == pytest.approx(3 * base, rel=0.1, abs=0.02)
    assert judge.calls == 0


# ---------------------------------------------------------------------------------------------
# (2, 5) Recorded usage and spend.json
# ---------------------------------------------------------------------------------------------


def test_each_row_records_its_judge_calls_usage(tool: Tool, judge: FakeOpenAI) -> None:
    assert tool.invoke() == 0
    run = tool.run_dir()
    assert run is not None
    rows = read_jsonl(run / "transcripts.jsonl")
    assert len(rows) == JUDGMENTS
    assert judge.calls == JUDGMENTS
    for row in rows:
        usage = row["judge"].get("usage")
        assert usage == [CALL_USAGE], (row["model"], row["scenario_id"], usage)


def test_the_run_writes_spend_json_with_totals_and_actual_usd(
    tool: Tool, judge: FakeOpenAI
) -> None:
    assert tool.invoke() == 0
    run = tool.run_dir()
    assert run is not None
    spend = read_spend(run)
    assert SPEND_FIELDS <= set(spend)
    assert spend["judge_model"] == JUDGE
    assert spend["calls"] == JUDGMENTS
    assert spend["input_tokens"] == JUDGMENTS * CALL_USAGE["input_tokens"]
    assert spend["output_tokens"] == JUDGMENTS * CALL_USAGE["output_tokens"]
    assert spend["reasoning_tokens"] == JUDGMENTS * CALL_USAGE["reasoning_tokens"]
    assert spend["cost_usd"] == pytest.approx(JUDGMENTS * CALL_COST)


def test_spend_follows_the_configured_price(
    tool: Tool, judge: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def double(cfg: dict[str, Any]) -> None:
        cfg["prices"][JUDGE] = {"input_per_mtok": 10.0, "output_per_mtok": 60.0}

    patched_config(monkeypatch, tmp_path, double)
    assert tool.invoke() == 0
    run = tool.run_dir()
    assert run is not None
    assert read_spend(run)["cost_usd"] == pytest.approx(2 * JUDGMENTS * CALL_COST)


# ---------------------------------------------------------------------------------------------
# (3, 5) --max-spend
# ---------------------------------------------------------------------------------------------


def test_max_spend_below_one_call_stops_before_any_call(
    tool: Tool, judge: FakeOpenAI, capsys: pytest.CaptureFixture[str]
) -> None:
    code = tool.invoke("--max-spend", "0.000001")
    err = capsys.readouterr().err
    not_an_unknown_flag(err)
    assert code != 0
    assert "--max-spend" in err, err
    assert judge.calls == 0
    run = tool.run_dir()
    assert run is None or not (run / "results.json").exists()


@pytest.mark.parametrize("concurrency", ["1", "4"])
def test_max_spend_never_lets_recorded_spend_pass_the_limit(
    tool: Tool, judge: FakeOpenAI, capsys: pytest.CaptureFixture[str], concurrency: str
) -> None:
    limit = 2.5 * CALL_COST  # room for two of the six calls at most
    code = tool.invoke("--max-spend", str(limit), "--concurrency", concurrency)
    err = capsys.readouterr().err
    not_an_unknown_flag(err)
    assert code != 0, "the cap is below the run's cost, so the run can't finish"
    assert "--max-spend" in err, err
    assert judge.calls <= 2
    run = tool.run_dir()
    if run is None:
        assert judge.calls == 0, "a run that made calls keeps what it judged"
        return
    assert not (run / "results.json").exists(), "a capped run is not finished"
    spend = read_spend(run)
    assert spend["cost_usd"] <= limit + 1e-12
    assert spend["calls"] == judge.calls
    rows = read_jsonl(run / "transcripts.jsonl")
    assert len(rows) == judge.calls, "every paid judgment is kept"


def test_a_max_spend_above_the_run_cost_lets_it_finish(
    tool: Tool, judge: FakeOpenAI, capsys: pytest.CaptureFixture[str]
) -> None:
    code = tool.invoke("--max-spend", "1000")
    err = capsys.readouterr().err
    not_an_unknown_flag(err)
    assert code == 0, err
    run = tool.run_dir()
    assert run is not None
    assert (run / "results.json").is_file()
    assert read_spend(run)["cost_usd"] == pytest.approx(JUDGMENTS * CALL_COST)
    assert judge.calls == JUDGMENTS


# ---------------------------------------------------------------------------------------------
# (4) --judge-from --resume RUN_DIR
# ---------------------------------------------------------------------------------------------

DONE_BEFORE_CRASH = 2


def judge_from(generations: Path, *extra: str) -> int:
    return eval_cli("--judge-from", str(generations), "--allow-dirty", *extra)


@pytest.fixture
def split(env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A --generate-only run's generations.jsonl; the cwd is a scratch dir so the default
    ``--out results`` can't touch the repo."""
    monkeypatch.chdir(tmp_path)
    with_key(env)
    return generate(tmp_path / "gen")


def interrupted_run(monkeypatch: pytest.MonkeyPatch, generations: Path, out: Path) -> Path:
    """A --judge-from run whose judge API died after DONE_BEFORE_CRASH calls."""
    broken = use_judge(monkeypatch, FakeOpenAI(fail_at=DONE_BEFORE_CRASH + 1))
    code = judge_from(generations, "--judge", JUDGE, "--concurrency", "1", "--out", str(out))
    assert code not in (0, 2), "an interrupted run is neither a success nor a usage error"
    assert broken.calls == DONE_BEFORE_CRASH + 1
    run = only_dir(out)
    assert run is not None
    assert len(read_jsonl(run / "transcripts.jsonl")) == DONE_BEFORE_CRASH
    assert not (run / "results.json").exists()
    return run


def normalised(run: Path, name: str) -> str:
    run_id = json.loads((run / "results.json").read_text(encoding="utf-8"))["run_id"]
    return (run / name).read_text(encoding="utf-8").replace(run_id, "<run_id>")


def sorted_rows(run: Path) -> list[dict[str, Any]]:
    rows = [
        {k: v for k, v in r.items() if k != "run_id"} for r in read_jsonl(run / "transcripts.jsonl")
    ]
    return sorted(rows, key=lambda r: (r["model"], r["strategy"], r["scenario_id"]))


def test_resume_judges_only_the_missing_rows(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    healthy = use_judge(monkeypatch, FakeOpenAI())
    capsys.readouterr()
    code = judge_from(split, "--judge", JUDGE, "--resume", str(run))
    err = capsys.readouterr().err
    not_an_unknown_flag(err)
    assert code == 0, err
    assert healthy.calls == JUDGMENTS - DONE_BEFORE_CRASH, "done rows are reused without calls"
    assert len(read_jsonl(run / "transcripts.jsonl")) == JUDGMENTS
    assert (run / "results.json").is_file(), "the resumed run finishes in RUN_DIR"
    assert not (tmp_path / "results").exists(), "--resume writes into RUN_DIR, not --out"


def test_resumed_results_equal_an_uninterrupted_run(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    use_judge(monkeypatch, FakeOpenAI())
    assert judge_from(split, "--judge", JUDGE, "--resume", str(run)) == 0

    use_judge(monkeypatch, FakeOpenAI())
    assert judge_from(split, "--judge", JUDGE, "--out", str(tmp_path / "whole")) == 0
    whole = only_dir(tmp_path / "whole")
    assert whole is not None

    assert normalised(run, "results.json") == normalised(whole, "results.json")
    assert normalised(run, "results.md") == normalised(whole, "results.md")
    assert sorted_rows(run) == sorted_rows(whole)


def test_resumed_spend_covers_both_invocations(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    assert read_spend(run)["calls"] == DONE_BEFORE_CRASH, "an interrupted run keeps its spend"
    use_judge(monkeypatch, FakeOpenAI())
    assert judge_from(split, "--judge", JUDGE, "--resume", str(run)) == 0
    spend = read_spend(run)
    assert spend["calls"] == JUDGMENTS
    assert spend["cost_usd"] == pytest.approx(JUDGMENTS * CALL_COST)


def test_resume_max_spend_counts_the_spend_already_recorded(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    before = read_jsonl(run / "transcripts.jsonl")
    healthy = use_judge(monkeypatch, FakeOpenAI())
    capsys.readouterr()
    # Two calls are already paid for: 0.04 USD recorded, so a 0.03 cap allows nothing more.
    limit = 1.5 * CALL_COST
    code = judge_from(split, "--judge", JUDGE, "--resume", str(run), "--max-spend", str(limit))
    err = capsys.readouterr().err
    not_an_unknown_flag(err)
    assert code != 0
    assert "--max-spend" in err, err
    assert healthy.calls == 0
    assert read_jsonl(run / "transcripts.jsonl") == before


def _refused(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    run: Path,
    generations: Path,
    *extra: str,
) -> str:
    """Resume RUN_DIR with ``generations`` and ``extra``; assert a clean refusal, return stderr."""
    before = (run / "transcripts.jsonl").read_text(encoding="utf-8")
    transport = use_judge(monkeypatch, FakeOpenAI())
    capsys.readouterr()
    code = judge_from(generations, "--resume", str(run), *extra)
    err = capsys.readouterr().err
    not_an_unknown_flag(err)
    assert code == 2, err
    assert transport.calls == 0
    assert (run / "transcripts.jsonl").read_text(encoding="utf-8") == before
    assert not (run / "results.json").exists()
    return err.lower()


def test_resume_refuses_different_generations(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    other = tmp_path / "other-gen"
    shutil.copytree(split.parent, other)
    edited = other / split.name
    rows = read_jsonl(edited)
    rows[0]["reply"] = rows[0]["reply"] + " (edited)"
    edited.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    err = _refused(capsys, monkeypatch, run, edited, "--judge", JUDGE)
    assert "generations" in err, err


def test_resume_refuses_a_different_eval_set(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    # Same scenarios, other bytes: the replies are identical, only the eval set's pin differs.
    eval_set = tmp_path / "ablation-copy.jsonl"
    eval_set.write_text(ABLATION.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    other = generate(tmp_path / "gen-copy", eval_set)
    assert other.read_bytes() == split.read_bytes(), "the replies themselves are the same"
    err = _refused(capsys, monkeypatch, run, other, "--judge", JUDGE)
    assert "eval set" in err, err


def test_resume_refuses_a_different_judge(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    err = _refused(capsys, monkeypatch, run, split, "--judge", "mock")
    assert "judge" in err, err


def test_resume_refuses_a_different_rubric(
    split: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = interrupted_run(monkeypatch, split, tmp_path / "out")
    # Make RUN_DIR look started (and its rows scored) under another rubric.
    other_rubric = "0" * 64
    manifest = run / "run.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["rubric_sha256"] = other_rubric
        manifest.write_text(json.dumps(data), encoding="utf-8")
    rows = read_jsonl(run / "transcripts.jsonl")
    for row in rows:
        row["rubric_sha256"] = other_rubric
    (run / "transcripts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    err = _refused(capsys, monkeypatch, run, split, "--judge", JUDGE)
    assert "rubric" in err, err
