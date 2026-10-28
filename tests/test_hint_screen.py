"""Tests for the semantic hint-leak screen in the R4 filter (R4, R13).

Training targets must never show that the teacher saw privileged inputs (the gold mode, the
supporting IDs or the hint note). The ``_HINT_WORDS`` regex catches the obvious wording, but
paraphrases such as "The intended reference here is ..." get through it. So every candidate that
survives the deterministic stages also goes through a short, separate classification prompt sent
to the judge model, ``groundcite.hint_screen``:

- its own versioned prompt file (``rubrics/hint_screen_<version>.md``), its own hash, and the
  output ``{"meta_leak": bool, "reason": str}``;
- the eval rubric is untouched, so ``rubric_sha256`` stays the one hash for eval, ablation and the
  dataset judge (R13);
- it runs per candidate just before the rubric judge (a flagged candidate is never paid a full
  judge call), and a flagged candidate is rejected as ``hint_leak_semantic``;
- the manifest and filter report count ``hint_leak_semantic`` apart from ``hint_leak``.

The 34 leaks are the round-4 validator's paraphrases, written as otherwise on-spec replies so
they reach the semantic stage; 11 of them hit the regex, 23 don't. Fakes only: sockets blocked,
keys unset.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from distill_data import (
    ScriptedJudge,
    candidate_row,
    read_jsonl,
    run_script,
    scenario_row,
    write_jsonl,
)

from groundcite import backends
from groundcite.evalset import parse_row

# The eval rubric hash before the screen existed (rubric v1 rendered with the current SPEC.md).
# Changing it splits eval and ablation scoring across two rubrics (R13); a deliberate rubric change
# must update this pin and rerun every scored cell.
EVAL_RUBRIC_SHA256 = "636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf"

_STEP = (
    "with the engine idling, inspect the hose at the intake elbow for cracks or a vacuum hiss "
    "[{id}]."
)
_ASK = "none of the references you provided cover this complaint. Which trouble code is stored?"

# The validator's 34 paraphrased leaks, as on-spec replies ({id} is the gold ID).
CITE_LEAKS = (
    "The note says this is the one to use, so " + _STEP,
    "Per the answer key, " + _STEP,
    "Since the gold reference is the bulletin, " + _STEP,
    "Going by the reference I was given, " + _STEP,
    "I've been told the bulletin applies, so " + _STEP,
    "As instructed, " + _STEP,
    "Going by the flagged snippet, " + _STEP,
    "The other bulletin is a decoy, so " + _STEP,
    "The correct citation here is the bulletin, so " + _STEP,
    "Based on the privileged information, " + _STEP,
    "My notes point to the bulletin, so " + _STEP,
    "I'm told the bulletin is the one that applies, so " + _STEP,
    "Per my briefing, the right reference is the bulletin, so " + _STEP,
    "Going by the reference marked for this case, " + _STEP,
    "According to the guidance provided to me, " + _STEP,
    "The target reference is the bulletin, so " + _STEP,
    "Ground truth points to the bulletin, so " + _STEP,
    "The intended reference here is the bulletin, so " + _STEP,
    "I was instructed to cite the bulletin, so " + _STEP,
    "The expected reference is the bulletin, so " + _STEP,
    "The gold one is the bulletin, so " + _STEP,
    "Following the instructions I received, " + _STEP,
    "The preselected snippet covers this, so " + _STEP,
    "The other snippets are distractors, so " + _STEP,
    "The others are traps, so " + _STEP,
    "Since the others don't apply, as the setup says, " + _STEP,
    "The answer is the bulletin, so " + _STEP,
    "Since cite mode is right here, " + _STEP,
    "The key points to the bulletin, so " + _STEP,
    "Per the rubric, " + _STEP,
)
ABSTAIN_LEAKS = (
    "As noted in my brief, " + _ASK,
    "The label for this case says abstain, and " + _ASK,
    "The right mode here is abstain: " + _ASK,
    "The provided answer is abstain, since " + _ASK,
)
INTENDED_REFERENCE = "The intended reference here is the bulletin"

# On-spec replies with the trap words the screen must not flag.
NORMAL = (
    "The bulletin covers your 2.5L, so " + _STEP,
    "Per the service manual section on vacuum leaks, " + _STEP,
    "Since you noticed a hint of coolant smell, " + _STEP,
    "Before anything else, check the gold-colored connector at the sensor for corrosion and a "
    "loose lock tab [{id}].",
    "Given the 2016 model year, the smoke test applies, so " + _STEP,
    "Check the labeled vacuum port on the intake for a leak with the engine idling [{id}].",
    "Follow the service manual instructions for the pressure test and compare the reading with "
    "the spec [{id}].",
    "The note in the bulletin says to flex the hose while idling, so " + _STEP,
    "Based on what you've told me, " + _STEP,
    "I understand you want a quick answer, but " + _STEP,
)


def _render(template: str, row: dict[str, Any]) -> str:
    ids = row["gold"]["supporting_ids"]
    text = template.format(id=ids[0] if ids else "")
    return text[0].upper() + text[1:]


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


class _Recorder(ScriptedJudge):
    """ScriptedJudge that records every request as (kind, system, user text)."""

    requests: list[tuple[str, str | None, str]] = []

    def generate(self, messages: list[dict[str, str]], system: str | None, params: Any) -> str:
        kind = "screen" if system is not None and '"meta_leak"' in system else "rubric"
        text = "\n".join(m["content"] for m in messages)
        type(self).requests.append((kind, system, text))
        return super().generate(messages, system, params)


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    for name, value in (
        ("verdicts", {}),
        ("calls", 0),
        ("leaks", ()),
        ("leak_calls", 0),
        ("requests", []),
    ):
        monkeypatch.setattr(_Recorder, name, value, raising=False)
    monkeypatch.setitem(backends.MOCKS, "judge_probe", _Recorder)
    return _Recorder


def _build(cands: Path, out: Path, judge_spec: str = "mock:judge_probe") -> int:
    return run_script(
        "build_dataset", ["--candidates", str(cands), "--out", str(out), "--judge", judge_spec]
    )


@pytest.fixture
def built(
    env: Path, judge: type[_Recorder], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """One candidate per scenario: the 10 normal replies, 30 cite leaks and 4 abstain leaks.

    The scripted screen flags exactly the 34 leak replies.
    """
    leaks: list[tuple[dict[str, Any], str]] = []
    for i, t in enumerate(CITE_LEAKS):
        row = scenario_row(100 + i)
        leaks.append((row, _render(t, row)))
    for i, t in enumerate(ABSTAIN_LEAKS):
        row = scenario_row(140 + i, category="no_snippet", mode="abstain")
        leaks.append((row, _render(t, row)))
    normals: list[tuple[dict[str, Any], str]] = []
    for i, t in enumerate(NORMAL):
        row = scenario_row(i)  # indices 0-9: distinct final messages, so no near-duplicates
        normals.append((row, _render(t, row)))
    monkeypatch.setattr(judge, "leaks", tuple(reply for _, reply in leaks))
    # Normals first: they claim their final messages before any leak could, so a leak the filter
    # wrongly keeps never pushes a normal reply out as a near-duplicate.
    rows = [candidate_row(row, 0, reply) for row, reply in normals + leaks]
    cands = write_jsonl(tmp_path / "candidates.jsonl", rows)
    out = tmp_path / "datasets" / "v1"
    assert _build(cands, out) == 0
    return {
        "leaks": leaks,
        "normals": normals,
        "out": out,
        "train": read_jsonl(out / "train.jsonl"),
        "rejections": read_jsonl(out / "rejections.jsonl"),
        "manifest": json.loads((out / "manifest.json").read_text(encoding="utf-8")),
        "report": (out / "filter_report.md").read_text(encoding="utf-8"),
    }


def _regex_caught(reply: str) -> bool:
    from groundcite.filter import _HINT_WORDS

    return bool(_HINT_WORDS.search(reply))


def _block(reply: str) -> str:
    return f"<<<\n{reply}\n>>>"


# ---------------------------------------------------------------------------------------------
# The screen prompt: separate file, separate hash, the right questions
# ---------------------------------------------------------------------------------------------


def test_screen_prompt_is_its_own_versioned_file() -> None:
    from groundcite import hint_screen as hs
    from groundcite.judge import RUBRIC_PATH

    assert isinstance(hs.HINT_SCREEN_VERSION, str) and hs.HINT_SCREEN_VERSION
    path = Path(hs.HINT_SCREEN_PATH)
    assert path.is_file()
    assert hs.HINT_SCREEN_VERSION in path.name
    assert path.parent == Path(RUBRIC_PATH).parent
    assert path.resolve() != Path(RUBRIC_PATH).resolve()


def test_screen_prompt_asks_about_every_kind_of_privileged_reference() -> None:
    from groundcite import hint_screen as hs

    text = hs.render_hint_screen().lower()
    for pattern in (
        r"hidden instructions?",
        r"answer key",
        r"intended",
        r"designated",
        r"(?:did ?n[o']t|does ?n[o']t|never) (?:provide|give|supply)",
    ):
        assert re.search(pattern, text), f"screen prompt never asks about {pattern!r}"


def test_screen_prompt_names_its_output_keys_and_the_eval_rubric_does_not() -> None:
    from groundcite import hint_screen as hs
    from groundcite.judge import render_rubric

    screen = hs.render_hint_screen()
    assert '"meta_leak"' in screen and '"reason"' in screen
    assert "meta_leak" not in render_rubric()


def test_screen_prompt_hash_is_its_own_and_hashes_the_rendered_text() -> None:
    from groundcite import hint_screen as hs
    from groundcite.judge import rubric_sha256

    expected = hashlib.sha256(hs.render_hint_screen().encode("utf-8")).hexdigest()
    assert hs.hint_screen_sha256() == expected
    assert hs.hint_screen_sha256() != rubric_sha256()


def test_eval_rubric_hash_is_unchanged_by_the_screen() -> None:
    from groundcite.judge import RUBRIC_VERSION, rubric_sha256

    assert RUBRIC_VERSION == "v1"
    assert rubric_sha256() == EVAL_RUBRIC_SHA256


def test_eval_judging_sends_only_the_eval_rubric() -> None:
    """judge_reply (eval, ablation and the dataset judge) makes one rubric call, no screen."""
    from groundcite.checks import run_checks
    from groundcite.judge import judge_reply, render_rubric

    row = scenario_row(1)
    scenario = parse_row(row)
    reply = _render(NORMAL[0], row)
    systems: list[str | None] = []

    class Backend:
        name = "mock:recording"

        def generate(self, messages: Any, system: str | None, params: Any) -> str:
            systems.append(system)
            return json.dumps(
                {"spec_adherence": 1, "robustness": 5, "failed_checks": [], "reasoning": "ok"}
            )

    result = judge_reply(scenario, reply, run_checks(reply, scenario), Backend(), params={})
    assert systems == [render_rubric()]
    assert result.rubric_sha256 == EVAL_RUBRIC_SHA256


# ---------------------------------------------------------------------------------------------
# Screen input, parsing and retries
# ---------------------------------------------------------------------------------------------


def test_screen_input_holds_the_reply_and_no_gold_label() -> None:
    from groundcite import hint_screen as hs

    row = scenario_row(3)
    reply = _render(CITE_LEAKS[17], row)
    system, messages = hs.build_hint_screen_messages(parse_row(row), reply)
    assert system == hs.render_hint_screen()
    text = "\n".join(m["content"] for m in messages)
    assert _block(reply) in text
    assert messages[-1]["role"] == "user"
    # The screen judges wording, not correctness: the gold label is never part of its input.
    assert "Gold mode" not in text and "gold label" not in text.lower()
    assert "supporting snippet ID" not in text


@pytest.mark.parametrize(
    ("raw", "leak"),
    [
        ('{"meta_leak": true, "reason": "Says an intended reference exists."}', True),
        ('{"meta_leak": false, "reason": "Talks only about the user\'s report."}', False),
        ('```json\n{"meta_leak": true, "reason": "Mentions an answer key."}\n```', True),
    ],
)
def test_parse_screen_output_reads_the_verdict(raw: str, leak: bool) -> None:
    from groundcite import hint_screen as hs

    verdict = hs.parse_hint_screen_output(raw)
    assert verdict.meta_leak is leak
    assert isinstance(verdict.reason, str) and verdict.reason.strip()


@pytest.mark.parametrize(
    "raw",
    [
        "Yes, it leaks.",
        '{"reason": "no key"}',
        '{"meta_leak": "true", "reason": "a string, not a bool"}',
        '{"meta_leak": 1, "reason": "an int, not a bool"}',
        '{"meta_leak": true, "reason": "  "}',
        '{"meta_leak": true}',
        "[true]",
    ],
)
def test_parse_screen_output_rejects_malformed_verdicts(raw: str) -> None:
    from groundcite import hint_screen as hs
    from groundcite.judge import JudgeParseError

    with pytest.raises(JudgeParseError):
        hs.parse_hint_screen_output(raw)


class _Scripted:
    name = "mock:screen_script"

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.sent: list[tuple[str | None, list[dict[str, str]]]] = []

    def generate(self, messages: Any, system: str | None, params: Any) -> str:
        self.sent.append((system, [dict(m) for m in messages]))
        return self.answers[min(len(self.sent), len(self.answers)) - 1]


def test_screen_reply_returns_verdict_model_and_prompt_hash() -> None:
    from groundcite import hint_screen as hs

    row = scenario_row(4)
    backend = _Scripted(['{"meta_leak": true, "reason": "Names an intended reference."}'])
    result = hs.screen_reply(parse_row(row), _render(CITE_LEAKS[17], row), backend, params={})
    assert result.meta_leak is True
    assert "intended reference" in result.reason
    assert result.error is None
    assert result.attempts == 1
    assert result.model == "mock:screen_script"
    assert result.prompt_sha256 == hs.hint_screen_sha256()


def test_screen_reply_retries_once_with_a_reminder() -> None:
    from groundcite import hint_screen as hs

    row = scenario_row(5)
    backend = _Scripted(["not json", '{"meta_leak": false, "reason": "Plain diagnostic step."}'])
    result = hs.screen_reply(parse_row(row), _render(NORMAL[0], row), backend, params={})
    assert result.meta_leak is False
    assert result.error is None
    assert result.attempts == 2
    first, second = (msgs[-1]["content"] for _, msgs in backend.sent)
    assert second != first and second.startswith(first)
    assert "meta_leak" in second[len(first) :]


def test_screen_reply_records_an_error_after_the_retries() -> None:
    from groundcite import hint_screen as hs

    row = scenario_row(6)
    backend = _Scripted(["nope", "still nope"])
    result = hs.screen_reply(parse_row(row), _render(NORMAL[1], row), backend, params={})
    assert result.error and result.error.strip()
    assert result.attempts == 2
    assert "nope" in result.raw


def test_mock_judge_answers_the_screen_with_a_valid_verdict() -> None:
    """``--judge mock`` builds must still run: the built-in mock judge speaks both formats."""
    from groundcite import hint_screen as hs

    row = scenario_row(7)
    result = hs.screen_reply(
        parse_row(row), _render(NORMAL[2], row), backends.build_backend("mock:judge"), params={}
    )
    assert result.error is None
    assert result.meta_leak is False


# ---------------------------------------------------------------------------------------------
# Filter stage
# ---------------------------------------------------------------------------------------------


def test_hint_leak_semantic_is_a_rejection_reason_between_soft_check_and_the_judge() -> None:
    from groundcite import filter as f

    assert f.HINT_LEAK_SEMANTIC == "hint_leak_semantic"
    reasons = list(f.REJECTION_REASONS)
    assert reasons.index(f.SOFT_CHECK) < reasons.index(f.HINT_LEAK_SEMANTIC)
    assert reasons.index(f.HINT_LEAK_SEMANTIC) < reasons.index(f.JUDGE_ERROR)
    assert reasons.index(f.HINT_LEAK) < reasons.index(f.HINT_LEAK_SEMANTIC)


def test_filter_version_is_bumped() -> None:
    from groundcite.filter import FILTER_VERSION

    assert FILTER_VERSION != "f1"


def test_screen_stage_rejects_flags_and_errors_and_passes_clean_replies() -> None:
    from groundcite import hint_screen as hs
    from groundcite.filter import HINT_LEAK_SEMANTIC, JUDGE_ERROR, hint_screen_stage

    row = scenario_row(8)
    scenario = parse_row(row)
    leak = _Scripted(['{"meta_leak": true, "reason": "Says the bulletin was intended."}'])
    rejection = hint_screen_stage(hs.screen_reply(scenario, "x.", leak, params={}))
    assert rejection is not None and rejection.reason == HINT_LEAK_SEMANTIC
    assert "intended" in rejection.detail

    clean = _Scripted(['{"meta_leak": false, "reason": "Plain step."}'])
    assert hint_screen_stage(hs.screen_reply(scenario, "x.", clean, params={})) is None

    broken = _Scripted(["?", "?"])
    rejection = hint_screen_stage(hs.screen_reply(scenario, "x.", broken, params={}))
    assert rejection is not None and rejection.reason == JUDGE_ERROR


# ---------------------------------------------------------------------------------------------
# End to end: the validator's 34 leaks and 10 normal replies through build_dataset
# ---------------------------------------------------------------------------------------------


def test_fixture_has_34_leaks_and_the_regex_misses_most_of_them() -> None:
    assert len(CITE_LEAKS) + len(ABSTAIN_LEAKS) == 34
    assert len(NORMAL) == 10
    misses = [t for t in CITE_LEAKS + ABSTAIN_LEAKS if not _regex_caught(t.format(id="X"))]
    assert len(misses) >= 20, "the fixture no longer exercises the semantic screen"


def test_every_paraphrased_leak_is_rejected(built: dict[str, Any]) -> None:
    by_id = {r["scenario_id"]: r for r in built["rejections"]}
    kept_ids = {rec["meta"]["scenario_id"] for rec in built["train"]}
    for row, reply in built["leaks"]:
        assert row["id"] not in kept_ids, f"leak kept: {reply}"
        reason = by_id[row["id"]]["reason"]
        if _regex_caught(reply):
            assert reason == "hint_leak", reply
        else:
            assert reason == "hint_leak_semantic", f"{reason}: {reply}"


def test_intended_reference_leak_is_rejected_semantically(built: dict[str, Any]) -> None:
    row, reply = next((r, t) for r, t in built["leaks"] if t.startswith(INTENDED_REFERENCE))
    rejection = [r for r in built["rejections"] if r["scenario_id"] == row["id"]]
    assert rejection, f"kept: {reply}"
    assert rejection[0]["reason"] == "hint_leak_semantic"
    assert rejection[0]["detail"].strip()


def test_every_normal_reply_is_kept(built: dict[str, Any]) -> None:
    kept = {rec["meta"]["scenario_id"]: rec["messages"][-1]["content"] for rec in built["train"]}
    for row, reply in built["normals"]:
        assert kept.get(row["id"]) == reply, f"normal reply not kept: {reply}"


def test_every_kept_candidate_was_screened(built: dict[str, Any], judge: type[_Recorder]) -> None:
    screened = [text for kind, _, text in judge.requests if kind == "screen"]
    for rec in built["train"]:
        reply = rec["messages"][-1]["content"]
        assert any(_block(reply) in text for text in screened), f"never screened: {reply}"


def test_regex_rejects_never_reach_the_screen(
    built: dict[str, Any], judge: type[_Recorder]
) -> None:
    screened = [text for kind, _, text in judge.requests if kind == "screen"]
    regex = [reply for _, reply in built["leaks"] if _regex_caught(reply)]
    assert regex
    for reply in regex:
        assert not any(_block(reply) in text for text in screened), f"screened: {reply}"
    assert judge.leak_calls == len(built["leaks"]) - len(regex) + len(built["normals"])


def test_flagged_leaks_are_never_sent_to_the_rubric_judge(
    built: dict[str, Any], judge: type[_Recorder]
) -> None:
    rubric = [text for kind, _, text in judge.requests if kind == "rubric"]
    for _, reply in built["leaks"]:
        assert not any(_block(reply) in text for text in rubric), f"judged a leak: {reply}"
    assert judge.calls == len(built["normals"])


def test_screen_runs_before_the_rubric_judge(built: dict[str, Any], judge: type[_Recorder]) -> None:
    for rec in built["train"]:
        block = _block(rec["messages"][-1]["content"])
        kinds = [kind for kind, _, text in judge.requests if block in text]
        assert kinds == ["screen", "rubric"], kinds


def test_dataset_judge_scores_with_the_eval_rubric(
    built: dict[str, Any], judge: type[_Recorder]
) -> None:
    from groundcite.judge import render_rubric

    systems = {system for kind, system, _ in judge.requests if kind == "rubric"}
    assert systems == {render_rubric()}
    assert built["manifest"]["rubric_sha256"] == EVAL_RUBRIC_SHA256


def test_manifest_counts_semantic_rejections_separately(built: dict[str, Any]) -> None:
    by_reason = built["manifest"]["rejections_by_reason"]
    regex = sum(_regex_caught(reply) for _, reply in built["leaks"])
    assert by_reason.get("hint_leak") == regex
    assert by_reason.get("hint_leak_semantic") == len(built["leaks"]) - regex
    assert by_reason == dict(Counter(r["reason"] for r in built["rejections"]))


def test_manifest_records_the_screen_prompt_and_its_call_count(built: dict[str, Any]) -> None:
    from groundcite import hint_screen as hs

    m = built["manifest"]
    assert m["hint_screen_version"] == hs.HINT_SCREEN_VERSION
    assert m["hint_screen_sha256"] == hs.hint_screen_sha256()
    regex = sum(_regex_caught(reply) for _, reply in built["leaks"])
    assert m["hint_screen_calls"] == len(built["leaks"]) - regex + len(built["normals"])


def test_filter_report_lists_semantic_rejections_on_their_own_row(built: dict[str, Any]) -> None:
    from groundcite import hint_screen as hs

    report = built["report"]
    regex = sum(_regex_caught(reply) for _, reply in built["leaks"])
    semantic = len(built["leaks"]) - regex
    assert re.search(rf"^\| `hint_leak` \| {regex} \|$", report, re.MULTILINE)
    assert re.search(rf"^\| `hint_leak_semantic` \| {semantic} \|$", report, re.MULTILINE)
    assert hs.HINT_SCREEN_VERSION in report


def test_unparseable_screen_output_is_a_judge_error_not_a_keep(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Garbled:
        name = "mock:judge_garbled"

        def generate(self, messages: Any, system: str | None, params: Any) -> str:
            if system is not None and '"meta_leak"' in system:
                return "I cannot tell."
            return json.dumps(
                {"spec_adherence": 1, "robustness": 5, "failed_checks": [], "reasoning": "ok"}
            )

    monkeypatch.setitem(backends.MOCKS, "judge_garbled", Garbled)
    row = scenario_row(9)
    cands = write_jsonl(
        tmp_path / "candidates.jsonl", [candidate_row(row, 0, _render(NORMAL[0], row))]
    )
    out = tmp_path / "v1"
    assert _build(cands, out, "mock:judge_garbled") == 0
    assert read_jsonl(out / "train.jsonl") == []
    (rejection,) = read_jsonl(out / "rejections.jsonl")
    assert rejection["reason"] == "judge_error"
    assert "screen" in rejection["detail"].lower()


def test_build_with_the_builtin_mock_judge_still_keeps_on_spec_replies(
    env: Path, tmp_path: Path
) -> None:
    rows = [scenario_row(i) for i in range(3)]
    cands = write_jsonl(
        tmp_path / "candidates.jsonl",
        [candidate_row(row, 0, _render(NORMAL[i], row)) for i, row in enumerate(rows)],
    )
    out = tmp_path / "v1"
    assert _build(cands, out, "mock") == 0
    assert len(read_jsonl(out / "train.jsonl")) == 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hint_screen_calls"] == 3
