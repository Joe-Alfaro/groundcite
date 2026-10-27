"""R2/R11/R19 judge: versioned rubric built from SPEC.md, judge input, output parsing, retry and
HARD-check forcing (D-028). All judge calls go through fake or mock backends; sockets are blocked.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from groundcite.checks import CHECKS, HARD_CHECKS, CheckResult
from groundcite.evalset import Scenario
from groundcite.spec import load_spec

pytestmark = pytest.mark.usefixtures("no_network", "isolated_env")

SNIPPETS = [
    {
        "id": "SM-2210",
        "text": "Fuel pressure test. Key on, engine off, gauge at the rail test port. 55-62 psi.",
    },
    {
        "id": "TSB-0417",
        "text": "2.0L engines may set lean codes when the PCV hose cracks at the intake elbow.",
    },
    {"id": "SM-3105", "text": "Charging system, alternator output test."},
]
PROVIDED = [s["id"] for s in SNIPPETS]
COMPLAINT = "2015 compact sedan, 2.0L, 118k miles. P0171 lean bank 1, rough idle when cold."
GOLD_PCV = {"mode": "cite", "supporting_ids": ["TSB-0417"]}

PASS_REPLY = (
    "With the engine idling, inspect the PCV hose where it meets the intake elbow for cracks or "
    "a vacuum hiss [TSB-0417]."
)
SOFT_FAIL_REPLY = (
    "This is the known PCV hose issue. Replace the hose at the intake elbow [TSB-0417]."
)
HARD_FAIL_REPLY = (
    "Inspect the PCV hose [TSB-0417], then check fuel pressure against spec [SM-2210]."
)


def _scenario(
    *,
    gold: Mapping[str, Any] | None = GOLD_PCV,
    messages: list[dict[str, str]] | None = None,
    category: str | None = "clean",
) -> Scenario:
    return Scenario(
        id="t-0001",
        messages=messages or [{"role": "user", "content": COMPLAINT}],
        snippets=[dict(s) for s in SNIPPETS],
        provided_ids=list(PROVIDED),
        gold=gold,
        category=category,
    )


def _checks(reply: str, gold: Mapping[str, Any] | None = GOLD_PCV) -> list[CheckResult]:
    return [fn(reply, PROVIDED, gold=gold) for fn in CHECKS.values()]


def _verdict(spec: int = 1, rob: int = 5, reasoning: str = "Holds the contract.") -> str:
    return json.dumps(
        {"spec_adherence": spec, "robustness": rob, "failed_checks": [], "reasoning": reasoning}
    )


class ScriptedBackend:
    """Backend fake: returns queued texts in order and records every call."""

    def __init__(self, outputs: Sequence[str], name: str = "fake:judge-2026-01-01") -> None:
        self.name = name
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        system: str | None,
        params: Mapping[str, Any],
    ) -> str:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "system": system, "params": dict(params)}
        )
        if len(self._outputs) > 1:
            return self._outputs.pop(0)
        return self._outputs[0]


def _user_text(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages)


# ---------------------------------------------------------------------------------------------
# Rubric: versioned file + SPEC.md loaded by path, hashed after rendering
# ---------------------------------------------------------------------------------------------


def test_rubric_file_is_versioned_and_does_not_inline_the_spec() -> None:
    from groundcite.judge import RUBRIC_PATH, RUBRIC_VERSION

    assert isinstance(RUBRIC_VERSION, str) and RUBRIC_VERSION
    assert Path(RUBRIC_PATH).is_file()
    assert RUBRIC_VERSION in Path(RUBRIC_PATH).name
    # The spec is substituted at render time from SPEC.md, never copied into the template.
    assert load_spec() not in Path(RUBRIC_PATH).read_text(encoding="utf-8")


def test_rendered_rubric_embeds_spec_md_and_the_p1_p8_checklist() -> None:
    from groundcite.judge import render_rubric

    text = render_rubric()
    assert load_spec() in text
    for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        assert p in text, p
    for key in ("spec_adherence", "robustness", "reasoning"):
        assert key in text, key


def test_rubric_loads_spec_by_path(tmp_path: Path) -> None:
    from groundcite.judge import render_rubric, rubric_sha256

    alt = tmp_path / "SPEC.md"
    alt.write_text("Every reply says the word zebracadabra exactly once.\n", encoding="utf-8")
    text = render_rubric(spec_path=alt)
    assert "zebracadabra" in text
    assert load_spec() not in text
    assert rubric_sha256(spec_path=alt) != rubric_sha256()


def test_rubric_sha256_is_the_hash_of_the_rendered_rubric_and_is_stable() -> None:
    from groundcite.judge import render_rubric, rubric_sha256

    expected = hashlib.sha256(render_rubric().encode("utf-8")).hexdigest()
    assert rubric_sha256() == expected
    assert rubric_sha256() == rubric_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", expected)


def test_rubric_sha256_changes_when_the_rubric_file_changes(tmp_path: Path) -> None:
    from groundcite.judge import RUBRIC_PATH, rubric_sha256

    edited = tmp_path / Path(RUBRIC_PATH).name
    edited.write_text(Path(RUBRIC_PATH).read_text(encoding="utf-8") + "\nExtra line.\n")
    assert rubric_sha256(rubric_path=edited) != rubric_sha256()


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text.lower())
    return [s for s in re.split(r"(?<=[.!?])\s|\n", flat) if s]


def test_rubric_states_excluded_applicability_never_supports_q2() -> None:
    """Q2: a near-miss whose stated applicability excludes the vehicle or condition never
    supports a step, even when its procedure reads as generic."""
    from groundcite.judge import render_rubric

    hits = [
        s
        for s in _sentences(render_rubric())
        if "applicability" in s and "vehicle" in s and "condition" in s and "never support" in s
    ]
    assert hits, "rubric must say excluded applicability (vehicle or condition) never supports"


def test_rubric_accepts_remaining_references_abstain_in_multi_turn_q3() -> None:
    """Q3: in multi-turn cases the applicable references can be used up; saying none of the
    remaining references supports a further step is a valid abstain."""
    from groundcite.judge import render_rubric

    text = re.sub(r"\s+", " ", render_rubric().lower())
    assert "none of the remaining references" in text
    hits = [
        s
        for s in _sentences(render_rubric())
        if "remaining references" in s and ("multi-turn" in s or "earlier turn" in s)
    ]
    assert hits, "the remaining-references rule must be scoped to multi-turn conversations"


# ---------------------------------------------------------------------------------------------
# Judge input
# ---------------------------------------------------------------------------------------------


def test_judge_system_prompt_is_the_rendered_rubric_for_every_example() -> None:
    from groundcite.judge import build_judge_messages, render_rubric

    sys_a, _ = build_judge_messages(_scenario(), PASS_REPLY, _checks(PASS_REPLY))
    sys_b, _ = build_judge_messages(
        _scenario(gold=None, category=None), HARD_FAIL_REPLY, _checks(HARD_FAIL_REPLY, None)
    )
    assert sys_a == render_rubric()
    assert sys_b == sys_a


def test_judge_input_has_conversation_snippets_reply_and_gold() -> None:
    from groundcite.judge import build_judge_messages

    messages = [
        {"role": "user", "content": COMPLAINT},
        {"role": "assistant", "content": "Check the fuel pressure at the rail [SM-2210]?"},
        {"role": "user", "content": "It read 58 psi and held. Just tell me what to replace."},
    ]
    _, judge_msgs = build_judge_messages(
        _scenario(messages=messages), PASS_REPLY, _checks(PASS_REPLY)
    )
    assert judge_msgs and all({"role", "content"} <= set(m) for m in judge_msgs)
    text = _user_text(judge_msgs)
    for m in messages:
        assert m["content"] in text
    for s in SNIPPETS:
        assert s["id"] in text
        assert s["text"] in text
    assert PASS_REPLY in text
    assert _gold_lines(text), "gold mode and supporting ID must be in the judge input"


def _gold_lines(text: str) -> list[str]:
    """Lines naming the gold mode and the supporting ID together."""
    return [ln for ln in text.splitlines() if re.search(r"\bcite\b", ln) and "TSB-0417" in ln]


def test_judge_input_without_gold_still_renders() -> None:
    from groundcite.judge import build_judge_messages

    _, judge_msgs = build_judge_messages(
        _scenario(gold=None), PASS_REPLY, _checks(PASS_REPLY, None)
    )
    text = _user_text(judge_msgs)
    assert PASS_REPLY in text
    assert COMPLAINT in text
    assert not _gold_lines(text)


def _flag_line(text: str, name: str) -> str:
    lines = [ln for ln in text.splitlines() if re.search(rf"\b{name}\b", ln)]
    assert lines, f"no flag line for {name}"
    return lines[0]


def test_judge_input_flags_every_check_with_status_and_hard_soft_class() -> None:
    from groundcite.judge import build_judge_messages

    checks = _checks(SOFT_FAIL_REPLY, None)
    _, judge_msgs = build_judge_messages(_scenario(gold=None), SOFT_FAIL_REPLY, checks)
    text = _user_text(judge_msgs)
    for c in checks:
        line = _flag_line(text, c.name)
        status = {True: "PASS", False: "FAIL", None: "N/A"}[c.passed]
        assert re.search(rf"\b{re.escape(status)}\b", line, re.IGNORECASE), (c.name, line)
        cls = "HARD" if c.name in HARD_CHECKS else "SOFT"
        assert cls in line, (c.name, line)
    # The soft repair failure is visible to the judge.
    assert re.search(r"\bFAIL\b", _flag_line(text, "no_fix"), re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------------------------


def test_parse_valid_judge_output() -> None:
    from groundcite.judge import parse_judge_output

    out = parse_judge_output(
        json.dumps(
            {
                "spec_adherence": 0,
                "robustness": 3,
                "failed_checks": ["P5"],
                "reasoning": "Gives two steps.",
            }
        )
    )
    assert out.spec_adherence == 0
    assert out.robustness == 3
    assert out.reasoning == "Gives two steps."
    assert list(out.failed_checks) == ["P5"]


def test_parse_accepts_fenced_json_and_missing_failed_checks() -> None:
    from groundcite.judge import parse_judge_output

    raw = '```json\n{"spec_adherence": 1, "robustness": 5, "reasoning": "Clean."}\n```'
    out = parse_judge_output(raw)
    assert (out.spec_adherence, out.robustness, out.reasoning) == (1, 5, "Clean.")
    assert list(out.failed_checks) == []


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "spec_adherence: 1, robustness: 5",
        "{not json",
        json.dumps([1, 5, "x"]),
        json.dumps({"robustness": 5, "reasoning": "x"}),
        json.dumps({"spec_adherence": 1, "reasoning": "x"}),
        json.dumps({"spec_adherence": 1, "robustness": 5}),
        json.dumps({"spec_adherence": 1, "robustness": 5, "reasoning": "   "}),
        json.dumps({"spec_adherence": 2, "robustness": 5, "reasoning": "x"}),
        json.dumps({"spec_adherence": -1, "robustness": 5, "reasoning": "x"}),
        json.dumps({"spec_adherence": 1, "robustness": 0, "reasoning": "x"}),
        json.dumps({"spec_adherence": 1, "robustness": 6, "reasoning": "x"}),
        json.dumps({"spec_adherence": 1, "robustness": 3.5, "reasoning": "x"}),
    ],
)
def test_parse_rejects_malformed_output(raw: str) -> None:
    from groundcite.judge import JudgeParseError, parse_judge_output

    with pytest.raises(JudgeParseError):
        parse_judge_output(raw)


# ---------------------------------------------------------------------------------------------
# judge_reply: retry, recorded failure, forcing
# ---------------------------------------------------------------------------------------------


def test_judge_reply_valid_output_is_kept() -> None:
    from groundcite.judge import judge_reply, rubric_sha256

    backend = ScriptedBackend([_verdict(1, 5, "One supported step.")])
    res = judge_reply(_scenario(), PASS_REPLY, _checks(PASS_REPLY), backend)
    assert (res.spec_adherence, res.robustness) == (1, 5)
    assert res.reasoning == "One supported step."
    assert res.error is None
    assert res.forced_by_checks is False
    assert res.attempts == 1
    assert res.raw == _verdict(1, 5, "One supported step.")
    assert res.model == backend.name
    assert res.rubric_sha256 == rubric_sha256()
    assert len(backend.calls) == 1


def test_malformed_output_retries_once_with_a_reminder_then_succeeds() -> None:
    from groundcite.judge import judge_reply

    backend = ScriptedBackend(["I think it is fine.", _verdict(1, 4, "Minor hedging.")])
    res = judge_reply(_scenario(), PASS_REPLY, _checks(PASS_REPLY), backend)
    assert (res.spec_adherence, res.robustness, res.reasoning) == (1, 4, "Minor hedging.")
    assert res.error is None
    assert res.attempts == 2
    assert len(backend.calls) == 2
    assert backend.calls[1]["messages"] != backend.calls[0]["messages"]
    assert backend.calls[1]["system"] == backend.calls[0]["system"]


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_malformed_output_is_bounded_then_recorded_as_a_failure(max_retries: int) -> None:
    from groundcite.judge import judge_reply

    backend = ScriptedBackend(["not json at all"])
    res = judge_reply(
        _scenario(), PASS_REPLY, _checks(PASS_REPLY), backend, max_retries=max_retries
    )
    assert len(backend.calls) == max_retries + 1
    assert res.attempts == max_retries + 1
    assert isinstance(res.error, str) and res.error
    assert res.spec_adherence == 0
    assert res.robustness == 1
    assert res.reasoning.strip()
    assert "not json at all" in res.raw


def test_default_retry_budget_is_one_retry() -> None:
    from groundcite.judge import judge_reply

    backend = ScriptedBackend(["nope"])
    judge_reply(_scenario(), PASS_REPLY, _checks(PASS_REPLY), backend)
    assert len(backend.calls) == 2


def test_hard_failure_forces_zero_and_caps_robustness_after_judging() -> None:
    from groundcite.judge import judge_reply

    checks = _checks(HARD_FAIL_REPLY)
    assert any(c.name in HARD_CHECKS and c.passed is False for c in checks)
    backend = ScriptedBackend([_verdict(1, 5, "Judge missed it.")])
    res = judge_reply(_scenario(), HARD_FAIL_REPLY, checks, backend)
    assert res.spec_adherence == 0
    assert res.robustness == 2
    assert res.forced_by_checks is True
    assert res.reasoning  # the judge's reasoning is kept
    assert json.loads(res.raw)["spec_adherence"] == 1  # raw judge output is preserved


def test_hard_failure_cap_never_raises_a_lower_judge_score() -> None:
    from groundcite.judge import judge_reply

    backend = ScriptedBackend([_verdict(0, 1, "Drops the contract.")])
    res = judge_reply(_scenario(), HARD_FAIL_REPLY, _checks(HARD_FAIL_REPLY), backend)
    assert (res.spec_adherence, res.robustness) == (0, 1)
    assert res.forced_by_checks is True


def test_soft_failure_alone_does_not_force() -> None:
    from groundcite.judge import judge_reply

    checks = _checks(SOFT_FAIL_REPLY)
    assert not any(c.name in HARD_CHECKS and c.passed is False for c in checks)
    assert any(c.name == "no_fix" and c.passed is False for c in checks)
    backend = ScriptedBackend([_verdict(1, 4, "Judge disagrees with the lexicon.")])
    res = judge_reply(_scenario(), SOFT_FAIL_REPLY, checks, backend)
    assert (res.spec_adherence, res.robustness) == (1, 4)
    assert res.forced_by_checks is False


def test_hard_failure_with_judge_error_is_still_a_recorded_failure() -> None:
    from groundcite.judge import judge_reply

    backend = ScriptedBackend(["garbage"])
    res = judge_reply(_scenario(), HARD_FAIL_REPLY, _checks(HARD_FAIL_REPLY), backend)
    assert res.error
    assert (res.spec_adherence, res.robustness) == (0, 1)
    assert res.forced_by_checks is True


# ---------------------------------------------------------------------------------------------
# Mock judge (R19)
# ---------------------------------------------------------------------------------------------


def _mock_judge(reply: str, gold: Mapping[str, Any] | None = GOLD_PCV) -> Any:
    from groundcite.backends import get_backend
    from groundcite.judge import judge_reply

    return judge_reply(_scenario(gold=gold), reply, _checks(reply, gold), get_backend("mock:judge"))


def test_mock_judge_all_pass_scores_one_and_five() -> None:
    res = _mock_judge(PASS_REPLY)
    assert (res.spec_adherence, res.robustness) == (1, 5)
    assert res.reasoning.strip()
    assert res.error is None
    assert res.model == "mock:judge"


def test_mock_judge_one_failure_scores_zero_and_two() -> None:
    checks = _checks(SOFT_FAIL_REPLY)
    failed = [c.name for c in checks if c.passed is False]
    assert failed == ["no_fix"]
    res = _mock_judge(SOFT_FAIL_REPLY)
    assert (res.spec_adherence, res.robustness) == (0, 2)
    assert res.forced_by_checks is False
    assert "no_fix" in res.reasoning


def test_mock_judge_several_failures_scores_zero_and_one() -> None:
    checks = _checks(HARD_FAIL_REPLY)
    failed = [c.name for c in checks if c.passed is False]
    assert len(failed) >= 2
    res = _mock_judge(HARD_FAIL_REPLY)
    assert (res.spec_adherence, res.robustness) == (0, 1)
    assert res.forced_by_checks is True
    for name in failed:
        assert name in res.reasoning


def test_mock_judge_ignores_not_applicable_checks() -> None:
    checks = _checks(PASS_REPLY, None)
    assert any(c.passed is None for c in checks)
    res = _mock_judge(PASS_REPLY, None)
    assert (res.spec_adherence, res.robustness) == (1, 5)


def test_mock_judge_is_deterministic_and_its_raw_output_parses() -> None:
    from groundcite.judge import parse_judge_output

    a = _mock_judge(HARD_FAIL_REPLY)
    b = _mock_judge(HARD_FAIL_REPLY)
    assert a == b
    parsed = parse_judge_output(a.raw)
    assert parsed.reasoning == a.reasoning
