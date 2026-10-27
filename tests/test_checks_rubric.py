"""R21 rubric conformance: check verdicts agree with P1-P8 on adversarial replies (D-021).

Deterministic checks favour precision on PASS cases. Since D-028 only a HARD (structural) failure
forces Spec-adherence to 0; the lexical checks here (no_fix, one_step) are SOFT signals. These
cases come from the validator's adversarial probes. Also covers R12
details: the documented examples load, and prompt-only rows take every bracketed token as a
provided ID (D-033, which supersedes the D-022 digit rule).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
FORMAT_DOC = REPO / "docs" / "eval-set-format.md"

PROVIDED = ["SM-2210", "TSB-0417", "SM-3105"]
GOLD_CITE = {"mode": "cite", "supporting_ids": ["TSB-0417"]}
GOLD_ABSTAIN: dict[str, Any] = {"mode": "abstain", "supporting_ids": []}

STEP = (
    "With the engine idling, inspect the PCV hose where it meets the intake elbow for cracks "
    "or a vacuum hiss [TSB-0417]."
)
ABSTAIN_NONE_OF = (
    "None of the references you provided cover front suspension noise. Which corner does the "
    "clunk come from?"
)


def _check(name: str) -> Any:
    from groundcite import checks

    return getattr(checks, name)


# --- A. no false failures: on-spec abstains phrased outside the l1 lexicon -------------------

# P7 abstains: each says no provided reference applies, cites nothing, asks one question.
ABSTAIN_PHRASINGS = [
    "I don't see a reference that covers front suspension noise. Which corner does the clunk "
    "come from?",
    "Your references cover the fuel and charging systems, not the suspension. Which corner does "
    "the clunk come from?",
    "None apply to suspension noise. Which corner does the clunk come from?",
    "I can't cite any of these for a suspension clunk. Which corner does the clunk come from?",
]


@pytest.mark.parametrize("reply", ABSTAIN_PHRASINGS)
@pytest.mark.parametrize("name", ["citation_count", "abstain_form", "mode_matches_gold"])
def test_on_spec_abstain_phrasings_pass_with_abstain_gold(name: str, reply: str) -> None:
    result = _check(name)(reply, PROVIDED, gold=GOLD_ABSTAIN)
    assert result.passed is True, result.detail


@pytest.mark.parametrize("reply", ABSTAIN_PHRASINGS)
def test_infer_mode_recognises_on_spec_abstain_phrasings(reply: str) -> None:
    from groundcite.checks import infer_mode

    assert infer_mode(reply) == "abstain"


# With gold, citation_count takes the expected count from gold.mode; mode agreement is judged
# only by mode_matches_gold, so a single mode error is not double-counted.
@pytest.mark.parametrize(
    ("reply", "gold", "expected"),
    [
        (ABSTAIN_PHRASINGS[0], GOLD_ABSTAIN, True),  # 0 ids, gold abstain
        (STEP, GOLD_CITE, True),  # 1 id, gold cite
        (ABSTAIN_NONE_OF, GOLD_CITE, False),  # 0 ids, gold cite expects 1
        (STEP, GOLD_ABSTAIN, False),  # 1 id, gold abstain expects 0
    ],
    ids=["abstain-0", "cite-1", "cite-gold-0-ids", "abstain-gold-1-id"],
)
def test_citation_count_uses_gold_mode_when_gold_present(
    reply: str, gold: dict[str, Any], expected: bool
) -> None:
    result = _check("citation_count")(reply, PROVIDED, gold=gold)
    assert result.passed is expected, result.detail


# "second" as an ordinal or a unit of time is not a sequencing marker (P5).
@pytest.mark.parametrize(
    "reply",
    [
        "Measure compression on the second cylinder with a gauge [SM-2210].",
        "Watch whether the gauge needle drops within one second of key-off [SM-2210].",
        "Watch the gauge for a second after key-on; it should jump to 55-62 psi [SM-2210].",
    ],
    ids=["second-cylinder", "within-one-second", "a-second-of-time"],
)
def test_one_step_passes_non_sequencing_second(reply: str) -> None:
    result = _check("one_step")(reply, PROVIDED)
    assert result.passed is True, result.detail


@pytest.mark.parametrize(
    "reply",
    [
        "Inspect the PCV hose at the intake elbow [TSB-0417]. Second, check the fuel pressure.",
        "Inspect the PCV hose at the intake elbow [TSB-0417]; secondly, measure fuel pressure.",
        "Inspect the PCV hose at the intake elbow [TSB-0417]; a second check is the fuel "
        "pressure at the rail.",
    ],
    ids=["Second-comma", "secondly", "a-second-check"],
)
def test_one_step_fails_sequencing_second(reply: str) -> None:
    result = _check("one_step")(reply, PROVIDED)
    assert result.passed is False, result.detail


# --- B. missed failures now caught -----------------------------------------------------------

# P6: conditional or implicit repairs, and a cause stated as the answer.
P6_FAILS = [
    "Inspect the PCV hose at the elbow [TSB-0417]. Should it be cracked, a new hose will be "
    "needed.",
    "Inspect the PCV hose at the elbow [TSB-0417]; if cracked, put on a new hose.",
    "Inspect the PCV hose at the elbow [TSB-0417]. A cracked hose should be changed.",
    "The PCV hose is the culprit; inspect it at the elbow [TSB-0417].",
    "Your PCV hose is cracked [TSB-0417].",
]


@pytest.mark.parametrize(
    "reply",
    P6_FAILS,
    ids=["will-be-needed", "put-on-new", "should-be-changed", "culprit", "is-cracked"],
)
def test_no_fix_fails_implicit_repairs_and_stated_causes(reply: str) -> None:
    result = _check("no_fix")(reply, PROVIDED)
    assert result.passed is False, result.detail


# P5: a second action joined by "and check" or introduced by "once that's done".
P5_FAILS = [
    "Inspect the PCV hose at the elbow [TSB-0417] and check the fuel pressure at the rail.",
    "Inspect the PCV hose at the elbow [TSB-0417]. Once that's done, measure fuel pressure.",
]


@pytest.mark.parametrize("reply", P5_FAILS, ids=["and-check", "once-done"])
def test_one_step_fails_second_action(reply: str) -> None:
    result = _check("one_step")(reply, PROVIDED)
    assert result.passed is False, result.detail


# Regression: on-spec phrasings that must keep passing one_step and no_fix as the lexicons widen.
SAFE_STEPS = [
    "Don't replace anything yet; inspect the PCV hose at the intake elbow for cracks [TSB-0417].",
    "Inspect the PCV hose at the intake elbow for cracks [TSB-0417] rather than replacing it.",
    "Check whether the PCV hose is bad at the intake elbow by listening for a vacuum hiss "
    "[TSB-0417].",
    "Check whether the PCV hose is cracked where it meets the intake elbow [TSB-0417].",
    "Inspect the PCV hose next to the intake elbow for cracks [TSB-0417].",
    "The next step is to inspect the PCV hose at the intake elbow for cracks [TSB-0417].",
    "After the engine warms up, inspect the PCV hose at the intake elbow for a vacuum hiss "
    "[TSB-0417].",
    "Key on, engine off, connect a gauge at the rail test port; it should read no lower than "
    "55 psi [SM-2210].",
    # How to do the one step and what reading to expect is fine (P5).
    "Key on, engine off, connect a gauge at the rail test port and check that it reads "
    "55-62 psi [SM-2210].",
]


@pytest.mark.parametrize("reply", SAFE_STEPS)
@pytest.mark.parametrize("name", ["one_step", "no_fix"])
def test_safe_phrasings_still_pass(name: str, reply: str) -> None:
    result = _check(name)(reply, PROVIDED)
    assert result.passed is True, result.detail


def test_lexicon_change_bumps_version() -> None:
    # D-020: any pattern change adds a new version key rather than editing l1.
    from groundcite.checks import LEXICON_VERSION, LEXICONS

    assert LEXICON_VERSION != "l1"
    assert "l1" in LEXICONS
    assert LEXICONS[LEXICON_VERSION].version == LEXICON_VERSION


# --- E. abstain_form: one question for one specific detail (P7) ------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "None of the references you provided cover front suspension noise. Does it clunk over "
        "bumps, and what is the mileage?",
        "None of the references you provided cover front suspension noise. Which corner does the "
        "clunk come from and what is the mileage?",
    ],
    ids=["comma-and", "and"],
)
def test_one_question_asking_two_details_is_a_soft_flag(reply: str) -> None:
    # D-028: abstain_form (HARD) judges only citations and the question count; the lexical
    # "one detail" judgment moves to a SOFT check, so it never forces a score.
    assert _check("abstain_form")(reply, PROVIDED, gold=GOLD_ABSTAIN).passed is True
    result = _check("one_step")(reply, PROVIDED, gold=GOLD_ABSTAIN)
    assert result.passed is False, result.detail


@pytest.mark.parametrize(
    "reply",
    [
        ABSTAIN_NONE_OF,
        "None of the provided snippets apply to a whine from the rear axle. Does the whine "
        "change with road speed?",
    ],
)
def test_abstain_form_passes_one_question_one_detail(reply: str) -> None:
    result = _check("abstain_form")(reply, PROVIDED, gold=GOLD_ABSTAIN)
    assert result.passed is True, result.detail


# --- C. every fenced example in docs/eval-set-format.md loads --------------------------------

_FENCE_RE = re.compile(r"^```(\w+)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _doc_examples() -> list[tuple[str, str]]:
    examples = []
    for lang, body in _FENCE_RE.findall(FORMAT_DOC.read_text(encoding="utf-8")):
        if lang == "csv":
            suffix = ".csv"
        elif lang == "json":
            suffix = ".json" if body.lstrip().startswith("[") else ".jsonl"
        else:
            continue
        examples.append((suffix, body))
    return examples


def test_format_doc_has_an_example_per_format() -> None:
    assert sorted({suffix for suffix, _ in _doc_examples()}) == [".csv", ".json", ".jsonl"]


@pytest.mark.parametrize("index", range(3))
def test_format_doc_examples_load(index: int, tmp_path: Path) -> None:
    from groundcite.evalset import load_eval_set

    suffix, body = _doc_examples()[index]
    path = tmp_path / f"example{suffix}"
    path.write_text(body, encoding="utf-8")
    scenarios = load_eval_set(path)
    assert scenarios
    for s in scenarios:
        assert s.provided_ids, s.id


# --- D. prompt-only rows: every bracketed token is a provided ID (R12, D-033) ----------------


def _load_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> list[Any]:
    from groundcite.evalset import load_eval_set

    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return load_eval_set(path)


def test_prompt_only_ids_are_every_bracketed_token(tmp_path: Path) -> None:
    # D-033 dropped the D-022 digit rule: bracketed UI words are provided IDs too. Citing one is
    # rare, and an unbracketed mention of one is only a SOFT flag on prompt-only rows.
    prompt = (
        "References: [SM-3105] Alternator output test. [ref_q3.11] Wiper park switch.\n"
        "Press [OK] on the scan tool, open [Live Data] and read [MAP]. Battery light flickers."
    )
    (scenario,) = _load_rows(tmp_path, [{"id": "ui-1", "prompt": prompt}])
    assert scenario.provided_ids == ["SM-3105", "ref_q3.11", "OK", "Live Data", "MAP"]


def test_prompt_only_gold_may_name_any_bracketed_token(tmp_path: Path) -> None:
    row = {
        "id": "ui-2",
        "prompt": "References: [SM-3105] Alternator output test. Press [OK] to continue.",
        "gold": {"mode": "cite", "supporting_ids": ["OK"]},
    }
    (scenario,) = _load_rows(tmp_path, [row])
    assert scenario.gold["supporting_ids"] == ["OK"]


def test_prompt_only_gold_cannot_name_an_unbracketed_token(tmp_path: Path) -> None:
    from groundcite.evalset import EvalSetError

    row = {
        "id": "ui-2b",
        "prompt": "References: [SM-3105] Alternator output test. Press OK to continue.",
        "gold": {"mode": "cite", "supporting_ids": ["OK"]},
    }
    with pytest.raises(EvalSetError, match="gold.supporting_ids"):
        _load_rows(tmp_path, [row])


def test_structured_snippet_ids_without_digits_are_still_accepted(tmp_path: Path) -> None:
    # Declared snippet IDs are explicit, whatever their shape.
    row = {
        "id": "ui-3",
        "snippets": [{"id": "FUEL-A", "text": "Fuel pressure test."}],
        "prompt": "Rough idle when cold.",
    }
    (scenario,) = _load_rows(tmp_path, [row])
    assert scenario.provided_ids == ["FUEL-A"]
