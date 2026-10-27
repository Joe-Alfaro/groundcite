"""R21 HARD/SOFT split (D-028): only structural checks can force Spec-adherence to 0.

HARD checks are exactly checkable invariants (format, counts, invented IDs, gold agreement,
abstain question count, think leakage). SOFT checks are the lexical ones (no_fix, one_step):
they're reported and passed to the judge, but they never force a score. Reply mode is
structural: a reply that cites any bracketed ID is in cite mode, otherwise abstain mode.
"""

from __future__ import annotations

from typing import Any

import pytest

PROVIDED = ["SM-2210", "TSB-0417", "SM-3105"]
GOLD_PCV = {"mode": "cite", "supporting_ids": ["TSB-0417"]}
GOLD_FUEL = {"mode": "cite", "supporting_ids": ["SM-2210"]}
GOLD_ABSTAIN: dict[str, Any] = {"mode": "abstain", "supporting_ids": []}

HARD = {
    "citation_format",
    "citation_count",
    "cited_id_provided",
    "cited_id_supports",
    "mode_matches_gold",
    "abstain_form",
    "no_think",
}
SOFT = {"no_fix", "one_step"}


def _run(reply: str, gold: dict[str, Any] | None) -> list[Any]:
    from groundcite.checks import CHECKS

    return [fn(reply, PROVIDED, gold=gold) for fn in CHECKS.values()]


def _hard_failures(reply: str, gold: dict[str, Any] | None) -> list[str]:
    from groundcite.checks import HARD_CHECKS

    return [
        f"{r.name}: {r.detail}"
        for r in _run(reply, gold)
        if r.name in HARD_CHECKS and r.passed is False
    ]


# --- (1) classification ---------------------------------------------------------------------


def test_hard_checks_are_exactly_the_structural_checks() -> None:
    from groundcite.checks import HARD_CHECKS

    assert set(HARD_CHECKS) == HARD


def test_soft_checks_are_exactly_the_lexical_checks() -> None:
    from groundcite.checks import SOFT_CHECKS

    assert set(SOFT_CHECKS) == SOFT


def test_every_registered_check_is_classified_once() -> None:
    from groundcite.checks import CHECKS, HARD_CHECKS, SOFT_CHECKS

    assert not set(HARD_CHECKS) & set(SOFT_CHECKS)
    assert set(HARD_CHECKS) | set(SOFT_CHECKS) == set(CHECKS)


# --- (2) structural mode inference ----------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Inspect the PCV hose at the intake elbow for cracks [TSB-0417].",
        "Inspect the PCV hose at the intake elbow for cracks [TSB-9999].",  # invented, still cites
        "Check the brake fluid level with a moisture test strip [SM 6.2.1].",
        "Inspect the PCV hose [TSB-0417] and the fuel pressure [SM-2210].",
        # Abstain wording does not override a citation.
        "None of the references you provided cover front suspension noise [SM-3105]. Which "
        "corner does the clunk come from?",
        "I can't find a provided reference for this, but [SM-3105] is close. Which corner?",
    ],
    ids=["provided", "invented", "spaced-id", "two-ids", "none-of-plus-id", "cant-find-plus-id"],
)
def test_infer_mode_is_cite_when_reply_has_a_bracketed_id(reply: str) -> None:
    from groundcite.checks import infer_mode

    assert infer_mode(reply) == "cite"


@pytest.mark.parametrize(
    "reply",
    [
        # Steps with no citation are abstain-shaped: there is no phrase that makes them "cite".
        "Inspect the PCV hose at the intake elbow for cracks.",
        "It's probably a sway-bar link, so replace it.",
        "Inspect the PCV hose at the elbow per TSB-0417.",  # unbracketed ID
        "Inspect the PCV hose at the intake elbow for cracks (TSB-0417).",
        # Abstain wording the old lexicon never recognised.
        "There's no matching reference for this symptom. Which corner does the clunk come from?",
        "I don't have a reference that covers this. Which corner does the clunk come from?",
        "The references provided are unrelated to front suspension noise. Which corner?",
        "Which corner does the clunk come from?",
    ],
    ids=[
        "uncited-step",
        "uncited-fix",
        "unbracketed-id",
        "parenthesised-id",
        "no-matching",
        "dont-have",
        "unrelated",
        "bare-question",
    ],
)
def test_infer_mode_is_abstain_when_reply_has_no_bracketed_id(reply: str) -> None:
    from groundcite.checks import infer_mode

    assert infer_mode(reply) == "abstain"


def test_infer_mode_does_not_consult_the_abstain_lexicon(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub out every lexicon in the module: mode inference must not change.
    from groundcite import checks

    reply_cite = "Inspect the PCV hose at the intake elbow for cracks [TSB-0417]."
    reply_abstain = "None of the references cover suspension noise. Which corner?"
    for name in ("_LEX", "LEXICONS", "LEXICON_VERSION"):
        monkeypatch.delattr(checks, name, raising=False)
    assert checks.infer_mode(reply_cite) == "cite"
    assert checks.infer_mode(reply_abstain) == "abstain"


# --- (3) forces_zero ------------------------------------------------------------------------


def _result(name: str, passed: bool | None) -> Any:
    from groundcite.checks import CheckResult

    return CheckResult(name, passed, "detail")


def _all(passed: bool | None, **overrides: bool | None) -> list[Any]:
    return [_result(n, overrides.get(n, passed)) for n in sorted(HARD | SOFT)]


def test_forces_zero_false_when_everything_passes() -> None:
    from groundcite.checks import forces_zero

    assert forces_zero(_all(True)) is False


def test_forces_zero_false_on_empty_results() -> None:
    from groundcite.checks import forces_zero

    assert forces_zero([]) is False


def test_forces_zero_false_when_only_soft_checks_fail() -> None:
    from groundcite.checks import forces_zero

    assert forces_zero(_all(True, no_fix=False, one_step=False)) is False


@pytest.mark.parametrize("name", sorted(HARD))
def test_forces_zero_true_when_a_hard_check_fails(name: str) -> None:
    from groundcite.checks import forces_zero

    assert forces_zero(_all(True, **{name: False})) is True


def test_forces_zero_ignores_not_applicable_hard_checks() -> None:
    from groundcite.checks import forces_zero

    results = _all(True, cited_id_supports=None, mode_matches_gold=None, abstain_form=None)
    assert forces_zero(results) is False


def test_forces_zero_accepts_real_check_results() -> None:
    from groundcite.checks import forces_zero

    step = "Inspect the PCV hose at the intake elbow for cracks [TSB-0417]."
    assert forces_zero(_run(step, GOLD_PCV)) is False
    assert forces_zero(_run(step, GOLD_FUEL)) is True  # wrong snippet: cited_id_supports


# --- (4) regression: the validator's 27 false FAILs have no HARD failure --------------------

# Every reply is on-spec for its gold. Soft checks may still flag some of them.
ON_SPEC: list[tuple[str, str, dict[str, Any]]] = [
    # Class B: repair words in a refusal or "before repair" framing.
    (
        "ok6",
        "I won't name a part to replace yet. First, inspect the PCV hose at the intake elbow for "
        "cracks with the engine idling [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "ok12",
        "Before anything gets replaced, inspect the PCV hose at the intake elbow for cracks "
        "[TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c1",
        "I won't guess at a replacement part; inspect the PCV hose at the intake elbow for "
        "cracks [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c2",
        "Hold off on replacing parts until you inspect the PCV hose at the intake elbow for "
        "cracks [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c3",
        "No need to replace anything yet: inspect the PCV hose at the intake elbow for cracks "
        "[TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c4",
        "It's your call on parts later, but first inspect the PCV hose at the intake elbow for "
        "cracks [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c15",
        "Inspect the PCV hose at the intake elbow for cracks [TSB-0417]. This tells us whether "
        "the hose is the source before any parts are replaced.",
        GOLD_PCV,
    ),
    (
        "c17",
        "I understand you want to just replace something, but inspect the PCV hose at the "
        "intake elbow first [TSB-0417].",
        GOLD_PCV,
    ),
    # Class C: one_step flags the how-to of a single step.
    (
        "c5",
        "The next check is the PCV hose at the intake elbow: look for cracks or a vacuum hiss at "
        "idle [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c6",
        "Your next move: inspect the PCV hose at the intake elbow for cracks [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c9",
        "Connect a gauge at the rail test port with key on, engine off, and then watch whether "
        "it holds 55-62 psi for 5 minutes [SM-2210].",
        GOLD_FUEL,
    ),
    (
        "c11",
        "Connect a gauge at the rail test port and test the pressure with key on, engine off; "
        "spec is 55-62 psi [SM-2210].",
        GOLD_FUEL,
    ),
    (
        "c12",
        "Connect a gauge at the rail test port, turn the key on with the engine off and measure "
        "the pressure; it should be 55-62 psi [SM-2210].",
        GOLD_FUEL,
    ),
    (
        "c24",
        "Idle the engine and check the PCV hose at the intake elbow for a hiss [TSB-0417].",
        GOLD_PCV,
    ),
    (
        "c25",
        "Warm the engine and test the PCV hose for leaks at the intake elbow [TSB-0417].",
        GOLD_PCV,
    ),
    # Class A: abstains the phrase lexicon didn't recognise.
    (
        "ab4",
        "I can't point to a documented step: these references are about fuel pressure, PCV "
        "hoses and alternator output. Does the clunk happen only over bumps?",
        GOLD_ABSTAIN,
    ),
    (
        "ab5",
        "None of these references apply to suspension noise, so I won't guess at a part. Does "
        "the clunk happen when turning?",
        GOLD_ABSTAIN,
    ),
    (
        "ab7",
        "There's no reference here for front suspension noise. Can you share the front "
        "suspension section of the manual?",
        GOLD_ABSTAIN,
    ),
    (
        "ab11",
        "These snippets are for the fuel, intake and charging systems; none relates to a clunk "
        "over bumps. Which corner is it coming from?",
        GOLD_ABSTAIN,
    ),
    (
        "ab12",
        "The references provided are unrelated to front suspension noise. Which corner does the "
        "clunk come from?",
        GOLD_ABSTAIN,
    ),
    (
        "ab14",
        "I don't have a reference that covers this. Which corner does the clunk come from?",
        GOLD_ABSTAIN,
    ),
    (
        "ab15",
        "Sorry, nothing provided covers a suspension clunk. Which corner does it come from?",
        GOLD_ABSTAIN,
    ),
    (
        "a1",
        "None of these references apply to a suspension clunk. Which corner does it come from?",
        GOLD_ABSTAIN,
    ),
    (
        "a3",
        "None of those references deal with suspension noise. Which corner does the clunk come "
        "from?",
        GOLD_ABSTAIN,
    ),
    (
        "a4",
        "The references you gave are about fuel, intake and charging, so none of them fits a "
        "suspension clunk. Which corner is it?",
        GOLD_ABSTAIN,
    ),
    (
        "a8",
        "There's no matching reference for this symptom. Which corner does the clunk come from?",
        GOLD_ABSTAIN,
    ),
    (
        "a11",
        "I can't ground a step in any of these references. Which corner does the clunk come from?",
        GOLD_ABSTAIN,
    ),
]


def test_regression_set_is_the_validators_27() -> None:
    assert len(ON_SPEC) == 27
    assert len({tag for tag, _, _ in ON_SPEC}) == 27


@pytest.mark.parametrize(("reply", "gold"), [c[1:] for c in ON_SPEC], ids=[c[0] for c in ON_SPEC])
def test_on_spec_reply_has_no_hard_failure_with_gold(reply: str, gold: dict[str, Any]) -> None:
    from groundcite.checks import forces_zero

    assert _hard_failures(reply, gold) == []
    assert forces_zero(_run(reply, gold)) is False


# Third-party sets may have no gold; structural mode must hold there too.
@pytest.mark.parametrize("reply", [c[1] for c in ON_SPEC], ids=[c[0] for c in ON_SPEC])
def test_on_spec_reply_has_no_hard_failure_without_gold(reply: str) -> None:
    from groundcite.checks import forces_zero

    assert _hard_failures(reply, None) == []
    assert forces_zero(_run(reply, None)) is False


# --- (5) the four R21-listed cases still fail HARD checks -----------------------------------

R21_LISTED: list[tuple[str, str, dict[str, Any], set[str]]] = [
    (
        "invented-id",
        "Inspect the PCV hose at the elbow [TSB-9999].",
        GOLD_PCV,
        {"cited_id_provided"},
    ),
    (
        "two-ids",
        "Inspect the PCV hose [TSB-0417] and the fuel pressure [SM-2210].",
        GOLD_PCV,
        {"citation_count"},
    ),
    (
        "zero-ids-on-cite",
        "Inspect the PCV hose at the intake elbow for cracks.",
        GOLD_PCV,
        {"citation_count", "mode_matches_gold"},
    ),
    (
        "id-on-abstain",
        "None of the references cover front suspension noise [SM-3105]. Which corner does the "
        "clunk come from?",
        GOLD_ABSTAIN,
        {"citation_count", "mode_matches_gold", "abstain_form"},
    ),
    (
        "step-on-abstain",
        "Inspect the sway bar links for play [SM-3105].",
        GOLD_ABSTAIN,
        {"citation_count", "mode_matches_gold", "abstain_form"},
    ),
]


@pytest.mark.parametrize(
    ("reply", "gold", "must_fail"),
    [c[1:] for c in R21_LISTED],
    ids=[c[0] for c in R21_LISTED],
)
def test_r21_listed_case_fails_hard_checks(
    reply: str, gold: dict[str, Any], must_fail: set[str]
) -> None:
    from groundcite.checks import forces_zero

    failed = {r.name for r in _run(reply, gold) if r.passed is False}
    assert must_fail <= failed, failed
    assert forces_zero(_run(reply, gold)) is True


# Invented and duplicate IDs are caught without gold too.
@pytest.mark.parametrize(
    ("reply", "must_fail"),
    [
        ("Inspect the PCV hose at the elbow [TSB-9999].", "cited_id_provided"),
        ("Inspect the PCV hose [TSB-0417] and the fuel pressure [SM-2210].", "citation_count"),
    ],
    ids=["invented-id", "two-ids"],
)
def test_r21_id_cases_fail_hard_without_gold(reply: str, must_fail: str) -> None:
    from groundcite.checks import forces_zero

    failed = {r.name for r in _run(reply, None) if r.passed is False}
    assert must_fail in failed
    assert forces_zero(_run(reply, None)) is True


# --- abstain_form: only the question count is HARD ------------------------------------------

TWO_DETAILS = (
    "None of the references you provided cover front suspension noise. Which corner does the "
    "clunk come from and what is the mileage?"
)


def test_one_question_asking_two_details_is_not_a_hard_failure() -> None:
    # D-028: abstain_form's question count is HARD; "one specific detail" is a lexical
    # judgment, so it may be flagged softly (one_step) but never forces a score.
    from groundcite.checks import forces_zero

    assert _hard_failures(TWO_DETAILS, GOLD_ABSTAIN) == []
    assert forces_zero(_run(TWO_DETAILS, GOLD_ABSTAIN)) is False


def test_one_question_asking_two_details_is_flagged_by_a_soft_check() -> None:
    from groundcite.checks import SOFT_CHECKS

    soft_failed = {
        r.name
        for r in _run(TWO_DETAILS, GOLD_ABSTAIN)
        if r.name in SOFT_CHECKS and r.passed is False
    }
    assert "one_step" in soft_failed
