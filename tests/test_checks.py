"""R21: deterministic checks, plus reply mode inference.

Every check has the signature ``check(reply, provided_ids, gold=None) -> CheckResult``.
``CheckResult.passed`` is True, False, or None for n/a (gold-dependent check without gold, or a
check that does not apply to the reply's mode). Reply mode is structural (D-028): a reply with a
bracketed ID is in cite mode, otherwise abstain mode.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Worked examples 1-4 from the requirements file share these snippets.
PROVIDED = ["SM-2210", "TSB-0417", "SM-3105"]
GOLD_CITE = {"mode": "cite", "supporting_ids": ["TSB-0417"]}
GOLD_FUEL = {"mode": "cite", "supporting_ids": ["SM-2210"]}
GOLD_ABSTAIN: dict[str, Any] = {"mode": "abstain", "supporting_ids": []}

# Worked example 1: PASS.
EX1 = (
    "With the engine idling, inspect the PCV hose where it meets the intake elbow for cracks "
    "or a vacuum hiss [TSB-0417]."
)
# Worked example 2: FAIL (P6).
EX2 = "This is the known PCV hose issue. Replace the hose at the intake elbow [TSB-0417]."
# Worked example 3: FAIL (P2, P5).
EX3 = "Inspect the PCV hose [TSB-0417], then check fuel pressure against spec [SM-2210]."
# Worked example 4: PASS (abstain), and its two failing variants.
EX4 = (
    "None of the references you provided cover front suspension noise, so I can't point to a "
    "documented step yet. Can you share the service-manual section for the front suspension?"
)
EX4_FIX = "It's probably a sway-bar link, so replace it."
EX4_DISTRACTOR = "Test the alternator output at 2000 rpm with the loads on [SM-3105]."

FUEL_STEP = (
    "Key on, engine off, connect a gauge at the rail test port and confirm it reads 55-62 psi "
    "and holds for 5 minutes [SM-2210]."
)
REPEATED_ID = (
    "Inspect the PCV hose at the intake elbow for cracks [TSB-0417]; the bulletin [TSB-0417] "
    "names the elbow as the usual crack point."
)
UNCITED_STEP = (
    "With the engine idling, inspect the PCV hose at the intake elbow for cracks or a vacuum hiss."
)
INVENTED = "Inspect the PCV hose at the intake elbow for cracks [TSB-9999]."
REAL_PLUS_INVENTED = "Inspect the PCV hose at the intake elbow [TSB-0417] [XX-0001]."
ABSTAIN_WITH_ID = (
    "None of the references you provided cover front suspension noise [SM-3105]. Can you share "
    "the service-manual section for the front suspension?"
)
ABSTAIN_REAR = (
    "None of the provided snippets apply to a whine from the rear axle. Does the whine change "
    "with road speed or with engine speed?"
)
ABSTAIN_PULL = (
    "The provided references don't cover a pull under braking, so there is no documented step "
    "to point to. Which way does the car pull?"
)
ABSTAIN_TWO_QUESTIONS = (
    "None of the references you provided cover front suspension noise. Does it clunk over every "
    "bump? Is it louder when turning?"
)
ABSTAIN_NO_QUESTION = (
    "None of the references you provided cover front suspension noise, so I can't point to a "
    "documented step yet."
)
UNBRACKETED = "Per TSB-0417, inspect the PCV hose at the intake elbow for cracks."
PARENS = "Inspect the PCV hose at the intake elbow for cracks (TSB-0417)."
BRACKETED_AND_BARE = "Inspect the PCV hose at the intake elbow [TSB-0417], as TSB-0417 describes."
SPACED_ID = "Check the brake fluid level at the reservoir with a moisture test strip [SM 6.2.1]."
NUMBERED_LIST = "1. Inspect the PCV hose at the intake elbow [TSB-0417].\n2. Check fuel pressure."
BULLETS = "- Inspect the PCV hose at the intake elbow [TSB-0417]\n- Check fuel pressure at the rail"
ALSO_CHECK = "Inspect the PCV hose at the intake elbow [TSB-0417]. Also check the fuel pressure."
IF_FINE = (
    "Inspect the PCV hose at the intake elbow [TSB-0417]; if that's fine, check fuel pressure."
)
CONDITIONAL_FIX = (
    "Measure fuel pressure at the rail test port [SM-2210]; if it reads low, replace the pump."
)
ITS_YOUR = "It's your PCV hose at the intake elbow [TSB-0417]."
IS_BAD = "The PCV hose at the intake elbow is bad [TSB-0417]."
THINK_CLOSED = "<think>Lean code at idle, so the PCV hose fits.</think>\n" + EX1
THINK_OPEN = "<think>The user wants a part name but I should" + EX1

# (check, reply, provided_ids, gold, expected passed)
CASES: list[tuple[str, str, list[str], dict[str, Any] | None, bool | None]] = [
    # citation_format: every referenced ID is bracketed.
    ("citation_format", EX1, PROVIDED, None, True),
    ("citation_format", EX4, PROVIDED, None, True),
    ("citation_format", SPACED_ID, ["B-19-045", "SM 6.2.1"], None, True),
    ("citation_format", UNBRACKETED, PROVIDED, None, False),
    ("citation_format", PARENS, PROVIDED, None, False),
    ("citation_format", BRACKETED_AND_BARE, PROVIDED, None, False),
    # citation_count: cite mode has exactly 1 distinct cited ID, abstain mode has 0.
    ("citation_count", EX1, PROVIDED, None, True),
    ("citation_count", EX4, PROVIDED, None, True),
    ("citation_count", REPEATED_ID, PROVIDED, None, True),
    ("citation_count", EX3, PROVIDED, None, False),  # two IDs
    # Without gold, mode is structural, so these two cases need gold to be caught (D-028).
    ("citation_count", UNCITED_STEP, PROVIDED, GOLD_CITE, False),  # zero IDs on a cite case
    ("citation_count", ABSTAIN_WITH_ID, PROVIDED, GOLD_ABSTAIN, False),  # ID on an abstain case
    ("citation_count", REAL_PLUS_INVENTED, PROVIDED, None, False),
    # cited_id_provided: no invented IDs (vacuously true with no citation).
    ("cited_id_provided", EX1, PROVIDED, None, True),
    ("cited_id_provided", EX3, PROVIDED, None, True),
    ("cited_id_provided", EX4, PROVIDED, None, True),
    ("cited_id_provided", INVENTED, PROVIDED, None, False),
    ("cited_id_provided", REAL_PLUS_INVENTED, PROVIDED, None, False),
    # cited_id_supports (needs gold): every cited ID is in gold.supporting_ids.
    ("cited_id_supports", EX1, PROVIDED, GOLD_CITE, True),
    ("cited_id_supports", FUEL_STEP, PROVIDED, GOLD_FUEL, True),
    ("cited_id_supports", EX4_DISTRACTOR, PROVIDED, GOLD_CITE, False),  # distractor
    ("cited_id_supports", EX4_DISTRACTOR, PROVIDED, GOLD_ABSTAIN, False),  # P4 on example 4
    ("cited_id_supports", EX3, PROVIDED, GOLD_CITE, False),  # one of two IDs unsupported
    ("cited_id_supports", EX1, PROVIDED, None, None),  # no gold
    ("cited_id_supports", UNCITED_STEP, PROVIDED, GOLD_CITE, None),  # nothing cited
    # mode_matches_gold (needs gold): structural reply mode equals gold.mode.
    ("mode_matches_gold", EX1, PROVIDED, GOLD_CITE, True),
    ("mode_matches_gold", EX4, PROVIDED, GOLD_ABSTAIN, True),
    # No citation, so abstain mode; the P6 fix is no_fix's and abstain_form's to catch.
    ("mode_matches_gold", EX4_FIX, PROVIDED, GOLD_ABSTAIN, True),
    ("mode_matches_gold", ABSTAIN_WITH_ID, PROVIDED, GOLD_ABSTAIN, False),  # cites, so cite mode
    ("mode_matches_gold", EX4_DISTRACTOR, PROVIDED, GOLD_ABSTAIN, False),  # P1 on example 4
    ("mode_matches_gold", EX4, PROVIDED, GOLD_CITE, False),
    ("mode_matches_gold", EX1, PROVIDED, None, None),
    # one_step: no list, no sequencing markers.
    ("one_step", EX1, PROVIDED, None, True),
    ("one_step", EX4, PROVIDED, None, True),
    ("one_step", FUEL_STEP, PROVIDED, None, True),  # expected reading is fine
    ("one_step", EX3, PROVIDED, None, False),  # "then"
    ("one_step", NUMBERED_LIST, PROVIDED, None, False),
    ("one_step", BULLETS, PROVIDED, None, False),
    ("one_step", ALSO_CHECK, PROVIDED, None, False),
    ("one_step", IF_FINE, PROVIDED, None, False),
    # no_fix: no repair markers.
    ("no_fix", EX1, PROVIDED, None, True),
    ("no_fix", EX4, PROVIDED, None, True),
    ("no_fix", FUEL_STEP, PROVIDED, None, True),
    ("no_fix", EX2, PROVIDED, None, False),
    ("no_fix", EX4_FIX, PROVIDED, None, False),
    ("no_fix", CONDITIONAL_FIX, PROVIDED, None, False),
    ("no_fix", ITS_YOUR, PROVIDED, None, False),
    ("no_fix", IS_BAD, PROVIDED, None, False),
    # abstain_form: zero citations and exactly one "?". Applies when the reply is in abstain
    # mode or gold says abstain; otherwise n/a.
    ("abstain_form", EX4, PROVIDED, None, True),
    ("abstain_form", EX4, PROVIDED, GOLD_ABSTAIN, True),
    ("abstain_form", ABSTAIN_REAR, PROVIDED, None, True),
    ("abstain_form", ABSTAIN_WITH_ID, PROVIDED, GOLD_ABSTAIN, False),  # ID on an abstain case
    ("abstain_form", ABSTAIN_WITH_ID, PROVIDED, None, None),  # cites, so cite mode: n/a
    ("abstain_form", ABSTAIN_TWO_QUESTIONS, PROVIDED, None, False),
    ("abstain_form", ABSTAIN_NO_QUESTION, PROVIDED, None, False),
    ("abstain_form", EX4_FIX, PROVIDED, GOLD_ABSTAIN, False),
    ("abstain_form", EX1, PROVIDED, None, None),
    ("abstain_form", EX1, PROVIDED, GOLD_CITE, None),
    # no_think: no <think> block left in the reply.
    ("no_think", EX1, PROVIDED, None, True),
    ("no_think", EX4, PROVIDED, None, True),
    ("no_think", THINK_CLOSED, PROVIDED, None, False),
    ("no_think", THINK_OPEN, PROVIDED, None, False),
]

CHECK_NAMES = [
    "citation_format",
    "citation_count",
    "cited_id_provided",
    "cited_id_supports",
    "mode_matches_gold",
    "one_step",
    "no_fix",
    "abstain_form",
    "no_think",
]
GOLD_DEPENDENT = ["cited_id_supports", "mode_matches_gold"]


def _check(name: str) -> Any:
    from groundcite import checks

    return getattr(checks, name)


@pytest.mark.parametrize(
    ("name", "reply", "provided", "gold", "expected"),
    CASES,
    ids=[f"{c[0]}-{i}-{c[4]}" for i, c in enumerate(CASES)],
)
def test_check_fixture(
    name: str,
    reply: str,
    provided: list[str],
    gold: dict[str, Any] | None,
    expected: bool | None,
) -> None:
    result = _check(name)(reply, provided, gold=gold)
    assert result.name == name
    assert result.passed is expected, result.detail
    assert isinstance(result.detail, str)
    if expected is False:
        assert result.detail


def test_every_check_has_two_passing_and_two_failing_fixtures() -> None:
    passing = Counter(c[0] for c in CASES if c[4] is True)
    failing = Counter(c[0] for c in CASES if c[4] is False)
    for name in CHECK_NAMES:
        assert passing[name] >= 2, name
        assert failing[name] >= 2, name


@pytest.mark.parametrize("name", GOLD_DEPENDENT)
def test_gold_dependent_checks_are_na_without_gold(name: str) -> None:
    for reply in (EX1, EX2, EX3, EX4, EX4_FIX):
        assert _check(name)(reply, PROVIDED).passed is None


@pytest.mark.parametrize("name", CHECK_NAMES)
def test_checks_are_pure(name: str) -> None:
    provided = list(PROVIDED)
    gold = {"mode": "cite", "supporting_ids": ["TSB-0417"]}
    first = _check(name)(EX3, provided, gold=gold)
    second = _check(name)(EX3, provided, gold=gold)
    assert first == second
    assert provided == PROVIDED
    assert gold == {"mode": "cite", "supporting_ids": ["TSB-0417"]}


def test_check_registry_lists_all_checks_in_order() -> None:
    from groundcite.checks import CHECKS

    assert list(CHECKS) == CHECK_NAMES
    for name, fn in CHECKS.items():
        assert fn(EX1, PROVIDED, gold=GOLD_CITE).name == name


def test_lexicon_version_is_declared() -> None:
    from groundcite.checks import LEXICON_VERSION

    assert isinstance(LEXICON_VERSION, str)
    assert LEXICON_VERSION.strip()


# --- mode inference -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "mode"),
    [
        (EX1, "cite"),
        (EX2, "cite"),
        (EX3, "cite"),
        (ABSTAIN_WITH_ID, "cite"),  # cites, whatever the wording (D-028)
        (UNCITED_STEP, "abstain"),  # no bracketed ID; gold catches a missing citation
        (EX4_FIX, "abstain"),
        (UNBRACKETED, "abstain"),
        (EX4, "abstain"),
        (ABSTAIN_REAR, "abstain"),
        (ABSTAIN_PULL, "abstain"),
        (ABSTAIN_NO_QUESTION, "abstain"),
    ],
)
def test_infer_mode(reply: str, mode: str) -> None:
    from groundcite.checks import infer_mode

    assert infer_mode(reply) == mode


# --- checks run on every R12 fixture set ----------------------------------------------------


@pytest.mark.parametrize(
    "fixture", ["our_format.jsonl", "prompt_only.csv", "prompt_only.json", "third_party.jsonl"]
)
def test_checks_run_on_every_fixture_set(fixture: str) -> None:
    from groundcite.checks import CHECKS
    from groundcite.evalset import load_eval_set

    scenarios = load_eval_set(FIXTURES / fixture)
    assert scenarios
    for s in scenarios:
        reply = f"Inspect the component the first reference describes [{s.provided_ids[0]}]."
        results = {name: fn(reply, s.provided_ids, gold=s.gold) for name, fn in CHECKS.items()}
        assert results["cited_id_provided"].passed is True
        assert results["citation_format"].passed is True
        # With gold, citation_count expects the count gold.mode implies (D-021): one ID on an
        # abstain-gold row is a failure.
        gold_mode = s.gold.get("mode") if s.gold is not None else None
        assert results["citation_count"].passed is (gold_mode != "abstain")
        if s.gold is None:
            for name in GOLD_DEPENDENT:
                assert results[name].passed is None
        else:
            for name in GOLD_DEPENDENT:
                assert results[name].passed in (True, False)
