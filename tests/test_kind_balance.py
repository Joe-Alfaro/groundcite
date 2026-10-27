"""Snippet kind must not be a shortcut for the correct citation (D-046, distill validation F1, F2).

These read the committed ablation, eval_own, heldout and train_pool sets as raw JSONL. Each
scenario snippet carries its library ``kind`` ("tsb" for bulletins, "manual" for service-manual
snippets); structured applicability is looked up in ``data/snippets/*.jsonl`` by snippet text.

Invariants, per set:

1. the ID scheme is drawn independently of kind;
2. among cite-gold scenarios, the gold snippet is a bulletin 35-65% of the time;
3. near-miss snippets are manual-kind 35-65% of the time;
4. no single surface feature (kind, ID-scheme family, a correction/"Repair:" line, an
   "Applies to all" opener) shifts the gold rate by more than 0.15 between its two values;
5. some cite-gold scenarios cite a vehicle-scoped bulletin, and some carry a manual near-miss
   scoped to a different engine or transmission.

F2: the R4 hint-leak screen catches the wider set of privileged-input phrases.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from distill_data import id_scheme, scenario_row

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
SNIPPET_DIR = REPO_ROOT / "data" / "snippets"

SET_NAMES = ("ablation", "eval_own", "heldout", "train_pool")

BALANCE_LOW, BALANCE_HIGH = 0.35, 0.65
MAX_SCHEME_KIND_SHARE = 0.80
# A scheme needs this many uses before its kind share is meaningful.
MIN_SCHEME_USES = 20
MAX_GOLD_RATE_GAP = 0.15
# The held-out set is small (60 rows, about 40 cite-gold), so its gold-rate gap is noisier.
HELDOUT_GOLD_RATE_GAP = 0.25

# Bulletin-style ID prefixes (TSB-#, SB #-#, B-#-#); everything else is manual-style (SM, SEC).
BULLETIN_SCHEME = re.compile(r"^(?:TSB|SB|B)[- ]")
CORRECTION_LINE = re.compile(r"\b(?:Repair|Correction|Corrective action)\s*:", re.IGNORECASE)
APPLIES_TO_ALL = re.compile(r"^\s*Applies to all\b", re.IGNORECASE)
YEAR = re.compile(r"\b(?:19|20)\d\d\b")


def _rows(name: str) -> list[dict[str, Any]]:
    path = SCENARIO_DIR / f"{name}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _library_by_text() -> dict[str, dict[str, Any]]:
    by_text = {}
    for path in sorted(SNIPPET_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                by_text[raw["text"]] = raw
    return by_text


def _cite_rows(name: str) -> list[dict[str, Any]]:
    return [row for row in _rows(name) if row["gold"]["mode"] == "cite"]


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


# --- (1) ID scheme independent of kind --------------------------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_each_kind_uses_at_least_three_id_schemes(name: str) -> None:
    schemes: dict[str, set[str]] = defaultdict(set)
    for row in _rows(name):
        for snippet in row["snippets"]:
            schemes[snippet["kind"]].add(id_scheme(snippet["id"]))
    for kind in ("tsb", "manual"):
        assert len(schemes[kind]) >= 3, f"{name}: {kind} uses only {sorted(schemes[kind])}"


@pytest.mark.parametrize("name", SET_NAMES)
def test_every_id_scheme_is_used_by_both_kinds(name: str) -> None:
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _rows(name):
        for snippet in row["snippets"]:
            kinds[id_scheme(snippet["id"])][snippet["kind"]] += 1
    single = {scheme: dict(c) for scheme, c in kinds.items() if len(c) < 2}
    assert not single, f"{name}: ID schemes used by only one kind: {single}"


@pytest.mark.parametrize("name", SET_NAMES)
def test_no_id_scheme_is_mostly_one_kind(name: str) -> None:
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _rows(name):
        for snippet in row["snippets"]:
            kinds[id_scheme(snippet["id"])][snippet["kind"]] += 1
    skewed = {
        scheme: dict(c)
        for scheme, c in kinds.items()
        if c.total() >= MIN_SCHEME_USES and max(c.values()) / c.total() > MAX_SCHEME_KIND_SHARE
    }
    assert not skewed, f"{name}: schemes > {MAX_SCHEME_KIND_SHARE:.0%} one kind: {skewed}"


# --- (2) gold kind balance and (3) near-miss kind balance --------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_gold_snippet_is_a_bulletin_35_to_65_percent_of_the_time(name: str) -> None:
    kinds = Counter()
    for row in _cite_rows(name):
        by_id = {s["id"]: s for s in row["snippets"]}
        for gold_id in row["gold"]["supporting_ids"]:
            kinds[by_id[gold_id]["kind"]] += 1
    share = _share(kinds["tsb"], kinds.total())
    assert BALANCE_LOW <= share <= BALANCE_HIGH, (
        f"{name}: gold is a bulletin {share:.1%} of the time ({dict(kinds)})"
    )


@pytest.mark.parametrize("name", SET_NAMES)
def test_near_misses_are_manual_kind_35_to_65_percent_of_the_time(name: str) -> None:
    kinds = Counter()
    for row in _rows(name):
        by_id = {s["id"]: s for s in row["snippets"]}
        for near_id in row["gold"]["near_miss_ids"]:
            kinds[by_id[near_id]["kind"]] += 1
    share = _share(kinds["manual"], kinds.total())
    assert BALANCE_LOW <= share <= BALANCE_HIGH, (
        f"{name}: near-misses are manual {share:.1%} of the time ({dict(kinds)})"
    )


# --- (4) no surface feature predicts gold ------------------------------------------------------

FEATURES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "kind is bulletin": lambda s: s["kind"] == "tsb",
    "bulletin-style ID scheme": lambda s: bool(BULLETIN_SCHEME.match(s["id"])),
    "correction/'Repair:' line": lambda s: bool(CORRECTION_LINE.search(s["text"])),
    "'Applies to all' opener": lambda s: bool(APPLIES_TO_ALL.match(s["text"])),
}


def _gold_rate_gap(name: str, feature: Callable[[dict[str, Any]], bool]) -> tuple[float, str]:
    """|gold rate with the feature - gold rate without it| over cite-gold scenarios' snippets.

    A feature that is always present or always absent can't predict gold, so its gap is 0.
    """
    gold = Counter[bool]()
    seen = Counter[bool]()
    for row in _cite_rows(name):
        supporting = set(row["gold"]["supporting_ids"])
        for snippet in row["snippets"]:
            has = feature(snippet)
            seen[has] += 1
            gold[has] += snippet["id"] in supporting
    if not seen[True] or not seen[False]:
        return 0.0, f"feature uniform ({dict(seen)})"
    with_rate, without_rate = gold[True] / seen[True], gold[False] / seen[False]
    detail = (
        f"gold rate {with_rate:.2f} with ({gold[True]}/{seen[True]}) vs "
        f"{without_rate:.2f} without ({gold[False]}/{seen[False]})"
    )
    return abs(with_rate - without_rate), detail


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("name", SET_NAMES)
def test_no_surface_feature_predicts_gold(name: str, feature: str) -> None:
    limit = HELDOUT_GOLD_RATE_GAP if name == "heldout" else MAX_GOLD_RATE_GAP
    if name == "heldout":
        print(
            f"heldout is small: gold-rate gap limit loosened to {limit} "
            f"(default {MAX_GOLD_RATE_GAP})"
        )
    gap, detail = _gold_rate_gap(name, FEATURES[feature])
    assert gap <= limit, f"{name}: {feature} predicts gold (gap {gap:.2f} > {limit}): {detail}"


# --- (5) scoped bulletin gold and scoped manual near-misses ------------------------------------

MIN_SCOPED = {"ablation": 2, "eval_own": 3, "heldout": 2, "train_pool": 10}


def _first_user_year(row: dict[str, Any]) -> int | None:
    first = next(m["content"] for m in row["messages"] if m["role"] == "user")
    match = YEAR.search(first)
    return int(match.group(0)) if match else None


def _is_scoped_bulletin_gold(row: dict[str, Any], library: dict[str, dict[str, Any]]) -> bool:
    """A gold bulletin with stated engine/year/body/transmission conditions the vehicle meets."""
    by_id = {s["id"]: s for s in row["snippets"]}
    year = _first_user_year(row)
    for gold_id in row["gold"]["supporting_ids"]:
        snippet = by_id[gold_id]
        source = library.get(snippet["text"], {})
        applies = source.get("applies") or {}
        if snippet["kind"] != "tsb" or not applies:
            continue
        years = applies.get("years")
        if years and (year is None or not years[0] <= year <= years[1]):
            continue
        return True
    return False


def _has_scoped_manual_near_miss(row: dict[str, Any], library: dict[str, dict[str, Any]]) -> bool:
    """A manual-kind near-miss scoped to an engine or transmission (one the vehicle lacks)."""
    by_id = {s["id"]: s for s in row["snippets"]}
    for near_id in row["gold"]["near_miss_ids"]:
        snippet = by_id[near_id]
        source = library.get(snippet["text"], {})
        applies = source.get("applies") or {}
        if (
            snippet["kind"] == "manual"
            and source.get("mismatch") == "vehicle"
            and (applies.get("engines") or applies.get("trans"))
        ):
            return True
    return False


@pytest.mark.parametrize("name", SET_NAMES)
def test_some_cite_gold_scenarios_cite_a_vehicle_scoped_bulletin(name: str) -> None:
    library = _library_by_text()
    hits = [row["id"] for row in _cite_rows(name) if _is_scoped_bulletin_gold(row, library)]
    assert len(hits) >= MIN_SCOPED[name], (
        f"{name}: {len(hits)} cite-gold scenarios cite a bulletin whose conditions match the "
        f"vehicle (need >= {MIN_SCOPED[name]})"
    )


@pytest.mark.parametrize("name", SET_NAMES)
def test_some_cite_gold_scenarios_have_a_scoped_manual_near_miss(name: str) -> None:
    library = _library_by_text()
    hits = [row["id"] for row in _cite_rows(name) if _has_scoped_manual_near_miss(row, library)]
    assert len(hits) >= MIN_SCOPED[name], (
        f"{name}: {len(hits)} cite-gold scenarios have a manual near-miss scoped to another "
        f"engine or transmission (need >= {MIN_SCOPED[name]})"
    )


def test_library_has_manual_near_misses_scoped_to_engine_or_transmission() -> None:
    scoped = [
        raw["id"]
        for raw in _library_by_text().values()
        if raw["kind"] == "manual"
        and raw.get("mismatch") == "vehicle"
        and ((raw.get("applies") or {}).get("engines") or (raw.get("applies") or {}).get("trans"))
    ]
    assert scoped, "no manual near-miss snippet is scoped to a different engine or transmission"


# --- (6) F2: hint-leak screen ------------------------------------------------------------------

LEAKY_REPLIES = (
    "The answer key says to inspect the hose at the intake elbow [TSB-4101].",
    "Per the ANSWER KEY, inspect the hose at the intake elbow for cracks [TSB-4101].",
    "The supporting IDs point to the hose check at the intake elbow [TSB-4101].",
    "Going by the Supporting ID list, inspect the hose at the intake elbow [TSB-4101].",
    "The note says to inspect the hose at the intake elbow for a vacuum hiss [TSB-4101].",
    "For grading purposes, inspect the hose at the intake elbow for cracks [TSB-4101].",
    "The reference answer is to inspect the hose at the intake elbow [TSB-4101].",
    "I was told to inspect the hose at the intake elbow for cracks [TSB-4101].",
    "According to the hint, inspect the hose at the intake elbow for cracks [TSB-4101].",
)

CLEAN_REPLIES = (
    "As the bulletin describes, with the engine idling, inspect the hose at the intake elbow "
    "for cracks or a vacuum hiss [TSB-4101].",
    "The service manual note covers this: with the engine idling, inspect the hose at the "
    "intake elbow for cracks or a vacuum hiss [TSB-4101].",
    "With the engine idling, inspect the hose at the intake elbow for cracks or a vacuum hiss, "
    "and note whether the idle changes when you flex it [TSB-4101].",
    "Note the idle speed first; with the engine idling, inspect the hose at the intake elbow "
    "for cracks or a vacuum hiss [TSB-4101].",
)


def _screen(reply: str) -> Any:
    from groundcite.evalset import parse_row
    from groundcite.filter import Candidate, deterministic_stage

    row = scenario_row(1)
    candidate = Candidate(
        scenario_id=row["id"],
        candidate_index=0,
        reply=reply,
        error=None,
        scenario=parse_row(row),
        row=row,
    )
    return deterministic_stage(candidate)


@pytest.mark.parametrize("reply", LEAKY_REPLIES)
def test_hint_leak_screen_rejects_privileged_phrases(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is not None and rejection.reason == HINT_LEAK, (
        f"not rejected as {HINT_LEAK}: {reply!r} -> {rejection}"
    )


@pytest.mark.parametrize("reply", CLEAN_REPLIES)
def test_hint_leak_screen_accepts_normal_replies(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is None or rejection.reason != HINT_LEAK, f"{reply!r} -> {rejection}"
