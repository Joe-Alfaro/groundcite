"""Twin priors, titles and unstated facts (round 4, kind-balance validation 3).

Validation 3 found that twins still leak gold without reading the complaint:

1. Year-range prior. Vehicle twins sat at the catalog's year tails while gold ranges spanned the
   middle, so "of two same-titled snippets, cite the one covering the median model year" picked
   the gold on 0.62-0.65 of distractors in every set. Between the two members of a twin pair
   (gold plus its twin), the member whose stated range covers the catalog's median model year
   must be gold at a rate of 0.40-0.60. The generic probe in ``test_twins.py`` now carries
   year-range, duplicate-title and pair features; here it is also scored fit on train_pool and
   pooled over the eval sets, and the two-cue rule is a gated heuristic.
2. Condition-twin titles. Eight condition twins kept the original title, which states the
   condition the scope edit removed ("Misfire under load ..." over text that says "only at
   idle"). No near-miss title may contain a condition word its scope edits removed from the
   text, so titles are either edited with the scope or carry no condition wording.
3. Unstated facts. Transmission-scoped twins decided gold, but no scenario states the
   transmission. Whenever a provided snippet is transmission-scoped, the vehicle description
   (first user message) states the vehicle's transmission. More generally, every applicability
   field and required condition of a gold or near-miss snippet is stated by the scenario.
4. Pair present means cite. No abstain row held a twin pair, so P(cite | pair) was 0.97-1.00.
   At least 15% of abstain scenarios contain a twin pair with both members out of scope, and
   P(cite | pair present) is at most 0.75.
5. Hint screen. The 10 paraphrased leaks validation 3 found are rejected, and "hint", "gold" and
   "the note" are rejected only in their reference senses.

``SETS`` lists the sets these gates cover. The ablation set may become a separately frozen set;
the PM can drop it from ``SETS`` then.
"""

from __future__ import annotations

import functools
import re
from typing import Any

import pytest
from test_kind_balance import _first_user_year, _rows, _screen
from test_scenario_coherence import _vehicle_of
from test_shortcuts import EPS, MAX_HEURISTIC_ACCURACY, _distractor_rows, _heuristic_accuracy
from test_twins import (
    CATALOG_MEDIAN_YEAR,
    MAX_PROBE_ACCURACY,
    _allowed,
    _complaint_facts,
    _complaint_index,
    _edits,
    _fails,
    _families,
    _library,
    _tokens,
    _toml,
    contains_median_year,
    pair_partners,
    probe_transfer_accuracy,
)

SETS = ("ablation", "eval_own", "heldout", "train_pool")
EVAL_SETS = tuple(name for name in SETS if name != "train_pool")

PRIOR_LOW, PRIOR_HIGH = 0.40, 0.60
MIN_ABSTAIN_PAIR_SHARE = 0.15
MAX_CITE_GIVEN_PAIR = 0.75

Row = dict[str, Any]
Raw = dict[str, Any]


@functools.cache
def _library_cached() -> dict[str, Raw]:
    return _library()


@functools.cache
def _by_text() -> dict[str, Raw]:
    return {raw["text"]: raw for raw in _library_cached().values()}


@functools.cache
def _families_cached() -> dict[str, dict[str, Any]]:
    return _families()


@functools.cache
def _vehicles() -> list[dict[str, Any]]:
    return list(_toml()["vehicle"])


def _source(snippet: dict[str, Any]) -> Raw | None:
    return _by_text().get(snippet["text"])


def _twin_pairs(row: Row) -> list[tuple[int, int]]:
    """(twin_of index, near-miss index) for every near-miss whose twin is also provided."""
    index = {s["text"]: i for i, s in enumerate(row["snippets"])}
    library = _library_cached()
    pairs = []
    for i, snippet in enumerate(row["snippets"]):
        raw = _source(snippet)
        if raw and raw.get("twin_of") in library:
            j = index.get(library[raw["twin_of"]]["text"])
            if j is not None:
                pairs.append((j, i))
    return pairs


# --- (1) year-range prior within a twin pair ---------------------------------------------------


def _year_prior(name: str) -> tuple[int, int, list[str]]:
    """Gold twin pairs where exactly one member covers the median year: (gold covers, total)."""
    hits = total = 0
    seen = []
    for row in _rows(name):
        gold = set(row["gold"]["supporting_ids"])
        for a, b in _twin_pairs(row):
            pair = [row["snippets"][a], row["snippets"][b]]
            golds = [s for s in pair if s["id"] in gold]
            covers = [s for s in pair if contains_median_year(s)]
            if len(golds) != 1 or len(covers) != 1:
                continue
            total += 1
            hits += covers[0] is golds[0]
            seen.append(f"{row['id']}:{'gold' if covers[0] is golds[0] else 'twin'}")
    return hits, total, seen


def test_catalog_median_year_is_inside_the_catalog() -> None:
    years = [y for v in _vehicles() for y in v["years"]]
    assert min(years) <= CATALOG_MEDIAN_YEAR <= max(years)


@pytest.mark.parametrize("name", SETS)
def test_member_covering_the_median_year_is_gold_about_half_the_time(name: str) -> None:
    hits, total, seen = _year_prior(name)
    rate = hits / total if total else 0.5
    print(f"{name}: median-year member is gold in {hits}/{total} informative twin pairs")
    assert PRIOR_LOW - EPS <= rate <= PRIOR_HIGH + EPS, (
        f"{name}: of {total} gold twin pairs where one member covers {CATALOG_MEDIAN_YEAR}, the "
        f"member covering it is gold {rate:.0%} of the time (allowed {PRIOR_LOW:.0%}-"
        f"{PRIOR_HIGH:.0%}): {seen[:12]}"
    )


def _prefer_median_in_pair(row: Row) -> list[int]:
    """Validation 3's complaint-free rule: among paired snippets, cite the one covering the
    catalog median year; with no pair, any snippet."""
    paired = [i for i in range(len(row["snippets"])) if pair_partners(row, i)]
    covering = [i for i in paired if contains_median_year(row["snippets"][i])]
    return covering or paired


@pytest.mark.parametrize("name", SETS)
def test_median_year_within_a_pair_does_not_solve_distractors(name: str) -> None:
    rows = _distractor_rows(name)
    accuracy = _heuristic_accuracy(rows, _prefer_median_in_pair)
    print(f"{name}: cite the paired snippet covering {CATALOG_MEDIAN_YEAR}: {accuracy:.2f}")
    assert accuracy <= MAX_HEURISTIC_ACCURACY + EPS, (
        f"{name}: 'cite the paired snippet covering {CATALOG_MEDIAN_YEAR}' picks the gold on "
        f"{accuracy:.0%} of {len(rows)} distractor scenarios (limit {MAX_HEURISTIC_ACCURACY:.0%})"
    )


def test_surface_probe_fit_on_train_pool_does_not_solve_pooled_eval_distractors() -> None:
    """Validation 3 scored 0.72 here: the per-set rows are few, so pool the eval sets too."""
    rows = [row for name in EVAL_SETS if name != "ablation" for row in _distractor_rows(name)]
    accuracy = probe_transfer_accuracy(_distractor_rows("train_pool"), rows)
    print(f"pooled eval: surface probe fit on train_pool: {accuracy:.2f} on {len(rows)}")
    assert accuracy <= MAX_PROBE_ACCURACY + EPS, (
        f"a surface probe fit on train_pool picks the gold on {accuracy:.0%} of {len(rows)} "
        f"pooled eval distractor scenarios (limit {MAX_PROBE_ACCURACY:.0%})"
    )


# --- (2) near-miss titles don't contradict their text ------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it its of on or the to when with".split()
)


def _removed_condition_words(raw: Raw) -> set[str]:
    """Words the scope edits removed that the near-miss text no longer contains anywhere."""
    removed: set[str] = set()
    for old, new in _edits(raw):
        removed |= set(_tokens(old)) - set(_tokens(new))
    left = set(_tokens(raw["text"]))
    return {w for w in removed - left if w not in _STOPWORDS and not w[0].isdigit()}


def _near_misses() -> list[Raw]:
    return [raw for raw in _library_cached().values() if raw.get("mismatch")]


@pytest.mark.parametrize("raw", _near_misses(), ids=lambda raw: raw["id"])
def test_near_miss_title_keeps_no_scope_word_the_edit_removed(raw: Raw) -> None:
    stale = sorted(set(_tokens(raw["title"])) & _removed_condition_words(raw))
    assert not stale, (
        f"{raw['id']}: title {raw['title']!r} still says {stale}, which its scope edits "
        f"{_edits(raw)} removed from the text; edit the title with the scope, or keep "
        "condition wording out of titles"
    )


def test_title_check_catches_a_stale_condition_title() -> None:
    raw = {
        "title": "Misfire under load from a fouled plug",
        "text": "A single-cylinder misfire only at idle can come from a fouled plug.",
        "scope_edits": [["misfire under load", "misfire only at idle"]],
    }
    assert set(_tokens(raw["title"])) & _removed_condition_words(raw) == {"under", "load"}


# --- (3) every fact that decides gold is stated -------------------------------------------------

# How a vehicle description states each catalog transmission, and what would contradict it
# (a CVT is also an automatic, so "automatic" doesn't contradict it).
TRANS_STATED = {
    "automatic": re.compile(r"\bautomatic\b|\bauto (?:trans(?:mission)?|box)\b", re.IGNORECASE),
    "CVT": re.compile(r"\bCVT\b|\bcontinuously variable\b", re.IGNORECASE),
    "manual": re.compile(r"\bmanual\b|\bstick[- ]?shift\b|\b\d-speed stick\b", re.IGNORECASE),
}
TRANS_CONTRADICTS = {
    "automatic": ("CVT", "manual"),
    "CVT": ("manual",),
    "manual": ("automatic", "CVT"),
}


def _first_user(row: Row) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def _trans_problem(row: Row, vehicle: dict[str, Any] | None) -> str | None:
    if vehicle is None:
        return "no catalog vehicle in the first message"
    first = _first_user(row)
    trans = vehicle["trans"]
    if not TRANS_STATED[trans].search(first):
        return f"{trans} {vehicle['body']} never states its transmission"
    wrong = [t for t in TRANS_CONTRADICTS[trans] if TRANS_STATED[t].search(first)]
    return f"{trans} {vehicle['body']} also says {wrong}" if wrong else None


def _trans_scoped(snippet: dict[str, Any]) -> bool:
    raw = _source(snippet)
    return bool(raw and (raw.get("applies") or {}).get("trans"))


@pytest.mark.parametrize("name", SETS)
def test_transmission_is_stated_whenever_a_snippet_is_transmission_scoped(name: str) -> None:
    bad = {}
    for row in _rows(name):
        scoped = [s["id"] for s in row["snippets"] if _trans_scoped(s)]
        if scoped and (problem := _trans_problem(row, _vehicle_of(row, _vehicles()))):
            bad[row["id"]] = f"{problem} (scoped: {scoped})"
    assert not bad, (
        f"{name}: {len(bad)} scenarios provide a transmission-scoped snippet without stating "
        f"the transmission, e.g. {dict(list(bad.items())[:8])}"
    )


def test_transmission_check_reads_the_vehicle_description() -> None:
    row = {"messages": [{"role": "user", "content": "2015 hatchback, 1.6L, 5-speed manual."}]}
    assert _trans_problem(row, {"trans": "manual", "body": "hatchback"}) is None
    row = {"messages": [{"role": "user", "content": "2015 compact sedan, 1.8L, 90k miles."}]}
    assert _trans_problem(row, {"trans": "CVT", "body": "compact sedan"}) is not None


def _unstated(row: Row, raw: Raw, vehicle: dict[str, Any], year: int | None) -> list[str]:
    """Applicability fields and required conditions of ``raw`` the scenario doesn't state."""
    first = _first_user(row)
    missing = []
    applies = raw.get("applies") or {}
    if applies.get("years") and year is None:
        missing.append("model year")
    if applies.get("bodies") and vehicle["body"] not in first:
        missing.append("body")
    if applies.get("engines") and not any(n in first for n in vehicle["names"]):
        missing.append("engine")
    if applies.get("trans") and _trans_problem(row, vehicle):
        missing.append("transmission")
    requires = raw.get("requires") or {}
    if requires:
        family = _families_cached()[raw["family"]]
        index = _complaint_index(row, family)
        facts = _complaint_facts(family)
        stated = facts[index] if index is not None and index < len(facts) else {}
        missing += [f"condition {k!r}" for k in requires if k not in stated]
    return missing


@pytest.mark.parametrize("name", SETS)
def test_every_fact_that_decides_gold_is_stated(name: str) -> None:
    """Gold and near-miss scope and conditions are all stated, so no label rests on a guess."""
    bad = {}
    for row in _rows(name):
        vehicle = _vehicle_of(row, _vehicles())
        if vehicle is None:
            bad[row["id"]] = ["no catalog vehicle"]
            continue
        year = _first_user_year(row)
        by_id = {s["id"]: s for s in row["snippets"]}
        deciding = row["gold"]["supporting_ids"] + row["gold"]["near_miss_ids"]
        problems = [
            f"[{sid}] {raw['id']}: {', '.join(missing)}"
            for sid in deciding
            if (raw := _source(by_id[sid])) and (missing := _unstated(row, raw, vehicle, year))
        ]
        if problems:
            bad[row["id"]] = problems
    assert not bad, (
        f"{name}: {len(bad)} scenarios rest gold on a fact they never state, e.g. "
        f"{dict(list(bad.items())[:8])}"
    )


# --- (4) abstain rows contain twin pairs too ----------------------------------------------------

CITATION = re.compile(r"\[([^\[\]]+)\]")


def _in_scope(row: Row, raw: Raw, vehicle: dict[str, Any], year: int) -> bool:
    applies = raw.get("applies") or {}
    if any(_fails(d, v, vehicle, year) for d, v in applies.items() if v):
        return False
    family = _families_cached()[raw["family"]]
    index = _complaint_index(row, family)
    if index is None:
        return False
    facts = _complaint_facts(family)[index]
    return all(facts.get(k) in _allowed(v) for k, v in (raw.get("requires") or {}).items())


def _out_of_scope_pair(row: Row) -> bool:
    """A twin pair on the complaint's family, neither member applying nor already cited."""
    vehicle = _vehicle_of(row, _vehicles())
    year = _first_user_year(row)
    if vehicle is None or year is None:
        return False
    cited = {
        c
        for m in row["messages"]
        if m["role"] == "assistant"
        for c in CITATION.findall(m["content"])
    }
    for a, b in _twin_pairs(row):
        pair = [row["snippets"][a], row["snippets"][b]]
        raws = [_source(s) for s in pair]
        if any(s["id"] in cited for s in pair) or None in raws:
            continue
        family = _families_cached()[raws[0]["family"]]
        if _complaint_index(row, family) is None:
            continue
        if not any(_in_scope(row, raw, vehicle, year) for raw in raws):
            return True
    return False


@pytest.mark.parametrize("name", SETS)
def test_some_abstain_scenarios_hold_an_out_of_scope_twin_pair(name: str) -> None:
    abstain = [row for row in _rows(name) if row["gold"]["mode"] == "abstain"]
    hits = [row["id"] for row in abstain if _out_of_scope_pair(row)]
    share = len(hits) / len(abstain) if abstain else 0.0
    print(f"{name}: {len(hits)}/{len(abstain)} abstain scenarios hold an out-of-scope twin pair")
    assert share >= MIN_ABSTAIN_PAIR_SHARE, (
        f"{name}: {share:.0%} of {len(abstain)} abstain scenarios contain a twin pair with both "
        f"members out of scope (need >= {MIN_ABSTAIN_PAIR_SHARE:.0%})"
    )


@pytest.mark.parametrize("name", SETS)
def test_a_twin_pair_does_not_imply_cite(name: str) -> None:
    with_pair = [row for row in _rows(name) if _twin_pairs(row)]
    cite = sum(row["gold"]["mode"] == "cite" for row in with_pair)
    rate = cite / len(with_pair) if with_pair else 0.0
    print(f"{name}: P(cite | twin pair present) = {cite}/{len(with_pair)}")
    assert rate <= MAX_CITE_GIVEN_PAIR + EPS, (
        f"{name}: {rate:.0%} of the {len(with_pair)} scenarios with a twin pair are cite-gold "
        f"(limit {MAX_CITE_GIVEN_PAIR:.0%}), so 'two near-identical snippets' predicts cite"
    )


# --- (5) hint screen: validation 3 paraphrases, narrow word senses ------------------------------

LEAKY_REPLIES = (
    "The reference I'm meant to use here is [TSB-4101]: inspect the hose at the intake elbow "
    "for cracks.",
    "Per the guidance I was handed, inspect the hose at the intake elbow for cracks [TSB-4101].",
    "The teacher note says this is the right one: inspect the hose at the intake elbow for "
    "cracks [TSB-4101].",
    "That's the correct citation for this case: inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "I was given this as the answer to point to: inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "My notes mark the others as not applicable, so inspect the hose at the intake elbow for "
    "cracks [TSB-4101].",
    "The snippet flagged as relevant is the elbow bulletin: inspect the hose at the intake elbow "
    "for cracks [TSB-4101].",
    "Going with the one the key points to, inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "As instructed, inspect the hose at the intake elbow for cracks [TSB-4101].",
    "The others are decoys, so inspect the hose at the intake elbow for cracks [TSB-4101].",
)

CLEAN_REPLIES = (
    "With the engine idling, unplug the gold-colored connector at the intake elbow sensor and "
    "inspect the hose behind it for cracks [TSB-4101].",
    "If you notice a hint of coolant smell near the elbow, inspect the hose at the intake elbow "
    "for cracks or a vacuum hiss [TSB-4101].",
    "Per the note in the bulletin about idle leaks, inspect the hose at the intake elbow for "
    "cracks or a vacuum hiss [TSB-4101].",
    "With the engine idling, listen for any hint of a vacuum hiss at the intake elbow and inspect "
    "the hose for cracks [TSB-4101].",
    "Remove the gold-colored test port cap and inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "Inspect the hose at the intake elbow for cracks and write what you find on the note card "
    "[TSB-4101].",
)


@pytest.mark.parametrize("reply", LEAKY_REPLIES)
def test_hint_leak_screen_rejects_validation_3_paraphrases(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is not None and rejection.reason == HINT_LEAK, (
        f"not rejected as {HINT_LEAK}: {reply!r} -> {rejection}"
    )


@pytest.mark.parametrize("reply", CLEAN_REPLIES)
def test_hint_leak_screen_passes_hint_gold_and_note_in_ordinary_senses(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is None or rejection.reason != HINT_LEAK, f"{reply!r} -> {rejection}"
