"""Twin near-misses, a generic surface probe and gold integrity (D-056, kind-balance validation 2).

Rebalancing one surface feature at a time only moved the shortcut (kind, then length, then commas
and units). D-056 builds every near-miss as a TWIN of a gold-capable snippet instead, so gold and
near-miss differ only in applicability scope. This module checks:

1. Twin construction (library). Every near-miss snippet (``mismatch`` set) declares
   ``twin_of: <gold-capable id>`` in its own family and ``scope_edits: [[old, new], ...]``.
   Applying the edits to the twin's text, in order and each exactly once, gives the near-miss
   text exactly. The edits are small (normalized token edit distance at most
   ``MAX_TWIN_TOKEN_EDITS``), and the numbers outside the edited phrases are identical. The
   scope differs from the twin's in engine, model years, transmission, body (``applies``) or an
   operating condition (``requires``). In every scenario where the snippet is a near-miss, that
   difference is what rules it out: a vehicle twin's changed dimension excludes the scenario
   vehicle, and a condition twin's ``requires`` contradicts the complaint the scenario uses.
2. A generic probe: ridge regression on surface features only, leave-one-scenario-out within
   each set (and fit on train_pool, scored on the eval sets), picks the gold on at most 0.60 of
   distractor scenarios, held-out included. sklearn and numpy are not dependencies, so the
   probe is a deterministic closed-form ridge on standardized, within-scenario-centered features.
3. Gold integrity. Families list structured ``complaint_facts`` (one table per complaint), and
   gold-capable snippets list structured ``requires``. Every key a gold-capable snippet requires
   is stated by every complaint of its family, with a compatible value. Regressions:
   ``ac_warm.condenser_fan`` (no "cool again at speed") and ``heater_cold.thermostat`` (no low
   gauge).
4. The R4 hint-leak screen catches the paraphrases validation 2 found, and still passes normal
   replies that use the same words in an ordinary sense.
5. ``one_step`` passes every canonical chain step (validation 2: 7% were SOFT-flagged).

Vocabulary (implementation chooses the rest): a ``requires``/``complaint_facts`` value is a string;
a ``requires`` value may also be a list of acceptable strings. heater_cold complaints state
``gauge = "normal"``; ac_warm complaints that say the air is warm all the time state
``warm_air = "constant"``.
"""

from __future__ import annotations

import functools
import json
import re
import statistics
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from distill_data import id_scheme
from test_kind_balance import (
    APPLIES_TO_ALL,
    BULLETIN_SCHEME,
    CORRECTION_LINE,
    SET_NAMES,
    YEAR,
    _first_user_year,
    _rows,
    _screen,
)
from test_scenario_coherence import _vehicle_of
from test_shortcuts import (
    APPLIES_TO_VEHICLES_WITH,
    EPS,
    MAX_HEURISTIC_ACCURACY,
    _distractor_rows,
    commas,
    digits,
    numbers,
    units,
    words,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNIPPET_DIR = REPO_ROOT / "data" / "snippets"
FAMILIES = REPO_ROOT / "data" / "families.toml"

# A twin changes an engine list, a year range, a transmission, a body or one condition phrase.
MAX_TWIN_TOKEN_EDITS = 10
SCOPE_DIMENSIONS = ("engines", "years", "trans", "bodies")
MAX_PROBE_ACCURACY = MAX_HEURISTIC_ACCURACY
RIDGE_LAMBDA = 1.0

Raw = dict[str, Any]


def _library() -> dict[str, Raw]:
    by_id: dict[str, Raw] = {}
    for path in sorted(SNIPPET_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                by_id[raw["id"]] = raw
    return by_id


def _toml() -> dict[str, Any]:
    with FAMILIES.open("rb") as f:
        return tomllib.load(f)


def _families() -> dict[str, dict[str, Any]]:
    return {family["name"]: family for family in _toml()["family"]}


def _near_misses() -> list[Raw]:
    return [raw for raw in _library().values() if raw.get("mismatch")]


def _gold_capable() -> list[Raw]:
    return [raw for raw in _library().values() if raw.get("step")]


def _by_text() -> dict[str, Raw]:
    return {raw["text"]: raw for raw in _library().values()}


# --- (1) twin construction ---------------------------------------------------------------------

TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _edits(raw: Raw) -> list[tuple[str, str]]:
    return [(old, new) for old, new in raw.get("scope_edits") or []]


def _twin(raw: Raw) -> Raw:
    twin_id = raw.get("twin_of")
    assert twin_id, f"{raw['id']}: near-miss declares no twin_of"
    library = _library()
    assert twin_id in library, f"{raw['id']}: twin_of {twin_id!r} is not a library snippet"
    return library[twin_id]


def _apply_edits(text: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        assert text.count(old) == 1, f"scope edit {old!r} must occur exactly once in {text!r}"
        text = text.replace(old, new)
    return text


def _without(text: str, phrases: list[str]) -> str:
    for phrase in phrases:
        text = text.replace(phrase, " ")
    return text


def test_library_has_near_misses() -> None:
    assert len(_near_misses()) >= 40


@pytest.mark.parametrize("raw", _near_misses(), ids=lambda raw: raw["id"])
def test_near_miss_declares_a_gold_capable_twin_in_its_family(raw: Raw) -> None:
    twin = _twin(raw)
    assert twin.get("step") and not twin.get("mismatch"), (
        f"{raw['id']}: twin_of {twin['id']} is not gold-capable (no chain step)"
    )
    assert twin["family"] == raw["family"], (
        f"{raw['id']} ({raw['family']}) twins {twin['id']} from another family ({twin['family']})"
    )


@pytest.mark.parametrize("raw", _near_misses(), ids=lambda raw: raw["id"])
def test_near_miss_text_is_its_twin_with_only_the_declared_scope_edits(raw: Raw) -> None:
    twin = _twin(raw)
    edits = _edits(raw)
    assert edits, f"{raw['id']}: no scope_edits declared"
    assert all(old != new for old, new in edits), f"{raw['id']}: a no-op scope edit: {edits}"
    rebuilt = _apply_edits(twin["text"], edits)
    assert rebuilt == raw["text"], (
        f"{raw['id']}: twin text with scope_edits applied differs from the snippet text\n"
        f"  rebuilt: {rebuilt!r}\n  actual:  {raw['text']!r}"
    )


@pytest.mark.parametrize("raw", _near_misses(), ids=lambda raw: raw["id"])
def test_near_miss_differs_from_its_twin_by_a_few_tokens(raw: Raw) -> None:
    twin = _twin(raw)
    distance = _edit_distance(_tokens(twin["text"]), _tokens(raw["text"]))
    assert 0 < distance <= MAX_TWIN_TOKEN_EDITS, (
        f"{raw['id']}: {distance} token edits from {twin['id']} (allowed 1-{MAX_TWIN_TOKEN_EDITS})"
    )


@pytest.mark.parametrize("raw", _near_misses(), ids=lambda raw: raw["id"])
def test_near_miss_numbers_outside_the_scope_edits_match_its_twin(raw: Raw) -> None:
    twin = _twin(raw)
    edits = _edits(raw)
    outside_twin = NUMBER.findall(_without(twin["text"], [old for old, _ in edits]))
    outside_near = NUMBER.findall(_without(raw["text"], [new for _, new in edits]))
    assert outside_near == outside_twin, (
        f"{raw['id']}: numbers outside the scope edits {outside_near} differ from "
        f"{twin['id']}'s {outside_twin}"
    )


def _applies(raw: Raw) -> dict[str, Any]:
    return raw.get("applies") or {}


def _requires(raw: Raw) -> dict[str, Any]:
    return raw.get("requires") or {}


def _changed_dimensions(raw: Raw, twin: Raw) -> list[str]:
    return [d for d in SCOPE_DIMENSIONS if _applies(raw).get(d) != _applies(twin).get(d)]


def _changed_conditions(raw: Raw, twin: Raw) -> list[str]:
    keys = set(_requires(raw)) | set(_requires(twin))
    return sorted(k for k in keys if _requires(raw).get(k) != _requires(twin).get(k))


@pytest.mark.parametrize("raw", _near_misses(), ids=lambda raw: raw["id"])
def test_near_miss_scope_differs_from_its_twin(raw: Raw) -> None:
    twin = _twin(raw)
    if raw["mismatch"] == "vehicle":
        changed = _changed_dimensions(raw, twin)
        assert any(_applies(raw).get(d) for d in changed), (
            f"{raw['id']}: a vehicle near-miss must narrow engine, years, transmission or body "
            f"relative to {twin['id']}: applies {_applies(raw)} vs {_applies(twin)}"
        )
    else:
        changed = _changed_conditions(raw, twin)
        assert any(k in _requires(raw) for k in changed), (
            f"{raw['id']}: a condition near-miss must require a condition {twin['id']} does not: "
            f"requires {_requires(raw)} vs {_requires(twin)}"
        )


def _allowed(value: Any) -> set[str]:
    return set(value) if isinstance(value, list) else {value}


def _conflicts(requires: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """Required conditions the complaint states with an incompatible value."""
    return [k for k, v in requires.items() if k in facts and facts[k] not in _allowed(v)]


def _complaint_facts(family: dict[str, Any]) -> list[dict[str, Any]]:
    return list(family.get("complaint_facts") or [])


@pytest.mark.parametrize(
    "raw", [r for r in _near_misses() if r["mismatch"] == "condition"], ids=lambda raw: raw["id"]
)
def test_condition_near_miss_contradicts_every_complaint_of_its_family(raw: Raw) -> None:
    family = _families()[raw["family"]]
    facts = _complaint_facts(family)
    assert len(facts) == len(family["complaints"]), f"{raw['family']}: complaint_facts missing"
    agreeing = [
        family["complaints"][i]
        for i, stated in enumerate(facts)
        if not _conflicts(_requires(raw), stated)
    ]
    assert not agreeing, (
        f"{raw['id']} requires {_requires(raw)}, which these {raw['family']} complaints don't "
        f"contradict: {agreeing}"
    )


def _complaint_pattern(template: str) -> re.Pattern[str]:
    pattern = re.escape(template).replace(r"\{cyl\}", r"\d+").replace(r"\{side\}", "(?:left|right)")
    return re.compile(pattern, re.IGNORECASE)


def _complaint_index(row: dict[str, Any], family: dict[str, Any]) -> int | None:
    user = " ".join(m["content"] for m in row["messages"] if m["role"] == "user")
    for i, template in enumerate(family["complaints"]):
        if _complaint_pattern(template).search(user):
            return i
    return None


def _fails(dimension: str, value: Any, vehicle: dict[str, Any], year: int) -> bool:
    if dimension == "years":
        return not value[0] <= year <= value[1]
    field = {"engines": "engine", "trans": "trans", "bodies": "body"}[dimension]
    return vehicle[field] not in value


def _near_miss_problems(row: dict[str, Any]) -> list[str]:
    library, by_text, families = _library(), _by_text(), _families()
    by_id = {s["id"]: s for s in row["snippets"]}
    vehicle = _vehicle_of(row, _toml()["vehicle"])
    year = _first_user_year(row)
    problems = []
    for near_id in row["gold"]["near_miss_ids"]:
        raw = by_text.get(by_id[near_id]["text"])
        if raw is None or not raw.get("twin_of") or raw["twin_of"] not in library:
            problems.append(f"[{near_id}] is not a library twin")
            continue
        twin = library[raw["twin_of"]]
        if raw["mismatch"] == "vehicle":
            if vehicle is None or year is None:
                problems.append(f"[{near_id}] {raw['id']}: no catalog vehicle and year")
                continue
            ruled_out = [
                d
                for d in _changed_dimensions(raw, twin)
                if _applies(raw).get(d) and _fails(d, _applies(raw)[d], vehicle, year)
            ]
            if not ruled_out:
                problems.append(
                    f"[{near_id}] {raw['id']}: no changed scope excludes the {year} "
                    f"{vehicle['engine']} {vehicle['trans']} {vehicle['body']}"
                )
        else:
            family = families[raw["family"]]
            index = _complaint_index(row, family)
            facts = _complaint_facts(family)
            if index is None or index >= len(facts):
                problems.append(f"[{near_id}] {raw['id']}: no {raw['family']} complaint facts")
                continue
            changed = set(_changed_conditions(raw, twin))
            if not changed & set(_conflicts(_requires(raw), facts[index])):
                problems.append(
                    f"[{near_id}] {raw['id']}: changed conditions {sorted(changed)} don't "
                    f"contradict complaint {index} facts {facts[index]}"
                )
    return problems


@pytest.mark.parametrize("name", SET_NAMES)
def test_every_scenario_near_miss_is_ruled_out_by_its_scope_change(name: str) -> None:
    bad = {row["id"]: p for row in _rows(name) if (p := _near_miss_problems(row))}
    shown = dict(list(bad.items())[:10])
    assert not bad, (
        f"{name}: {len(bad)} scenarios with a near-miss not ruled out by its twin's scope "
        f"change, e.g. {shown}"
    )


# --- (2) generic surface probe -----------------------------------------------------------------

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
TIP = re.compile(
    r"\b(?:helps?|quicker|easier|handy|a helper|save|record|note how|keep|caution|careful|avoid)\b",
    re.IGNORECASE,
)

Snippet = dict[str, Any]
Row = dict[str, Any]


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.strip()) if s]


def _has_tip(snippet: Snippet) -> bool:
    return any(TIP.search(s) and not CORRECTION_LINE.search(s) for s in _sentences(snippet["text"]))


SURFACE: dict[str, Callable[[Snippet], float]] = {
    "chars": lambda s: len(s["text"]),
    "words": words,
    "commas": commas,
    "digits": digits,
    "numbers": numbers,
    "units": units,
    "sentences": lambda s: len(_sentences(s["text"])),
    "year": lambda s: float(bool(YEAR.search(s["text"]))),
    "bulletin kind": lambda s: float(s.get("kind") == "tsb"),
    "bulletin-style ID": lambda s: float(bool(BULLETIN_SCHEME.match(s["id"]))),
    "correction line": lambda s: float(bool(CORRECTION_LINE.search(s["text"]))),
    "'Applies to all' opener": lambda s: float(bool(APPLIES_TO_ALL.match(s["text"]))),
    "'Applies to vehicles with' opener": lambda s: float(
        bool(APPLIES_TO_VEHICLES_WITH.match(s["text"]))
    ),
    "tip sentence": lambda s: float(_has_tip(s)),
}


# Round 4 (kind-balance validation 3): twins sat at the catalog's year tails while gold ranges
# spanned the middle, and duplicate titles located the pair. A probe with year-range values and a
# duplicate-title feature scored 0.72 fit on train_pool, so the probe carries both, plus pair
# features (a snippet with a near-identical partner) and their interaction with the year prior.
YEAR_RANGE = re.compile(r"\b((?:19|20)\d\d)\s*[-\u2013]\s*((?:19|20)\d\d)\b")


def _catalog_median_year() -> int:
    years = [y for v in _toml()["vehicle"] for y in range(v["years"][0], v["years"][1] + 1)]
    return int(statistics.median_low(years))


CATALOG_MEDIAN_YEAR = _catalog_median_year()


def year_range(snippet: Snippet) -> tuple[int, int] | None:
    """The model-year range the snippet text states, or ``None`` (no year limit)."""
    match = YEAR_RANGE.search(snippet["text"])
    return (int(match.group(1)), int(match.group(2))) if match else None


def contains_median_year(snippet: Snippet) -> bool:
    """True when the stated range covers the catalog's median model year (no range covers it)."""
    span = year_range(snippet)
    return span is None or span[0] <= CATALOG_MEDIAN_YEAR <= span[1]


@functools.cache
def _near_identical(a: str, b: str) -> bool:
    return _edit_distance(_tokens(a), _tokens(b)) <= MAX_TWIN_TOKEN_EDITS


def pair_partners(row: Row, i: int) -> list[int]:
    """Other snippets a reader would see as the same snippet: same title or near-identical text.
    Label-free, so a model can compute it too."""
    snippets = row["snippets"]
    me = snippets[i]
    return [
        j
        for j, other in enumerate(snippets)
        if j != i
        and (
            (me.get("title") and other.get("title") == me["title"])
            or _near_identical(me["text"], other["text"])
        )
    ]


def _year_features(row: Row, i: int) -> list[float]:
    snippet = row["snippets"][i]
    span = year_range(snippet)
    start, end = span if span else (CATALOG_MEDIAN_YEAR, CATALOG_MEDIAN_YEAR)
    mid = (start + end) / 2
    contains = float(contains_median_year(snippet))
    titles = [s.get("title") for s in row["snippets"]]
    duplicate_titles = float(titles.count(snippet["title"]) - 1) if snippet.get("title") else 0.0
    in_pair = float(bool(pair_partners(row, i)))
    return [
        float(span is not None),
        start,
        end,
        end - start,
        mid,
        abs(mid - CATALOG_MEDIAN_YEAR),
        contains,
        duplicate_titles,
        in_pair,
        in_pair * contains,
        in_pair * abs(mid - CATALOG_MEDIAN_YEAR),
    ]


def _features(row: Row, schemes: Sequence[str]) -> list[list[float]]:
    """Surface features, year-range and pair features, position and a one-hot ID scheme, one
    vector per snippet."""
    snippets = row["snippets"]
    last = max(len(snippets) - 1, 1)
    vectors = []
    for i, snippet in enumerate(snippets):
        vector = [float(fn(snippet)) for fn in SURFACE.values()]
        vector += _year_features(row, i)
        vector += [i / last, float(i == 0), float(i == len(snippets) - 1)]
        vector += [float(id_scheme(snippet["id"]) == scheme) for scheme in schemes]
        vectors.append(vector)
    return vectors


def _labels(row: Row) -> list[float]:
    gold = set(row["gold"]["supporting_ids"])
    return [float(s["id"] in gold) for s in row["snippets"]]


def _schemes(rows: Sequence[Row]) -> list[str]:
    return sorted({id_scheme(s["id"]) for row in rows for s in row["snippets"]})


Scaler = tuple[list[float], list[float]]


def _scaler(rows: Sequence[Row], schemes: Sequence[str]) -> Scaler:
    """Label-free per-feature mean and standard deviation (constant features get scale 0)."""
    columns = list(zip(*[v for row in rows for v in _features(row, schemes)], strict=True))
    means = [statistics.fmean(c) for c in columns]
    scales = [statistics.pstdev(c) for c in columns]
    return means, [1 / s if s > 0 else 0.0 for s in scales]


def _design(
    row: Row, schemes: Sequence[str], scaler: Scaler
) -> tuple[list[list[float]], list[float]]:
    """Standardized features and labels, both centered within the scenario."""
    means, inverse = scaler
    x = [
        [(v - m) * k for v, m, k in zip(vector, means, inverse, strict=True)]
        for vector in _features(row, schemes)
    ]
    y = _labels(row)
    x_mean = [statistics.fmean(c) for c in zip(*x, strict=True)]
    y_mean = statistics.fmean(y)
    return [[v - m for v, m in zip(r, x_mean, strict=True)] for r in x], [v - y_mean for v in y]


Normal = tuple[list[list[float]], list[float]]


def _normal(designs: Sequence[tuple[list[list[float]], list[float]]], size: int) -> Normal:
    a = [[0.0] * size for _ in range(size)]
    b = [0.0] * size
    for x, y in designs:
        for r, t in zip(x, y, strict=True):
            for i in range(size):
                b[i] += r[i] * t
                for j in range(size):
                    a[i][j] += r[i] * r[j]
    return a, b


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan elimination with partial pivoting (a is symmetric positive definite)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(n):
            if r != col and m[r][col]:
                factor = m[r][col]
                m[r] = [v - factor * p for v, p in zip(m[r], m[col], strict=True)]
    return [m[i][n] for i in range(n)]


def _ridge(normal: Normal, minus: Normal | None = None) -> list[float]:
    a, b = normal
    if minus is not None:
        a = [[v - w for v, w in zip(r, s, strict=True)] for r, s in zip(a, minus[0], strict=True)]
        b = [v - w for v, w in zip(b, minus[1], strict=True)]
    a = [[v + (RIDGE_LAMBDA if i == j else 0.0) for j, v in enumerate(r)] for i, r in enumerate(a)]
    return _solve(a, b)


def _pick_accuracy(row: Row, x: list[list[float]], w: Sequence[float]) -> float:
    """Share of the top-scoring snippets that are gold (uniform tie-breaking)."""
    scores = [sum(v * c for v, c in zip(r, w, strict=True)) for r in x]
    best = max(scores)
    top = [i for i, s in enumerate(scores) if s >= best - 1e-12]
    labels = _labels(row)
    return sum(labels[i] for i in top) / len(top)


def probe_loso_accuracy(rows: Sequence[Row]) -> float:
    """Leave-one-scenario-out accuracy of the surface ridge probe at picking the gold."""
    schemes = _schemes(rows)
    scaler = _scaler(rows, schemes)
    designs = [_design(row, schemes, scaler) for row in rows]
    size = len(designs[0][0][0])
    total = _normal(designs, size)
    hits = 0.0
    for row, design in zip(rows, designs, strict=True):
        w = _ridge(total, minus=_normal([design], size))
        hits += _pick_accuracy(row, design[0], w)
    return hits / len(rows)


def probe_transfer_accuracy(train: Sequence[Row], test: Sequence[Row]) -> float:
    """Fit the probe on ``train`` and score it on ``test`` (scaling from ``train``)."""
    schemes = _schemes(train)
    scaler = _scaler(train, schemes)
    designs = [_design(row, schemes, scaler) for row in train]
    w = _ridge(_normal(designs, len(designs[0][0][0])))
    return statistics.fmean(
        _pick_accuracy(row, _design(row, schemes, scaler)[0], w) for row in test
    )


def _synthetic_rows(cue: Callable[[int, bool], str], n: int = 30) -> list[Row]:
    """Three-snippet scenarios; gold sits at a varying position and ``cue`` writes the texts."""
    rows = []
    for k in range(n):
        gold_at = k % 3
        snippets = [
            {"id": f"SM {k}.{i}", "kind": "manual", "text": cue(k * 3 + i, i == gold_at)}
            for i in range(3)
        ]
        rows.append({"snippets": snippets, "gold": {"supporting_ids": [f"SM {k}.{gold_at}"]}})
    return rows


def test_probe_finds_a_planted_comma_cue() -> None:
    def cue(seed: int, gold: bool) -> str:
        filler = " ".join(["word"] * (8 + seed % 5))
        return f"Applies to all engines. {filler}, a, b{', c, d' if gold else ''}."

    assert probe_loso_accuracy(_synthetic_rows(cue)) >= 0.9


def test_probe_is_at_chance_without_a_cue() -> None:
    def cue(seed: int, gold: bool) -> str:
        return "Applies to all engines. " + " ".join(["word"] * (8 + (seed * 7) % 5)) + "."

    assert probe_loso_accuracy(_synthetic_rows(cue)) <= 0.5


def test_probe_finds_a_planted_year_prior_within_a_twin_pair() -> None:
    """Round 4: gold is the pair member whose range covers the catalog median; a lone snippet
    that also covers it is never gold, so only the pair interaction separates them."""
    mid = CATALOG_MEDIAN_YEAR
    rows = []
    for k in range(30):
        body = " ".join(["word"] * (8 + k % 5)) + "."
        texts = [
            (f"Pair {k}", f"Applies to model years {mid - 3}-{mid + 3}. {body}"),
            (f"Pair {k}", f"Applies to model years {mid - 14}-{mid - 5}. {body}"),
            (f"Lone {k}", f"Applies to model years {mid - 2}-{mid + 4}. Other {body}"),
        ]
        order = [(k + n) % 3 for n in range(3)]
        snippets = [
            {"id": f"SM {k}.{n}", "kind": "manual", "title": texts[t][0], "text": texts[t][1]}
            for n, t in enumerate(order)
        ]
        gold = f"SM {k}.{order.index(0)}"
        rows.append({"snippets": snippets, "gold": {"supporting_ids": [gold]}})
    assert probe_loso_accuracy(rows) >= 0.9


@pytest.mark.parametrize("name", SET_NAMES)
def test_surface_probe_does_not_solve_distractors(name: str) -> None:
    rows = _distractor_rows(name)
    accuracy = probe_loso_accuracy(rows)
    print(f"{name}: surface probe (leave-one-scenario-out): {accuracy:.2f} on {len(rows)}")
    assert accuracy <= MAX_PROBE_ACCURACY + EPS, (
        f"{name}: a ridge probe on surface features picks the gold on {accuracy:.0%} of "
        f"{len(rows)} distractor scenarios (limit {MAX_PROBE_ACCURACY:.0%})"
    )


@pytest.mark.parametrize("name", [n for n in SET_NAMES if n != "train_pool"])
def test_surface_probe_fit_on_train_pool_does_not_solve_eval_distractors(name: str) -> None:
    rows = _distractor_rows(name)
    accuracy = probe_transfer_accuracy(_distractor_rows("train_pool"), rows)
    print(f"{name}: surface probe fit on train_pool: {accuracy:.2f} on {len(rows)}")
    assert accuracy <= MAX_PROBE_ACCURACY + EPS, (
        f"{name}: a surface probe fit on train_pool distractors picks the gold on {accuracy:.0%} "
        f"of {len(rows)} distractor scenarios (limit {MAX_PROBE_ACCURACY:.0%})"
    )


# --- (3) gold integrity: declared conditions fit every complaint --------------------------------


@pytest.mark.parametrize("name", sorted(_families()))
def test_every_complaint_has_structured_facts(name: str) -> None:
    family = _families()[name]
    facts = _complaint_facts(family)
    assert len(facts) == len(family["complaints"]), (
        f"{name}: {len(facts)} complaint_facts tables for {len(family['complaints'])} complaints"
    )
    for i, stated in enumerate(facts):
        assert isinstance(stated, dict) and stated, f"{name}: complaint {i} states no facts"
        assert all(isinstance(v, str) for v in stated.values()), f"{name}[{i}]: {stated}"


@pytest.mark.parametrize("raw", _gold_capable(), ids=lambda raw: raw["id"])
def test_gold_capable_snippet_conditions_hold_for_every_complaint(raw: Raw) -> None:
    """Every required condition is stated by each complaint of the family, with a fitting value."""
    assert isinstance(raw.get("requires"), dict), (
        f"{raw['id']}: gold-capable snippets declare their conditions as a `requires` table "
        "(empty when the text states none)"
    )
    family = _families()[raw["family"]]
    facts = _complaint_facts(family)
    assert len(facts) == len(family["complaints"]), f"{raw['family']}: complaint_facts missing"
    problems = []
    for key, value in raw["requires"].items():
        for i, stated in enumerate(facts):
            if key not in stated:
                problems.append(f"complaint {i} doesn't state {key!r}")
            elif stated[key] not in _allowed(value):
                problems.append(
                    f"complaint {i} says {key}={stated[key]!r}, snippet needs {value!r}"
                )
    assert not problems, f"{raw['id']} requires {raw['requires']}: {problems}"


def test_condenser_fan_gold_does_not_require_cooling_again_at_speed() -> None:
    raw = _library()["ac_warm.condenser_fan"]
    assert not re.search(r"\bcool\w*\b[^.]{0,20}\bat (?:highway )?speed", raw["text"], re.I), (
        f"ac_warm.condenser_fan says the air cools again at speed, but ac_warm complaints say "
        f"it is warm all the time: {raw['text']!r}"
    )
    family = _families()["ac_warm"]
    facts = _complaint_facts(family)
    constant = [
        i
        for i, text in enumerate(family["complaints"])
        if re.search(r"all the time|from the moment I start", text)
    ]
    assert constant and len(facts) == len(family["complaints"])
    assert all(facts[i].get("warm_air") == "constant" for i in constant), (
        f"ac_warm complaints {constant} say the air is warm all the time; their facts must state "
        f"warm_air = 'constant': {facts}"
    )
    required = _requires(raw).get("warm_air")
    assert required is None or "constant" in _allowed(required), (
        f"ac_warm.condenser_fan requires warm_air={required!r}"
    )


def test_heater_thermostat_gold_does_not_require_a_low_gauge() -> None:
    raw = _library()["heater_cold.thermostat"]
    assert not re.search(
        r"\blow\b[^.]{0,20}\bgauge|\bgauge\b[^.]{0,20}\blow\b", raw["text"], re.I
    ), (
        f"heater_cold.thermostat says the gauge reads low, but heater_cold complaints say it "
        f"reads normal: {raw['text']!r}"
    )
    facts = _complaint_facts(_families()["heater_cold"])
    assert facts and all(f.get("gauge") == "normal" for f in facts), (
        f"every heater_cold complaint says the gauge is normal; facts must state gauge = "
        f"'normal': {facts}"
    )
    required = _requires(raw).get("gauge")
    assert required is None or "normal" in _allowed(required), (
        f"heater_cold.thermostat requires gauge={required!r}"
    )


# --- (4) hint-leak screen: validation 2 paraphrases --------------------------------------------

LEAKY_REPLIES = (
    "The intended answer is this check: inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "Per my instructions, inspect the hose at the intake elbow for cracks [TSB-4101].",
    "As per my instructions, with the engine idling, inspect the hose at the intake elbow "
    "[TSB-4101].",
    "Since the labeled reference applies, inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "The labelled reference covers it: inspect the hose at the intake elbow for cracks [TSB-4101].",
    "Going by the reference I was given, inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "Inspect the hose at the intake elbow for cracks [TSB-4101]. That is the designated snippet.",
)

CLEAN_REPLIES = (
    "Following the service manual instructions, with the engine idling, inspect the hose at "
    "the intake elbow for cracks or a vacuum hiss [TSB-4101].",
    "With the engine idling, inspect the hose at the labeled vacuum port on the intake elbow "
    "for cracks or a vacuum hiss [TSB-4101].",
    "Connect a gauge to the designated test port on the rail and note the reading at idle "
    "[TSB-4101].",
    "The intended idle speed is on the emissions label; with the engine idling, inspect the "
    "hose at the intake elbow for cracks [TSB-4101].",
    "The reference you gave covers this: inspect the hose at the intake elbow for cracks or a "
    "vacuum hiss [TSB-4101].",
)


@pytest.mark.parametrize("reply", LEAKY_REPLIES)
def test_hint_leak_screen_rejects_validation_2_paraphrases(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is not None and rejection.reason == HINT_LEAK, (
        f"not rejected as {HINT_LEAK}: {reply!r} -> {rejection}"
    )


@pytest.mark.parametrize("reply", CLEAN_REPLIES)
def test_hint_leak_screen_passes_ordinary_uses_of_the_same_words(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is None or rejection.reason != HINT_LEAK, f"{reply!r} -> {rejection}"


# --- (5) one_step accepts the canonical chain steps --------------------------------------------


@pytest.mark.parametrize("raw", _gold_capable(), ids=lambda raw: raw["id"])
def test_one_step_accepts_the_canonical_chain_step(raw: Raw) -> None:
    from groundcite.checks import one_step

    reply = f"{raw['step']} [SM 1.1]."
    result = one_step(reply, ["SM 1.1"])
    assert result.passed, f"{raw['id']}: one_step SOFT-flags its own library step: {result.detail}"
