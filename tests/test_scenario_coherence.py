"""Scenario coherence notes from the scenario-set validation (Q4-Q7).

- Q4: a harsh-shift complaint only appears on an automatic-transmission vehicle.
- Q5: the "that reading is in spec" follow-up only follows a check that produces a reading, never a
  visual inspection.
- Q6: a scenario holds one snippet list for the whole conversation, so the generator can't show
  a reference being added between turns. The "I found one more reference" phrasing is therefore
  absent from the phrasing pools and from every set (D-034).
- Q7: rows with two supporting IDs use any-of gold, and the docs say so.

Q4-Q6 are checked on every committed set and on fresh generator runs with other seeds, so a
pass reflects the generator, not one lucky seed.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SCENARIO_DIR = DATA_DIR / "scenarios"
SNIPPET_DIR = DATA_DIR / "snippets"
FAMILIES_TOML = DATA_DIR / "families.toml"
PHRASING_TOML = DATA_DIR / "phrasing.toml"
DOCS = (REPO_ROOT / "docs" / "scenario-coverage.md", REPO_ROOT / "ARCHITECTURE.md")

COMMITTED = ("ablation", "eval_own", "heldout")
FRESH_SEEDS = (1028, 1031)
CITATION = re.compile(r"\[([^\[\]]+)\]")

# A step "produces a reading" when it names an instrument or a measurement. Steps without one
# (inspect, look, listen, feel, confirm the fan runs) are visual or sensory checks.
READING_STEP = re.compile(
    r"\b(measure\w*|read|reading|gauge|meter|ammeter|voltmeter|micrometer|dial indicator|"
    r"thermometer|tester|test strip|balancer|load test|compression test|leak test|"
    r"pressure[- ]test|pressurize|back-probe|cap adapter|psi|mA|kV|V AC)\b",
    re.IGNORECASE,
)
READING_PHRASE = re.compile(r"\breading\b|\bin spec\b", re.IGNORECASE)
ADDED_REFERENCE = re.compile(
    r"\b(one more|another|an extra|a new) (reference|service document|snippet|bulletin|document)\b"
    r"|\badded (it|one|another|them) to the list\b|\bput it in with the others\b",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _committed() -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, row) for name in COMMITTED for row in _load_jsonl(SCENARIO_DIR / f"{name}.jsonl")
    ]


def _fresh(seed: int) -> list[tuple[str, dict[str, Any]]]:
    from groundcite.scenarios.generator import generate
    from groundcite.scenarios.library import load_library

    sets = generate(load_library(), seed)
    return [(f"seed{seed}/{name}", row) for name, rows in sets.items() for row in rows]


SOURCES = ["committed", *(f"seed{s}" for s in FRESH_SEEDS)]


@pytest.fixture(scope="module", params=SOURCES)
def rows(request: pytest.FixtureRequest) -> list[tuple[str, dict[str, Any]]]:
    if request.param == "committed":
        return _committed()
    return _fresh(int(request.param.removeprefix("seed")))


@pytest.fixture(scope="module")
def toml_data() -> dict[str, Any]:
    return tomllib.loads(FAMILIES_TOML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def steps_by_text() -> dict[tuple[str, str], str]:
    """(title, text) -> the library snippet's step ('' when it has none)."""
    out = {}
    for path in sorted(SNIPPET_DIR.glob("*.jsonl")):
        for row in _load_jsonl(path):
            out[(row["title"], row["text"])] = row.get("step") or ""
    return out


def _user_texts(row: dict[str, Any]) -> list[str]:
    return [m["content"] for m in row["messages"] if m["role"] == "user"]


# --- Q4: harsh-shift complaints only on automatic-transmission vehicles ------------------------


def _vehicle_of(row: dict[str, Any], vehicles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The catalog vehicle named in the first user message (body plus an engine name)."""
    first = _user_texts(row)[0]
    matches = [
        v
        for v in vehicles
        if v["body"] in first and any(re.search(rf"\b{re.escape(n)}", first) for n in v["names"])
    ]
    # Prefer the most specific engine name when one name is a prefix of another.
    matches.sort(key=lambda v: -max(len(n) for n in v["names"] if n in first))
    return matches[0] if matches else None


def _has_harsh_shift_complaint(row: dict[str, Any], complaints: list[str]) -> bool:
    if row.get("family") == "harsh_shift":
        return True
    text = " ".join(_user_texts(row)).lower()
    return any(c.lower() in text for c in complaints)


def test_harsh_shift_complaints_only_on_automatic_vehicles(
    rows: list[tuple[str, dict[str, Any]]], toml_data: dict[str, Any]
) -> None:
    vehicles = toml_data["vehicle"]
    (fam,) = [f for f in toml_data["family"] if f["name"] == "harsh_shift"]
    problems = []
    for where, row in rows:
        if not _has_harsh_shift_complaint(row, fam["complaints"]):
            continue
        vehicle = _vehicle_of(row, vehicles)
        if vehicle is None:
            problems.append(f"{where}/{row['id']}: no catalog vehicle found in the first message")
        elif vehicle["trans"] != "automatic":
            problems.append(
                f"{where}/{row['id']} ({row['category']}): harsh-shift complaint on a "
                f"{vehicle['trans']} {vehicle['body']}"
            )
    assert not problems, "\n".join(problems)


# --- Q5: "reading is in spec" only after a check that produces a reading -----------------------


def test_reading_step_pattern_classifies_library_steps(
    steps_by_text: dict[tuple[str, str], str],
) -> None:
    steps = {s for s in steps_by_text.values() if s}
    visual = [s for s in steps if not READING_STEP.search(s)]
    reading = [s for s in steps if READING_STEP.search(s)]
    # The validator's example (the belt inspection) is visual; a gauge check is a reading.
    assert any("inspect the accessory belt" in s for s in visual)
    assert any("fuel rail test port" in s for s in reading)


def test_reading_follow_up_never_follows_a_visual_check(
    rows: list[tuple[str, dict[str, Any]]], steps_by_text: dict[tuple[str, str], str]
) -> None:
    problems = []
    for where, row in rows:
        if row["category"] != "multi_turn":
            continue
        step_of = {s["id"]: steps_by_text.get((s["title"], s["text"]), "") for s in row["snippets"]}
        previous: str | None = None
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            content = message["content"]
            if previous is not None and READING_PHRASE.search(content):
                if not READING_STEP.search(previous):
                    problems.append(
                        f"{where}/{row['id']}: {content[:40]!r} follows visual check "
                        f"{previous[:60]!r}"
                    )
            cited = [c for c in CITATION.findall(content) if c in step_of]
            if cited:
                previous = step_of[cited[0]]
    assert not problems, "\n".join(problems)


# --- Q6: no "added a reference" phrasing, since the snippet list never changes -----------------


def test_phrasing_pools_have_no_added_reference_phrase() -> None:
    pools = tomllib.loads(PHRASING_TOML.read_text(encoding="utf-8"))
    hits = [
        f"{name}: {entry!r}"
        for name, entries in pools.items()
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, str) and ADDED_REFERENCE.search(entry)
    ]
    assert not hits, "\n".join(hits)


def test_no_scenario_says_a_reference_was_added(rows: list[tuple[str, dict[str, Any]]]) -> None:
    hits = [
        f"{where}/{row['id']}"
        for where, row in rows
        for text in _user_texts(row)
        if ADDED_REFERENCE.search(text)
    ]
    assert not hits, "\n".join(sorted(set(hits)))


@pytest.mark.parametrize(
    "text",
    [
        "I found one more reference and put it in with the others.",
        "I also dug up another service document and added it to the list.",
    ],
)
def test_added_reference_pattern_matches_the_flagged_wording(text: str) -> None:
    assert ADDED_REFERENCE.search(text)


# --- Q7: any-of gold for rows with two supporting IDs is documented ----------------------------


def test_two_supporting_id_rows_exist() -> None:
    multi = [r for _, r in _committed() if len(r["gold"]["supporting_ids"]) >= 2]
    assert multi, "no committed row has two supporting ids; this test guards nothing"


def _paragraphs(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))]


def test_docs_state_any_of_gold_for_two_supporting_ids() -> None:
    hits = [
        f"{path.name}: {p[:80]}"
        for path in DOCS
        for p in _paragraphs(path)
        if re.search(r"supporting[_ ]ids?", p, re.IGNORECASE)
        and re.search(r"\b(two|both|more than one|several|multiple)\b", p, re.IGNORECASE)
        and re.search(r"\b(any[- ]of|any one|either|one of (them|these|those|the))\b", p, re.I)
    ]
    assert hits, (
        "neither docs/scenario-coverage.md nor ARCHITECTURE.md says that citing any one of "
        "several supporting ids is correct"
    )
