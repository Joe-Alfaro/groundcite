"""No vehicle or bulletin detail may pin a specific real platform (R22 scrub; validator note Q1).

Generic vehicles are fine; a combination of displacement, body, transmission and year range that
points at one manufacturer's line is not, and neither is a bulletin that echoes a widely reported
real issue on such a platform. Checked on the snippet library, the vehicle catalog and every
committed scenario set:

- a library snippet is read together with every catalog vehicle it applies to (ignoring years),
  since "Applies to minivans with rear auxiliary heat" pins a platform when the only minivan in
  the catalog has a 3.5L V6;
- a scenario row is read as one text (all snippets and messages), since a reader sees them
  together.

This extends ``PLATFORM_DENYLIST`` in ``tests/test_scenario_chains.py``.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
SNIPPET_DIR = REPO_ROOT / "data" / "snippets"
FAMILIES_TOML = REPO_ROOT / "data" / "families.toml"

V8_53 = r"\b5\.3[- ]?(?:L|liter)\b.{0,10}\bV8\b"
TURBO_15 = r"\b1\.5[- ]?(?:L|liter)\b.{0,10}\bturbo\b"
V6_36 = r"\b3\.6[- ]?(?:L|liter)\b.{0,10}\bV6\b"
V6_35 = r"\b3\.5[- ]?(?:L|liter)\b.{0,10}\bV6\b"

# Each entry is a set of patterns that must not all appear in one text.
PLATFORM_COMBOS: dict[str, tuple[str, ...]] = {
    "5.3L V8 full-size pickup": (V8_53, r"\bfull[- ]size pickups?\b"),
    "1.5L turbo CVT crossover": (TURBO_15, r"\bCVT\b", r"\bcrossovers?\b"),
    "3.6L V6 coolant crossover": (V6_36, r"\b(?:coolant|intake manifold) crossover\b"),
    "3.5L V6 minivan rear auxiliary heat": (
        V6_35,
        r"\bminivans?\b",
        r"\brear auxiliary heat(?:er)?\b",
    ),
}

# Widely reported real issues on those platforms. Each must not appear in any single text.
REAL_ISSUE_ECHOES: dict[str, tuple[str, ...]] = {
    "collapsed lifter": (r"\bcollapsed (?:valve )?lifters?\b",),
    "fuel pump driver module relocation": (
        r"\bfuel pump (?:driver|control) module\b",
        r"\b(?:relocat\w*|heat[- ]soak\w*|overheat\w*)\b",
    ),
    "telematics after a cellular network change": (r"\btelematics\b", r"\bcellular\b"),
    "LSPI 2017-2019": (
        r"\b(?:low[- ]speed pre-?ignition|pre-?ignition|LSPI)\b",
        r"\b2017\s*[-–]\s*2019\b",
    ),
}

SET_PATHS = sorted(SCENARIO_DIR.glob("*.jsonl"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _library() -> list[dict[str, Any]]:
    return [row for path in sorted(SNIPPET_DIR.glob("*.jsonl")) for row in _load_jsonl(path)]


def _vehicles() -> list[dict[str, Any]]:
    return list(tomllib.loads(FAMILIES_TOML.read_text(encoding="utf-8"))["vehicle"])


def _vehicle_text(vehicle: dict[str, Any]) -> str:
    names = ", ".join(vehicle.get("names", []))
    years = vehicle.get("years", ["", ""])
    return (
        f"{vehicle['body']} {vehicle['engine']} ({names}) {vehicle['trans']} {years[0]}-{years[1]}"
    )


def _applies_to(applies: dict[str, Any], vehicle: dict[str, Any]) -> bool:
    """Structural applicability, ignoring model years."""
    for key, field in (("engines", "engine"), ("bodies", "body"), ("trans", "trans")):
        allowed = applies.get(key)
        if allowed is not None and vehicle[field] not in allowed:
            return False
    return True


def _snippet_text(row: dict[str, Any]) -> str:
    return f"{row.get('title', '')}. {row.get('text', '')}"


def _hits(where: str, text: str, table: dict[str, tuple[str, ...]]) -> list[str]:
    return [
        f"{where}: {label}"
        for label, patterns in table.items()
        if all(re.search(p, text, re.IGNORECASE) for p in patterns)
    ]


def test_scenario_sets_are_present() -> None:
    names = {p.stem for p in SET_PATHS}
    assert {"ablation", "eval_own", "heldout"} <= names, names


def test_no_catalog_vehicle_pins_a_platform() -> None:
    hits = [
        hit
        for v in _vehicles()
        for hit in _hits(f"vehicle {v['body']} {v['engine']}", _vehicle_text(v), PLATFORM_COMBOS)
    ]
    assert not hits, "\n".join(hits)


def test_no_library_snippet_pins_a_platform_with_the_vehicles_it_applies_to() -> None:
    vehicles = _vehicles()
    hits = []
    for row in _library():
        matching = [v for v in vehicles if _applies_to(row.get("applies") or {}, v)]
        text = " ".join([_snippet_text(row), *(_vehicle_text(v) for v in matching)])
        hits += _hits(f"snippet {row['id']}", text, PLATFORM_COMBOS)
    assert not hits, "\n".join(hits)


def test_no_library_snippet_echoes_a_real_platform_issue() -> None:
    hits = [
        hit
        for row in _library()
        for hit in _hits(f"snippet {row['id']}", _snippet_text(row), REAL_ISSUE_ECHOES)
    ]
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("path", SET_PATHS, ids=lambda p: p.name)
def test_no_scenario_row_pins_a_platform(path: Path) -> None:
    hits = []
    for row in _load_jsonl(path):
        parts = [_snippet_text(s) for s in row.get("snippets", [])]
        parts += [m["content"] for m in row.get("messages", [])]
        hits += _hits(f"{path.stem}/{row['id']}", " ".join(parts), PLATFORM_COMBOS)
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("path", SET_PATHS, ids=lambda p: p.name)
def test_no_scenario_text_echoes_a_real_platform_issue(path: Path) -> None:
    hits = []
    for row in _load_jsonl(path):
        texts = [(f"snippet {s['id']}", _snippet_text(s)) for s in row.get("snippets", [])]
        texts += [(f"message {i}", m["content"]) for i, m in enumerate(row.get("messages", []))]
        for where, text in texts:
            hits += _hits(f"{path.stem}/{row['id']} {where}", text, REAL_ISSUE_ECHOES)
    assert not hits, "\n".join(sorted(set(hits)))


# --- the patterns themselves: guard against a denylist that can never match --------------------


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Applies to full-size pickups with the 5.3L V8, model years 2022-2024.", "5.3L V8"),
        ("small crossover 1.5L turbo I4 (1.5L turbo) CVT 2017-2022", "1.5L turbo"),
        ("Applies to 3.6L V6 engines. Intake manifold gasket at the coolant crossover.", "3.6L"),
        ("Applies to minivans with rear auxiliary heat. minivan 3.5L V6 automatic", "minivan"),
    ],
)
def test_platform_combos_match_the_flagged_wording(text: str, label: str) -> None:
    assert [h for h in _hits("probe", text, PLATFORM_COMBOS) if label in h]


@pytest.mark.parametrize(
    "text",
    [
        "A misfire code with a ticking noise at idle may indicate a collapsed valve lifter.",
        "The electronic fuel pump driver module can overheat. Repair: relocate the module.",
        "The telematics module may stay awake after a cellular network change.",
        "Applies to 1.5L turbo I4 engines, model years 2017-2019. Low-speed pre-ignition.",
    ],
)
def test_echo_patterns_match_the_flagged_wording(text: str) -> None:
    assert _hits("probe", text, REAL_ISSUE_ECHOES)
