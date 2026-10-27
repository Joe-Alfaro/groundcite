"""Tests for multi-turn chain order and platform-identifying vehicle details (R22 gold fixes).

Multi-turn conversations walk through a family's diagnostic chain: the assistant asks for a
check, the user reports the result, and the gold label is the next step. A gold step that an
earlier reported result already rules out is a gold error. To make "already ruled out"
checkable, every family in data/families.toml declares ``chain_order``: an ordered list of
stages, each stage a list of that family's chain snippet ids. Steps in one stage are
independent; a step in a later stage only makes sense once the earlier stages are done, and a
reported result from a later stage makes every earlier-stage step moot.

The committed sets are read as JSONL; scenario snippets are mapped back to library snippets by
(title, text), since scenario ids are re-assigned per scenario.
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

SET_NAMES = ("ablation", "eval_own", "heldout")
CITATION = re.compile(r"\[([^\[\]]+)\]")

# Domain facts this test holds independently of the data: a reported result of the key step
# makes the listed steps moot. Normal AC operating pressures show the compressor is engaged and
# pumping and the condenser is shedding heat, so the clutch and condenser fan checks are done.
MOOTED_BY = {
    "ac_warm.pressures": ("ac_warm.clutch", "ac_warm.condenser_fan"),
}

# Detail combinations that pin a specific real platform without naming it. Each entry is a set
# of patterns that must not all appear in one text.
PLATFORM_DENYLIST: tuple[tuple[str, ...], ...] = (
    (r"5\.3[- ]?(?:L|liter)\b.{0,10}V8", r"\b10[- ]speed\b"),
    (r"fuel pump control module", r"under the cargo bed"),
)
DISPLACEMENT = re.compile(r"\b\d\.\d[- ]?(?:L|liter)\b", re.IGNORECASE)
SPEED_COUNT = re.compile(r"\b\d{1,2}[- ]speed\b", re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def library() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(SNIPPET_DIR.glob("*.jsonl")):
        for row in _load_jsonl(path):
            rows[row["id"]] = row
    return rows


@pytest.fixture(scope="module")
def by_text(library: dict[str, dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {(row["title"], row["text"]): snippet_id for snippet_id, row in library.items()}


@pytest.fixture(scope="module")
def families() -> dict[str, dict[str, Any]]:
    raw = tomllib.loads(FAMILIES_TOML.read_text(encoding="utf-8"))
    return {fam["name"]: fam for fam in raw["family"]}


@pytest.fixture(scope="module")
def vehicles() -> list[dict[str, Any]]:
    return list(tomllib.loads(FAMILIES_TOML.read_text(encoding="utf-8"))["vehicle"])


@pytest.fixture(scope="module")
def sets() -> dict[str, list[dict[str, Any]]]:
    return {name: _load_jsonl(SCENARIO_DIR / f"{name}.jsonl") for name in SET_NAMES}


def _stages(family: dict[str, Any]) -> dict[str, int]:
    """Chain snippet id -> stage index, from the family's ``chain_order``."""
    order = family.get("chain_order")
    assert order is not None, f"{family['name']}: no chain_order declared in families.toml"
    return {snippet_id: i for i, stage in enumerate(order) for snippet_id in stage}


def _library_ids(row: dict[str, Any], by_text: dict[tuple[str, str], str]) -> dict[str, str]:
    """Scenario snippet id -> library snippet id."""
    return {s["id"]: by_text[(s["title"], s["text"])] for s in row["snippets"]}


def _reported_steps(row: dict[str, Any], to_lib: dict[str, str]) -> list[str]:
    """Library ids of the checks the assistant asked for earlier, in conversation order.

    Every assistant turn before the final user message asks for one check, and the user's next
    message reports its result.
    """
    steps: list[str] = []
    for message in row["messages"]:
        if message["role"] != "assistant":
            continue
        cited = [c for c in CITATION.findall(message["content"]) if c in to_lib]
        assert len(cited) == 1, f"{row['id']}: assistant turn cites {cited}"
        if not steps or steps[-1] != to_lib[cited[0]]:  # a held turn repeats the same check
            steps.append(to_lib[cited[0]])
    return steps


def _multi_turn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["category"] == "multi_turn"]


# --- (1) chain-order metadata and the general multi-turn invariant ---------------------------


def test_every_family_declares_chain_order(families: dict[str, dict[str, Any]]) -> None:
    problems = []
    for name, fam in families.items():
        order = fam.get("chain_order")
        if order is None:
            problems.append(f"{name}: missing chain_order")
            continue
        if not isinstance(order, list) or not all(isinstance(s, list) and s for s in order):
            problems.append(f"{name}: chain_order must be a list of non-empty lists")
            continue
        flat = [snippet_id for stage in order for snippet_id in stage]
        if len(flat) != len(set(flat)):
            problems.append(f"{name}: chain_order lists a snippet twice")
        if set(flat) != set(fam["chain"]):
            problems.append(f"{name}: chain_order {sorted(flat)} != chain {sorted(fam['chain'])}")
        if len(order) < 2:
            problems.append(f"{name}: chain_order needs at least 2 stages to order a conversation")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("set_name", SET_NAMES)
def test_multi_turn_gold_is_next_unchecked_step(
    set_name: str,
    sets: dict[str, list[dict[str, Any]]],
    families: dict[str, dict[str, Any]],
    by_text: dict[tuple[str, str], str],
) -> None:
    problems = []
    for row in _multi_turn(sets[set_name]):
        fam = families[row["family"]]
        stage = _stages(fam)
        to_lib = _library_ids(row, by_text)
        reported = _reported_steps(row, to_lib)
        where = f"{row['id']} ({row['family']}, reported {reported})"

        if any(stage[a] > stage[b] for a, b in zip(reported, reported[1:], strict=False)):
            problems.append(f"{where}: earlier turns go backwards in chain_order")
        done = max(stage[s] for s in reported)
        # Provided chain steps that are neither checked yet nor made moot by a reported result.
        open_steps = {
            lib_id
            for lib_id in to_lib.values()
            if lib_id in stage and lib_id not in reported and stage[lib_id] >= done
        }
        gold = [to_lib[g] for g in row["gold"]["supporting_ids"]]
        if row["gold"]["mode"] == "cite":
            for g in gold:
                if g in reported:
                    problems.append(f"{where}: gold {g} was already checked")
                elif stage.get(g, -1) < done:
                    problems.append(f"{where}: gold {g} is moot after the reported results")
                elif any(stage[o] < stage[g] for o in open_steps):
                    problems.append(f"{where}: gold {g} skips an earlier open step {open_steps}")
        elif open_steps:
            problems.append(f"{where}: abstain gold but open next step {open_steps} is provided")
    assert not problems, "\n".join(problems)


# --- (2) regression: ac_warm gold cited the clutch check after normal pressures --------------


def test_ac_warm_chain_order_puts_pressures_last(families: dict[str, dict[str, Any]]) -> None:
    stage = _stages(families["ac_warm"])
    for key, mooted in MOOTED_BY.items():
        for step in mooted:
            assert stage[step] < stage[key], f"ac_warm: {step} must come before {key}"


@pytest.mark.parametrize("set_name", SET_NAMES)
def test_no_gold_step_is_moot_after_a_reported_result(
    set_name: str,
    sets: dict[str, list[dict[str, Any]]],
    by_text: dict[tuple[str, str], str],
) -> None:
    """Holds without chain_order: e.g. no clutch check after normal AC pressures were reported."""
    problems = []
    for row in _multi_turn(sets[set_name]):
        to_lib = _library_ids(row, by_text)
        reported = _reported_steps(row, to_lib)
        if row["gold"]["mode"] != "cite":
            continue
        for g in (to_lib[i] for i in row["gold"]["supporting_ids"]):
            for key in reported:
                if g in MOOTED_BY.get(key, ()):
                    problems.append(f"{row['id']} ({row['family']}): gold {g} is moot after {key}")
    assert not problems, "\n".join(problems)


# --- (3) vehicle details must not pin a specific real platform (Q5) --------------------------


def _texts(
    library: dict[str, dict[str, Any]],
    vehicles: list[dict[str, Any]],
    sets: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, str]]:
    """(where, text) for every vehicle description: library snippets, catalog, scenario rows."""
    out = [(f"snippet {k}", f"{v['title']}. {v['text']}") for k, v in library.items()]
    out += [(f"vehicle {v['body']} {v['engine']}", json.dumps(v)) for v in vehicles]
    for set_name, rows in sets.items():
        for row in rows:
            for s in row["snippets"]:
                out.append((f"{set_name}/{row['id']} snippet", f"{s['title']}. {s['text']}"))
            for i, m in enumerate(row["messages"]):
                out.append((f"{set_name}/{row['id']} message {i}", m["content"]))
    return out


def test_no_denylisted_platform_combo(
    library: dict[str, dict[str, Any]],
    vehicles: list[dict[str, Any]],
    sets: dict[str, list[dict[str, Any]]],
) -> None:
    hits = sorted(
        {
            f"{where}: {' + '.join(combo)}"
            for where, text in _texts(library, vehicles, sets)
            for combo in PLATFORM_DENYLIST
            if all(re.search(p, text, re.IGNORECASE) for p in combo)
        }
    )
    assert not hits, "\n".join(hits)


def test_no_displacement_with_transmission_speed_count(
    library: dict[str, dict[str, Any]],
    vehicles: list[dict[str, Any]],
    sets: dict[str, list[dict[str, Any]]],
) -> None:
    hits = sorted(
        {
            where
            for where, text in _texts(library, vehicles, sets)
            if DISPLACEMENT.search(text) and SPEED_COUNT.search(text)
        }
    )
    assert not hits, "\n".join(hits)
