"""Red tests for the rollout pilot's gold rule (hardening option H1, speculative).

``data/scenarios/rollout_pilot.jsonl`` is held to the any-of gold rule on every turn:

- ``supporting_ids`` is exactly the set of live snippets that are in scope for the vehicle and the
  complaint and whose step hasn't been performed or reported earlier in the conversation (the
  current turn's reported result included). Chain order is library metadata the model never sees,
  so it doesn't decide gold. The one exception, an order stated in the snippet texts themselves,
  never applies here: ``test_no_pilot_snippet_states_an_order`` keeps it that way.
- ``mode`` is ``abstain`` exactly when that set is empty.
- Adding a snippet never drops a still-undone, in-scope gold snippet.
- A withdrawn snippet is never gold.

The expected set is computed here, independently of the pilot builder: scope from the library's
snippet metadata (``applies``, ``requires`` and the family), the vehicle from the opener plus the
year corrections below, and done steps from the hand-labelled reported results below. Pilot
snippets are matched to library snippets by title and text, because the pilot re-IDs them.

Also: explicit regressions for every gold disagreement in the rollout validation report (17
turns, three of them dropped golds), rp-022's ambiguous crossover twin, and the pilot in
``scripts/check.sh``'s leakage checks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from groundcite.scenarios.library import Library, Snippet, Vehicle, allowed, load_library

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "scenarios" / "rollout_pilot.jsonl"
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"

LIB: Library = load_library()

# Steps each reported result performs, keyed by (conversation, turn index). Turn 0 is the opener
# turn, so an entry there with no ``reported_result`` is a step the opener already reports.
REPORTED_STEPS: dict[tuple[str, int], tuple[str, ...]] = {
    ("rp-001", 2): ("brake_pull.tire_check",),
    ("rp-001", 6): ("brake_pull.hose_drag",),
    ("rp-001", 9): ("brake_pull.rotor_temp",),
    ("rp-002", 4): ("brake_pull.slide_pins",),
    # Rotating the tires front to back is not the tire match and pressure check.
    ("rp-002", 8): (),
    ("rp-003", 2): ("brake_pull.slide_pins",),
    ("rp-003", 8): ("brake_pull.rotor_temp",),
    ("rp-004", 5): ("brake_pull.slide_pins",),
    ("rp-005", 1): ("brake_pull.tire_check",),
    ("rp-005", 7): ("brake_pull.slide_pins",),
    ("rp-006", 2): ("brake_pull.hose_drag",),
    ("rp-006", 9): ("brake_pull.slide_pins",),
    ("rp-007", 2): ("no_crank_click.battery_crank",),
    ("rp-007", 6): ("no_crank_click.cable_drop",),
    # The opener: "I already put a meter on the battery while my wife tried to start it".
    ("rp-008", 0): ("no_crank_click.battery_crank",),
    ("rp-008", 7): ("no_crank_click.s_terminal",),
    ("rp-009", 4): ("no_crank_click.battery_crank",),
    ("rp-009", 9): ("no_crank_click.s_terminal",),
    ("rp-010", 3): ("no_crank_click.battery_crank",),
    ("rp-011", 2): ("no_crank_click.battery_crank",),
    ("rp-011", 8): ("no_crank_click.cable_drop",),
    ("rp-012", 3): ("no_crank_click.s_terminal",),
    ("rp-013", 2): ("heater_cold.coolant_level",),
    ("rp-013", 5): ("heater_cold.thermostat",),
    ("rp-014", 2): ("heater_cold.blend_door",),
    ("rp-015", 2): ("heater_cold.thermostat",),
    ("rp-015", 5): ("heater_cold.hose_temps",),
    ("rp-016", 1): ("heater_cold.coolant_level",),
    ("rp-016", 6): ("heater_cold.thermostat",),
    ("rp-016", 8): ("heater_cold.blend_door",),
    ("rp-017", 3): ("heater_cold.coolant_level",),
    ("rp-017", 5): ("heater_cold.hose_temps",),
    # "The coolant is full and both heater hoses are hot": two steps reported at once.
    ("rp-018", 6): ("heater_cold.coolant_level", "heater_cold.hose_temps"),
    ("rp-018", 8): ("heater_cold.blend_door",),
    ("rp-019", 2): ("soft_pedal.fluid_leak",),
    ("rp-019", 6): ("soft_pedal.mc_bypass",),
    ("rp-020", 3): ("soft_pedal.moisture",),
    ("rp-021", 4): ("soft_pedal.fluid_leak",),
    ("rp-021", 7): ("soft_pedal.moisture",),
    ("rp-021", 9): ("soft_pedal.mc_bypass",),
    ("rp-022", 2): ("soft_pedal.fluid_leak",),
    ("rp-022", 6): ("soft_pedal.moisture",),
    ("rp-023", 5): ("soft_pedal.moisture",),
    ("rp-023", 8): ("soft_pedal.mc_bypass",),
    ("rp-024", 1): ("soft_pedal.fluid_leak",),
    ("rp-024", 4): ("soft_pedal.moisture",),
    ("rp-025", 3): ("highway_shake.radial_runout",),
    ("rp-025", 8): ("highway_shake.balance",),
    ("rp-026", 3): ("highway_shake.tread",),
    ("rp-027", 2): ("highway_shake.tread",),
    ("rp-027", 3): ("highway_shake.radial_runout",),
    ("rp-027", 9): ("highway_shake.balance",),
    ("rp-028", 5): ("highway_shake.radial_runout",),
    ("rp-028", 10): ("highway_shake.balance",),
    ("rp-029", 1): ("highway_shake.radial_runout",),
    ("rp-029", 6): ("highway_shake.tread",),
    ("rp-030", 3): ("highway_shake.balance",),
}

# Model-year corrections the user makes mid-conversation: (conversation, turn) -> new year.
YEAR_CORRECTIONS: dict[tuple[str, int], int] = {
    ("rp-004", 1): 2016,
    ("rp-008", 5): 2016,
    ("rp-029", 3): 2010,
}

# A snippet text that orders steps relative to each other ("only after confirming the fluid level,
# ..."), as opposed to ordering the actions inside its own step.
STATED_ORDER = re.compile(
    r"\b(only after|after (confirming|ruling out|checking|completing)|once (you have|the \w+ "
    r"(has|have) been)|before (moving on|any other)|do this (first|last)|step \d+ of)\b",
    re.IGNORECASE,
)

# Every gold disagreement in the rollout validation report: (conversation, turn, IDs the gold must
# include). Rows 8, 9 and 11 are golds dropped when a snippet was added.
VALIDATOR_DISAGREEMENTS: list[tuple[str, int, tuple[str, ...]]] = [
    ("rp-001", 0, ("SB 14-022", "TSB-3318")),
    ("rp-001", 1, ("SB 14-022", "TSB-3318")),
    ("rp-007", 0, ("SM 4.1.2", "TSB-2210")),
    ("rp-007", 1, ("SM 4.1.2", "TSB-2210")),
    ("rp-011", 0, ("TSB-1120", "TSB-0915")),
    ("rp-011", 1, ("TSB-1120", "TSB-0915")),
    ("rp-016", 0, ("TSB-5501", "TSB-5560")),
    ("rp-017", 2, ("SM 8.0.3", "SM 8.2.4")),
    ("rp-021", 3, ("B-16-200", "B-16-210", "B-16-212")),
    ("rp-024", 0, ("B-17-061", "B-17-077")),
    ("rp-025", 2, ("SB 19-450", "SB 19-440")),
    ("rp-027", 0, ("TSB-6610", "TSB-6614", "TSB-6620")),
    ("rp-027", 1, ("TSB-6610", "TSB-6614", "TSB-6620")),
    ("rp-027", 2, ("TSB-6614", "TSB-6620")),
    ("rp-029", 0, ("SM 4.2.6", "SM 4.2.2")),
    ("rp-030", 0, ("TSB-1601", "TSB-1602")),
    ("rp-030", 1, ("TSB-1601", "TSB-1602")),
]
DROPPED_GOLD: list[tuple[str, int, str]] = [
    ("rp-017", 2, "SM 8.2.4"),
    ("rp-021", 3, "B-16-210"),
    ("rp-021", 3, "B-16-212"),
    ("rp-025", 2, "SB 19-440"),
]


def _load_rows() -> list[dict[str, Any]]:
    lines = PILOT.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


ROWS = _load_rows()
BY_ID = {row["id"]: row for row in ROWS}
LIB_BY_TEXT: dict[tuple[str, str], Snippet] = {(s.title, s.text): s for s in LIB.snippets.values()}


def _lib(snippet: Mapping[str, Any]) -> Snippet:
    return LIB_BY_TEXT[(snippet["title"], snippet["text"])]


def _step(s: Snippet) -> str:
    """The step a snippet describes; a near-miss twin describes its original's step."""
    return s.twin_of or s.id


def _vehicle(row: Mapping[str, Any]) -> tuple[Vehicle, int]:
    """The vehicle and model year the opener states."""
    opener = row["opener"].lower()
    bodies = sorted({v.body for v in LIB.vehicles}, key=len, reverse=True)
    body = next(b for b in bodies if b.lower() in opener)
    displacement = re.search(r"\b(\d\.\d)", opener)
    year = re.search(r"\b(20[0-2]\d)\b", opener)
    assert displacement and year, f"{row['id']}: the opener doesn't state an engine and a year"
    matches = [
        v for v in LIB.vehicles if v.body == body and v.engine.startswith(displacement.group(1))
    ]
    assert len(matches) == 1, f"{row['id']}: {body} {displacement.group(1)} is not one vehicle"
    return matches[0], int(year.group(1))


def _in_scope(s: Snippet, family: str, vehicle: Vehicle, year: int) -> bool:
    """In scope for this complaint (the family and its stated facts) and this vehicle and year."""
    if s.family != family or not s.applies.matches(vehicle, year):
        return False
    facts = LIB.families[family].complaint_facts[0]
    return all(facts.get(k) in allowed(v) for k, v in s.requires.items())


@dataclass(frozen=True)
class TurnState:
    """The conversation's state as of a turn, after that turn's own user message."""

    index: int
    family: str
    vehicle: Vehicle
    live: Mapping[str, Snippet]
    withdrawn: frozenset[str]
    done: frozenset[str]
    year: int
    gold_mode: str
    gold_ids: frozenset[str]

    @property
    def expected(self) -> frozenset[str]:
        return frozenset(i for i, s in self.live.items() if self.undone_in_scope(s))

    def undone_in_scope(self, s: Snippet) -> bool:
        return _step(s) not in self.done and _in_scope(s, self.family, self.vehicle, self.year)


def _states(row: Mapping[str, Any]) -> Iterator[TurnState]:
    vehicle, year = _vehicle(row)
    live = {s["id"]: _lib(s) for s in row["snippets_initial"]}
    withdrawn: set[str] = set()
    done: set[str] = set()
    for i, turn in enumerate(row["turns"]):
        for r in turn.get("retract", []):
            live.pop(r["id"], None)
            withdrawn.add(r["id"])
        for s in turn.get("add", []):
            live[s["id"]] = _lib(s)
        year = YEAR_CORRECTIONS.get((row["id"], i), year)
        done.update(REPORTED_STEPS.get((row["id"], i), ()))
        gold = turn["gold"]
        yield TurnState(
            index=i,
            family=row["family"],
            vehicle=vehicle,
            live=dict(live),
            withdrawn=frozenset(withdrawn),
            done=frozenset(done),
            year=year,
            gold_mode=gold["mode"],
            gold_ids=frozenset(gold["supporting_ids"]),
        )


def _state(conv: str, turn: int) -> TurnState:
    return next(s for s in _states(BY_ID[conv]) if s.index == turn)


ALL_TURNS = [(row["id"], i) for row in ROWS for i in range(len(row["turns"]))]
TURN_IDS = [f"{c}-t{i:02d}" for c, i in ALL_TURNS]


# ---------------------------------------------------------------------------------------------
# The inputs of the independent derivation hold, so a gold failure below is a gold problem.
# ---------------------------------------------------------------------------------------------


def test_pilot_has_thirty_conversations() -> None:
    assert len(ROWS) == 30
    assert len(BY_ID) == 30


def test_every_pilot_snippet_is_a_library_snippet() -> None:
    missing = [
        f"{row['id']}: {s['id']}"
        for row in ROWS
        for s in [*row["snippets_initial"], *(a for t in row["turns"] for a in t.get("add", []))]
        if (s["title"], s["text"]) not in LIB_BY_TEXT
    ]
    assert not missing, f"snippets whose title and text match no library snippet: {missing}"


@pytest.mark.parametrize("conv", sorted(BY_ID))
def test_opener_states_one_catalog_vehicle(conv: str) -> None:
    vehicle, year = _vehicle(BY_ID[conv])
    assert vehicle.years[0] <= year <= vehicle.years[1]


def test_year_corrections_are_stated_in_the_turn_text() -> None:
    for (conv, turn), year in YEAR_CORRECTIONS.items():
        assert str(year) in BY_ID[conv]["turns"][turn].get("text", ""), (conv, turn)


def test_every_reported_result_has_its_steps_labelled() -> None:
    reported = {
        (row["id"], i)
        for row in ROWS
        for i, turn in enumerate(row["turns"])
        if "reported_result" in turn
    }
    unlabelled = sorted(reported - set(REPORTED_STEPS))
    assert not unlabelled, f"reported results missing from REPORTED_STEPS: {unlabelled}"
    stale = sorted(k for k in set(REPORTED_STEPS) - reported if k[1] != 0)
    assert not stale, f"REPORTED_STEPS entries with no reported result: {stale}"
    steps = {step for steps in REPORTED_STEPS.values() for step in steps}
    assert steps <= set(LIB.snippets), f"unknown steps: {sorted(steps - set(LIB.snippets))}"


def test_no_pilot_snippet_states_an_order() -> None:
    """Gold ignores step order because no snippet text states one. Adding such a text needs a
    rule for it here first."""
    texts = {
        s["id"]: s["text"]
        for row in ROWS
        for s in [*row["snippets_initial"], *(a for t in row["turns"] for a in t.get("add", []))]
    }
    ordered = sorted(i for i, text in texts.items() if STATED_ORDER.search(text))
    assert not ordered, f"snippets that state an order between steps: {ordered}"


# ---------------------------------------------------------------------------------------------
# The any-of gold rule, on every turn of every conversation.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("conv", "turn"), ALL_TURNS, ids=TURN_IDS)
def test_gold_is_every_live_in_scope_undone_snippet(conv: str, turn: int) -> None:
    state = _state(conv, turn)
    missing = sorted(state.expected - state.gold_ids)
    extra = sorted(state.gold_ids - state.expected)
    assert not missing and not extra, (
        f"{conv} turn {turn}: gold {sorted(state.gold_ids)} != expected "
        f"{sorted(state.expected)} (missing {missing}, not in scope or done {extra})"
    )


@pytest.mark.parametrize(("conv", "turn"), ALL_TURNS, ids=TURN_IDS)
def test_gold_abstains_iff_nothing_is_left_to_cite(conv: str, turn: int) -> None:
    state = _state(conv, turn)
    want = "cite" if state.expected else "abstain"
    assert state.gold_mode == want, (
        f"{conv} turn {turn}: mode {state.gold_mode!r}, expected {want!r} "
        f"(live, in scope and undone: {sorted(state.expected)})"
    )
    assert (state.gold_mode == "abstain") == (not state.gold_ids)


ADD_TURNS = [
    (row["id"], i)
    for row in ROWS
    for i, turn in enumerate(row["turns"])
    if i > 0 and turn.get("add")
]


@pytest.mark.parametrize(("conv", "turn"), ADD_TURNS, ids=[f"{c}-t{i:02d}" for c, i in ADD_TURNS])
def test_adding_a_snippet_never_drops_an_undone_gold(conv: str, turn: int) -> None:
    before, now = _state(conv, turn - 1), _state(conv, turn)
    still_due = {i for i in before.gold_ids if i in now.live and now.undone_in_scope(now.live[i])}
    dropped = sorted(still_due - now.gold_ids)
    assert not dropped, f"{conv} turn {turn}: the add dropped still-undone gold {dropped}"


@pytest.mark.parametrize("conv", sorted(BY_ID))
def test_withdrawn_snippets_are_never_gold(conv: str) -> None:
    for state in _states(BY_ID[conv]):
        assert not state.gold_ids & state.withdrawn, (conv, state.index)
        assert state.gold_ids <= set(state.live), (conv, state.index)


# ---------------------------------------------------------------------------------------------
# Regressions from the rollout validation report.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conv", "turn", "ids"),
    VALIDATOR_DISAGREEMENTS,
    ids=[f"{c}-t{t:02d}" for c, t, _ in VALIDATOR_DISAGREEMENTS],
)
def test_validator_disagreement_accepts_every_undone_step(
    conv: str, turn: int, ids: tuple[str, ...]
) -> None:
    gold = BY_ID[conv]["turns"][turn]["gold"]
    assert gold["mode"] == "cite"
    missing = sorted(set(ids) - set(gold["supporting_ids"]))
    assert not missing, f"{conv} turn {turn}: gold {gold['supporting_ids']} leaves out {missing}"


@pytest.mark.parametrize(
    ("conv", "turn", "snippet_id"),
    DROPPED_GOLD,
    ids=[f"{c}-t{t:02d}-{s}" for c, t, s in DROPPED_GOLD],
)
def test_previously_gold_snippet_stays_gold_after_an_add(
    conv: str, turn: int, snippet_id: str
) -> None:
    turns = BY_ID[conv]["turns"]
    assert snippet_id in turns[turn - 1]["gold"]["supporting_ids"], "precondition"
    assert turns[turn].get("add"), "precondition: the turn adds a snippet"
    assert snippet_id in turns[turn]["gold"]["supporting_ids"]


def _twin_pairs(state: TurnState) -> list[tuple[str, str]]:
    by_lib = {s.id: i for i, s in state.live.items()}
    return [
        (by_lib[s.twin_of], i)
        for i, s in state.live.items()
        if s.twin_of is not None and s.twin_of in by_lib
    ]


def test_rp022_vehicle_puts_exactly_one_of_each_twin_pair_in_scope() -> None:
    """The opener's body must decide every live twin pair: one twin in scope, the other out."""
    row = BY_ID["rp-022"]
    vehicle, _ = _vehicle(row)
    pairs = 0
    for state in _states(row):
        for original, twin in _twin_pairs(state):
            pairs += 1
            in_scope = [
                i
                for i in (original, twin)
                if _in_scope(state.live[i], row["family"], vehicle, state.year)
            ]
            assert len(in_scope) == 1, (
                f"rp-022 turn {state.index}: {original} / {twin} has {len(in_scope)} in scope "
                f"for a {state.year} {vehicle.body} {vehicle.engine}"
            )
    assert pairs, "rp-022 should keep a live twin pair"


def test_rp022_out_of_scope_snippets_do_not_name_the_vehicle_body() -> None:
    """A twin ruled out by body must not name the vehicle's body class ("compact crossovers"
    against a small crossover reads as covering it)."""
    row = BY_ID["rp-022"]
    vehicle, _ = _vehicle(row)
    noun = vehicle.body.split()[-1].lower()
    for state in _states(row):
        for i, s in state.live.items():
            if s.family == row["family"] and not _in_scope(s, row["family"], vehicle, state.year):
                assert noun not in s.text.lower(), (
                    f"rp-022 turn {state.index}: out-of-scope {i} names {noun!r}"
                )


# ---------------------------------------------------------------------------------------------
# The quality gate runs the leakage check on the pilot.
# ---------------------------------------------------------------------------------------------


def _leakage_commands() -> list[str]:
    """check.sh's ``check_leakage.py`` invocations, with continuation lines joined."""
    commands, current = [], ""
    for raw in CHECK_SH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        current += " " + line.rstrip("\\").strip()
        if not line.endswith("\\"):
            commands.append(current.strip())
            current = ""
    return [c for c in commands if "check_leakage.py" in c]


def test_check_sh_checks_the_pilot_against_the_training_pool() -> None:
    hits = [
        c
        for c in _leakage_commands()
        if "--train data/scenarios/train_pool.jsonl" in c and "rollout_pilot.jsonl" in c
    ]
    assert hits, "check.sh never runs check_leakage.py on the pilot against train_pool"


def test_check_sh_checks_the_pilot_against_the_few_shot_examples() -> None:
    hits = [c for c in _leakage_commands() if "--few-shot" in c and "rollout_pilot.jsonl" in c]
    assert hits, "check.sh never runs check_leakage.py on the pilot against the few-shot examples"


def test_check_sh_checks_the_pilot_against_every_training_set() -> None:
    hits = [
        c for c in _leakage_commands() if '--train "$train"' in c and "rollout_pilot.jsonl" in c
    ]
    assert hits, "check.sh's training-set loop doesn't include the pilot"
