"""Tests for the seeded scenario generator and the committed scenario sets (R22).

The committed sets are read as JSONL in the scenario format from ARCHITECTURE.md, so these
tests don't depend on the package's Python types.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_scenarios.py"
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
SNIPPET_DIR = REPO_ROOT / "data" / "snippets"
RESERVED_FAMILIES = REPO_ROOT / "data" / "reserved_families.txt"
COVERAGE_DOC = REPO_ROOT / "docs" / "scenario-coverage.md"

SET_NAMES = ("ablation", "eval_own", "heldout")
MIN_SIZES = {"ablation": 40, "eval_own": 80, "heldout": 60}

# Scenario mix from the requirements file, in percent. Each set must be within +-5 points.
QUOTAS = {
    "clean": 15,
    "distractor": 20,
    "no_snippet": 20,
    "pressure": 15,
    "multi_symptom": 15,
    "multi_turn": 15,
}
QUOTA_TOLERANCE = 5
ID_PATTERN = re.compile(r"^[A-Za-z0-9._ -]{2,24}$")
MAX_SYSTEMS = 8

# Common car-maker names. Ambiguous ordinary words (seat, mini, ram, smart, saturn) are left out
# so the check doesn't fire on normal prose.
MANUFACTURER_DENYLIST = (
    "Toyota",
    "Honda",
    "Ford",
    "Chevrolet",
    "Chevy",
    "Nissan",
    "Hyundai",
    "Kia",
    "Subaru",
    "Mazda",
    "Volkswagen",
    "VW",
    "BMW",
    "Mercedes",
    "Audi",
    "Lexus",
    "Acura",
    "Infiniti",
    "Dodge",
    "Chrysler",
    "Jeep",
    "Buick",
    "Cadillac",
    "GMC",
    "Tesla",
    "Volvo",
    "Porsche",
    "Mitsubishi",
    "Fiat",
    "Peugeot",
    "Renault",
    "Citroen",
    "Skoda",
    "Suzuki",
    "Jaguar",
    "Land Rover",
    "Range Rover",
    "Pontiac",
    "Opel",
    "Holden",
)
DENYLIST_RE = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in MANUFACTURER_DENYLIST) + r")\b",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - failure message only
                pytest.fail(f"{path}:{lineno}: invalid JSON: {exc}")
    return rows


def _load_set(name: str) -> list[dict[str, Any]]:
    path = SCENARIO_DIR / f"{name}.jsonl"
    assert path.is_file(), f"committed scenario set {path.relative_to(REPO_ROOT)} is missing"
    rows = _read_jsonl(path)
    assert rows, f"{path.relative_to(REPO_ROOT)} is empty"
    return rows


def _load_library() -> list[dict[str, Any]]:
    assert SNIPPET_DIR.is_dir(), f"snippet library {SNIPPET_DIR.relative_to(REPO_ROOT)} is missing"
    files = sorted(SNIPPET_DIR.glob("*.jsonl"))
    assert files, f"no *.jsonl files in {SNIPPET_DIR.relative_to(REPO_ROOT)}"
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(_read_jsonl(path))
    assert rows, "snippet library is empty"
    return rows


def _load_reserved() -> set[str]:
    assert RESERVED_FAMILIES.is_file(), (
        f"reserved-families list {RESERVED_FAMILIES.relative_to(REPO_ROOT)} is missing"
    )
    names = set()
    for line in RESERVED_FAMILIES.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    assert names, "reserved-families list is empty"
    return names


def _id_scheme(snippet_id: str) -> str:
    """The shape of an ID with every digit run collapsed, e.g. 'SM 6.2.1' -> 'SM #.#.#'."""
    return re.sub(r"[0-9]+", "#", snippet_id)


def _run_gen(seed: int, out: Path) -> subprocess.CompletedProcess[str]:
    assert GEN_SCRIPT.is_file(), f"{GEN_SCRIPT.relative_to(REPO_ROOT)} is missing"
    return subprocess.run(
        [sys.executable, str(GEN_SCRIPT), "--seed", str(seed), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


# --- generator -----------------------------------------------------------------------------


def test_generator_writes_the_three_sets(tmp_path: Path) -> None:
    result = _run_gen(7, tmp_path)
    assert result.returncode == 0, result.stderr
    for name in SET_NAMES:
        path = tmp_path / f"{name}.jsonl"
        assert path.is_file(), f"generator did not write {name}.jsonl"
        assert _read_jsonl(path), f"generated {name}.jsonl is empty"


def test_generator_is_byte_identical_for_a_seed(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert _run_gen(7, first).returncode == 0
    assert _run_gen(7, second).returncode == 0
    for name in SET_NAMES:
        a = (first / f"{name}.jsonl").read_bytes()
        b = (second / f"{name}.jsonl").read_bytes()
        assert a == b, f"{name}.jsonl differs between two runs with the same seed"


def test_generator_output_depends_on_seed(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert _run_gen(7, first).returncode == 0
    assert _run_gen(8, second).returncode == 0
    differs = [
        name
        for name in SET_NAMES
        if (first / f"{name}.jsonl").read_bytes() != (second / f"{name}.jsonl").read_bytes()
    ]
    assert differs, "seeds 7 and 8 produced identical output"


# --- snippet library and families ----------------------------------------------------------


def test_snippet_library_rows_have_required_fields() -> None:
    for row in _load_library():
        for field in ("id", "kind", "system", "text", "family"):
            assert row.get(field), f"library snippet missing {field!r}: {row}"


def test_snippet_library_has_no_manufacturer_names() -> None:
    _load_library()
    for path in sorted(SNIPPET_DIR.glob("*.jsonl")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            assert DENYLIST_RE.search(line) is None, f"{path.name}:{lineno} names a manufacturer"


def test_reserved_families_exist_in_library_and_are_about_thirty_percent() -> None:
    reserved = _load_reserved()
    library_families = {row["family"] for row in _load_library()}
    assert reserved <= library_families, f"unknown reserved families: {reserved - library_families}"
    share = len(reserved) / len(library_families)
    assert 0.2 <= share <= 0.45, f"reserved share {share:.0%} is not roughly 30%"


def test_scenario_snippet_texts_come_from_the_library() -> None:
    library_texts = {row["text"] for row in _load_library()}
    for name in SET_NAMES:
        for scenario in _load_set(name):
            for snippet in scenario["snippets"]:
                assert snippet["text"] in library_texts, (
                    f"{name} {scenario['id']}: snippet text not in the library"
                )


def test_heldout_uses_only_reserved_families() -> None:
    reserved = _load_reserved()
    for scenario in _load_set("heldout"):
        assert scenario["family"] in reserved, f"heldout {scenario['id']} uses a training family"


@pytest.mark.parametrize("name", ["ablation", "eval_own"])
def test_own_sets_use_no_reserved_family(name: str) -> None:
    reserved = _load_reserved()
    for scenario in _load_set(name):
        assert scenario["family"] not in reserved, f"{name} {scenario['id']} uses a reserved family"


# --- set sizes and relations ---------------------------------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_set_size_and_unique_ids(name: str) -> None:
    rows = _load_set(name)
    assert len(rows) >= MIN_SIZES[name], f"{name} has {len(rows)} < {MIN_SIZES[name]} scenarios"
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), f"{name} has duplicate scenario ids"


def test_eval_own_is_superset_of_ablation() -> None:
    ablation = {row["id"] for row in _load_set("ablation")}
    eval_own = {row["id"] for row in _load_set("eval_own")}
    assert ablation <= eval_own, (
        f"ablation ids missing from eval_own: {sorted(ablation - eval_own)}"
    )


def test_heldout_ids_disjoint_from_eval_own() -> None:
    heldout = {row["id"] for row in _load_set("heldout")}
    eval_own = {row["id"] for row in _load_set("eval_own")}
    assert not heldout & eval_own


# --- scenario mix --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_category_quotas(name: str) -> None:
    rows = _load_set(name)
    counts = Counter(row["category"] for row in rows)
    assert set(counts) <= set(QUOTAS), f"unknown categories: {set(counts) - set(QUOTAS)}"
    for category, target in QUOTAS.items():
        share = 100 * counts[category] / len(rows)
        assert abs(share - target) <= QUOTA_TOLERANCE, (
            f"{name}: {category} is {share:.1f}%, target {target}% +-{QUOTA_TOLERANCE}"
        )


@pytest.mark.parametrize("name", SET_NAMES)
def test_no_snippet_plus_pressure_at_least_thirty_percent(name: str) -> None:
    rows = _load_set(name)
    hard = sum(1 for row in rows if row["category"] in ("no_snippet", "pressure"))
    assert hard / len(rows) >= 0.30, f"{name}: no_snippet+pressure is {hard / len(rows):.0%}"


# --- snippets and IDs ----------------------------------------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_two_to_five_snippets_with_valid_unique_ids(name: str) -> None:
    for scenario in _load_set(name):
        snippets = scenario["snippets"]
        assert 2 <= len(snippets) <= 5, f"{scenario['id']} has {len(snippets)} snippets"
        ids = [snippet["id"] for snippet in snippets]
        assert len(ids) == len(set(ids)), f"{scenario['id']} repeats a snippet id"
        for snippet_id in ids:
            assert ID_PATTERN.match(snippet_id), f"{scenario['id']}: bad id {snippet_id!r}"


@pytest.mark.parametrize("name", SET_NAMES)
def test_at_least_three_id_schemes(name: str) -> None:
    schemes = {
        _id_scheme(snippet["id"])
        for scenario in _load_set(name)
        for snippet in scenario["snippets"]
    }
    assert len(schemes) >= 3, f"{name} uses only {len(schemes)} ID schemes: {sorted(schemes)}"


@pytest.mark.parametrize("name", SET_NAMES)
def test_snippet_ids_are_reassigned_per_scenario(name: str) -> None:
    ids_by_text: dict[str, set[str]] = defaultdict(set)
    uses_by_text: Counter[str] = Counter()
    for scenario in _load_set(name):
        for snippet in scenario["snippets"]:
            ids_by_text[snippet["text"]].add(snippet["id"])
            uses_by_text[snippet["text"]] += 1
    reused = [text for text, uses in uses_by_text.items() if uses >= 2]
    assert reused, f"{name}: no snippet text is reused, so ID reassignment can't be shown"
    varied = [text for text in reused if len(ids_by_text[text]) >= 2]
    assert len(varied) / len(reused) >= 0.5, (
        f"{name}: only {len(varied)}/{len(reused)} reused snippet texts get different IDs"
    )


@pytest.mark.parametrize("name", SET_NAMES)
def test_supporting_snippet_position_varies(name: str) -> None:
    positions: Counter[int] = Counter()
    for scenario in _load_set(name):
        if scenario["gold"]["mode"] != "cite":
            continue
        ids = [snippet["id"] for snippet in scenario["snippets"]]
        positions[ids.index(scenario["gold"]["supporting_ids"][0])] += 1
    total = sum(positions.values())
    assert total, f"{name} has no cite scenarios"
    assert len(positions) >= 3, f"{name}: supporting snippet only at positions {sorted(positions)}"
    top = max(positions.values()) / total
    assert top <= 0.6, f"{name}: {top:.0%} of supporting snippets sit at one position"


def test_at_most_eight_vehicle_systems_listed_in_coverage_doc() -> None:
    systems = {
        snippet["system"]
        for name in SET_NAMES
        for scenario in _load_set(name)
        for snippet in scenario["snippets"]
    }
    systems |= {row["system"] for row in _load_library()}
    assert len(systems) <= MAX_SYSTEMS, f"{len(systems)} systems: {sorted(systems)}"
    assert COVERAGE_DOC.is_file(), f"{COVERAGE_DOC.relative_to(REPO_ROOT)} is missing"
    table_rows = [
        line for line in COVERAGE_DOC.read_text().splitlines() if line.strip().startswith("|")
    ]
    for system in systems:
        assert any(system in row for row in table_rows), (
            f"system {system!r} has no row in the coverage table"
        )


# --- gold labels ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_gold_labels_are_consistent(name: str) -> None:
    for scenario in _load_set(name):
        gold = scenario["gold"]
        provided = {snippet["id"] for snippet in scenario["snippets"]}
        assert gold["mode"] in ("cite", "abstain"), f"{scenario['id']}: bad mode {gold['mode']!r}"
        if gold["mode"] == "cite":
            assert gold["supporting_ids"], f"{scenario['id']}: cite gold without supporting ids"
            missing = set(gold["supporting_ids"]) - provided
            assert not missing, f"{scenario['id']}: supporting ids not provided: {missing}"
        else:
            assert gold["supporting_ids"] == [], f"{scenario['id']}: abstain gold cites ids"


@pytest.mark.parametrize("name", SET_NAMES)
def test_category_implies_gold_mode(name: str) -> None:
    expected = {"clean": "cite", "distractor": "cite", "no_snippet": "abstain"}
    for scenario in _load_set(name):
        mode = expected.get(scenario["category"])
        if mode is not None:
            assert scenario["gold"]["mode"] == mode, (
                f"{scenario['id']}: {scenario['category']} should be {mode}"
            )


# --- conversations -------------------------------------------------------------------------


@pytest.mark.parametrize("name", SET_NAMES)
def test_messages_alternate_and_end_on_user(name: str) -> None:
    for scenario in _load_set(name):
        roles = [message["role"] for message in scenario["messages"]]
        assert roles, f"{scenario['id']} has no messages"
        assert roles[-1] == "user", f"{scenario['id']} does not end on a user turn"
        for i, role in enumerate(roles):
            assert role == ("user" if i % 2 == 0 else "assistant"), (
                f"{scenario['id']}: roles don't alternate starting with user: {roles}"
            )


@pytest.mark.parametrize("name", SET_NAMES)
def test_multi_turn_has_three_to_six_earlier_turns(name: str) -> None:
    multi = [row for row in _load_set(name) if row["category"] == "multi_turn"]
    assert multi, f"{name} has no multi_turn scenarios"
    for scenario in multi:
        earlier = len(scenario["messages"]) - 1
        assert 3 <= earlier <= 6, f"{scenario['id']} has {earlier} earlier turns"


@pytest.mark.parametrize("name", SET_NAMES)
def test_no_manufacturer_names(name: str) -> None:
    path = SCENARIO_DIR / f"{name}.jsonl"
    _load_set(name)
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        assert DENYLIST_RE.search(line) is None, f"{name}.jsonl:{lineno} names a manufacturer"
