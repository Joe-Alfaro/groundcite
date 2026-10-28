"""General shortcut gate: no surface cue may stand in for reading the references (D-046 follow-up).

``test_kind_balance.py`` covers kind, ID scheme, the correction line and the "Applies to all"
opener. Validation of the kind-balance fix found that snippet length and year presence still
predict gold, so this module adds:

1. trivial-heuristic accuracy on distractor scenarios, at most 0.60 (0.67 on the small held-out
   set). Accuracy is the expected value with uniform tie-breaking: a scenario scores
   |picked & gold| / |picked|, and a heuristic that picks nothing falls back to a random snippet.
   Word overlap with the complaint is a legitimate cue, so it is reported but not gated;
2. gold-rate gaps by length tercile and year presence, at most 0.15 (0.25 on heldout);
3. at least one few-shot cite example whose gold is not the longest snippet;
4. no multi-turn abstain scenario leaves a documented, unperformed check in a chain snippet it
   already cited (for example the ground side of a starter voltage-drop test);
5. the R4 hint-leak screen also catches the teacher note's own wording;
6. build_dataset warns, and marks the manifest, when the kept abstain share is outside 20-30%.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from distill_data import (
    ABSTAIN_REPLY,
    FEW_SHOT,
    FINAL_MESSAGES,
    candidate_row,
    cite_reply,
    read_jsonl,
    run_script,
    scenario_row,
    write_jsonl,
)
from test_kind_balance import (
    CORRECTION_LINE,
    HELDOUT_GOLD_RATE_GAP,
    MAX_GOLD_RATE_GAP,
    SET_NAMES,
    YEAR,
    _cite_rows,
    _library_by_text,
    _rows,
    _screen,
)

from groundcite import backends

MAX_HEURISTIC_ACCURACY = 0.60
# The held-out set has only 12 distractor scenarios, so one scenario moves accuracy by 0.08.
HELDOUT_HEURISTIC_ACCURACY = 0.67
# Accuracy is a mean of fractions; 12/20 must not fail on float rounding.
EPS = 1e-9
# Gold-rate groups smaller than this are too noisy to compare.
MIN_GROUP = 5

Snippet = dict[str, Any]
Picker = Callable[[dict[str, Any]], list[int]]


def _distractor_rows(name: str) -> list[dict[str, Any]]:
    rows = [row for row in _rows(name) if row["category"] == "distractor"]
    assert rows and all(row["gold"]["mode"] == "cite" for row in rows), name
    return rows


def _where(row: dict[str, Any], keep: Callable[[Snippet], bool]) -> list[int]:
    return [i for i, snippet in enumerate(row["snippets"]) if keep(snippet)]


def _extreme(row: dict[str, Any], key: Callable[[Snippet], float], pick: Callable) -> list[int]:
    values = [key(snippet) for snippet in row["snippets"]]
    best = pick(values)
    return [i for i, value in enumerate(values) if value == best]


_STOPWORDS = frozenset(
    "a an and are at be but by do does for from has have i if in is it its it's me my no not of "
    "on or so that the then there this to up was what when with would you your".split()
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def _complaint(row: dict[str, Any]) -> str:
    return " ".join(m["content"] for m in row["messages"] if m["role"] == "user")


def _overlap(row: dict[str, Any]) -> list[int]:
    complaint = _words(_complaint(row))
    return _extreme(row, lambda s: len(_words(s["text"]) & complaint), max)


def _length(snippet: Snippet) -> float:
    return len(snippet["text"])


GATED_HEURISTICS: dict[str, Picker] = {
    "cite the longest snippet": lambda row: _extreme(row, _length, max),
    "cite the shortest snippet": lambda row: _extreme(row, _length, min),
    "cite the first snippet": lambda row: [0],
    "cite the last snippet": lambda row: [len(row["snippets"]) - 1],
    "cite a snippet with no model year": lambda row: _where(
        row, lambda s: not YEAR.search(s["text"])
    ),
    "cite a manual-kind snippet": lambda row: _where(row, lambda s: s["kind"] == "manual"),
    "cite a bulletin-kind snippet": lambda row: _where(row, lambda s: s["kind"] == "tsb"),
    "cite a snippet with no correction line": lambda row: _where(
        row, lambda s: not CORRECTION_LINE.search(s["text"])
    ),
}

REPORTED_HEURISTICS: dict[str, Picker] = {
    "cite the snippet with the most word overlap with the complaint": _overlap,
}


def _heuristic_accuracy(rows: list[dict[str, Any]], pick: Picker) -> float:
    """Expected accuracy with uniform tie-breaking; an empty pick falls back to any snippet."""
    total = 0.0
    for row in rows:
        picked = pick(row) or list(range(len(row["snippets"])))
        gold = set(row["gold"]["supporting_ids"])
        total += sum(row["snippets"][i]["id"] in gold for i in picked) / len(picked)
    return total / len(rows)


def _accuracy_limit(name: str) -> float:
    if name == "heldout":
        print(
            f"heldout is small: heuristic accuracy limit loosened to {HELDOUT_HEURISTIC_ACCURACY} "
            f"(default {MAX_HEURISTIC_ACCURACY})"
        )
        return HELDOUT_HEURISTIC_ACCURACY
    return MAX_HEURISTIC_ACCURACY


# --- (1) trivial heuristics on distractor scenarios --------------------------------------------


@pytest.mark.parametrize("heuristic", GATED_HEURISTICS)
@pytest.mark.parametrize("name", SET_NAMES)
def test_no_trivial_heuristic_solves_distractors(name: str, heuristic: str) -> None:
    rows = _distractor_rows(name)
    limit = _accuracy_limit(name)
    accuracy = _heuristic_accuracy(rows, GATED_HEURISTICS[heuristic])
    print(f"{name}: {heuristic}: {accuracy:.2f} on {len(rows)} distractor scenarios")
    assert accuracy <= limit + EPS, (
        f"{name}: '{heuristic}' picks the gold on {accuracy:.0%} of {len(rows)} distractor "
        f"scenarios (limit {limit:.0%})"
    )


@pytest.mark.parametrize("heuristic", REPORTED_HEURISTICS)
@pytest.mark.parametrize("name", SET_NAMES)
def test_report_legitimate_cue_accuracy(name: str, heuristic: str) -> None:
    """Word overlap is a legitimate cue (the right step talks about the complaint): report only."""
    rows = _distractor_rows(name)
    accuracy = _heuristic_accuracy(rows, REPORTED_HEURISTICS[heuristic])
    print(f"{name}: {heuristic} (reported, not gated): {accuracy:.2f} on {len(rows)} distractors")
    assert 0.0 <= accuracy <= 1.0


def test_heuristic_accuracy_uses_uniform_tie_breaking() -> None:
    row = {
        "snippets": [
            {"id": "A", "text": "xx"},
            {"id": "B", "text": "yy"},
            {"id": "C", "text": "z"},
        ],
        "gold": {"supporting_ids": ["A"]},
    }
    assert _heuristic_accuracy([row], GATED_HEURISTICS["cite the longest snippet"]) == 0.5
    assert _heuristic_accuracy([row], lambda r: []) == pytest.approx(1 / 3)


# --- (2) gold-rate gaps by length tercile and year presence ------------------------------------


def _length_tercile(name: str) -> Callable[[Snippet], int]:
    lengths = [len(s["text"]) for row in _cite_rows(name) for s in row["snippets"]]
    low, high = statistics.quantiles(lengths, n=3)
    return lambda s: 0 if len(s["text"]) <= low else 1 if len(s["text"]) <= high else 2


GROUPINGS: dict[str, Callable[[str], Callable[[Snippet], object]]] = {
    "length tercile": _length_tercile,
    "contains a model year": lambda name: lambda s: bool(YEAR.search(s["text"])),
}


def _group_gold_rate_gap(name: str, group_of: Callable[[Snippet], object]) -> tuple[float, str]:
    """Largest minus smallest gold rate across groups with at least MIN_GROUP snippets."""
    gold = Counter[object]()
    seen = Counter[object]()
    for row in _cite_rows(name):
        supporting = set(row["gold"]["supporting_ids"])
        for snippet in row["snippets"]:
            group = group_of(snippet)
            seen[group] += 1
            gold[group] += snippet["id"] in supporting
    rates = {g: gold[g] / n for g, n in seen.items() if n >= MIN_GROUP}
    detail = ", ".join(f"{g}: {rates[g]:.2f} ({gold[g]}/{seen[g]})" for g in sorted(rates, key=str))
    if len(rates) < 2:
        return 0.0, detail
    return max(rates.values()) - min(rates.values()), detail


@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("name", SET_NAMES)
def test_length_and_year_do_not_predict_gold(name: str, grouping: str) -> None:
    limit = HELDOUT_GOLD_RATE_GAP if name == "heldout" else MAX_GOLD_RATE_GAP
    if name == "heldout":
        print(
            f"heldout is small: gold-rate gap limit loosened to {limit} "
            f"(default {MAX_GOLD_RATE_GAP})"
        )
    gap, detail = _group_gold_rate_gap(name, GROUPINGS[grouping](name))
    print(f"{name}: {grouping}: gap {gap:.2f} ({detail})")
    assert gap <= limit + EPS, (
        f"{name}: {grouping} predicts gold (gap {gap:.2f} > {limit}): {detail}"
    )


# --- (3) few-shot examples don't teach "cite the longest" --------------------------------------


def test_some_few_shot_cite_example_gold_is_not_the_longest_snippet() -> None:
    examples = [row for row in read_jsonl(FEW_SHOT) if row["gold"]["mode"] == "cite"]
    assert examples
    not_longest = []
    for row in examples:
        longest = max(len(s["text"]) for s in row["snippets"])
        gold = set(row["gold"]["supporting_ids"])
        if all(len(s["text"]) < longest for s in row["snippets"] if s["id"] in gold):
            not_longest.append(row["id"])
    assert not_longest, (
        f"every few-shot cite example's gold is the longest snippet: {[r['id'] for r in examples]}"
    )


# --- (4) multi-turn abstain leaves no documented check unperformed ----------------------------

# Two-sided checks a snippet can document. A step or reported result covers the check when it
# names both sides, or says "both"/"each".
TWO_SIDED = {
    "positive and ground side": (
        re.compile(r"\bpositive\b", re.IGNORECASE),
        re.compile(r"\b(?:negative|ground)\b", re.IGNORECASE),
    ),
    "inlet and outlet": (
        re.compile(r"\binlet\b", re.IGNORECASE),
        re.compile(r"\boutlet\b", re.IGNORECASE),
    ),
    "low and high side": (
        re.compile(r"\blow[- ]side\b", re.IGNORECASE),
        re.compile(r"\bhigh[- ]side\b", re.IGNORECASE),
    ),
}
BOTH = re.compile(r"\b(?:both|each)\b", re.IGNORECASE)
CITATION = re.compile(r"\[([^\[\]]+)\]")


def _documented_sides(text: str) -> list[str]:
    return [name for name, (a, b) in TWO_SIDED.items() if a.search(text) and b.search(text)]


def _covers(text: str, side: str) -> bool:
    a, b = TWO_SIDED[side]
    return bool(BOTH.search(text)) or bool(a.search(text) and b.search(text))


def _chain_snippets() -> list[dict[str, Any]]:
    return [raw for raw in _library_by_text().values() if raw.get("step")]


def test_cable_drop_step_or_its_results_cover_both_sides() -> None:
    raw = next(r for r in _chain_snippets() if r["id"] == "no_crank_click.cable_drop")
    side = "positive and ground side"
    assert side in _documented_sides(raw["text"])
    covered = _covers(raw["step"], side) or all(
        _covers(f"{raw['step']} {result}", side) for result in raw["results"]
    )
    assert covered, (
        "no_crank_click.cable_drop documents a positive-side and a ground-side drop, but its "
        f"step and results cover only one: step={raw['step']!r} results={raw['results']!r}"
    )


@pytest.mark.parametrize("raw", _chain_snippets(), ids=lambda raw: raw["id"])
def test_every_chain_step_or_result_covers_each_documented_two_sided_check(
    raw: dict[str, Any],
) -> None:
    """Chain metadata: a performed step plus any of its results covers every documented side."""
    uncovered = [
        (side, result)
        for side in _documented_sides(raw["text"])
        for result in raw["results"]
        if not _covers(f"{raw['step']} {result}", side)
    ]
    assert not uncovered, (
        f"{raw['id']}: snippet documents a two-sided check the step leaves half done: "
        f"step={raw['step']!r}, uncovered (side, result) = {uncovered}"
    )


def _unperformed_checks(row: dict[str, Any], library: dict[str, dict[str, Any]]) -> list[str]:
    """Cited chain snippets whose documented two-sided check the dialogue performed only in part."""
    by_id = {s["id"]: s for s in row["snippets"]}
    messages = row["messages"]
    problems = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        reported = messages[i + 1]["content"] if i + 1 < len(messages) else ""
        for cited in CITATION.findall(message["content"]):
            source = library.get(by_id[cited]["text"], {}) if cited in by_id else {}
            if not source.get("step"):
                continue
            for side in _documented_sides(source["text"]):
                if not _covers(f"{message['content']} {reported}", side):
                    problems.append(f"[{cited}] ({source['id']}): {side} not both done")
    return problems


@pytest.mark.parametrize("name", SET_NAMES)
def test_multi_turn_abstain_leaves_no_documented_check_unperformed(name: str) -> None:
    """ho-0056: gold abstains, yet the ground side of the cited voltage-drop test is undone."""
    library = _library_by_text()
    rows = [
        r for r in _rows(name) if r["category"] == "multi_turn" and r["gold"]["mode"] == "abstain"
    ]
    bad = {row["id"]: p for row in rows if (p := _unperformed_checks(row, library))}
    assert not bad, (
        f"{name}: multi-turn abstain scenarios leave a documented check in an already cited "
        f"snippet unperformed, so citing it again is a valid next step: {bad}"
    )


@pytest.mark.parametrize("name", SET_NAMES)
def test_multi_turn_cite_leaves_no_documented_check_unperformed(name: str) -> None:
    """ho-0028 pattern: a half-done cited check is a second valid next step besides the gold."""
    library = _library_by_text()
    rows = [r for r in _rows(name) if r["category"] == "multi_turn" and r["gold"]["mode"] == "cite"]
    bad = {row["id"]: p for row in rows if (p := _unperformed_checks(row, library))}
    assert not bad, (
        f"{name}: multi-turn cite scenarios leave a documented check in an already cited "
        f"snippet unperformed, so the gold is not the only valid next step: {bad}"
    )


# --- (5) hint-leak screen: the teacher note's own wording -------------------------------------

LEAKY_REPLIES = (
    "The other references are near-misses, so inspect the hose at the intake elbow [TSB-4101].",
    "SB 22-104 is only a near miss for this engine; inspect the hose at the intake elbow "
    "[TSB-4101].",
    "That one is a Near-Miss. With the engine idling, inspect the hose at the intake elbow "
    "[TSB-4101].",
    "The cap bulletin is already used up, so inspect the hose at the intake elbow [TSB-4101].",
    "The snippet that supports the next step is the elbow check: inspect the hose for cracks "
    "[TSB-4101].",
    "This is the reference that supports the next step: inspect the hose at the intake "
    "elbow [TSB-4101].",
    "I've been told to inspect the hose at the intake elbow for cracks [TSB-4101].",
    "I’ve been told the elbow hose is the place to look, so inspect it for cracks [TSB-4101].",
    "I have been told to inspect the hose at the intake elbow for cracks [TSB-4101].",
    "The expected answer here is to inspect the hose at the intake elbow [TSB-4101].",
    "The correct ID is the elbow bulletin: inspect the hose at the intake elbow for cracks "
    "[TSB-4101].",
    "Citing the correct id, inspect the hose at the intake elbow for cracks [TSB-4101].",
)

CLEAN_REPLIES = (
    "The bulletin covers this idle complaint: with the engine idling, inspect the hose at the "
    "intake elbow for cracks or a vacuum hiss [TSB-4101].",
    "Per the service manual section on vacuum leaks, inspect the hose at the intake elbow for "
    "cracks or a vacuum hiss [TSB-4101].",
    "Your steady fuel trim reading supports the next step: with the engine idling, inspect the "
    "hose at the intake elbow for cracks or a vacuum hiss [TSB-4101].",
    "The expected reading at idle is a steady vacuum; inspect the hose at the intake elbow for "
    "cracks or a vacuum hiss [TSB-4101].",
    "If you've been told it's a vacuum leak, confirm it first: inspect the hose at the intake "
    "elbow for cracks or a vacuum hiss [TSB-4101].",
    "With the engine at its correct idle speed, inspect the hose at the intake elbow for cracks "
    "or a vacuum hiss [TSB-4101].",
    "The next step is to inspect the hose at the intake elbow for cracks or a vacuum hiss with "
    "the engine idling [TSB-4101].",
)


@pytest.mark.parametrize("reply", LEAKY_REPLIES)
def test_hint_leak_screen_rejects_teacher_note_wording(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is not None and rejection.reason == HINT_LEAK, (
        f"not rejected as {HINT_LEAK}: {reply!r} -> {rejection}"
    )


@pytest.mark.parametrize("reply", CLEAN_REPLIES)
def test_hint_leak_screen_still_accepts_normal_replies(reply: str) -> None:
    from groundcite.filter import HINT_LEAK

    rejection = _screen(reply)
    assert rejection is None or rejection.reason != HINT_LEAK, f"{reply!r} -> {rejection}"


# --- (6) build_dataset flags a kept abstain share outside 20-30% -------------------------------


@pytest.fixture
def build_with_abstain(
    isolated_env: Path,
    no_network: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Callable[[int, int], tuple[dict[str, Any], str]]:
    """Build a dataset where every candidate is kept: ``abstain`` of ``total`` abstain."""
    from distill_data import ScriptedJudge

    monkeypatch.setattr(ScriptedJudge, "verdicts", {})
    monkeypatch.setitem(backends.MOCKS, "judge_probe", ScriptedJudge)

    def build(abstain: int, total: int) -> tuple[dict[str, Any], str]:
        assert total <= len(FINAL_MESSAGES)
        candidates = []
        for i in range(total):
            if i < abstain:
                row = scenario_row(i, category="no_snippet", mode="abstain")
                candidates.append(candidate_row(row, 0, ABSTAIN_REPLY))
            else:
                row = scenario_row(i, category="clean")
                candidates.append(
                    candidate_row(row, 0, cite_reply(row["gold"]["supporting_ids"][0], i))
                )
        work = tmp_path / f"a{abstain}of{total}"
        cands = write_jsonl(work / "candidates.jsonl", candidates)
        out = work / "v1"
        capsys.readouterr()
        code = run_script(
            "build_dataset",
            ["--candidates", str(cands), "--out", str(out), "--judge", "mock:judge_probe"],
        )
        captured = capsys.readouterr()
        assert code == 0, captured.out + captured.err
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert m["kept"] == total, f"fixture expects every candidate kept: {m}"
        return m, captured.out + captured.err

    return build


ABSTAIN_WARNING = re.compile(r"warn\w*\b.*abstain|abstain.*\bwarn", re.IGNORECASE)


@pytest.mark.parametrize(("abstain", "total"), [(1, 10), (0, 10), (4, 10), (8, 10)])
def test_build_dataset_warns_when_kept_abstain_share_is_out_of_range(
    build_with_abstain: Callable[[int, int], tuple[dict[str, Any], str]], abstain: int, total: int
) -> None:
    m, output = build_with_abstain(abstain, total)
    assert m["abstain_share"] == pytest.approx(abstain / total)
    assert m.get("abstain_share_in_range") is False, m
    assert ABSTAIN_WARNING.search(output), f"no abstain-share warning printed:\n{output}"


@pytest.mark.parametrize(("abstain", "total"), [(2, 10), (3, 10), (1, 4)])
def test_build_dataset_is_quiet_when_kept_abstain_share_is_in_range(
    build_with_abstain: Callable[[int, int], tuple[dict[str, Any], str]], abstain: int, total: int
) -> None:
    m, output = build_with_abstain(abstain, total)
    assert m["abstain_share"] == pytest.approx(abstain / total)
    assert m.get("abstain_share_in_range") is True, m
    assert not ABSTAIN_WARNING.search(output), f"unexpected abstain-share warning:\n{output}"
