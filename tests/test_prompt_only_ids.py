"""Prompt-only provided IDs and the citation_format downgrade (R12, R21; D-033, supersedes D-022).

A prompt-only row (no structured ``snippets``) takes every bracketed ID in its user and system
text as provided, as R12 says: there is no digit rule, so all-letter IDs such as ``[TSB-ABC]``
are provided too. Because that inferred list can also hold bracketed trouble codes (``[P0171]``)
or UI words (``[OK]``), the "unbracketed provided ID" part of ``citation_format`` is only a SOFT
flag on prompt-only rows. Structured-snippet rows keep it HARD. ``cited_id_provided`` and
``citation_count`` stay HARD on both kinds of row.

Row-level scoring goes through ``checks.run_checks(reply, scenario)``, which runs every check
with the scenario's provided IDs and gold and tags each result with ``severity`` ("HARD" or
"SOFT"); ``forces_zero`` counts only HARD failures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
FORMAT_DOC = REPO / "docs" / "eval-set-format.md"
REQUIREMENTS = REPO / "docs" / "requirements" / "small-model-finetune.md"

PROMPT_ALPHA = (
    "References:\n"
    "[TSB-ABC] Some 2.0L engines may set lean codes at idle when the PCV hose cracks at the "
    "intake elbow. Inspect the hose at the elbow for cracks or a vacuum hiss with the engine "
    "idling.\n"
    "[SM-FUEL] Fuel system, fuel pressure test at the rail test port.\n\n"
    "Complaint: 2015 compact sedan, 2.0L, rough idle when cold."
)
STEP_ALPHA = (
    "With the engine idling, inspect the PCV hose where it meets the intake elbow for cracks "
    "or a vacuum hiss [TSB-ABC]."
)

PROMPT_DTC = (
    "Codes: [P0171]\n"
    "References:\n"
    "[TSB-0417] Some 2.0L engines may set lean codes at idle when the PCV hose cracks at the "
    "intake elbow.\n"
    "[SM-2210] Fuel system, fuel pressure test at the rail test port.\n\n"
    "Complaint: 2015 compact sedan, 2.0L, rough idle when cold."
)
PROMPT_PLAIN = (
    "References:\n"
    "[TSB-0417] Some 2.0L engines may set lean codes at idle when the PCV hose cracks at the "
    "intake elbow.\n"
    "[SM-2210] Fuel system, fuel pressure test at the rail test port.\n\n"
    "Complaint: 2015 compact sedan, 2.0L, P0171, rough idle when cold."
)
SNIPPETS = [
    {"id": "TSB-0417", "text": "PCV hose cracks at the intake elbow on some 2.0L engines."},
    {"id": "SM-2210", "text": "Fuel system, fuel pressure test at the rail test port."},
]
GOLD_PCV = {"mode": "cite", "supporting_ids": ["TSB-0417"]}

# On-spec replies that mention a provided ID without brackets.
MENTIONS_DTC = (
    "With P0171 set, inspect the PCV hose where it meets the intake elbow for cracks or a "
    "vacuum hiss [TSB-0417]."
)
MENTIONS_REF = (
    "Before the SM-2210 pressure test, inspect the PCV hose where it meets the intake elbow for "
    "cracks or a vacuum hiss [TSB-0417]."
)


def _load(tmp_path: Path, rows: list[dict[str, Any]]) -> list[Any]:
    from groundcite.evalset import load_eval_set

    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return load_eval_set(path)


def _one(tmp_path: Path, row: dict[str, Any]) -> Any:
    (scenario,) = _load(tmp_path, [row])
    return scenario


def _run(reply: str, scenario: Any) -> list[Any]:
    from groundcite.checks import run_checks

    return list(run_checks(reply, scenario))


def _failed(results: list[Any], name: str) -> Any:
    hits = [r for r in results if r.name == name and r.passed is False]
    assert hits, f"{name} did not flag the reply: {[(r.name, r.passed, r.detail) for r in results]}"
    return hits[0]


# --- (1) every bracketed ID in a prompt-only row is provided, including all-letter IDs ----------


def test_prompt_only_row_provides_all_letter_ids(tmp_path: Path) -> None:
    scenario = _one(tmp_path, {"id": "po-alpha", "prompt": PROMPT_ALPHA})
    assert scenario.provided_ids == ["TSB-ABC", "SM-FUEL"]


def test_prompt_only_ids_come_from_system_text_too(tmp_path: Path) -> None:
    row = {
        "id": "po-sys",
        "messages": [
            {"role": "system", "content": "References: [REF-A] PCV hose. [REF-B] Fuel pressure."},
            {"role": "user", "content": "Rough idle when cold. Where do I start?"},
        ],
    }
    assert _one(tmp_path, row).provided_ids == ["REF-A", "REF-B"]


def test_prompt_only_gold_may_name_an_all_letter_id(tmp_path: Path) -> None:
    gold = {"mode": "cite", "supporting_ids": ["TSB-ABC"]}
    scenario = _one(tmp_path, {"id": "po-alpha", "prompt": PROMPT_ALPHA, "gold": gold})
    assert scenario.gold["supporting_ids"] == ["TSB-ABC"]


# --- (2) an on-spec reply citing an all-letter ID passes every HARD check ---------------------


@pytest.mark.parametrize("with_gold", [True, False], ids=["gold", "no-gold"])
def test_on_spec_reply_citing_all_letter_id_passes_hard_checks(
    tmp_path: Path, with_gold: bool
) -> None:
    from groundcite.checks import CHECKS, HARD_CHECKS

    row: dict[str, Any] = {"id": "po-alpha", "prompt": PROMPT_ALPHA}
    if with_gold:
        row["gold"] = {"mode": "cite", "supporting_ids": ["TSB-ABC"]}
    scenario = _one(tmp_path, row)
    failures = [
        f"{name}: {result.detail}"
        for name, fn in CHECKS.items()
        if name in HARD_CHECKS
        and (result := fn(STEP_ALPHA, scenario.provided_ids, gold=scenario.gold)).passed is False
    ]
    assert not failures, failures


def test_on_spec_reply_citing_all_letter_id_does_not_force_zero(tmp_path: Path) -> None:
    from groundcite.checks import forces_zero

    gold = {"mode": "cite", "supporting_ids": ["TSB-ABC"]}
    scenario = _one(tmp_path, {"id": "po-alpha", "prompt": PROMPT_ALPHA, "gold": gold})
    results = _run(STEP_ALPHA, scenario)
    assert not forces_zero(results), [(r.name, r.passed, r.detail) for r in results]


# --- (3) unbracketed provided ID: SOFT on prompt-only rows, HARD on structured rows ----------


def test_run_checks_tags_every_result_with_a_severity(tmp_path: Path) -> None:
    from groundcite.checks import CHECKS, HARD_CHECKS

    scenario = _one(tmp_path, {"id": "st", "snippets": SNIPPETS, "prompt": "Rough idle cold."})
    results = _run(STEP_ALPHA.replace("TSB-ABC", "TSB-0417"), scenario)
    assert {r.name for r in results} == set(CHECKS)
    for r in results:
        assert r.severity == ("HARD" if r.name in HARD_CHECKS else "SOFT"), (r.name, r.severity)


@pytest.mark.parametrize(
    ("prompt", "reply", "bare"),
    [(PROMPT_DTC, MENTIONS_DTC, "P0171"), (PROMPT_PLAIN, MENTIONS_REF, "SM-2210")],
    ids=["bracketed-dtc", "reference-id"],
)
def test_prompt_only_unbracketed_provided_id_is_a_soft_flag(
    tmp_path: Path, prompt: str, reply: str, bare: str
) -> None:
    from groundcite.checks import forces_zero

    scenario = _one(tmp_path, {"id": "po", "prompt": prompt, "gold": GOLD_PCV})
    assert bare in scenario.provided_ids
    results = _run(reply, scenario)
    flag = _failed(results, "citation_format")
    assert bare in flag.detail
    assert flag.severity == "SOFT"
    assert not forces_zero(results), [(r.name, r.passed, r.severity) for r in results]


def test_prompt_only_soft_flag_holds_without_gold(tmp_path: Path) -> None:
    from groundcite.checks import forces_zero

    scenario = _one(tmp_path, {"id": "po", "prompt": PROMPT_DTC})
    results = _run(MENTIONS_DTC, scenario)
    assert _failed(results, "citation_format").severity == "SOFT"
    assert not forces_zero(results)


def test_structured_row_unbracketed_provided_id_stays_hard(tmp_path: Path) -> None:
    from groundcite.checks import forces_zero

    row = {"id": "st", "snippets": SNIPPETS, "prompt": "Rough idle when cold.", "gold": GOLD_PCV}
    results = _run(MENTIONS_REF, _one(tmp_path, row))
    flag = _failed(results, "citation_format")
    assert "SM-2210" in flag.detail
    assert flag.severity == "HARD"
    assert forces_zero(results)


def test_structured_row_with_dtc_snippet_id_stays_hard(tmp_path: Path) -> None:
    from groundcite.checks import forces_zero

    snippets = [*SNIPPETS, {"id": "P0171", "text": "Lean condition, bank 1: diagnostic overview."}]
    row = {"id": "st", "snippets": snippets, "prompt": "Rough idle when cold.", "gold": GOLD_PCV}
    results = _run(MENTIONS_DTC, _one(tmp_path, row))
    assert _failed(results, "citation_format").severity == "HARD"
    assert forces_zero(results)


@pytest.mark.parametrize(
    ("reply", "check"),
    [
        ("Inspect the PCV hose at the intake elbow for cracks [TSB-9999].", "cited_id_provided"),
        (
            "Inspect the PCV hose at the intake elbow [TSB-0417] and test fuel pressure [SM-2210].",
            "citation_count",
        ),
    ],
    ids=["invented-id", "two-ids"],
)
def test_prompt_only_id_errors_stay_hard(tmp_path: Path, reply: str, check: str) -> None:
    from groundcite.checks import forces_zero

    scenario = _one(tmp_path, {"id": "po", "prompt": PROMPT_DTC, "gold": GOLD_PCV})
    results = _run(reply, scenario)
    assert _failed(results, check).severity == "HARD"
    assert forces_zero(results)


def test_prompt_only_clean_reply_has_no_citation_format_flag(tmp_path: Path) -> None:
    scenario = _one(tmp_path, {"id": "po", "prompt": PROMPT_DTC, "gold": GOLD_PCV})
    results = _run(MENTIONS_DTC.replace("With P0171 set, inspect", "Inspect"), scenario)
    assert all(r.passed is not False for r in results), [(r.name, r.detail) for r in results]


# --- (4) the rule is written down where third parties read it (D-033) --------------------------


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()))


@pytest.mark.parametrize("doc", [FORMAT_DOC, REQUIREMENTS], ids=lambda p: p.name)
def test_doc_states_unbracketed_mention_is_soft_on_prompt_only_rows(doc: Path) -> None:
    hits = [
        s
        for s in _sentences(doc.read_text(encoding="utf-8"))
        if re.search(r"prompt-only", s, re.IGNORECASE)
        and re.search(r"unbracketed", s, re.IGNORECASE)
        and re.search(r"\bSOFT\b", s)
    ]
    assert hits, f"{doc.name}: no sentence says an unbracketed mention is SOFT on prompt-only rows"


@pytest.mark.parametrize("doc", [FORMAT_DOC, REQUIREMENTS], ids=lambda p: p.name)
def test_doc_does_not_describe_a_digit_rule_for_provided_ids(doc: Path) -> None:
    text = " ".join(doc.read_text(encoding="utf-8").split())
    assert not re.search(r"(contain|need|have)s? (at least one |a )digit", text, re.IGNORECASE)
