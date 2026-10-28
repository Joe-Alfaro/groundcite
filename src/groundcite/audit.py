"""Audits over finished runs: one rubric hash (R13) and the judge hand audit (R29), plus the run
readers ``groundcite.rerun`` (R28) shares. ``scripts/check_rubric.py`` and
``scripts/judge_audit.py`` are thin CLIs over this module.

Every function raises ``AuditError`` on bad input (missing file, invalid JSON, missing field); the
scripts turn it into exit 2.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from groundcite.evalset import Scenario

RESULTS = "results.json"
TRANSCRIPTS = "transcripts.jsonl"

AGREEMENT_BAR = 0.90  # R29: at least 18 of 20 hand labels agree with the judge.
_EPS = 1e-9

RowKey = tuple[str, str, str]


class AuditError(ValueError):
    """Bad audit input. The message names the file or run at fault."""


# ---------------------------------------------------------------------------------------------
# Reading runs
# ---------------------------------------------------------------------------------------------


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuditError(f"{label}: can't read {path.name}: {exc.strerror or exc}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditError(f"{label}: {path.name} is not valid JSON: {exc}") from None


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """Every non-blank line of ``path`` as a JSON object; errors give ``label`` and the line."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"{label}: can't read {path.name}: {exc.strerror or exc}") from None
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label}: {path.name} is not UTF-8: {exc}") from None
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(
                f"{label}: {path.name} line {line_no} is not valid JSON: {exc}"
            ) from None
        if not isinstance(row, dict):
            raise AuditError(f"{label}: {path.name} line {line_no} is not a JSON object")
        rows.append(row)
    return rows


def row_key(row: Mapping[str, Any]) -> RowKey:
    """A transcript row's identity within one run: (scenario_id, model, strategy)."""
    return (str(row.get("scenario_id")), str(row.get("model")), str(row.get("strategy")))


def index_rows(rows: Iterable[Mapping[str, Any]], label: str) -> dict[RowKey, Mapping[str, Any]]:
    """Rows by ``row_key``; a key seen twice is an error (which row would count?)."""
    indexed: dict[RowKey, Mapping[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise AuditError(f"{label}: duplicate row {fmt_key(key)}")
        indexed[key] = row
    return indexed


def fmt_key(key: RowKey) -> str:
    return " / ".join(key)


def spec_score(row: Mapping[str, Any], label: str) -> int:
    """The judge's 0/1 Spec-adherence for a transcript row."""
    score = row.get("score")
    value = score.get("spec_adherence") if isinstance(score, Mapping) else None
    if value not in (0, 1):
        raise AuditError(f"{label}: row {fmt_key(row_key(row))} has no 0/1 score.spec_adherence")
    return int(value)


# ---------------------------------------------------------------------------------------------
# One rubric hash (R13)
# ---------------------------------------------------------------------------------------------


def find_runs(results_dir: Path) -> list[Path]:
    """Every directory under ``results_dir``, at any depth, holding results.json or transcripts."""
    if not results_dir.is_dir():
        raise AuditError(f"{results_dir}: no such directory")
    found = {p.parent for name in (RESULTS, TRANSCRIPTS) for p in results_dir.rglob(name)}
    return sorted(found)


def run_rubric_hashes(run_dir: Path, label: str) -> set[str]:
    """``pins.rubric_sha256`` from results.json plus every transcript row's ``rubric_sha256``."""
    hashes: set[str] = set()
    results_path = run_dir / RESULTS
    if results_path.exists():
        results = read_json(results_path, label)
        pins = results.get("pins") if isinstance(results, dict) else None
        rubric = pins.get("rubric_sha256") if isinstance(pins, dict) else None
        if not isinstance(rubric, str):
            raise AuditError(f"{label}: {RESULTS} has no pins.rubric_sha256")
        hashes.add(rubric)
    transcripts_path = run_dir / TRANSCRIPTS
    if transcripts_path.exists():
        for i, row in enumerate(read_jsonl(transcripts_path, label), start=1):
            rubric = row.get("rubric_sha256")
            if not isinstance(rubric, str):
                raise AuditError(f"{label}: {TRANSCRIPTS} row {i} has no rubric_sha256")
            hashes.add(rubric)
    return hashes


def rubric_groups(results_dir: Path) -> dict[str, list[str]]:
    """Each rubric hash and the runs (relative to ``results_dir``) that carry it. A run scored
    under two rubrics is listed under both."""
    groups: dict[str, list[str]] = {}
    for run_dir in find_runs(results_dir):
        label = run_dir.relative_to(results_dir).as_posix()
        for rubric in sorted(run_rubric_hashes(run_dir, label)):
            groups.setdefault(rubric, []).append(label)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------------------------
# Judge hand audit (R29)
# ---------------------------------------------------------------------------------------------


def _allocate(sizes: Mapping[Any, int], n: int, rng: random.Random) -> dict[Any, int]:
    """Split ``n`` across strata as equally as their sizes allow: a stratum smaller than its share
    is taken whole and the rest fill up; leftover single rows go to random strata."""
    alloc = dict.fromkeys(sizes, 0)
    remaining = n
    while remaining:
        open_strata = [s for s in sizes if alloc[s] < sizes[s]]
        base, extra = divmod(remaining, len(open_strata))
        if base == 0:
            for s in rng.sample(open_strata, extra):
                alloc[s] += 1
            break
        for s in open_strata:
            give = min(base, sizes[s] - alloc[s])
            alloc[s] += give
            remaining -= give
    return alloc


def sample_sheet(
    rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Scenario],
    *,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """A blind audit sheet of ``n`` transcript rows, stratified over (category, judge verdict).

    The sheet carries what a human needs to label the reply (the conversation, the snippets from
    the eval set and the gold) and no judge verdict. Rows are shuffled so the verdict can't be read
    from the order either.
    """
    by_id = {s.id: s for s in scenarios}
    indexed = index_rows(rows, "transcripts")
    missing = sorted({key[0] for key in indexed} - by_id.keys())
    if missing:
        raise AuditError(f"scenarios missing from the eval set: {', '.join(missing)}")
    if len(indexed) < n:
        raise AuditError(f"only {len(indexed)} transcript rows; can't sample {n}")

    strata: dict[tuple[str, int], list[RowKey]] = {}
    for key, row in indexed.items():
        verdict = spec_score(row, "transcripts")
        strata.setdefault((str(row.get("category")), verdict), []).append(key)
    strata = dict(sorted(strata.items()))

    rng = random.Random(seed)
    alloc = _allocate({s: len(keys) for s, keys in strata.items()}, n, rng)
    picked = [key for s, keys in strata.items() for key in rng.sample(keys, alloc[s])]
    rng.shuffle(picked)

    sheet: list[dict[str, Any]] = []
    for key in picked:
        row, scenario = indexed[key], by_id[key[0]]
        sheet.append(
            {
                "scenario_id": key[0],
                "model": key[1],
                "strategy": key[2],
                "category": row.get("category"),
                "messages": scenario.messages,
                "reply": row.get("reply"),
                "snippets": scenario.snippets,
                "gold": row.get("gold"),
                "human_label": {"spec_adherence": None, "p_failed": []},
            }
        )
    return sheet


def score_sheet(
    sheet: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Agreement between the hand labels and the judge on Spec-adherence."""
    if not sheet:
        raise AuditError("the audit sheet is empty")
    indexed = index_rows(rows, "transcripts")
    seen: set[RowKey] = set()
    agree = 0
    disagreements: list[dict[str, Any]] = []
    for i, entry in enumerate(sheet, start=1):
        key = row_key(entry)
        if key in seen:
            raise AuditError(f"sheet row {i} ({fmt_key(key)}) is listed twice")
        seen.add(key)
        if key not in indexed:
            raise AuditError(f"sheet row {i} ({fmt_key(key)}) is not in the transcripts")
        label = entry.get("human_label")
        human = label.get("spec_adherence") if isinstance(label, Mapping) else None
        if human not in (0, 1):
            raise AuditError(
                f"sheet row {i} ({fmt_key(key)}) has no 0/1 human_label.spec_adherence"
            )
        judge = spec_score(indexed[key], "transcripts")
        if int(human) == judge:
            agree += 1
        else:
            disagreements.append(
                {
                    "scenario_id": key[0],
                    "model": key[1],
                    "strategy": key[2],
                    "human": int(human),
                    "judge": judge,
                }
            )
    n = len(sheet)
    return {
        "n": n,
        "agree": agree,
        "agreement_pct": round(100 * agree / n, 2),
        "disagreements": disagreements,
    }


def meets_agreement_bar(report: Mapping[str, Any]) -> bool:
    return bool(report["agree"] >= AGREEMENT_BAR * report["n"] - _EPS)
