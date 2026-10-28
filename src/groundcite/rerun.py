"""Spec-adherence delta between two runs of the same pins (R28).

``scripts/rerun_delta.py`` runs ``main``. Each run directory needs results.json and
transcripts.jsonl. The runs must have equal pins (run_id and timestamps aside) and scenario_ids,
and their transcripts must hold the same rows. Deltas are B minus A, from the transcripts: cell
means go through ``aggregate_scores`` like every reported score, and ``changed_rows`` lists each
(scenario_id, model, strategy) whose Spec-adherence flipped. Exit 1 when any cell moves by more
than 0.03 (exactly 0.03 passes), 2 when the runs can't be compared.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from groundcite.audit import (
    RESULTS,
    TRANSCRIPTS,
    AuditError,
    index_rows,
    read_json,
    read_jsonl,
    spec_score,
)
from groundcite.report import aggregate_scores
from groundcite.runner import write_json

RERUN_TOLERANCE = 0.03
_EPS = 1e-9
# Keys that legitimately differ between two runs of the same pins (D-053).
IGNORED_PIN_KEYS = frozenset({"run_id", "created_at", "started_at", "finished_at", "timestamp"})
_MISSING = object()


def pin_mismatches(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """The pin keys whose values differ, ignoring the run id and timestamps."""
    keys = (a.keys() | b.keys()) - IGNORED_PIN_KEYS
    return sorted(k for k in keys if a.get(k, _MISSING) != b.get(k, _MISSING))


def _cell_means(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    # The delta reads only the Spec-adherence mean, so the rows are cut to what that needs; checks
    # and gold feed metrics it doesn't report. One bootstrap draw: the CI is unused.
    slim = [
        {
            "model": r["model"],
            "strategy": r["strategy"],
            "category": r.get("category"),
            "score": r["score"],
            "checks": {},
        }
        for r in rows
    ]
    return {
        (c["model"], c["strategy"]): c["spec_adherence"]["mean"]
        for c in aggregate_scores(slim, seed=0, n_boot=1)
    }


def rerun_delta(
    results_a: Mapping[str, Any],
    rows_a: Sequence[Mapping[str, Any]],
    results_b: Mapping[str, Any],
    rows_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-cell and per-row Spec-adherence deltas, B minus A."""
    pins_a, pins_b = results_a.get("pins"), results_b.get("pins")
    if not isinstance(pins_a, Mapping) or not isinstance(pins_b, Mapping):
        raise AuditError(f"both runs need a pins object in {RESULTS}")
    differing = pin_mismatches(pins_a, pins_b)
    if differing:
        raise AuditError(f"the runs have different pins: {', '.join(differing)}")
    if results_a.get("scenario_ids") != results_b.get("scenario_ids"):
        raise AuditError("the runs have different scenario_ids")

    indexed_a, indexed_b = index_rows(rows_a, "run A"), index_rows(rows_b, "run B")
    if indexed_a.keys() != indexed_b.keys():
        only = len(indexed_a.keys() - indexed_b.keys()), len(indexed_b.keys() - indexed_a.keys())
        raise AuditError(
            f"the runs' transcripts hold different rows ({only[0]} only in A, {only[1]} only in B)"
        )

    changed_rows: list[dict[str, Any]] = []
    for key, row_a in indexed_a.items():
        a, b = spec_score(row_a, "run A"), spec_score(indexed_b[key], "run B")
        if a != b:
            scenario_id, model, strategy = key
            changed_rows.append(
                {"scenario_id": scenario_id, "model": model, "strategy": strategy, "a": a, "b": b}
            )

    means_a, means_b = _cell_means(rows_a), _cell_means(rows_b)
    cells: list[dict[str, Any]] = [
        {
            "model": model,
            "strategy": strategy,
            "a": a,
            "b": means_b[(model, strategy)],
            "delta": means_b[(model, strategy)] - a,
        }
        for (model, strategy), a in means_a.items()
    ]
    return {
        "tolerance": RERUN_TOLERANCE,
        "max_abs_delta": max((abs(c["delta"]) for c in cells), default=0.0),
        "cells": cells,
        "changed_rows": changed_rows,
    }


def within_tolerance(report: Mapping[str, Any]) -> bool:
    """|delta| <= 0.03, with an epsilon so 0.53 - 0.50 (0.030000000000000027) still passes."""
    return bool(report["max_abs_delta"] <= report["tolerance"] + _EPS)


def _load(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = str(run_dir)
    for name in (RESULTS, TRANSCRIPTS):
        if not (run_dir / name).is_file():
            raise AuditError(f"{label}: no {name}")
    results = read_json(run_dir / RESULTS, label)
    if not isinstance(results, dict):
        raise AuditError(f"{label}: {RESULTS} is not a JSON object")
    return results, read_jsonl(run_dir / TRANSCRIPTS, label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rerun_delta.py", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("run_a", type=Path, help="the first run directory")
    parser.add_argument("run_b", type=Path, help="the rerun directory")
    parser.add_argument("--out", type=Path, help="also write the report as JSON here")
    args = parser.parse_args(argv)

    try:
        report = rerun_delta(*_load(args.run_a), *_load(args.run_b))
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for c in report["cells"]:
        print(
            f"{c['model']} x {c['strategy']}: A {c['a']:.3f}, B {c['b']:.3f}, "
            f"delta {c['delta']:+.3f}"
        )
    print(f"changed rows: {len(report['changed_rows'])}")
    for r in report["changed_rows"]:
        print(f"  - {r['scenario_id']} {r['model']} {r['strategy']}: {r['a']} -> {r['b']}")
    if args.out:
        write_json(args.out, report)
    ok = within_tolerance(report)
    verdict = "ok" if ok else "FAIL"
    print(f"{verdict}: max |delta| {report['max_abs_delta']:.3f}, tolerance {report['tolerance']}")
    return 0 if ok else 1
