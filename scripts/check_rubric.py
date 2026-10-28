#!/usr/bin/env python3
"""Check that one judge rubric hash covers every run under a results directory (R13).

    uv run scripts/check_rubric.py results/

A run is any directory, at any depth, holding a results file or transcripts.jsonl; its hashes are
pins.rubric_sha256 and every transcript row's rubric_sha256. Exit 0 when one hash covers every run
(or there are no runs), 1 when there is more than one (each hash is listed with its runs), 2 on a
missing directory, an unreadable or invalid file, or a missing hash field.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundcite.audit import AuditError, rubric_groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("results_dir", type=Path, help="results root, e.g. results/")
    args = parser.parse_args(argv)

    try:
        groups = rubric_groups(args.results_dir)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not groups:
        print(f"no runs under {args.results_dir}; nothing to check")
        return 0
    if len(groups) == 1:
        [(rubric, runs)] = groups.items()
        print(f"ok: {len(runs)} run(s), one rubric_sha256 {rubric}")
        return 0
    print(f"FAIL: {len(groups)} rubric hashes across the runs")
    for rubric, runs in groups.items():
        print(f"rubric_sha256 {rubric}:")
        for run in runs:
            print(f"  - {run}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
