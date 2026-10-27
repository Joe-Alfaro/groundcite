#!/usr/bin/env python3
"""Fail if training or few-shot data overlaps the eval sets (R23).

    uv run scripts/check_leakage.py --train PATH --eval PATH [PATH ...]
        [--few-shot PATH] [--reserved PATH] [--report PATH]

Exit 0 when clean, 1 when any rule fires (each finding names its rule code), 2 on bad input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundcite.leakage import LeakageInputError, find_leaks, load_rows, render_report
from groundcite.scenarios.library import DATA_DIR, load_reserved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True, help="training rows (JSONL)")
    parser.add_argument("--eval", type=Path, nargs="+", required=True, help="eval sets (JSONL)")
    parser.add_argument("--few-shot", type=Path, help="few-shot examples (JSONL)")
    parser.add_argument(
        "--reserved",
        type=Path,
        default=DATA_DIR / "reserved_families.txt",
        help="families reserved for held-out (default: data/reserved_families.txt)",
    )
    parser.add_argument("--report", type=Path, help="also write a markdown report here")
    args = parser.parse_args(argv)

    try:
        reserved = load_reserved(args.reserved)
        train = load_rows(args.train)
        evals = [row for path in args.eval for row in load_rows(path)]
        few_shot = load_rows(args.few_shot) if args.few_shot else []
    except (OSError, LeakageInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    findings = find_leaks(train, evals, few_shot, reserved)
    for finding in findings:
        print(finding)
    if args.report:
        inputs = {
            "train": [args.train],
            "eval": list(args.eval),
            "few-shot": [args.few_shot] if args.few_shot else [],
        }
        counts = {"train": len(train), "eval": len(evals), "few-shot": len(few_shot)}
        args.report.write_text(render_report(findings, inputs, counts), encoding="utf-8")
    if findings:
        print(f"leakage check failed: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(
        f"leakage check passed: {len(train)} train, {len(evals)} eval, "
        f"{len(few_shot)} few-shot rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
