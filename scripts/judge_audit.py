#!/usr/bin/env python3
"""Hand-audit the judge on a sample of replies and measure agreement (R29).

    uv run scripts/judge_audit.py --sample TRANSCRIPTS --eval-set EVAL --n 20 --seed S --out SHEET
    uv run scripts/judge_audit.py --score SHEET --transcripts TRANSCRIPTS --out REPORT

--sample writes a blind JSONL sheet (no judge verdict), stratified over (category, judge
pass/fail); fill each row's human_label with a 0/1 pass label (and p_failed for the write-up).
--score compares the labels with the judge's pass/fail verdicts and writes a JSON report.
Exit 1 when agreement is below 90% (18/20), 2 on bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from groundcite.audit import (
    AuditError,
    meets_agreement_bar,
    read_jsonl,
    sample_sheet,
    score_sheet,
)
from groundcite.evalset import EvalSetError, load_eval_set
from groundcite.runner import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=Path, metavar="TRANSCRIPTS", help="write an audit sheet")
    mode.add_argument("--score", type=Path, metavar="SHEET", help="score a filled audit sheet")
    parser.add_argument("--eval-set", type=Path, help="--sample: the run's eval set")
    parser.add_argument("--n", type=int, default=20, help="--sample: rows to sample (default 20)")
    parser.add_argument("--seed", type=int, help="--sample: sampling seed")
    parser.add_argument("--transcripts", type=Path, help="--score: the run's transcripts.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="sheet (JSONL) or report (JSON)")
    return parser


def _sample(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.sample, str(args.sample))
    try:
        scenarios = load_eval_set(args.eval_set)
    except (OSError, EvalSetError) as exc:
        raise AuditError(f"{args.eval_set}: {exc}") from None
    sheet = sample_sheet(rows, scenarios, n=args.n, seed=args.seed)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sheet), encoding="utf-8"
    )
    print(f"wrote {len(sheet)} rows to {args.out}; fill human_label in each, then run --score")
    return 0


def _score(args: argparse.Namespace) -> int:
    sheet = read_jsonl(args.score, str(args.score))
    rows = read_jsonl(args.transcripts, str(args.transcripts))
    report = score_sheet(sheet, rows)
    write_json(args.out, report)
    print(f"agreement: {report['agree']}/{report['n']} ({report['agreement_pct']}%)")
    for d in report["disagreements"]:
        print(
            f"  - {d['scenario_id']} {d['model']} {d['strategy']}: "
            f"human {d['human']}, judge {d['judge']}"
        )
    if not meets_agreement_bar(report):
        print("FAIL: agreement is below 90%")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample and (args.eval_set is None or args.seed is None):
        parser.error("--sample needs --eval-set and --seed")
    if args.score and args.transcripts is None:
        parser.error("--score needs --transcripts")
    try:
        return _sample(args) if args.sample else _score(args)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
