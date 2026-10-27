#!/usr/bin/env python3
"""Prompt-ceiling ablation: models x strategies x the same scenarios (R3, R14, R20).

    uv run scripts/run_ablation.py [--model M ...] [--strategy S ...] [--limit N] [--dry-run]
    uv run scripts/run_ablation.py --run-id ID --resume      # continue an interrupted run

Defaults: the config [ablation] models, all three strategies, data/scenarios/ablation.jsonl and
the pinned judge. Rows are appended to transcripts.jsonl as they finish, so --resume re-pays
nothing. Transient API errors are retried with backoff; subject errors that survive the retries
are scored as failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from groundcite.ablation import (
    ROBUSTNESS_BAR,
    SPEC_ADHERENCE_BAR,
    model_family,
    plateau_verdict,
    strategy_deltas,
)
from groundcite.backends import BackendError
from groundcite.cli import check_clean_tree, load_scenarios
from groundcite.config import load_models_config
from groundcite.judge import rubric_sha256
from groundcite.keys import MissingKeyError
from groundcite.prompts import STRATEGIES
from groundcite.report import aggregate_scores
from groundcite.runner import (
    BOOTSTRAP_SEED,
    DEFAULT_CONCURRENCY,
    System,
    UsageError,
    build_backend,
    category_table,
    fmt,
    new_run_id,
    normalize_spec,
    pins,
    pins_section,
    print_plan,
    read_transcripts,
    results_table,
    run_grid,
    split_hf,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
DEFAULT_OUT = Path("results") / "ablation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ablation.py", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--model", action="append", help="repeatable (default: config)")
    parser.add_argument(
        "--strategy", action="append", choices=STRATEGIES, help="repeatable (default: all three)"
    )
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--limit", type=int, help="only the first N scenarios")
    parser.add_argument("--judge", help="judge model (default: config; 'mock' = mock:judge)")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel requests"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no calls")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root")
    parser.add_argument("--run-id", help="run directory name (default: a timestamp)")
    parser.add_argument(
        "--resume", action="store_true", help="continue --run-id, skipping finished rows"
    )
    return parser


def _revision(model: str) -> str | None:
    """API models are pinned by their dated id; an hf: model records its requested revision."""
    return split_hf(model)[1] if model.startswith("hf:") else None


def _check_resumable(run_dir: Path, judge: str) -> None:
    """Rows from an earlier attempt must come from the same judge and rubric."""
    rubric = rubric_sha256()
    for row in read_transcripts(run_dir / "transcripts.jsonl"):
        if row.get("judge_model") != judge or row.get("rubric_sha256") != rubric:
            raise UsageError(
                f"{run_dir} was scored by {row.get('judge_model')} with rubric "
                f"{row.get('rubric_sha256')}; this run uses {judge} with {rubric}"
            )


def _results_md(results: dict[str, Any]) -> str:
    p = results["pins"]
    cells = results["cells"]
    for c in cells:
        c["_extra"] = "yes" if c["shares_judge_family"] else "no"
    lines = [
        f"# Prompt-ceiling ablation `{results['run_id']}`",
        "",
        f"{len(results['scenario_ids'])} scenarios, the same IDs in every cell. Judge "
        f"`{p['judge_model']}`, rubric sha256 `{p['rubric_sha256']}`, eval code `{p['git_sha']}`. "
        "Spec-adherence is the mean with a 95% bootstrap CI; Robustness is the mean 1-5 judge "
        "score. Errors and refusals are scored as failures.",
        "",
        *results_table(cells, extra="Shares judge family"),
        "",
        "A cell marked as sharing the judge family was graded by a model from its own vendor; "
        "read it with that bias in mind (GOTCHA G12).",
        "",
        "## Spec-adherence by category",
        "",
        *category_table(cells),
        "",
        "## Best vs second-best strategy per model",
        "",
        "| Model | Best | Second | Delta (Spec-adherence) |",
        "|---|---|---|---|",
    ]
    for c in cells:
        c.pop("_extra", None)
    for model, d in results["deltas"].items():
        lines.append(f"| {model} | {d['best']} | {d['second']} | {fmt(d['delta'])} |")
    plateau = results["plateau"]
    meeting = ", ".join(f"{m['model']} x {m['strategy']}" for m in plateau["cells_meeting_bar"])
    verdict = (
        "**plateau**: no cell meets the reliability bar."
        if plateau["plateau"]
        else f"**no plateau**: {meeting} meet(s) the reliability bar."
    )
    lines += [
        "",
        "## Plateau check",
        "",
        f"Reliability bar: Spec-adherence >= {SPEC_ADHERENCE_BAR:.2f} and mean Robustness >= "
        f"{ROBUSTNESS_BAR} (inclusive). Verdict: {verdict}",
        "",
        *pins_section(p),
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    config = load_models_config()
    if args.resume and not args.run_id:
        raise UsageError("--resume needs --run-id")
    scenarios = load_scenarios(args.eval_set, args.limit)
    judge_spec = normalize_spec(args.judge or config["judge"]["model"])
    models = [normalize_spec(m) for m in (args.model or config["ablation"]["models"])]
    strategies = list(dict.fromkeys(args.strategy or STRATEGIES))
    systems = [System(m, s) for m in dict.fromkeys(models) for s in strategies]
    if args.dry_run:
        print_plan(systems, scenarios, judge_spec, args.eval_set)
        return 0

    run_id = args.run_id or new_run_id()
    run_dir = args.out / run_id
    if run_dir.exists() and not args.resume:
        raise UsageError(f"{run_dir} already exists; pass --resume to continue it")
    dirty = check_clean_tree(args.allow_dirty)
    model_ids = list(dict.fromkeys(models))
    revisions = {m: _revision(m) for m in model_ids}
    backends = {m: build_backend(m) for m in model_ids}
    judge = build_backend(judge_spec)
    if run_dir.exists():
        _check_resumable(run_dir, judge.name)

    run_dir.mkdir(parents=True, exist_ok=True)
    rows = run_grid(
        run_id=run_id,
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        judge=judge,
        transcripts=run_dir / "transcripts.jsonl",
        concurrency=args.concurrency,
    )
    cells = aggregate_scores(rows, seed=BOOTSTRAP_SEED)
    judge_family = model_family(judge.name)
    for cell in cells:
        cell["shares_judge_family"] = model_family(cell["model"]) == judge_family
    results = {
        "run_id": run_id,
        "pins": pins(
            eval_set=args.eval_set,
            judge=judge,
            systems=systems,
            backends=backends,
            revisions=revisions,
            git_dirty=dirty,
        ),
        "scenario_ids": [s.id for s in scenarios],
        "cells": cells,
        "deltas": strategy_deltas(cells),
        "plateau": plateau_verdict(cells),
    }
    write_json(run_dir / "results.json", results)
    (run_dir / "results.md").write_text(_results_md(results), encoding="utf-8")
    print(f"wrote {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BackendError as exc:
        print(
            f"error: run stopped: {exc}; finished rows are saved, rerun with --resume",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
