"""``eval.py``: score one or more models on an eval set in one command (R10, R11, R13, R20, R26).

With exactly one ``hf:`` model and no ``--no-base``, the base model is added automatically (from
``--base`` or the repo's ``adapter_config.json``) and the table has three rows: base with the
zero-shot prompt, base with the best ablation strategy, and the tuned model (GOTCHA G16, G39).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from groundcite import hub, versions
from groundcite.backends import BackendError
from groundcite.config import load_models_config
from groundcite.evalset import EvalSetError, Scenario, load_eval_set
from groundcite.hub import HubError
from groundcite.keys import MissingKeyError
from groundcite.prompts import STRATEGIES
from groundcite.report import aggregate_scores
from groundcite.runner import (
    BOOTSTRAP_SEED,
    DEFAULT_CONCURRENCY,
    Retrying,
    System,
    UsageError,
    build_backend,
    category_table,
    new_run_id,
    normalize_spec,
    pins,
    pins_section,
    print_plan,
    results_table,
    run_grid,
    split_hf,
    write_json,
)

DEFAULT_OUT = Path("results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.py", description="Score models on an eval set with the pinned judge."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="hf:<repo>[@rev] (a bare org/repo means hf:), anthropic:<m>, openai:<m> or "
        "mock:<name>; repeat to add rows",
    )
    parser.add_argument("--eval-set", type=Path, required=True, help="JSONL, JSON or CSV")
    parser.add_argument(
        "--judge", help="judge model (default: config [judge] model; 'mock' = mock:judge)"
    )
    parser.add_argument("--base", help="base model for the automatic base rows (skips the lookup)")
    parser.add_argument("--no-base", action="store_true", help="don't add the base rows")
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default="zero_shot", help="prompt strategy"
    )
    parser.add_argument("--limit", type=int, help="only the first N scenarios")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel requests"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no calls")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on a dirty tree (recorded in results)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root")
    return parser


def load_scenarios(path: Path, limit: int | None) -> list[Scenario]:
    try:
        scenarios = load_eval_set(path)
    except EvalSetError as exc:
        raise UsageError(f"malformed eval set {path}: {exc}") from None
    except OSError as exc:
        raise UsageError(f"can't read eval set {path}: {exc}") from None
    if limit is not None:
        if limit < 1:
            raise UsageError("--limit must be at least 1")
        scenarios = scenarios[:limit]
    return scenarios


def check_clean_tree(allow_dirty: bool) -> bool:
    dirty = versions.git_dirty()
    if dirty and not allow_dirty:
        raise UsageError(
            "the working tree is dirty (tracked files have changes); commit them or pass "
            "--allow-dirty, which is recorded in results.json"
        )
    return dirty


def build_backends(
    models: list[str], judge: str, revisions: dict[str, str | None]
) -> tuple[dict[str, Retrying], Retrying]:
    """Every subject backend and the judge; a missing key raises before any request."""
    backends = {m: build_backend(m, revision=revisions.get(m)) for m in models}
    return backends, build_backend(judge)


def _base_systems(args: argparse.Namespace, tuned: str, plan_only: bool) -> list[System] | None:
    """The two base rows for one ``hf:`` model, or None when they are off."""
    if args.no_base or not tuned.startswith("hf:"):
        return None
    if args.base:
        base = normalize_spec(args.base)
    elif plan_only:
        base = f"hf:<base from adapter_config.json of {split_hf(tuned)[0]}>"
    else:
        repo, rev = split_hf(tuned)
        config = hub.fetch_adapter_config(repo, rev)
        name = (config or {}).get("base_model_name_or_path")
        if not name:
            raise UsageError(
                f"{tuned} has no adapter_config.json naming its base model (a merged or full "
                "model); pass --base <model> to name the base, or --no-base to skip the base rows"
            )
        base = f"hf:{name}"
    best = load_models_config()["eval"]["best_strategy"]
    return [System(base, "zero_shot"), System(base, best)]


def _revisions(models: list[str]) -> dict[str, str | None]:
    """Resolved commit SHAs for hf: models (R13); API and mock models have none."""
    revisions: dict[str, str | None] = {}
    for model in models:
        if model.startswith("hf:"):
            repo, rev = split_hf(model)
            revisions[model] = hub.resolve_revision(repo, rev)
        else:
            revisions[model] = None
    return revisions


def _results_md(run_id: str, rows: list[dict[str, Any]], p: dict[str, Any]) -> str:
    lines = [
        f"# Eval results `{run_id}`",
        "",
        f"Judge `{p['judge_model']}`, rubric sha256 `{p['rubric_sha256']}`, eval code "
        f"`{p['git_sha']}`. Spec-adherence is the mean with a 95% bootstrap CI; Robustness is "
        "the mean 1-5 judge score.",
        "",
        *results_table(rows),
        "",
        "## Spec-adherence by category",
        "",
        *category_table(rows),
        "",
        *pins_section(p),
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.eval_set, args.limit)
    judge_spec = normalize_spec(args.judge or load_models_config()["judge"]["model"])
    models = [normalize_spec(m) for m in args.model]

    base_rows = None
    if len(models) == 1:
        base_rows = _base_systems(args, models[0], plan_only=True)
    systems = (base_rows or []) + [System(m, args.strategy) for m in models]
    if args.dry_run:
        print_plan(systems, scenarios, judge_spec, args.eval_set)
        return 0

    dirty = check_clean_tree(args.allow_dirty)
    # Keys first, so a missing key fails before any Hub lookup or request.
    build_backends(models, judge_spec, {})
    if len(models) == 1:
        base_rows = _base_systems(args, models[0], plan_only=False)
    systems = (base_rows or []) + [System(m, args.strategy) for m in models]
    if len(set(systems)) != len(systems):
        raise UsageError("the same model and strategy appear twice")
    all_models = list(dict.fromkeys(s.model for s in systems))
    revisions = _revisions(all_models)
    backends, judge = build_backends(all_models, judge_spec, revisions)

    run_id = new_run_id()
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    transcripts = run_dir / "transcripts.jsonl"
    rows = run_grid(
        run_id=run_id,
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        judge=judge,
        transcripts=transcripts,
        concurrency=args.concurrency,
    )
    cells = aggregate_scores(rows, seed=BOOTSTRAP_SEED)
    p = pins(
        eval_set=args.eval_set,
        judge=judge,
        systems=systems,
        backends=backends,
        revisions=revisions,
        git_dirty=dirty,
    )
    results = {
        "run_id": run_id,
        "pins": p,
        "scenario_ids": [s.id for s in scenarios],
        "rows": cells,
    }
    write_json(run_dir / "results.json", results)
    (run_dir / "results.md").write_text(_results_md(run_id, cells, p), encoding="utf-8")
    print(f"wrote {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (BackendError, HubError) as exc:
        print(f"error: run stopped: {exc}", file=sys.stderr)
        return 1
