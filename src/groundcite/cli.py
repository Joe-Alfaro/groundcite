"""``eval.py``: score one or more models on an eval set in one command (R10, R11, R13, R20, R26).

With exactly one ``hf:`` model and no ``--no-base``, the base model is added automatically (from
``--base`` or the repo's ``adapter_config.json``) and the table has three rows: base with the
zero-shot prompt, base with the best ablation strategy, and the tuned model (GOTCHA G16, G39).

The same run can be split in two (D-078): ``--generate-only`` generates every reply with no
judge and writes ``generations.jsonl`` plus ``generation_pins.json`` (on a Kaggle GPU), and
``--judge-from <generations.jsonl>`` judges those replies elsewhere with no generation or Hub
calls. For the same replies and judge, its ``results.json`` equals the one-command run's apart
from ``run_id``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from groundcite import hub, versions
from groundcite.backends import BackendError
from groundcite.config import load_models_config
from groundcite.evalset import EvalSetError, Scenario, load_eval_set
from groundcite.hub import HubError
from groundcite.keys import MissingKeyError
from groundcite.prompts import STRATEGIES, prompt_sha256
from groundcite.report import aggregate_scores
from groundcite.runner import (
    BOOTSTRAP_SEED,
    DEFAULT_CONCURRENCY,
    MANIFEST,
    RecordedBackend,
    Retrying,
    System,
    UsageError,
    build_backend,
    category_table,
    discard_if_empty,
    display_path,
    generate_grid,
    model_decoding,
    model_pins,
    new_run_id,
    normalize_spec,
    pins,
    pins_section,
    print_plan,
    results_table,
    run_grid,
    run_manifest,
    sha256_file,
    split_hf,
    stop_message,
    write_json,
)

DEFAULT_OUT = Path("results")
GENERATIONS = "generations.jsonl"
GENERATION_PINS = "generation_pins.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.py", description="Score models on an eval set with the pinned judge."
    )
    parser.add_argument(
        "--model",
        action="append",
        help="hf:<repo>[@rev] (a bare org/repo means hf:), anthropic:<m>, openai:<m> or "
        "mock:<name>; repeat to add rows (required unless --judge-from)",
    )
    parser.add_argument(
        "--eval-set", type=Path, help="JSONL, JSON or CSV (required unless --judge-from)"
    )
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
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--generate-only",
        action="store_true",
        help=f"generate every reply with no judge; write {GENERATIONS} and {GENERATION_PINS}",
    )
    split.add_argument(
        "--judge-from",
        type=Path,
        metavar="GENERATIONS",
        help=f"judge the replies in a --generate-only run's {GENERATIONS} (no generation)",
    )
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


def _plan_systems(
    args: argparse.Namespace, scenarios: list[Scenario], judge_spec: str | None
) -> tuple[list[System], dict[str, str | None], dict[str, Retrying], bool] | None:
    """The systems, resolved revisions and subject backends of a generating run; None after a
    dry run. ``judge_spec`` None means no judge (``--generate-only``): no judge key is needed."""
    models = [normalize_spec(m) for m in args.model]
    base_rows = None
    if len(models) == 1:
        base_rows = _base_systems(args, models[0], plan_only=True)
    systems = (base_rows or []) + [System(m, args.strategy) for m in models]
    if args.dry_run:
        print_plan(systems, scenarios, judge_spec or "none (--generate-only)", args.eval_set)
        return None

    dirty = check_clean_tree(args.allow_dirty)
    # Keys first, so a missing key fails before any Hub lookup or request.
    for model in models:
        build_backend(model)
    if judge_spec is not None:
        build_backend(judge_spec)
    if len(models) == 1:
        base_rows = _base_systems(args, models[0], plan_only=False)
    systems = (base_rows or []) + [System(m, args.strategy) for m in models]
    if len(set(systems)) != len(systems):
        raise UsageError("the same model and strategy appear twice")
    all_models = list(dict.fromkeys(s.model for s in systems))
    revisions = _revisions(all_models)
    backends = {m: build_backend(m, revision=revisions.get(m)) for m in all_models}
    return systems, revisions, backends, dirty


def _finish(
    *,
    run_id: str,
    run_dir: Path,
    eval_set: Path,
    systems: list[System],
    scenarios: list[Scenario],
    backends: dict[str, Retrying],
    revisions: dict[str, str | None],
    judge: Retrying,
    dirty: bool,
    concurrency: int,
    generated: dict[tuple[str, str, str], tuple[str, str | None, dict[str, Any] | None]]
    | None = None,
) -> None:
    """Judge the grid and write transcripts.jsonl, results.json and results.md into run_dir."""
    write_json(run_dir / MANIFEST, run_manifest(eval_set, scenarios, judge.name))
    transcripts = run_dir / "transcripts.jsonl"
    try:
        rows = run_grid(
            run_id=run_id,
            systems=systems,
            scenarios=scenarios,
            backends=backends,
            revisions=revisions,
            judge=judge,
            transcripts=transcripts,
            concurrency=concurrency,
            generated=generated,
        )
    except BaseException:
        discard_if_empty(run_dir)
        raise
    cells = aggregate_scores(rows, seed=BOOTSTRAP_SEED)
    p = pins(
        eval_set=eval_set,
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


def run(args: argparse.Namespace) -> int:
    if args.judge_from is not None:
        return run_judge_from(args)
    if not args.model or args.eval_set is None:
        raise UsageError("--model and --eval-set are required (unless --judge-from)")
    scenarios = load_scenarios(args.eval_set, args.limit)
    judge_spec = None
    if not args.generate_only:
        judge_spec = normalize_spec(args.judge or load_models_config()["judge"]["model"])
    planned = _plan_systems(args, scenarios, judge_spec)
    if planned is None:
        return 0
    systems, revisions, backends, dirty = planned

    if judge_spec is None:
        return _generate_only(args, scenarios, systems, revisions, backends, dirty)

    judge = build_backend(judge_spec)
    run_id = new_run_id()
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    _finish(
        run_id=run_id,
        run_dir=run_dir,
        eval_set=args.eval_set,
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        judge=judge,
        dirty=dirty,
        concurrency=args.concurrency,
    )
    print(f"wrote {run_dir}")
    return 0


# ---------------------------------------------------------------------------------------------
# Split mode: --generate-only (GPU) and --judge-from (anywhere)
# ---------------------------------------------------------------------------------------------


def _generate_only(
    args: argparse.Namespace,
    scenarios: list[Scenario],
    systems: list[System],
    revisions: dict[str, str | None],
    backends: dict[str, Retrying],
    dirty: bool,
) -> int:
    rows = generate_grid(
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        concurrency=args.concurrency,
    )
    # The pins the one-command run records, minus the judge's (R13).
    models = list(dict.fromkeys(s.model for s in systems))
    generation_pins = {
        "eval_set": {"path": display_path(args.eval_set), "sha256": sha256_file(args.eval_set)},
        "scenario_ids": [s.id for s in scenarios],
        "systems": [{"model": s.model, "strategy": s.strategy} for s in systems],
        "models": model_pins(models, backends, revisions),
        "decoding": model_decoding(models, backends),
        "git_sha": versions.git_sha(),
        "git_dirty": dirty,
        "library_versions": versions.library_versions(),
        "prompt_sha256": {s.strategy: prompt_sha256(s.strategy) for s in systems},
    }
    run_dir = args.out / new_run_id()
    run_dir.mkdir(parents=True)
    write_json(run_dir / GENERATION_PINS, generation_pins)
    with (run_dir / GENERATIONS).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {run_dir / GENERATIONS}")
    return 0


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"can't read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise UsageError(f"{path} is not JSON: {exc.msg}") from None


def _recorded_eval_set(recorded: Any) -> Path:
    """The generating run's eval set, refused if its bytes changed since."""
    if not isinstance(recorded, dict) or not recorded.get("path") or not recorded.get("sha256"):
        raise UsageError(f"{GENERATION_PINS} has no eval_set path and sha256")
    path = Path(recorded["path"])
    if not path.is_absolute():
        path = versions.REPO_ROOT / path
    if not path.is_file():
        raise UsageError(f"the recorded eval set is missing: {path}")
    if sha256_file(path) != recorded["sha256"]:
        raise UsageError(
            f"the eval set {path} changed since the replies were generated (sha256 "
            f"{sha256_file(path)}, recorded {recorded['sha256']}); regenerate them"
        )
    return path


def run_judge_from(args: argparse.Namespace) -> int:
    given = [
        flag
        for flag, value in (
            ("--model", args.model),
            ("--eval-set", args.eval_set),
            ("--limit", args.limit),
            ("--base", args.base),
            ("--no-base", args.no_base),
        )
        if value
    ]
    if given:
        raise UsageError(
            f"--judge-from takes the models and eval set from {GENERATION_PINS}; drop "
            + ", ".join(given)
        )
    generations: Path = args.judge_from
    pins_path = generations.parent / GENERATION_PINS
    if not generations.is_file():
        raise UsageError(f"--judge-from: no such file: {generations}")
    if not pins_path.is_file():
        raise UsageError(f"--judge-from: {GENERATION_PINS} is missing next to {generations}")
    recorded = _read_json(pins_path)
    if not isinstance(recorded, dict):
        raise UsageError(f"{pins_path} is not a JSON object")
    eval_set = _recorded_eval_set(recorded.get("eval_set"))
    by_id = {s.id: s for s in load_scenarios(eval_set, None)}
    try:
        scenarios = [by_id[i] for i in recorded["scenario_ids"]]
        systems = [System(s["model"], s["strategy"]) for s in recorded["systems"]]
        models_info = {m["model"]: m for m in recorded["models"]}
        decoding = dict(recorded["decoding"])
    except (KeyError, TypeError) as exc:
        raise UsageError(f"{pins_path} is incomplete or names an unknown scenario: {exc}") from None
    for strategy, sha in recorded.get("prompt_sha256", {}).items():
        if strategy in STRATEGIES and prompt_sha256(strategy) != sha:
            raise UsageError(
                f"the {strategy} prompt changed since the replies were generated; regenerate them"
            )

    generated: dict[tuple[str, str, str], tuple[str, str | None, dict[str, Any] | None]] = {}
    try:
        lines = generations.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                row = json.loads(line)
                key = (row["model"], row["strategy"], row["scenario_id"])
                generated[key] = (row["reply"], row["generation_error"], row["decoding"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise UsageError(f"{generations} has a malformed row: {exc}") from None
    missing = [
        (s.model, s.strategy, sc.id)
        for s in systems
        for sc in scenarios
        if (s.model, s.strategy, sc.id) not in generated
    ]
    if missing:
        raise UsageError(f"{generations} has no reply for {len(missing)} cells, e.g. {missing[0]}")

    dirty = check_clean_tree(args.allow_dirty)
    judge = build_backend(normalize_spec(args.judge or load_models_config()["judge"]["model"]))
    models = list(dict.fromkeys(s.model for s in systems))
    revisions = {m: (models_info.get(m) or {}).get("model_revision") for m in models}
    backends = {
        m: Retrying(
            RecordedBackend(
                m, decoding.get(m), (models_info.get(m) or {}).get("chat_template_sha256")
            )
        )
        for m in models
    }
    run_id = new_run_id()
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    # Keep what generated the replies next to the scores (R13).
    shutil.copy2(pins_path, run_dir / GENERATION_PINS)
    _finish(
        run_id=run_id,
        run_dir=run_dir,
        eval_set=eval_set,
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        judge=judge,
        dirty=dirty,
        concurrency=args.concurrency,
        generated=generated,
    )
    print(f"wrote {run_dir} (judged {display_path(generations)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BackendError as exc:
        if exc.stops_run:
            print(stop_message(exc), file=sys.stderr)
            return 2
        print(f"error: run stopped: {exc}", file=sys.stderr)
        return 1
    except HubError as exc:
        print(f"error: run stopped: {exc}", file=sys.stderr)
        return 1
