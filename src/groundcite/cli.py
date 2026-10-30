"""``eval.py``: score one or more models on an eval set in one command (R10, R11, R13, R20, R26).

With exactly one ``hf:`` model and no ``--no-base``, the base model is added automatically (from
``--base`` or the repo's ``adapter_config.json``) and the table has three rows: base with the
zero-shot prompt, base with the best ablation strategy, and the tuned model (GOTCHA G16, G39).

The same run can be split in two (D-078): ``--generate-only`` generates every reply with no
judge and writes ``generations.jsonl`` plus ``generation_pins.json`` (on a Kaggle GPU), and
``--judge-from <generations.jsonl>`` judges those replies elsewhere with no generation or Hub
calls. For the same replies and judge, its ``results.json`` equals the one-command run's apart
from ``run_id``. ``--judge-from GENERATIONS --resume RUN_DIR`` continues an interrupted judging
run in RUN_DIR, judging only the rows it lacks.

Every judging run records each judge call's token usage in its transcript row and the run's
total, with the actual dollars from ``[prices]``, in ``spend.json``. ``--dry-run`` makes no calls
and prints the judge call count and an estimated cost; ``--max-spend USD`` stops the run before a
call that could push the recorded spend past USD (D-123).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

from groundcite import hub, spend, versions
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
    MANIFEST_LABELS,
    RecordedBackend,
    Retrying,
    System,
    UsageError,
    build_backend,
    category_table,
    check_resumable,
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
    read_transcripts,
    results_table,
    row_key,
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
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="with --judge-from: continue an interrupted judging run in RUN_DIR (not --out)",
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        metavar="USD",
        help="stop before a judge call that could push the run's recorded spend past USD",
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
        revision = (config or {}).get("revision") or _configured_base_revision(name)
        base = f"hf:{name}@{revision}" if revision else f"hf:{name}"
    best = load_models_config()["eval"]["best_strategy"]
    return [System(base, "zero_shot"), System(base, best)]


def _configured_base_revision(name: str) -> str | None:
    """train/config.toml's pinned base commit when the adapter's base is the configured base
    (GOTCHA G27), so the base rows load the weights training used rather than ``main``."""
    path = versions.REPO_ROOT / "train" / "config.toml"
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        model = tomllib.load(fh).get("model", {})
    if model.get("base") != name:
        return None
    revision = model.get("revision")
    return str(revision) if revision else None


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
        if judge_spec is not None:
            _print_one_shot_estimate(judge_spec, systems, scenarios)
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
    meter: spend.MeteredJudge,
    dirty: bool,
    concurrency: int,
    generated: dict[tuple[str, str, str], tuple[str, str | None, dict[str, Any] | None]]
    | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge the grid and write transcripts.jsonl, results.json, results.md and spend.json into
    run_dir; return the spend. Stopped early, it keeps the finished rows and their spend."""
    write_json(run_dir / MANIFEST, manifest or run_manifest(eval_set, scenarios, judge.name))
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
            meter=meter,
        )
    except BaseException:
        if meter.usage or (run_dir / spend.SPEND).is_file():
            spend.record_spend(run_dir, judge.name, meter.usage)
        else:
            discard_if_empty(run_dir)
        raise
    recorded = spend.record_spend(run_dir, judge.name, meter.usage)
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
    return recorded


def _print_one_shot_estimate(
    judge_spec: str, systems: list[System], scenarios: list[Scenario]
) -> None:
    """The dry run's cost line. The replies don't exist yet, so each judge prompt is sized with an
    empty reply: the rubric and scenario dominate it."""
    prompts = [spend.judge_prompt(sc, "") for sc in scenarios]
    spend.print_estimate(judge_spec, spend.estimate_cost(judge_spec, prompts * len(systems)))


def run(args: argparse.Namespace) -> int:
    if args.judge_from is not None:
        return run_judge_from(args)
    if args.resume is not None:
        raise UsageError("--resume RUN_DIR continues a --judge-from run; pass --judge-from too")
    if not args.model or args.eval_set is None:
        raise UsageError("--model and --eval-set are required (unless --judge-from)")
    if args.generate_only and args.max_spend is not None:
        raise UsageError("--max-spend caps judge spend; --generate-only makes no judge calls")
    scenarios = load_scenarios(args.eval_set, args.limit)
    judge_spec = None
    if not args.generate_only:
        judge_spec = normalize_spec(args.judge or load_models_config()["judge"]["model"])
        spend.check_limit(judge_spec, args.max_spend)
    planned = _plan_systems(args, scenarios, judge_spec)
    if planned is None:
        return 0
    systems, revisions, backends, dirty = planned

    if judge_spec is None:
        return _generate_only(args, scenarios, systems, revisions, backends, dirty)

    judge = build_backend(judge_spec)
    meter = spend.MeteredJudge(judge, limit=args.max_spend)
    run_id = new_run_id()
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    recorded = _finish(
        run_id=run_id,
        run_dir=run_dir,
        eval_set=args.eval_set,
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        judge=judge,
        meter=meter,
        dirty=dirty,
        concurrency=args.concurrency,
    )
    print(f"wrote {run_dir} ({spend.spend_note(recorded)})")
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

    judge_spec = normalize_spec(args.judge or load_models_config()["judge"]["model"])
    spend.check_limit(judge_spec, args.max_spend)
    manifest = run_manifest(eval_set, scenarios, judge_spec)
    manifest["generations_sha256"] = sha256_file(generations)
    done: set[tuple[str, str, str, str | None]] = set()
    if args.resume is not None:
        _check_resume(args.resume, manifest, generated, generations, pins_path)
        done = {row_key(r) for r in read_transcripts(args.resume / "transcripts.jsonl")}
    if args.dry_run:
        hashes = {s.strategy: prompt_sha256(s.strategy) for s in systems}
        todo = [
            (sc, generated[(s.model, s.strategy, sc.id)])
            for s in systems
            for sc in scenarios
            if (s.model, s.strategy, sc.id, hashes[s.strategy]) not in done
        ]
        # A reply whose generation failed is scored as a failure with no judge call.
        prompts = [spend.judge_prompt(sc, reply) for sc, (reply, error, _) in todo if error is None]
        total = len(systems) * len(scenarios)
        print(f"generations: {display_path(generations)} ({total} replies)")
        print(f"judge: {judge_spec}")
        print(f"rows to judge: {len(todo)} ({total - len(todo)} already done)")
        print(f"judge calls: {len(prompts)}")
        spend.print_estimate(judge_spec, spend.estimate_cost(judge_spec, prompts))
        print("dry run: no requests made, nothing written")
        return 0

    dirty = check_clean_tree(args.allow_dirty)
    judge = build_backend(judge_spec)
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
    if args.resume is not None:
        run_dir = args.resume
        run_id = run_dir.name
        meter = spend.MeteredJudge(judge, limit=args.max_spend, spent=spend.recorded_cost(run_dir))
    else:
        meter = spend.MeteredJudge(judge, limit=args.max_spend)
        run_id = new_run_id()
        run_dir = args.out / run_id
        run_dir.mkdir(parents=True)
    # Keep what generated the replies next to the scores (R13).
    shutil.copy2(pins_path, run_dir / GENERATION_PINS)
    recorded = _finish(
        run_id=run_id,
        run_dir=run_dir,
        eval_set=eval_set,
        systems=systems,
        scenarios=scenarios,
        backends=backends,
        revisions=revisions,
        judge=judge,
        meter=meter,
        dirty=dirty,
        concurrency=args.concurrency,
        generated=generated,
        manifest=manifest,
    )
    print(f"wrote {run_dir} (judged {display_path(generations)}; {spend.spend_note(recorded)})")
    return 0


def _check_resume(
    run_dir: Path,
    manifest: dict[str, Any],
    generated: dict[tuple[str, str, str], tuple[str, str | None, dict[str, Any] | None]],
    generations: Path,
    pins_path: Path,
) -> None:
    """Refuse to resume ``run_dir`` unless it was started from the same generations, eval set,
    judge and rubric. A run started before run.json recorded the generations' sha256 is checked
    reply by reply, and by its copy of generation_pins.json."""
    if not run_dir.is_dir():
        raise UsageError(f"--resume: no such run directory: {run_dir}")
    labels = dict(MANIFEST_LABELS)
    recorded = _read_json(run_dir / MANIFEST) if (run_dir / MANIFEST).is_file() else {}
    legacy = not isinstance(recorded, dict) or "generations_sha256" not in recorded
    if not legacy:
        labels["generations_sha256"] = "generations (sha256)"
    try:
        check_resumable(run_dir, manifest, labels)
    except json.JSONDecodeError as exc:
        raise UsageError(f"{run_dir} has a malformed row: {exc.msg}") from None
    if not legacy:
        return
    copied = run_dir / GENERATION_PINS
    if copied.is_file() and _read_json(copied) != _read_json(pins_path):
        raise UsageError(
            f"{run_dir} was judged from other generations ({GENERATION_PINS} differs); resume "
            "only continues the same run"
        )
    for row in read_transcripts(run_dir / "transcripts.jsonl"):
        key = (row["model"], row["strategy"], row["scenario_id"])
        reply = generated.get(key)
        if reply is None or (row.get("reply"), row.get("generation_error")) != reply[:2]:
            raise UsageError(
                f"{run_dir} was judged from other generations: its {key} reply is not the one "
                f"in {generations}; resume only continues the same run"
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except spend.SpendCapReached as exc:
        print(f"error: run stopped: {exc}", file=sys.stderr)
        return 1
    except BackendError as exc:
        if exc.stops_run:
            print(stop_message(exc), file=sys.stderr)
            return 2
        print(f"error: run stopped: {exc}", file=sys.stderr)
        return 1
    except HubError as exc:
        print(f"error: run stopped: {exc}", file=sys.stderr)
        return 1
