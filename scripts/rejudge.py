#!/usr/bin/env python3
"""Re-judge a finished run's stored replies with another judge (D-080, R11).

    uv run scripts/rejudge.py RUN_DIR --judge SPEC --out NEW_DIR [--dry-run] [--resume]

RUN_DIR is a finished run (results.json + transcripts.jsonl) from eval.py, run_ablation.py or
run_rollouts.py. Its eval set (or conversation scripts) is reloaded from ``pins.eval_set`` and
refused if its SHA-256 changed. Every stored reply is checked and judged again with --judge; no
reply is generated and nothing touches the Hub. A reply whose generation failed stays a scored
failure with no judge call. NEW_DIR is the new run directory itself (its name is the run_id); it
gets results.json, results.md and transcripts.jsonl in the source run's shape, with the new
judge's pins and ``rejudged_from`` naming the source run and judge.

Rows are appended as they are judged, so an interrupted re-judge continues with --resume into
the same NEW_DIR, judging only the missing rows; --resume refuses a NEW_DIR started from another
source run or judge. Without --resume a non-empty NEW_DIR is refused. --dry-run prints
``judge calls: N`` and an estimated cost from ``[prices]`` and writes nothing, with no key
needed. Each row records its judge calls' usage, NEW_DIR/spend.json the actual spend across
resumed invocations, and --max-spend USD stops before a call that could push it past USD (D-123).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

from groundcite import spend, versions
from groundcite.ablation import model_family, plateau_verdict, strategy_deltas
from groundcite.backends import BackendError
from groundcite.checks import run_checks
from groundcite.cli import load_scenarios
from groundcite.config import decoding_params
from groundcite.evalset import EvalSetError, Scenario
from groundcite.judge import judge_reply, rubric_sha256
from groundcite.keys import MissingKeyError
from groundcite.report import aggregate_scores, transcript_row
from groundcite.rollout import load_scripts, meets_rollout_bar, turn_scenario
from groundcite.runner import (
    BOOTSTRAP_SEED,
    DEFAULT_CONCURRENCY,
    MANIFEST,
    Retrying,
    UsageError,
    _generation_failure,
    build_backend,
    normalize_spec,
    read_transcripts,
    sha256_file,
    stop_message,
    write_json,
)

SCRIPTS_DIR = Path(__file__).resolve().parent
TRANSCRIPTS = "transcripts.jsonl"
RESULTS = "results.json"
# Copied next to the scores when the source was judged from a split (--judge-from) run.
GENERATION_PINS = "generation_pins.json"
# Source-row fields a re-judge carries over unchanged.
KEPT_FIELDS = (
    "model",
    "model_revision",
    "strategy",
    "scenario_id",
    "category",
    "gold",
    "reply",
)

RowKey = tuple[str, str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rejudge.py", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("run_dir", type=Path, help="the finished run to re-judge")
    parser.add_argument("--judge", required=True, help="the new judge ('mock' = mock:judge)")
    parser.add_argument("--out", type=Path, required=True, help="the new run directory")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel judge calls"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the call count; no calls")
    parser.add_argument(
        "--resume", action="store_true", help="continue an interrupted re-judge into --out"
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        metavar="USD",
        help="stop before a judge call that could push NEW_DIR's recorded spend past USD",
    )
    return parser


def row_key(row: Mapping[str, Any]) -> RowKey:
    return (row["model"], row["strategy"], row["scenario_id"])


# ---------------------------------------------------------------------------------------------
# The source run
# ---------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"can't read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise UsageError(f"{path} is not JSON: {exc.msg}") from None


def load_source(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The source run's results.json and transcript rows; a run that never finished is refused."""
    if not (run_dir / RESULTS).is_file():
        raise UsageError(f"{run_dir} has no {RESULTS}; only a finished run can be re-judged")
    if not (run_dir / TRANSCRIPTS).is_file():
        raise UsageError(f"{run_dir} has no {TRANSCRIPTS}")
    results = _read_json(run_dir / RESULTS)
    if not isinstance(results, dict) or not isinstance(results.get("pins"), dict):
        raise UsageError(f"{run_dir / RESULTS} has no pins")
    try:
        rows = read_transcripts(run_dir / TRANSCRIPTS)
    except json.JSONDecodeError as exc:
        raise UsageError(f"{run_dir / TRANSCRIPTS} has a malformed row: {exc.msg}") from None
    if not rows:
        raise UsageError(f"{run_dir / TRANSCRIPTS} has no rows")
    keys = [row_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise UsageError(f"{run_dir / TRANSCRIPTS} has duplicate rows")
    return results, rows


def recorded_eval_set(pins: Mapping[str, Any]) -> Path:
    """The eval set (or conversation scripts) the run used, refused if its bytes changed."""
    recorded = pins.get("eval_set")
    if not isinstance(recorded, dict) or not recorded.get("path") or not recorded.get("sha256"):
        raise UsageError("the source run's pins have no eval_set path and sha256")
    path = Path(recorded["path"])
    if not path.is_absolute():
        path = versions.REPO_ROOT / path
    if not path.is_file():
        raise UsageError(f"the recorded eval set is missing: {path}")
    actual = sha256_file(path)
    if actual != recorded["sha256"]:
        raise UsageError(
            f"the eval set {path} changed since the run (sha256 {actual}, recorded "
            f"{recorded['sha256']}); its replies can't be re-judged against it"
        )
    return path


def row_scenarios(
    results: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], eval_set: Path
) -> list[Scenario]:
    """The scenario each stored reply answered, in row order. A rollout turn's scenario is rebuilt
    from its script and the conversation's stored earlier replies, as the rollout rendered it."""
    if "conversation_ids" in results:
        try:
            scripts = {s.id: s for s in load_scripts(eval_set)}
        except EvalSetError as exc:
            raise UsageError(f"malformed conversation scripts {eval_set}: {exc}") from None
        replies: dict[str, dict[int, str]] = {}
        for row in rows:
            replies.setdefault(row["conversation_id"], {})[int(row["turn_index"])] = row["reply"]
        scenarios = []
        for row in rows:
            cid, turn = row["conversation_id"], int(row["turn_index"])
            if cid not in scripts:
                raise UsageError(f"conversation {cid} is not in {eval_set}")
            try:
                earlier = [replies[cid][i] for i in range(turn)]
            except KeyError:
                raise UsageError(f"conversation {cid} is missing turns before {turn}") from None
            scenarios.append(turn_scenario(scripts[cid], turn, earlier))
        return scenarios
    by_id = {s.id: s for s in load_scenarios(eval_set, None)}
    missing = sorted({r["scenario_id"] for r in rows} - by_id.keys())
    if missing:
        raise UsageError(f"scenarios missing from {eval_set}: {', '.join(missing)}")
    return [by_id[r["scenario_id"]] for r in rows]


def grid_order(
    results: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], eval_set: Path
) -> list[Mapping[str, Any]]:
    """``rows`` sorted by (model, strategy), then in the eval set's scenario (or conversation and
    turn) order: the order the source run aggregated each cell in. The source transcripts are in
    completion order, so aggregating them as written would make the cells, and the bootstrap
    intervals within them, depend on thread timing."""
    if "conversation_ids" in results:
        position = {s.id: i for i, s in enumerate(load_scripts(eval_set))}

        def key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
            return (
                row["model"],
                row["strategy"],
                position[row["conversation_id"]],
                int(row["turn_index"]),
            )

    else:
        position = {s.id: i for i, s in enumerate(load_scenarios(eval_set, None))}

        def key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
            return (row["model"], row["strategy"], position[row["scenario_id"]], 0)

    return sorted(rows, key=key)


# ---------------------------------------------------------------------------------------------
# The new run directory (resume safety)
# ---------------------------------------------------------------------------------------------


def manifest(results: Mapping[str, Any], judge: str) -> dict[str, Any]:
    """What a NEW_DIR's rows depend on: a resume must match it."""
    pins = results["pins"]
    return {
        "rejudged_from": {"run_id": results.get("run_id"), "judge_model": pins.get("judge_model")},
        "eval_set_sha256": pins["eval_set"]["sha256"],
        "judge_model": judge,
        "rubric_sha256": rubric_sha256(),
    }


def done_rows(
    new_dir: Path, wanted: Mapping[str, Any], resume: bool, source_keys: set[RowKey]
) -> dict[RowKey, dict[str, Any]]:
    """The rows already judged into ``new_dir`` (none for a new one). A non-empty ``new_dir`` is
    refused without --resume, and with --resume unless it was started from the same source run
    and judge."""
    if not new_dir.exists() or not any(new_dir.iterdir()):
        return {}
    if not resume:
        raise UsageError(f"{new_dir} is not empty; pass --resume to continue a re-judge into it")
    path = new_dir / MANIFEST
    if not path.is_file():
        raise UsageError(f"{new_dir} has no {MANIFEST}; it can't be resumed safely")
    recorded = _read_json(path)
    labels = {
        "rejudged_from": "source run",
        "eval_set_sha256": "eval set (sha256)",
        "judge_model": "judge (--judge)",
        "rubric_sha256": "judge rubric",
    }
    for key, label in labels.items():
        if not isinstance(recorded, dict) or recorded.get(key) != wanted[key]:
            raise UsageError(
                f"{new_dir} was started with a different {label}; --resume only continues the "
                "same re-judge (use a new --out instead)"
            )
    done: dict[RowKey, dict[str, Any]] = {}
    try:
        rows = read_transcripts(new_dir / TRANSCRIPTS)
    except json.JSONDecodeError as exc:
        raise UsageError(f"{new_dir / TRANSCRIPTS} has a malformed row: {exc.msg}") from None
    for row in rows:
        if row.get("judge_model") != wanted["judge_model"] or row_key(row) not in source_keys:
            raise UsageError(f"{new_dir} has a row from another re-judge: {row_key(row)}")
        done[row_key(row)] = row
    return done


# ---------------------------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------------------------


def rejudge_row(
    src: Mapping[str, Any],
    scenario: Scenario,
    *,
    run_id: str,
    judge: Retrying,
    judge_params: Mapping[str, Any],
    meter: spend.MeteredJudge | None = None,
) -> dict[str, Any]:
    """One source row checked and judged again; every other field is carried over. ``meter``
    (default: an uncapped one) records the row's judge calls."""
    reply = src["reply"]
    checks = run_checks(reply, scenario)
    error = src.get("generation_error")
    calls = (meter or spend.MeteredJudge(judge)).for_row()
    if error is None:
        result = judge_reply(scenario, reply, checks, calls, params=judge_params)
    else:
        result = _generation_failure(judge.name, error)
    row = transcript_row(
        run_id=run_id,
        scenario=scenario,
        model=src["model"],
        model_revision=src.get("model_revision"),
        strategy=src["strategy"],
        reply=reply,
        checks=checks,
        judge=result,
    )
    for field in KEPT_FIELDS:
        row[field] = src.get(field)
    # The fields after the shared schema (prompt hash, generation error, decoding, conversation
    # and turn ids), in the source row's order.
    for field, value in src.items():
        if field not in row:
            row[field] = value
    row["judge"]["decoding"] = judge.effective_params(judge_params)
    row["judge"]["usage"] = calls.usage
    return row


def judge_all(
    todo: Sequence[tuple[Mapping[str, Any], Scenario]],
    *,
    run_id: str,
    judge: Retrying,
    transcripts: Path,
    done: dict[RowKey, dict[str, Any]],
    concurrency: int,
    meter: spend.MeteredJudge | None = None,
) -> None:
    """Judge every ``todo`` row in a bounded pool, appending each to ``transcripts`` as it
    finishes. The first exception stops the run; finished rows stay on disk for --resume."""
    judge_params = decoding_params(judge.name)
    for row in done.values():
        recorded = (row.get("judge") or {}).get("decoding")
        if isinstance(recorded, Mapping):
            judge.restore_effective_params(judge_params, recorded)
    lock = threading.Lock()
    stop = threading.Event()
    print(f"{run_id}: {len(done)} rows done, {len(todo)} to judge", file=sys.stderr)
    with transcripts.open("a", encoding="utf-8") as fh:

        def task(src: Mapping[str, Any], scenario: Scenario) -> None:
            if stop.is_set():
                return
            try:
                row = rejudge_row(
                    src,
                    scenario,
                    run_id=run_id,
                    judge=judge,
                    judge_params=judge_params,
                    meter=meter,
                )
            except BaseException:
                stop.set()
                raise
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[row_key(row)] = row

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(task, src, scenario) for src, scenario in todo]
            for future in futures:
                future.result()


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------


def _sibling(name: str) -> ModuleType:
    """A sibling script (run_ablation.py, run_rollouts.py), for its results.md writer."""
    spec = importlib.util.spec_from_file_location(f"_rejudge_{name}", SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_pins(source: Mapping[str, Any], judge: Retrying) -> dict[str, Any]:
    """The source's eval-set and model pins with the new judge, its decoding, the current rubric
    and the re-judging checkout's code and library versions."""
    pins = dict(source)
    old_judge = pins.get("judge_model")
    decoding = dict(pins.get("decoding") or {})
    subjects = {m.get("model") for m in pins.get("models") or [] if isinstance(m, Mapping)}
    if old_judge not in subjects:
        decoding.pop(old_judge, None)
    decoding[judge.name] = judge.effective_params(decoding_params(judge.name))
    git_dirty = versions.git_dirty()
    pins.update(
        git_sha=versions.git_sha(),
        git_dirty=git_dirty,
        judge_model=judge.name,
        rubric_sha256=rubric_sha256(),
        decoding=decoding,
        library_versions=versions.library_versions(),
    )
    return pins


def build_results(
    source: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], run_id: str, judge: Retrying
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], str]]:
    """results.json for the re-judged rows, in the source run's shape, and its results.md
    writer."""
    cells = aggregate_scores(rows, seed=BOOTSTRAP_SEED)
    results = dict(source)
    results["run_id"] = run_id
    results["pins"] = new_pins(source["pins"], judge)
    if "conversation_ids" in source:
        for cell in cells:
            cell["meets_bar"] = meets_rollout_bar(cell)
        results["cells"] = cells
        writer: Callable[[dict[str, Any]], str] = _sibling("run_rollouts")._results_md
    elif "cells" in source:
        judge_family = model_family(judge.name)
        for cell in cells:
            cell["shares_judge_family"] = model_family(cell["model"]) == judge_family
        results.update(cells=cells, deltas=strategy_deltas(cells), plateau=plateau_verdict(cells))
        writer = _sibling("run_ablation")._results_md
    else:
        from groundcite.cli import _results_md

        results["rows"] = cells

        def writer(r: dict[str, Any]) -> str:
            return _results_md(r["run_id"], r["rows"], r["pins"])

    source_pins = source["pins"]
    results["rejudged_from"] = {
        "run_id": source.get("run_id"),
        "judge_model": source_pins.get("judge_model"),
        "git_sha": source_pins.get("git_sha"),
    }
    return results, writer


def results_md(results: dict[str, Any], writer: Callable[[dict[str, Any]], str]) -> str:
    """The source kind's results.md with a re-judge note under the title."""
    title, _, body = writer(results).partition("\n")
    origin = results["rejudged_from"]
    note = (
        f"Re-judged from run `{origin['run_id']}` (judge `{origin['judge_model']}`): its stored "
        f"replies scored again by `{results['pins']['judge_model']}` with "
        "`scripts/rejudge.py`. No reply was generated again."
    )
    return f"{title}\n\n{note}\n{body}"


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    source, src_rows = load_source(args.run_dir)
    eval_set = recorded_eval_set(source["pins"])
    scenarios = row_scenarios(source, src_rows, eval_set)
    judge_spec = normalize_spec(args.judge)
    new_dir: Path = args.out
    run_id = new_dir.name
    wanted = manifest(source, judge_spec)
    done = done_rows(new_dir, wanted, args.resume, {row_key(r) for r in src_rows})
    todo = [(r, s) for r, s in zip(src_rows, scenarios, strict=True) if row_key(r) not in done]
    calls = sum(r.get("generation_error") is None for r, _ in todo)
    if args.dry_run:
        source_judge = wanted["rejudged_from"]["judge_model"]
        print(f"source: {args.run_dir} ({len(src_rows)} rows, judge {source_judge})")
        print(f"judge: {judge_spec}")
        print(f"rows to re-judge: {len(todo)} ({len(done)} already done)")
        print(f"judge calls: {calls}")
        prompts = [
            spend.judge_prompt(s, r["reply"]) for r, s in todo if r.get("generation_error") is None
        ]
        spend.print_estimate(judge_spec, spend.estimate_cost(judge_spec, prompts))
        print("dry run: no requests made, nothing written")
        return 0

    spend.check_limit(judge_spec, args.max_spend)
    judge = build_backend(judge_spec)  # the key check, before anything is written
    spent = spend.recorded_cost(new_dir) if new_dir.is_dir() else 0.0
    meter = spend.MeteredJudge(judge, limit=args.max_spend, spent=spent)
    created = not new_dir.exists()
    new_dir.mkdir(parents=True, exist_ok=True)
    write_json(new_dir / MANIFEST, wanted)
    if (args.run_dir / GENERATION_PINS).is_file():
        shutil.copy2(args.run_dir / GENERATION_PINS, new_dir / GENERATION_PINS)
    transcripts = new_dir / TRANSCRIPTS
    try:
        judge_all(
            todo,
            run_id=run_id,
            judge=judge,
            transcripts=transcripts,
            done=done,
            concurrency=args.concurrency,
            meter=meter,
        )
    except BaseException:
        if meter.usage or (new_dir / spend.SPEND).is_file():
            spend.record_spend(new_dir, judge.name, meter.usage)
        elif created and not read_transcripts(transcripts):
            shutil.rmtree(new_dir, ignore_errors=True)
        raise
    recorded = spend.record_spend(new_dir, judge.name, meter.usage)
    rows = [done[row_key(r)] for r in src_rows]
    # Rows were appended as they finished; the final file keeps the source order.
    transcripts.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    results, writer = build_results(source, grid_order(source, rows, eval_set), run_id, judge)
    write_json(new_dir / RESULTS, results)
    (new_dir / "results.md").write_text(results_md(results, writer), encoding="utf-8")
    print(f"wrote {new_dir} ({spend.spend_note(recorded)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except spend.SpendCapReached as exc:
        print(f"error: re-judge stopped: {exc}; rerun with --resume to continue", file=sys.stderr)
        return 1
    except BackendError as exc:
        if exc.stops_run:
            print(stop_message(exc), file=sys.stderr)
            return 2
        print(
            f"error: re-judge stopped: {exc}; finished rows are saved, rerun with --resume",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
