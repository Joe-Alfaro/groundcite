#!/usr/bin/env python3
"""The judge-agreement check (D-095, D-096): candidate judges against the reference judge.

    uv run scripts/judge_agreement.py --build-sample --seed S --out PATH
    uv run scripts/judge_agreement.py --run --judge SPEC --sample PATH --out DIR
        [--dry-run] [--resume] [--concurrency N]
    uv run scripts/judge_agreement.py --report --dirs DIR [DIR ...] --out PATH
        [--adjudication PATH] [--full-dirs DIR [DIR ...]]

--build-sample draws the sample from the committed M1 ablation and rollout pilot transcripts
(both scored by the reference judge, ``[judge_candidates] reference``) with no API call. Only
judge-decided rows are drawn: no HARD check failed, so the reference verdict was not forced to 0
(D-099). Within them the draw follows D-096: 60 M1 rows (10 per subject model x strategy cell,
every category in every cell, up to 30 reference fails) and 40 pilot rows (every caliper turn,
up to 20 reference fails, early and late turns, cite and abstain golds, every family). Next to
the sample, ``<stem>.manifest.json`` records the seed, the sample's sha256 and each quota the
judge-decided rows can't fill. The same seed gives byte-identical output.

--run re-scores every sample row through the scripts/rejudge.py machinery (stored replies,
current checks, unchanged rubric) into DIR/transcripts.jsonl, resumable with --resume. --dry-run
needs no key, writes nothing and prints the call count and an estimated cost from ``[prices]``.
DIR/spend.json holds the actual spend from recorded token usage, across resumed invocations.

--report compares each run with the reference verdicts (agreement, Cohen's kappa, confusion
matrix, robustness delta, disagreements by category, spend) and applies the D-096 bar on the
judge-decided sample rows; a run row with a failed HARD check is refused (D-099). Always-pass and
always-fail baselines on the same rows must fail the bar. --full-dirs names rejudge.py re-judges
of all of M1 or the pilot, whose all-rows numbers are shown for reference only. Next to PATH it
writes disagreements.jsonl: one blind item per sample row where any candidate disagrees, for
adjudication. The adjudication file is JSONL of ``{item_id, spec_adherence}``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

from groundcite import versions
from groundcite.backends import BackendError
from groundcite.checks import HARD_CHECKS, run_checks
from groundcite.config import cost_usd, decoding_params, load_models_config
from groundcite.evalset import Scenario
from groundcite.judge import build_judge_messages, context_sections, rubric_sha256
from groundcite.keys import MissingKeyError
from groundcite.rollout import load_scripts
from groundcite.runner import (
    DEFAULT_CONCURRENCY,
    MANIFEST,
    Retrying,
    UsageError,
    build_backend,
    normalize_spec,
    read_transcripts,
    sha256_file,
    stop_message,
    write_json,
)

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = versions.REPO_ROOT
M1_DIR = "results/ablation/m1"
PILOT_DIR = "results/rollout_pilot/pilot1"
TRANSCRIPTS = "transcripts.jsonl"
SPEND = "spend.json"
DISAGREEMENTS = "disagreements.jsonl"

# The D-096 sample shape.
M1_PER_CELL = 10
M1_FAILS = 30
PILOT_SIZE = 40
PILOT_FAILS = 20
PILOT_MIN_EACH = 8
EARLY_MAX_TURN = 2
LATE_MIN_TURN = 6
# A caliper turn cites a snippet that applies only to floating front calipers: the pilot's
# hardest applicability call, so every one of them is in the sample.
CALIPER_TEXT = "floating front caliper"

# The D-096 acceptance bar, fixed before the run.
BAR = {"agreement": 0.9, "kappa": 0.75, "win_rate": 0.5}

# Dry-run estimate: about 4 characters per input token, and a fixed output allowance per call
# (a ~200-token verdict plus reasoning). Reasoning length is unmeasured (D-095), so this is an
# assumption, not a quote; spend.json records what a run actually used.
CHARS_PER_TOKEN = 4
EST_OUTPUT_TOKENS_PER_CALL = 2000


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_agreement_{name}", SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rejudge = _sibling("rejudge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judge_agreement.py", description=__doc__.splitlines()[0] if __doc__ else None
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-sample", action="store_true", help="draw the sample (no calls)")
    mode.add_argument("--run", action="store_true", help="re-score the sample with --judge")
    mode.add_argument("--report", action="store_true", help="compare runs with the reference")
    parser.add_argument("--seed", type=int, help="--build-sample: the draw's seed")
    parser.add_argument("--out", type=Path, required=True, help="sample file, run dir or report")
    parser.add_argument("--judge", help="--run: the candidate judge spec")
    parser.add_argument("--sample", type=Path, help="--run: the sample file")
    parser.add_argument("--dry-run", action="store_true", help="--run: print calls and cost only")
    parser.add_argument("--resume", action="store_true", help="--run: continue into --out")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="--run: parallel calls"
    )
    parser.add_argument("--dirs", type=Path, nargs="+", help="--report: candidate run dirs")
    parser.add_argument("--adjudication", type=Path, help="--report: adjudicated labels (JSONL)")
    parser.add_argument(
        "--full-dirs",
        type=Path,
        nargs="+",
        default=[],
        help="--report: rejudge.py re-judges of M1 or the pilot, for all-rows numbers",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return read_transcripts(path)
    except OSError as exc:
        raise UsageError(f"can't read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise UsageError(f"{path} has a malformed row: {exc.msg}") from None


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def is_fail(row: Mapping[str, Any]) -> bool:
    return bool(row["score"]["spec_adherence"] == 0)


def is_forced(row: Mapping[str, Any]) -> bool:
    """A HARD check failed, so D-028 pinned Spec-adherence to 0 whatever the judge said."""
    checks = row.get("checks") or {}
    return any(checks.get(name) is False for name in HARD_CHECKS)


Shortfall = dict[str, Any]


def shortfall(stratum: str, wanted: int, got: int) -> list[Shortfall]:
    return [{"stratum": stratum, "wanted": wanted, "got": got}] if got < wanted else []


# ---------------------------------------------------------------------------------------------
# The source runs
# ---------------------------------------------------------------------------------------------


class Source:
    """One committed run: its rows and the scenario each stored reply answered."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.results, self.rows = rejudge.load_source(REPO_ROOT / run_dir)
        self.eval_set: Path = rejudge.recorded_eval_set(self.results["pins"])
        self._scenarios: list[Scenario] | None = None

    @property
    def scenarios(self) -> list[Scenario]:
        if self._scenarios is None:
            self._scenarios = rejudge.row_scenarios(self.results, self.rows, self.eval_set)
        return self._scenarios

    def find(self, sample_row: Mapping[str, Any]) -> tuple[dict[str, Any], Scenario]:
        key = (sample_row["model"], sample_row["strategy"], sample_row["scenario_id"])
        for row, scenario in zip(self.rows, self.scenarios, strict=True):
            if rejudge.row_key(row) == key:
                return row, scenario
        raise UsageError(f"sample row {sample_row['sample_id']} is not in {self.run_dir}")


_SOURCES: dict[str, Source] = {}


def source(run_dir: str) -> Source:
    if run_dir not in _SOURCES:
        _SOURCES[run_dir] = Source(run_dir)
    return _SOURCES[run_dir]


# ---------------------------------------------------------------------------------------------
# --build-sample
# ---------------------------------------------------------------------------------------------


def draw_m1(
    rows: Sequence[dict[str, Any]], categories: Sequence[str], rng: random.Random
) -> tuple[list[dict[str, Any]], list[Shortfall]]:
    """10 rows per (model, strategy) cell covering every category, with as many reference fails
    as the data allows up to 30 in all. A cell missing a category, or short of rows, takes other
    rows of the cell and records the shortfall."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["model"], row["strategy"]), []).append(row)

    # Each cell's fail quota: every category needs a row, so a category with no fail in the cell
    # takes a pass slot. Over the total, quotas come down from the largest first.
    targets: dict[tuple[str, str], int] = {}
    for cell, cell_rows in sorted(cells.items()):
        fail_cats = {r["category"] for r in cell_rows if is_fail(r)}
        cap = M1_PER_CELL - (len(categories) - len(fail_cats))
        targets[cell] = min(sum(map(is_fail, cell_rows)), cap)
    while sum(targets.values()) > M1_FAILS:
        largest = max(targets.values())
        cell = next(c for c in sorted(targets) if targets[c] == largest)
        targets[cell] -= 1

    picked: list[dict[str, Any]] = []
    shortfalls: list[Shortfall] = []
    for cell, cell_rows in sorted(cells.items()):
        pool = list(cell_rows)
        rng.shuffle(pool)
        fails = [r for r in pool if is_fail(r)]
        passes = [r for r in pool if not is_fail(r)]
        chosen: list[dict[str, Any]] = []
        for category in categories:
            row = next((r for r in fails if r["category"] == category), None)
            if row is not None and sum(map(is_fail, chosen)) < targets[cell]:
                chosen.append(row)
        for category in categories:
            if any(r["category"] == category for r in chosen):
                continue
            row = next((r for r in passes if r["category"] == category), None)
            if row is None:
                shortfalls += shortfall(f"m1 {cell[0]} {cell[1]} {category}", 1, 0)
                continue
            chosen.append(row)
        for row in fails:
            if sum(map(is_fail, chosen)) >= targets[cell]:
                break
            if row not in chosen:
                chosen.append(row)
        for row in passes:
            if len(chosen) >= M1_PER_CELL:
                break
            if row not in chosen:
                chosen.append(row)
        for row in fails:
            if len(chosen) >= M1_PER_CELL:
                break
            if row not in chosen:
                chosen.append(row)
        shortfalls += shortfall(f"m1 {cell[0]} {cell[1]} rows", M1_PER_CELL, len(chosen))
        picked.extend(chosen)
    shortfalls += shortfall("m1 reference fails", M1_FAILS, sum(map(is_fail, picked)))
    return picked, shortfalls


def caliper_ids(scripts_path: Path) -> set[str]:
    """Snippet IDs whose text limits them to floating front calipers, anywhere in the scripts."""
    ids: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "id" in obj and CALIPER_TEXT in str(obj.get("text", "")).lower():
                ids.add(str(obj["id"]))
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(read_jsonl(scripts_path))
    return ids


def draw_pilot(
    rows: Sequence[dict[str, Any]],
    all_rows: Sequence[dict[str, Any]],
    scripts_path: Path,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[Shortfall]]:
    """Every caliper turn, up to 20 reference fails, then passes until 40 rows, drawn first to
    reach 8 early, 8 late, 8 cite and 8 abstain turns and every family. ``rows`` are the
    judge-decided rows; ``all_rows`` (every turn) sets the caliper quota."""
    calipers = caliper_ids(scripts_path)
    families = {s.id: s.family for s in load_scripts(scripts_path)}

    def is_caliper(row: Mapping[str, Any]) -> bool:
        gold = row["gold"]
        return gold["mode"] == "cite" and bool(set(gold["supporting_ids"]) & calipers)

    pool = list(rows)
    rng.shuffle(pool)
    chosen = [r for r in pool if is_caliper(r)]
    fail_target = min(PILOT_FAILS, sum(map(is_fail, rows)))
    for row in pool:
        if sum(map(is_fail, chosen)) >= fail_target:
            break
        if is_fail(row) and row not in chosen:
            chosen.append(row)
    if len(chosen) > PILOT_SIZE:
        raise UsageError("the pilot's caliper turns and fails don't fit the sample")
    shortfalls = shortfall(
        "pilot caliper turns", sum(map(is_caliper, all_rows)), sum(map(is_caliper, chosen))
    )
    shortfalls += shortfall("pilot reference fails", PILOT_FAILS, sum(map(is_fail, chosen)))

    needs: list[tuple[str, Callable[[Mapping[str, Any]], bool], int]] = [
        ("pilot early turns", lambda r: int(r["turn_index"]) <= EARLY_MAX_TURN, PILOT_MIN_EACH),
        ("pilot late turns", lambda r: int(r["turn_index"]) >= LATE_MIN_TURN, PILOT_MIN_EACH),
        ("pilot cite golds", lambda r: r["gold"]["mode"] == "cite", PILOT_MIN_EACH),
        ("pilot abstain golds", lambda r: r["gold"]["mode"] == "abstain", PILOT_MIN_EACH),
    ]
    for family in sorted(set(families.values())):
        needs.append(
            (f"pilot family {family}", lambda r, f=family: families[r["conversation_id"]] == f, 1)
        )
    passes = [r for r in pool if not is_fail(r)]
    for _, wanted, count in needs:
        for row in passes:
            if sum(map(wanted, chosen)) >= count or len(chosen) >= PILOT_SIZE:
                break
            if wanted(row) and row not in chosen:
                chosen.append(row)
    for row in passes:
        if len(chosen) >= PILOT_SIZE:
            break
        if row not in chosen:
            chosen.append(row)
    for name, wanted, count in needs:
        shortfalls += shortfall(name, count, sum(map(wanted, chosen)))
    shortfalls += shortfall("pilot rows", PILOT_SIZE, len(chosen))
    return chosen, shortfalls


def sample_row(
    sample_id: str, source_name: str, run_dir: str, row: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": sample_id,
        "source": source_name,
        "run_dir": run_dir,
        "model": row["model"],
        "strategy": row["strategy"],
        "scenario_id": row["scenario_id"],
        "category": row["category"],
        "seed": seed,
        "reference": {
            "judge_model": row["judge_model"],
            "spec_adherence": row["score"]["spec_adherence"],
            "robustness": row["score"]["robustness"],
        },
    }
    if "conversation_id" in row:
        out["conversation_id"] = row["conversation_id"]
        out["turn_index"] = row["turn_index"]
    return out


def judged_rows(src: Source, reference: str) -> list[dict[str, Any]]:
    """The source's delivered replies, checked to carry the reference judge's verdicts."""
    wrong = {r["judge_model"] for r in src.rows} - {reference}
    if wrong:
        raise UsageError(f"{src.run_dir} was judged by {sorted(wrong)}, not {reference}")
    return [r for r in src.rows if not r.get("generation_error")]


def manifest_path(sample: Path) -> Path:
    return sample.with_name(sample.stem + ".manifest.json")


def build_sample(seed: int, out: Path) -> int:
    reference = str(load_models_config()["judge_candidates"]["reference"])
    rng = random.Random(seed)
    m1, pilot = source(M1_DIR), source(PILOT_DIR)
    m1_rows, pilot_rows = judged_rows(m1, reference), judged_rows(pilot, reference)
    categories = sorted({r["category"] for r in m1_rows})
    m1_picked, m1_short = draw_m1([r for r in m1_rows if not is_forced(r)], categories, rng)
    pilot_picked, pilot_short = draw_pilot(
        [r for r in pilot_rows if not is_forced(r)], pilot_rows, pilot.eval_set, rng
    )
    parts = [("m1", m1, m1_picked), ("pilot", pilot, pilot_picked)]
    rows: list[dict[str, Any]] = []
    for name, src, picked in parts:
        # Source order, so the file doesn't depend on the order of the draw.
        position = {id(r): i for i, r in enumerate(src.rows)}
        for i, row in enumerate(sorted(picked, key=lambda r: position[id(r)]), 1):
            rows.append(sample_row(f"{name}-{i:02d}", name, src.run_dir, row, seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    shortfalls = m1_short + pilot_short
    write_json(
        manifest_path(out),
        {"seed": seed, "sample_sha256": sha256_file(out), "shortfalls": shortfalls},
    )
    for entry in shortfalls:
        print(f"shortfall: {entry['stratum']}: {entry['got']} of {entry['wanted']}")
    fails = Counter(r["source"] for r in rows if r["reference"]["spec_adherence"] == 0)
    counts = Counter(r["source"] for r in rows)
    print(
        f"wrote {out}: {len(rows)} rows (m1 {counts['m1']}, {fails['m1']} reference fails; "
        f"pilot {counts['pilot']}, {fails['pilot']} reference fails), seed {seed}"
    )
    return 0


# ---------------------------------------------------------------------------------------------
# --run
# ---------------------------------------------------------------------------------------------


def run_manifest(sample: Path, judge: str) -> dict[str, Any]:
    return {
        "sample_sha256": sha256_file(sample),
        "judge_model": judge,
        "rubric_sha256": rubric_sha256(),
    }


def done_rows(
    out: Path, wanted: Mapping[str, Any], resume: bool, sample_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """The rows already judged into ``out``; a non-empty ``out`` needs --resume and a matching
    sample, judge and rubric."""
    if not out.exists() or not any(out.iterdir()):
        return {}
    if not resume:
        raise UsageError(f"{out} is not empty; pass --resume to continue a run into it")
    path = out / MANIFEST
    if not path.is_file():
        raise UsageError(f"{out} has no {MANIFEST}; it can't be resumed safely")
    recorded = rejudge._read_json(path)
    labels = {
        "sample_sha256": "sample (sha256)",
        "judge_model": "judge (--judge)",
        "rubric_sha256": "judge rubric",
    }
    for key, label in labels.items():
        if not isinstance(recorded, dict) or recorded.get(key) != wanted[key]:
            raise UsageError(
                f"{out} was started with a different {label}; --resume only continues the same "
                "run (use a new --out instead)"
            )
    done: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(out / TRANSCRIPTS):
        if (
            row.get("judge_model") != wanted["judge_model"]
            or row.get("sample_id") not in sample_ids
        ):
            raise UsageError(f"{out} has a row from another run: {row.get('sample_id')}")
        done[row["sample_id"]] = row
    return done


def load_sample(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise UsageError(f"{path} has no sample rows")
    ids = [r.get("sample_id") for r in rows]
    if len(set(ids)) != len(ids):
        raise UsageError(f"{path} has duplicate sample_ids")
    return rows


Item = tuple[dict[str, Any], dict[str, Any], Scenario]


def estimate(spec: str, todo: Sequence[Item]) -> tuple[int, float | None]:
    """Estimated input tokens and dollars for judging ``todo`` (None without a price)."""
    usages = []
    for _, src_row, scenario in todo:
        reply = src_row["reply"]
        system, messages = build_judge_messages(scenario, reply, run_checks(reply, scenario))
        chars = len(system) + sum(len(m["content"]) for m in messages)
        usages.append(
            {
                "input_tokens": math.ceil(chars / CHARS_PER_TOKEN),
                "output_tokens": EST_OUTPUT_TOKENS_PER_CALL,
                "reasoning_tokens": None,
            }
        )
    tokens = sum(u["input_tokens"] for u in usages)
    try:
        return tokens, cost_usd(spec, usages)
    except KeyError:
        return tokens, None


def judge_rows(
    todo: Sequence[Item],
    *,
    run_id: str,
    judge: Retrying,
    transcripts: Path,
    done: dict[str, dict[str, Any]],
    concurrency: int,
) -> None:
    """Judge ``todo`` in a bounded pool, appending each row as it finishes. The first exception
    stops the run; finished rows stay on disk for --resume."""
    judge_params = decoding_params(judge.name)
    for row in done.values():
        recorded = (row.get("judge") or {}).get("decoding")
        if isinstance(recorded, Mapping):
            judge.restore_effective_params(judge_params, recorded)
    lock = threading.Lock()
    stop = threading.Event()
    print(f"{run_id}: {len(done)} rows done, {len(todo)} to judge", file=sys.stderr)
    with transcripts.open("a", encoding="utf-8") as fh:

        def task(item: Item) -> None:
            if stop.is_set():
                return
            sample, src_row, scenario = item
            try:
                row = rejudge.rejudge_row(
                    src_row, scenario, run_id=run_id, judge=judge, judge_params=judge_params
                )
            except BaseException:
                stop.set()
                raise
            row.update(
                sample_id=sample["sample_id"],
                source=sample["source"],
                run_dir=sample["run_dir"],
                reference=sample["reference"],
            )
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[sample["sample_id"]] = row

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(task, item) for item in todo]
            for future in futures:
                future.result()


def update_spend(out: Path, judge: Retrying) -> dict[str, Any]:
    """Add this invocation's recorded usage to DIR/spend.json."""
    usage: list[dict[str, Any]] = list(getattr(judge.inner, "usage", []))
    path = out / SPEND
    spend: dict[str, Any] = {
        "judge_model": judge.name,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    if path.is_file():
        spend.update(rejudge._read_json(path))
    spend["calls"] += len(usage)
    for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
        spend[field] += sum(u.get(field) or 0 for u in usage)
    if usage:
        try:
            added = cost_usd(judge.name, usage)
        except KeyError:
            print(
                f"warning: no [prices] entry for {judge.name}; cost not recorded", file=sys.stderr
            )
            spend["cost_usd"] = None
        else:
            if spend["cost_usd"] is not None:
                spend["cost_usd"] += added
    write_json(path, spend)
    return spend


def run(args: argparse.Namespace) -> int:
    if not args.judge or not args.sample:
        raise UsageError("--run needs --judge and --sample")
    judge_spec = normalize_spec(args.judge)
    sample = load_sample(args.sample)
    out: Path = args.out
    wanted = run_manifest(args.sample, judge_spec)
    done = done_rows(out, wanted, args.resume, {r["sample_id"] for r in sample})
    items: list[Item] = []
    for row in sample:
        src_row, scenario = source(row["run_dir"]).find(row)
        items.append((row, src_row, scenario))
    todo = [item for item in items if item[0]["sample_id"] not in done]
    calls = sum(src.get("generation_error") is None for _, src, _ in todo)

    if args.dry_run:
        tokens, cost = estimate(
            judge_spec, [i for i in todo if i[1].get("generation_error") is None]
        )
        print(f"sample: {args.sample} ({len(sample)} rows)")
        print(f"judge: {judge_spec}")
        print(f"rows to judge: {len(todo)} ({len(done)} already done)")
        print(f"judge calls: {calls}")
        print(
            f"estimated tokens: {tokens} input (~{CHARS_PER_TOKEN} chars/token), "
            f"{EST_OUTPUT_TOKENS_PER_CALL * calls} output incl. reasoning "
            f"({EST_OUTPUT_TOKENS_PER_CALL}/call assumed)"
        )
        if cost is None:
            print(f"estimated cost: unknown (no [prices] entry for {judge_spec})")
        else:
            print(f"estimated cost: ${cost:.2f}")
        print("dry run: no requests made, nothing written")
        return 0

    judge = build_backend(judge_spec)  # the key check, before anything is written
    created = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / MANIFEST, wanted)
    transcripts = out / TRANSCRIPTS
    try:
        judge_rows(
            todo,
            run_id=out.name,
            judge=judge,
            transcripts=transcripts,
            done=done,
            concurrency=args.concurrency,
        )
    except BaseException:
        if created and not read_transcripts(transcripts) and not getattr(judge.inner, "usage", []):
            shutil.rmtree(out, ignore_errors=True)
        else:
            update_spend(out, judge)
        raise
    spend = update_spend(out, judge)
    # Rows were appended as they finished; the final file keeps the sample order.
    write_jsonl(transcripts, [done[r["sample_id"]] for r in sample])
    cost = spend["cost_usd"]
    print(
        f"wrote {out} ({spend['calls']} judge calls in all, "
        f"spend {'unknown' if cost is None else f'${cost:.4f}'})"
    )
    return 0


# ---------------------------------------------------------------------------------------------
# --report
# ---------------------------------------------------------------------------------------------


def cohens_kappa(ref: Sequence[int], cand: Sequence[int]) -> float:
    """Cohen's kappa for two binary raters; 1.0 when chance agreement is already 1."""
    n = len(ref)
    po = sum(a == b for a, b in zip(ref, cand, strict=True)) / n
    pr, pc = sum(ref) / n, sum(cand) / n
    pe = pr * pc + (1 - pr) * (1 - pc)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def item_id(sample_id: str) -> str:
    return "item-" + hashlib.sha256(f"judge-agreement:{sample_id}".encode()).hexdigest()[:12]


def disagrees(row: Mapping[str, Any]) -> bool:
    return bool(row["score"]["spec_adherence"] != row["reference"]["spec_adherence"])


def verdict_of(score: Mapping[str, Any], reasoning: Any, failed: Any) -> dict[str, Any]:
    return {
        "spec_adherence": score["spec_adherence"],
        "robustness": score["robustness"],
        "reasoning": reasoning,
        "failed_checks": list(failed or []),
    }


def blind_item(sample_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The context, gold, reply and every verdict (reference included) in a shuffled order, with
    no model or judge name."""
    first = rows[0]
    src_row, scenario = source(first["run_dir"]).find(first)
    verdicts = [
        verdict_of(
            src_row["score"], src_row.get("reasoning"), src_row["judge"].get("failed_checks")
        )
    ]
    verdicts += [
        verdict_of(r["score"], r.get("reasoning"), (r.get("judge") or {}).get("failed_checks"))
        for r in rows
    ]
    oid = item_id(sample_id)
    random.Random(oid).shuffle(verdicts)
    return {
        "item_id": oid,
        "context": "\n\n".join(context_sections(scenario)),
        "gold": first["gold"],
        "reply": first["reply"],
        "checks": first["checks"],
        "verdicts": verdicts,
    }


def agreement_stats(ref: Sequence[int], cand: Sequence[int]) -> dict[str, Any]:
    """Spec-adherence agreement, Cohen's kappa and the confusion matrix (reference x candidate)."""
    label = {1: "pass", 0: "fail"}
    confusion = {a: {b: 0 for b in ("pass", "fail")} for a in ("pass", "fail")}
    for a, b in zip(ref, cand, strict=True):
        confusion[label[a]][label[b]] += 1
    return {
        "n": len(ref),
        "agreement": sum(a == b for a, b in zip(ref, cand, strict=True)) / len(ref),
        "kappa": cohens_kappa(ref, cand),
        "confusion": confusion,
    }


def meets_agreement_bar(stats: Mapping[str, Any]) -> bool:
    return bool(stats["agreement"] >= BAR["agreement"] and stats["kappa"] >= BAR["kappa"])


def candidate_metrics(
    rows: Sequence[Mapping[str, Any]], adjudication: Mapping[str, int], spend: Any
) -> dict[str, Any]:
    ref = [int(r["reference"]["spec_adherence"]) for r in rows]
    stats = agreement_stats(ref, [int(r["score"]["spec_adherence"]) for r in rows])
    split = [r for r in rows if disagrees(r)]
    by_category = Counter(f"{r['source']}:{r['category']}" for r in split)
    judged = [r for r in split if item_id(r["sample_id"]) in adjudication]
    wins = sum(
        adjudication[item_id(r["sample_id"])] == r["score"]["spec_adherence"] for r in judged
    )
    win_rate = wins / len(judged) if judged else None
    if not meets_agreement_bar(stats):
        verdict = "fail"
    elif len(judged) < len(split):
        verdict = "pending"
    else:
        # With nothing to adjudicate the win-rate criterion passes vacuously.
        verdict = "pass" if win_rate is None or win_rate >= BAR["win_rate"] else "fail"
    return {
        **stats,
        "mean_abs_robustness_delta": sum(
            abs(r["score"]["robustness"] - r["reference"]["robustness"]) for r in rows
        )
        / len(rows),
        "disagreements": len(split),
        "disagreements_by_category": dict(by_category),
        "adjudicated": len(judged),
        "wins": wins,
        "win_rate": win_rate,
        "spend_usd": spend,
        "verdict": verdict,
        "all_rows": None,
    }


def baselines(references: Sequence[int]) -> dict[str, dict[str, Any]]:
    """Constant judges on the same rows: both must miss the bar, or the bar can't tell a judge
    from a constant (D-099)."""
    out = {}
    for name, constant in (("always_pass", 1), ("always_fail", 0)):
        stats = agreement_stats(references, [constant] * len(references))
        out[name] = {
            "n": stats["n"],
            "agreement": stats["agreement"],
            "kappa": stats["kappa"],
            "verdict": "pass" if meets_agreement_bar(stats) else "fail",
        }
    return out


def full_run_stats(dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """All-rows numbers per judge from rejudge.py re-judges of M1 or the pilot, every row
    (forced ones included) paired with the source run's reference verdict. Reference only."""
    known = {source(d).results.get("run_id"): source(d) for d in (M1_DIR, PILOT_DIR)}
    pairs: dict[str, tuple[list[int], list[int]]] = {}
    for run_dir in dirs:
        manifest = rejudge._read_json(run_dir / MANIFEST)
        origin = manifest.get("rejudged_from") if isinstance(manifest, dict) else None
        spec = manifest.get("judge_model") if isinstance(manifest, dict) else None
        if not isinstance(origin, dict) or not isinstance(spec, str):
            raise UsageError(
                f"{run_dir} is not a rejudge.py output (no rejudged_from in {MANIFEST})"
            )
        src = known.get(origin.get("run_id"))
        if src is None:
            raise UsageError(f"{run_dir} re-judges {origin.get('run_id')!r}, not M1 or the pilot")
        reference = {rejudge.row_key(r): r for r in src.rows}
        ref, cand = pairs.setdefault(spec, ([], []))
        for row in read_jsonl(run_dir / TRANSCRIPTS):
            key = rejudge.row_key(row)
            if key not in reference:
                raise UsageError(f"{run_dir} has a row that is not in {src.run_dir}: {key}")
            ref.append(int(reference[key]["score"]["spec_adherence"]))
            cand.append(int(row["score"]["spec_adherence"]))
    return {spec: agreement_stats(ref, cand) for spec, (ref, cand) in pairs.items() if ref}


def load_adjudication(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    labels: dict[str, int] = {}
    for row in read_jsonl(path):
        if row.get("spec_adherence") not in (0, 1) or not isinstance(row.get("item_id"), str):
            raise UsageError(f"{path}: each row needs an item_id and spec_adherence 0 or 1")
        labels[row["item_id"]] = int(row["spec_adherence"])
    return labels


def report(args: argparse.Namespace) -> int:
    if not args.dirs:
        raise UsageError("--report needs --dirs")
    adjudication = load_adjudication(args.adjudication)
    runs: dict[str, list[dict[str, Any]]] = {}
    spends: dict[str, Any] = {}
    for run_dir in args.dirs:
        manifest = rejudge._read_json(run_dir / MANIFEST)
        spec = manifest.get("judge_model") if isinstance(manifest, dict) else None
        if not isinstance(spec, str):
            raise UsageError(f"{run_dir / MANIFEST} names no judge_model")
        if spec in runs:
            raise UsageError(f"two run dirs were judged by {spec}")
        rows = read_jsonl(run_dir / TRANSCRIPTS)
        if not rows:
            raise UsageError(f"{run_dir / TRANSCRIPTS} has no rows")
        forced = [r["sample_id"] for r in rows if is_forced(r)]
        if forced:
            raise UsageError(
                f"{run_dir} has rows with a failed HARD check ({', '.join(forced)}); the bar only "
                "uses judge-decided rows (D-099): rebuild the sample and re-run"
            )
        runs[spec] = rows
        spend_path = run_dir / SPEND
        spends[spec] = (
            rejudge._read_json(spend_path).get("cost_usd") if spend_path.is_file() else None
        )

    references = {r["reference"]["judge_model"] for rows in runs.values() for r in rows}
    if len(references) != 1:
        raise UsageError(f"the runs compare against different reference judges: {references}")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for rows in runs.values():
        for row in rows:
            by_sample.setdefault(row["sample_id"], []).append(row)
    split = sorted(sid for sid, rows in by_sample.items() if any(map(disagrees, rows)))
    items = [blind_item(sid, by_sample[sid]) for sid in split]
    items.sort(key=lambda i: i["item_id"])

    candidates = {
        spec: candidate_metrics(rows, adjudication, spends[spec]) for spec, rows in runs.items()
    }
    for spec, stats in full_run_stats(args.full_dirs).items():
        if spec not in candidates:
            raise UsageError(f"--full-dirs has a re-judge by {spec}, which is not a candidate")
        candidates[spec]["all_rows"] = stats
    sample_ids = sorted(by_sample)
    result = {
        "reference_judge": references.pop(),
        "bar": dict(BAR),
        "candidates": candidates,
        "baselines": baselines(
            [int(by_sample[sid][0]["reference"]["spec_adherence"]) for sid in sample_ids]
        ),
        "items": {item_id(sid): sid for sid in split},
    }
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    write_jsonl(out.parent / DISAGREEMENTS, items)
    for spec, c in result["candidates"].items():
        print(
            f"{spec}: agreement {c['agreement']:.3f}, kappa {c['kappa']:.3f}, "
            f"{c['disagreements']} disagreements ({c['adjudicated']} adjudicated), {c['verdict']}"
        )
    for name, b in result["baselines"].items():
        note = "" if b["verdict"] == "fail" else " (WARNING: the bar does not reject it)"
        print(
            f"baseline {name.replace('_', '-')}: agreement {b['agreement']:.3f}, "
            f"kappa {b['kappa']:.3f}, {b['verdict']}{note}"
        )
    print(f"wrote {out} and {out.parent / DISAGREEMENTS} ({len(items)} items)")
    return 0


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.build_sample:
            if args.seed is None:
                raise UsageError("--build-sample needs --seed")
            return build_sample(args.seed, args.out)
        if args.run:
            return run(args)
        return report(args)
    except (UsageError, MissingKeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BackendError as exc:
        if exc.stops_run:
            print(stop_message(exc), file=sys.stderr)
            return 2
        print(
            f"error: run stopped: {exc}; finished rows are saved, rerun with --resume",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
