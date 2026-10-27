#!/usr/bin/env python3
"""Distill k teacher candidates per training scenario (R4, R20, R25).

    uv run scripts/distill.py --scenarios data/scenarios/train_pool.jsonl --k 2 \\
        --out data/distill/v1 [--teacher SPEC] [--limit N] [--concurrency C] [--dry-run] [--resume]

Writes ``<out>/candidates.jsonl``: one line per (scenario, candidate) with the full scenario row,
so ``scripts/build_dataset.py`` needs only this file. Lines are appended in scenario order as soon
as they are done, so an interrupted run continues with ``--resume`` and pays only for the missing
(scenario_id, candidate_index) pairs. Transient API errors are retried with backoff; a teacher
error that survives the retries is recorded on its line (``reply`` empty), never dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from groundcite.backends import BackendError
from groundcite.config import decoding_params, load_models_config
from groundcite.distill import TEACHER_PROMPT_VERSION, render_teacher_prompt, teacher_prompt_sha256
from groundcite.evalset import Scenario, _RowError, parse_row
from groundcite.keys import MissingKeyError
from groundcite.runner import DEFAULT_CONCURRENCY, Retrying, build_backend, normalize_spec

MIN_K = 2  # R4: at least two candidates per scenario
CANDIDATES = "candidates.jsonl"


class UsageError(RuntimeError):
    """Bad arguments or input: reported on stderr, exit 2, nothing written."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distill.py", description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", type=Path, required=True, help="training scenarios JSONL")
    parser.add_argument("--teacher", help="teacher model (default: config [teacher] model)")
    parser.add_argument(
        "--k", type=int, required=True, help=f"candidates per scenario (>= {MIN_K})"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--limit", type=int, help="only the first N scenarios")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel teacher requests"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no calls")
    parser.add_argument("--resume", action="store_true", help="append only missing candidates")
    return parser


def load_pool(path: Path, limit: int | None) -> list[tuple[dict[str, Any], Scenario]]:
    """Raw rows (kept verbatim for the candidates file) with their parsed scenarios."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc}") from None
    pool = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            pool.append((row, parse_row(row)))
        except json.JSONDecodeError as exc:
            raise UsageError(f"{path}: line {lineno}: invalid JSON: {exc.msg}") from None
        except _RowError as exc:
            raise UsageError(f"{path}: line {lineno}: {exc.field}: {exc.message}") from None
        if limit is not None and len(pool) >= limit:
            break
    return pool


def existing_keys(path: Path, teacher: str, prompt_sha: str) -> set[tuple[str, int]]:
    """(scenario_id, candidate_index) pairs already on disk. Refuses to mix teachers or prompts."""
    keys = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (str(row["scenario_id"]), int(row["candidate_index"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise UsageError(f"{path}: line {lineno} is not a candidate line") from None
        if row.get("teacher") != teacher or row.get("teacher_prompt_sha256") != prompt_sha:
            raise UsageError(
                f"{path}: line {lineno} came from teacher {row.get('teacher')!r} with prompt sha "
                f"{str(row.get('teacher_prompt_sha256'))[:12]}; resuming needs the same teacher "
                "and teacher prompt"
            )
        keys.add(key)
    return keys


def candidate_line(
    row: Mapping[str, Any], index: int, teacher: str, sha: str, reply: str, error: str | None
) -> str:
    record = {
        "scenario_id": row["id"],
        "candidate_index": index,
        "teacher": teacher,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "teacher_prompt_sha256": sha,
        "reply": reply,
        "error": error,
        "scenario": row,
    }
    return json.dumps(record) + "\n"


def run(args: argparse.Namespace) -> int:
    if args.k < MIN_K:
        raise UsageError(f"--k must be at least {MIN_K} (R4), got {args.k}")
    if args.concurrency < 1:
        raise UsageError("--concurrency must be at least 1")
    teacher = normalize_spec(args.teacher or load_models_config()["teacher"]["model"])
    pool = load_pool(args.scenarios, args.limit)
    sha = teacher_prompt_sha256()
    out_path = args.out / CANDIDATES

    done: set[tuple[str, int]] = set()
    if out_path.exists() and out_path.stat().st_size:
        if not args.resume:
            raise UsageError(f"{out_path} already exists; pass --resume to continue it")
        done = existing_keys(out_path, teacher, sha)
    tasks = [
        (row, scenario, i)
        for row, scenario in pool
        for i in range(args.k)
        if (scenario.id, i) not in done
    ]

    if args.dry_run:
        print(f"scenarios: {args.scenarios} ({len(pool)} scenarios, k={args.k})")
        print(f"teacher: {teacher}")
        print(f"teacher prompt: {TEACHER_PROMPT_VERSION} (sha256 {sha[:12]})")
        if done:
            print(f"already done: {len(done)} candidates")
        print(f"teacher calls: {len(tasks)}")
        print("dry run: no requests made, nothing written")
        return 0

    backend = build_backend(teacher)  # fails here, before any work, when the key is missing
    params = decoding_params(teacher)
    args.out.mkdir(parents=True, exist_ok=True)
    errors = generate_all(backend, params, tasks, teacher, sha, out_path, args.concurrency)
    print(f"wrote {len(tasks)} candidates to {out_path} ({errors} teacher errors)")
    return 0


def generate_all(
    backend: Retrying,
    params: Mapping[str, Any],
    tasks: list[tuple[dict[str, Any], Scenario, int]],
    teacher: str,
    sha: str,
    out_path: Path,
    concurrency: int,
) -> int:
    """Run the tasks in a bounded pool and append each line in task order as soon as every
    earlier line is written, so the file stays ordered and a crash loses only unfinished work."""
    lock = threading.Lock()
    results: dict[int, str] = {}
    next_to_write = 0
    errors = 0

    def one(row: dict[str, Any], scenario: Scenario, index: int) -> tuple[str, str | None]:
        system, messages = render_teacher_prompt(scenario, variant=index)
        try:
            return backend.generate(messages, system, params), None
        except BackendError as exc:
            return "", str(exc) or type(exc).__name__

    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(concurrency) as pool:

        def finished(position: int, future: Future[tuple[str, str | None]]) -> None:
            nonlocal next_to_write, errors
            row, _, index = tasks[position]
            reply, error = future.result()
            with lock:
                errors += error is not None
                results[position] = candidate_line(row, index, teacher, sha, reply, error)
                while next_to_write in results:
                    fh.write(results.pop(next_to_write))
                    next_to_write += 1
                fh.flush()

        futures = []
        for position, (row, scenario, index) in enumerate(tasks):
            future = pool.submit(one, row, scenario, index)
            future.add_done_callback(lambda f, p=position: finished(p, f))
            futures.append(future)
            if position == 0:
                # The first request runs alone: a model that rejects a decoding parameter is
                # learned once (the backend caches it), not once per concurrent request.
                future.result()
        for future in futures:
            future.result()  # re-raise anything unexpected
    return errors


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
