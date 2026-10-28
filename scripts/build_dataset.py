#!/usr/bin/env python3
"""Filter teacher candidates into a training dataset (R4, R6, R23, R30).

    uv run scripts/build_dataset.py --candidates data/distill/v1/candidates.jsonl \\
        --out data/datasets/v1 [--judge SPEC] [--concurrency C]

Stages, in order (the first failure is the rejection reason; see ``groundcite.filter``):
generation_error, hint_leak, unclean_end, gold_mismatch, hard_check, soft_check, then the judge
(judge_error, judge_spec, judge_robustness), near_duplicate and not_selected. One example is kept
per scenario. Writes train.jsonl, rejections.jsonl, manifest.json and filter_report.md into a new
or empty ``--out``; the same input gives the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from groundcite.backends import BackendError
from groundcite.config import load_models_config
from groundcite.evalset import _RowError, parse_row
from groundcite.filter import (
    FILTER_VERSION,
    JUDGE_ERROR,
    MAX_ROBUSTNESS,
    NEAR_DUPLICATE,
    NOT_SELECTED,
    REJECTION_REASONS,
    SELECTION_RULE,
    Candidate,
    NearDuplicateIndex,
    Outcome,
    Rejection,
    deterministic_stage,
    final_user_message,
    judge_stage,
)
from groundcite.judge import judge_reply, rubric_sha256
from groundcite.keys import MissingKeyError
from groundcite.prompts import render_messages
from groundcite.runner import DEFAULT_CONCURRENCY, Retrying, build_backend, normalize_spec

KEPT_RATE_RANGE = (0.20, 0.95)
# R23/R30: the training set should be 20-30% abstain examples, inclusive.
ABSTAIN_SHARE_RANGE = (0.20, 0.30)
_STR_FIELDS = ("scenario_id", "teacher", "teacher_prompt_version", "teacher_prompt_sha256")


class UsageError(RuntimeError):
    """Bad arguments or input: reported on stderr, exit 2, nothing written."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_dataset.py", description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, required=True, help="distill candidates JSONL")
    parser.add_argument(
        "--out", type=Path, required=True, help="new dataset dir, e.g. data/datasets/v1"
    )
    parser.add_argument("--judge", help="judge model (default: config; 'mock' = mock:judge)")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel judge requests"
    )
    return parser


# ---------------------------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------------------------


def _parse_candidate(raw: Any, where: str) -> tuple[Candidate, dict[str, Any]]:
    def bad(name: str, message: str) -> UsageError:
        return UsageError(f"{where}: {name}: {message}")

    if not isinstance(raw, dict):
        raise bad("line", "must be a JSON object")
    for name in _STR_FIELDS:
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise bad(name, "required: a non-empty string")
    index = raw.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise bad("candidate_index", "required: an integer >= 0")
    if not isinstance(raw.get("reply"), str):
        raise bad("reply", "required: a string")
    if "error" not in raw or not (raw["error"] is None or isinstance(raw["error"], str)):
        raise bad("error", "required: null or a string")
    row = raw.get("scenario")
    if not isinstance(row, dict):
        raise bad("scenario", "required: the scenario row object")
    try:
        scenario = parse_row(row)
    except _RowError as exc:
        raise bad(f"scenario.{exc.field}", exc.message) from None
    if scenario.id != raw["scenario_id"]:
        raise bad("scenario_id", f"{raw['scenario_id']!r} differs from scenario.id {scenario.id!r}")
    if scenario.gold is None:
        raise bad("scenario.gold", "required: training scenarios need a gold label")
    for name in ("category", "family"):
        if not isinstance(row.get(name), str):
            raise bad(f"scenario.{name}", "required: a string")
    candidate = Candidate(raw["scenario_id"], index, raw["reply"], raw["error"], scenario, row)
    return candidate, raw


def load_candidates(path: Path) -> list[tuple[Candidate, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc}") from None
    loaded = []
    seen: set[tuple[str, int]] = set()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}: line {lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{where}: invalid JSON: {exc.msg}") from None
        candidate, raw = _parse_candidate(raw, where)
        key = (candidate.scenario_id, candidate.candidate_index)
        if key in seen:
            raise UsageError(f"{where}: candidate_index: duplicate candidate {key}")
        seen.add(key)
        loaded.append((candidate, raw))
    if not loaded:
        raise UsageError(f"{path} holds no candidates")
    return loaded


# ---------------------------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------------------------


def judge_all(outcomes: Sequence[Outcome], backend: Retrying, concurrency: int) -> None:
    """Judge the candidates that survived the deterministic stages; scenarios run in parallel.

    Within a scenario, candidates are judged in index order and judging stops at the first one
    that passes with the top Robustness: no later candidate can beat it (ties go to the lowest
    index), so selection is unchanged and the judge is paid less.
    """

    def scenario(group: list[Outcome]) -> None:
        for outcome in group:
            c = outcome.candidate
            try:
                outcome.judge = judge_reply(c.scenario, c.reply.strip(), c.checks, backend)
            except BackendError as exc:
                outcome.rejection = Rejection(JUDGE_ERROR, f"judge request failed: {exc}")
                continue
            outcome.rejection = judge_stage(outcome.judge)
            if outcome.rejection is None and outcome.judge.robustness == MAX_ROBUSTNESS:
                return

    groups: dict[str, list[Outcome]] = {}
    for o in outcomes:
        if o.rejection is None:
            groups.setdefault(o.candidate.scenario_id, []).append(o)
    ordered = [sorted(g, key=lambda o: o.candidate.candidate_index) for g in groups.values()]
    if not ordered:
        return
    # The first scenario runs alone: a judge that rejects a decoding parameter is learned once
    # (the backend caches it), not once per concurrent request.
    scenario(ordered[0])
    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        list(pool.map(scenario, ordered[1:]))


def select(outcomes: Sequence[Outcome]) -> list[Outcome]:
    """Near-duplicate and one-per-scenario selection, in candidates-file scenario order."""
    by_scenario: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        by_scenario.setdefault(outcome.candidate.scenario_id, []).append(outcome)
    kept = []
    index = NearDuplicateIndex()
    for scenario_id, group in by_scenario.items():
        survivors = [o for o in group if o.rejection is None]
        if not survivors:
            continue
        final = final_user_message(survivors[0].candidate.scenario)
        if (dup := index.match(final)) is not None:
            for o in survivors:
                o.rejection = Rejection(NEAR_DUPLICATE, dup)
            continue
        best = min(
            survivors,
            key=lambda o: (-(o.judge.robustness if o.judge else 0), o.candidate.candidate_index),
        )
        for o in survivors:
            if o is not best:
                unjudged = "" if o.judge else " (not judged: an earlier candidate scored the top)"
                o.rejection = Rejection(
                    NOT_SELECTED,
                    f"candidate {best.candidate.candidate_index} of {scenario_id} was selected"
                    + unjudged,
                )
        index.add(scenario_id, final)
        kept.append(best)
    return kept


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def train_record(outcome: Outcome) -> dict[str, Any]:
    c, judge = outcome.candidate, outcome.judge
    assert judge is not None
    system, messages = render_messages(c.scenario, "zero_shot")
    return {
        "messages": [
            {"role": "system", "content": system},
            *messages,
            {"role": "assistant", "content": c.reply.strip()},
        ],
        "meta": {
            "scenario_id": c.scenario_id,
            "category": c.row["category"],
            "family": c.row["family"],
            "teacher": outcome.teacher,
            "filter_version": FILTER_VERSION,
            "candidate_index": c.candidate_index,
            "judge": judge.scores(),
        },
    }


def rejection_record(outcome: Outcome) -> dict[str, Any]:
    c, r = outcome.candidate, outcome.rejection
    assert r is not None
    record: dict[str, Any] = {
        "scenario_id": c.scenario_id,
        "candidate_index": c.candidate_index,
        "reason": r.reason,
        "detail": r.detail,
    }
    if r.failed_checks:
        record["failed_checks"] = list(r.failed_checks)
    if outcome.judge is not None:
        record["judge"] = outcome.judge.scores()
    return record


def _one_value(values: Sequence[str]) -> str:
    distinct = sorted(set(values))
    return distinct[0] if len(distinct) == 1 else ", ".join(distinct)


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def manifest(
    version: str,
    kept: Sequence[Outcome],
    outcomes: Sequence[Outcome],
    raws: Sequence[Mapping[str, Any]],
    judge_model: str,
    train_bytes: bytes,
    candidates_bytes: bytes,
) -> dict[str, Any]:
    rows = [o.candidate.row for o in kept]
    n_cands, n_kept = len(outcomes), len(kept)
    kept_rate = n_kept / n_cands if n_cands else 0.0
    reasons = Counter(o.rejection.reason for o in outcomes if o.rejection is not None)
    abstain_share = sum(r["gold"]["mode"] == "abstain" for r in rows) / n_kept if n_kept else 0.0
    return {
        "version": version,
        "count": n_kept,
        "counts_by_category": _sorted_counts(Counter(str(r["category"]) for r in rows)),
        "id_scheme_mix": _sorted_counts(
            Counter(re.sub(r"[0-9]+", "#", str(s["id"])) for r in rows for s in r["snippets"])
        ),
        "abstain_share": abstain_share,
        "abstain_share_in_range": (
            ABSTAIN_SHARE_RANGE[0] <= abstain_share <= ABSTAIN_SHARE_RANGE[1]
        ),
        "source_seeds": sorted({r["seed"] for r in rows if "seed" in r}),
        "teacher_model": _one_value([str(r["teacher"]) for r in raws]),
        "teacher_prompt_version": _one_value([str(r["teacher_prompt_version"]) for r in raws]),
        "teacher_prompt_sha256": _one_value([str(r["teacher_prompt_sha256"]) for r in raws]),
        "filter_version": FILTER_VERSION,
        "judge_model": judge_model,
        "rubric_sha256": rubric_sha256(),
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "candidates_sha256": hashlib.sha256(candidates_bytes).hexdigest(),
        "selection_rule": SELECTION_RULE,
        "candidates": n_cands,
        "kept": n_kept,
        "kept_rate": kept_rate,
        "kept_rate_in_range": KEPT_RATE_RANGE[0] <= kept_rate <= KEPT_RATE_RANGE[1],
        "rejections_by_reason": {r: reasons[r] for r in REJECTION_REASONS if reasons[r]},
    }


def filter_report(m: Mapping[str, Any]) -> str:
    low, high = KEPT_RATE_RANGE
    in_range = "yes" if m["kept_rate_in_range"] else "NO"
    lines = [
        f"# Filter report: {m['version']}",
        "",
        "Generated by `scripts/build_dataset.py` (R4).",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Candidates | {m['candidates']} |",
        f"| Kept | {m['kept']} |",
        f"| Kept rate | {100 * m['kept_rate']:.1f}% |",
        f"| Kept rate within {100 * low:.0f}-{100 * high:.0f}% | {in_range} |",
        f"| Abstain share of kept | {100 * m['abstain_share']:.1f}% |",
        f"| Abstain share within {100 * ABSTAIN_SHARE_RANGE[0]:.0f}-"
        f"{100 * ABSTAIN_SHARE_RANGE[1]:.0f}% | {'yes' if m['abstain_share_in_range'] else 'NO'} |",
        f"| Teacher | `{m['teacher_model']}` (prompt {m['teacher_prompt_version']}) |",
        f"| Judge | `{m['judge_model']}` |",
        f"| Filter version | {m['filter_version']} |",
        "",
        "## Rejections by reason",
        "",
        "Stages in order; a candidate is counted at the first stage it fails.",
        "",
        "| Reason | Count |",
        "| --- | --- |",
    ]
    by_reason = m["rejections_by_reason"]
    lines += [f"| `{reason}` | {by_reason.get(reason, 0)} |" for reason in REJECTION_REASONS]
    lines += ["", "## Kept by category", "", "| Category | Count |", "| --- | --- |"]
    lines += [f"| {cat} | {n} |" for cat, n in m["counts_by_category"].items()]
    lines += ["", f"Selection rule: {m['selection_rule']}.", ""]
    return "\n".join(lines)


def _jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise UsageError(f"{args.out} is not empty; datasets are never overwritten")
    judge_model = normalize_spec(args.judge or load_models_config()["judge"]["model"])
    loaded = load_candidates(args.candidates)
    backend = build_backend(judge_model)  # fails here when the judge key is missing

    outcomes = [Outcome(c, str(raw["teacher"])) for c, raw in loaded]
    for outcome in outcomes:
        outcome.rejection = deterministic_stage(outcome.candidate)
    judge_all(outcomes, backend, args.concurrency)
    kept = select(outcomes)

    train_bytes = _jsonl([train_record(o) for o in kept])
    rejections = _jsonl([rejection_record(o) for o in outcomes if o.rejection is not None])
    m = manifest(
        args.out.name,
        kept,
        outcomes,
        [raw for _, raw in loaded],
        judge_model,
        train_bytes,
        args.candidates.read_bytes(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_bytes(train_bytes)
    (args.out / "rejections.jsonl").write_bytes(rejections)
    (args.out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    (args.out / "filter_report.md").write_text(filter_report(m), encoding="utf-8")
    print(
        f"kept {m['kept']} of {m['candidates']} candidates ({100 * m['kept_rate']:.1f}%); "
        f"wrote {args.out}"
    )
    for reason, count in m["rejections_by_reason"].items():
        print(f"  {reason}: {count}")
    if not m["abstain_share_in_range"]:
        low, high = ABSTAIN_SHARE_RANGE
        print(
            f"warning: kept abstain share {100 * m['abstain_share']:.1f}% is outside "
            f"{100 * low:.0f}-{100 * high:.0f}%",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, MissingKeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
