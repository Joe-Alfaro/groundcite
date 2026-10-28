#!/usr/bin/env python3
"""Conversation rollouts: the model writes every assistant turn of scripted 8-12 turn chats (H1).

    uv run scripts/run_rollouts.py --model M --strategy S --scripts PATH [--limit N] [--dry-run]
    uv run scripts/run_rollouts.py --model M --strategy S --scripts PATH --run-id ID --resume

Every turn is checked and judged against its live snippets and gold, and the cell adds the
per-conversation pass rate (every turn on spec), which the bar is measured against. Conversations
are appended to transcripts.jsonl as they finish, so --resume skips finished ones, and refuses to
continue a run whose scripts, limit, model, strategy, prompts, judge or rubric changed. The
script format is docs/conversation-format.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from groundcite.ablation import ROBUSTNESS_BAR, SPEC_ADHERENCE_BAR
from groundcite.backends import BackendError
from groundcite.cli import check_clean_tree
from groundcite.config import load_models_config
from groundcite.evalset import EvalSetError
from groundcite.judge import rubric_sha256
from groundcite.keys import MissingKeyError
from groundcite.prompts import STRATEGIES, prompt_sha256
from groundcite.report import aggregate_scores
from groundcite.rollout import (
    ConversationScript,
    load_scripts,
    meets_rollout_bar,
    run_conversations,
)
from groundcite.runner import (
    BOOTSTRAP_SEED,
    DEFAULT_CONCURRENCY,
    MANIFEST,
    System,
    UsageError,
    build_backend,
    check_resumable,
    discard_if_empty,
    display_path,
    fmt,
    new_run_id,
    normalize_spec,
    pins,
    pins_section,
    results_table,
    sha256_file,
    split_hf,
    stop_message,
    write_json,
)

DEFAULT_OUT = Path("results") / "rollouts"
# What a resume must match; each key's label names the flag or input that changed.
MANIFEST_LABELS = {
    "scripts_sha256": "conversation scripts (sha256)",
    "conversation_ids": "conversation list (--scripts / --limit)",
    "model": "model (--model)",
    "strategy": "strategy (--strategy)",
    "prompt_sha256": "prompt hashes",
    "judge_model": "judge model",
    "rubric_sha256": "judge rubric",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_rollouts.py", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--model", required=True, help="the model under test (any backend spec)")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES, help="prompt strategy")
    parser.add_argument("--scripts", type=Path, required=True, help="conversation scripts (JSONL)")
    parser.add_argument("--limit", type=int, help="only the first N conversations")
    parser.add_argument("--judge", help="judge model (default: config; 'mock' = mock:judge)")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="conversations in parallel"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no calls")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root")
    parser.add_argument("--run-id", help="run directory name (default: a timestamp)")
    parser.add_argument(
        "--resume", action="store_true", help="continue --run-id, skipping finished conversations"
    )
    return parser


def _load(path: Path, limit: int | None) -> list[ConversationScript]:
    try:
        scripts = load_scripts(path)
    except EvalSetError as exc:
        raise UsageError(f"malformed conversation scripts {path}: {exc}") from None
    except OSError as exc:
        raise UsageError(f"can't read conversation scripts {path}: {exc}") from None
    if limit is not None:
        if limit < 1:
            raise UsageError("--limit must be at least 1")
        scripts = scripts[:limit]
    return scripts


def _print_plan(
    model: str, strategy: str, scripts: list[ConversationScript], judge: str, path: Path
) -> None:
    turns = sum(len(s.turns) for s in scripts)
    print(f"scripts: {display_path(path)} ({len(scripts)} conversations, {turns} turns)")
    print(f"system: {model} x {strategy}")
    print(f"judge: {judge}")
    print(f"generation calls: {turns}")
    print(f"judge calls: {turns}")
    print("dry run: no requests made, nothing written")


def _manifest(
    path: Path, scripts: list[ConversationScript], model: str, strategy: str, judge: str
) -> dict[str, Any]:
    return {
        "scripts_sha256": sha256_file(path),
        "conversation_ids": [s.id for s in scripts],
        "model": model,
        "strategy": strategy,
        "prompt_sha256": {s: prompt_sha256(s) for s in STRATEGIES},
        "judge_model": judge,
        "rubric_sha256": rubric_sha256(),
    }


def _results_md(results: dict[str, Any]) -> str:
    p = results["pins"]
    cells = results["cells"]
    for c in cells:
        c["_extra"] = fmt(c["conversation_pass_rate"])
    table = results_table(cells, extra="Per-conversation pass")
    for c in cells:
        c.pop("_extra", None)
    lines = [
        f"# Conversation rollouts `{results['run_id']}`",
        "",
        f"{len(results['conversation_ids'])} conversations; the model wrote every assistant turn "
        f"and each turn was scored on its own. Judge `{p['judge_model']}`, rubric sha256 "
        f"`{p['rubric_sha256']}`, eval code `{p['git_sha']}`. The per-turn columns pool every "
        "turn; the per-conversation pass rate counts a conversation only when every one of its "
        "turns is on spec. Generation errors are scored as failures.",
        "",
        *table,
        "",
        f"Bar: per-conversation pass rate >= {SPEC_ADHERENCE_BAR:.2f} and mean Robustness >= "
        f"{ROBUSTNESS_BAR} (inclusive).",
        "",
    ]
    for c in cells:
        verdict = "meets the bar" if c["meets_bar"] else "does not meet the bar"
        first = ", ".join(f"turn {t}: {n}" for t, n in c["first_failure_turns"].items()) or "none"
        by_turn = " | ".join(fmt(v) for v in c["spec_adherence_by_turn"].values())
        lines += [
            f"## {c['model']} x {c['strategy']}: {verdict}",
            "",
            f"- Conversations: {c['n_conversations']}, per-conversation pass "
            f"{fmt(c['conversation_pass_rate'])}",
            f"- First failing turn (conversations): {first}",
            "",
            "| " + " | ".join(c["spec_adherence_by_turn"]) + " |",
            "|" + "---|" * len(c["spec_adherence_by_turn"]),
            f"| {by_turn} |",
            "",
            "Spec-adherence by turn index (header row: turn index).",
            "",
        ]
    lines += [*pins_section(p), ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if args.resume and not args.run_id:
        raise UsageError("--resume needs --run-id")
    scripts = _load(args.scripts, args.limit)
    judge_spec = normalize_spec(args.judge or load_models_config()["judge"]["model"])
    model = normalize_spec(args.model)
    if args.dry_run:
        _print_plan(model, args.strategy, scripts, judge_spec, args.scripts)
        return 0

    run_id = args.run_id or new_run_id()
    run_dir = args.out / run_id
    if run_dir.exists() and not args.resume:
        raise UsageError(f"{run_dir} already exists; pass --resume to continue it")
    dirty = check_clean_tree(args.allow_dirty)
    revision = split_hf(model)[1] if model.startswith("hf:") else None
    subject = build_backend(model)
    judge = build_backend(judge_spec)
    manifest = _manifest(args.scripts, scripts, model, args.strategy, judge.name)
    resuming = run_dir.exists()
    if resuming:
        check_resumable(run_dir, manifest, MANIFEST_LABELS)
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / MANIFEST, manifest)
    try:
        rows = run_conversations(
            run_id=run_id,
            scripts=scripts,
            subject=subject,
            strategy=args.strategy,
            judge=judge,
            transcripts=run_dir / "transcripts.jsonl",
            model_revision=revision,
            concurrency=args.concurrency,
        )
    except BaseException:
        if not resuming:
            discard_if_empty(run_dir)
        raise
    cells = aggregate_scores(rows, seed=BOOTSTRAP_SEED)
    for cell in cells:
        cell["meets_bar"] = meets_rollout_bar(cell)
    results = {
        "run_id": run_id,
        "pins": pins(
            eval_set=args.scripts,
            judge=judge,
            systems=[System(model, args.strategy)],
            backends={model: subject},
            revisions={model: revision},
            git_dirty=dirty,
        ),
        "conversation_ids": [s.id for s in scripts],
        "cells": cells,
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
        if exc.stops_run:
            print(stop_message(exc), file=sys.stderr)
            return 2
        print(
            f"error: run stopped: {exc}; finished conversations are saved, rerun with --resume",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
