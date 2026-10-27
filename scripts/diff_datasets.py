#!/usr/bin/env python3
"""Summarize what changed between two dataset versions (R7, GOTCHA G21).

    uv run scripts/diff_datasets.py data/datasets/v1 data/datasets/v2 \\
        [--json PATH] [--out PATH]

Examples are keyed by ``meta.scenario_id``. Reports added and removed scenarios, relabeled ones
(the assistant target text differs), mode changes (cite vs abstain, as the checks infer it) and a
per-category breakdown. Prints markdown; ``--out`` also writes it and ``--json`` writes the
summary object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from groundcite.checks import infer_mode


class UsageError(RuntimeError):
    pass


def load(ds: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = ds / "train.jsonl"
    if not path.is_file():
        raise UsageError(f"{path} is missing")
    data = path.read_bytes()
    records: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[str(record["meta"]["scenario_id"])] = record
        except (json.JSONDecodeError, KeyError, TypeError):
            raise UsageError(f"{path}: line {lineno} has no meta.scenario_id") from None
    version = ds.name
    manifest = ds / "manifest.json"
    if manifest.is_file():
        version = str(json.loads(manifest.read_text(encoding="utf-8")).get("version", version))
    side = {"version": version, "sha256": hashlib.sha256(data).hexdigest(), "count": len(records)}
    return side, records


def _target(record: Mapping[str, Any]) -> str:
    return str(record["messages"][-1]["content"])


def _category(record: Mapping[str, Any]) -> str:
    return str(record.get("meta", {}).get("category", "uncategorized"))


def diff(old: Path, new: Path) -> dict[str, Any]:
    old_side, a = load(old)
    new_side, b = load(new)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    common = sorted(set(a) & set(b))
    relabeled = [sid for sid in common if _target(a[sid]) != _target(b[sid])]
    mode_changes = [
        sid for sid in relabeled if infer_mode(_target(a[sid])) != infer_mode(_target(b[sid]))
    ]
    by_category: dict[str, dict[str, int]] = {}

    def bump(category: str, kind: str) -> None:
        counts = by_category.setdefault(category, {"added": 0, "removed": 0, "relabeled": 0})
        counts[kind] += 1

    for sid in added:
        bump(_category(b[sid]), "added")
    for sid in removed:
        bump(_category(a[sid]), "removed")
    for sid in relabeled:
        bump(_category(b[sid]), "relabeled")
    return {
        "old": old_side,
        "new": new_side,
        "added": added,
        "removed": removed,
        "relabeled": relabeled,
        "mode_changes": mode_changes,
        "unchanged": len(common) - len(relabeled),
        "by_category": dict(sorted(by_category.items())),
    }


def render(summary: Mapping[str, Any]) -> str:
    old, new = summary["old"], summary["new"]
    lines = [
        f"# Dataset diff: {old['version']} -> {new['version']}",
        "",
        f"- {old['version']}: {old['count']} examples, sha256 `{old['sha256'][:12]}`",
        f"- {new['version']}: {new['count']} examples, sha256 `{new['sha256'][:12]}`",
        "",
        "| Change | Count |",
        "| --- | --- |",
    ]
    for kind in ("added", "removed", "relabeled", "mode_changes"):
        lines.append(f"| {kind} | {len(summary[kind])} |")
    lines.append(f"| unchanged | {summary['unchanged']} |")
    if summary["by_category"]:
        lines += ["", "| Category | Added | Removed | Relabeled |", "| --- | --- | --- | --- |"]
        for cat, c in summary["by_category"].items():
            lines.append(f"| {cat} | {c['added']} | {c['removed']} | {c['relabeled']} |")
    for kind in ("added", "removed", "relabeled", "mode_changes"):
        ids = summary[kind]
        lines += ["", f"{kind.replace('_', ' ').capitalize()}: " + (", ".join(ids) or "none")]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diff_datasets.py", description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path, help="old dataset dir")
    parser.add_argument("new", type=Path, help="new dataset dir")
    parser.add_argument("--json", type=Path, help="write the summary JSON here")
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args(argv)
    try:
        summary = diff(args.old, args.new)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    md = render(summary)
    print(md, end="")
    if args.out:
        args.out.write_text(md, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
