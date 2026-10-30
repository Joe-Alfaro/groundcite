#!/usr/bin/env python3
"""Assemble a dataset version from an unchanged base and newly filtered additions (R7, G21).

    uv run scripts/assemble_dataset.py --base data/datasets/v1 \\
        --additions data/datasets/v2/additions --out data/datasets/v2 [--expect-base-sha256 HEX]

Both parts are ``scripts/build_dataset.py`` outputs. ``<out>/train.jsonl`` is the base's
train.jsonl bytes followed by the additions', so every base line stays byte-identical, and
``<out>/manifest.json`` sums the parts and keeps each part's manifest under ``parts``. Refused
with exit 2, nothing written: an existing ``<out>/train.jsonl`` or ``manifest.json``, a part whose
train.jsonl differs from its manifest's ``train_sha256`` (or the base from
``--expect-base-sha256``), a scenario id in both parts, or parts built with a different filter
version, teacher prompt or selection rule. The judge may differ between parts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# R23/R30, as in scripts/build_dataset.py.
ABSTAIN_SHARE_RANGE = (0.20, 0.30)
# v2 changes the data only: the parts must share how they were generated and filtered.
SAME_IN_EVERY_PART = (
    "teacher_model",
    "teacher_prompt_version",
    "teacher_prompt_sha256",
    "filter_version",
    "selection_rule",
)


class UsageError(RuntimeError):
    """Bad arguments or input: reported on stderr, exit 2, nothing written."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assemble_dataset.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--base", type=Path, required=True, help="unchanged dataset dir (v1)")
    parser.add_argument("--additions", type=Path, required=True, help="filtered additions dir")
    parser.add_argument("--out", type=Path, required=True, help="new dataset dir, e.g. v2")
    parser.add_argument("--expect-base-sha256", help="the base train.jsonl must hash to this")
    return parser


def load_part(path: Path) -> tuple[bytes, dict[str, Any], list[str]]:
    """A part's train.jsonl bytes, its manifest and its scenario ids, checked against each other."""
    try:
        data = (path / "train.jsonl").read_bytes()
        manifest = dict(json.loads((path / "manifest.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read the dataset in {path}: {exc}") from None
    sha = hashlib.sha256(data).hexdigest()
    if manifest.get("train_sha256") != sha:
        raise UsageError(f"{path}/train.jsonl hashes to {sha}, not its manifest's train_sha256")
    ids = [
        str(json.loads(line)["meta"]["scenario_id"])
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != manifest.get("count"):
        raise UsageError(
            f"{path}: {len(ids)} examples, but the manifest count is {manifest['count']}"
        )
    return data, manifest, ids


def _summed(dicts: list[dict[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for d in dicts:
        total.update(d)
    return dict(sorted(total.items()))


def assemble(
    version: str, parts: list[tuple[Path, bytes, dict[str, Any]]]
) -> tuple[bytes, dict[str, Any]]:
    data = b"".join(part_data for _, part_data, _ in parts)
    manifests = [m for _, _, m in parts]
    count = sum(int(m["count"]) for m in manifests)
    abstain = sum(round(float(m["abstain_share"]) * int(m["count"])) for m in manifests)
    share = abstain / count if count else 0.0
    manifest: dict[str, Any] = {
        "version": version,
        "count": count,
        "counts_by_category": _summed([m["counts_by_category"] for m in manifests]),
        "id_scheme_mix": _summed([m["id_scheme_mix"] for m in manifests]),
        "abstain_share": share,
        "abstain_share_in_range": ABSTAIN_SHARE_RANGE[0] <= share <= ABSTAIN_SHARE_RANGE[1],
        "source_seeds": sorted({s for m in manifests for s in m.get("source_seeds", [])}),
        **{key: manifests[0].get(key) for key in SAME_IN_EVERY_PART},
        "train_sha256": hashlib.sha256(data).hexdigest(),
        "parts": [
            {"path": str(path), **{k: v for k, v in m.items() if k != "source_seeds"}}
            for path, _, m in parts
        ],
    }
    return data, manifest


def run(args: argparse.Namespace) -> int:
    for name in ("train.jsonl", "manifest.json"):
        if (args.out / name).exists():
            raise UsageError(f"{args.out / name} already exists; datasets are never overwritten")
    base_data, base, base_ids = load_part(args.base)
    if args.expect_base_sha256 and base["train_sha256"] != args.expect_base_sha256:
        raise UsageError(
            f"{args.base}/train.jsonl sha256 is {base['train_sha256']}, "
            f"not the expected {args.expect_base_sha256}"
        )
    add_data, additions, add_ids = load_part(args.additions)
    both = sorted(set(base_ids) & set(add_ids))
    if both:
        raise UsageError(f"scenario ids in both parts: {', '.join(both[:5])}")
    for key in SAME_IN_EVERY_PART:
        if base.get(key) != additions.get(key):
            raise UsageError(
                f"{key} differs: {base.get(key)!r} (base) vs {additions.get(key)!r} (additions)"
            )

    data, manifest = assemble(
        args.out.name, [(args.base, base_data, base), (args.additions, add_data, additions)]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_bytes(data)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out}: {manifest['count']} examples ({base['count']} base + "
        f"{additions['count']} added), abstain share {100 * manifest['abstain_share']:.1f}%"
    )
    if not manifest["abstain_share_in_range"]:
        low, high = ABSTAIN_SHARE_RANGE
        print(
            f"warning: abstain share is outside {100 * low:.0f}-{100 * high:.0f}%", file=sys.stderr
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
