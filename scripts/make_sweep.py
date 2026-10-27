#!/usr/bin/env python3
"""Nested, category-stratified subsets of a dataset for the data-efficiency curve (R8, G29).

    uv run scripts/make_sweep.py --dataset data/datasets/v1 --sizes 320 160 80 40 --seed 7

Shuffles ``<dataset>/train.jsonl`` once with the seed, stratified by ``meta.category``, and writes
each size as a prefix of that one order to ``<dataset>/subsets/n<N>.jsonl``, so every smaller
subset is contained in every larger one. Lines are copied unchanged. Also writes
``subsets/sweep_manifest.json`` with the seed and every file's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


def stratified_order(categories: list[str], seed: int) -> list[int]:
    """Row indices in one seeded order whose every prefix is stratified by category.

    Rows are shuffled within each category; the j-th of a category's n rows gets the position
    (j + offset) / n, with one random offset per category, and rows are sorted by position. Any
    prefix then holds each category within about one row of its pool share.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        by_cat.setdefault(cat, []).append(i)
    keyed = []
    for rank, cat in enumerate(sorted(by_cat)):
        rows = by_cat[cat]
        rng.shuffle(rows)
        offset = rng.random()
        keyed += [((j + offset) / len(rows), rank, i) for j, i in enumerate(rows)]
    return [i for _, _, i in sorted(keyed)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_sweep.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="dataset dir with train.jsonl")
    parser.add_argument("--sizes", type=int, nargs="+", required=True, help="subset sizes")
    parser.add_argument("--seed", type=int, required=True, help="shuffle seed")
    args = parser.parse_args(argv)

    pool_path = args.dataset / "train.jsonl"
    if not pool_path.is_file():
        print(f"error: {pool_path} is missing", file=sys.stderr)
        return 2
    pool_bytes = pool_path.read_bytes()
    lines = [line for line in pool_bytes.decode("utf-8").splitlines() if line.strip()]
    sizes = sorted(set(args.sizes))
    if sizes[0] < 1 or sizes[-1] > len(lines):
        print(f"error: sizes must be between 1 and the pool size {len(lines)}", file=sys.stderr)
        return 2
    try:
        categories = [str(json.loads(line)["meta"]["category"]) for line in lines]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"error: {pool_path}: every line needs meta.category ({exc!r})", file=sys.stderr)
        return 2

    order = stratified_order(categories, args.seed)
    out = args.dataset / "subsets"
    out.mkdir(exist_ok=True)
    subsets = {}
    for n in sorted(sizes, reverse=True):
        picked = order[:n]
        data = "".join(lines[i] + "\n" for i in picked).encode("utf-8")
        (out / f"n{n}.jsonl").write_bytes(data)
        subsets[f"n{n}"] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "count": n,
            "counts_by_category": dict(sorted(Counter(categories[i] for i in picked).items())),
        }
        print(f"wrote {out / f'n{n}.jsonl'} ({n} rows)")
    manifest = {
        "seed": args.seed,
        "pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "pool_count": len(lines),
        "sizes": sorted(sizes, reverse=True),
        "subsets": subsets,
    }
    (out / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
