#!/usr/bin/env python3
"""Generate the dataset v2 additions: in-scope cite twins of the v1 abstain scenarios (R7).

    uv run scripts/gen_v2_additions.py [--seed S] [--out PATH]

Defaults reproduce the committed ``data/scenarios/train_v2_additions.jsonl`` byte for byte. Each
row is a training scenario plus ``source_id``, the v1 abstain scenario (``train_pool.jsonl``,
kept in ``data/datasets/v1``) it twins: same family, snippet texts and conversation shape, with
the vehicle moved so exactly one provided snippet applies and is cited. No final user message is
a near-duplicate of an eval final, a training-pool final or another twin's (R23).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from groundcite.leakage import LeakageInputError, load_rows
from groundcite.scenarios import GenerationError, LibraryError, load_library, load_train_phrasing
from groundcite.scenarios.library import DATA_DIR
from groundcite.scenarios.twins import TWIN_SEED, abstain_sources, generate_twins

SCENARIOS_DIR = DATA_DIR / "scenarios"
TRAIN_POOL = SCENARIOS_DIR / "train_pool.jsonl"
V1_MANIFEST = DATA_DIR / "datasets" / "v1" / "manifest.json"
DEFAULT_OUT = SCENARIOS_DIR / "train_v2_additions.jsonl"
EVAL_SETS = ("ablation.jsonl", "eval_own.jsonl", "heldout.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=TWIN_SEED, help="master seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL path")
    args = parser.parse_args(argv)

    try:
        library = load_library()
        phrasing = load_train_phrasing()
        pool = [json.loads(line) for line in TRAIN_POOL.read_text(encoding="utf-8").splitlines()]
        kept = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))["source_seeds"]
        sources = abstain_sources(pool, kept)
        avoid = [row.final_user for name in EVAL_SETS for row in load_rows(SCENARIOS_DIR / name)]
        avoid += [row["messages"][-1]["content"] for row in pool]
        rows = generate_twins(library, sources, args.seed, phrasing, avoid)
    except (LibraryError, GenerationError, LeakageInputError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} in-scope twins of {len(sources)} abstain scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
