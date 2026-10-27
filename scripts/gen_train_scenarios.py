#!/usr/bin/env python3
"""Generate the training scenario pool the teacher distills from (R4, R22, R23, R30).

    uv run scripts/gen_train_scenarios.py [--seed S] [--n N] [--out PATH]

Defaults reproduce the committed ``data/scenarios/train_pool.jsonl`` byte for byte. Training
scenarios use only non-reserved families, their own master seed and their own phrase bank
(``data/train_phrasing.toml``), and no final user message is a near-duplicate of an eval final.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from groundcite.leakage import LeakageInputError, load_rows
from groundcite.scenarios import (
    TRAIN_POOL_SIZE,
    TRAIN_SEED,
    GenerationError,
    LibraryError,
    generate_train,
    load_library,
    load_train_phrasing,
)
from groundcite.scenarios.library import DATA_DIR

SCENARIOS_DIR = DATA_DIR / "scenarios"
DEFAULT_OUT = SCENARIOS_DIR / "train_pool.jsonl"
EVAL_SETS = ("ablation.jsonl", "eval_own.jsonl", "heldout.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=TRAIN_SEED, help="master seed")
    parser.add_argument("--n", type=int, default=TRAIN_POOL_SIZE, help="number of scenarios")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL path")
    args = parser.parse_args(argv)

    try:
        library = load_library()
        phrasing = load_train_phrasing()
        avoid = [row.final_user for name in EVAL_SETS for row in load_rows(SCENARIOS_DIR / name)]
        rows = generate_train(library, args.seed, args.n, phrasing, avoid)
    except (LibraryError, GenerationError, LeakageInputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} training scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
