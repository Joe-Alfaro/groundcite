#!/usr/bin/env python3
"""Generate the ablation, eval_own and heldout scenario sets from the synthetic library (R22).

    uv run scripts/gen_scenarios.py --seed 1027 --out data/scenarios \\
        --coverage-doc docs/scenario-coverage.md

The same seed always produces the same files, byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from groundcite.scenarios import (
    GenerationError,
    LibraryError,
    generate,
    load_library,
    render_coverage,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="master seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--coverage-doc", type=Path, help="also write the coverage markdown here")
    args = parser.parse_args(argv)

    try:
        library = load_library()
        sets = generate(library, args.seed)
    except (LibraryError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        path = args.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(f"wrote {len(rows)} scenarios to {path}")
    if args.coverage_doc:
        args.coverage_doc.write_text(render_coverage(library, sets, args.seed), encoding="utf-8")
        print(f"wrote {args.coverage_doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
