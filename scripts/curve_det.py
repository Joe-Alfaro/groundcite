#!/usr/bin/env python3
"""Deterministic-check metrics (no judge) per data-efficiency curve point (R8).

    uv run scripts/curve_det.py --scenarios data/scenarios/heldout.jsonl \
        --transcripts RUN/transcripts.jsonl --point 56=GENS.jsonl ... --out OUT.json

All logic lives in ``groundcite.detmetrics``; run with ``--help`` for every flag.
"""

import sys

from groundcite.detmetrics import main

if __name__ == "__main__":
    sys.exit(main())
