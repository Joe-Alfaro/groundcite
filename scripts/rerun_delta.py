#!/usr/bin/env python3
"""Score delta between two runs of the same pins (R28).

    uv run scripts/rerun_delta.py RUN_A RUN_B [--out REPORT]

All logic lives in ``groundcite.rerun``; run with ``--help`` for every flag.
"""

import sys

from groundcite.rerun import main

if __name__ == "__main__":
    sys.exit(main())
