#!/usr/bin/env python3
"""Score models on an eval set with the pinned judge (R10).

    uv run eval.py --model <hf-repo-id> --eval-set <path>

All logic lives in ``groundcite.cli``; run with ``--help`` for every flag.
"""

import sys

from groundcite.cli import main

if __name__ == "__main__":
    sys.exit(main())
