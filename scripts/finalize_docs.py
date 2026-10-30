#!/usr/bin/env python3
"""Render the final docs for one outcome of the pending items.

    uv run scripts/finalize_docs.py --mode approved|not-run [--values VALUES.json] [--out DIR]
    uv run scripts/finalize_docs.py --list-placeholders

All logic lives in ``groundcite.finalize``; run with ``--help`` for every flag.
"""

import sys

from groundcite.finalize import main

if __name__ == "__main__":
    sys.exit(main())
