"""Render the final docs for one outcome of the pending items.

``scripts/finalize_docs.py`` runs ``main``. A pending item X has two alternative blocks in each
doc that mentions it, each on its own marker lines:

    <!-- PENDING-X:approved -->
    ... text with {{placeholders}} ...
    <!-- /PENDING-X -->
    <!-- PENDING-X:not-run -->
    ... final-ready text ...
    <!-- /PENDING-X -->

The chosen mode keeps its own blocks' text, drops the other blocks and removes every marker
line. In approved mode each ``{{name}}`` is filled from a JSON object of strings. A placeholder
left over, a value no placeholder uses, or a malformed marker is an error, and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

MODES = ("approved", "not-run")
DOC_FILES = (
    "README.md",
    "docs/writeup.md",
    "docs/final-checklist.md",
    "docs/checklist.md",
    "results/curve/heldout/README.md",
    "results/training/curve/README.md",
)

OPEN = re.compile(r"^<!-- PENDING-([A-Za-z0-9]+):([a-z-]+) -->$")
CLOSE = re.compile(r"^<!-- /PENDING-([A-Za-z0-9]+) -->$")
ANY_MARKER = re.compile(r"<!-- /?PENDING-")
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


class FinalizeError(Exception):
    """The docs or the values can't be rendered."""


def placeholders(text: str) -> list[str]:
    """Placeholder names in order of first appearance, without repeats."""
    return list(dict.fromkeys(PLACEHOLDER.findall(text)))


def _select(text: str, mode: str) -> str:
    kept: list[str] = []
    current: tuple[str, str] | None = None
    seen: dict[str, set[str]] = {}
    for lineno, line in enumerate(text.splitlines(keepends=True), start=1):
        bare = line.rstrip("\r\n")
        opened, closed = OPEN.match(bare), CLOSE.match(bare)
        if opened:
            item, block_mode = opened.groups()
            if block_mode not in MODES:
                raise FinalizeError(
                    f"line {lineno}: unknown mode {block_mode!r} for PENDING-{item}"
                )
            if current is not None:
                raise FinalizeError(
                    f"line {lineno}: PENDING-{item} opens inside PENDING-{current[0]}"
                )
            current = (item, block_mode)
            seen.setdefault(item, set()).add(block_mode)
        elif closed:
            if current is None or closed.group(1) != current[0]:
                raise FinalizeError(
                    f"line {lineno}: PENDING-{closed.group(1)} closes no open block"
                )
            current = None
        elif ANY_MARKER.search(line):
            raise FinalizeError(f"line {lineno}: a PENDING marker must be alone on its line")
        elif current is None or current[1] == mode:
            kept.append(line)
    if current is not None:
        raise FinalizeError(f"PENDING-{current[0]} is never closed")
    for item, modes in seen.items():
        if set(MODES) - modes:
            raise FinalizeError(f"PENDING-{item} needs both alternatives: {', '.join(MODES)}")
    return "".join(kept)


def render(text: str, mode: str, values: Mapping[str, str], *, strict_unused: bool = False) -> str:
    """Keep ``mode``'s blocks, drop the rest and fill placeholders from ``values``."""
    if mode not in MODES:
        raise FinalizeError(f"unknown mode {mode!r}")
    for name, value in values.items():
        if not isinstance(value, str):
            raise FinalizeError(f"value {name!r} must be a string, written as it should appear")
    selected = _select(text, mode)
    missing = [name for name in placeholders(selected) if name not in values]
    if missing:
        raise FinalizeError(f"no value for placeholder(s): {', '.join(missing)}")
    if strict_unused:
        unused = sorted(set(values) - set(placeholders(selected)))
        if unused:
            raise FinalizeError(f"value(s) used by no placeholder: {', '.join(unused)}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], selected)


def _load_values(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise FinalizeError(f"{path}: expected a JSON object of placeholder values")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--values", type=Path, help="JSON object of strings (approved mode)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, help="write here instead of rewriting in place")
    parser.add_argument("--list-placeholders", action="store_true")
    args = parser.parse_args(argv)

    sources = {rel: (args.root / rel).read_text() for rel in DOC_FILES}
    if args.list_placeholders:
        print("\n".join(placeholders("".join(sources.values()))))
        return 0
    if args.mode is None:
        parser.error("--mode is required")
    if args.mode == "approved" and args.values is None:
        parser.error("--mode approved needs --values")

    try:
        values = _load_values(args.values)
        rendered = {rel: render(text, args.mode, values) for rel, text in sources.items()}
        left = [f"{rel}: {name}" for rel, text in rendered.items() for name in placeholders(text)]
        if left:
            raise FinalizeError(f"placeholder(s) left unfilled: {'; '.join(left)}")
        used = set(placeholders("".join(_select(t, args.mode) for t in sources.values())))
        unused = sorted(set(values) - used)
        if unused:
            raise FinalizeError(f"value(s) used by no placeholder: {', '.join(unused)}")
    except FinalizeError as err:
        print(f"finalize_docs: {err}", file=sys.stderr)
        return 1

    target = args.out if args.out is not None else args.root
    for rel, text in rendered.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")
    return 0
