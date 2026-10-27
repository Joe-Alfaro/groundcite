"""The canonical behavior spec (R1) and the citation/ID grammar every other module shares."""

from __future__ import annotations

import re
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parents[2] / "SPEC.md"

# IDs are opaque strings of 2-24 characters from [A-Za-z0-9._ -] with no brackets.
ID_CHARS = r"[A-Za-z0-9._ -]"
ID_RE = re.compile(rf"{ID_CHARS}{{2,24}}")
CITATION_RE = re.compile(rf"\[({ID_CHARS}{{2,24}})\]")


def load_spec(path: str | Path | None = None) -> str:
    """Return the spec text from ``path`` (default: the repo's ``SPEC.md``), stripped."""
    return Path(path if path is not None else SPEC_PATH).read_text(encoding="utf-8").strip()


def is_valid_id(value: str) -> bool:
    """True if ``value`` is a well-formed snippet ID."""
    return ID_RE.fullmatch(value) is not None


def parse_cited_ids(text: str) -> list[str]:
    """Return the distinct bracketed IDs in ``text``, in order of first appearance."""
    return list(dict.fromkeys(CITATION_RE.findall(text)))
