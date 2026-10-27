"""R1: SPEC.md is the canonical spec, at most 2 sentences, quoted verbatim in the requirements."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = REPO_ROOT / "SPEC.md"
REQUIREMENTS_FILE = REPO_ROOT / "docs" / "requirements" / "small-model-finetune.md"


def _normalize(text: str) -> str:
    """Drop markdown emphasis and collapse whitespace."""
    text = text.replace("**", "").replace("__", "")
    return " ".join(text.split())


def _requirements_quote() -> str:
    lines = REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Behavior Spec (R1)")
    quote: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith(">"):
            quote.append(line[1:].strip())
        elif quote:
            break
    assert quote, "no blockquote found under '## Behavior Spec (R1)'"
    return " ".join(quote)


def test_spec_file_exists_and_is_not_empty() -> None:
    assert SPEC_FILE.is_file()
    assert SPEC_FILE.read_text(encoding="utf-8").strip()


def test_spec_is_at_most_two_sentences() -> None:
    text = _normalize(SPEC_FILE.read_text(encoding="utf-8"))
    sentence_ends = re.findall(r"[.!?](?=\s|$)", text)
    assert 1 <= len(sentence_ends) <= 2


def test_requirements_quote_matches_spec_file() -> None:
    assert _normalize(_requirements_quote()) == _normalize(SPEC_FILE.read_text(encoding="utf-8"))


def test_load_spec_returns_spec_file_text() -> None:
    from groundcite.spec import load_spec

    assert load_spec() == SPEC_FILE.read_text(encoding="utf-8").strip()


def test_spec_path_points_at_repo_spec_file() -> None:
    from groundcite.spec import SPEC_PATH

    assert Path(SPEC_PATH).resolve() == SPEC_FILE.resolve()


def test_load_spec_reads_the_given_path(tmp_path: Path) -> None:
    from groundcite.spec import load_spec

    other = tmp_path / "SPEC.md"
    other.write_text("\nOne sentence spec.\n", encoding="utf-8")
    assert load_spec(other) == "One sentence spec."


def test_parse_cited_ids_returns_distinct_ids_in_order() -> None:
    from groundcite.spec import parse_cited_ids

    text = "Inspect the hose [TSB-0417], see [SM 6.2.1] and again [TSB-0417]; also [B-19-045]."
    assert parse_cited_ids(text) == ["TSB-0417", "SM 6.2.1", "B-19-045"]


def test_parse_cited_ids_ignores_brackets_outside_the_id_grammar() -> None:
    from groundcite.spec import parse_cited_ids

    # 1 char (too short), 25 chars (too long), a colon and a hash are outside the grammar.
    text = "[x] [ABCDEFGHIJKLMNOPQRSTUVWXY] [Note: see below] [Ref#Q7/22] [ref_q7.22]"
    assert parse_cited_ids(text) == ["ref_q7.22"]
