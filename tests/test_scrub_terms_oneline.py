"""The scrub terms may be given on one line, comma- or pipe-separated.

Kaggle's secret Value field is single-line, so a multi-line ``SCRUB_TERMS`` paste keeps only its
first line. ``load_terms`` in ``scripts/scrub_check.py`` (which ``train/train_qlora.py`` uses)
therefore also splits each line on ``,`` and ``|``. Items are stripped (inner spaces kept);
blank items and ``#`` comment items are dropped, and a line that starts with ``#`` is a comment
line, dropped whole. A multi-line file without separators reads exactly as before.

Uses dummy terms only; the real terms file is never read.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from notebook_sandbox import run_notebook, source
from train_data import NOTEBOOK, REPO_ROOT, load_train_module

SCRIPT = REPO_ROOT / "scripts" / "scrub_check.py"
FILLED = {"HUB_REPO": "someone/gc-test", "KAGGLE_DATASET": ""}
FAKE_REPO_URL = "https://example.invalid/someone/repo.git"


def _scrub_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scrub_check_oneline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terms(tmp_path: Path, content: str) -> list[str]:
    path = tmp_path / "terms.txt"
    path.write_text(content, encoding="utf-8")
    terms: list[str] = _scrub_check().load_terms(path)
    return terms


# ---------------------------------------------------------------------------------------------
# load_terms
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ending", ["", "\n"], ids=["no-newline", "newline"])
def test_a_one_line_comma_list_is_split_into_stripped_terms(tmp_path: Path, ending: str) -> None:
    terms = _terms(tmp_path, "zebracorn,quuxfoo bar, baz-co" + ending)
    assert terms == ["zebracorn", "quuxfoo bar", "baz-co"]


def test_a_pipe_separates_terms_too(tmp_path: Path) -> None:
    assert _terms(tmp_path, "zebracorn|quuxfoo bar | baz-co") == [
        "zebracorn",
        "quuxfoo bar",
        "baz-co",
    ]


def test_blank_items_are_dropped(tmp_path: Path) -> None:
    assert _terms(tmp_path, " , zebracorn,, quuxfoo bar ,|, ") == ["zebracorn", "quuxfoo bar"]


def test_a_comment_item_is_dropped(tmp_path: Path) -> None:
    assert _terms(tmp_path, "zebracorn, # a note, quuxfoo bar") == ["zebracorn", "quuxfoo bar"]


def test_a_leading_comment_line_is_dropped(tmp_path: Path) -> None:
    assert _terms(tmp_path, "# dummy terms\nzebracorn, quuxfoo bar\n") == [
        "zebracorn",
        "quuxfoo bar",
    ]


def test_a_comment_line_with_commas_is_dropped_whole(tmp_path: Path) -> None:
    assert _terms(tmp_path, "# note, zebracorn\nquuxfoo bar, baz-co\n") == [
        "quuxfoo bar",
        "baz-co",
    ]


def test_newlines_and_commas_mix(tmp_path: Path) -> None:
    content = "zebracorn\nquuxfoo bar, baz-co\n\n# skip, this\nwibblet | flonk\n"
    assert _terms(tmp_path, content) == ["zebracorn", "quuxfoo bar", "baz-co", "wibblet", "flonk"]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_a_multi_line_file_without_separators_reads_as_before(tmp_path: Path, newline: str) -> None:
    lines = ["# dummy terms", "Zebracorn", "  quuxfoo bar  ", "", "   ", "baz-co", "#x"]
    assert _terms(tmp_path, newline.join(lines) + newline) == ["zebracorn", "quuxfoo bar", "baz-co"]


@pytest.mark.parametrize(
    "content",
    ["", "\n", "# only a comment", "# note, zebracorn", " , ,| ", ",#x, # y"],
    ids=["empty", "newline", "comment", "comment-with-comma", "separators", "comment-items"],
)
def test_an_empty_or_comment_only_one_line_value_fails_with_no_terms(
    tmp_path: Path,
    content: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "terms.txt"
    path.write_text(content, encoding="utf-8")
    module = _scrub_check()
    assert module.load_terms(path) == []
    monkeypatch.setenv("SCRUB_TERMS_PATH", str(path))
    assert module.main(["--repo", str(tmp_path)]) == 2
    assert "no terms" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# train_qlora and the Kaggle training notebook
# ---------------------------------------------------------------------------------------------


def test_train_qlora_reads_a_one_line_comma_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("zebracorn,quuxfoo bar, baz-co", encoding="utf-8")
    monkeypatch.setenv("SCRUB_TERMS_PATH", str(path))
    assert load_train_module()._load_scrub_terms() == ["zebracorn", "quuxfoo bar", "baz-co"]


def test_train_qlora_rejects_a_separator_only_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "terms.txt"
    path.write_text(" , ,| ", encoding="utf-8")
    monkeypatch.setenv("SCRUB_TERMS_PATH", str(path))
    module = load_train_module()
    with pytest.raises(module.UsageError, match="no terms"):
        module._load_scrub_terms()


def test_a_one_line_scrub_terms_secret_reaches_the_push_as_every_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook: dict[str, Any] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    run = run_notebook(
        notebook,
        params=FILLED,
        stop_at=lambda cmd: "train/train_qlora.py" in cmd,
        secrets={"SCRUB_TERMS": "zebracorn,quuxfoo bar, baz-co", "REPO_URL": FAKE_REPO_URL},
    )
    assert run.error is None, f"the filled-in notebook failed: {run.error!r}"
    path = Path(run.stop_env.get("SCRUB_TERMS_PATH", ""))
    try:
        assert path.is_absolute() and path.is_file(), "SCRUB_TERMS_PATH names the written file"
        monkeypatch.setenv("SCRUB_TERMS_PATH", str(path))
        terms = load_train_module()._load_scrub_terms()
        assert terms == ["zebracorn", "quuxfoo bar", "baz-co"]
    finally:
        if path.is_absolute() and path.is_file():
            path.unlink()


def test_the_notebook_says_scrub_terms_is_comma_separated_on_one_line() -> None:
    notebook: dict[str, Any] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = " ".join(
        " ".join(source(c).split()) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ).lower()
    note = markdown[markdown.find("`scrub_terms`") :][:300]
    assert "`scrub_terms`" in markdown, "the secrets note names SCRUB_TERMS"
    assert "comma-separated" in note, "SCRUB_TERMS is described as comma-separated"
    assert "one line" in note, "SCRUB_TERMS is described as fitting on one line"
