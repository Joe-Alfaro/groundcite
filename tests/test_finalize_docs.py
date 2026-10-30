"""Tests for scripts/finalize_docs.py: render the final docs for one outcome of the pending items.

CLI: finalize_docs.py --mode approved|not-run [--values VALUES.json] [--root REPO] [--out DIR]
     finalize_docs.py --list-placeholders [--root REPO]

A pending item X has two alternative blocks, each opened by a marker line and closed by
``<!-- /PENDING-X -->``:

    <!-- PENDING-X:approved -->   ... text with {{placeholders}} ...
    <!-- PENDING-X:not-run -->    ... final-ready text ...

The mode keeps its own block's text, drops the other block, and removes every marker line. In
approved mode each ``{{name}}`` is filled from VALUES (a JSON object of strings). Any placeholder
left, a value that no placeholder uses, or a malformed marker is an error (exit 1, nothing
written). ``--out`` writes the rendered files under DIR at the same relative paths and leaves the
sources alone; without it the files are rewritten in place.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "finalize_docs.py"

_spec = importlib.util.spec_from_file_location("finalize_docs", SCRIPT)
assert _spec is not None and _spec.loader is not None
finalize_docs = importlib.util.module_from_spec(_spec)
sys.modules["finalize_docs"] = finalize_docs
_spec.loader.exec_module(finalize_docs)
DOC_FILES = finalize_docs.DOC_FILES
FinalizeError = finalize_docs.FinalizeError
placeholders = finalize_docs.placeholders
render = finalize_docs.render

SOURCE = """# Title

Before.

<!-- PENDING-A:approved -->
The curve gives N = {{min_n}}.
<!-- /PENDING-A -->
<!-- PENDING-A:not-run -->
The curve was not run.

Second paragraph.
<!-- /PENDING-A -->

| Req | Status |
| --- | --- |
| R1 | met |
<!-- PENDING-B:approved -->
| R7 | met: {{v2_spec}} |
<!-- /PENDING-B -->
<!-- PENDING-B:not-run -->
| R7 | designed, not run |
<!-- /PENDING-B -->
| R9 | met |

After.
"""


def test_not_run_keeps_its_blocks_and_drops_markers() -> None:
    out = render(SOURCE, "not-run", {})
    assert out == (
        "# Title\n\nBefore.\n\nThe curve was not run.\n\nSecond paragraph.\n\n"
        "| Req | Status |\n| --- | --- |\n| R1 | met |\n| R7 | designed, not run |\n"
        "| R9 | met |\n\nAfter.\n"
    )


def test_approved_fills_placeholders() -> None:
    out = render(SOURCE, "approved", {"min_n": "224", "v2_spec": "0.61"})
    assert "The curve gives N = 224." in out
    assert "| R1 | met |\n| R7 | met: 0.61 |\n| R9 | met |" in out
    assert "not run" not in out
    assert "<!--" not in out and "{{" not in out


def test_approved_missing_value_fails_and_names_it() -> None:
    with pytest.raises(FinalizeError, match="v2_spec"):
        render(SOURCE, "approved", {"min_n": "224"})


def test_unused_value_fails() -> None:
    with pytest.raises(FinalizeError, match="typo"):
        render(SOURCE, "approved", {"min_n": "1", "v2_spec": "2", "typo": "3"}, strict_unused=True)


def test_non_string_value_fails() -> None:
    with pytest.raises(FinalizeError, match="min_n"):
        render(SOURCE, "approved", {"min_n": 224, "v2_spec": "0.61"})  # type: ignore[dict-item]


def test_placeholder_outside_a_block_fails_in_not_run() -> None:
    with pytest.raises(FinalizeError, match="stray"):
        render("Text {{stray}}.\n", "not-run", {})


@pytest.mark.parametrize(
    "text",
    [
        "<!-- PENDING-A:approved -->\nx\n",  # never closed
        "x\n<!-- /PENDING-A -->\n",  # closed, never opened
        "<!-- PENDING-A:approved -->\nx\n<!-- /PENDING-B -->\n",  # wrong close
        "<!-- PENDING-A:maybe -->\nx\n<!-- /PENDING-A -->\n",  # unknown mode
        "<!-- PENDING-A:approved -->\n<!-- PENDING-B:approved -->\nx\n<!-- /PENDING-B -->\n"
        "<!-- /PENDING-A -->\n",  # nested
        "Inline <!-- PENDING-A:approved --> x <!-- /PENDING-A -->\n",  # not on its own line
    ],
)
def test_malformed_markers_fail(text: str) -> None:
    with pytest.raises(FinalizeError):
        render(text, "not-run", {})


def test_item_needs_both_alternatives() -> None:
    text = "<!-- PENDING-A:approved -->\nx\n<!-- /PENDING-A -->\n"
    with pytest.raises(FinalizeError, match="PENDING-A"):
        render(text, "not-run", {})


def test_placeholders_lists_names_in_order() -> None:
    assert placeholders(SOURCE) == ["min_n", "v2_spec"]


def _copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in DOC_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((REPO_ROOT / rel).read_text())
    return root


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_repo_docs_render_clean_not_run(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _run("--mode", "not-run", "--root", str(REPO_ROOT), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    for rel in DOC_FILES:
        text = (out / rel).read_text()
        assert "PENDING-" not in text, rel
        assert "{{" not in text and "}}" not in text, rel


def test_repo_docs_have_pending_items_and_render_approved(tmp_path: Path) -> None:
    names = placeholders("".join((REPO_ROOT / rel).read_text() for rel in DOC_FILES))
    assert names, "the approved blocks carry placeholders"
    values = tmp_path / "values.json"
    values.write_text(json.dumps({name: f"<{name}>" for name in names}))
    out = tmp_path / "out"
    proc = _run(
        "--mode", "approved", "--values", str(values), "--root", str(REPO_ROOT), "--out", str(out)
    )
    assert proc.returncode == 0, proc.stderr
    for rel in DOC_FILES:
        text = (out / rel).read_text()
        assert "PENDING-" not in text and "{{" not in text, rel


def test_cli_approved_missing_values_fails_and_writes_nothing(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    before = {rel: (root / rel).read_text() for rel in DOC_FILES}
    values = tmp_path / "values.json"
    values.write_text("{}")
    proc = _run("--mode", "approved", "--values", str(values), "--root", str(root))
    assert proc.returncode == 1
    assert "placeholder" in proc.stderr
    assert {rel: (root / rel).read_text() for rel in DOC_FILES} == before


def test_cli_approved_requires_values(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    proc = _run("--mode", "approved", "--root", str(root))
    assert proc.returncode != 0


def test_cli_in_place_rewrites_sources(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    proc = _run("--mode", "not-run", "--root", str(root))
    assert proc.returncode == 0, proc.stderr
    for rel in DOC_FILES:
        assert "PENDING-" not in (root / rel).read_text()


def test_cli_list_placeholders() -> None:
    proc = _run("--list-placeholders", "--root", str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr
    listed = proc.stdout.split()
    assert listed == placeholders("".join((REPO_ROOT / rel).read_text() for rel in DOC_FILES))
