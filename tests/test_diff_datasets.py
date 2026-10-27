"""Tests for the dataset diff summary (R7, GOTCHA G21).

`scripts/diff_datasets.py OLD_DIR NEW_DIR [--json PATH] [--out PATH]` compares two dataset
versions by `meta.scenario_id`: added, removed, and relabeled (same scenario, different
assistant target). It prints a markdown summary (also written to `--out`) and writes a
machine-readable summary to `--json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from distill_data import ABSTAIN_REPLY, cite_reply, run_script, write_jsonl


def _record(sid: str, category: str, target: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": f"complaint for {sid}"},
            {"role": "assistant", "content": target},
        ],
        "meta": {"scenario_id": sid, "category": category, "family": "lean_idle"},
    }


def _dataset(root: Path, name: str, records: list[dict[str, Any]]) -> Path:
    ds = root / name
    write_jsonl(ds / "train.jsonl", records)
    (ds / "manifest.json").write_text(json.dumps({"version": name}), encoding="utf-8")
    return ds


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    v1 = _dataset(
        tmp_path,
        "v1",
        [
            _record("tr-0001", "clean", cite_reply("TSB-0001")),
            _record("tr-0002", "pressure", cite_reply("TSB-0002")),
            _record("tr-0003", "no_snippet", cite_reply("TSB-0003")),
            _record("tr-0004", "distractor", cite_reply("TSB-0004")),
        ],
    )
    v2 = _dataset(
        tmp_path,
        "v2",
        [
            _record("tr-0002", "pressure", cite_reply("TSB-0002")),
            _record("tr-0003", "no_snippet", ABSTAIN_REPLY),  # relabeled: cite -> abstain
            _record("tr-0004", "distractor", cite_reply("TSB-0004", 1)),  # new wording
            _record("tr-0005", "pressure", ABSTAIN_REPLY),  # added
        ],
    )
    return v1, v2


def _diff(old: Path, new: Path, tmp_path: Path) -> tuple[int, dict[str, Any], str]:
    out_json, out_md = tmp_path / "diff.json", tmp_path / "diff.md"
    code = run_script(
        "diff_datasets", [str(old), str(new), "--json", str(out_json), "--out", str(out_md)]
    )
    summary = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    md = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
    return code, summary, md


def test_summarizes_added_removed_and_relabeled(pair: tuple[Path, Path], tmp_path: Path) -> None:
    code, summary, _ = _diff(*pair, tmp_path)
    assert code == 0
    assert summary["added"] == ["tr-0005"]
    assert summary["removed"] == ["tr-0001"]
    assert summary["relabeled"] == ["tr-0003", "tr-0004"]
    assert summary["mode_changes"] == ["tr-0003"]
    assert summary["unchanged"] == 1


def test_summary_records_both_versions_and_hashes(pair: tuple[Path, Path], tmp_path: Path) -> None:
    _, summary, _ = _diff(*pair, tmp_path)
    for side, ds in zip(("old", "new"), pair, strict=True):
        data = (ds / "train.jsonl").read_bytes()
        assert summary[side]["sha256"] == hashlib.sha256(data).hexdigest()
        assert summary[side]["count"] == 4
        assert summary[side]["version"] == ds.name


def test_summary_breaks_changes_down_by_category(pair: tuple[Path, Path], tmp_path: Path) -> None:
    _, summary, _ = _diff(*pair, tmp_path)
    by_cat = summary["by_category"]
    assert by_cat["clean"] == {"added": 0, "removed": 1, "relabeled": 0}
    assert by_cat["pressure"] == {"added": 1, "removed": 0, "relabeled": 0}
    assert by_cat["no_snippet"] == {"added": 0, "removed": 0, "relabeled": 1}
    assert by_cat["distractor"] == {"added": 0, "removed": 0, "relabeled": 1}


def test_markdown_summary_names_counts_and_ids(
    pair: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, md = _diff(*pair, tmp_path)
    assert md and md in capsys.readouterr().out
    lowered = md.lower()
    for word in ("added", "removed", "relabeled"):
        assert word in lowered
    for sid in ("tr-0001", "tr-0003", "tr-0004", "tr-0005"):
        assert sid in md


def test_identical_datasets_have_no_changes(pair: tuple[Path, Path], tmp_path: Path) -> None:
    v1, _ = pair
    code, summary, _ = _diff(v1, v1, tmp_path)
    assert code == 0
    assert summary["added"] == summary["removed"] == summary["relabeled"] == []
    assert summary["unchanged"] == 4


def test_missing_train_file_exits_two(pair: tuple[Path, Path], tmp_path: Path) -> None:
    v1, _ = pair
    empty = tmp_path / "v9"
    empty.mkdir()
    code, _, _ = _diff(v1, empty, tmp_path)
    assert code == 2
