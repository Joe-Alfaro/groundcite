"""Pre-staged dataset v2 build (R7, G21, D-146): assemble v1 + the kept twins, and the gated script.

``scripts/assemble_dataset.py --base DIR --additions DIR --out DIR [--expect-base-sha256 HEX]``
writes ``<out>/train.jsonl`` (the base train.jsonl bytes, unchanged, then the additions') and
``<out>/manifest.json``. It refuses, with exit 2 and nothing written, when ``<out>`` already holds
a train.jsonl or manifest.json, when either part's train.jsonl differs from its manifest's
``train_sha256`` (or the base from ``--expect-base-sha256``), when a scenario id is in both parts,
or when the parts differ in filter version, teacher prompt or selection rule (v2 changes the data
only). The judge may differ; each part's manifest is kept under ``parts``.

``scripts/build_v2.sh [--approved] [--filter opus|gpt-5.5]`` defaults to a dry run: it prints the
plan and runs ``distill.py --dry-run`` (no request, nothing written). Only ``--approved`` with a
``--filter`` runs the paid steps. The approved path is never run here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from distill_data import ABSTAIN_REPLY, cite_reply, run_script, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_V2 = REPO_ROOT / "scripts" / "build_v2.sh"
V1_SHA = "ae470ee1a6eb771bae3e7f1b4dcaa5764e457e101b4023ad101711d7afcbd179"
SHARED = {
    "teacher_model": "anthropic:claude-opus-5-5",
    "teacher_prompt_version": "t1",
    "teacher_prompt_sha256": "a" * 64,
    "filter_version": "f2",
    "selection_rule": "one example per scenario",
}


def _record(sid: str, category: str, target: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"complaint for {sid}"},
            {"role": "assistant", "content": target},
        ],
        "meta": {"scenario_id": sid, "category": category, "family": "lean_idle"},
    }


def _part(
    root: Path, name: str, records: list[dict[str, Any]], abstain: int, **override: Any
) -> Path:
    write_jsonl(root / name / "train.jsonl", records)
    data = (root / name / "train.jsonl").read_bytes()
    manifest = {
        "version": name,
        "count": len(records),
        "counts_by_category": {},
        "id_scheme_mix": {"TSB-#": len(records)},
        "abstain_share": abstain / len(records),
        "source_seeds": [len(name), 7],
        **SHARED,
        "judge_model": "anthropic:claude-opus-5-5",
        "train_sha256": hashlib.sha256(data).hexdigest(),
        "kept": len(records),
        "candidates": 2 * len(records),
        **override,
    }
    for r in records:
        cat = r["meta"]["category"]
        manifest["counts_by_category"][cat] = manifest["counts_by_category"].get(cat, 0) + 1
    (root / name / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / name


@pytest.fixture
def parts(tmp_path: Path) -> tuple[Path, Path]:
    base = _part(
        tmp_path,
        "v1",
        [
            _record("tr-0001", "clean", cite_reply("TSB-0001")),
            _record("tr-0002", "no_snippet", ABSTAIN_REPLY),
            _record("tr-0003", "pressure", cite_reply("TSB-0003")),
        ],
        abstain=1,
    )
    additions = _part(
        tmp_path / "v2",
        "additions",
        [
            _record("v2-0001", "distractor", cite_reply("TSB-0002")),
            _record("v2-0002", "pressure", cite_reply("TSB-0004")),
        ],
        abstain=0,
        judge_model="openai:gpt-5.5-2026-04-23",
    )
    return base, additions


def _assemble(base: Path, additions: Path, out: Path, *extra: str) -> int:
    argv = ["--base", str(base), "--additions", str(additions), "--out", str(out), *extra]
    return run_script("assemble_dataset", argv)


def test_train_is_the_base_bytes_then_the_additions(parts: tuple[Path, Path]) -> None:
    base, additions = parts
    out = additions.parent
    assert _assemble(base, additions, out) == 0
    expected = (base / "train.jsonl").read_bytes() + (additions / "train.jsonl").read_bytes()
    assert (out / "train.jsonl").read_bytes() == expected


def test_the_manifest_sums_the_parts(parts: tuple[Path, Path]) -> None:
    base, additions = parts
    out = additions.parent
    assert _assemble(base, additions, out) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    data = (out / "train.jsonl").read_bytes()
    assert m["version"] == "v2"
    assert m["count"] == 5
    assert m["counts_by_category"] == {"clean": 1, "distractor": 1, "no_snippet": 1, "pressure": 2}
    assert m["id_scheme_mix"] == {"TSB-#": 5}
    assert m["abstain_share"] == pytest.approx(1 / 5)
    assert m["abstain_share_in_range"] is True
    assert m["train_sha256"] == hashlib.sha256(data).hexdigest()
    assert m["source_seeds"] == [2, 7, 9]
    for key, value in SHARED.items():
        assert m[key] == value
    assert [p["version"] for p in m["parts"]] == ["v1", "additions"]
    assert m["parts"][1]["judge_model"] == "openai:gpt-5.5-2026-04-23"
    assert (
        m["parts"][0]["train_sha256"]
        == json.loads((base / "manifest.json").read_text(encoding="utf-8"))["train_sha256"]
    )


def test_the_abstain_share_is_flagged_outside_20_to_30_percent(tmp_path: Path) -> None:
    base = _part(tmp_path, "v1", [_record("tr-0001", "clean", cite_reply("TSB-0001"))], 0)
    additions = _part(
        tmp_path / "v2", "additions", [_record("v2-0001", "clean", cite_reply("TSB-0002"))], 0
    )
    assert _assemble(base, additions, additions.parent) == 0
    m = json.loads((additions.parent / "manifest.json").read_text(encoding="utf-8"))
    assert m["abstain_share"] == 0
    assert m["abstain_share_in_range"] is False


def test_the_expected_base_hash_must_match(
    parts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base, additions = parts
    out = additions.parent
    assert _assemble(base, additions, out, "--expect-base-sha256", V1_SHA) == 2
    assert "sha256" in capsys.readouterr().err
    assert not (out / "train.jsonl").exists()
    assert not (out / "manifest.json").exists()


def _refused(base: Path, additions: Path, out: Path, capsys: pytest.CaptureFixture[str]) -> str:
    assert _assemble(base, additions, out) == 2
    assert not (out / "manifest.json").exists()
    return capsys.readouterr().err


def test_an_existing_dataset_is_never_overwritten(
    parts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base, additions = parts
    out = additions.parent
    (out / "train.jsonl").write_text("old\n", encoding="utf-8")
    assert "train.jsonl" in _refused(base, additions, out, capsys)
    assert (out / "train.jsonl").read_text(encoding="utf-8") == "old\n"


def test_a_part_that_differs_from_its_manifest_is_refused(
    parts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base, additions = parts
    with (additions / "train.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_record("v2-0003", "clean", cite_reply("TSB-0005"))) + "\n")
    assert "train_sha256" in _refused(base, additions, additions.parent, capsys)
    assert not (additions.parent / "train.jsonl").exists()


def test_a_scenario_in_both_parts_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _part(tmp_path, "v1", [_record("tr-0001", "clean", cite_reply("TSB-0001"))], 0)
    additions = _part(
        tmp_path / "v2", "additions", [_record("tr-0001", "clean", cite_reply("TSB-0001"))], 0
    )
    assert "tr-0001" in _refused(base, additions, additions.parent, capsys)


@pytest.mark.parametrize("key", ["filter_version", "teacher_prompt_sha256", "selection_rule"])
def test_parts_built_differently_are_refused(
    tmp_path: Path, key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _part(tmp_path, "v1", [_record("tr-0001", "clean", cite_reply("TSB-0001"))], 0)
    additions = _part(
        tmp_path / "v2",
        "additions",
        [_record("v2-0001", "clean", cite_reply("TSB-0002"))],
        0,
        **{key: "other"},
    )
    assert key in _refused(base, additions, additions.parent, capsys)


def test_the_real_v1_is_a_valid_base(tmp_path: Path) -> None:
    v1 = REPO_ROOT / "data" / "datasets" / "v1"
    additions = _part(
        tmp_path / "v2",
        "additions",
        [_record("v2-0001", "distractor", cite_reply("TSB-0002"))],
        0,
        **{k: json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))[k] for k in SHARED},
    )
    out = additions.parent
    assert _assemble(v1, additions, out, "--expect-base-sha256", V1_SHA) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["count"] == 449
    assert m["abstain_share"] == pytest.approx(126 / 449)


# ---------------------------------------------------------------------------------------------
# scripts/build_v2.sh
# ---------------------------------------------------------------------------------------------


def _build(*args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    return subprocess.run(
        ["bash", str(BUILD_V2), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


OUTPUTS = (REPO_ROOT / "data" / "distill" / "v2_additions", REPO_ROOT / "data" / "datasets" / "v2")


def _snapshot() -> dict[Path, list[str]]:
    """What the script could write; v2 may already be built, so compare before and after."""
    return {p: sorted(str(f) for f in p.rglob("*")) if p.exists() else [] for p in OUTPUTS}


@pytest.fixture
def untouched() -> Iterator[None]:
    before = _snapshot()
    yield
    assert _snapshot() == before, "the script wrote outputs"


def test_build_v2_is_an_executable_strict_shell_script() -> None:
    text = BUILD_V2.read_text(encoding="utf-8")
    assert text.startswith("#!")
    assert "set -euo pipefail" in text
    assert os.access(BUILD_V2, os.X_OK)


def test_build_v2_runs_the_pipeline_in_order() -> None:
    text = BUILD_V2.read_text(encoding="utf-8")
    steps = [
        "scripts/distill.py",
        "scripts/build_dataset.py",
        "scripts/assemble_dataset.py",
        "scripts/diff_datasets.py",
        "scripts/check_leakage.py",
        "scripts/scrub_check.py",
        "git commit",
    ]
    # The last mention of each: the scrub check also runs as a preflight, before any paid call.
    positions = [text.rfind(step) for step in steps]
    assert all(p >= 0 for p in positions), dict(zip(steps, positions, strict=True))
    assert positions == sorted(positions)
    assert "--teacher anthropic:claude-opus-5-5" in text
    assert "--k 2" in text
    assert "data/scenarios/train_v2_additions.jsonl" in text
    assert V1_SHA in text


def test_the_default_is_a_dry_run_that_writes_nothing(untouched: None) -> None:
    result = _build()
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "dry run" in out.lower()
    assert "teacher calls: 210" in out
    assert "--approved" in out


@pytest.mark.parametrize(
    ("name", "spec"),
    [("opus", "anthropic:claude-opus-5-5"), ("gpt-5.5", "openai:gpt-5.5-2026-04-23")],
)
def test_a_dry_run_names_the_filter_judge(name: str, spec: str, untouched: None) -> None:
    result = _build("--filter", name)
    assert result.returncode == 0, result.stderr
    assert spec in result.stdout


@pytest.mark.parametrize(
    "args",
    [["--approved"], ["--filter", "sonnet"], ["--approved", "--filter", "haiku"], ["--bogus"]],
    ids=["approved-without-filter", "unknown-filter", "approved-unknown-filter", "unknown-flag"],
)
def test_bad_arguments_exit_2_before_any_step(args: list[str], untouched: None) -> None:
    result = _build(*args)
    assert result.returncode == 2
    assert "teacher calls" not in result.stdout
