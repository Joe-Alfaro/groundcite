"""Tests for scripts/check_rubric.py (R13): one rubric_sha256 covers every committed run.

CLI: check_rubric.py RESULTS_DIR

A run is any directory under RESULTS_DIR (at any depth) holding a ``results.json`` or a
``transcripts.jsonl``. Its hashes are ``pins.rubric_sha256`` from results.json and the
``rubric_sha256`` of every transcript row.

- Exit 0 when exactly one hash covers every run; the hash is printed.
- Exit 1 when there is more than one; each hash is printed with the runs (paths relative to
  RESULTS_DIR) that carry it.
- Exit 0 with a "no runs" note when there are none.
- Exit 2 on a missing RESULTS_DIR, an unreadable file, invalid JSON or a missing hash field.

D-080 (the judge moved from gpt-5.5 to Claude Opus) adds a same-judge check: one results table
must not mix rows scored by different judges. Each run's ``pins.judge_model`` and every
transcript row's ``judge_model`` must be one judge; a run that mixes them exits 1, naming the run
and its judges. Different runs may use different judges (a re-judged run sits next to its
source), as long as they share the one rubric hash. A row without ``judge_model`` exits 2.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_rubric.py"

HASH_A = "a" * 64
HASH_B = "b" * 64


JUDGE_A = "mock-judge"
JUDGE_B = "anthropic:claude-opus-5-5"


def _results(rubric: str | None, run_id: str, judge: str = JUDGE_A) -> dict[str, Any]:
    pins: dict[str, Any] = {"git_sha": "0" * 40, "judge_model": judge}
    if rubric is not None:
        pins["rubric_sha256"] = rubric
    return {"run_id": run_id, "pins": pins, "scenario_ids": ["s1"], "rows": []}


def _row(rubric: str, run_id: str, scenario_id: str = "s1", judge: str = JUDGE_A) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "model": "m",
        "strategy": "zero_shot",
        "rubric_sha256": rubric,
        "judge_model": judge,
    }


def _make_run(
    results_dir: Path,
    rel: str,
    rubric: str,
    *,
    row_rubrics: tuple[str, ...] | None = None,
    with_results: bool = True,
    with_transcripts: bool = True,
    judge: str = JUDGE_A,
    row_judges: tuple[str, ...] | None = None,
) -> Path:
    run_dir = results_dir / rel
    run_dir.mkdir(parents=True)
    run_id = run_dir.name
    if with_results:
        results = _results(rubric, run_id, judge)
        (run_dir / "results.json").write_text(json.dumps(results), "utf-8")
    if with_transcripts:
        rubrics = row_rubrics if row_rubrics is not None else (rubric, rubric)
        judges = row_judges if row_judges is not None else (judge,) * len(rubrics)
        lines = [
            json.dumps(_row(r, run_id, f"s{i}", j))
            for i, (r, j) in enumerate(zip(rubrics, judges, strict=True))
        ]
        (run_dir / "transcripts.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    return run_dir


def _run(results_dir: Path) -> tuple[int, str]:
    assert SCRIPT.is_file(), f"{SCRIPT.name} does not exist yet"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(results_dir)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    return done.returncode, done.stdout + done.stderr


def test_one_hash_across_all_runs_exits_zero(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "20261027T000000Z-aaaaaa", HASH_A)
    _make_run(results, "ablation/20261027T010000Z-bbbbbb", HASH_A)
    code, out = _run(results)
    assert code == 0, out
    assert HASH_A in out


def test_two_hashes_exit_one_and_list_runs_per_hash(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "run-old", HASH_A)
    _make_run(results, "ablation/run-new", HASH_B)
    _make_run(results, "run-also-new", HASH_B)
    code, out = _run(results)
    assert code == 1, out
    assert HASH_A in out and HASH_B in out
    # Each run is listed after its own hash and before the next hash.
    expected = {"run-old": HASH_A, "ablation/run-new": HASH_B, "run-also-new": HASH_B}
    for run, rubric in expected.items():
        assert run in out, f"run {run} not listed:\n{out}"
        at = out.index(run)
        nearest = max((HASH_A, HASH_B), key=lambda h: out.rfind(h, 0, at))
        assert nearest == rubric, f"{run} is not listed under {rubric[:8]}:\n{out}"


def test_mismatch_inside_one_runs_transcripts_is_caught(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "run-mixed", HASH_A, row_rubrics=(HASH_A, HASH_B))
    code, out = _run(results)
    assert code == 1, out
    assert HASH_B in out and "run-mixed" in out


def test_transcripts_disagreeing_with_results_json_is_caught(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "run-split", HASH_A, row_rubrics=(HASH_B, HASH_B))
    code, out = _run(results)
    assert code == 1, out


def test_run_with_only_transcripts_counts(tmp_path: Path) -> None:
    """A run stopped before results.json was written still has scored rows."""
    results = tmp_path / "results"
    _make_run(results, "run-done", HASH_A)
    _make_run(results, "run-partial", HASH_B, with_results=False)
    code, out = _run(results)
    assert code == 1, out
    assert "run-partial" in out


def test_no_runs_exits_zero_with_note(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "README.md").write_text("nothing here yet\n", "utf-8")
    code, out = _run(results)
    assert code == 0, out
    assert "no runs" in out.lower()


def test_missing_results_dir_exits_two(tmp_path: Path) -> None:
    code, out = _run(tmp_path / "does-not-exist")
    assert code == 2, out
    assert "does-not-exist" in out


def test_malformed_results_json_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _make_run(results, "run-bad", HASH_A)
    (run / "results.json").write_text("{not json", "utf-8")
    code, out = _run(results)
    assert code == 2, out
    assert "run-bad" in out


def test_results_json_without_rubric_hash_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _make_run(results, "run-nohash", HASH_A)
    (run / "results.json").write_text(json.dumps(_results(None, "run-nohash")), "utf-8")
    code, out = _run(results)
    assert code == 2, out
    assert "run-nohash" in out and "rubric_sha256" in out


def test_malformed_transcript_line_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _make_run(results, "run-badline", HASH_A)
    with (run / "transcripts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{truncated\n")
    code, out = _run(results)
    assert code == 2, out
    assert "run-badline" in out


def test_transcript_row_without_rubric_hash_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _make_run(results, "run-rowhash", HASH_A)
    row = _row(HASH_A, "run-rowhash")
    del row["rubric_sha256"]
    with (run / "transcripts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    code, out = _run(results)
    assert code == 2, out
    assert "run-rowhash" in out


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
def test_unreadable_results_json_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _make_run(results, "run-locked", HASH_A)
    target = run / "results.json"
    target.chmod(0)
    try:
        code, out = _run(results)
    finally:
        target.chmod(0o644)
    assert code == 2, out
    assert "run-locked" in out


# ---------------------------------------------------------------------------------------------
# D-080: one judge per results table
# ---------------------------------------------------------------------------------------------

PINNED_RUBRIC = "636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf"


def test_the_committed_results_still_share_one_rubric_hash() -> None:
    """The judge switch leaves the rubric alone: every committed run keeps the pinned hash."""
    code, out = _run(REPO_ROOT / "results")
    assert code == 0, out
    assert PINNED_RUBRIC in out


def test_runs_with_different_judges_but_one_rubric_pass(tmp_path: Path) -> None:
    """A gpt-5.5-scored run and its Opus re-judge can sit side by side as separate runs."""
    results = tmp_path / "results"
    _make_run(results, "m1", HASH_A, judge=JUDGE_A)
    _make_run(results, "m1-rejudged", HASH_A, judge=JUDGE_B)
    code, out = _run(results)
    assert code == 0, out


def test_rows_from_two_judges_in_one_run_fail(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "run-ok", HASH_A)
    _make_run(results, "run-mixed-judges", HASH_A, row_judges=(JUDGE_A, JUDGE_B))
    code, out = _run(results)
    assert code == 1, out
    assert "run-mixed-judges" in out
    assert JUDGE_A in out and JUDGE_B in out
    assert "judge" in out.lower()


def test_rows_disagreeing_with_the_pinned_judge_fail(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "run-split-judge", HASH_A, judge=JUDGE_A, row_judges=(JUDGE_B, JUDGE_B))
    code, out = _run(results)
    assert code == 1, out
    assert "run-split-judge" in out and JUDGE_B in out


def test_mixed_judges_fail_even_when_the_rubric_matches_everywhere(tmp_path: Path) -> None:
    """The rubric check alone would pass; the judge check is what catches it."""
    results = tmp_path / "results"
    _make_run(
        results,
        "only-run",
        HASH_A,
        row_rubrics=(HASH_A, HASH_A, HASH_A),
        row_judges=(JUDGE_A, JUDGE_B, JUDGE_A),
    )
    code, out = _run(results)
    assert code == 1, out


def test_transcript_row_without_judge_model_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _make_run(results, "run-nojudge", HASH_A)
    row = _row(HASH_A, "run-nojudge")
    del row["judge_model"]
    with (run / "transcripts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    code, out = _run(results)
    assert code == 2, out
    assert "run-nojudge" in out and "judge_model" in out
