"""R29 on the committed held-out base-vs-tuned run: sample a blind audit sheet, score a filled one.

``tests/test_judge_audit.py`` covers the general ``--sample`` / ``--score`` contract on a synthetic
run. This file adds what a base-vs-tuned audit needs:

- ``--sample`` works on the committed run ``results/base_vs_tuned/heldout/<run>/`` with the eval
  set its ``generation_pins.json`` names.
- ``--blind-system`` hides which system wrote each reply, so the hand labels can't favor the
  tuned model. Each sheet row gets an opaque, unique ``audit_id`` and carries no ``model``,
  ``model_revision``, ``strategy`` or ``run_id`` (and, as always, no judge verdict). The mapping
  from ``audit_id`` to (scenario_id, model, strategy) goes to a key file next to the sheet,
  ``SHEET.with_suffix(".key.json")``, which the labeler doesn't open.
- With ``--blind-system`` the sample is split across systems (model, strategy) first, as evenly
  as their sizes allow (counts differ by at most one), then stratified over (category, judge
  verdict) within each system as before. The same seed gives the same sheet and key.
- ``--score`` on a blind sheet reads the key file to match each row to its transcript row, and
  reports agreement and disagreements (with model and strategy) exactly as for an open sheet. A
  missing key file is exit 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "judge_audit.py"
HELDOUT = REPO_ROOT / "results" / "base_vs_tuned" / "heldout"
SYSTEM_KEYS = ("model", "model_revision", "strategy", "run_id")
VERDICT_KEYS = ("score", "judge", "reasoning", "checks")
SHEET_KEYS = {
    "audit_id",
    "scenario_id",
    "category",
    "messages",
    "reply",
    "snippets",
    "gold",
    "human_label",
}


def _run_dir() -> Path:
    runs = sorted(p.parent for p in HELDOUT.glob("*/transcripts.jsonl"))
    assert len(runs) == 1, f"expected one committed held-out run, found {runs}"
    return runs[0]


RUN = _run_dir()
TRANSCRIPTS = RUN / "transcripts.jsonl"


def _eval_set() -> Path:
    pins = json.loads((RUN / "generation_pins.json").read_text(encoding="utf-8"))
    return REPO_ROOT / pins["eval_set"]["path"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


ROWS = _read_jsonl(TRANSCRIPTS)
SYSTEMS = sorted({(r["model"], r["strategy"]) for r in ROWS})


def _run(*args: str) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    return done.returncode, done.stdout + done.stderr


def _sample(tmp_path: Path, *extra: str, seed: int = 7, name: str = "sheet.jsonl") -> Path:
    sheet = tmp_path / name
    code, out = _run(
        "--sample",
        str(TRANSCRIPTS),
        "--eval-set",
        str(_eval_set()),
        "--n",
        "20",
        "--seed",
        str(seed),
        "--out",
        str(sheet),
        *extra,
    )
    assert code == 0, out
    return sheet


def _source_row(entry: dict[str, Any]) -> dict[str, Any]:
    """The transcript row a blind sheet row came from, found by its scenario and reply text (each
    (scenario_id, reply) pair is unique in the committed run)."""
    matches = [
        r for r in ROWS if r["scenario_id"] == entry["scenario_id"] and r["reply"] == entry["reply"]
    ]
    assert len(matches) == 1, entry["scenario_id"]
    return matches[0]


def test_the_committed_run_has_base_and_tuned_systems() -> None:
    assert len(SYSTEMS) >= 2, SYSTEMS
    assert _eval_set().is_file()
    pairs = Counter((r["scenario_id"], r["reply"]) for r in ROWS)
    assert max(pairs.values()) == 1, "the blind tests match rows by (scenario_id, reply)"


# ---------------------------------------------------------------------------------------------
# Open sheet on the committed run
# ---------------------------------------------------------------------------------------------


def test_sample_draws_twenty_rows_from_the_committed_heldout_run(tmp_path: Path) -> None:
    sheet = _read_jsonl(_sample(tmp_path))
    assert len(sheet) == 20
    keys = {(r["scenario_id"], r["model"], r["strategy"]) for r in ROWS}
    assert {(r["scenario_id"], r["model"], r["strategy"]) for r in sheet} <= keys
    for entry in sheet:
        assert not [k for k in VERDICT_KEYS if k in entry], entry.keys()


# ---------------------------------------------------------------------------------------------
# --blind-system
# ---------------------------------------------------------------------------------------------


def test_blind_sheet_hides_the_system_and_the_verdict(tmp_path: Path) -> None:
    sheet = _read_jsonl(_sample(tmp_path, "--blind-system"))
    assert len(sheet) == 20
    models = {r["model"] for r in ROWS} | {r["model_revision"] for r in ROWS if r["model_revision"]}
    for entry in sheet:
        assert set(entry) == SHEET_KEYS, sorted(set(entry) ^ SHEET_KEYS)
        assert not [k for k in (*SYSTEM_KEYS, *VERDICT_KEYS) if k in entry]
        text = json.dumps(entry)
        leaked = [m for m in models if m in text]
        assert not leaked, f"sheet row names the system: {leaked}"
        assert entry["human_label"] == {"spec_adherence": None, "p_failed": []}
        source = _source_row(entry)
        assert entry["category"] == source["category"]
        assert entry["gold"] == source["gold"]


def test_blind_sheet_audit_ids_are_unique_and_opaque(tmp_path: Path) -> None:
    sheet = _read_jsonl(_sample(tmp_path, "--blind-system"))
    ids = [entry["audit_id"] for entry in sheet]
    assert len(set(ids)) == 20
    for audit_id in ids:
        assert isinstance(audit_id, str) and audit_id
        assert not any(model in audit_id for model, _ in SYSTEMS)
        assert not any(strategy in audit_id for _, strategy in SYSTEMS)


def test_blind_sample_writes_a_key_file_next_to_the_sheet(tmp_path: Path) -> None:
    sheet_path = _sample(tmp_path, "--blind-system")
    key_path = sheet_path.with_suffix(".key.json")
    assert key_path.is_file(), "the blind sample needs a key file to unblind the scores"
    key_text = key_path.read_text(encoding="utf-8")
    for entry in _read_jsonl(sheet_path):
        assert entry["audit_id"] in key_text
        source = _source_row(entry)
        assert source["model"] in key_text


def test_blind_sample_is_balanced_across_systems(tmp_path: Path) -> None:
    sheet = _read_jsonl(_sample(tmp_path, "--blind-system"))
    per_system = Counter((r["model"], r["strategy"]) for r in map(_source_row, sheet))
    assert set(per_system) == set(SYSTEMS), per_system
    assert max(per_system.values()) - min(per_system.values()) <= 1, per_system


def test_blind_sample_is_stratified_within_each_system(tmp_path: Path) -> None:
    sheet = _read_jsonl(_sample(tmp_path, "--blind-system"))
    sources = [_source_row(entry) for entry in sheet]
    strata = Counter(
        (r["model"], r["strategy"], r["category"], r["score"]["spec_adherence"]) for r in sources
    )
    # Every system has more (category, verdict) strata than its 6-7 rows, so none repeats.
    assert max(strata.values()) == 1, strata
    for system in SYSTEMS:
        categories = {r["category"] for r in sources if (r["model"], r["strategy"]) == system}
        assert len(categories) >= 4, (system, categories)


def test_blind_sample_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    first = _sample(tmp_path, "--blind-system", seed=11, name="a.jsonl")
    again = _sample(tmp_path, "--blind-system", seed=11, name="b.jsonl")
    other = _sample(tmp_path, "--blind-system", seed=12, name="c.jsonl")
    assert first.read_bytes() == again.read_bytes()
    first_key = first.with_suffix(".key.json").read_text(encoding="utf-8")
    again_key = again.with_suffix(".key.json").read_text(encoding="utf-8")
    assert json.loads(first_key) == json.loads(again_key)
    picked = {(r["scenario_id"], r["reply"]) for r in _read_jsonl(first)}
    assert picked != {(r["scenario_id"], r["reply"]) for r in _read_jsonl(other)}


# ---------------------------------------------------------------------------------------------
# --score a filled blind sheet
# ---------------------------------------------------------------------------------------------


def _fill(sheet_path: Path, disagree: int) -> list[tuple[str, str, str]]:
    """Label every row with the judge's verdict, flipping the first ``disagree`` rows."""
    sheet = _read_jsonl(sheet_path)
    flipped = []
    for i, entry in enumerate(sheet):
        source = _source_row(entry)
        verdict = source["score"]["spec_adherence"]
        if i < disagree:
            verdict = 1 - verdict
            flipped.append((source["scenario_id"], source["model"], source["strategy"]))
        entry["human_label"] = {"spec_adherence": verdict, "p_failed": []}
    _write_jsonl(sheet_path, sheet)
    return flipped


def _score(tmp_path: Path, sheet: Path) -> tuple[int, str, Path]:
    report = tmp_path / "agreement.json"
    code, out = _run("--score", str(sheet), "--transcripts", str(TRANSCRIPTS), "--out", str(report))
    return code, out, report


def test_score_unblinds_a_fully_agreeing_sheet(tmp_path: Path) -> None:
    sheet = _sample(tmp_path, "--blind-system")
    _fill(sheet, disagree=0)
    code, out, report = _score(tmp_path, sheet)
    assert code == 0, out
    assert "20/20" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["n"] == 20 and data["agree"] == 20
    assert data["disagreements"] == []


def test_score_names_the_systems_behind_blind_disagreements(tmp_path: Path) -> None:
    sheet = _sample(tmp_path, "--blind-system")
    flipped = _fill(sheet, disagree=3)
    code, out, report = _score(tmp_path, sheet)
    assert code == 1, out
    assert "17/20" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["agree"] == 17 and data["agreement_pct"] == 85.0
    listed = {(d["scenario_id"], d["model"], d["strategy"]) for d in data["disagreements"]}
    assert listed == set(flipped)


def test_score_without_the_key_file_exits_2(tmp_path: Path) -> None:
    sheet = _sample(tmp_path, "--blind-system")
    _fill(sheet, disagree=0)
    sheet.with_suffix(".key.json").unlink()
    code, out, _ = _score(tmp_path, sheet)
    assert code == 2, out
    assert "key" in out.lower()


@pytest.mark.parametrize("field", ["scenario_id", "audit_id"])
def test_score_refuses_a_blind_row_edited_away_from_its_key(tmp_path: Path, field: str) -> None:
    """The key records each audit_id's scenario; a row whose scenario no longer matches is bad
    input, not a silent mismatch."""
    sheet_path = _sample(tmp_path, "--blind-system")
    _fill(sheet_path, disagree=0)
    sheet = _read_jsonl(sheet_path)
    sheet[0][field] = f"not-a-{field}"
    _write_jsonl(sheet_path, sheet)
    code, out, _ = _score(tmp_path, sheet_path)
    assert code == 2, out
