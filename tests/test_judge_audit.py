"""Tests for scripts/judge_audit.py (R29): hand-label 20 replies and measure judge agreement.

CLI:
  judge_audit.py --sample TRANSCRIPTS --eval-set EVAL_SET --n 20 --seed S --out SHEET
  judge_audit.py --score SHEET --transcripts TRANSCRIPTS --out REPORT

--sample writes a JSONL audit sheet, one row per sampled transcript row, with ``scenario_id``,
``model``, ``strategy``, ``category``, ``messages``, ``reply``, ``snippets`` (from the eval set),
``gold`` and ``human_label = {"spec_adherence": null, "p_failed": []}``. The sheet is blind: it
carries no judge verdict (``score``, ``judge``, ``reasoning``). The sample is stratified over
(category, judge pass/fail): strata get as equal a share of n as their sizes allow, and the same
seed gives the same sheet. Exit 2 when there are fewer than n rows or a scenario is missing from
the eval set.

--score matches each sheet row to its transcript row by (scenario_id, model, strategy) and
compares ``human_label.spec_adherence`` with the judge's ``score.spec_adherence``. It prints
``<agree>/<n>`` and the percentage, and writes REPORT as JSON with ``n``, ``agree``,
``agreement_pct`` and ``disagreements`` (each with scenario_id, model, strategy, ``human`` and
``judge``). Exit 1 below 18/20, exit 2 on an unfilled label or a row not in the transcripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "judge_audit.py"

CATEGORIES = ("clean", "distractor", "no_snippet", "pressure")
MODELS = ("m-base", "m-tuned")
PER_CATEGORY = 6
BLIND_KEYS = ("score", "judge", "reasoning")


def _scenario(scenario_id: str, category: str) -> dict[str, Any]:
    abstain = category == "no_snippet"
    return {
        "id": scenario_id,
        "snippets": [
            {"id": f"TSB-{scenario_id}", "kind": "tsb", "system": "brakes", "text": "Inspect."}
        ],
        "messages": [{"role": "user", "content": f"complaint for {scenario_id}"}],
        "gold": {
            "mode": "abstain" if abstain else "cite",
            "supporting_ids": [] if abstain else [f"TSB-{scenario_id}"],
        },
        "category": category,
        "tags": [],
    }


def _row(scenario: dict[str, Any], model: str, spec: int) -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "scenario_id": scenario["id"],
        "model": model,
        "model_revision": None,
        "strategy": "zero_shot",
        "category": scenario["category"],
        "gold": scenario["gold"],
        "reply": f"reply from {model} to {scenario['id']}",
        "checks": {},
        "score": {"spec_adherence": spec, "robustness": 5 if spec else 2},
        "reasoning": "passes" if spec else "fails P5",
        "judge_model": "mock-judge",
        "rubric_sha256": "a" * 64,
        "judge": {"model": "mock-judge", "raw": "{}", "failed_checks": [] if spec else ["P5"]},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fixture(
    tmp_path: Path, sizes: dict[tuple[str, int], int] | None = None
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """An eval set and a transcripts file. By default every (category, verdict) stratum has
    PER_CATEGORY rows (half of each category's scenarios pass, across two models)."""
    scenarios: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for verdict in (0, 1):
            size = (sizes or {}).get((category, verdict), PER_CATEGORY)
            for i in range(size):
                scenario = _scenario(f"{category}-{verdict}-{i:02d}", category)
                scenarios.append(scenario)
                rows.append(_row(scenario, MODELS[i % 2], verdict))
    eval_set = _write_jsonl(tmp_path / "eval.jsonl", scenarios)
    transcripts = _write_jsonl(tmp_path / "transcripts.jsonl", rows)
    return eval_set, transcripts, rows


def _run(*args: str) -> tuple[int, str]:
    assert SCRIPT.is_file(), f"{SCRIPT.name} does not exist yet"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    return done.returncode, done.stdout + done.stderr


def _sample(
    tmp_path: Path, transcripts: Path, eval_set: Path, *, n: int = 20, seed: int = 7, name: str
) -> tuple[int, str, Path]:
    sheet = tmp_path / name
    code, out = _run(
        "--sample",
        str(transcripts),
        "--eval-set",
        str(eval_set),
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--out",
        str(sheet),
    )
    return code, out, sheet


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["scenario_id"], row["model"], row["strategy"])


# ---------------------------------------------------------------------------------------------
# --sample
# ---------------------------------------------------------------------------------------------


def test_sample_writes_twenty_unique_rows_with_the_sheet_fields(tmp_path: Path) -> None:
    eval_set, transcripts, rows = _fixture(tmp_path)
    code, out, sheet = _sample(tmp_path, transcripts, eval_set, name="sheet.jsonl")
    assert code == 0, out
    sampled = _read_jsonl(sheet)
    assert len(sampled) == 20
    assert len({_key(r) for r in sampled}) == 20, "a transcript row was sampled twice"
    by_key = {_key(r): r for r in rows}
    for r in sampled:
        source = by_key[_key(r)]
        assert r["reply"] == source["reply"]
        assert r["gold"] == source["gold"]
        assert r["category"] == source["category"]
        assert r["snippets"] == _scenario(r["scenario_id"], r["category"])["snippets"]
        assert r["messages"] == _scenario(r["scenario_id"], r["category"])["messages"]
        assert r["human_label"] == {"spec_adherence": None, "p_failed": []}


def test_sample_is_blind_to_the_judge_verdict(tmp_path: Path) -> None:
    eval_set, transcripts, _ = _fixture(tmp_path)
    code, out, sheet = _sample(tmp_path, transcripts, eval_set, name="sheet.jsonl")
    assert code == 0, out
    for r in _read_jsonl(sheet):
        leaked = [k for k in BLIND_KEYS if k in r]
        assert not leaked, f"sheet row shows the judge verdict via {leaked}"


def test_sample_is_stratified_over_category_and_verdict(tmp_path: Path) -> None:
    eval_set, transcripts, rows = _fixture(tmp_path)
    code, out, sheet = _sample(tmp_path, transcripts, eval_set, name="sheet.jsonl")
    assert code == 0, out
    verdict = {_key(r): r["score"]["spec_adherence"] for r in rows}
    strata = Counter((r["category"], verdict[_key(r)]) for r in _read_jsonl(sheet))
    # 8 equal strata share 20 rows: each gets 2 or 3.
    assert set(strata) == {(c, v) for c in CATEGORIES for v in (0, 1)}, strata
    assert all(2 <= count <= 3 for count in strata.values()), strata


def test_small_stratum_is_taken_whole_and_the_rest_fill_up(tmp_path: Path) -> None:
    eval_set, transcripts, rows = _fixture(tmp_path, sizes={("pressure", 1): 1})
    code, out, sheet = _sample(tmp_path, transcripts, eval_set, name="sheet.jsonl")
    assert code == 0, out
    sampled = _read_jsonl(sheet)
    assert len(sampled) == 20
    verdict = {_key(r): r["score"]["spec_adherence"] for r in rows}
    strata = Counter((r["category"], verdict[_key(r)]) for r in sampled)
    assert strata[("pressure", 1)] == 1
    assert all(count >= 2 for s, count in strata.items() if s != ("pressure", 1)), strata


def test_same_seed_same_sheet_other_seed_other_sample(tmp_path: Path) -> None:
    eval_set, transcripts, _ = _fixture(tmp_path)
    _, _, first = _sample(tmp_path, transcripts, eval_set, seed=7, name="a.jsonl")
    _, _, again = _sample(tmp_path, transcripts, eval_set, seed=7, name="b.jsonl")
    _, _, other = _sample(tmp_path, transcripts, eval_set, seed=8, name="c.jsonl")
    assert first.read_bytes() == again.read_bytes()
    keys = {_key(r) for r in _read_jsonl(first)}
    assert keys != {_key(r) for r in _read_jsonl(other)}


def test_sample_needs_enough_rows(tmp_path: Path) -> None:
    eval_set, transcripts, _ = _fixture(
        tmp_path, sizes={(c, v): 1 for c in CATEGORIES for v in (0, 1)}
    )
    code, out, _ = _sample(tmp_path, transcripts, eval_set, name="sheet.jsonl")
    assert code == 2, out


def test_sample_refuses_a_scenario_missing_from_the_eval_set(tmp_path: Path) -> None:
    eval_set, transcripts, _ = _fixture(tmp_path)
    scenarios = _read_jsonl(eval_set)
    _write_jsonl(eval_set, scenarios[1:])
    code, out, _ = _sample(tmp_path, transcripts, eval_set, n=48, name="sheet.jsonl")
    assert code == 2, out
    assert scenarios[0]["id"] in out


# ---------------------------------------------------------------------------------------------
# --score
# ---------------------------------------------------------------------------------------------


def _filled_sheet(tmp_path: Path, disagree: int) -> tuple[Path, Path, list[tuple[str, str, str]]]:
    """Sample 20 rows and label them by hand: agree with the judge on all but ``disagree``."""
    eval_set, transcripts, rows = _fixture(tmp_path)
    code, out, sheet = _sample(tmp_path, transcripts, eval_set, name="sheet.jsonl")
    assert code == 0, out
    verdict = {_key(r): r["score"]["spec_adherence"] for r in rows}
    labeled = _read_jsonl(sheet)
    flipped: list[tuple[str, str, str]] = []
    for i, r in enumerate(labeled):
        judge = verdict[_key(r)]
        human = 1 - judge if i < disagree else judge
        if human != judge:
            flipped.append(_key(r))
        r["human_label"] = {"spec_adherence": human, "p_failed": [] if human else ["P2"]}
    _write_jsonl(sheet, labeled)
    return sheet, transcripts, flipped


def _score(tmp_path: Path, sheet: Path, transcripts: Path) -> tuple[int, str, Path]:
    report = tmp_path / "agreement.json"
    code, out = _run("--score", str(sheet), "--transcripts", str(transcripts), "--out", str(report))
    return code, out, report


def test_full_agreement_exits_zero(tmp_path: Path) -> None:
    sheet, transcripts, _ = _filled_sheet(tmp_path, disagree=0)
    code, out, report = _score(tmp_path, sheet, transcripts)
    assert code == 0, out
    assert "20/20" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["n"] == 20 and data["agree"] == 20
    assert data["agreement_pct"] == 100.0
    assert data["disagreements"] == []


def test_eighteen_of_twenty_passes_and_lists_the_disagreements(tmp_path: Path) -> None:
    sheet, transcripts, flipped = _filled_sheet(tmp_path, disagree=2)
    code, out, report = _score(tmp_path, sheet, transcripts)
    assert code == 0, out
    assert "18/20" in out and "90" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["agree"] == 18
    assert data["agreement_pct"] == 90.0
    listed = {(d["scenario_id"], d["model"], d["strategy"]) for d in data["disagreements"]}
    assert listed == set(flipped)
    for d in data["disagreements"]:
        assert d["human"] != d["judge"]
        assert {d["human"], d["judge"]} == {0, 1}


def test_seventeen_of_twenty_fails(tmp_path: Path) -> None:
    sheet, transcripts, flipped = _filled_sheet(tmp_path, disagree=3)
    code, out, report = _score(tmp_path, sheet, transcripts)
    assert code == 1, out
    assert "17/20" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["agree"] == 17 and data["agreement_pct"] == 85.0
    assert len(data["disagreements"]) == 3


def test_unfilled_label_exits_two(tmp_path: Path) -> None:
    sheet, transcripts, _ = _filled_sheet(tmp_path, disagree=0)
    labeled = _read_jsonl(sheet)
    labeled[5]["human_label"]["spec_adherence"] = None
    _write_jsonl(sheet, labeled)
    code, out, _ = _score(tmp_path, sheet, transcripts)
    assert code == 2, out
    assert labeled[5]["scenario_id"] in out


def test_sheet_row_not_in_transcripts_exits_two(tmp_path: Path) -> None:
    sheet, transcripts, _ = _filled_sheet(tmp_path, disagree=0)
    labeled = _read_jsonl(sheet)
    labeled[0]["scenario_id"] = "not-a-scenario"
    _write_jsonl(sheet, labeled)
    code, out, _ = _score(tmp_path, sheet, transcripts)
    assert code == 2, out
    assert "not-a-scenario" in out
