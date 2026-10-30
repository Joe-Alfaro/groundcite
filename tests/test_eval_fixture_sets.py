"""End to end: eval.py scores a third-party held-out set (R12, GOTCHA G18).

Each fixture set runs through ``eval.py`` (``groundcite.cli:main``) with two mock models and the
mock judge, and must produce scored output: ``results.json``, ``results.md`` and
``transcripts.jsonl`` with one row per scenario and system. The sets are (a) our format, (b)
prompt-only rows (CSV and JSON array) and (c) a hand-written set with an ID scheme the training
data never used. A malformed row stops the run before any output with exit 2, the row's location
and the field at fault.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groundcite.evalset import load_eval_set
from groundcite.report import validate_transcript_row

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
MODELS = ("mock:compliant", "mock:naive")
GOLD_CHECKS = ("cited_id_supports", "mode_matches_gold")

FIXTURE_SETS = [
    pytest.param("our_format.jsonl", id="a-our-format"),
    pytest.param("prompt_only.csv", id="b-prompt-only-csv"),
    pytest.param("prompt_only.json", id="b-prompt-only-json"),
    pytest.param("third_party.jsonl", id="c-unfamiliar-id-scheme"),
]


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


def _run_cli(argv: list[str]) -> int:
    from groundcite.cli import main

    try:
        return main(argv)
    except SystemExit as exc:  # argparse errors
        return int(exc.code or 0)


def _argv(eval_set: Path, out: Path) -> list[str]:
    models = [arg for m in MODELS for arg in ("--model", m)]
    return [
        *models,
        "--eval-set",
        str(eval_set),
        "--judge",
        "mock",
        "--no-base",
        "--allow-dirty",
        "--out",
        str(out),
    ]


def _scored_run(eval_set: Path, out: Path) -> Path:
    assert _run_cli(_argv(eval_set, out)) == 0
    dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _transcripts(run: Path) -> list[dict[str, Any]]:
    lines = (run / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.parametrize("fixture", FIXTURE_SETS)
def test_fixture_set_produces_scored_output(env: Path, tmp_path: Path, fixture: str) -> None:
    eval_set = FIXTURES / fixture
    ids = [s.id for s in load_eval_set(eval_set)]
    run = _scored_run(eval_set, tmp_path / "results")
    for name in ("results.json", "results.md", "transcripts.jsonl"):
        assert (run / name).is_file(), name

    rows = _transcripts(run)
    assert len(rows) == len(ids) * len(MODELS)
    for model in MODELS:
        assert sorted(r["scenario_id"] for r in rows if r["model"] == model) == sorted(ids)
    for row in rows:
        validate_transcript_row(row)
        assert row["score"]["spec_adherence"] in (0, 1)
        assert row["reasoning"]
        assert row["judge_model"] == "mock:judge"

    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["scenario_ids"] == ids
    assert [r["model"] for r in results["rows"]] == list(MODELS)
    assert all(r["n"] == len(ids) for r in results["rows"])


@pytest.mark.parametrize("fixture", FIXTURE_SETS)
def test_rows_without_gold_report_the_gold_checks_as_na(
    env: Path, tmp_path: Path, fixture: str
) -> None:
    eval_set = FIXTURES / fixture
    gold = {s.id: s.gold for s in load_eval_set(eval_set)}
    rows = _transcripts(_scored_run(eval_set, tmp_path / "results"))
    for row in rows:
        if gold[row["scenario_id"]] is None:
            for check in GOLD_CHECKS:
                assert row["checks"][check] is None, (row["scenario_id"], check)


def test_unfamiliar_id_scheme_is_cited_as_provided(env: Path, tmp_path: Path) -> None:
    eval_set = FIXTURES / "third_party.jsonl"
    provided = {s.id: s.provided_ids for s in load_eval_set(eval_set)}
    rows = _transcripts(_scored_run(eval_set, tmp_path / "results"))
    compliant = {r["scenario_id"]: r for r in rows if r["model"] == "mock:compliant"}
    assert compliant["tp-7"]["checks"]["cited_id_provided"] is True
    assert compliant["tp-7"]["checks"]["cited_id_supports"] is True
    assert f"[{provided['tp-7'][0]}]" in compliant["tp-7"]["reply"]


# ---------------------------------------------------------------------------------------------
# Malformed rows fail fast: exit 2, the location and the field, no run written
# ---------------------------------------------------------------------------------------------

_OK = {"id": "ok-1", "prompt": "References: [A-1] Hose check. Complaint: rough idle."}

MALFORMED = [
    pytest.param(
        "bad.jsonl",
        json.dumps(_OK) + "\n" + json.dumps({"id": "bad-2"}) + "\n",
        "line 2",
        "messages",
        id="jsonl-no-messages-or-prompt",
    ),
    pytest.param(
        "bad.jsonl",
        json.dumps(_OK)
        + "\n\n"
        + json.dumps({"id": "bad-3", "prompt": "x [A-1]", "snippets": [{"id": "A-1"}]})
        + "\n",
        "line 3",
        "snippets[0].text",
        id="jsonl-snippet-without-text",
    ),
    pytest.param(
        "bad.jsonl",
        json.dumps(_OK)
        + "\n"
        + json.dumps({"id": "bad-2", "prompt": "x [A-1]", "gold": {"mode": "maybe"}})
        + "\n",
        "line 2",
        "gold",
        id="jsonl-bad-gold-mode",
    ),
    pytest.param(
        "bad.csv",
        'id,prompt\nok-1,"[A-1] Hose check. Rough idle."\n,"no id here [A-1]"\n',
        "line 3",
        "id",
        id="csv-empty-id",
    ),
    pytest.param(
        "bad.json",
        json.dumps([_OK, {"id": "bad-2", "messages": [{"role": "assistant", "content": "hi"}]}]),
        "item 2",
        "messages",
        id="json-array-ends-on-assistant",
    ),
]


@pytest.mark.parametrize(("name", "text", "where", "field"), MALFORMED)
def test_malformed_row_exits_2_with_the_location_and_field(
    env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    text: str,
    where: str,
    field: str,
) -> None:
    bad = tmp_path / name
    bad.write_text(text, encoding="utf-8")
    out = tmp_path / "results"
    assert _run_cli(_argv(bad, out)) == 2
    err = capsys.readouterr().err
    assert where in err, err
    assert field in err, err
    assert not out.exists() or not any(out.iterdir()), "a malformed set must not start a run"
