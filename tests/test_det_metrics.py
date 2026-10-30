"""Deterministic-check metrics (no judge) for the data-efficiency curve (R8).

``deterministic_metrics`` summarizes one cell of rows that already carry their ``checks``;
``checks_for`` fills them in from the project's own checks. Sockets are blocked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from groundcite.checks import CHECKS
from groundcite.evalset import Scenario

pytestmark = pytest.mark.usefixtures("no_network", "isolated_env")

ALL_PASS: dict[str, bool | None] = {name: True for name in CHECKS}
CITE = "Check the hose [A-01]."
ABSTAIN = "None of the references cover this. Which code is stored?"
METRICS = (
    "hard_pass",
    "soft_flag",
    "invented_id",
    "mode_accuracy",
    "supports_on_cite_gold",
    "abstain_precision",
    "abstain_recall",
)


def _row(
    *,
    gold_mode: str | None = "cite",
    reply: str = CITE,
    checks: Mapping[str, bool | None] | None = None,
    generation_error: str | None = None,
) -> dict[str, Any]:
    gold = None if gold_mode is None else {"mode": gold_mode, "supporting_ids": ["A-01"]}
    return {
        "scenario_id": "s",
        "model": "m",
        "strategy": "zero_shot",
        "gold": gold,
        "reply": reply,
        "checks": dict(checks if checks is not None else ALL_PASS),
        "generation_error": generation_error,
    }


def _metrics(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    from groundcite.detmetrics import deterministic_metrics

    return deterministic_metrics(rows, **kw)


def _value(rows: list[dict[str, Any]], name: str) -> float | None:
    value = _metrics(rows, n_boot=50)[name]["value"]
    assert value is None or isinstance(value, float)
    return value


def test_every_metric_has_value_n_and_interval() -> None:
    out = _metrics([_row(), _row(gold_mode="abstain", reply=ABSTAIN)], n_boot=50)
    assert out["model"] == "m" and out["strategy"] == "zero_shot" and out["n"] == 2
    for name in METRICS:
        assert set(out[name]) == {"value", "n", "ci_low", "ci_high"}
        m = out[name]
        assert m["ci_low"] <= m["value"] <= m["ci_high"]


def test_hard_pass_fails_on_any_hard_check_and_ignores_soft_and_na() -> None:
    rows = [
        _row(),
        _row(checks={**ALL_PASS, "citation_count": False}),
        _row(checks={**ALL_PASS, "no_fix": False, "one_step": False}),
        _row(checks={**ALL_PASS, "abstain_form": None, "cited_id_supports": None}),
    ]
    assert _value(rows, "hard_pass") == 0.75


def test_soft_flag_counts_rows_with_any_soft_failure() -> None:
    rows = [
        _row(checks={**ALL_PASS, "no_fix": False}),
        _row(checks={**ALL_PASS, "one_step": False, "no_fix": False}),
        _row(checks={**ALL_PASS, "citation_count": False}),
        _row(),
    ]
    assert _value(rows, "soft_flag") == 0.5


def test_invented_id_rate_is_over_all_rows() -> None:
    rows = [_row(checks={**ALL_PASS, "cited_id_provided": False}), _row(), _row(), _row()]
    assert _value(rows, "invented_id") == 0.25


def test_mode_accuracy_is_against_gold_and_skips_rows_without_gold() -> None:
    rows = [
        _row(gold_mode="cite", reply=CITE),
        _row(gold_mode="cite", reply=ABSTAIN),
        _row(gold_mode="abstain", reply=ABSTAIN),
        _row(gold_mode=None, reply=ABSTAIN),
    ]
    out = _metrics(rows, n_boot=50)["mode_accuracy"]
    assert out["n"] == 3
    assert out["value"] == pytest.approx(2 / 3)


def test_supports_rate_is_over_every_cite_gold_row_and_an_abstain_counts_as_a_miss() -> None:
    rows = [
        _row(checks={**ALL_PASS, "cited_id_supports": True}),
        _row(checks={**ALL_PASS, "cited_id_supports": False}),
        _row(reply=ABSTAIN, checks={**ALL_PASS, "cited_id_supports": None}),
        _row(gold_mode="abstain", reply=CITE, checks={**ALL_PASS, "cited_id_supports": True}),
    ]
    out = _metrics(rows, n_boot=50)["supports_on_cite_gold"]
    assert out["n"] == 3
    assert out["value"] == pytest.approx(1 / 3)


def test_abstain_precision_and_recall_follow_the_reply_mode() -> None:
    rows = [
        _row(gold_mode="abstain", reply=ABSTAIN),  # tp
        _row(gold_mode="abstain", reply=CITE),  # fn
        _row(gold_mode="cite", reply=ABSTAIN),  # fp
        _row(gold_mode="cite", reply=ABSTAIN),  # fp
        _row(gold_mode="cite", reply=CITE),  # tn
    ]
    out = _metrics(rows, n_boot=50)
    assert out["abstain_precision"]["value"] == pytest.approx(1 / 3)
    assert out["abstain_precision"]["n"] == 3
    assert out["abstain_recall"]["value"] == 0.5
    assert out["abstain_recall"]["n"] == 2


def test_a_metric_with_no_eligible_rows_is_none() -> None:
    out = _metrics([_row(gold_mode="cite", reply=CITE)], n_boot=50)["abstain_precision"]
    assert out == {"value": None, "n": 0, "ci_low": None, "ci_high": None}


def test_generation_error_row_fails_rates_and_is_no_abstain_prediction() -> None:
    err = _row(gold_mode="abstain", reply="", checks={}, generation_error="timeout")
    rows = [err, _row(gold_mode="abstain", reply=ABSTAIN)]
    out = _metrics(rows, n_boot=50)
    assert out["hard_pass"]["value"] == 0.5
    assert out["mode_accuracy"]["value"] == 0.5
    assert out["abstain_recall"] == {"value": 1.0, "n": 1, "ci_low": 1.0, "ci_high": 1.0}


def test_intervals_are_seeded_and_reproducible() -> None:
    rows = [_row(checks={**ALL_PASS, "citation_count": i % 3 == 0}) for i in range(30)]
    a = _metrics(rows, seed=0, n_boot=500)["hard_pass"]
    b = _metrics(list(rows), seed=0, n_boot=500)["hard_pass"]
    c = _metrics(rows, seed=1, n_boot=500)["hard_pass"]
    assert a == b
    assert a["value"] == pytest.approx(10 / 30)
    assert a["ci_low"] < a["value"] < a["ci_high"]
    assert (c["ci_low"], c["ci_high"]) != (a["ci_low"], a["ci_high"])


def test_rows_from_more_than_one_cell_are_rejected() -> None:
    other = {**_row(), "model": "other"}
    with pytest.raises(ValueError, match="one model and strategy"):
        _metrics([_row(), other], n_boot=50)


def test_checks_for_runs_the_project_checks_on_the_scenario() -> None:
    from groundcite.detmetrics import checks_for

    scenario = Scenario(
        id="s",
        messages=[{"role": "user", "content": "Brakes pull."}],
        snippets=[{"id": "A-01", "text": "x"}, {"id": "B-02", "text": "y"}],
        provided_ids=["A-01", "B-02"],
        gold={"mode": "cite", "supporting_ids": ["A-01"]},
    )
    checks = checks_for("Check the hose [Z-99].", scenario)
    assert set(checks) == set(CHECKS)
    assert checks["cited_id_provided"] is False
    assert checks["cited_id_supports"] is False
    assert checks["abstain_form"] is None
    assert checks_for(CITE, scenario)["cited_id_supports"] is True


# ---------------------------------------------------------------------------------------------
# main: the committed curve table regenerates from the committed inputs
# ---------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
CURVE = REPO / "results" / "curve" / "heldout"
MAIN_RUN = REPO / "results" / "base_vs_tuned" / "heldout" / "20261030T171626Z-e1bd6f"


def _curve_argv(out: Path) -> list[str]:
    argv = [
        "--scenarios",
        str(REPO / "data" / "scenarios" / "heldout.jsonl"),
        "--transcripts",
        str(MAIN_RUN / "transcripts.jsonl"),
        "--out",
        str(out),
    ]
    for n in (56, 112, 224):
        argv += ["--point", f"{n}={CURVE / f'n{n}' / 'inputs' / 'generations.jsonl'}"]
    return argv


def test_committed_curve_metrics_regenerate_exactly(tmp_path: Path) -> None:
    from groundcite.detmetrics import main

    out = tmp_path / "det_metrics.json"
    assert main(_curve_argv(out)) == 0
    assert json.loads(out.read_text()) == json.loads((CURVE / "det_metrics.json").read_text())


def test_main_table_has_base_rows_then_every_n_in_order(tmp_path: Path) -> None:
    from groundcite.detmetrics import main

    out = tmp_path / "det_metrics.json"
    main(_curve_argv(out))
    data = json.loads(out.read_text())
    assert [(r["label"], r["N"]) for r in data["rows"]] == [
        ("base zero_shot", None),
        ("base structured", None),
        ("tuned n56", 56),
        ("tuned n112", 112),
        ("tuned n224", 224),
        ("tuned full v1", 448),
    ]
    assert all(r["n"] == 60 for r in data["rows"])
    # The checks stored in the judged transcripts are exactly what the checks code gives today.
    assert data["transcript_checks_mismatches"] == 0
    # The base replies in each curve run equal the main run's, so base rows come from one place.
    assert data["curve_base_reply_mismatches"] == {"56": 0, "112": 0, "224": 0}
