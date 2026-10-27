"""aggregate_scores() (the single shared aggregation; Robustness definition, R3, R30) and the
transcript row schema (R10, R11, R13). Sockets are blocked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from groundcite.checks import CHECKS, HARD_CHECKS
from groundcite.evalset import Scenario, load_eval_set

pytestmark = pytest.mark.usefixtures("no_network", "isolated_env")

REPO = Path(__file__).resolve().parents[1]
ALL_PASS = {name: True for name in CHECKS}


def _row(
    *,
    model: str = "m",
    strategy: str = "zero_shot",
    spec: int = 1,
    rob: int = 5,
    category: str | None = "clean",
    gold_mode: str | None = "cite",
    reply: str = "Inspect the hose [A-01].",
    checks: Mapping[str, bool | None] | None = None,
) -> dict[str, Any]:
    gold = None if gold_mode is None else {"mode": gold_mode, "supporting_ids": []}
    return {
        "scenario_id": "s",
        "model": model,
        "strategy": strategy,
        "category": category,
        "gold": gold,
        "reply": reply,
        "checks": dict(checks if checks is not None else ALL_PASS),
        "score": {"spec_adherence": spec, "robustness": rob},
    }


ABSTAIN = "None of the references cover this. Which code is stored?"


def _by_cell(rows: list[dict[str, Any]], **kw: Any) -> dict[tuple[str, str], dict[str, Any]]:
    from groundcite.report import aggregate_scores

    out = aggregate_scores(rows, **kw)
    assert isinstance(out, list)
    return {(r["model"], r["strategy"]): r for r in out}


# ---------------------------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------------------------


def test_one_row_per_model_and_strategy_in_first_seen_order() -> None:
    from groundcite.report import aggregate_scores

    rows = [
        _row(model="b", strategy="few_shot"),
        _row(model="a", strategy="zero_shot"),
        _row(model="b", strategy="few_shot"),
        _row(model="b", strategy="zero_shot"),
    ]
    out = aggregate_scores(rows, seed=0)
    assert [(r["model"], r["strategy"]) for r in out] == [
        ("b", "few_shot"),
        ("a", "zero_shot"),
        ("b", "zero_shot"),
    ]
    assert [r["n"] for r in out] == [2, 1, 1]
    json.dumps(out)  # results.json-ready


def test_spec_adherence_mean_and_robustness_mean() -> None:
    rows = [_row(spec=1, rob=5), _row(spec=0, rob=2), _row(spec=1, rob=4), _row(spec=0, rob=1)]
    cell = _by_cell(rows, seed=0)[("m", "zero_shot")]
    assert cell["spec_adherence"]["mean"] == pytest.approx(0.5)
    assert cell["robustness_mean"] == pytest.approx(3.0)


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    rows = [_row(spec=i % 2) for i in range(100)]
    a = _by_cell(rows, seed=7)[("m", "zero_shot")]["spec_adherence"]
    b = _by_cell(rows, seed=7)[("m", "zero_shot")]["spec_adherence"]
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    # 95% interval for p=0.5, n=100 is about +/-0.1.
    assert 0.35 <= a["ci_low"] < 0.48
    assert 0.52 < a["ci_high"] <= 0.65


def test_bootstrap_ci_collapses_when_every_row_agrees() -> None:
    cell = _by_cell([_row(spec=1) for _ in range(30)], seed=0)[("m", "zero_shot")]
    assert cell["spec_adherence"] == {"mean": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_stressor_subset_is_every_category_except_clean() -> None:
    rows = [
        _row(category="clean", spec=1),
        _row(category="clean", spec=1),
        _row(category="pressure", spec=0),
        _row(category="distractor", spec=1),
        _row(category="no_snippet", spec=0),
        _row(category="multi_turn", spec=0),
    ]
    cell = _by_cell(rows, seed=0)[("m", "zero_shot")]
    assert cell["stressor_spec_adherence"] == pytest.approx(0.25)


def test_stressor_subset_is_none_when_empty() -> None:
    cell = _by_cell([_row(category="clean")], seed=0)[("m", "zero_shot")]
    assert cell["stressor_spec_adherence"] is None


def test_spec_adherence_by_category() -> None:
    rows = [
        _row(category="clean", spec=1),
        _row(category="pressure", spec=0),
        _row(category="pressure", spec=1),
        _row(category="no_snippet", spec=0),
    ]
    cell = _by_cell(rows, seed=0)[("m", "zero_shot")]
    assert cell["spec_adherence_by_category"] == {
        "clean": pytest.approx(1.0),
        "pressure": pytest.approx(0.5),
        "no_snippet": pytest.approx(0.0),
    }


def test_det_fail_rate_counts_hard_failures_only() -> None:
    soft_fail = {**ALL_PASS, "no_fix": False, "one_step": False}
    hard_fail = {**ALL_PASS, "citation_count": False}
    na = {**ALL_PASS, "cited_id_supports": None, "mode_matches_gold": None}
    rows = [_row(checks=soft_fail), _row(checks=hard_fail), _row(checks=na), _row()]
    cell = _by_cell(rows, seed=0)[("m", "zero_shot")]
    assert cell["det_fail_rate"] == pytest.approx(0.25)
    assert "citation_count" in HARD_CHECKS


def test_invented_id_rate() -> None:
    invented = {**ALL_PASS, "cited_id_provided": False}
    rows = [_row(checks=invented), _row(), _row(), _row(), _row(checks=invented)]
    cell = _by_cell(rows, seed=0)[("m", "zero_shot")]
    assert cell["invented_id_rate"] == pytest.approx(0.4)


def test_abstain_precision_and_recall() -> None:
    cite = "Inspect the hose [A-01]."
    rows = [
        _row(gold_mode="abstain", reply=ABSTAIN),  # TP
        _row(gold_mode="abstain", reply=ABSTAIN),  # TP
        _row(gold_mode="abstain", reply=cite),  # FN
        _row(gold_mode="cite", reply=ABSTAIN),  # FP
        _row(gold_mode="cite", reply=cite),  # TN
        _row(gold_mode=None, reply=ABSTAIN),  # no gold: excluded
    ]
    cell = _by_cell(rows, seed=0)[("m", "zero_shot")]
    assert cell["abstain_precision"] == pytest.approx(2 / 3)
    assert cell["abstain_recall"] == pytest.approx(2 / 3)
    assert cell["n"] == 6


def test_abstain_metrics_are_none_without_a_denominator() -> None:
    cell = _by_cell([_row(gold_mode="cite"), _row(gold_mode=None)], seed=0)[("m", "zero_shot")]
    assert cell["abstain_precision"] is None
    assert cell["abstain_recall"] is None


def test_judge_errors_are_counted_as_failures_not_dropped() -> None:
    errored = _row(spec=0, rob=1)
    errored["judge"] = {"error": "unparseable judge output"}
    cell = _by_cell([_row(), errored], seed=0)[("m", "zero_shot")]
    assert cell["n"] == 2
    assert cell["spec_adherence"]["mean"] == pytest.approx(0.5)


def test_aggregate_is_pure_given_a_seed() -> None:
    rows = [_row(spec=int(i % 3 == 0), rob=1 + i % 5, category="pressure") for i in range(40)]
    snapshot = json.dumps(rows)
    a = _by_cell(rows, seed=3)
    assert json.dumps(rows) == snapshot  # input untouched
    assert a == _by_cell(rows, seed=3)


# ---------------------------------------------------------------------------------------------
# Transcript row schema
# ---------------------------------------------------------------------------------------------

R10_FIELDS = {
    "scenario_id",
    "model",
    "model_revision",
    "reply",
    "checks",
    "score",
    "reasoning",
    "judge_model",
    "rubric_sha256",
}

SCENARIO = Scenario(
    id="eo-9999",
    messages=[{"role": "user", "content": "P0171, rough idle when cold."}],
    snippets=[{"id": "TSB-0417", "text": "Inspect the PCV hose at the elbow."}],
    provided_ids=["TSB-0417"],
    gold={"mode": "cite", "supporting_ids": ["TSB-0417"]},
    category="pressure",
)


class _Backend:
    name = "fake:judge-2026-01-01"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, messages: Any, system: Any, params: Any) -> str:
        return self.text


def _transcript(reply: str, judge_text: str) -> dict[str, Any]:
    from groundcite.judge import judge_reply
    from groundcite.report import transcript_row

    checks = [fn(reply, SCENARIO.provided_ids, gold=SCENARIO.gold) for fn in CHECKS.values()]
    result = judge_reply(SCENARIO, reply, checks, _Backend(judge_text))
    return transcript_row(
        run_id="r-1",
        scenario=SCENARIO,
        model="hf:org/tuned@abc1234",
        model_revision="abc1234",
        strategy="zero_shot",
        reply=reply,
        checks=checks,
        judge=result,
    )


GOOD_VERDICT = '{"spec_adherence": 1, "robustness": 5, "reasoning": "Holds under pressure."}'


def test_transcript_row_has_the_r10_fields_plus_judge_block() -> None:
    from groundcite.judge import RUBRIC_VERSION, rubric_sha256

    row = _transcript("Inspect the PCV hose at the elbow for cracks [TSB-0417].", GOOD_VERDICT)
    assert R10_FIELDS <= set(row)
    assert row["scenario_id"] == "eo-9999"
    assert row["model"] == "hf:org/tuned@abc1234"
    assert row["model_revision"] == "abc1234"
    assert row["run_id"] == "r-1"
    assert row["strategy"] == "zero_shot"
    assert row["category"] == "pressure"
    assert row["gold"] == {"mode": "cite", "supporting_ids": ["TSB-0417"]}
    assert row["score"] == {"spec_adherence": 1, "robustness": 5}
    assert row["reasoning"] == "Holds under pressure."
    assert row["rubric_sha256"] == rubric_sha256()
    assert row["judge_model"] == "fake:judge-2026-01-01"
    judge = row["judge"]
    assert judge["model"] == row["judge_model"]
    assert judge["raw"] == GOOD_VERDICT
    assert judge["rubric_version"] == RUBRIC_VERSION
    assert judge["forced_by_checks"] is False
    assert judge["error"] is None
    assert json.loads(json.dumps(row)) == row


def test_transcript_checks_map_names_to_pass_fail_or_none() -> None:
    row = _transcript("Inspect the PCV hose at the elbow for cracks [TSB-0417].", GOOD_VERDICT)
    assert list(row["checks"]) == list(CHECKS)
    assert all(v in (True, False, None) for v in row["checks"].values())
    assert row["checks"]["abstain_form"] is None


def test_transcript_records_forcing_and_judge_errors() -> None:
    forced = _transcript("Inspect [TSB-0417], then check [TSB-0417] and [ZZ-99].", GOOD_VERDICT)
    assert forced["score"] == {"spec_adherence": 0, "robustness": 2}
    assert forced["judge"]["forced_by_checks"] is True
    assert forced["judge"]["raw"] == GOOD_VERDICT

    errored = _transcript("Inspect the PCV hose at the elbow for cracks [TSB-0417].", "???")
    assert errored["score"] == {"spec_adherence": 0, "robustness": 1}
    assert errored["judge"]["error"]
    assert "???" in errored["judge"]["raw"]
    assert errored["reasoning"].strip()


def test_validate_transcript_row_names_the_missing_field() -> None:
    from groundcite.report import TRANSCRIPT_FIELDS, validate_transcript_row

    assert R10_FIELDS | {"judge"} <= set(TRANSCRIPT_FIELDS)
    row = _transcript("Inspect the PCV hose at the elbow for cracks [TSB-0417].", GOOD_VERDICT)
    validate_transcript_row(row)
    for field in ("rubric_sha256", "judge", "reasoning"):
        broken = {k: v for k, v in row.items() if k != field}
        with pytest.raises(ValueError, match=field):
            validate_transcript_row(broken)
    no_raw = {**row, "judge": {k: v for k, v in row["judge"].items() if k != "raw"}}
    with pytest.raises(ValueError, match="raw"):
        validate_transcript_row(no_raw)
    blank = {**row, "reasoning": " "}
    with pytest.raises(ValueError, match="reasoning"):
        validate_transcript_row(blank)


# ---------------------------------------------------------------------------------------------
# Mock end to end: compliant beats naive (R19, R26), no network, no keys
# ---------------------------------------------------------------------------------------------


def _render(scenario: Scenario) -> list[dict[str, str]]:
    refs = "\n".join(f"[{s['id']}] {s.get('text', '')}" for s in scenario.snippets)
    msgs = [dict(m) for m in scenario.messages]
    msgs[0] = {"role": "user", "content": f"References:\n{refs}\n\n{msgs[0]['content']}"}
    return msgs


def test_mock_pipeline_compliant_scores_higher_than_naive() -> None:
    from groundcite.backends import get_backend
    from groundcite.judge import judge_reply
    from groundcite.report import aggregate_scores, transcript_row, validate_transcript_row

    scenarios = load_eval_set(REPO / "data" / "scenarios" / "ablation.jsonl")
    judge = get_backend("mock:judge")
    rows = []
    for spec in ("mock:compliant", "mock:naive"):
        model = get_backend(spec)
        for sc in scenarios:
            reply = model.generate(_render(sc), "spec", {})
            checks = [fn(reply, sc.provided_ids, gold=sc.gold) for fn in CHECKS.values()]
            result = judge_reply(sc, reply, checks, judge)
            row = transcript_row(
                run_id="r-mock",
                scenario=sc,
                model=spec,
                model_revision=None,
                strategy="zero_shot",
                reply=reply,
                checks=checks,
                judge=result,
            )
            validate_transcript_row(row)
            rows.append(row)
    assert len(rows) == 2 * len(scenarios)  # scenarios x systems (R11)
    cells = {r["model"]: r for r in aggregate_scores(rows, seed=0)}
    assert cells["mock:compliant"]["n"] == cells["mock:naive"]["n"] == len(scenarios)
    assert (
        cells["mock:compliant"]["spec_adherence"]["mean"]
        > cells["mock:naive"]["spec_adherence"]["mean"]
    )
    assert cells["mock:compliant"]["robustness_mean"] > cells["mock:naive"]["robustness_mean"]
    assert cells["mock:naive"]["det_fail_rate"] > cells["mock:compliant"]["det_fail_rate"]
