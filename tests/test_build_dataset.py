"""Tests for the R4 quality filter and dataset build (R4, R6, R23, R27, R30, GOTCHA G22, G23).

`scripts/build_dataset.py --candidates P --out data/datasets/vN [--judge SPEC]` keeps a
candidate only if every HARD and SOFT check passes, it agrees with gold, the judge gives
Spec-adherence 1 and Robustness >= 4, and its scenario is not a near-duplicate (R23 rules on the
final user message) of an already kept example. At most one candidate per scenario is kept:
the highest judge Robustness wins, ties go to the lowest `candidate_index`.

It writes `train.jsonl` (chat records rendered with the shared `render_messages(…, "zero_shot")`),
`manifest.json`, `rejections.jsonl` and `filter_report.md`. Judges are fakes registered under
`backends.MOCKS`; sockets are blocked and keys unset.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from distill_data import (
    ABSTAIN_REPLY,
    EVAL_SETS,
    FINAL_MESSAGES,
    REPO_ROOT,
    RecordingTeacher,
    ScriptedJudge,
    candidate_row,
    cite_reply,
    id_scheme,
    read_jsonl,
    run_script,
    scenario_row,
    write_jsonl,
)

from groundcite import backends
from groundcite.evalset import parse_row
from groundcite.judge import rubric_sha256

CHECK_LEAKAGE = REPO_ROOT / "scripts" / "check_leakage.py"
HINT_WORDS = re.compile(r"\b(gold|supporting_ids|privileged|hint)\b", re.IGNORECASE)


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedJudge]:
    monkeypatch.setattr(ScriptedJudge, "verdicts", {})
    monkeypatch.setattr(ScriptedJudge, "calls", 0)
    seen: list[str] = []
    original = ScriptedJudge.generate

    def recording(self: ScriptedJudge, messages: Any, system: Any, params: Any) -> str:
        seen.append("\n".join(m["content"] for m in messages))
        return original(self, messages, system, params)

    monkeypatch.setattr(ScriptedJudge, "generate", recording)
    monkeypatch.setattr(ScriptedJudge, "seen", seen, raising=False)
    monkeypatch.setitem(backends.MOCKS, "judge_probe", ScriptedJudge)
    return ScriptedJudge


# ---------------------------------------------------------------------------------------------
# One candidates file that exercises every filter stage
# ---------------------------------------------------------------------------------------------


def _gold(row: dict[str, Any]) -> str:
    return str(row["gold"]["supporting_ids"][0])


def _near(row: dict[str, Any]) -> str:
    return str(row["gold"]["near_miss_ids"][0])


def build_fixture(judge: type[ScriptedJudge]) -> dict[str, Any]:
    """Scenarios s1..s11 with two candidates each, the expected keeps and rejection reasons."""
    s = {
        1: scenario_row(1, category="clean"),
        2: scenario_row(2, category="distractor"),
        3: scenario_row(3, category="no_snippet", mode="abstain"),
        4: scenario_row(4, category="pressure"),
        5: scenario_row(5, category="multi_symptom", mode="abstain"),
        6: scenario_row(6, category="multi_turn"),
        7: scenario_row(7, category="clean"),
        8: scenario_row(8, category="distractor"),
        9: scenario_row(9, category="pressure"),
        10: scenario_row(
            10,
            category="pressure",
            final=FINAL_MESSAGES[9 % 16].replace("driveway at all", "driveway at night"),
        ),
        11: scenario_row(11, category="clean"),
    }
    replies: dict[tuple[int, int], tuple[str, str | None]] = {
        (1, 0): (cite_reply(_gold(s[1]), 0), None),
        (1, 1): (cite_reply(_gold(s[1]), 1), None),
        (2, 0): (cite_reply(_near(s[2]), 0), None),
        (2, 1): (cite_reply(_gold(s[2]), 2), None),
        (3, 0): (ABSTAIN_REPLY, None),
        (3, 1): ("", "teacher overloaded"),
        (4, 0): (f"Replace the hose at the intake elbow [{_gold(s[4])}].", None),
        (4, 1): (f"Inspect the hose [{_gold(s[4])}], then check the fuel pressure.", None),
        (5, 0): (
            "None of the references you provided cover this complaint. Which trouble code is "
            "stored? What is the mileage?",
            None,
        ),
        (5, 1): ("<think>No reference fits.</think> " + ABSTAIN_REPLY, None),
        (6, 0): (cite_reply(_gold(s[6]), 0), None),
        (6, 1): (cite_reply(_gold(s[6]), 1), None),
        (7, 0): (f"Per the hint, measure the voltage at the connector [{_gold(s[7])}].", None),
        (7, 1): (
            "Key on, engine off, measure the voltage at the sensor connector against the",
            None,
        ),
        (8, 0): (cite_reply(_gold(s[8]), 0), None),
        (8, 1): (cite_reply(_gold(s[8]), 1), None),
        (9, 0): (cite_reply(_gold(s[9]), 0), None),
        (9, 1): (cite_reply(_gold(s[9]), 1), None),
        (10, 0): (cite_reply(_gold(s[10]), 0), None),
        (10, 1): (cite_reply(_gold(s[10]), 1), None),
        (11, 0): (cite_reply(_gold(s[11]), 0), None),
        (11, 1): (cite_reply(_gold(s[11]), 1), None),
    }
    judge.verdicts = {
        replies[(6, 0)][0]: (1, 3),
        replies[(6, 1)][0]: (1, 4),
        replies[(8, 0)][0]: (0, 4),
        replies[(11, 0)][0]: (1, 4),
        replies[(11, 1)][0]: (1, 5),
    }
    kept = {1: 0, 2: 1, 3: 0, 6: 1, 8: 1, 9: 0, 11: 1}
    reasons = {
        (1, 1): "not_selected",
        (2, 0): "gold_mismatch",
        (3, 1): "generation_error",
        (4, 0): "soft_check",
        (4, 1): "soft_check",
        (5, 0): "hard_check",
        (5, 1): "hard_check",
        (6, 0): "judge_robustness",
        (7, 0): "hint_leak",
        (7, 1): "unclean_end",
        (8, 0): "judge_spec",
        (9, 1): "not_selected",
        (10, 0): "near_duplicate",
        (10, 1): "near_duplicate",
        (11, 0): "not_selected",
    }
    candidates = [
        candidate_row(s[n], i, reply, error=error) for (n, i), (reply, error) in replies.items()
    ]
    pre_judge = {
        replies[key][0]
        for key, why in reasons.items()
        if why
        in {
            "gold_mismatch",
            "generation_error",
            "soft_check",
            "hard_check",
            "hint_leak",
            "unclean_end",
        }
    }
    return {
        "scenarios": s,
        "replies": replies,
        "candidates": candidates,
        "kept": kept,
        "reasons": reasons,
        "pre_judge_replies": pre_judge,
    }


def _build(*args: str) -> int:
    return run_script("build_dataset", list(args))


@pytest.fixture
def built(env: Path, judge: type[ScriptedJudge], tmp_path: Path) -> dict[str, Any]:
    fx = build_fixture(judge)
    cands = write_jsonl(tmp_path / "distill" / "candidates.jsonl", fx["candidates"])
    out = tmp_path / "datasets" / "v1"
    assert _build("--candidates", str(cands), "--out", str(out), "--judge", "mock:judge_probe") == 0
    fx["out"] = out
    fx["candidates_path"] = cands
    fx["train"] = read_jsonl(out / "train.jsonl")
    fx["manifest"] = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    fx["rejections"] = read_jsonl(out / "rejections.jsonl")
    return fx


# --- filter decisions ------------------------------------------------------------------------


def test_keeps_exactly_the_expected_candidate_per_scenario(built: dict[str, Any]) -> None:
    s, replies = built["scenarios"], built["replies"]
    got = {rec["meta"]["scenario_id"]: rec["meta"]["candidate_index"] for rec in built["train"]}
    want = {s[n]["id"]: i for n, i in built["kept"].items()}
    assert got == want
    for rec in built["train"]:
        n = next(k for k, v in s.items() if v["id"] == rec["meta"]["scenario_id"])
        assert rec["messages"][-1]["content"] == replies[(n, rec["meta"]["candidate_index"])][0]


def test_at_most_one_example_per_scenario(built: dict[str, Any]) -> None:
    ids = [rec["meta"]["scenario_id"] for rec in built["train"]]
    assert len(ids) == len(set(ids))


def test_every_rejection_has_the_expected_reason(built: dict[str, Any]) -> None:
    from groundcite.filter import REJECTION_REASONS

    s = built["scenarios"]
    got = {(r["scenario_id"], r["candidate_index"]): r["reason"] for r in built["rejections"]}
    want = {(s[n]["id"], i): why for (n, i), why in built["reasons"].items()}
    assert got == want
    assert set(want.values()) <= set(REJECTION_REASONS)
    assert all(isinstance(r["detail"], str) and r["detail"].strip() for r in built["rejections"])
    assert len(built["rejections"]) + len(built["train"]) == len(built["candidates"])


def test_check_rejections_name_the_failed_checks(built: dict[str, Any]) -> None:
    by_key = {(r["scenario_id"], r["candidate_index"]): r for r in built["rejections"]}
    s = built["scenarios"]
    assert "no_fix" in by_key[(s[4]["id"], 0)]["failed_checks"]
    assert "one_step" in by_key[(s[4]["id"], 1)]["failed_checks"]
    assert "abstain_form" in by_key[(s[5]["id"], 0)]["failed_checks"]
    assert "no_think" in by_key[(s[5]["id"], 1)]["failed_checks"]
    assert "cited_id_supports" in by_key[(s[2]["id"], 0)]["failed_checks"]


def test_judge_is_not_called_for_deterministic_rejects(
    built: dict[str, Any], judge: type[ScriptedJudge]
) -> None:
    seen = judge.seen  # type: ignore[attr-defined]
    assert seen, "the judge was never called"
    for reply in built["pre_judge_replies"]:
        if reply:
            assert not any(reply in text for text in seen), f"judged a pre-judge reject: {reply}"
    assert judge.calls <= 15


def test_robustness_four_is_kept_and_three_is_not(built: dict[str, Any]) -> None:
    s6 = built["scenarios"][6]["id"]
    rec = next(r for r in built["train"] if r["meta"]["scenario_id"] == s6)
    assert rec["meta"]["judge"]["robustness"] == 4
    assert rec["meta"]["judge"]["spec_adherence"] == 1


# --- train.jsonl -----------------------------------------------------------------------------


def test_train_records_render_through_the_shared_renderer(built: dict[str, Any]) -> None:
    from groundcite.prompts import render_messages

    by_id = {row["id"]: row for row in built["scenarios"].values()}
    for rec in built["train"]:
        system, messages = render_messages(
            parse_row(by_id[rec["meta"]["scenario_id"]]), "zero_shot"
        )
        assert rec["messages"][0] == {"role": "system", "content": system}
        assert rec["messages"][1:-1] == messages
        assert rec["messages"][-1]["role"] == "assistant"


def test_targets_hold_no_hint_and_end_cleanly(built: dict[str, Any]) -> None:
    from groundcite.distill import HINT_HEADER, render_hint

    by_id = {row["id"]: row for row in built["scenarios"].values()}
    for rec in built["train"]:
        target = rec["messages"][-1]["content"]
        assert target == target.strip()
        assert target[-1] in ".?!", f"target does not end cleanly: {target!r}"
        assert "<think" not in target.lower()
        assert not HINT_WORDS.search(target)
        assert HINT_HEADER.lower() not in target.lower()
        assert render_hint(parse_row(by_id[rec["meta"]["scenario_id"]])) not in target
        prompt = "\n".join(m["content"] for m in rec["messages"][:-1])
        assert HINT_HEADER not in prompt


def test_train_meta_fields(built: dict[str, Any]) -> None:
    from groundcite.filter import FILTER_VERSION

    by_id = {row["id"]: row for row in built["scenarios"].values()}
    for rec in built["train"]:
        meta = rec["meta"]
        scenario = by_id[meta["scenario_id"]]
        assert meta["category"] == scenario["category"]
        assert meta["family"] == scenario["family"]
        assert meta["teacher"] == "mock:teacher_probe"
        assert meta["filter_version"] == FILTER_VERSION
        assert set(meta["judge"]) >= {"spec_adherence", "robustness"}


def test_train_jsonl_passes_the_leakage_check(built: dict[str, Any]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_LEAKAGE),
            "--train",
            str(built["out"] / "train.jsonl"),
            "--eval",
            *(str(p) for p in EVAL_SETS),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- manifest and report ---------------------------------------------------------------------


def test_manifest_records_counts_mix_and_provenance(built: dict[str, Any]) -> None:
    from groundcite.distill import TEACHER_PROMPT_VERSION, teacher_prompt_sha256
    from groundcite.filter import FILTER_VERSION

    m = built["manifest"]
    s = built["scenarios"]
    kept_rows = [s[n] for n in built["kept"]]
    assert m["version"] == "v1"
    assert m["count"] == len(kept_rows) == len(built["train"])
    assert m["counts_by_category"] == dict(Counter(r["category"] for r in kept_rows))
    assert m["id_scheme_mix"] == dict(
        Counter(id_scheme(sn["id"]) for r in kept_rows for sn in r["snippets"])
    )
    abstain = sum(r["gold"]["mode"] == "abstain" for r in kept_rows)
    assert m["abstain_share"] == pytest.approx(abstain / len(kept_rows))
    assert m["source_seeds"] == sorted({r["seed"] for r in kept_rows})
    assert m["teacher_model"] == "mock:teacher_probe"
    assert m["teacher_prompt_version"] == TEACHER_PROMPT_VERSION
    assert m["teacher_prompt_sha256"] == teacher_prompt_sha256()
    assert m["filter_version"] == FILTER_VERSION
    assert m["judge_model"] == "mock:judge_probe"
    assert m["rubric_sha256"] == rubric_sha256()
    train_bytes = (built["out"] / "train.jsonl").read_bytes()
    assert m["train_sha256"] == hashlib.sha256(train_bytes).hexdigest()
    cand_bytes = built["candidates_path"].read_bytes()
    assert m["candidates_sha256"] == hashlib.sha256(cand_bytes).hexdigest()
    assert isinstance(m["selection_rule"], str) and m["selection_rule"].strip()


def test_manifest_holds_the_filter_report_numbers(built: dict[str, Any]) -> None:
    m = built["manifest"]
    n_cands, n_kept = len(built["candidates"]), len(built["kept"])
    assert m["candidates"] == n_cands
    assert m["kept"] == n_kept
    assert m["kept_rate"] == pytest.approx(n_kept / n_cands)
    assert m["kept_rate_in_range"] is (0.20 <= n_kept / n_cands <= 0.95)
    assert m["rejections_by_reason"] == dict(Counter(built["reasons"].values()))


def test_filter_report_lists_kept_rate_and_each_reason(built: dict[str, Any]) -> None:
    report = (built["out"] / "filter_report.md").read_text(encoding="utf-8")
    n_cands, n_kept = len(built["candidates"]), len(built["kept"])
    assert re.search(rf"\b{n_cands}\b", report)
    assert re.search(rf"\b{n_kept}\b", report)
    assert f"{100 * n_kept / n_cands:.1f}%" in report
    for reason, count in Counter(built["reasons"].values()).items():
        assert re.search(rf"{reason}\D*\b{count}\b", report), f"{reason} {count} not in report"


def test_build_is_deterministic(env: Path, judge: type[ScriptedJudge], tmp_path: Path) -> None:
    fx = build_fixture(judge)
    cands = write_jsonl(tmp_path / "candidates.jsonl", fx["candidates"])
    outs = [tmp_path / "a" / "v1", tmp_path / "b" / "v1"]
    for out in outs:
        assert (
            _build("--candidates", str(cands), "--out", str(out), "--judge", "mock:judge_probe")
            == 0
        )
    for name in ("train.jsonl", "rejections.jsonl", "manifest.json", "filter_report.md"):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name


# --- failures --------------------------------------------------------------------------------


def test_missing_judge_key_exits_two_and_writes_nothing(
    env: Path, judge: type[ScriptedJudge], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fx = build_fixture(judge)
    cands = write_jsonl(tmp_path / "candidates.jsonl", fx["candidates"])
    out = tmp_path / "v1"
    assert _build("--candidates", str(cands), "--out", str(out)) == 2  # default judge: OpenAI
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not out.exists() or not any(out.iterdir())


def test_existing_dataset_dir_is_never_overwritten(
    env: Path, judge: type[ScriptedJudge], tmp_path: Path
) -> None:
    fx = build_fixture(judge)
    cands = write_jsonl(tmp_path / "candidates.jsonl", fx["candidates"])
    out = tmp_path / "v1"
    out.mkdir()
    (out / "train.jsonl").write_text("{}\n", encoding="utf-8")
    assert _build("--candidates", str(cands), "--out", str(out), "--judge", "mock:judge_probe") == 2
    assert (out / "train.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_malformed_candidate_fails_fast_with_line_and_field(
    env: Path, judge: type[ScriptedJudge], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fx = build_fixture(judge)
    rows = fx["candidates"][:3]
    del rows[1]["scenario"]
    cands = write_jsonl(tmp_path / "candidates.jsonl", rows)
    out = tmp_path / "v1"
    assert _build("--candidates", str(cands), "--out", str(out), "--judge", "mock:judge_probe") == 2
    err = capsys.readouterr().err
    assert "line 2" in err and "scenario" in err
    assert not out.exists() or not any(out.iterdir())


# --- end to end ------------------------------------------------------------------------------


def test_distill_then_build_then_sweep_with_mocks(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R5-style smoke: the teacher (fake, hint-following), the mock judge, then the sweep."""
    monkeypatch.setattr(RecordingTeacher, "calls", [])
    rows = [
        scenario_row(i, category=c, mode="abstain" if c == "no_snippet" else "cite")
        for i, c in enumerate(
            ["clean", "distractor", "no_snippet", "pressure", "multi_symptom", "multi_turn"] * 2,
            start=20,
        )
    ]
    monkeypatch.setattr(
        RecordingTeacher,
        "gold_ids",
        {r["id"]: r["gold"]["supporting_ids"][0] for r in rows if r["gold"]["supporting_ids"]},
    )
    monkeypatch.setitem(backends.MOCKS, "teacher_probe", RecordingTeacher)
    pool = write_jsonl(tmp_path / "pool.jsonl", rows)
    distill_out, ds = tmp_path / "distill", tmp_path / "datasets" / "v1"

    assert (
        run_script(
            "distill",
            [
                "--scenarios",
                str(pool),
                "--teacher",
                "mock:teacher_probe",
                "--k",
                "2",
                "--out",
                str(distill_out),
            ],
        )
        == 0
    )
    assert (
        _build(
            "--candidates",
            str(distill_out / "candidates.jsonl"),
            "--out",
            str(ds),
            "--judge",
            "mock",
        )
        == 0
    )
    train = read_jsonl(ds / "train.jsonl")
    assert len(train) == len(rows)
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_model"] == "mock:judge"
    assert (
        run_script("make_sweep", ["--dataset", str(ds), "--sizes", "12", "6", "3", "--seed", "7"])
        == 0
    )
    assert len(read_jsonl(ds / "subsets" / "n3.jsonl")) == 3
