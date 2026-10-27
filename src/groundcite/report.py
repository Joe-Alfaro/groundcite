"""Score aggregation and the transcript row schema (Robustness definition, R3, R10, R11, R30).

``aggregate_scores`` is the single aggregation every runner imports (ablation, base vs tuned,
curve). Judge-error rows are already scored (0, 1) and count like any other row.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from groundcite.checks import CHECKS, HARD_CHECKS, CheckResult, infer_mode
from groundcite.evalset import Scenario
from groundcite.judge import RUBRIC_VERSION, JudgeResult

TRANSCRIPT_FIELDS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "model",
    "model_revision",
    "strategy",
    "category",
    "gold",
    "reply",
    "checks",
    "score",
    "reasoning",
    "judge_model",
    "rubric_sha256",
    "judge",
)
CLEAN_CATEGORY = "clean"
UNCATEGORIZED = "uncategorized"


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _bootstrap_ci(
    values: Sequence[float], rng: random.Random, n_boot: int, confidence: float
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    n = len(values)
    means = sorted(_mean(rng.choices(values, k=n)) for _ in range(n_boot))
    tail = (1 - confidence) / 2
    low = means[int(tail * (n_boot - 1))]
    high = means[int(round((1 - tail) * (n_boot - 1)))]
    return low, high


def _cell(
    rows: Sequence[Mapping[str, Any]], rng: random.Random, n_boot: int, confidence: float
) -> dict[str, Any]:
    spec = [float(r["score"]["spec_adherence"]) for r in rows]
    robustness = [float(r["score"]["robustness"]) for r in rows]
    ci_low, ci_high = _bootstrap_ci(spec, rng, n_boot, confidence)

    by_category: dict[str, list[float]] = {}
    for r, s in zip(rows, spec, strict=True):
        by_category.setdefault(r.get("category") or UNCATEGORIZED, []).append(s)
    # Stressors are the tagged non-clean categories; untagged third-party rows are not stressors.
    stressor = [
        s
        for r, s in zip(rows, spec, strict=True)
        if r.get("category") not in (None, CLEAN_CATEGORY)
    ]

    hard_fail = sum(any(r["checks"].get(name) is False for name in HARD_CHECKS) for r in rows)
    invented = sum(r["checks"].get("cited_id_provided") is False for r in rows)

    tp = fp = fn = 0
    for r in rows:
        gold_mode = (r.get("gold") or {}).get("mode")
        if gold_mode is None:
            continue
        predicted = infer_mode(r["reply"]) == "abstain"
        actual = gold_mode == "abstain"
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted

    n = len(rows)
    return {
        "model": rows[0]["model"],
        "strategy": rows[0]["strategy"],
        "n": n,
        "spec_adherence": {"mean": _mean(spec), "ci_low": ci_low, "ci_high": ci_high},
        "robustness_mean": _mean(robustness),
        "stressor_spec_adherence": _mean(stressor) if stressor else None,
        "spec_adherence_by_category": {k: _mean(v) for k, v in by_category.items()},
        "det_fail_rate": hard_fail / n,
        "invented_id_rate": invented / n,
        "abstain_precision": _ratio(tp, tp + fp),
        "abstain_recall": _ratio(tp, tp + fn),
    }


def aggregate_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    n_boot: int = 1000,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """One JSON-ready summary per (model, strategy), in first-seen order.

    Spec-adherence carries a seeded percentile-bootstrap interval. Rows need ``model``,
    ``strategy``, ``category``, ``gold``, ``reply``, ``checks`` and ``score``.
    """
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["model"], row["strategy"]), []).append(row)
    return [
        _cell(cell_rows, _cell_rng(seed, model, strategy), n_boot, confidence)
        for (model, strategy), cell_rows in cells.items()
    ]


def _cell_rng(seed: int, model: str, strategy: str) -> random.Random:
    """An RNG for one cell, independent of the other cells and of PYTHONHASHSEED (D-039)."""
    digest = hashlib.sha256(f"{seed}\x00{model}\x00{strategy}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------------------------
# Transcript rows
# ---------------------------------------------------------------------------------------------


def transcript_row(
    *,
    run_id: str,
    scenario: Scenario,
    model: str,
    model_revision: str | None,
    strategy: str,
    reply: str,
    checks: Sequence[CheckResult],
    judge: JudgeResult,
) -> dict[str, Any]:
    """One JSON-ready line of ``transcripts.jsonl`` (R10, R11)."""
    by_name = {c.name: c.passed for c in checks}
    return {
        "run_id": run_id,
        "scenario_id": scenario.id,
        "model": model,
        "model_revision": model_revision,
        "strategy": strategy,
        "category": scenario.category,
        "gold": dict(scenario.gold) if scenario.gold is not None else None,
        "reply": reply,
        "checks": {name: by_name.get(name) for name in CHECKS},
        "score": {"spec_adherence": judge.spec_adherence, "robustness": judge.robustness},
        "reasoning": judge.reasoning,
        "judge_model": judge.model,
        "rubric_sha256": judge.rubric_sha256,
        "judge": {
            "model": judge.model,
            "raw": judge.raw,
            "rubric_version": RUBRIC_VERSION,
            "forced_by_checks": judge.forced_by_checks,
            "error": judge.error,
            "attempts": judge.attempts,
            "failed_checks": list(judge.failed_checks),
        },
    }


def validate_transcript_row(row: Mapping[str, Any]) -> None:
    """Raise ValueError naming the first missing or invalid field."""
    for field in TRANSCRIPT_FIELDS:
        if field not in row:
            raise ValueError(f"transcript row is missing field {field!r}")
    score = row["score"]
    for key in ("spec_adherence", "robustness"):
        if not isinstance(score, Mapping) or key not in score:
            raise ValueError(f"transcript row score is missing {key!r}")
    judge = row["judge"]
    if not isinstance(judge, Mapping) or not isinstance(judge.get("raw"), str):
        raise ValueError("transcript row judge block is missing 'raw'")
    reasoning = row["reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("transcript row has blank 'reasoning'")
