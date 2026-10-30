"""Deterministic-check metrics for one cell of replies, with no judge (R8 data-efficiency curve).

Every metric is a mean of 0/1 values over the rows it applies to, with the same seeded
percentile bootstrap as ``aggregate_scores``. HARD and SOFT follow the check names (D-028), as
``aggregate_scores`` does for its HARD-check fail rate. A generation-error row fails every rate it
belongs to but makes no abstain prediction, as in ``aggregate_scores`` (R30, D-043).

``scripts/curve_det.py`` runs ``main``: base and full-v1 rows come from the judged main run's
transcripts (their stored checks), each curve point's tuned rows from its generations, checked
here. Run with ``--help`` for every flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from groundcite.checks import HARD_CHECKS, SOFT_CHECKS, infer_mode, run_checks
from groundcite.evalset import Scenario, load_eval_set
from groundcite.report import _bootstrap_ci

METRICS: tuple[str, ...] = (
    "hard_pass",
    "soft_flag",
    "invented_id",
    "mode_accuracy",
    "supports_on_cite_gold",
    "abstain_precision",
    "abstain_recall",
)


def checks_for(reply: str, scenario: Scenario) -> dict[str, bool | None]:
    """The project's checks on one reply, as the transcripts store them (name to pass/fail/n/a)."""
    return {r.name: r.passed for r in run_checks(reply, scenario)}


def _values(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    for r in rows:
        checks = r["checks"]
        error = bool(r.get("generation_error"))
        gold_mode = (r.get("gold") or {}).get("mode")
        values["hard_pass"].append(
            float(not error and not any(checks.get(n) is False for n in HARD_CHECKS))
        )
        values["soft_flag"].append(float(any(checks.get(n) is False for n in SOFT_CHECKS)))
        values["invented_id"].append(float(checks.get("cited_id_provided") is False))
        if gold_mode is None:
            continue
        mode = None if error else infer_mode(r["reply"])
        values["mode_accuracy"].append(float(mode == gold_mode))
        if gold_mode == "cite":
            values["supports_on_cite_gold"].append(
                float(not error and checks.get("cited_id_supports") is True)
            )
        if mode == "abstain":
            values["abstain_precision"].append(float(gold_mode == "abstain"))
        if gold_mode == "abstain" and not error:
            values["abstain_recall"].append(float(mode == "abstain"))
    return values


def _rng(seed: int, model: str, strategy: str, metric: str) -> random.Random:
    """One RNG per metric, independent of the others and of PYTHONHASHSEED (D-039)."""
    digest = hashlib.sha256(f"{seed}\x00{model}\x00{strategy}\x00{metric}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def deterministic_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    n_boot: int = 2000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Each metric's value, row count and bootstrap interval for one (model, strategy) cell.

    Rows need ``model``, ``strategy``, ``gold``, ``reply`` and ``checks``. A metric with no
    eligible rows is all None.
    """
    cells = {(r["model"], r["strategy"]) for r in rows}
    if len(cells) != 1:
        raise ValueError(f"rows must share one model and strategy, got {sorted(cells)}")
    ((model, strategy),) = cells
    out: dict[str, Any] = {"model": model, "strategy": strategy, "n": len(rows)}
    for name, vals in _values(rows).items():
        if not vals:
            out[name] = {"value": None, "n": 0, "ci_low": None, "ci_high": None}
            continue
        low, high = _bootstrap_ci(vals, _rng(seed, model, strategy, name), n_boot, confidence)
        out[name] = {"value": sum(vals) / len(vals), "n": len(vals), "ci_low": low, "ci_high": high}
    return out


FULL_V1_N = 448
BASE_STRATEGIES = ("zero_shot", "structured")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _is_tuned(model: str) -> bool:
    return "groundcite" in model


def _point(arg: str) -> tuple[int, Path]:
    n, _, path = arg.partition("=")
    return int(n), Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic-check metrics per curve point.")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True, help="judged main run")
    parser.add_argument("--point", type=_point, action="append", default=[], help="N=gens.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)

    scenarios = {s.id: s for s in load_eval_set(args.scenarios)}
    main_rows = _read_jsonl(args.transcripts)
    mismatches = sum(
        checks_for(r["reply"], scenarios[r["scenario_id"]]) != r["checks"] for r in main_rows
    )
    base_replies = {
        (r["model"], r["strategy"], r["scenario_id"]): r["reply"]
        for r in main_rows
        if not _is_tuned(r["model"])
    }

    labelled: list[tuple[str, int | None, Sequence[Mapping[str, Any]]]] = []
    for strategy in BASE_STRATEGIES:
        rows = [r for r in main_rows if not _is_tuned(r["model"]) and r["strategy"] == strategy]
        labelled.append((f"base {strategy}", None, rows))
    base_mismatches: dict[str, int] = {}
    for n, path in sorted(args.point):
        gens = _read_jsonl(path)
        base_mismatches[str(n)] = sum(
            base_replies.get((g["model"], g["strategy"], g["scenario_id"])) != g["reply"]
            for g in gens
            if not _is_tuned(g["model"])
        )
        rows = [
            {
                **g,
                "gold": scenarios[g["scenario_id"]].gold,
                "checks": checks_for(g["reply"], scenarios[g["scenario_id"]]),
            }
            for g in gens
            if _is_tuned(g["model"])
        ]
        labelled.append((f"tuned n{n}", n, rows))
    labelled.append(("tuned full v1", FULL_V1_N, [r for r in main_rows if _is_tuned(r["model"])]))

    out = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "transcript_checks_mismatches": mismatches,
        "curve_base_reply_mismatches": base_mismatches,
        "rows": [
            {
                "label": label,
                "N": n,
                **deterministic_metrics(rows, seed=args.seed, n_boot=args.n_boot),
            }
            for label, n, rows in labelled
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0
