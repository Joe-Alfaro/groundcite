"""Pure helpers for the prompt-ceiling ablation (R3, GOTCHA G1, G2, G12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# The reliability bar from the requirements: a system holds the behavior when both are met.
SPEC_ADHERENCE_BAR = 0.90
ROBUSTNESS_BAR = 4.3
# Means are ratios of small integers; this keeps 27/30 from missing 0.90 on float noise.
_EPS = 1e-9


def model_family(spec: str) -> str:
    """The vendor family of a backend spec: its prefix, with a bare ``org/repo`` meaning ``hf``."""
    if ":" in spec:
        return spec.split(":", 1)[0]
    return "hf"


def meets_bar(cell: Mapping[str, Any]) -> bool:
    return bool(
        cell["spec_adherence"]["mean"] >= SPEC_ADHERENCE_BAR - _EPS
        and cell["robustness_mean"] >= ROBUSTNESS_BAR - _EPS
    )


def plateau_verdict(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Plateau = no model x strategy cell meets the bar (inclusive)."""
    meeting = [{"model": c["model"], "strategy": c["strategy"]} for c in cells if meets_bar(c)]
    return {
        "plateau": not meeting,
        "cells_meeting_bar": meeting,
        "bar": {"spec_adherence": SPEC_ADHERENCE_BAR, "robustness": ROBUSTNESS_BAR},
    }


def strategy_deltas(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per model, the best and second-best strategy by mean Spec-adherence and their gap.

    Ties keep cell order. Models with a single strategy have no delta and are left out.
    """
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        by_model.setdefault(cell["model"], []).append(cell)
    deltas = {}
    for model, model_cells in by_model.items():
        if len(model_cells) < 2:
            continue
        best, second = sorted(model_cells, key=lambda c: -c["spec_adherence"]["mean"])[:2]
        deltas[model] = {
            "best": best["strategy"],
            "second": second["strategy"],
            "delta": best["spec_adherence"]["mean"] - second["spec_adherence"]["mean"],
        }
    return deltas
