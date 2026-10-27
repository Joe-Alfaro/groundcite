"""The per-cell bootstrap interval must not depend on which other cells are aggregated (D-039).

eval.py, the ablation and the curve all call `aggregate_scores()` on different mixes of cells, so
the same cell has to get the same CI wherever it is computed. The RNG is seeded per cell from
(seed, model, strategy), in a way that is stable across processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from groundcite.report import aggregate_scores


def _rows(model: str, strategy: str, specs: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "strategy": strategy,
            "category": "clean",
            "gold": None,
            "reply": "Check it [A-1].",
            "checks": {},
            "score": {"spec_adherence": s, "robustness": 5 if s else 2},
        }
        for s in specs
    ]


A = _rows("mock:a", "zero_shot", [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])
B = _rows("mock:b", "zero_shot", [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
C = _rows("mock:b", "structured", [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])


# Few resamples make the interval sensitive to which RNG draws a cell gets.
N_BOOT = 30


def _cell(cells: list[dict[str, Any]], model: str, strategy: str) -> dict[str, Any]:
    return next(c for c in cells if c["model"] == model and c["strategy"] == strategy)


def test_a_cell_ci_is_the_same_alone_or_after_other_cells() -> None:
    alone = _cell(aggregate_scores(B, seed=7, n_boot=N_BOOT), "mock:b", "zero_shot")
    after_a = _cell(aggregate_scores(A + B, seed=7, n_boot=N_BOOT), "mock:b", "zero_shot")
    after_a_c = _cell(aggregate_scores(C + A + B, seed=7, n_boot=N_BOOT), "mock:b", "zero_shot")
    assert alone["spec_adherence"] == after_a["spec_adherence"] == after_a_c["spec_adherence"]


def test_cell_order_does_not_change_any_cell() -> None:
    forward = aggregate_scores(A + B + C, seed=3, n_boot=N_BOOT)
    backward = aggregate_scores(C + B + A, seed=3, n_boot=N_BOOT)
    for model, strategy in [
        ("mock:a", "zero_shot"),
        ("mock:b", "zero_shot"),
        ("mock:b", "structured"),
    ]:
        assert _cell(forward, model, strategy) == _cell(backward, model, strategy)


def test_the_seed_still_matters() -> None:
    cis = {
        tuple(
            _cell(aggregate_scores(B, seed=s, n_boot=N_BOOT), "mock:b", "zero_shot")[
                "spec_adherence"
            ].values()
        )
        for s in range(6)
    }
    assert len(cis) > 1, "different seeds should give some different bootstrap intervals"


_SNIPPET = """
import json, sys
from groundcite.report import aggregate_scores
rows = json.loads(sys.stdin.read())
print(json.dumps(aggregate_scores(rows, seed=11, n_boot=30)))
"""


def test_cell_ci_is_stable_across_processes_and_hash_seeds() -> None:
    """Seeding must not use Python's randomized str hash()."""
    outputs = []
    for hash_seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        done = subprocess.run(
            [sys.executable, "-c", _SNIPPET],
            input=json.dumps(A + B + C),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=True,
        )
        outputs.append(json.loads(done.stdout))
    assert outputs[0] == outputs[1] == outputs[2]
