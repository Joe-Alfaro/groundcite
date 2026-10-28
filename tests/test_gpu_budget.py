"""The R31 GPU-hour budget table (R31, GOTCHA G28, D-078).

``docs/checklist.md`` or ``train/README.md`` holds a Markdown table with a ``Planned`` and a
``Used`` column (GPU-hours), one row per Kaggle run (at least v1, v2, the curve points and the
eval generation), and optionally a ``Total`` row. A line in the same file states the weekly Kaggle
GPU quota in hours (a line containing ``quota``, ``week`` and a number of hours, e.g.
"Weekly Kaggle GPU quota: 30 h"). The planned total leaves at least 25% slack: it is at most 75%
of that quota. Used cells are numbers or blank/``-`` until the run happens.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (REPO_ROOT / "docs" / "checklist.md", REPO_ROOT / "train" / "README.md")
NUMBER = re.compile(r"~?\s*(\d+(?:\.\d+)?)")
QUOTA = re.compile(
    r"(?i)^(?=.*quota)(?=.*week).*?(\d+(?:\.\d+)?)\s*(?:h\b|hrs?\b|hours|gpu[- ]hours)"
)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _budget() -> tuple[Path, list[str], list[list[str]], str]:
    """(file, header, body rows, file text) of the first table with Planned and Used columns."""
    for path in CANDIDATES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines[:-1]):
            if not line.lstrip().startswith("|") or not re.match(
                r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
            ):
                continue
            header = _cells(line)
            if any("planned" in h.lower() for h in header) and any(
                "used" in h.lower() for h in header
            ):
                body = []
                for row in lines[i + 2 :]:
                    if not row.lstrip().startswith("|"):
                        break
                    body.append(_cells(row))
                return path, header, body, text
    pytest.fail("no table with Planned and Used columns in docs/checklist.md or train/README.md")


def _column(header: list[str], word: str) -> int:
    return next(i for i, h in enumerate(header) if word in h.lower())


def _hours(cell: str) -> float | None:
    match = NUMBER.search(cell)
    return float(match.group(1)) if match else None


def test_budget_table_has_planned_and_used_columns_for_each_run() -> None:
    _, header, body, _ = _budget()
    runs = [r for r in body if not r[0].lower().startswith("total")]
    assert len(runs) >= 3
    names = " ".join(r[0].lower() for r in runs)
    for run in ("v1", "v2", "curve", "eval"):
        assert run in names, f"no budget row for {run}"
    planned, used = _column(header, "planned"), _column(header, "used")
    for row in runs:
        assert _hours(row[planned]) is not None, f"no planned hours: {row}"
        cell = row[used].strip()
        assert cell in {"", "-", "—", "–"} or _hours(cell) is not None, f"bad used cell: {row}"


def test_budget_states_the_weekly_kaggle_quota() -> None:
    _, _, _, text = _budget()
    assert any(QUOTA.search(line) for line in text.splitlines()), (
        "state the weekly Kaggle GPU quota in hours next to the table"
    )


def test_planned_hours_leave_at_least_25_percent_slack() -> None:
    _, header, body, text = _budget()
    quota = next(float(m.group(1)) for line in text.splitlines() if (m := QUOTA.search(line)))
    planned = _column(header, "planned")
    runs = [r for r in body if not r[0].lower().startswith("total")]
    total_rows = [r for r in body if r[0].lower().startswith("total")]
    hours = [_hours(r[planned]) or 0.0 for r in runs]
    total = _hours(total_rows[0][planned]) if total_rows else sum(hours)
    assert total is not None and total > 0
    if total_rows:
        assert abs(total - sum(hours)) < 0.05 + 1e-9, "the Total row must add up"
    assert total <= 0.75 * quota, f"planned {total} h exceeds 75% of the {quota} h weekly quota"


def test_checklist_r31_row_is_current() -> None:
    checklist = (REPO_ROOT / "docs" / "checklist.md").read_text(encoding="utf-8")
    row = next(line for line in checklist.splitlines() if line.startswith("| R31 "))
    assert "(planned)" not in row and "not started" not in row
