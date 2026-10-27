"""Tests for the nested data-efficiency subsets (R8, R14, GOTCHA G29).

`scripts/make_sweep.py --dataset DIR --sizes N1 N2 ... --seed S` shuffles `DIR/train.jsonl`
once (seeded, stratified by `meta.category`) and writes each size as a prefix of that order to
`DIR/subsets/n<N>.jsonl`, plus `DIR/subsets/sweep_manifest.json`. So N/8 ⊂ N/4 ⊂ N/2 ⊂ N.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from distill_data import CATEGORIES, run_script, write_jsonl

QUOTAS = {
    "clean": 15,
    "distractor": 20,
    "no_snippet": 20,
    "pressure": 15,
    "multi_symptom": 15,
    "multi_turn": 15,
}
POOL_SIZE = 320
SIZES = (320, 160, 80, 40)
# A prefix of a stratified order can be off by about one row per category, never more.
STRATA_TOLERANCE = 1.5


def _record(i: int, category: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": f"complaint number {i}"},
            {"role": "assistant", "content": f"Measure the reading at point {i} [TSB-{i:04d}]."},
        ],
        "meta": {"scenario_id": f"tr-{i:04d}", "category": category, "family": "lean_idle"},
    }


def _dataset(tmp_path: Path, name: str = "v1") -> Path:
    cats = [c for c in CATEGORIES for _ in range(POOL_SIZE * QUOTAS[c] // 100)]
    assert len(cats) == POOL_SIZE
    # Grouped by category on disk, so an unshuffled prefix would be badly unstratified.
    ds = tmp_path / name
    write_jsonl(ds / "train.jsonl", [_record(i, c) for i, c in enumerate(cats)])
    return ds


def _sweep(ds: Path, *sizes: int, seed: int = 7) -> int:
    return run_script(
        "make_sweep",
        ["--dataset", str(ds), "--sizes", *(str(n) for n in sizes), "--seed", str(seed)],
    )


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def swept(tmp_path: Path) -> Path:
    ds = _dataset(tmp_path)
    assert _sweep(ds, *SIZES) == 0
    return ds


def test_writes_one_file_per_size_with_exactly_n_rows(swept: Path) -> None:
    for n in SIZES:
        path = swept / "subsets" / f"n{n}.jsonl"
        assert path.is_file(), f"{path} is missing"
        assert len(_lines(path)) == n


def test_subsets_are_nested_prefixes(swept: Path) -> None:
    ordered = sorted(SIZES)
    for small, large in zip(ordered, ordered[1:], strict=False):
        a = _lines(swept / "subsets" / f"n{small}.jsonl")
        b = _lines(swept / "subsets" / f"n{large}.jsonl")
        assert b[:small] == a, f"n{small} is not a prefix of n{large}"
        assert set(a) <= set(b)


def test_rows_are_unchanged_pool_lines_without_duplicates(swept: Path) -> None:
    pool = set(_lines(swept / "train.jsonl"))
    for n in SIZES:
        lines = _lines(swept / "subsets" / f"n{n}.jsonl")
        assert len(set(lines)) == len(lines)
        assert set(lines) <= pool


def test_every_subset_is_stratified_by_category(swept: Path) -> None:
    for n in SIZES:
        rows = [json.loads(line) for line in _lines(swept / "subsets" / f"n{n}.jsonl")]
        counts = Counter(r["meta"]["category"] for r in rows)
        for category, quota in QUOTAS.items():
            expected = n * quota / 100
            assert abs(counts[category] - expected) <= STRATA_TOLERANCE, (
                f"n{n}: {category} has {counts[category]}, expected about {expected}"
            )


def test_the_order_is_shuffled_not_the_file_head(swept: Path) -> None:
    head = _lines(swept / "train.jsonl")[:40]
    assert _lines(swept / "subsets" / "n40.jsonl") != head


def test_same_seed_same_bytes_other_seed_other_order(tmp_path: Path) -> None:
    a, b, c = _dataset(tmp_path, "a"), _dataset(tmp_path, "b"), _dataset(tmp_path, "c")
    assert _sweep(a, *SIZES, seed=7) == 0
    assert _sweep(b, *SIZES, seed=7) == 0
    assert _sweep(c, *SIZES, seed=8) == 0
    for n in SIZES:
        name = f"subsets/n{n}.jsonl"
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert (a / "subsets/n40.jsonl").read_bytes() != (c / "subsets/n40.jsonl").read_bytes()


def test_sweep_manifest_records_seed_pool_hash_and_subset_hashes(swept: Path) -> None:
    m = json.loads((swept / "subsets" / "sweep_manifest.json").read_text(encoding="utf-8"))
    assert m["seed"] == 7
    assert m["pool_sha256"] == hashlib.sha256((swept / "train.jsonl").read_bytes()).hexdigest()
    assert sorted(m["sizes"]) == sorted(SIZES)
    for n in SIZES:
        entry = m["subsets"][f"n{n}"]
        data = (swept / "subsets" / f"n{n}.jsonl").read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["count"] == n
        assert sum(entry["counts_by_category"].values()) == n


def test_size_larger_than_the_pool_exits_two(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    assert _sweep(ds, POOL_SIZE + 1, 40) == 2
    assert not (ds / "subsets").exists() or not any((ds / "subsets").iterdir())


def test_missing_dataset_exits_two(tmp_path: Path) -> None:
    assert _sweep(tmp_path / "nope", 40) == 2
