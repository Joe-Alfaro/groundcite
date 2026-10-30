"""The committed data-efficiency subsets of dataset v1 and the frozen config (D-126, R8, G29).

``data/datasets/v1/subsets/`` holds n56, n112 and n224 from
``scripts/make_sweep.py --dataset data/datasets/v1 --sizes 56 112 224 448 --seed 7`` (n448 is
the full v1 pool, already trained, so it is not committed). Every committed file matches its
manifest hash and count, each is a line prefix of the next, every line is a v1 line, each
category is within about one row of its v1 share, and the manifest's pool is the dataset the v1
run trained on. Every curve point trains with the same config: ``train/config.toml`` still
resolves to the v1 run's ``config_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from train_data import REPO_ROOT, load_train_module

V1 = REPO_ROOT / "data" / "datasets" / "v1"
SUBSETS = V1 / "subsets"
V1_TRAIN_LOG = REPO_ROOT / "results" / "training" / "v1" / "train_log.json"
CURVE_SIZES = (56, 112, 224)
SWEEP_SEED = 7
STRATA_TOLERANCE = 1.5


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return dict(json.loads((SUBSETS / "sweep_manifest.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def v1_log() -> dict[str, Any]:
    return dict(json.loads(V1_TRAIN_LOG.read_text(encoding="utf-8")))


def test_the_manifest_records_the_sweep_seed_and_the_trained_v1_pool(
    manifest: dict[str, Any], v1_log: dict[str, Any]
) -> None:
    assert manifest["seed"] == SWEEP_SEED
    pool = (V1 / "train.jsonl").read_bytes()
    assert manifest["pool_sha256"] == hashlib.sha256(pool).hexdigest()
    assert manifest["pool_sha256"] == v1_log["dataset_sha256"]
    assert manifest["pool_count"] == v1_log["trained_examples"]


@pytest.mark.parametrize("n", CURVE_SIZES)
def test_each_subset_matches_its_manifest_entry(manifest: dict[str, Any], n: int) -> None:
    path = SUBSETS / f"n{n}.jsonl"
    entry = manifest["subsets"][f"n{n}"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    lines = _lines(path)
    assert len(lines) == entry["count"] == n
    cats = Counter(json.loads(line)["meta"]["category"] for line in lines)
    assert dict(sorted(cats.items())) == entry["counts_by_category"]


def test_the_subsets_are_nested_prefixes_of_v1_lines() -> None:
    pool = _lines(V1 / "train.jsonl")
    subsets = [_lines(SUBSETS / f"n{n}.jsonl") for n in CURVE_SIZES]
    for small, large in zip(subsets, subsets[1:], strict=False):
        assert large[: len(small)] == small
    largest = subsets[-1]
    assert len(set(largest)) == len(largest), "no duplicate lines"
    assert set(largest) <= set(pool)


@pytest.mark.parametrize("n", CURVE_SIZES)
def test_each_subset_is_category_stratified(n: int) -> None:
    pool = Counter(json.loads(line)["meta"]["category"] for line in _lines(V1 / "train.jsonl"))
    total = sum(pool.values())
    cats = Counter(json.loads(line)["meta"]["category"] for line in _lines(SUBSETS / f"n{n}.jsonl"))
    for cat, count in pool.items():
        assert abs(cats[cat] - n * count / total) <= STRATA_TOLERANCE, cat


def test_the_config_is_frozen_at_the_v1_run_hash(v1_log: dict[str, Any]) -> None:
    """G29: every curve point trains with the config the full v1 run used."""
    train = load_train_module()
    assert train.config_sha256(train.resolve_config()) == v1_log["config_sha256"]
