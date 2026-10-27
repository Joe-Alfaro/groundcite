"""Tests for the training scenario pool (R4, R22, R23, R30, GOTCHA G23, G24).

`scripts/gen_train_scenarios.py [--seed S] [--n N] [--out PATH]` writes the pool the teacher
distills from. Its defaults (`groundcite.scenarios.TRAIN_SEED`, `TRAIN_POOL_SIZE` and
`data/scenarios/train_pool.jsonl`) reproduce the committed pool byte for byte.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from distill_data import EVAL_SETS, FEW_SHOT, RESERVED_FAMILIES, id_scheme, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_TRAIN = REPO_ROOT / "scripts" / "gen_train_scenarios.py"
CHECK_LEAKAGE = REPO_ROOT / "scripts" / "check_leakage.py"
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"
POOL = REPO_ROOT / "data" / "scenarios" / "train_pool.jsonl"
PHRASING = REPO_ROOT / "data" / "phrasing.toml"

QUOTAS = {
    "clean": 15,
    "distractor": 20,
    "no_snippet": 20,
    "pressure": 15,
    "multi_symptom": 15,
    "multi_turn": 15,
}
QUOTA_TOLERANCE = 5
MIN_POOL = 400
# User-turn templates the eval sets draw pressure and multi-turn phrasing from. Training data
# must use its own phrasings, so a model can't score by recognizing a memorized line.
EVAL_USER_TEMPLATE_KEYS = (
    "pressure",
    "pressure_lead",
    "final_pressure",
    "final_ask",
    "user_pushback",
    "user_relent",
)


def _gen(*args: str) -> subprocess.CompletedProcess[str]:
    assert GEN_TRAIN.is_file(), f"{GEN_TRAIN.relative_to(REPO_ROOT)} is missing"
    return subprocess.run(
        [sys.executable, str(GEN_TRAIN), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def _pool() -> list[dict[str, Any]]:
    assert POOL.is_file(), f"committed training pool {POOL.relative_to(REPO_ROOT)} is missing"
    rows = read_jsonl(POOL)
    assert rows, f"{POOL.relative_to(REPO_ROOT)} is empty"
    return rows


def _eval_rows() -> list[dict[str, Any]]:
    return [row for path in EVAL_SETS for row in read_jsonl(path)]


def _reserved() -> set[str]:
    lines = RESERVED_FAMILIES.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _template_regex(template: str) -> re.Pattern[str]:
    """A pattern matching the template with any text in each ``{placeholder}``."""
    parts = re.split(r"\{\w+\}", _normalize(template))
    return re.compile(".+?".join(re.escape(p) for p in parts))


def _copies_template(text: str, template: str) -> bool:
    norm, tmpl = _normalize(text), _normalize(template)
    if not tmpl:
        return False
    if "{" not in tmpl and len(tmpl.split()) < 3:
        # Very short lines ("What now?") only count as a copy when they close the message.
        return norm.endswith(tmpl)
    return _template_regex(template).search(norm) is not None


# --- generator -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("train_pool") / "train_pool.jsonl"
    result = _gen("--out", str(out))
    assert result.returncode == 0, result.stderr
    return out


def test_defaults_reproduce_the_committed_pool_byte_for_byte(default_run: Path) -> None:
    assert default_run.read_bytes() == POOL.read_bytes()


def test_same_seed_same_bytes_and_seed_changes_output(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    for out, seed in ((a, "4401"), (b, "4401"), (c, "4402")):
        result = _gen("--seed", seed, "--n", "60", "--out", str(out))
        assert result.returncode == 0, result.stderr
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    assert len(read_jsonl(a)) == 60


def test_train_seed_is_separate_from_the_eval_seed() -> None:
    from groundcite.scenarios import TRAIN_POOL_SIZE, TRAIN_SEED

    eval_masters = {row["seed"] // 1_000_000 for row in _eval_rows()}
    assert TRAIN_SEED not in eval_masters
    assert TRAIN_POOL_SIZE >= MIN_POOL
    eval_seeds = {row["seed"] for row in _eval_rows()}
    assert not eval_seeds & {row["seed"] for row in _pool()}


# --- the committed pool ----------------------------------------------------------------------


def test_pool_size_split_and_unique_ids() -> None:
    rows = _pool()
    assert len(rows) >= MIN_POOL
    ids = [row["id"] for row in rows]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("tr-") for i in ids)
    assert {row["split"] for row in rows} == {"train"}
    assert not set(ids) & {row["id"] for row in _eval_rows()}


def test_pool_uses_only_training_families() -> None:
    families = {row["family"] for row in _pool()}
    assert families, "pool rows carry no family"
    assert not families & _reserved(), f"reserved families in pool: {families & _reserved()}"


def test_pool_meets_the_category_mix() -> None:
    rows = _pool()
    counts = Counter(row["category"] for row in rows)
    assert set(counts) == set(QUOTAS)
    for category, quota in QUOTAS.items():
        share = 100 * counts[category] / len(rows)
        assert abs(share - quota) <= QUOTA_TOLERANCE, f"{category}: {share:.1f}% vs {quota}%"


def test_pool_abstain_share_is_twenty_to_thirty_percent() -> None:
    rows = _pool()
    share = sum(row["gold"]["mode"] == "abstain" for row in rows) / len(rows)
    assert 0.20 <= share <= 0.30, f"abstain share {share:.1%} is outside 20-30% (R30)"


def test_pool_uses_at_least_three_id_schemes() -> None:
    schemes = {id_scheme(s["id"]) for row in _pool() for s in row["snippets"]}
    assert len(schemes) >= 3, sorted(schemes)


def test_pool_ids_are_reassigned_per_scenario() -> None:
    ids_by_text: dict[str, set[str]] = defaultdict(set)
    uses: Counter[str] = Counter()
    for row in _pool():
        for snippet in row["snippets"]:
            ids_by_text[snippet["text"]].add(snippet["id"])
            uses[snippet["text"]] += 1
    reused = [text for text, n in uses.items() if n >= 2]
    assert reused
    varied = [text for text in reused if len(ids_by_text[text]) >= 2]
    assert len(varied) / len(reused) >= 0.9


def test_pool_supporting_snippet_position_varies() -> None:
    positions: Counter[int] = Counter()
    for row in _pool():
        if row["gold"]["mode"] == "cite":
            ids = [s["id"] for s in row["snippets"]]
            positions[ids.index(row["gold"]["supporting_ids"][0])] += 1
    total = sum(positions.values())
    assert len(positions) >= 3
    assert max(positions.values()) / total <= 0.6


# --- leakage ---------------------------------------------------------------------------------


def test_pool_has_no_leakage_against_eval_sets_and_few_shot(tmp_path: Path) -> None:
    assert FEW_SHOT.is_file(), f"{FEW_SHOT.relative_to(REPO_ROOT)} is missing"
    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_LEAKAGE),
            "--train",
            str(POOL),
            "--eval",
            *(str(p) for p in EVAL_SETS),
            "--few-shot",
            str(FEW_SHOT),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr


def test_pool_user_turns_never_copy_an_eval_pressure_or_multi_turn_template() -> None:
    """Validator note: templated pressure lines recurred between sets. Training phrasing for
    pressure and multi-turn user turns must not reuse any eval template, placeholders aside."""
    phrasing = tomllib.loads(PHRASING.read_text(encoding="utf-8"))
    templates = [(key, t) for key in EVAL_USER_TEMPLATE_KEYS for t in phrasing[key] if t.strip()]
    assert templates
    copies = []
    for row in _pool():
        for message in row["messages"]:
            if message["role"] != "user":
                continue
            for key, template in templates:
                if _copies_template(message["content"], template):
                    copies.append((row["id"], key, template))
    assert not copies, f"{len(copies)} copied eval templates, e.g. {copies[:5]}"


def test_pool_final_user_messages_never_equal_an_eval_final_message() -> None:
    eval_finals = {
        _normalize([m for m in row["messages"] if m["role"] == "user"][-1]["content"])
        for row in _eval_rows()
    }
    same = [
        row["id"]
        for row in _pool()
        if _normalize([m for m in row["messages"] if m["role"] == "user"][-1]["content"])
        in eval_finals
    ]
    assert not same


def test_template_matcher_flags_a_filled_copy_and_spares_a_new_line() -> None:
    template = "Is it the {part}? A simple yes or no is fine."
    assert _copies_template("2.0L sedan. Is it the PCV hose? A simple yes or no is fine.", template)
    assert not _copies_template("Is it the PCV hose? Just a yes or a no, please.", template)
    assert _copies_template("Done, 13.9 volts. What now?", "What now?")
    assert not _copies_template("What now? I checked it.", "What now?")


def test_quality_gate_checks_the_pool_against_every_eval_set_and_few_shot() -> None:
    text = CHECK_SH.read_text(encoding="utf-8").replace("\\\n", " ")
    commands = [line for line in text.splitlines() if "check_leakage.py" in line]
    pool_cmds = [c for c in commands if "train_pool.jsonl" in c]
    assert pool_cmds, "check.sh never runs check_leakage.py on data/scenarios/train_pool.jsonl"
    for cmd in pool_cmds:
        for name in ("ablation.jsonl", "eval_own.jsonl", "heldout.jsonl", "--few-shot"):
            assert name in cmd, f"pool leakage command lacks {name}: {cmd.strip()}"


def test_pool_rows_load_as_eval_scenarios() -> None:
    from groundcite.evalset import parse_row

    for row in _pool()[:50]:
        scenario = parse_row(json.loads(json.dumps(row)))
        assert scenario.gold is not None
