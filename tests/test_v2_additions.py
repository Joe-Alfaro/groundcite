"""Dataset v2 additions: in-scope cite twins of the v1 abstain scenarios (R7, G21, D-056, D-061).

``results/base_vs_tuned/heldout/DIAGNOSIS.md`` found the tuned model abstains by template: every
v1 abstain example rejects an on-topic snippet for its scope, and no v1 example cites that same
snippet for a vehicle inside its scope. The v2 data change adds, for each v1 abstain scenario,
its in-scope counterpart (a minimal pair):

    uv run scripts/gen_v2_additions.py [--seed S] [--out PATH]

writes ``data/scenarios/train_v2_additions.jsonl`` deterministically (the defaults reproduce the
committed file byte for byte). Each row is a training scenario in the usual row shape, plus
``source_id``: the v1 abstain scenario (``train_pool.jsonl``) it is the twin of. A twin keeps its
source's family, snippet texts and conversation shape, and changes the vehicle (and/or model
year) so that exactly one provided on-topic snippet now applies: the gold, cited alone.
A ``no_snippet`` source becomes a ``distractor`` twin (the other near-misses stay provided);
``pressure``, ``multi_symptom`` and ``multi_turn`` keep their category.

Only the scenarios are built here. Teacher replies are a paid step (about N x 2 calls) and stay
on hold, so this module also pins what that step will cost: ``distill.py --dry-run`` on the
additions plans exactly N x k calls and makes no request. ``docs/dataset-v2-plan.md`` records
the plan: dataset v2 = v1 (448 examples, unchanged) + the kept additions, with the training
config frozen at the v1 run's hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import test_kind_balance
import test_scenario_chains
import test_scenarios
import test_shortcuts
import test_twin_priors
import test_twins
from distill_data import (
    EVAL_SETS,
    FEW_SHOT,
    RESERVED_FAMILIES,
    cite_reply,
    run_script,
    write_jsonl,
)
from test_kind_balance import _first_user_year
from test_scenario_coherence import _vehicle_of
from train_data import load_train_module

from groundcite.leakage import (
    JACCARD_THRESHOLD,
    find_leaks,
    jaccard,
    load_rows,
    text_hash,
    word_ngrams,
)
from groundcite.scenarios import load_library, load_reserved

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
SET = "train_v2_additions"
ADDITIONS = SCENARIO_DIR / f"{SET}.jsonl"
TRAIN_POOL = SCENARIO_DIR / "train_pool.jsonl"
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_v2_additions.py"
V1 = REPO_ROOT / "data" / "datasets" / "v1"
V1_TRAIN_LOG = REPO_ROOT / "results" / "training" / "v1" / "train_log.json"
PLAN_DOC = REPO_ROOT / "docs" / "dataset-v2-plan.md"
FAMILIES_TOML = REPO_ROOT / "data" / "families.toml"

# Every one of the 126 v1 abstain scenarios is kept in dataset v1; about 115 of them have a
# vehicle in the catalog that brings exactly one provided snippet into scope with the same
# snippet list. A twin per source at most, and at least 110 of them.
V1_COUNT = 448
V1_ABSTAIN = 126
# D-138: lowered from 110 to 105; 21 of the 126 sources can't yield a twin that passes the gates.
MIN_ADDITIONS = 105
# v1 distilled two candidates per scenario (960 for 480); v2 keeps the same k.
K = 2
ABSTAIN_SHARE_RANGE = (0.20, 0.30)
# A per-scenario keep rate this low would still leave v2 inside the abstain-share range.
LOW_KEEP_RATE = 0.5
CITATION = re.compile(r"\[([^\[\]]+)\]")


# --- loading ----------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} is missing"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def additions() -> list[dict[str, Any]]:
    rows = _jsonl(ADDITIONS)
    assert rows, f"{ADDITIONS.relative_to(REPO_ROOT)} is empty"
    return rows


@pytest.fixture(scope="module")
def pool() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _jsonl(TRAIN_POOL)}


@pytest.fixture(scope="module")
def v1_manifest() -> dict[str, Any]:
    return dict(json.loads((V1 / "manifest.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def vehicles() -> list[dict[str, Any]]:
    return list(tomllib.loads(FAMILIES_TOML.read_text(encoding="utf-8"))["vehicle"])


@pytest.fixture(scope="module")
def library() -> Any:
    return load_library()


def _first_user(row: dict[str, Any]) -> str:
    return str(next(m["content"] for m in row["messages"] if m["role"] == "user"))


def _final(row: dict[str, Any]) -> str:
    return str(row["messages"][-1]["content"])


def _cited_earlier(row: dict[str, Any]) -> set[str]:
    """Scenario snippet ids an earlier assistant turn already cited (multi-turn)."""
    return {
        found
        for m in row["messages"]
        if m["role"] == "assistant"
        for found in CITATION.findall(m["content"])
    }


def _lib_vehicle(row: dict[str, Any], vehicles: list[dict[str, Any]], library: Any) -> Any:
    """The library ``Vehicle`` the first user message names, or ``None``."""
    named = _vehicle_of(row, vehicles)
    if named is None:
        return None
    return next(
        (
            v
            for v in library.vehicles
            if (v.body, v.engine, v.trans) == (named["body"], named["engine"], named["trans"])
        ),
        None,
    )


def _on_topic_families(row: dict[str, Any], library: Any) -> set[str]:
    """The row's family plus the families of its gold and near-miss snippets."""
    by_text = {s.text: s for s in library.snippets.values()}
    by_id = {s["id"]: by_text.get(s["text"]) for s in row["snippets"]}
    ids = row["gold"]["supporting_ids"] + row["gold"]["near_miss_ids"]
    return {row["family"]} | {by_id[i].family for i in ids if by_id.get(i) is not None}


def _applying(row: dict[str, Any], vehicles: list[dict[str, Any]], library: Any) -> list[str]:
    """Provided on-topic snippets that are a valid next step for the stated vehicle and year:
    in scope, not a condition near-miss (the complaint contradicts those) and not already cited
    in an earlier turn."""
    vehicle, year = _lib_vehicle(row, vehicles, library), _first_user_year(row)
    assert vehicle is not None and year is not None, f"{row['id']}: no catalog vehicle and year"
    by_text = {s.text: s for s in library.snippets.values()}
    families = _on_topic_families(row, library)
    earlier = _cited_earlier(row)
    out = []
    for snippet in row["snippets"]:
        lib = by_text.get(snippet["text"])
        assert lib is not None, f"{row['id']}: [{snippet['id']}] is not a library snippet"
        if (
            lib.family in families
            and lib.mismatch != "condition"
            and snippet["id"] not in earlier
            and lib.applies.matches(vehicle, year)
        ):
            out.append(snippet["id"])
    return out


# --- (1) the generator: deterministic, committed, one twin per v1 abstain scenario -----------


def _run_gen(out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    assert GEN_SCRIPT.is_file(), f"{GEN_SCRIPT.relative_to(REPO_ROOT)} is missing"
    return subprocess.run(
        [sys.executable, str(GEN_SCRIPT), "--out", str(out), *extra],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def test_generator_reproduces_the_committed_additions_byte_for_byte(tmp_path: Path) -> None:
    out = tmp_path / "additions.jsonl"
    result = _run_gen(out)
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == ADDITIONS.read_bytes(), "defaults must reproduce the committed file"


def test_generator_output_depends_only_on_its_seed(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    for out, seed in ((a, "11"), (b, "11"), (c, "12")):
        result = _run_gen(out, "--seed", seed)
        assert result.returncode == 0, result.stderr
    assert a.read_bytes() == b.read_bytes(), "same seed, different bytes"
    assert a.read_bytes() != c.read_bytes(), "the seed changes nothing"


def test_about_one_twin_per_v1_abstain_scenario(
    additions: list[dict[str, Any]],
    pool: dict[str, dict[str, Any]],
    v1_manifest: dict[str, Any],
) -> None:
    v1_seeds = set(v1_manifest["source_seeds"])
    v1_abstain = {
        sid
        for sid, row in pool.items()
        if row["gold"]["mode"] == "abstain" and row["seed"] in v1_seeds
    }
    assert len(v1_abstain) == V1_ABSTAIN, "the v1 abstain scenarios moved"
    sources = [row.get("source_id") for row in additions]
    assert MIN_ADDITIONS <= len(additions) <= V1_ABSTAIN, f"{len(additions)} additions"
    assert len(set(sources)) == len(sources), "a v1 abstain scenario has two twins"
    stray = sorted(str(s) for s in set(sources) - v1_abstain)
    assert not stray, f"source_id is not a v1 abstain scenario: {stray[:10]}"


def test_ids_and_seeds_are_new_and_unique(
    additions: list[dict[str, Any]], pool: dict[str, dict[str, Any]]
) -> None:
    ids = [row["id"] for row in additions]
    seeds = [row["seed"] for row in additions]
    assert len(set(ids)) == len(ids), "duplicate scenario ids"
    assert all(isinstance(s, int) for s in seeds) and len(set(seeds)) == len(seeds)
    taken_ids, taken_seeds = set(pool), {row["seed"] for row in pool.values()}
    for path in EVAL_SETS:
        for row in _jsonl(path):
            taken_ids.add(row["id"])
            taken_seeds.add(row["seed"])
    assert not set(ids) & taken_ids, f"ids reused: {sorted(set(ids) & taken_ids)[:10]}"
    assert not set(seeds) & taken_seeds, f"seeds reused: {sorted(set(seeds) & taken_seeds)[:10]}"
    assert {row["split"] for row in additions} == {"train"}


def test_twins_keep_the_source_family_and_use_no_reserved_family(
    additions: list[dict[str, Any]], pool: dict[str, dict[str, Any]]
) -> None:
    reserved = load_reserved(RESERVED_FAMILIES)
    for row in additions:
        source = pool[row["source_id"]]
        assert row["family"] == source["family"], row["id"]
        assert row["family"] not in reserved, row["id"]


def test_twins_keep_the_source_snippets_and_conversation_shape(
    additions: list[dict[str, Any]], pool: dict[str, dict[str, Any]]
) -> None:
    """A minimal pair: the snippet that was out of scope in the source is the one cited here."""
    for row in additions:
        source = pool[row["source_id"]]
        assert sorted(s["text"] for s in row["snippets"]) == sorted(
            s["text"] for s in source["snippets"]
        ), f"{row['id']}: snippet texts differ from {source['id']}"
        roles = [m["role"] for m in row["messages"]]
        assert roles == [m["role"] for m in source["messages"]], row["id"]


def test_twins_change_the_vehicle_or_year(
    additions: list[dict[str, Any]],
    pool: dict[str, dict[str, Any]],
    vehicles: list[dict[str, Any]],
) -> None:
    for row in additions:
        source = pool[row["source_id"]]
        new = (_vehicle_of(row, vehicles), _first_user_year(row))
        old = (_vehicle_of(source, vehicles), _first_user_year(source))
        assert None not in new, f"{row['id']}: no catalog vehicle and year stated"
        assert new != old, f"{row['id']}: same vehicle and year as {source['id']}"


def test_categories_and_tags_follow_the_cite_mode(
    additions: list[dict[str, Any]], pool: dict[str, dict[str, Any]]
) -> None:
    for row in additions:
        source = pool[row["source_id"]]
        expected = "distractor" if source["category"] == "no_snippet" else source["category"]
        assert row["category"] == expected, f"{row['id']}: {row['category']} for {expected}"
        assert "no_snippet" not in row["tags"], row["id"]


# --- (2) gold integrity: the gold is the one snippet that now applies -------------------------


def test_every_twin_cites_exactly_one_snippet(additions: list[dict[str, Any]]) -> None:
    for row in additions:
        gold = row["gold"]
        provided = {s["id"] for s in row["snippets"]}
        assert gold["mode"] == "cite", row["id"]
        assert len(gold["supporting_ids"]) == 1, row["id"]
        assert set(gold["supporting_ids"]) <= provided, row["id"]
        assert set(gold["near_miss_ids"]) <= provided, row["id"]
        assert not set(gold["supporting_ids"]) & set(gold["near_miss_ids"]), row["id"]


def test_the_gold_is_the_only_provided_snippet_that_applies(
    additions: list[dict[str, Any]], vehicles: list[dict[str, Any]], library: Any
) -> None:
    bad = {}
    for row in additions:
        applying = _applying(row, vehicles, library)
        if applying != row["gold"]["supporting_ids"]:
            bad[row["id"]] = {"applying": applying, "gold": row["gold"]["supporting_ids"]}
    assert not bad, f"{len(bad)} twins whose gold is not the one applying snippet: {bad}"


def test_the_gold_was_out_of_scope_in_the_source(
    additions: list[dict[str, Any]],
    pool: dict[str, dict[str, Any]],
    vehicles: list[dict[str, Any]],
    library: Any,
) -> None:
    """The source rejected the very snippet the twin cites, so the pair isolates the scope."""
    for row in additions:
        source = pool[row["source_id"]]
        gold_text = next(
            s["text"] for s in row["snippets"] if s["id"] == row["gold"]["supporting_ids"][0]
        )
        assert _applying(source, vehicles, library) == [], f"{source['id']} is not an abstain"
        assert gold_text in {s["text"] for s in source["snippets"]}, row["id"]


def test_a_provided_twin_of_the_gold_is_a_near_miss(
    additions: list[dict[str, Any]], library: Any
) -> None:
    by_text = {s.text: s for s in library.snippets.values()}
    for row in additions:
        lib = {s["id"]: by_text[s["text"]] for s in row["snippets"]}
        gold = lib[row["gold"]["supporting_ids"][0]]
        partners = [
            sid
            for sid, s in lib.items()
            if s is not gold and (s.twin_of == gold.id or gold.twin_of == s.id)
        ]
        missing = [sid for sid in partners if sid not in row["gold"]["near_miss_ids"]]
        assert not missing, f"{row['id']}: the gold's twin {missing} is not a near-miss"


# --- (3) the existing gates, run on the additions --------------------------------------------


@pytest.mark.parametrize(
    "gate",
    [
        "test_gold_labels_are_consistent",
        "test_category_implies_gold_mode",
        "test_two_to_five_snippets_with_valid_unique_ids",
        "test_messages_alternate_and_end_on_user",
        "test_no_manufacturer_names",
    ],
)
def test_scenario_gates_hold(gate: str) -> None:
    getattr(test_scenarios, gate)(SET)


@pytest.mark.parametrize(
    ("module", "gate"),
    [
        (test_twins, "test_every_scenario_near_miss_is_ruled_out_by_its_scope_change"),
        (test_twin_priors, "test_every_fact_that_decides_gold_is_stated"),
        (test_twin_priors, "test_transmission_is_stated_whenever_a_snippet_is_transmission_scoped"),
        (test_shortcuts, "test_multi_turn_cite_leaves_no_documented_check_unperformed"),
    ],
)
def test_twin_and_gold_integrity_gates_hold(module: Any, gate: str) -> None:
    getattr(module, gate)(SET)


def test_multi_turn_gold_is_the_next_unchecked_step(additions: list[dict[str, Any]]) -> None:
    library = {
        raw["id"]: raw
        for path in sorted((REPO_ROOT / "data" / "snippets").glob("*.jsonl"))
        for raw in _jsonl(path)
    }
    by_text = {(raw["title"], raw["text"]): sid for sid, raw in library.items()}
    families = {
        fam["name"]: fam
        for fam in tomllib.loads(FAMILIES_TOML.read_text(encoding="utf-8"))["family"]
    }
    test_scenario_chains.test_multi_turn_gold_is_next_unchecked_step(
        SET, {SET: additions}, families, by_text
    )


@pytest.mark.parametrize("heuristic", test_shortcuts.GATED_HEURISTICS)
def test_no_trivial_heuristic_solves_the_additions(heuristic: str) -> None:
    test_shortcuts.test_no_trivial_heuristic_solves_distractors(SET, heuristic)


def test_median_year_within_a_pair_does_not_solve_the_additions() -> None:
    test_twin_priors.test_median_year_within_a_pair_does_not_solve_distractors(SET)


def test_surface_probe_does_not_solve_the_additions() -> None:
    test_twins.test_surface_probe_does_not_solve_distractors(SET)


def _pooled_distractors() -> list[dict[str, Any]]:
    """train_pool distractors plus the additions' distractor twins: the v2 scenario pool."""
    return test_shortcuts._distractor_rows("train_pool") + test_shortcuts._distractor_rows(SET)


def test_surface_probe_does_not_solve_the_v2_pool() -> None:
    rows = _pooled_distractors()
    accuracy = test_twins.probe_loso_accuracy(rows)
    print(f"v2 pool: surface probe (leave-one-scenario-out): {accuracy:.2f} on {len(rows)}")
    assert accuracy <= test_twins.MAX_PROBE_ACCURACY + test_shortcuts.EPS, accuracy


def test_surface_probe_fit_on_the_v2_pool_does_not_solve_eval_distractors() -> None:
    evals = [
        row for name in ("eval_own", "heldout") for row in test_shortcuts._distractor_rows(name)
    ]
    accuracy = test_twins.probe_transfer_accuracy(_pooled_distractors(), evals)
    print(f"surface probe fit on the v2 pool: {accuracy:.2f} on {len(evals)} eval distractors")
    assert accuracy <= test_twins.MAX_PROBE_ACCURACY + test_shortcuts.EPS, accuracy


@pytest.mark.parametrize("heuristic", test_shortcuts.GATED_HEURISTICS)
def test_no_trivial_heuristic_solves_the_v2_pool(heuristic: str) -> None:
    rows = _pooled_distractors()
    accuracy = test_shortcuts._heuristic_accuracy(rows, test_shortcuts.GATED_HEURISTICS[heuristic])
    assert accuracy <= test_shortcuts.MAX_HEURISTIC_ACCURACY + test_shortcuts.EPS, (
        f"v2 pool: '{heuristic}' picks the gold on {accuracy:.0%} of {len(rows)} distractors"
    )


def test_gold_kind_stays_balanced_in_the_v2_pool(additions: list[dict[str, Any]]) -> None:
    kinds: Counter[str] = Counter()
    for row in test_kind_balance._cite_rows("train_pool") + additions:
        by_id = {s["id"]: s for s in row["snippets"]}
        kinds.update(by_id[g]["kind"] for g in row["gold"]["supporting_ids"])
    share = kinds["tsb"] / kinds.total()
    assert test_kind_balance.BALANCE_LOW <= share <= test_kind_balance.BALANCE_HIGH, kinds


# --- (4) leakage: nothing overlaps the eval sets, and no twin is a duplicate final -----------


def test_additions_pass_the_leakage_rules() -> None:
    evals = [row for path in EVAL_SETS for row in load_rows(path)]
    findings = find_leaks(
        load_rows(ADDITIONS), evals, load_rows(FEW_SHOT), load_reserved(RESERVED_FAMILIES)
    )
    assert not findings, [str(f) for f in findings[:10]]


def test_leakage_script_passes_on_the_additions(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--train", str(ADDITIONS), "--eval", *map(str, EVAL_SETS), "--few-shot", str(FEW_SHOT)]
    code = run_script("check_leakage", argv)
    assert code == 0, capsys.readouterr().out


def test_no_final_is_a_near_duplicate_of_a_training_final(
    additions: list[dict[str, Any]], pool: dict[str, dict[str, Any]]
) -> None:
    """build_dataset drops near-duplicate finals (R23), so a twin that repeats a v1 final, or
    another twin's, would be a teacher call thrown away."""
    seen = [(row["id"], _final(row)) for row in pool.values()]
    bad = []
    for row in additions:
        final = _final(row)
        grams = word_ngrams(final)
        for other_id, other in seen:
            if text_hash(final) == text_hash(other) or (
                jaccard(grams, word_ngrams(other)) >= JACCARD_THRESHOLD
            ):
                bad.append((row["id"], other_id))
                break
        seen.append((row["id"], final))
    assert not bad, f"{len(bad)} near-duplicate finals, e.g. {bad[:10]}"


# --- (5) the paid step, planned but not run --------------------------------------------------


def test_distill_dry_run_plans_n_times_k_calls_and_makes_none(
    isolated_env: Path,
    no_network: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    additions: list[dict[str, Any]],
) -> None:
    out = tmp_path / "distill_v2"
    argv = ["--scenarios", str(ADDITIONS), "--k", str(K), "--out", str(out), "--dry-run"]
    assert run_script("distill", argv) == 0
    stdout = capsys.readouterr().out
    n = len(additions)
    assert f"({n} scenarios, k={K})" in stdout, stdout
    assert re.search(rf"^teacher calls: {n * K}$", stdout, re.MULTILINE), stdout
    assert "no requests made" in stdout
    assert not out.exists() or not any(out.iterdir()), "a dry run wrote files"


def test_v2_abstain_share_stays_in_range(additions: list[dict[str, Any]]) -> None:
    """v2 = v1 + kept cite twins: in range whether every twin is kept or only half of them."""
    low, high = ABSTAIN_SHARE_RANGE
    for kept in (len(additions), int(len(additions) * LOW_KEEP_RATE)):
        share = V1_ABSTAIN / (V1_COUNT + kept)
        assert low <= share <= high, f"{kept} kept twins: abstain share {share:.3f}"


def test_diff_datasets_summarizes_v1_plus_the_additions(
    tmp_path: Path, additions: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    """v2 = v1's 448 lines unchanged + one kept example per twin: only additions show."""
    v2 = tmp_path / "v2"
    v1_lines = (V1 / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(v1_lines) == V1_COUNT
    new = [
        {
            "messages": [
                {"role": "user", "content": _final(row)},
                {"role": "assistant", "content": cite_reply(row["gold"]["supporting_ids"][0])},
            ],
            "meta": {"scenario_id": row["id"], "category": row["category"]},
        }
        for row in additions
    ]
    write_jsonl(v2 / "train.jsonl", [json.loads(line) for line in v1_lines] + new)
    (v2 / "manifest.json").write_text(json.dumps({"version": "v2"}), encoding="utf-8")
    summary_path = tmp_path / "diff.json"
    assert run_script("diff_datasets", [str(V1), str(v2), "--json", str(summary_path)]) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["added"] == sorted(row["id"] for row in additions)
    assert summary["removed"] == [] and summary["relabeled"] == []
    assert summary["unchanged"] == V1_COUNT
    added = Counter(row["category"] for row in additions)
    assert {c: v["added"] for c, v in summary["by_category"].items()} == dict(added)


def test_training_config_is_frozen_at_the_v1_hash() -> None:
    """G21/G29: v2 changes the data only; the config still resolves to the v1 run's hash."""
    log = json.loads(V1_TRAIN_LOG.read_text(encoding="utf-8"))
    train = load_train_module()
    assert train.config_sha256(train.resolve_config()) == log["config_sha256"]


def test_v2_plan_is_documented(additions: list[dict[str, Any]]) -> None:
    assert PLAN_DOC.is_file(), f"{PLAN_DOC.relative_to(REPO_ROOT)} is missing"
    text = PLAN_DOC.read_text(encoding="utf-8")
    log = json.loads(V1_TRAIN_LOG.read_text(encoding="utf-8"))
    n = len(additions)
    v1_sha = hashlib.sha256((V1 / "train.jsonl").read_bytes()).hexdigest()
    required = {
        "the generator": "scripts/gen_v2_additions.py",
        "the additions file": "data/scenarios/train_v2_additions.jsonl",
        "the v1 example count": str(V1_COUNT),
        "the number of twins": str(n),
        "the teacher-call count": str(n * K),
        "the dry run": "--dry-run",
        "the diff command": "scripts/diff_datasets.py data/datasets/v1 data/datasets/v2",
        "the v1 dataset hash": v1_sha,
        "the frozen config hash": log["config_sha256"],
    }
    missing = [what for what, needle in required.items() if needle not in text]
    assert not missing, f"{PLAN_DOC.name} does not state {missing}"
