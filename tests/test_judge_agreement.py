"""The judge-agreement check (D-095, D-096, D-099): ``scripts/judge_agreement.py``.

    uv run scripts/judge_agreement.py --build-sample --seed S \
        --out data/judge_agreement/sample.jsonl
    uv run scripts/judge_agreement.py --run --judge SPEC --sample PATH --out DIR
        [--dry-run] [--resume] [--concurrency N]
    uv run scripts/judge_agreement.py --report --dirs DIR [DIR ...] --out PATH
        [--adjudication PATH] [--full-dirs DIR [DIR ...]]

``--build-sample`` draws the sample deterministically from the committed transcripts
(``results/ablation/m1`` and ``results/rollout_pilot/pilot1``, both judged by gpt-5.5). Each JSONL
row has ``sample_id`` (unique), ``source`` (``m1`` or ``pilot``), ``run_dir`` (repo-relative),
``model``, ``strategy``, ``scenario_id``, ``category``, ``seed`` and ``reference`` =
``{judge_model, spec_adherence, robustness}`` copied from the source row; pilot rows also carry
``conversation_id`` and ``turn_index``.

D-099: every sampled row is judge-decided. No HARD check (D-028) failed on it, so its
Spec-adherence is the judge's verdict and was not forced to 0. Within those rows the draw is the
D-096 stratification:

- 60 M1 rows: 10 per subject model x strategy cell, every (model, strategy, category) stratum
  present, and as close to half fails as the data allows: ``min(30, available)`` fails, where
  only 5 unforced M1 fails exist.
- 40 pilot rows: every unforced caliper turn (a cite turn whose gold snippet applies only to
  floating front calipers; 10 of the 20 are unforced, all passes), ``min(20, available)`` fails
  (1 unforced pilot fail exists), at least 8 early turns (index <= 2), 8 late turns (index >= 6),
  8 cite and 8 abstain golds, and every held-out family.

Next to the sample, ``<stem>.manifest.json`` records ``seed``, ``sample_sha256`` and
``shortfalls``: one ``{stratum, wanted, got}`` entry for each D-096 quota that the unforced rows
can't fill (the M1 fails, the pilot fails and the caliper turns). It names no path. The same seed
gives byte-identical output. The committed sample uses seed 96 and rebuilds from it.

``--run`` re-scores every sample row through the rejudge machinery (stored replies, current
checks, unchanged rubric): ``DIR/transcripts.jsonl`` holds one rejudge-shaped row per sample row
plus ``sample_id``, ``source`` and ``reference``. It is resumable like ``rejudge.py`` (``--resume``
continues the same sample and judge; a non-empty DIR without it, or a different judge, exits 2).
A missing key or a non-retryable 4xx exits 2 with a redacted message before or instead of any
row. ``--dry-run`` needs no key, writes nothing and prints ``judge calls: N`` and
``estimated cost: $X.XX`` from the ``[prices]`` table. ``DIR/spend.json`` gives the actual spend
from recorded usage: ``{judge_model, calls, input_tokens, output_tokens, reasoning_tokens,
cost_usd}``, across every invocation of a resumed run.

``--report`` writes a JSON report to PATH with, per candidate (keyed by judge spec) vs the
gpt-5.5 reference: ``n``, ``agreement`` (Spec-adherence, 0-1), ``kappa`` (Cohen's),
``confusion`` (``confusion[reference][candidate]`` over ``pass``/``fail``),
``mean_abs_robustness_delta``, ``disagreements``, ``disagreements_by_category`` (keyed
``<source>:<category>``), ``adjudicated``, ``wins``, ``win_rate``, ``spend_usd``, ``verdict`` and
``all_rows``. ``verdict`` is ``pass`` only if agreement >= 0.90 AND kappa >= 0.75 AND the
candidate wins >= 50% of its adjudicated disagreements (vacuous with none); ``fail`` if a bar is
missed; ``pending`` while any of its disagreements is unadjudicated. The bar is computed on the
sample rows only, and they must all be judge-decided: a run row with a failed HARD check exits 2
naming its sample_id (D-099). ``report["baselines"]`` holds ``always_pass`` and ``always_fail``
(``n``, ``agreement``, ``kappa``, ``verdict``) for constant judges on the same rows. Both must
fail the bar, and the report prints one line for each. ``all_rows`` is ``None`` unless
``--full-dirs`` names a ``rejudge.py`` re-judge of M1 or the pilot by that candidate. Then it
holds ``{n, agreement, kappa, confusion}`` against the source run's reference verdicts on every
row, forced rows included, for reference only (the verdict never uses it).

The report also writes ``disagreements.jsonl`` next to PATH, one blind item per sample row where
any candidate disagrees with the reference: an opaque ``item_id``, the context, gold, reply and
the anonymized verdicts, with no judge or model name. ``report["items"]`` maps item_id ->
sample_id. The adjudication file is JSONL of ``{item_id, spec_adherence}``; a candidate wins an
item when its verdict matches.

Sockets are blocked and keys are fake. API judges are the real backend classes over fake
transports, so no test calls a real API.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import random
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from config_helpers import patched_config

from groundcite import backends, config
from groundcite.checks import HARD_CHECKS
from groundcite.judge import rubric_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "judge_agreement.py"
COMMITTED_SAMPLE = REPO_ROOT / "data" / "judge_agreement" / "sample.jsonl"
COMMITTED_SEED = 96
M1_DIR = "results/ablation/m1"
PILOT_DIR = "results/rollout_pilot/pilot1"
PILOT_SCRIPTS = REPO_ROOT / "data" / "scenarios" / "rollout_pilot.jsonl"
REJUDGE = REPO_ROOT / "scripts" / "rejudge.py"

XAI = "xai:grok-4.20-0309-reasoning"
SONNET = "anthropic:claude-sonnet-5-5"
HAIKU = "anthropic:claude-haiku-5-5"
REFERENCE = "openai:gpt-5.5-2026-04-23"
RUBRIC_SHA = "636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf"
FAKE_XAI = "xai-FAKE-5555-AGREEMENT-TESTVALUE"
FAKE_ANTHROPIC = "FAKE-ANTHROPIC-6666-AGREEMENT"
SEED = 7

# Per call, the fake xAI judge reports 1000 input, 100 visible output and 400 reasoning tokens:
# (1000 x 1.25 + 500 x 2.50) / 1e6 = $0.0025.
XAI_CALL_COST = 0.0025
FORBIDDEN_IN_BLIND = (
    "gpt-5.5",
    "gpt",
    "grok",
    "sonnet",
    "haiku",
    "claude",
    "openai",
    "anthropic",
    "xai:",
    "opus",
)


# ---------------------------------------------------------------------------------------------
# Loading the script
# ---------------------------------------------------------------------------------------------


def _load() -> ModuleType:
    assert SCRIPT.is_file(), "scripts/judge_agreement.py does not exist yet"
    spec = importlib.util.spec_from_file_location("judge_agreement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ja(*argv: str) -> int:
    try:
        return int(_load().main(list(argv)))
    except SystemExit as exc:
        return int(exc.code or 0)


def rejudge(*argv: str) -> int:
    spec = importlib.util.spec_from_file_location("rejudge", REJUDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return int(module.main(list(argv)))
    except SystemExit as exc:
        return int(exc.code or 0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def manifest_of(sample: Path) -> dict[str, Any]:
    path = sample.with_name(sample.stem + ".manifest.json")
    assert path.is_file(), f"--build-sample wrote no {path.name} next to the sample"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def forced(row: Mapping[str, Any]) -> bool:
    """A HARD check failed, so D-028 pinned Spec-adherence to 0 whatever the judge said."""
    return any(row["checks"].get(name) is False for name in HARD_CHECKS)


# ---------------------------------------------------------------------------------------------
# The committed sources
# ---------------------------------------------------------------------------------------------


@functools.cache
def source_rows() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """(run_dir, model, strategy, scenario_id) -> the stored gpt-5.5-judged row."""
    rows = {}
    for run_dir in (M1_DIR, PILOT_DIR):
        for row in read_jsonl(REPO_ROOT / run_dir / "transcripts.jsonl"):
            rows[(run_dir, row["model"], row["strategy"], row["scenario_id"])] = row
    return rows


def source_of(sample_row: Mapping[str, Any]) -> dict[str, Any]:
    key = (
        sample_row["run_dir"],
        sample_row["model"],
        sample_row["strategy"],
        sample_row["scenario_id"],
    )
    assert key in source_rows(), f"sample row {sample_row['sample_id']} is not in its source run"
    return source_rows()[key]


@functools.cache
def pilot_scripts() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in read_jsonl(PILOT_SCRIPTS)}


@functools.cache
def caliper_ids() -> frozenset[str]:
    """Snippet IDs whose text limits them to floating front calipers."""
    ids: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            text = str(obj.get("text", "")).lower()
            if "id" in obj and "floating front caliper" in text:
                ids.add(obj["id"])
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(list(pilot_scripts().values()))
    return frozenset(ids)


def caliper_keys() -> set[tuple[str, str, str, str]]:
    return {
        key
        for key, row in source_rows().items()
        if key[0] == PILOT_DIR
        and row["gold"]["mode"] == "cite"
        and set(row["gold"]["supporting_ids"]) & caliper_ids()
    }


def key_of(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (row["run_dir"], row["model"], row["strategy"], row["scenario_id"])


def unforced(run_dir: str) -> list[dict[str, Any]]:
    """The source run's judge-decided rows (D-099), in source order."""
    return [
        row
        for key, row in source_rows().items()
        if key[0] == run_dir and not forced(row) and not row.get("generation_error")
    ]


# ---------------------------------------------------------------------------------------------
# The built sample (built once per seed per test session)
# ---------------------------------------------------------------------------------------------


@functools.cache
def _build_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="judge_agreement_sample_"))


@functools.cache
def built_sample(seed: int = SEED) -> Path:
    out = _build_dir() / f"sample-{seed}.jsonl"
    code = ja("--build-sample", "--seed", str(seed), "--out", str(out))
    assert code == 0, f"--build-sample exited {code}"
    return out


def sample_rows(seed: int = SEED) -> list[dict[str, Any]]:
    return read_jsonl(built_sample(seed))


# ---------------------------------------------------------------------------------------------
# --build-sample
# ---------------------------------------------------------------------------------------------


def test_sample_has_60_m1_and_40_pilot_rows_with_the_reference_verdicts() -> None:
    rows = sample_rows()
    assert len(rows) == 100
    assert Counter(r["source"] for r in rows) == {"m1": 60, "pilot": 40}
    assert len({r["sample_id"] for r in rows}) == 100
    assert len({key_of(r) for r in rows}) == 100, "no row is sampled twice"
    for row in rows:
        src = source_of(row)
        assert row["run_dir"] == (M1_DIR if row["source"] == "m1" else PILOT_DIR)
        assert row["category"] == src["category"]
        assert row["seed"] == SEED
        assert row["reference"] == {
            "judge_model": REFERENCE,
            "spec_adherence": src["score"]["spec_adherence"],
            "robustness": src["score"]["robustness"],
        }
        assert src["judge_model"] == REFERENCE
        assert not src.get("generation_error")
        if row["source"] == "pilot":
            assert row["conversation_id"] == src["conversation_id"]
            assert row["turn_index"] == src["turn_index"]


def test_every_sampled_row_is_judge_decided() -> None:
    """D-099: no sampled row failed a HARD check, so no reference verdict was forced to 0."""
    for row in sample_rows():
        src = source_of(row)
        failed = sorted(n for n in HARD_CHECKS if src["checks"].get(n) is False)
        assert not failed, f"{row['sample_id']} failed HARD checks {failed}"
        assert src["judge"]["forced_by_checks"] is False, row["sample_id"]


def test_m1_part_is_stratified_by_cell_and_category_with_as_many_fails_as_exist() -> None:
    m1 = [r for r in sample_rows() if r["source"] == "m1"]
    cells = Counter((r["model"], r["strategy"]) for r in m1)
    all_cells = {(k[1], k[2]) for k in source_rows() if k[0] == M1_DIR}
    assert set(cells) == all_cells and len(all_cells) == 6
    assert set(cells.values()) == {10}, f"10 rows per model x strategy cell, got {cells}"

    categories = {r["category"] for k, r in source_rows().items() if k[0] == M1_DIR}
    assert len(categories) == 6
    strata = {(r["model"], r["strategy"], r["category"]) for r in m1}
    missing = {(m, s, c) for (m, s) in all_cells for c in categories} - strata
    assert not missing, f"strata with no sampled row: {sorted(missing)}"

    fails = {key_of(r) for r in m1 if r["reference"]["spec_adherence"] == 0}
    available = {
        (M1_DIR, r["model"], r["strategy"], r["scenario_id"])
        for r in unforced(M1_DIR)
        if r["score"]["spec_adherence"] == 0
    }
    assert len(fails) == min(30, len(available)) == 5
    assert fails == available, "every unforced M1 fail is sampled"


def test_pilot_part_covers_caliper_turns_fails_turn_depth_modes_and_families() -> None:
    pilot = [r for r in sample_rows() if r["source"] == "pilot"]
    keys = {key_of(r) for r in pilot}

    unforced_keys = {
        (PILOT_DIR, r["model"], r["strategy"], r["scenario_id"]) for r in unforced(PILOT_DIR)
    }
    calipers = caliper_keys() & unforced_keys
    assert len(caliper_keys()) == 20, "the pilot has 20 caliper turns"
    assert len(calipers) == 10, "10 of them are judge-decided"
    assert calipers <= keys, "every unforced caliper turn is in the sample"

    fails = [r for r in pilot if r["reference"]["spec_adherence"] == 0]
    available = sum(r["score"]["spec_adherence"] == 0 for r in unforced(PILOT_DIR))
    assert len(fails) == min(20, available) == 1

    assert sum(r["turn_index"] <= 2 for r in pilot) >= 8, "early turns"
    assert sum(r["turn_index"] >= 6 for r in pilot) >= 8, "late turns"
    modes = Counter(source_of(r)["gold"]["mode"] for r in pilot)
    assert modes["cite"] >= 8 and modes["abstain"] >= 8, modes

    families = {s["family"] for s in pilot_scripts().values()}
    sampled = {pilot_scripts()[r["conversation_id"]]["family"] for r in pilot}
    assert sampled == families


def _shortfall(manifest: Mapping[str, Any], *words: str) -> dict[str, Any]:
    entries = manifest["shortfalls"]
    assert isinstance(entries, list)
    for entry in entries:
        assert {"stratum", "wanted", "got"} <= set(entry), entry
        assert entry["got"] < entry["wanted"], f"not a shortfall: {entry}"
    found = [e for e in entries if all(w in str(e["stratum"]).lower() for w in words)]
    assert len(found) == 1, f"one shortfall naming {words} expected in {entries}"
    return dict(found[0])


def test_the_sample_manifest_records_seed_hash_and_every_stratum_shortfall() -> None:
    """D-099: where a stratum lacks unforced rows the draw takes the nearest one and records the
    shortfall."""
    sample = built_sample()
    manifest = manifest_of(sample)
    assert manifest["seed"] == SEED
    assert manifest["sample_sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()

    m1_fails = _shortfall(manifest, "m1", "fail")
    assert (m1_fails["wanted"], m1_fails["got"]) == (30, 5)
    pilot_fails = _shortfall(manifest, "pilot", "fail")
    assert (pilot_fails["wanted"], pilot_fails["got"]) == (20, 1)
    calipers = _shortfall(manifest, "caliper")
    assert (calipers["wanted"], calipers["got"]) == (20, 10)
    assert len(manifest["shortfalls"]) == 3, "the cell, stratum and turn quotas all fill"


def test_same_seed_is_byte_identical_and_another_seed_differs(tmp_path: Path) -> None:
    again = tmp_path / "again.jsonl"
    assert ja("--build-sample", "--seed", str(SEED), "--out", str(again)) == 0
    assert again.read_bytes() == built_sample(SEED).read_bytes()
    assert manifest_of(again) == manifest_of(built_sample(SEED))

    other = {key_of(r) for r in sample_rows(SEED + 1)}
    assert other != {key_of(r) for r in sample_rows(SEED)}
    assert all(r["seed"] == SEED + 1 for r in sample_rows(SEED + 1))


def test_the_committed_sample_rebuilds_from_seed_96(tmp_path: Path) -> None:
    """D-096, D-099: the sample file, its manifest and its seed are committed before any judge
    call."""
    assert COMMITTED_SAMPLE.is_file(), "data/judge_agreement/sample.jsonl is not committed"
    rows = read_jsonl(COMMITTED_SAMPLE)
    assert len(rows) == 100
    [seed] = {r["seed"] for r in rows}
    assert seed == COMMITTED_SEED
    rebuilt = tmp_path / COMMITTED_SAMPLE.name
    assert ja("--build-sample", "--seed", str(seed), "--out", str(rebuilt)) == 0
    assert rebuilt.read_bytes() == COMMITTED_SAMPLE.read_bytes()
    assert manifest_of(rebuilt) == manifest_of(COMMITTED_SAMPLE)


# ---------------------------------------------------------------------------------------------
# Fake judges over the real backend classes
# ---------------------------------------------------------------------------------------------

_REPLY_RE = re.compile(r"## Reply to grade\n\n<<<\n(.*?)\n>>>", re.DOTALL)

# (reply, reference verdict) -> (spec_adherence, robustness)
Policy = Callable[[str, Mapping[str, Any]], tuple[int, int]]


@functools.cache
def reference_by_reply() -> dict[str, dict[str, Any]]:
    """reply text -> the reference verdict (replies are unique across M1 and the pilot)."""
    return {
        row["reply"]: {"spec_adherence": s["spec_adherence"], "robustness": s["robustness"]}
        for row in source_rows().values()
        for s in [row["score"]]
    }


def oracle(reply: str, ref: Mapping[str, Any]) -> tuple[int, int]:
    return ref["spec_adherence"], ref["robustness"]


def contrarian(reply: str, ref: Mapping[str, Any]) -> tuple[int, int]:
    return 1 - ref["spec_adherence"], 5 if ref["spec_adherence"] == 0 else 1


def _strings(obj: Any) -> list[str]:
    """Every string in a request body, whatever shape its message content takes."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, Mapping):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _verdict_text(body: Mapping[str, Any], policy: Policy) -> str:
    match = _REPLY_RE.search("\n".join(_strings(body)))
    assert match, "the judge input has no reply section"
    reply = match.group(1)
    spec, robustness = policy(reply, reference_by_reply()[reply])
    return json.dumps(
        {
            "spec_adherence": spec,
            "robustness": robustness,
            "failed_checks": [],
            "reasoning": "Fake verdict for the agreement tests.",
        }
    )


class FakeXAI:
    """Answers either xAI API (by URL) with a policy verdict and fixed usage. ``fail_at`` makes
    the n-th call (1-based) raise ``failure``."""

    def __init__(
        self, policy: Policy = oracle, fail_at: int | None = None, failure: Exception | None = None
    ) -> None:
        self.policy = policy
        self.fail_at = fail_at
        self.failure = failure
        self.calls = 0

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            assert self.failure is not None
            raise self.failure
        if url.endswith("/responses"):
            verdict = _verdict_text(body, self.policy)
            return {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": verdict}],
                    },
                ],
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                    "output_tokens_details": {"reasoning_tokens": 400},
                },
            }
        verdict = _verdict_text(body, self.policy)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": verdict},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1500,
                "completion_tokens_details": {"reasoning_tokens": 400},
            },
        }


class FakeAnthropic:
    """Answers each Anthropic model with its own policy and fixed usage."""

    def __init__(self, policies: Mapping[str, Policy]) -> None:
        self.policies = dict(policies)
        self.calls: Counter[str] = Counter()

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        model = body["model"]
        self.calls[model] += 1
        verdict = _verdict_text(body, self.policies[model])
        return {
            "content": [{"type": "text", "text": verdict}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }


class FakeHTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status


@pytest.fixture
def env(isolated_env: Path, no_network: None, monkeypatch: pytest.MonkeyPatch) -> Path:
    isolated_env.write_text(
        f"XAI_API_KEY={FAKE_XAI}\nANTHROPIC_API_KEY={FAKE_ANTHROPIC}\n", encoding="utf-8"
    )
    monkeypatch.setattr(backends, "_REJECTED", {})
    return isolated_env


def use_xai(monkeypatch: pytest.MonkeyPatch, transport: FakeXAI) -> FakeXAI:
    monkeypatch.setitem(
        backends.BACKENDS,
        "xai:",
        lambda spec, _transport=None: backends.XAIBackend(spec, transport),
    )
    return transport


def use_anthropic(monkeypatch: pytest.MonkeyPatch, transport: FakeAnthropic) -> FakeAnthropic:
    monkeypatch.setitem(
        backends.BACKENDS,
        "anthropic:",
        lambda spec, _transport=None: backends.AnthropicBackend(spec, transport),
    )
    return transport


def run_argv(judge: str, sample: Path, out: Path, *extra: str) -> list[str]:
    return ["--run", "--judge", judge, "--sample", str(sample), "--out", str(out), *extra]


def all_text(root: Path) -> str:
    if not root.exists():
        return ""
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*") if p.is_file()
    )


# ---------------------------------------------------------------------------------------------
# --run --dry-run
# ---------------------------------------------------------------------------------------------


def _dry_run_cost(capsys: pytest.CaptureFixture[str], out: Path) -> tuple[int, float]:
    code = ja(*run_argv(XAI, built_sample(), out, "--dry-run"))
    stdout = capsys.readouterr().out
    assert code == 0
    calls = re.search(r"(?mi)^\s*judge calls:\s*(\d+)\s*$", stdout)
    cost = re.search(r"(?mi)^\s*estimated cost:\s*\$(\d+\.\d+)", stdout)
    assert calls, f"no 'judge calls: N' line in:\n{stdout}"
    assert cost, f"no 'estimated cost: $X' line in:\n{stdout}"
    return int(calls.group(1)), float(cost.group(1))


def test_dry_run_prints_calls_and_estimated_cost_without_a_key_or_any_write(
    isolated_env: Path,
    no_network: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = use_xai(monkeypatch, FakeXAI())
    out = tmp_path / "dry"
    calls, cost = _dry_run_cost(capsys, out)
    assert calls == 100
    assert 0 < cost < 5, "a 100-reply re-judge is estimated at cents to a few dollars"
    assert transport.calls == 0
    assert not out.exists()


def test_dry_run_estimate_follows_the_configured_price(
    isolated_env: Path,
    no_network: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, base = _dry_run_cost(capsys, tmp_path / "a")

    def double(cfg: dict[str, Any]) -> None:
        price = cfg["prices"][XAI]
        price["input_per_mtok"] *= 2
        price["output_per_mtok"] *= 2

    patched_config(monkeypatch, tmp_path, double)
    _, doubled = _dry_run_cost(capsys, tmp_path / "b")
    assert doubled == pytest.approx(2 * base, rel=0.05)


# ---------------------------------------------------------------------------------------------
# --run
# ---------------------------------------------------------------------------------------------


def test_run_rescores_every_sample_row_with_the_unchanged_rubric(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = use_xai(monkeypatch, FakeXAI(oracle))
    out = tmp_path / "grok"
    assert ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1")) == 0
    assert transport.calls == 100

    sample = {r["sample_id"]: r for r in sample_rows()}
    rows = read_jsonl(out / "transcripts.jsonl")
    assert len(rows) == 100
    assert {r["sample_id"] for r in rows} == set(sample)
    assert rubric_sha256() == RUBRIC_SHA, "the rubric must be unchanged (D-096)"
    for row in rows:
        want = sample[row["sample_id"]]
        src = source_of(want)
        assert row["judge_model"] == XAI
        assert row["rubric_sha256"] == RUBRIC_SHA
        assert row["source"] == want["source"]
        assert row["reference"] == want["reference"]
        for field in ("model", "strategy", "scenario_id", "category", "reply"):
            assert row[field] == src[field], field
        # The oracle copies the reference, and the current checks reproduce the stored ones.
        assert row["score"] == src["score"]


def test_run_records_actual_spend_from_usage_and_prices(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_xai(monkeypatch, FakeXAI(oracle))
    out = tmp_path / "grok"
    assert ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1")) == 0
    spend = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    assert spend["judge_model"] == XAI
    assert spend["calls"] == 100
    assert spend["input_tokens"] == 100 * 1000
    assert spend["output_tokens"] == 100 * 100
    assert spend["reasoning_tokens"] == 100 * 400
    assert spend["cost_usd"] == pytest.approx(100 * XAI_CALL_COST)


def test_run_is_resumable_and_spend_covers_both_invocations(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "grok"
    broken = use_xai(
        monkeypatch, FakeXAI(oracle, fail_at=40, failure=RuntimeError("connection reset"))
    )
    code = ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1"))
    assert code not in (0, 2), "an interrupted run is neither a success nor a usage error"
    assert broken.calls == 40
    assert len(read_jsonl(out / "transcripts.jsonl")) == 39

    healthy = use_xai(monkeypatch, FakeXAI(oracle))
    assert ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1")) == 2, (
        "a non-empty DIR without --resume is refused"
    )
    assert healthy.calls == 0
    assert ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1", "--resume")) == 0
    assert healthy.calls == 61, "only the missing rows are judged"

    rows = read_jsonl(out / "transcripts.jsonl")
    assert len(rows) == 100 and len({r["sample_id"] for r in rows}) == 100
    spend = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    assert spend["calls"] == 100
    assert spend["cost_usd"] == pytest.approx(100 * XAI_CALL_COST)


def test_resume_refuses_another_judge(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "grok"
    use_xai(monkeypatch, FakeXAI(oracle, fail_at=5, failure=RuntimeError("connection reset")))
    ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1"))
    assert read_jsonl(out / "transcripts.jsonl")
    fake = use_anthropic(monkeypatch, FakeAnthropic({"claude-haiku-5-5": oracle}))
    assert ja(*run_argv(HAIKU, built_sample(), out, "--resume")) == 2
    assert sum(fake.calls.values()) == 0


def test_missing_key_exits_2_naming_xai_api_key_before_any_work(
    isolated_env: Path,
    no_network: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = use_xai(monkeypatch, FakeXAI())
    out = tmp_path / "grok"
    assert ja(*run_argv(XAI, built_sample(), out)) == 2
    assert "XAI_API_KEY" in capsys.readouterr().err
    assert transport.calls == 0
    assert not out.exists()


@pytest.mark.parametrize("status", [400, 401, 403])
def test_a_4xx_stops_the_run_with_exit_2_and_a_redacted_message(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    failure = FakeHTTPError(status, f"Error code: {status} (Authorization: Bearer {FAKE_XAI})")
    transport = use_xai(monkeypatch, FakeXAI(fail_at=1, failure=failure))
    out = tmp_path / "grok"
    code = ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1"))
    captured = capsys.readouterr()
    assert code == 2
    assert transport.calls == 1, "no further requests after a non-retryable 4xx"
    assert str(status) in captured.err
    assert FAKE_XAI not in captured.err + captured.out
    assert FAKE_XAI not in all_text(out)
    if (out / "transcripts.jsonl").exists():
        assert read_jsonl(out / "transcripts.jsonl") == [], "no failure rows are scored"


def test_the_key_never_reaches_logs_outputs_or_spend(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    use_xai(monkeypatch, FakeXAI(oracle))
    out = tmp_path / "grok"
    assert ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1")) == 0
    report = tmp_path / "report" / "report.json"
    assert ja("--report", "--dirs", str(out), "--out", str(report)) == 0
    captured = capsys.readouterr()
    # Only what the script wrote: tmp_path also holds the isolated env file with the fake keys.
    written = [all_text(out), all_text(report.parent)]
    blob = "\n".join([caplog.text, captured.out, captured.err, *written])
    assert FAKE_XAI not in blob
    assert FAKE_ANTHROPIC not in blob


# ---------------------------------------------------------------------------------------------
# --report
# ---------------------------------------------------------------------------------------------


def as_sample_row(sample_id: str, run_dir: str, src: Mapping[str, Any]) -> dict[str, Any]:
    """A sample row for a stored source row, in the --build-sample shape."""
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "source": "m1" if run_dir == M1_DIR else "pilot",
        "run_dir": run_dir,
        "model": src["model"],
        "strategy": src["strategy"],
        "scenario_id": src["scenario_id"],
        "category": src["category"],
        "seed": SEED,
        "reference": {
            "judge_model": src["judge_model"],
            "spec_adherence": src["score"]["spec_adherence"],
            "robustness": src["score"]["robustness"],
        },
    }
    if "conversation_id" in src:
        row["conversation_id"] = src["conversation_id"]
        row["turn_index"] = src["turn_index"]
    return row


def write_sample(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def small_rows() -> list[dict[str, Any]]:
    """20 judge-decided rows, independent of --build-sample: every unforced reference fail (5 in
    M1, 1 in the pilot) and the first 7 unforced passes of each run."""
    picked: list[tuple[str, dict[str, Any]]] = []
    for run_dir in (M1_DIR, PILOT_DIR):
        rows = unforced(run_dir)
        picked += [(run_dir, r) for r in rows if r["score"]["spec_adherence"] == 0]
        picked += [(run_dir, r) for r in rows if r["score"]["spec_adherence"] == 1][:7]
    rows = [as_sample_row(f"t-{i:02d}", d, r) for i, (d, r) in enumerate(picked, 1)]
    assert len(rows) == 20
    assert Counter(r["reference"]["spec_adherence"] for r in rows) == {1: 14, 0: 6}
    return rows


def small_sample(tmp_path: Path) -> Path:
    return write_sample(tmp_path / "small.jsonl", small_rows())


def first_pass_reply(sample: Path) -> str:
    row = next(r for r in read_jsonl(sample) if r["reference"]["spec_adherence"] == 1)
    return str(source_of(row)["reply"])


def near_policy(flipped_reply: str) -> Policy:
    """Agrees with the reference except on one reference pass, which it fails."""

    def policy(reply: str, ref: Mapping[str, Any]) -> tuple[int, int]:
        return (0, 2) if reply == flipped_reply else oracle(reply, ref)

    return policy


def cohens_kappa(ref: Sequence[int], cand: Sequence[int]) -> float:
    n = len(ref)
    po = sum(a == b for a, b in zip(ref, cand, strict=True)) / n
    pr, pc = sum(ref) / n, sum(cand) / n
    pe = pr * pc + (1 - pr) * (1 - pc)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def expected_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ref = [r["reference"]["spec_adherence"] for r in rows]
    cand = [r["score"]["spec_adherence"] for r in rows]
    label = {1: "pass", 0: "fail"}
    confusion = {a: {b: 0 for b in ("pass", "fail")} for a in ("pass", "fail")}
    for a, b in zip(ref, cand, strict=True):
        confusion[label[a]][label[b]] += 1
    by_category = Counter(
        f"{r['source']}:{r['category']}"
        for r in rows
        if r["score"]["spec_adherence"] != r["reference"]["spec_adherence"]
    )
    deltas = [abs(r["score"]["robustness"] - r["reference"]["robustness"]) for r in rows]
    return {
        "n": len(rows),
        "agreement": sum(a == b for a, b in zip(ref, cand, strict=True)) / len(rows),
        "kappa": cohens_kappa(ref, cand),
        "confusion": confusion,
        "mean_abs_robustness_delta": sum(deltas) / len(deltas),
        "disagreements": sum(by_category.values()),
        "disagreements_by_category": dict(by_category),
    }


@pytest.fixture
def three_runs(env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Grok agrees with the reference everywhere, Sonnet nowhere, Haiku on all but one row."""
    sample = small_sample(tmp_path)
    use_xai(monkeypatch, FakeXAI(oracle))
    use_anthropic(
        monkeypatch,
        FakeAnthropic(
            {
                "claude-sonnet-5-5": contrarian,
                "claude-haiku-5-5": near_policy(first_pass_reply(sample)),
            }
        ),
    )
    dirs = {}
    for spec, name in ((XAI, "grok"), (SONNET, "sonnet"), (HAIKU, "haiku")):
        dirs[spec] = tmp_path / "runs" / name
        assert ja(*run_argv(spec, sample, dirs[spec], "--concurrency", "1")) == 0
    return dirs


def make_report(
    tmp_path: Path, dirs: Sequence[Path], adjudication: Path | None = None
) -> tuple[dict[str, Any], Path]:
    out = tmp_path / "report" / "report.json"
    argv = ["--report", "--dirs", *map(str, dirs), "--out", str(out)]
    if adjudication is not None:
        argv += ["--adjudication", str(adjudication)]
    assert ja(*argv) == 0
    return json.loads(out.read_text(encoding="utf-8")), out.parent / "disagreements.jsonl"


def test_report_metrics_match_the_scored_rows(three_runs: dict[str, Path], tmp_path: Path) -> None:
    report, _ = make_report(tmp_path, list(three_runs.values()))
    assert report["reference_judge"] == REFERENCE
    assert report["bar"] == {"agreement": 0.9, "kappa": 0.75, "win_rate": 0.5}
    assert set(report["candidates"]) == set(three_runs)
    for spec, run_dir in three_runs.items():
        got = report["candidates"][spec]
        want = expected_metrics(read_jsonl(run_dir / "transcripts.jsonl"))
        assert got["n"] == want["n"] == 20
        assert got["agreement"] == pytest.approx(want["agreement"])
        assert got["kappa"] == pytest.approx(want["kappa"])
        assert got["confusion"] == want["confusion"]
        assert got["mean_abs_robustness_delta"] == pytest.approx(want["mean_abs_robustness_delta"])
        assert got["disagreements"] == want["disagreements"]
        assert got["disagreements_by_category"] == want["disagreements_by_category"]
        spend = json.loads((run_dir / "spend.json").read_text(encoding="utf-8"))
        assert got["spend_usd"] == pytest.approx(spend["cost_usd"])

    assert report["candidates"][XAI]["agreement"] == 1.0
    assert report["candidates"][SONNET]["agreement"] == 0.0
    assert report["candidates"][HAIKU]["disagreements"] == 1


def test_anthropic_spend_uses_its_own_configured_prices(
    three_runs: dict[str, Path],
) -> None:
    for spec in (SONNET, HAIKU):
        spend = json.loads((three_runs[spec] / "spend.json").read_text(encoding="utf-8"))
        usage = [{"input_tokens": 1000, "output_tokens": 500, "reasoning_tokens": None}] * 20
        assert spend["calls"] == 20
        assert spend["cost_usd"] == pytest.approx(config.cost_usd(spec, usage))


def test_report_verdicts_follow_the_d096_bar(three_runs: dict[str, Path], tmp_path: Path) -> None:
    report, _ = make_report(tmp_path, list(three_runs.values()))
    verdicts = {spec: c["verdict"] for spec, c in report["candidates"].items()}
    # Grok: agreement 1.0, kappa 1.0, nothing to adjudicate. Sonnet misses agreement and kappa.
    # Haiku clears agreement and kappa but its one disagreement is not adjudicated yet.
    assert verdicts == {XAI: "pass", SONNET: "fail", HAIKU: "pending"}


@pytest.mark.parametrize(("label", "verdict"), [(0, "pass"), (1, "fail")])
def test_adjudication_decides_a_candidate_that_clears_agreement_and_kappa(
    three_runs: dict[str, Path], tmp_path: Path, label: int, verdict: str
) -> None:
    dirs = [three_runs[XAI], three_runs[HAIKU]]
    first, disagreements = make_report(tmp_path, dirs)
    [item] = read_jsonl(disagreements)
    adjudication = tmp_path / "adjudication.jsonl"
    adjudication.write_text(
        json.dumps({"item_id": item["item_id"], "spec_adherence": label}) + "\n", encoding="utf-8"
    )
    report, _ = make_report(tmp_path, dirs, adjudication)
    haiku = report["candidates"][HAIKU]
    assert haiku["adjudicated"] == 1
    assert haiku["wins"] == (1 if label == 0 else 0)
    assert haiku["win_rate"] == (1.0 if label == 0 else 0.0)
    assert haiku["verdict"] == verdict


def test_disagreements_file_is_blind_and_covers_every_disagreement(
    three_runs: dict[str, Path], tmp_path: Path
) -> None:
    report, disagreements = make_report(tmp_path, list(three_runs.values()))
    items = read_jsonl(disagreements)

    disagreeing = {
        r["sample_id"]
        for run_dir in three_runs.values()
        for r in read_jsonl(run_dir / "transcripts.jsonl")
        if r["score"]["spec_adherence"] != r["reference"]["spec_adherence"]
    }
    assert len(items) == len(disagreeing) == 20, "one item per disagreeing sample row"
    assert {report["items"][i["item_id"]] for i in items} == disagreeing
    assert len({i["item_id"] for i in items}) == len(items)

    sample_ids = {r["sample_id"] for r in small_rows()}
    for item in items:
        assert item["item_id"] not in sample_ids, "item IDs are opaque, not sample IDs"
        assert item["reply"]
        assert item["context"]
        assert item["verdicts"], "the competing verdicts, anonymized"
        for verdict in item["verdicts"]:
            assert set(verdict) <= {"spec_adherence", "robustness", "reasoning", "failed_checks"}
        assert not {"model", "judge_model", "judge", "strategy"} & set(item)

    text = disagreements.read_text(encoding="utf-8").lower()
    for name in FORBIDDEN_IN_BLIND:
        assert name not in text, f"the blind file names {name!r}"


def test_disagreement_item_order_does_not_reveal_the_reference(
    three_runs: dict[str, Path], tmp_path: Path
) -> None:
    """The reference verdict is not always in the same position among an item's verdicts."""
    report, disagreements = make_report(tmp_path, [three_runs[SONNET]])
    by_sample = {r["sample_id"]: r for r in small_rows()}
    positions = set()
    for item in read_jsonl(disagreements):
        ref = by_sample[report["items"][item["item_id"]]]["reference"]["spec_adherence"]
        specs = [v["spec_adherence"] for v in item["verdicts"]]
        positions.add(specs.index(ref))
    assert len(positions) > 1


# ---------------------------------------------------------------------------------------------
# --report on judge-decided rows (D-099)
# ---------------------------------------------------------------------------------------------


def always_pass(reply: str, ref: Mapping[str, Any]) -> tuple[int, int]:
    return 1, 5


def bar_verdict(agreement: float, kappa: float) -> str:
    return "pass" if agreement >= 0.9 and kappa >= 0.75 else "fail"


def test_report_refuses_a_run_with_a_hard_check_failure(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bar is only about the judge's own verdicts, so a forced row can't be in it."""
    src = next(r for r in source_rows().values() if forced(r) and not r.get("generation_error"))
    run_dir = next(k[0] for k, r in source_rows().items() if r is src)
    stale = write_sample(
        tmp_path / "stale.jsonl", [*small_rows(), as_sample_row("t-forced", run_dir, src)]
    )
    use_xai(monkeypatch, FakeXAI(oracle))
    out = tmp_path / "grok"
    assert ja(*run_argv(XAI, stale, out, "--concurrency", "1")) == 0
    capsys.readouterr()
    report = tmp_path / "report" / "report.json"
    assert ja("--report", "--dirs", str(out), "--out", str(report)) == 2
    assert "t-forced" in capsys.readouterr().err, "the error names the forced sample row"
    assert not report.exists()


def test_report_baselines_always_pass_and_always_fail_miss_the_bar(
    three_runs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    report, _ = make_report(tmp_path, list(three_runs.values()))
    stdout = capsys.readouterr().out
    ref = [r["reference"]["spec_adherence"] for r in small_rows()]
    for name, constant in (("always_pass", 1), ("always_fail", 0)):
        baseline = report["baselines"][name]
        cand = [constant] * len(ref)
        agreement = sum(a == b for a, b in zip(ref, cand, strict=True)) / len(ref)
        assert baseline["n"] == 20
        assert baseline["agreement"] == pytest.approx(agreement)
        assert baseline["kappa"] == pytest.approx(cohens_kappa(ref, cand)) == 0.0
        assert baseline["verdict"] == "fail"
        line = re.search(rf"(?mi)^.*{name.replace('_', '[-_ ]')}.*$", stdout)
        assert line, f"no {name} line in:\n{stdout}"
        assert re.search(r"(?i)\bfail", line.group(0)), line.group(0)


def test_the_bar_is_computed_on_the_judge_decided_sample(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_xai(monkeypatch, FakeXAI(oracle))
    out = tmp_path / "grok"
    assert ja(*run_argv(XAI, built_sample(), out, "--concurrency", "1")) == 0
    assert not any(forced(r) for r in read_jsonl(out / "transcripts.jsonl"))
    report, _ = make_report(tmp_path, [out])
    grok = report["candidates"][XAI]
    assert (grok["n"], grok["agreement"], grok["kappa"], grok["verdict"]) == (100, 1.0, 1.0, "pass")

    # On the D-099 sample a judge that passes everything is no longer close to the bar's
    # agreement for free through forced fails: its kappa is 0.
    ref = [r["reference"]["spec_adherence"] for r in sample_rows()]
    always = report["baselines"]["always_pass"]
    assert always["n"] == 100
    assert always["agreement"] == pytest.approx(sum(ref) / 100)
    assert always["kappa"] == pytest.approx(0.0)
    assert always["verdict"] == bar_verdict(always["agreement"], always["kappa"]) == "fail"
    assert report["baselines"]["always_fail"]["verdict"] == "fail"


def test_all_rows_is_none_without_a_full_run_dir(
    three_runs: dict[str, Path], tmp_path: Path
) -> None:
    report, _ = make_report(tmp_path, list(three_runs.values()))
    for spec in three_runs:
        candidate = report["candidates"][spec]
        assert "all_rows" in candidate and candidate["all_rows"] is None, spec


def test_all_rows_numbers_come_from_a_full_run_dir_and_never_change_the_verdict(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rejudge.py re-judge of all of M1, forced rows included, is compared row by row with
    M1's reference verdicts. The candidate passes everything; HARD forcing pins the forced rows
    at 0 for both judges."""
    use_xai(monkeypatch, FakeXAI(oracle))
    grok = tmp_path / "grok"
    assert ja(*run_argv(XAI, small_sample(tmp_path), grok, "--concurrency", "1")) == 0
    use_xai(monkeypatch, FakeXAI(always_pass))
    full = tmp_path / "full-m1"
    argv = [str(REPO_ROOT / M1_DIR), "--judge", XAI, "--out", str(full), "--concurrency", "1"]
    assert rejudge(*argv) == 0

    out = tmp_path / "report" / "report.json"
    argv = ["--report", "--dirs", str(grok), "--out", str(out), "--full-dirs", str(full)]
    assert ja(*argv) == 0
    candidate = json.loads(out.read_text(encoding="utf-8"))["candidates"][XAI]

    m1 = [r for k, r in source_rows().items() if k[0] == M1_DIR]
    ref = [r["score"]["spec_adherence"] for r in m1]
    cand = [0 if forced(r) else 1 for r in m1]
    label = {1: "pass", 0: "fail"}
    confusion = {a: {b: 0 for b in ("pass", "fail")} for a in ("pass", "fail")}
    for a, b in zip(ref, cand, strict=True):
        confusion[label[a]][label[b]] += 1
    all_rows = candidate["all_rows"]
    assert all_rows["n"] == len(m1) == 240
    assert all_rows["agreement"] == pytest.approx(
        sum(a == b for a, b in zip(ref, cand, strict=True)) / len(m1)
    )
    assert all_rows["kappa"] == pytest.approx(cohens_kappa(ref, cand))
    assert all_rows["confusion"] == confusion

    # The verdict is still the oracle's on the sample rows.
    assert (candidate["n"], candidate["agreement"], candidate["verdict"]) == (20, 1.0, "pass")


# ---------------------------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------------------------


def test_cohens_kappa_matches_the_hand_computed_values() -> None:
    module = _load()
    # 40 both pass, 10 ref pass / cand fail, 5 ref fail / cand pass, 45 both fail:
    # po = 0.85, pe = 0.5 x 0.45 + 0.5 x 0.55 = 0.5, kappa = 0.7.
    ref = [1] * 50 + [0] * 50
    cand = [1] * 40 + [0] * 10 + [1] * 5 + [0] * 45
    assert module.cohens_kappa(ref, cand) == pytest.approx(0.7)
    assert module.cohens_kappa(ref, ref) == pytest.approx(1.0)
    assert module.cohens_kappa([1] * 10, [1] * 10) == 1.0, "identical constant ratings"
    rng = random.Random(0)
    noisy = [rng.randint(0, 1) for _ in ref]
    assert module.cohens_kappa(ref, noisy) == pytest.approx(cohens_kappa(ref, noisy))
