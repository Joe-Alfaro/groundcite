"""The judge reverts to OpenAI gpt-5.5, and Grok is rejected (D-107, replacing D-080 and D-095).

- ``config/models.toml`` pins the judge as ``openai:gpt-5.5-2026-04-23`` with the fixed
  ``[decoding.openai]`` parameters, the same effective decoding that scored M1 and the H1 pilot.
  The eval rubric and its hash are unchanged.
- ``[judge_candidates]`` keeps the three measured candidates for the record, each marked in
  ``status``: Grok ``rejected`` (it failed the D-096/D-099 agreement check), Sonnet and Haiku
  ``rejected`` or ``not-selected``.
- Every judged step that defaults to the configured judge uses it: the dataset build's hint
  screen and rubric judge, and ``eval.py``, whose ``results.json`` pins record the judge model and
  its effective decoding. Dataset v1 stays as built, with its Opus filter pins.
- The docs state that the eval judge is gpt-5.5 (a different family from the Anthropic teacher)
  and disclose that dataset v1's filter used Claude Opus: README, the architecture's judge
  section, the requirements' R2 and G12, the write-up's limitations, and the ablation and pilot
  judge notes. The write-up reports the judge agreement study (``results/judge_agreement``).

The configured judge is replaced in-process by a recording fake under the ``openai:`` prefix,
so no key or network is needed.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from distill_data import candidate_row, cite_reply, run_script, scenario_row, write_jsonl

from groundcite import backends
from groundcite.backends import MockJudge
from groundcite.config import decoding_params, load_models_config

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
README = REPO_ROOT / "README.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
REQUIREMENTS = REPO_ROOT / "docs" / "requirements" / "small-model-finetune.md"
WRITEUP = REPO_ROOT / "docs" / "writeup.md"
ABLATION_README = REPO_ROOT / "results" / "ablation" / "README.md"
PILOT_README = REPO_ROOT / "results" / "rollout_pilot" / "README.md"
AGREEMENT_REPORT = REPO_ROOT / "results" / "judge_agreement" / "report.json"
DATASET_V1_MANIFEST = REPO_ROOT / "data" / "datasets" / "v1" / "manifest.json"
JUDGE = "openai:gpt-5.5-2026-04-23"
OPUS = "anthropic:claude-opus-5-5"
GROK = "xai:grok-4.20-0309-reasoning"
SONNET = "anthropic:claude-sonnet-5-5"
HAIKU = "anthropic:claude-haiku-5-5"
PINNED_RUBRIC = "636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf"
SCORED_BY_GPT55 = ("results/ablation/m1", "results/rollout_pilot/pilot1")


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


class RecordingJudge:
    """Stands in for the OpenAI backend: answers like the mock judge (rubric and hint screen) and
    records which spec served each request and whether it was the screen."""

    seen: list[tuple[str, str]] = []
    lock = threading.Lock()

    def __init__(self, spec: str, transport: Any = None) -> None:
        self.name = spec
        self._inner = MockJudge()

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        kind = "screen" if system is not None and '"meta_leak"' in system else "rubric"
        with RecordingJudge.lock:
            RecordingJudge.seen.append((self.name, kind))
        return self._inner.generate(messages, system, params)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[RecordingJudge]:
    monkeypatch.setattr(RecordingJudge, "seen", [])
    monkeypatch.setitem(backends.BACKENDS, "openai:", RecordingJudge)
    return RecordingJudge


# ---------------------------------------------------------------------------------------------
# (1) Config: the judge, its decoding, the unchanged rubric, the rejected candidates
# ---------------------------------------------------------------------------------------------


def test_the_configured_judge_is_gpt55() -> None:
    assert load_models_config()["judge"]["model"] == JUDGE


def test_the_config_comment_explains_the_revert_not_the_opus_switch() -> None:
    """The [judge] block cites D-107 and no longer describes Claude Opus as the judge."""
    text = (REPO_ROOT / "config" / "models.toml").read_text(encoding="utf-8")
    block = text.split("[judge]", 1)[1].split("\n[", 1)[0]
    assert "D-107" in block
    assert "Claude Opus replaced" not in block
    assert "self-preference risk the R29" not in block


def test_the_judge_decoding_is_the_fixed_openai_table() -> None:
    cfg = load_models_config()
    params = decoding_params(cfg["judge"]["model"])
    assert params == cfg["decoding"]["openai"]
    assert params["max_completion_tokens"] > 0
    assert params["reasoning_effort"] in {"minimal", "low", "medium", "high"}
    if "temperature" in params:
        assert params["temperature"] == 0


@pytest.mark.parametrize("run", SCORED_BY_GPT55)
def test_the_judge_decoding_matches_what_scored_m1_and_the_pilot(run: str) -> None:
    """M1 and the pilot stay directly comparable (D-107) only if the judge runs with the same
    effective decoding they recorded. A rejected sampling param (temperature) may be dropped."""
    results = json.loads((REPO_ROOT / run / "results.json").read_text(encoding="utf-8"))
    assert results["pins"]["judge_model"] == JUDGE
    recorded = results["pins"]["decoding"][JUDGE]
    params = decoding_params(load_models_config()["judge"]["model"])
    assert recorded == {k: v for k, v in params.items() if k in recorded}
    assert set(params) - set(recorded) <= {"temperature"}, params


def test_the_rubric_and_its_hash_are_unchanged() -> None:
    from groundcite.judge import RUBRIC_VERSION, rubric_sha256

    assert RUBRIC_VERSION == "v1"
    assert rubric_sha256() == PINNED_RUBRIC


def test_judge_candidates_keep_the_measured_models_for_the_record() -> None:
    candidates = load_models_config()["judge_candidates"]
    assert candidates["models"] == [GROK, SONNET, HAIKU]
    assert candidates["reference"] == JUDGE
    assert JUDGE not in candidates["models"]


def test_judge_candidates_are_marked_rejected_or_not_selected() -> None:
    candidates = load_models_config()["judge_candidates"]
    status = candidates.get("status")
    assert isinstance(status, dict), "[judge_candidates] needs status = { <spec> = <mark> }"
    assert set(status) == set(candidates["models"]), "every measured candidate is marked"
    assert status[GROK] == "rejected"
    for spec in (SONNET, HAIKU):
        assert status[spec] in {"rejected", "not-selected"}, (spec, status[spec])


def test_the_agreement_report_records_the_rejection() -> None:
    """The committed study backs the config: every candidate failed the bar, Grok with 0 wins."""
    report = json.loads(AGREEMENT_REPORT.read_text(encoding="utf-8"))
    assert report["reference_judge"] == JUDGE
    cands = report["candidates"]
    assert set(cands) == {GROK, SONNET, HAIKU}
    assert all(c["verdict"] == "fail" for c in cands.values())
    assert cands[GROK]["adjudicated"] == cands[GROK]["disagreements"] > 0
    assert cands[GROK]["wins"] == 0


def test_dataset_v1_keeps_its_opus_filter_pins() -> None:
    """Dataset v1 is kept as built (D-107): its manifest still records the Opus filter judge."""
    manifest = json.loads(DATASET_V1_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["judge_model"] == OPUS


# ---------------------------------------------------------------------------------------------
# (1) The configured judge serves the hint screen, the rubric judge and the eval pins
# ---------------------------------------------------------------------------------------------


def test_build_dataset_screens_and_judges_with_the_configured_judge(
    env: Path, tmp_path: Path, recorder: type[RecordingJudge]
) -> None:
    rows = [scenario_row(i) for i in range(1, 4)]
    cands = [candidate_row(r, 0, cite_reply(r["gold"]["supporting_ids"][0])) for r in rows]
    path = write_jsonl(tmp_path / "candidates.jsonl", cands)
    out = tmp_path / "datasets" / "v1"
    # No --judge: the build must default to config/models.toml.
    assert run_script("build_dataset", ["--candidates", str(path), "--out", str(out)]) == 0

    kinds = {kind for _, kind in recorder.seen}
    assert kinds == {"screen", "rubric"}, f"both the screen and the judge run: {recorder.seen}"
    assert {name for name, _ in recorder.seen} == {JUDGE}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_model"] == JUDGE
    assert manifest["hint_screen_calls"] == sum(k == "screen" for _, k in recorder.seen)


def test_eval_results_pin_the_configured_judge(
    env: Path, tmp_path: Path, recorder: type[RecordingJudge]
) -> None:
    from groundcite.cli import main

    out = tmp_path / "results"
    argv = ["--model", "mock:compliant", "--eval-set", str(ABLATION), "--no-base"]
    argv += ["--limit", "2", "--allow-dirty", "--out", str(out)]
    try:
        code = main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 0
    [run] = [p for p in out.iterdir() if p.is_dir()]
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["pins"]["judge_model"] == JUDGE
    assert results["pins"]["decoding"][JUDGE] == decoding_params(JUDGE)
    rows = [json.loads(line) for line in (run / "transcripts.jsonl").read_text().splitlines()]
    assert rows and all(r["judge_model"] == JUDGE for r in rows)
    assert {name for name, _ in recorder.seen} == {JUDGE}
    md = (run / "results.md").read_text(encoding="utf-8")
    assert JUDGE in md


# ---------------------------------------------------------------------------------------------
# (3) Docs: gpt-5.5 is the cross-family eval judge; dataset v1's filter used Opus
# ---------------------------------------------------------------------------------------------

CROSS_FAMILY = re.compile(
    r"cross[- ]family|different (vendor|model )?family|differs? in family|another family",
    re.IGNORECASE,
)
DECISION_ID = re.compile(r"\bD-\d{3}\b")


def _section(text: str, heading: re.Pattern[str]) -> str:
    """The body under the first Markdown heading matching ``heading``, up to the next heading of
    the same or a higher level."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m and heading.search(m.group(2)):
            level = len(m.group(1))
            body = []
            for nxt in lines[i + 1 :]:
                n = re.match(r"^(#+)\s", nxt)
                if n and len(n.group(1)) <= level:
                    break
                body.append(nxt)
            return "\n".join(body)
    raise AssertionError(f"no heading matching {heading.pattern!r}")


def _paragraphs(text: str) -> list[str]:
    """Blank-line paragraphs, with each top-level list item split out as its own paragraph and
    line wraps collapsed to single spaces."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        out.extend(" ".join(p.split()) for p in re.split(r"\n(?=- )", block) if p.strip())
    return out


def _states_cross_family_judge(text: str) -> bool:
    return any(
        "gpt-5.5" in p and CROSS_FAMILY.search(p) and re.search(r"teacher", p, re.IGNORECASE)
        for p in _paragraphs(text)
    )


def _discloses_the_opus_filter(text: str) -> bool:
    """One paragraph says dataset v1 was filtered with Claude Opus while gpt-5.5 judges evals."""
    return any(
        re.search(r"dataset v1|\bv1\b", p, re.IGNORECASE)
        and re.search(r"filter", p, re.IGNORECASE)
        and re.search(r"claude-opus-5-5|Claude Opus|\bOpus\b", p)
        and "gpt-5.5" in p
        for p in _paragraphs(text)
    )


def _block(text: str, start: re.Pattern[str]) -> str:
    """From the first line matching ``start`` to the next blank line or heading."""
    m = start.search(text)
    assert m, f"nothing matches {start.pattern!r}"
    return re.split(r"\n\s*\n|\n#", text[m.start() :], maxsplit=1)[0]


def test_readme_names_the_gpt55_judge_and_its_key() -> None:
    text = README.read_text(encoding="utf-8")
    assert JUDGE in text
    assert re.search(r"OPENAI_API_KEY[^.\n]*(required|needed)[^.\n]*judge", text), (
        "the env-var note says the judge needs OPENAI_API_KEY"
    )
    assert not re.search(r"ANTHROPIC_API_KEY[^.\n]*required for the pinned judge", text)
    assert "The judge is Claude Opus" not in text


def test_readme_states_the_cross_family_judge_and_the_opus_filter() -> None:
    text = README.read_text(encoding="utf-8")
    assert _states_cross_family_judge(text), "say gpt-5.5 is a different family from the teacher"
    assert _discloses_the_opus_filter(text), "disclose that dataset v1's filter used Opus"
    assert not DECISION_ID.search(text), "decision ids live in the private log, not the README"


def test_architecture_judge_section_pins_gpt55() -> None:
    body = _section(ARCHITECTURE.read_text(encoding="utf-8"), re.compile(r"^Judge design"))
    pinned = _block(body, re.compile(r"\*\*Pinned judge:\*\*"))
    assert JUDGE in pinned
    assert OPUS not in pinned.split(JUDGE, 1)[0], "the pinned judge is named first, not Opus"
    assert "same model as the teacher" not in body
    assert _states_cross_family_judge(body)
    assert _discloses_the_opus_filter(body)


def test_requirements_r2_names_the_gpt55_judge() -> None:
    text = REQUIREMENTS.read_text(encoding="utf-8")
    r2 = text.split("**R2 —", 1)[1].split("\n**R3 —", 1)[0]
    assert f"The judge is `{JUDGE}`" in r2
    assert f"The judge is `{OPUS}`" not in r2


def test_requirements_g12_describes_the_gpt55_judge_and_the_agreement_check() -> None:
    text = REQUIREMENTS.read_text(encoding="utf-8")
    g12 = text.split("**G12 —", 1)[1].split("\n- **", 1)[0]
    assert "gpt-5.5" in g12
    assert "same model as the teacher" not in g12
    assert CROSS_FAMILY.search(g12) and re.search(r"teacher", g12)
    assert re.search(r"agreement (check|study)", g12, re.IGNORECASE)
    assert re.search(r"Grok", g12) and re.search(r"rejected", g12, re.IGNORECASE)
    assert re.search(r"dataset v1", g12, re.IGNORECASE) and re.search(r"Opus", g12)
    assert re.search(r"filter", g12, re.IGNORECASE)


def test_writeup_limitations_state_the_gpt55_judge_and_the_opus_filter() -> None:
    body = _section(WRITEUP.read_text(encoding="utf-8"), re.compile(r"Limitations"))
    assert "The judge shares the teacher's family" not in body
    assert "re-judged by Claude Opus" not in body
    assert _states_cross_family_judge(body)
    assert _discloses_the_opus_filter(body)
    assert re.search(r"self[- ]preference", body, re.IGNORECASE), (
        "name the self-preference risk the Opus filter carries"
    )


def test_writeup_validity_audit_no_longer_plans_an_opus_rejudge() -> None:
    body = _section(WRITEUP.read_text(encoding="utf-8"), re.compile(r"Judge validity audit"))
    assert "re-judged by Claude Opus" not in body


@pytest.mark.parametrize("path", [ABLATION_README, PILOT_README], ids=["ablation", "pilot"])
def test_results_judge_notes_name_gpt55_as_the_eval_judge(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?im)^(#+\s*judge note\b.*|\*\*judge note[.:]?\*\*.*)$", text)
    assert match, f"{path.relative_to(REPO_ROOT)} has a 'Judge note'"
    note = _block(text, re.compile(re.escape(match.group(0))))
    assert JUDGE in note or "gpt-5.5" in note
    assert re.search(r"eval(uation)? judge|project judge|current judge", note, re.IGNORECASE)
    assert f"now `{OPUS}`" not in note
    assert "re-judged by Claude Opus" not in note
    assert _discloses_the_opus_filter(note), "the note discloses dataset v1's Opus filter"


# ---------------------------------------------------------------------------------------------
# (3) Docs: the judge agreement study in the write-up
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def study() -> str:
    """The write-up section headed "Judge agreement ..." ("" if missing, so each test fails)."""
    try:
        return _section(WRITEUP.read_text(encoding="utf-8"), re.compile(r"judge agreement", re.I))
    except AssertionError:
        return ""


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return dict(json.loads(AGREEMENT_REPORT.read_text(encoding="utf-8")))


def test_study_points_at_the_committed_results(study: str) -> None:
    assert study, "docs/writeup.md has a 'Judge agreement' section"
    assert "results/judge_agreement" in study
    assert "data/judge_agreement/sample.jsonl" in study


def test_study_states_the_method(study: str) -> None:
    assert re.search(r"\b100\b", study) and re.search(r"\b60\b", study)
    assert re.search(r"\b40\b", study)
    assert re.search(r"judge-decided|no HARD check", study, re.IGNORECASE)
    assert re.search(r"seed[^.\n]*\b96\b", study, re.IGNORECASE)
    assert re.search(r"rubric", study, re.IGNORECASE) and re.search(r"unchanged", study)
    assert re.search(r"90%|0\.90", study) and re.search(r"0\.75", study)
    assert re.search(r"50%|half", study), "state the pre-registered bar"


def test_study_table_matches_the_report(study: str, report: dict[str, Any]) -> None:
    rows = [line for line in study.splitlines() if line.lstrip().startswith("|")]
    assert len(rows) >= 2 + len(report["candidates"]), "a Markdown table, one row per candidate"
    for spec, c in report["candidates"].items():
        model = spec.split(":", 1)[1]
        [row] = [r for r in rows if model in r]
        assert f"{c['agreement']:.2f}" in row, (model, row)
        assert f"{c['kappa']:.2f}" in row, (model, row)
        assert re.search(rf"\b{c['wins']}\s*(/|of)\s*{c['adjudicated']}\b", row), (model, row)
        assert re.search(r"\bfail\b", row, re.IGNORECASE), (model, row)
    base = report["baselines"]["always_pass"]
    [row] = [r for r in rows if re.search(r"always[- ]pass", r, re.IGNORECASE)]
    assert f"{base['agreement']:.2f}" in row and f"{base['kappa']:.2f}" in row


def test_study_describes_the_blind_adjudication(study: str, report: dict[str, Any]) -> None:
    assert re.search(r"blind", study, re.IGNORECASE)
    assert re.search(r"adjudicat", study, re.IGNORECASE)
    n_items = sum(
        1 for line in (AGREEMENT_REPORT.parent / "adjudication.jsonl").open() if line.strip()
    )
    assert re.search(rf"\b{n_items}\b", study), "say how many disagreements were adjudicated"


def test_study_states_the_conclusion(study: str) -> None:
    assert re.search(r"Grok[^.]*rejected|rejected[^.]*Grok", study, re.IGNORECASE)
    assert "gpt-5.5" in study
    assert re.search(r"(stays|remains|kept)[^.]*judge|judge[^.]*(stays|remains|kept)", study)


def test_study_states_its_limits(study: str) -> None:
    assert re.search(r"\b6\b[^.]*(reference )?fail", study), "only 6 reference fails"
    assert re.search(r"kappa[^.]*(fragile|unstable|sensitive)", study, re.IGNORECASE)
