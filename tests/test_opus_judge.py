"""The judge switches from OpenAI gpt-5.5 to Anthropic Claude Opus (D-080).

- ``config/models.toml`` pins the judge as ``anthropic:claude-opus-5-5`` with the fixed
  ``[decoding.anthropic]`` parameters. The eval rubric and its hash are unchanged.
- Every judged step that defaults to the configured judge uses it: the dataset build's hint
  screen and rubric judge, and ``eval.py``, whose ``results.json`` pins record the judge model and
  its effective decoding.
- The docs state the switch and the self-preference risk (the judge now shares a family with the
  teacher): README, the write-up's limitations section and the ablation README's judge note (M1
  and the H1 pilot were scored by gpt-5.5; side-by-side comparisons use re-judged runs).

The configured judge is replaced in-process by a recording fake under the ``anthropic:`` prefix,
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
WRITEUP = REPO_ROOT / "docs" / "writeup.md"
ABLATION_README = REPO_ROOT / "results" / "ablation" / "README.md"
OPUS_JUDGE = "anthropic:claude-opus-5-5"
OLD_JUDGE = "openai:gpt-5.5-2026-04-23"
PINNED_RUBRIC = "636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf"


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


class RecordingJudge:
    """Stands in for the Anthropic backend: answers like the mock judge (rubric and hint screen)
    and records which spec served each request and whether it was the screen."""

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
    monkeypatch.setitem(backends.BACKENDS, "anthropic:", RecordingJudge)
    return RecordingJudge


# ---------------------------------------------------------------------------------------------
# (1) Config: the judge, its decoding, the unchanged rubric
# ---------------------------------------------------------------------------------------------


def test_the_configured_judge_is_claude_opus() -> None:
    assert load_models_config()["judge"]["model"] == OPUS_JUDGE


def test_the_config_comment_no_longer_claims_a_different_family_from_the_teacher() -> None:
    """The [judge] block explains the D-080 choice instead of the old D-019 reasoning."""
    text = (REPO_ROOT / "config" / "models.toml").read_text(encoding="utf-8")
    block = text.split("[judge]", 1)[1].split("\n[", 1)[0]
    assert "differs in family from the teacher" not in block
    assert "D-080" in block


def test_the_judge_decoding_is_the_fixed_anthropic_table() -> None:
    cfg = load_models_config()
    params = decoding_params(cfg["judge"]["model"])
    assert params == cfg["decoding"]["anthropic"]
    assert params["max_tokens"] > 0
    assert params["output_config"]["effort"] in {"low", "medium", "high", "xhigh", "max"}


def test_the_rubric_and_its_hash_are_unchanged() -> None:
    from groundcite.judge import RUBRIC_VERSION, rubric_sha256

    assert RUBRIC_VERSION == "v1"
    assert rubric_sha256() == PINNED_RUBRIC


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
    assert {name for name, _ in recorder.seen} == {OPUS_JUDGE}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_model"] == OPUS_JUDGE
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
    assert results["pins"]["judge_model"] == OPUS_JUDGE
    assert results["pins"]["decoding"][OPUS_JUDGE] == decoding_params(OPUS_JUDGE)
    rows = [json.loads(line) for line in (run / "transcripts.jsonl").read_text().splitlines()]
    assert rows and all(r["judge_model"] == OPUS_JUDGE for r in rows)
    assert {name for name, _ in recorder.seen} == {OPUS_JUDGE}
    md = (run / "results.md").read_text(encoding="utf-8")
    assert OPUS_JUDGE in md


# ---------------------------------------------------------------------------------------------
# (5) Docs: the switch and the self-preference risk
# ---------------------------------------------------------------------------------------------

SELF_PREFERENCE = re.compile(r"self[- ]preference", re.IGNORECASE)


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


def test_readme_names_the_opus_judge_and_its_key() -> None:
    text = README.read_text(encoding="utf-8")
    assert OPUS_JUDGE in text
    assert re.search(r"ANTHROPIC_API_KEY[^.\n]*(required|needed)[^.\n]*judge", text), (
        "the env-var note says the judge needs ANTHROPIC_API_KEY"
    )
    assert not re.search(r"OPENAI_API_KEY[^.\n]*required for the pinned judge", text)


def test_readme_states_the_switch_and_the_self_preference_risk() -> None:
    text = README.read_text(encoding="utf-8")
    assert SELF_PREFERENCE.search(text)
    assert "gpt-5.5" in text, "say which judge the earlier numbers used"
    assert "D-080" not in text, "decision ids live in the private log, not the README"


def test_writeup_limitations_state_the_judge_switch_and_self_preference() -> None:
    body = _section(WRITEUP.read_text(encoding="utf-8"), re.compile(r"Limitations"))
    assert SELF_PREFERENCE.search(body)
    assert "claude-opus-5-5" in body or "Claude Opus" in body
    assert "gpt-5.5" in body
    assert re.search(r"teacher", body, re.IGNORECASE), "the risk is judge = teacher family"
    assert re.search(r"re-?judged", body, re.IGNORECASE)


def test_ablation_readme_has_a_judge_note() -> None:
    text = ABLATION_README.read_text(encoding="utf-8")
    match = re.search(r"(?im)^(#+\s*judge note\b.*|\*\*judge note[.:]?\*\*.*)$", text)
    assert match, "results/ablation/README.md has a 'Judge note'"
    note = text[match.start() :].split("\n## ", 1)[0]
    assert OLD_JUDGE in note or "gpt-5.5" in note
    assert re.search(r"\bM1\b", note) and re.search(r"pilot", note, re.IGNORECASE)
    assert re.search(r"re-?judged", note, re.IGNORECASE)
    assert re.search(r"side[- ]by[- ]side", note, re.IGNORECASE)
    assert "rejudge.py" in note, "name the tool that produces the comparable runs"
