"""Tests for teacher distillation (R1, R4, R20, R25, GOTCHA G22, G30).

`scripts/distill.py --scenarios P --teacher <spec> --k 2 --out DIR [--limit N] [--dry-run]
[--resume]` writes `DIR/candidates.jsonl`. The teacher prompt lives in `groundcite.distill`
(a versioned template file that loads SPEC.md by path and shows the privileged gold hint).
Fake teachers are registered under `backends.MOCKS`; sockets are blocked and keys unset.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest
from distill_data import (
    ABSTAIN_REPLY,
    REPO_ROOT,
    RecordingTeacher,
    read_jsonl,
    run_script,
    scenario_row,
    write_jsonl,
)

from groundcite import backends
from groundcite.config import load_models_config
from groundcite.evalset import parse_row
from groundcite.spec import SPEC_PATH, load_spec

K = 2


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


@pytest.fixture
def teacher(monkeypatch: pytest.MonkeyPatch) -> type[RecordingTeacher]:
    monkeypatch.setattr(RecordingTeacher, "calls", [])
    monkeypatch.setattr(RecordingTeacher, "gold_ids", {})
    monkeypatch.setitem(backends.MOCKS, "teacher_probe", RecordingTeacher)
    return RecordingTeacher


def _scenarios(tmp_path: Path, n: int = 4) -> tuple[Path, list[dict[str, Any]]]:
    rows = [
        scenario_row(
            i, category="clean" if i % 2 else "no_snippet", mode="cite" if i % 2 else "abstain"
        )
        for i in range(1, n + 1)
    ]
    return write_jsonl(tmp_path / "pool.jsonl", rows), rows


def _distill(*args: str) -> int:
    return run_script("distill", list(args))


# --- teacher prompt ------------------------------------------------------------------------


def test_teacher_prompt_file_is_versioned_and_tracked() -> None:
    from groundcite.distill import TEACHER_PROMPT_PATH, TEACHER_PROMPT_VERSION

    path = Path(TEACHER_PROMPT_PATH)
    assert path.is_file()
    assert TEACHER_PROMPT_VERSION and TEACHER_PROMPT_VERSION in path.name
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)], cwd=REPO_ROOT, capture_output=True
    )
    assert tracked.returncode == 0, f"{path} is not tracked by git"


def test_teacher_prompt_loads_the_spec_by_path(tmp_path: Path) -> None:
    from groundcite.distill import TEACHER_PROMPT_PATH, render_teacher_prompt

    template = Path(TEACHER_PROMPT_PATH).read_text(encoding="utf-8")
    assert load_spec(SPEC_PATH) not in template, "the template must load SPEC.md, not copy it"
    scenario = parse_row(scenario_row(1))
    system, messages = render_teacher_prompt(scenario)
    assert load_spec() in system

    other_spec = tmp_path / "SPEC.md"
    other_spec.write_text("Sentinel spec sentence for the teacher prompt test.\n", encoding="utf-8")
    system, _ = render_teacher_prompt(scenario, spec_path=other_spec)
    assert "Sentinel spec sentence for the teacher prompt test." in system


def test_teacher_prompt_sha_covers_the_rendered_template(tmp_path: Path) -> None:
    from groundcite.distill import teacher_prompt_sha256

    sha = teacher_prompt_sha256()
    assert len(sha) == 64 and sha == teacher_prompt_sha256()
    other_spec = tmp_path / "SPEC.md"
    other_spec.write_text("A different spec.\n", encoding="utf-8")
    assert teacher_prompt_sha256(spec_path=other_spec) != sha


def test_teacher_prompt_shows_every_snippet_and_the_conversation() -> None:
    from groundcite.distill import render_teacher_prompt

    row = scenario_row(2)
    system, messages = render_teacher_prompt(parse_row(row))
    text = "\n".join([system] + [m["content"] for m in messages])
    for snippet in row["snippets"]:
        assert f"[{snippet['id']}]" in text
        assert snippet["text"] in text
    assert messages[-1]["role"] == "user"
    assert row["messages"][-1]["content"] in messages[-1]["content"]


def test_cite_hint_names_the_gold_mode_and_supporting_id() -> None:
    from groundcite.distill import HINT_HEADER, render_hint, render_teacher_prompt

    row = scenario_row(3, mode="cite")
    scenario = parse_row(row)
    hint = render_hint(scenario)
    assert HINT_HEADER in hint
    assert "cite" in hint.lower()
    assert row["gold"]["supporting_ids"][0] in hint
    system, messages = render_teacher_prompt(scenario)
    assert hint in "\n".join([system] + [m["content"] for m in messages])


def test_abstain_hint_names_abstain_and_no_supporting_id() -> None:
    from groundcite.distill import HINT_HEADER, render_hint

    row = scenario_row(4, mode="abstain")
    hint = render_hint(parse_row(row))
    assert HINT_HEADER in hint
    assert "abstain" in hint.lower()
    assert row["snippets"][0]["id"] not in hint


def test_eval_prompt_never_shows_the_hint() -> None:
    from groundcite.distill import HINT_HEADER, render_hint
    from groundcite.prompts import STRATEGIES, render_messages

    scenario = parse_row(scenario_row(5))
    for strategy in STRATEGIES:
        system, messages = render_messages(scenario, strategy)
        text = "\n".join([system] + [m["content"] for m in messages])
        assert HINT_HEADER not in text
        assert render_hint(scenario) not in text


# --- distill.py ------------------------------------------------------------------------------


def test_distill_writes_k_candidates_per_scenario(
    env: Path, teacher: type[RecordingTeacher], tmp_path: Path
) -> None:
    from groundcite.distill import (
        HINT_HEADER,
        TEACHER_PROMPT_VERSION,
        teacher_prompt_sha256,
    )

    pool, rows = _scenarios(tmp_path)
    teacher.gold_ids = {
        r["id"]: r["gold"]["supporting_ids"][0] for r in rows if r["gold"]["supporting_ids"]
    }
    out = tmp_path / "distill"
    assert (
        _distill(
            "--scenarios",
            str(pool),
            "--teacher",
            "mock:teacher_probe",
            "--k",
            str(K),
            "--out",
            str(out),
        )
        == 0
    )

    cands = read_jsonl(out / "candidates.jsonl")
    assert len(cands) == len(rows) * K
    assert len(teacher.calls) == len(rows) * K
    by_scenario: dict[str, list[int]] = {}
    for cand in cands:
        by_scenario.setdefault(cand["scenario_id"], []).append(cand["candidate_index"])
        assert cand["teacher"] == "mock:teacher_probe"
        assert cand["teacher_prompt_version"] == TEACHER_PROMPT_VERSION
        assert cand["teacher_prompt_sha256"] == teacher_prompt_sha256()
        assert cand["error"] is None
        assert isinstance(cand["reply"], str) and cand["reply"]
    assert {sid: sorted(idx) for sid, idx in by_scenario.items()} == {
        r["id"]: list(range(K)) for r in rows
    }
    # Each candidate carries its scenario, so build_dataset needs only the candidates file.
    assert [c["scenario"] for c in cands[::K]] == rows

    for system, messages in teacher.calls:
        text = "\n".join([system or ""] + [m["content"] for m in messages])
        assert load_spec() in text
        assert HINT_HEADER in text


def test_distill_limit_takes_the_first_scenarios(
    env: Path, teacher: type[RecordingTeacher], tmp_path: Path
) -> None:
    pool, rows = _scenarios(tmp_path, n=5)
    out = tmp_path / "distill"
    assert (
        _distill(
            "--scenarios",
            str(pool),
            "--teacher",
            "mock:teacher_probe",
            "--k",
            "2",
            "--limit",
            "2",
            "--out",
            str(out),
        )
        == 0
    )
    ids = [c["scenario_id"] for c in read_jsonl(out / "candidates.jsonl")]
    assert sorted(set(ids)) == [rows[0]["id"], rows[1]["id"]]


def test_distill_rejects_k_below_two(
    env: Path, teacher: type[RecordingTeacher], tmp_path: Path
) -> None:
    pool, _ = _scenarios(tmp_path)
    out = tmp_path / "distill"
    assert (
        _distill(
            "--scenarios",
            str(pool),
            "--teacher",
            "mock:teacher_probe",
            "--k",
            "1",
            "--out",
            str(out),
        )
        == 2
    )
    assert not (out / "candidates.jsonl").exists()


def test_dry_run_prints_the_call_count_and_needs_no_key(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool, rows = _scenarios(tmp_path, n=3)
    out = tmp_path / "distill"
    # Default teacher is the pinned Anthropic model, and no key is set.
    assert _distill("--scenarios", str(pool), "--k", "2", "--out", str(out), "--dry-run") == 0
    stdout = capsys.readouterr().out
    assert f"teacher calls: {len(rows) * 2}" in stdout
    assert load_models_config()["teacher"]["model"] in stdout
    assert not out.exists() or not any(out.iterdir())


def test_missing_teacher_key_exits_two_before_any_work(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool, _ = _scenarios(tmp_path)
    out = tmp_path / "distill"
    code = _distill(
        "--scenarios",
        str(pool),
        "--teacher",
        "anthropic:claude-opus-5-5",
        "--k",
        "2",
        "--out",
        str(out),
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (out / "candidates.jsonl").exists()


def test_existing_output_without_resume_exits_two_untouched(
    env: Path, teacher: type[RecordingTeacher], tmp_path: Path
) -> None:
    pool, _ = _scenarios(tmp_path)
    out = tmp_path / "distill"
    out.mkdir()
    existing = out / "candidates.jsonl"
    existing.write_text('{"scenario_id": "keep-me"}\n', encoding="utf-8")
    before = hashlib.sha256(existing.read_bytes()).hexdigest()
    assert (
        _distill(
            "--scenarios",
            str(pool),
            "--teacher",
            "mock:teacher_probe",
            "--k",
            "2",
            "--out",
            str(out),
        )
        == 2
    )
    assert hashlib.sha256(existing.read_bytes()).hexdigest() == before
    assert teacher.calls == []


def test_resume_only_generates_missing_candidates(
    env: Path, teacher: type[RecordingTeacher], tmp_path: Path
) -> None:
    pool, rows = _scenarios(tmp_path, n=4)
    out = tmp_path / "distill"
    base = [
        "--scenarios",
        str(pool),
        "--teacher",
        "mock:teacher_probe",
        "--k",
        "2",
        "--out",
        str(out),
    ]
    assert _distill(*base, "--limit", "2") == 0
    first = (out / "candidates.jsonl").read_text(encoding="utf-8")
    teacher.calls.clear()

    assert _distill(*base, "--resume") == 0
    assert len(teacher.calls) == 2 * 2
    text = (out / "candidates.jsonl").read_text(encoding="utf-8")
    assert text.startswith(first), "resume must append, not rewrite"
    keys = [(c["scenario_id"], c["candidate_index"]) for c in read_jsonl(out / "candidates.jsonl")]
    assert len(keys) == len(set(keys)) == len(rows) * 2


def test_teacher_errors_are_recorded_not_dropped(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingTeacher:
        """Fails on its second call only."""

        name = "mock:teacher_fail"
        calls = 0

        def generate(self, messages: Any, system: Any, params: Any) -> str:
            FailingTeacher.calls += 1
            if FailingTeacher.calls == 2:
                raise backends.BackendError("teacher overloaded")
            return ABSTAIN_REPLY

    monkeypatch.setitem(backends.MOCKS, "teacher_fail", FailingTeacher)
    pool, rows = _scenarios(tmp_path, n=2)
    out = tmp_path / "distill"
    assert (
        _distill(
            "--scenarios",
            str(pool),
            "--teacher",
            "mock:teacher_fail",
            "--k",
            "2",
            "--out",
            str(out),
        )
        == 0
    )
    cands = read_jsonl(out / "candidates.jsonl")
    assert len(cands) == 4
    failed = [c for c in cands if c["error"]]
    assert len(failed) == 1 and "teacher overloaded" in failed[0]["error"]
    assert failed[0]["reply"] == ""
