"""Tests for the inference demo CLI, base vs tuned (R16).

    uv run scripts/demo.py --complaint TEXT [--snippets FILE | --snippet "ID: text" ...]
                           [--model SPEC] [--base SPEC] [--json]
    uv run scripts/demo.py --scenario EVAL_SET --id SCENARIO_ID [--model SPEC] [--base SPEC]

The contract these tests pin:

- ``scripts/demo.py`` exposes ``main(argv) -> int``. It builds one scenario, renders it with the
  project's zero-shot prompt (``render_messages(scenario, "zero_shot")``), sends the same prompt
  to the base and the tuned model, and runs ``run_checks`` on both replies.
- Input: ``--complaint`` is one user turn. Snippets come from ``--snippets FILE`` (a JSON array of
  ``{"id", "text", "title"?}``) or from repeated ``--snippet "ID: text"`` (the ID is the text
  before the first ``": "``); the two are mutually exclusive. ``--scenario FILE --id ID`` takes a
  row (messages, snippets and gold) from any eval set the loader reads; it excludes
  ``--complaint`` and the snippet flags.
- Gold: a scenario row keeps its own gold. A free-text complaint with snippets has no gold (the
  checks that need it are n/a). A free-text complaint with no snippets is scored against
  ``{"mode": "abstain", "supporting_ids": []}``: no snippet can apply, so the on-spec reply
  abstains.
- Models: ``--model`` is the tuned system and ``--base`` the base, in eval.py's spec syntax (a
  bare ``org/repo`` means ``hf:``). Backends come from ``groundcite.backends.get_backend``, so
  ``hf:`` reuses ``HFBackend`` and loads nothing until the first generate call. The defaults are
  the public adapter at the commit the demo notebook pins, and the base at train/config.toml's
  pin.
- Output: base and tuned replies side by side, then one row per check with its severity (HARD or
  SOFT) and a PASS / FAIL / n/a verdict for each model. ``--json`` prints one JSON object instead:
  ``{"scenario_id", "snippet_ids", "gold", "replies": {"base": R, "tuned": R}}``, where
  ``snippet_ids`` is the scenario's provided IDs (inferred from brackets on prompt-only rows)
  and R is ``{"model", "reply", "checks": [{"name", "severity", "passed", "detail"}]}`` in
  ``run_checks`` order.
- Exit 0 after printing (failed checks are output, not errors); exit 2 on bad usage or input.

No test loads a real model or makes a network call: the models are mocks, or ``HFBackend`` with
``generate`` replaced.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from groundcite import backends
from groundcite.checks import CHECKS, run_checks
from groundcite.evalset import load_eval_set, parse_row
from groundcite.prompts import render_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "scripts" / "demo.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
NOTEBOOK = REPO_ROOT / "train" / "demo_inference.ipynb"

COMPLAINT = (
    "My 2016 hatchback takes three or four tries to restart after a short stop on a hot day, "
    "but it starts right up every morning."
)
SNIPPETS = [
    {
        "id": "TSB-4417",
        "title": "Hot restart: fuel pressure leak-down test",
        "text": "Connect a fuel pressure gauge at the rail test port and watch the leak-down.",
    },
    {"id": "SM 4.2.6", "text": "Wiper motor park switch check with the key on."},
]
MOCKS = ["--model", "mock:compliant", "--base", "mock:naive"]
ABSTAIN_GOLD = {"mode": "abstain", "supporting_ids": []}
HEAVY_MODULES = ("torch", "transformers", "peft")


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


def _load() -> ModuleType:
    assert DEMO.is_file(), f"{DEMO.relative_to(REPO_ROOT)} does not exist yet"
    spec = importlib.util.spec_from_file_location("groundcite_demo", DEMO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main(argv: list[str]) -> int:
    try:
        return int(_load().main(argv))
    except SystemExit as exc:  # argparse errors
        return int(exc.code or 0)


def _run_json(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    code = _main([*argv, "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    return data


def _checks(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in result["checks"]}


def _expected_checks(reply: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    scenario = parse_row(row)
    return [
        {"name": r.name, "severity": r.severity, "passed": r.passed, "detail": r.detail}
        for r in run_checks(reply, scenario)
    ]


def _free_text_row(snippets: list[dict[str, Any]] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"id": "demo", "messages": [{"role": "user", "content": COMPLAINT}]}
    if snippets:
        row["snippets"] = snippets
    else:
        row["gold"] = ABSTAIN_GOLD
    return row


def _snippet_file(tmp_path: Path, snippets: list[dict[str, Any]]) -> Path:
    path = tmp_path / "snippets.json"
    path.write_text(json.dumps(snippets), encoding="utf-8")
    return path


def _notebook_param(name: str) -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        match = re.search(rf'(?m)^{name} = "([^"]+)"', "".join(cell["source"]))
        if match:
            return match.group(1)
    raise AssertionError(f"{name} is not set in {NOTEBOOK.name}")


def _pinned_specs() -> tuple[str, str]:
    """(tuned, base) defaults: the notebook's adapter pin and train/config.toml's base pin."""
    with (REPO_ROOT / "train" / "config.toml").open("rb") as fh:
        model = tomllib.load(fh)["model"]
    tuned = f"hf:{_notebook_param('ADAPTER_REPO')}@{_notebook_param('ADAPTER_REVISION')}"
    base = f"hf:{model['base']}@{model['revision']}"
    return tuned, base


class _RecordingHF:
    """Replaces ``HFBackend.generate`` and ``_load``: records each call, never loads weights."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, str | None, list[dict[str, str]]]] = []
        recorder = self

        def generate(
            self: backends.HFBackend,
            messages: list[dict[str, str]],
            system: str | None,
            params: Mapping[str, Any],
        ) -> str:
            recorder.calls.append((self.name, system, [dict(m) for m in messages]))
            return f"Reply from {self.name}: check the rail pressure [TSB-4417]."

        def load(self: backends.HFBackend) -> None:
            raise AssertionError("the demo loaded real weights")

        monkeypatch.setattr(backends.HFBackend, "generate", generate)
        monkeypatch.setattr(backends.HFBackend, "_load", load)

    @property
    def names(self) -> set[str]:
        return {name for name, _, _ in self.calls}


# ---------------------------------------------------------------------------------------------
# Free-text complaint with snippets
# ---------------------------------------------------------------------------------------------


def test_snippet_flags_give_both_replies_with_their_checks(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--complaint", COMPLAINT, *MOCKS]
    for s in SNIPPETS:
        argv += ["--snippet", f"{s['id']}: {s['text']}"]
    data = _run_json(argv, capsys)
    assert data["snippet_ids"] == ["TSB-4417", "SM 4.2.6"]
    assert data["gold"] is None
    base, tuned = data["replies"]["base"], data["replies"]["tuned"]
    assert base["model"] == "mock:naive"
    assert tuned["model"] == "mock:compliant"
    assert "[TSB-4417]" in tuned["reply"]
    row = _free_text_row([{"id": s["id"], "text": s["text"]} for s in SNIPPETS])
    assert tuned["checks"] == _expected_checks(tuned["reply"], row)
    assert base["checks"] == _expected_checks(base["reply"], row)


def test_check_rows_carry_every_check_with_a_hard_or_soft_severity(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--complaint", COMPLAINT, "--snippets", str(_snippet_file(tmp_path, SNIPPETS))]
    data = _run_json([*argv, *MOCKS], capsys)
    for name in ("base", "tuned"):
        checks = data["replies"][name]["checks"]
        assert [c["name"] for c in checks] == list(CHECKS)
        assert {c["severity"] for c in checks} == {"HARD", "SOFT"}
        assert all(c["passed"] in (True, False, None) for c in checks)


def test_compliant_tuned_passes_and_naive_base_fails_hard_checks(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--complaint", COMPLAINT, "--snippets", str(_snippet_file(tmp_path, SNIPPETS))]
    data = _run_json([*argv, *MOCKS], capsys)
    tuned = _checks(data["replies"]["tuned"])
    base = _checks(data["replies"]["base"])
    assert not [n for n, c in tuned.items() if c["passed"] is False], tuned
    # Free text with snippets has no gold: the gold checks are n/a.
    assert tuned["mode_matches_gold"]["passed"] is None
    assert tuned["cited_id_supports"]["passed"] is None
    assert base["citation_count"]["passed"] is False
    assert base["citation_count"]["severity"] == "HARD"
    assert base["no_fix"]["passed"] is False
    assert base["no_fix"]["severity"] == "SOFT"


def test_snippets_file_matches_the_snippet_flags(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from_file = _run_json(
        ["--complaint", COMPLAINT, "--snippets", str(_snippet_file(tmp_path, SNIPPETS)), *MOCKS],
        capsys,
    )
    flags = ["--complaint", COMPLAINT, *MOCKS]
    for s in SNIPPETS:
        flags += ["--snippet", f"{s['id']}: {s['text']}"]
    from_flags = _run_json(flags, capsys)
    assert from_file["snippet_ids"] == from_flags["snippet_ids"]
    assert from_file["replies"] == from_flags["replies"]


def test_both_models_get_the_same_zero_shot_prompt(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hf = _RecordingHF(monkeypatch)
    argv = ["--complaint", COMPLAINT, "--snippets", str(_snippet_file(tmp_path, SNIPPETS))]
    _run_json([*argv, "--model", "hf:org/tuned@abc", "--base", "hf:org/base@def"], capsys)
    assert len(hf.calls) == 2
    (_, system_a, messages_a), (_, system_b, messages_b) = hf.calls
    expected = render_messages(parse_row(_free_text_row(SNIPPETS)), "zero_shot")
    assert (system_a, messages_a) == expected
    assert (system_b, messages_b) == expected


# ---------------------------------------------------------------------------------------------
# No snippets: the on-spec reply abstains (R16)
# ---------------------------------------------------------------------------------------------


def test_no_snippet_path_is_scored_as_an_abstain_case(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _run_json(["--complaint", COMPLAINT, *MOCKS], capsys)
    assert data["snippet_ids"] == []
    assert data["gold"] == ABSTAIN_GOLD


def test_no_snippet_path_abstains_on_spec_for_the_compliant_mock(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _run_json(["--complaint", COMPLAINT, *MOCKS], capsys)
    tuned = data["replies"]["tuned"]
    checks = _checks(tuned)
    assert not re.search(r"\[[^\]]+\]", tuned["reply"]), "an abstain cites nothing"
    assert tuned["reply"].count("?") == 1, "an abstain asks one question"
    assert not [n for n, c in checks.items() if c["passed"] is False], checks
    assert checks["mode_matches_gold"]["passed"] is True
    assert checks["abstain_form"]["passed"] is True
    assert tuned["checks"] == _expected_checks(tuned["reply"], _free_text_row(None))


def test_no_snippet_path_flags_the_naive_base(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _run_json(["--complaint", COMPLAINT, *MOCKS], capsys)
    base = _checks(data["replies"]["base"])
    assert base["mode_matches_gold"]["passed"] is False
    assert base["cited_id_provided"]["passed"] is False
    assert base["abstain_form"]["passed"] is False


# ---------------------------------------------------------------------------------------------
# --scenario FILE --id ID
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "scenario_id"),
    [
        ("our_format.jsonl", "fx-a-002"),
        ("prompt_only.csv", "po-csv-001"),
        ("prompt_only.json", "po-json-001"),
        ("third_party.jsonl", "tp-7"),
    ],
)
def test_scenario_row_is_scored_with_its_own_snippets_and_gold(
    env: Path, fixture: str, scenario_id: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = FIXTURES / fixture
    scenario = next(s for s in load_eval_set(path) if s.id == scenario_id)
    data = _run_json(["--scenario", str(path), "--id", scenario_id, *MOCKS], capsys)
    assert data["scenario_id"] == scenario_id
    assert data["snippet_ids"] == scenario.provided_ids
    assert data["gold"] == (dict(scenario.gold) if scenario.gold is not None else None)
    for name in ("base", "tuned"):
        reply = data["replies"][name]["reply"]
        expected = [
            {"name": r.name, "severity": r.severity, "passed": r.passed, "detail": r.detail}
            for r in run_checks(reply, scenario)
        ]
        assert data["replies"][name]["checks"] == expected


def test_scenario_with_an_unfamiliar_id_scheme_cites_a_provided_id(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = FIXTURES / "third_party.jsonl"
    data = _run_json(["--scenario", str(path), "--id", "tp-7", *MOCKS], capsys)
    tuned = _checks(data["replies"]["tuned"])
    assert "[ref_q9.04]" in data["replies"]["tuned"]["reply"]
    assert tuned["cited_id_provided"]["passed"] is True
    assert tuned["cited_id_supports"]["passed"] is True


def test_unknown_scenario_id_exits_2_naming_it(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--scenario", str(FIXTURES / "our_format.jsonl"), "--id", "no-such-id", *MOCKS]
    assert _main(argv) == 2
    assert "no-such-id" in capsys.readouterr().err


def test_malformed_scenario_file_exits_2_with_the_line_and_field(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "ok-1", "prompt": "Rough idle. [A-1] Check the hose."})
        + "\n"
        + json.dumps({"id": "bad-2"})
        + "\n",
        encoding="utf-8",
    )
    assert _main(["--scenario", str(bad), "--id", "ok-1", *MOCKS]) == 2
    err = capsys.readouterr().err
    assert "line 2" in err
    assert "messages" in err or "prompt" in err


# ---------------------------------------------------------------------------------------------
# Usage errors
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([*MOCKS], id="no-input"),
        pytest.param(
            ["--complaint", COMPLAINT, "--scenario", "x.jsonl", "--id", "a", *MOCKS],
            id="complaint-and-scenario",
        ),
        pytest.param(
            ["--scenario", str(FIXTURES / "our_format.jsonl"), *MOCKS], id="scenario-without-id"
        ),
        pytest.param(
            ["--complaint", COMPLAINT, "--snippet", "no separator here", *MOCKS],
            id="snippet-without-id",
        ),
        pytest.param(
            ["--complaint", COMPLAINT, "--snippet", "!!: text", *MOCKS],
            id="snippet-id-outside-grammar",
        ),
        pytest.param(
            ["--complaint", COMPLAINT, "--model", "mock:nope", "--base", "mock:naive"],
            id="unknown-mock",
        ),
    ],
)
def test_bad_usage_exits_2(env: Path, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(argv) == 2
    assert capsys.readouterr().err.strip(), "a usage error says what is wrong"


def test_snippets_file_and_snippet_flags_are_exclusive(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--complaint",
        COMPLAINT,
        "--snippets",
        str(_snippet_file(tmp_path, SNIPPETS)),
        "--snippet",
        "TSB-1: text",
        *MOCKS,
    ]
    assert _main(argv) == 2


def test_snippets_file_with_a_bad_row_exits_2_naming_the_field(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _snippet_file(tmp_path, [{"id": "TSB-1", "text": "ok"}, {"id": "TSB-2"}])
    assert _main(["--complaint", COMPLAINT, "--snippets", str(path), *MOCKS]) == 2
    assert "text" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# Text output: side by side
# ---------------------------------------------------------------------------------------------


def _squash(text: str) -> str:
    return " ".join(text.split())


def test_text_output_shows_both_replies_and_a_check_table(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--complaint", COMPLAINT, "--snippets", str(_snippet_file(tmp_path, SNIPPETS))]
    data = _run_json([*argv, *MOCKS], capsys)
    assert _main([*argv, *MOCKS]) == 0
    out = capsys.readouterr().out
    assert re.search(r"(?m)^\s*BASE\b.*mock:naive", out), out
    assert re.search(r"(?m)^\s*TUNED\b.*mock:compliant", out), out
    for name in ("base", "tuned"):
        assert _squash(data["replies"][name]["reply"]) in _squash(out)
    verdicts = {True: "PASS", False: "FAIL", None: "n/a"}
    base, tuned = _checks(data["replies"]["base"]), _checks(data["replies"]["tuned"])
    for check in CHECKS:
        line = next((ln for ln in out.splitlines() if re.match(rf"\s*{check}\b", ln)), None)
        assert line is not None, f"no table row for {check}"
        assert base[check]["severity"] in line
        cells = re.findall(r"PASS|FAIL|n/a", line)
        assert cells == [verdicts[base[check]["passed"]], verdicts[tuned[check]["passed"]]], line


def test_text_output_says_when_no_snippets_were_given(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(["--complaint", COMPLAINT, *MOCKS]) == 0
    out = capsys.readouterr().out
    assert "(none)" in out or "no snippets" in out.lower()


# ---------------------------------------------------------------------------------------------
# hf: models reuse HFBackend, lazily
# ---------------------------------------------------------------------------------------------


def test_hf_specs_go_through_hf_backend(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hf = _RecordingHF(monkeypatch)
    data = _run_json(
        ["--complaint", COMPLAINT, "--model", "org/tuned@abc", "--base", "hf:org/base@def"], capsys
    )
    assert hf.names == {"hf:org/tuned@abc", "hf:org/base@def"}
    assert data["replies"]["tuned"]["model"] == "hf:org/tuned@abc"
    assert data["replies"]["base"]["model"] == "hf:org/base@def"
    assert data["replies"]["tuned"]["reply"].startswith("Reply from hf:org/tuned@abc")
    assert data["replies"]["base"]["reply"].startswith("Reply from hf:org/base@def")


def test_defaults_are_the_pinned_adapter_and_base(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hf = _RecordingHF(monkeypatch)
    data = _run_json(["--complaint", COMPLAINT], capsys)
    tuned, base = _pinned_specs()
    assert hf.names == {tuned, base}
    assert data["replies"]["tuned"]["model"] == tuned
    assert data["replies"]["base"]["model"] == base


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))}
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    env["GROUNDCITE_ENV_FILE"] = str(empty)
    return env


def test_demo_with_hf_specs_imports_no_model_library_until_generate(tmp_path: Path) -> None:
    """In a fresh interpreter, with HFBackend.generate replaced, an hf: run never imports torch,
    transformers or peft: the backend is built lazily and the demo adds no eager import."""
    assert DEMO.is_file(), f"{DEMO.relative_to(REPO_ROOT)} does not exist yet"
    code = f"""
import importlib.util, json, sys
from groundcite import backends

def generate(self, messages, system, params):
    return "Reply from " + self.name + ". Which trouble code is stored?"

def load(self):
    raise AssertionError("loaded real weights")

backends.HFBackend.generate = generate
backends.HFBackend._load = load
spec = importlib.util.spec_from_file_location("groundcite_demo", {str(DEMO)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rc = module.main(["--complaint", "Engine cranks but will not start.",
                  "--model", "hf:org/tuned@abc", "--base", "hf:org/base@def", "--json"])
heavy = sorted(n for n in {HEAVY_MODULES!r} if n in sys.modules)
print("RESULT " + json.dumps({{"rc": rc, "heavy": heavy}}), file=sys.stderr)
"""
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(tmp_path),
    )
    match = re.search(r"^RESULT (.*)$", done.stderr, re.MULTILINE)
    assert match, done.stderr
    result = json.loads(match.group(1))
    assert result["rc"] == 0, done.stderr
    assert result["heavy"] == [], f"imported {result['heavy']} without a real model"
    data = json.loads(done.stdout)
    assert data["replies"]["tuned"]["reply"].startswith("Reply from hf:org/tuned@abc")


def test_demo_help_lists_every_flag(tmp_path: Path) -> None:
    assert DEMO.is_file(), f"{DEMO.relative_to(REPO_ROOT)} does not exist yet"
    done = subprocess.run(
        [sys.executable, str(DEMO), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=_subprocess_env(tmp_path),
    )
    assert done.returncode == 0, done.stderr
    for flag in (
        "--complaint",
        "--snippets",
        "--snippet",
        "--scenario",
        "--id",
        "--model",
        "--base",
        "--json",
    ):
        assert flag in done.stdout, flag
