"""Split eval: generate on a Kaggle GPU, judge locally (R10, R13, R27, D-078, GOTCHA G27, G28).

    uv run eval.py --model <hf-repo-id> --eval-set <path> --generate-only [--out DIR]
    uv run eval.py --judge-from <run_dir>/generations.jsonl [--judge J] [--out DIR]

``--generate-only`` runs every system the one-command eval would run (one ``hf:`` model brings
the base with the zero-shot and the best ablation prompt), makes no judge calls and needs no judge
key. It writes a new run dir under ``--out`` holding:

- ``generations.jsonl``: one row per (system, scenario) in system-then-scenario order, with
  ``model``, ``model_revision``, ``strategy``, ``scenario_id``, ``prompt_sha256``, ``reply``,
  ``generation_error`` and ``decoding``;
- ``generation_pins.json``: ``eval_set`` {path, sha256}, ``scenario_ids``, ``systems``
  [{model, strategy}], ``models`` [{model, model_revision, chat_template_sha256}], ``decoding``,
  ``git_sha``, ``git_dirty``, ``library_versions`` and ``prompt_sha256`` {strategy: sha}.

``--judge-from`` reads those two files (``generation_pins.json`` sits next to the JSONL), loads
the recorded eval set and refuses it (exit 2) if its SHA-256 changed, and judges the stored
replies with no generation and no Hub calls. It writes a new run dir with the same
``results.json``, ``results.md`` and ``transcripts.jsonl`` the one-command eval writes: for the
same replies and judge, ``results.json`` is byte-identical apart from ``run_id``. The two flags
are mutually exclusive; ``--model`` and ``--eval-set`` are required except with ``--judge-from``.

The R13 pins gain ``chat_template_sha256`` per model (``None`` for API and mock models), taken
from the backend that generated. HFBackend loads in fp16 on CUDA, since the T4 has no bf16.

Every test runs with sockets blocked and no keys; the Hub lookups and the ``hf:`` backend are
fakes, as in tests/test_eval_cli.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from groundcite import backends, hub
from groundcite.backends import MockCompliant, MockJudge, MockNaive
from groundcite.config import load_models_config
from groundcite.prompts import prompt_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
BASE_REPO = "Qwen/Qwen3-1.7B"
TUNED = "hf:org/groundcite-tuned@rev1"
TEMPLATE_SHA = "ab" * 32
GEN_FIELDS = {
    "model",
    "model_revision",
    "strategy",
    "scenario_id",
    "prompt_sha256",
    "reply",
    "generation_error",
    "decoding",
}


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


def run_cli(argv: list[str]) -> int:
    from groundcite.cli import main

    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def run_dir(out: Path) -> Path:
    dirs = [p for p in out.iterdir() if p.is_dir()] if out.exists() else []
    assert len(dirs) == 1, f"expected one run dir in {out}, found {dirs}"
    return dirs[0]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def fake_sha(repo: str, revision: str | None) -> str:
    return hashlib.sha1(f"{repo}@{revision}".encode()).hexdigest()


class FakeHF:
    """Stands in for HFBackend: the tuned repo answers on-spec, anything else off-spec. It
    reports a chat-template hash, as HFBackend does once loaded."""

    chat_template_sha256 = TEMPLATE_SHA

    def __init__(self, spec: str, transport: Any = None) -> None:
        self.name = spec
        self._inner = MockCompliant() if "tuned" in spec else MockNaive()

    def generate(self, messages: Any, system: str | None, params: Mapping[str, Any]) -> str:
        return self._inner.generate(messages, system, params)


class NoGeneration:
    def __init__(self, spec: str, transport: Any = None) -> None:
        self.name = spec

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("--judge-from must not generate")


def _patch_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hub,
        "fetch_adapter_config",
        lambda repo, revision=None: {"base_model_name_or_path": BASE_REPO, "peft_type": "LORA"},
    )
    monkeypatch.setattr(
        hub, "resolve_revision", lambda repo, revision=None: fake_sha(repo, revision)
    )
    monkeypatch.setitem(backends.BACKENDS, "hf:", FakeHF)


def _no_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--judge-from must not call the Hub")

    monkeypatch.setattr(hub, "fetch_adapter_config", refuse)
    monkeypatch.setattr(hub, "resolve_revision", refuse)
    monkeypatch.setitem(backends.BACKENDS, "hf:", NoGeneration)


SETUPS = {
    "hf-with-base-rows": ["--model", TUNED],
    "mock-models": ["--model", "mock:compliant", "--model", "mock:naive", "--no-base"],
}


def _subject(setup: str, eval_set: Path, out: Path, *extra: str) -> list[str]:
    return [
        *SETUPS[setup],
        "--eval-set",
        str(eval_set),
        "--limit",
        "3",
        "--allow-dirty",
        "--out",
        str(out),
        *extra,
    ]


def _generate(out: Path, setup: str = "hf-with-base-rows", eval_set: Path = ABLATION) -> Path:
    assert run_cli(_subject(setup, eval_set, out, "--generate-only")) == 0
    path = run_dir(out) / "generations.jsonl"
    assert path.is_file(), "--generate-only writes generations.jsonl"
    return path


def _judge_from(generations: Path, out: Path, *extra: str) -> int:
    argv = ["--judge-from", str(generations), "--judge", "mock", "--allow-dirty"]
    return run_cli([*argv, "--out", str(out), *extra])


# ---------------------------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------------------------


def test_help_lists_the_split_flags() -> None:
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert "--generate-only" in done.stdout and "--judge-from" in done.stdout


def test_generate_only_and_judge_from_are_mutually_exclusive(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hub(monkeypatch)
    generations = _generate(tmp_path / "gen")
    out = tmp_path / "both"
    argv = _subject("mock-models", ABLATION, out, "--generate-only")
    assert run_cli([*argv, "--judge-from", str(generations)]) == 2
    assert not out.exists() or not any(out.iterdir())


def test_judge_from_a_missing_file_exits_2(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _judge_from(tmp_path / "nope" / "generations.jsonl", tmp_path / "out") == 2
    assert "generations.jsonl" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# --generate-only (the Kaggle half)
# ---------------------------------------------------------------------------------------------


def test_generate_only_needs_no_judge_and_makes_no_judge_calls(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --judge: the pinned API judge's key is missing, and generation still runs."""
    _patch_hub(monkeypatch)

    def no_judging(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("--generate-only must not judge")

    monkeypatch.setattr(MockJudge, "generate", no_judging)
    out = tmp_path / "gen"
    assert run_cli(_subject("hf-with-base-rows", ABLATION, out, "--generate-only")) == 0
    run = run_dir(out)
    assert (run / "generations.jsonl").is_file()
    assert (run / "generation_pins.json").is_file()
    assert not (run / "results.json").exists()
    assert not (run / "transcripts.jsonl").exists()


def test_generations_cover_every_system_and_scenario(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hub(monkeypatch)
    rows = _jsonl(_generate(tmp_path / "gen"))
    best = load_models_config()["eval"]["best_strategy"]
    base = f"hf:{BASE_REPO}"
    systems = list(dict.fromkeys((r["model"], r["strategy"]) for r in rows))
    assert systems == [(base, "zero_shot"), (base, best), (TUNED, "zero_shot")]
    assert len(rows) == 3 * 3
    for row in rows:
        assert GEN_FIELDS <= set(row)
        assert row["prompt_sha256"] == prompt_sha256(row["strategy"])
        assert isinstance(row["reply"], str) and row["reply"]
        assert row["generation_error"] is None
        repo = BASE_REPO if row["model"] == base else "org/groundcite-tuned"
        rev = None if row["model"] == base else "rev1"
        assert row["model_revision"] == fake_sha(repo, rev)


def test_generation_pins_record_what_generated(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hub(monkeypatch)
    generations = _generate(tmp_path / "gen")
    pins = json.loads((generations.parent / "generation_pins.json").read_text(encoding="utf-8"))
    assert pins["eval_set"]["sha256"] == hashlib.sha256(ABLATION.read_bytes()).hexdigest()
    assert len(pins["scenario_ids"]) == 3
    assert [s["strategy"] for s in pins["systems"]][-1] == "zero_shot"
    models = {m["model"]: m for m in pins["models"]}
    assert models[TUNED]["model_revision"] == fake_sha("org/groundcite-tuned", "rev1")
    assert models[TUNED]["chat_template_sha256"] == TEMPLATE_SHA
    assert TUNED in pins["decoding"]
    assert len(pins["git_sha"]) == 40 and isinstance(pins["git_dirty"], bool)
    assert {"torch", "transformers", "peft"} <= set(pins["library_versions"])
    assert set(pins["prompt_sha256"]) >= {s["strategy"] for s in pins["systems"]}


# ---------------------------------------------------------------------------------------------
# --judge-from (the local half) equals the one-command eval
# ---------------------------------------------------------------------------------------------


def _normalised(run: Path, name: str) -> str:
    run_id = json.loads((run / "results.json").read_text(encoding="utf-8"))["run_id"]
    return (run / name).read_text(encoding="utf-8").replace(run_id, "<run_id>")


def _transcripts(run: Path) -> list[dict[str, Any]]:
    rows = [
        {k: v for k, v in r.items() if k != "run_id"} for r in _jsonl(run / "transcripts.jsonl")
    ]
    return sorted(rows, key=lambda r: (r["model"], r["strategy"], r["scenario_id"]))


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_split_mode_gives_byte_identical_results_to_one_shot(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: str
) -> None:
    _patch_hub(monkeypatch)
    one_shot = tmp_path / "one-shot"
    assert run_cli(_subject(setup, ABLATION, one_shot, "--judge", "mock")) == 0
    generations = _generate(tmp_path / "gen", setup)

    _no_hub(monkeypatch)
    judged = tmp_path / "judged"
    assert _judge_from(generations, judged) == 0

    a, b = run_dir(one_shot), run_dir(judged)
    assert _normalised(b, "results.json") == _normalised(a, "results.json")
    assert _normalised(b, "results.md") == _normalised(a, "results.md")
    assert _transcripts(b) == _transcripts(a)


def test_judge_from_refuses_a_changed_eval_set(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_hub(monkeypatch)
    eval_set = tmp_path / "ablation.jsonl"
    shutil.copy(ABLATION, eval_set)
    generations = _generate(tmp_path / "gen", eval_set=eval_set)
    eval_set.write_text(eval_set.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    _no_hub(monkeypatch)
    out = tmp_path / "judged"
    assert _judge_from(generations, out) == 2
    err = capsys.readouterr().err
    assert "sha256" in err.lower() or "eval set" in err.lower()
    assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------------------------------------
# The one-command R10 path is unchanged
# ---------------------------------------------------------------------------------------------


def test_one_shot_still_writes_results_and_no_generations(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hub(monkeypatch)
    out = tmp_path / "results"
    assert run_cli(_subject("hf-with-base-rows", ABLATION, out, "--judge", "mock")) == 0
    run = run_dir(out)
    for name in ("results.json", "results.md", "transcripts.jsonl"):
        assert (run / name).is_file()
    assert not (run / "generations.jsonl").exists()
    models = {
        m["model"]: m for m in json.loads((run / "results.json").read_text())["pins"]["models"]
    }
    assert models[TUNED]["chat_template_sha256"] == TEMPLATE_SHA


def test_mock_models_pin_no_chat_template(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "results"
    assert run_cli(_subject("mock-models", ABLATION, out, "--judge", "mock")) == 0
    pins = json.loads((run_dir(out) / "results.json").read_text(encoding="utf-8"))["pins"]
    assert all(m["chat_template_sha256"] is None for m in pins["models"])


# ---------------------------------------------------------------------------------------------
# HFBackend on the Kaggle T4
# ---------------------------------------------------------------------------------------------


class _Model:
    device = "cpu"

    def to(self, *args: Any, **kwargs: Any) -> _Model:
        return self

    def eval(self) -> _Model:
        return self

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        return input_ids


def test_hf_backend_loads_fp16_on_cuda(
    isolated_env: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The T4 (sm_75) has no bf16 (GOTCHA G28)."""
    import torch
    import transformers

    class _Tok:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, chat: Any, **kwargs: Any) -> str:
            return "prompt"

        def __call__(self, text: str, **kwargs: Any) -> Any:
            class _Enc(dict):  # type: ignore[type-arg]
                def to(self, device: Any) -> Any:
                    return self

            return _Enc(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens: Any, **kwargs: Any) -> str:
            return "Check the hose [A-1]."

    loads: list[dict[str, Any]] = []

    def load_model(*args: Any, **kwargs: Any) -> _Model:
        loads.append(kwargs)
        return _Model()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(hub, "fetch_adapter_config", lambda repo, revision=None: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)

    backend = backends.HFBackend("hf:org/merged-model")
    backend.generate([{"role": "user", "content": "It stalls."}], None, {})
    assert [kw.get("dtype") for kw in loads] == [torch.float16]
    assert backend.chat_template_sha256 == hashlib.sha256(b"{{ messages }}").hexdigest()
