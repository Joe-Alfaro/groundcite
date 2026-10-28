"""The R5 smoke loop, run with every tool stubbed (R5, D-078).

    scripts/smoke_loop.sh [OUT_DIR]            # local, CPU only, capped
    scripts/smoke_loop.sh --kaggle [OUT_DIR]   # the real-model smoke loop on a Kaggle GPU

Local mode (the default) is the only model run allowed on the dev Mac: a CPU-only check under
``timeout 120``. It builds a tiny small-vocab base with ``scripts/make_tiny_base.py --out DIR``
(the same shape as the test fixture in ``tests/train_data.py``: a random two-layer Qwen3 and a
byte-level BPE tokenizer of a few hundred tokens with the Qwen3-shaped chat template) and trains
on it with ``--device cpu --batch-size 1``, at most 20 steps, on at most 32 examples (one per
distilled scenario). Every stage runs under ``timeout 120`` (or ``gtimeout``) with
``CUDA_VISIBLE_DEVICES=''`` and ``PYTORCH_ENABLE_MPS_FALLBACK=0``. Without either timeout
command the script fails before running any stage. It never names a Qwen3 checkpoint.

``--kaggle`` trains the real ``Qwen/Qwen3-0.6B`` with ``--device cuda``; a notebook under
``train/`` runs it.

The script is executed for real, but ``uv``, ``python``, ``python3``, ``timeout`` and
``gtimeout`` are stubs that record their arguments and environment, so nothing is trained.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from train_data import REPO_ROOT, SMOKE_LOOP, load_train_module, scenario_rows, train_record

MAKE_TINY_BASE = REPO_ROOT / "scripts" / "make_tiny_base.py"
SEP = "\x1f"

# Records "<tool> SEP <timeout secs or -> SEP <CUDA_VISIBLE_DEVICES or <unset>> SEP
# <PYTORCH_ENABLE_MPS_FALLBACK or <unset>> SEP args..." and creates any --out directory.
TOOL_STUB = r"""#!/bin/bash
{
  printf '%s' "$(basename "$0")"
  printf '\x1f%s' "${SMOKE_STUB_TIMEOUT:--}"
  printf '\x1f%s' "${CUDA_VISIBLE_DEVICES-<unset>}"
  printf '\x1f%s' "${PYTORCH_ENABLE_MPS_FALLBACK-<unset>}"
  for a in "$@"; do printf '\x1f%s' "$a"; done
  printf '\n'
} >> "$SMOKE_STUB_LOG"
prev=""
for a in "$@"; do
  if [ "$prev" = "--out" ]; then mkdir -p "$a"; fi
  prev="$a"
done
exit 0
"""

# A timeout stand-in: skips options, exports the duration, then runs the command.
TIMEOUT_STUB = r"""#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in
    -k|-s|--kill-after|--signal) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
export SMOKE_STUB_TIMEOUT="$1"
shift
exec "$@"
"""


@dataclass
class Call:
    tool: str
    timeout: str
    cuda_visible_devices: str
    mps_fallback: str
    args: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.args)

    def flag(self, name: str) -> str | None:
        for i, arg in enumerate(self.args):
            if arg == name and i + 1 < len(self.args):
                return self.args[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
        return None


@dataclass
class Run:
    code: int
    stderr: str
    calls: list[Call]

    def stage(self, script: str) -> Call:
        matches = [c for c in self.calls if any(a.endswith(script) for a in c.args)]
        assert len(matches) == 1, f"expected one {script} call, got {[c.text for c in matches]}"
        return matches[0]


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stubs(directory: Path, *, timeout: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for tool in ("uv", "python", "python3"):
        _write(directory / tool, TOOL_STUB)
    if timeout:
        for tool in ("timeout", "gtimeout"):
            _write(directory / tool, TIMEOUT_STUB)
    return directory


def _run(tmp_path: Path, *args: str, timeout: bool = True, system_path: str | None = None) -> Run:
    assert SMOKE_LOOP.is_file(), f"{SMOKE_LOOP} is missing"
    stubs = _stubs(tmp_path / "bin", timeout=timeout)
    log = tmp_path / "calls.log"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"SMOKE_BASE", "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK"}
    }
    env["PATH"] = f"{stubs}{os.pathsep}{system_path or env.get('PATH', '')}"
    env["SMOKE_STUB_LOG"] = str(log)
    bash = shutil.which("bash") or "/bin/bash"
    done = subprocess.run(
        [bash, str(SMOKE_LOOP), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    calls = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            tool, secs, cuda, mps, *rest = line.split(SEP)
            calls.append(Call(tool, secs, cuda, mps, rest))
    return Run(done.returncode, done.stderr, calls)


def _seconds(text: str) -> float | None:
    """A timeout duration (``120``, ``120s``, ``2m``) in seconds, or None."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", text)
    if match is None:
        return None
    return float(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


@pytest.fixture
def local(tmp_path: Path) -> Run:
    run = _run(tmp_path, str(tmp_path / "out"))
    assert run.code == 0, run.stderr
    return run


@pytest.fixture
def kaggle(tmp_path: Path) -> Run:
    run = _run(tmp_path, "--kaggle", str(tmp_path / "out"))
    assert run.code == 0, run.stderr
    return run


# ---------------------------------------------------------------------------------------------
# Local mode: CPU only, tiny base, capped
# ---------------------------------------------------------------------------------------------


def test_local_mode_runs_every_stage_in_order(local: Run) -> None:
    order = [
        "scripts/make_tiny_base.py",
        "scripts/distill.py",
        "scripts/build_dataset.py",
        "train/train_qlora.py",
        "eval.py",
    ]
    positions = [local.calls.index(local.stage(s)) for s in order]
    assert positions == sorted(positions)


def test_local_mode_trains_the_tiny_base_it_just_built(local: Run) -> None:
    tiny = local.stage("scripts/make_tiny_base.py").flag("--out")
    assert tiny, "make_tiny_base.py takes --out DIR"
    assert local.stage("train/train_qlora.py").flag("--base") == tiny


def test_local_mode_never_touches_a_qwen3_checkpoint(local: Run) -> None:
    for call in local.calls:
        assert "Qwen3-0.6B" not in call.text and "Qwen/" not in call.text, call.text


def test_local_mode_trains_on_cpu_with_batch_1_and_at_most_20_steps(local: Run) -> None:
    train = local.stage("train/train_qlora.py")
    assert train.flag("--device") == "cpu"
    assert train.flag("--backend") == "peft"
    assert train.flag("--batch-size") == "1"
    steps = train.flag("--max-steps")
    assert steps is not None and 1 <= int(steps) <= 20


def test_local_mode_uses_at_most_32_examples(local: Run) -> None:
    """build_dataset keeps one example per scenario, so the distill limit bounds the examples."""
    limit = local.stage("scripts/distill.py").flag("--limit")
    assert limit is not None and 1 <= int(limit) <= 32


def test_every_local_stage_runs_under_a_120_second_timeout(local: Run) -> None:
    assert local.calls
    for call in local.calls:
        seconds = _seconds(call.timeout)
        assert seconds is not None and 0 < seconds <= 120, f"not under timeout 120: {call.text}"


def test_every_local_stage_hides_cuda_and_disables_the_mps_fallback(local: Run) -> None:
    for call in local.calls:
        assert call.cuda_visible_devices == "", call.text
        assert call.mps_fallback == "0", call.text


def test_local_mode_keeps_the_mock_teacher_and_judges(local: Run) -> None:
    assert (local.stage("scripts/distill.py").flag("--teacher") or "").startswith("mock")
    assert (local.stage("scripts/build_dataset.py").flag("--judge") or "").startswith("mock")
    assert (local.stage("eval.py").flag("--judge") or "").startswith("mock")


def test_local_mode_without_a_timeout_command_fails_before_any_stage(tmp_path: Path) -> None:
    system_path = "/usr/bin:/bin"
    if any(shutil.which(t, path=system_path) for t in ("timeout", "gtimeout")):
        pytest.skip("this system has a timeout command on /usr/bin:/bin")
    run = _run(tmp_path, str(tmp_path / "out"), timeout=False, system_path=system_path)
    assert run.code != 0
    assert "timeout" in run.stderr
    assert run.calls == [], "no stage may run without the 120-second cap"


# ---------------------------------------------------------------------------------------------
# --kaggle: the real-model smoke loop on a GPU
# ---------------------------------------------------------------------------------------------


def test_kaggle_mode_trains_the_real_small_qwen3_on_cuda(kaggle: Run) -> None:
    train = kaggle.stage("train/train_qlora.py")
    assert train.flag("--base") == "Qwen/Qwen3-0.6B"
    assert train.flag("--device") == "cuda"
    steps = train.flag("--max-steps")
    assert steps is not None and 1 <= int(steps) <= 20
    assert not any(c.text.endswith("make_tiny_base.py") for c in kaggle.calls)
    assert all(c.cuda_visible_devices != "" for c in kaggle.calls), "Kaggle must see its GPU"


def test_a_kaggle_notebook_runs_the_kaggle_smoke_loop() -> None:
    sources = []
    for notebook in sorted((REPO_ROOT / "train").glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        for cell in cells:
            if cell["cell_type"] == "code":
                src = cell["source"]
                sources.append(src if isinstance(src, str) else "".join(src))
    assert any("smoke_loop.sh --kaggle" in s for s in sources), (
        "a notebook in train/ runs scripts/smoke_loop.sh --kaggle"
    )


def test_readme_documents_both_smoke_modes() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "smoke_loop.sh --kaggle" in readme
    assert "timeout 120" in readme


# ---------------------------------------------------------------------------------------------
# scripts/make_tiny_base.py: the local smoke base
# ---------------------------------------------------------------------------------------------


def _make_tiny_base(out: Path) -> int:
    import importlib.util

    assert MAKE_TINY_BASE.is_file(), f"{MAKE_TINY_BASE} is missing"
    spec = importlib.util.spec_from_file_location("make_tiny_base", MAKE_TINY_BASE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return int(module.main(["--out", str(out)]))
    except SystemExit as exc:
        return int(exc.code or 0)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    mp = pytest.MonkeyPatch()
    out = tmp_path_factory.mktemp("made") / "tiny"
    try:
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("TRANSFORMERS_OFFLINE", "1")
        assert _make_tiny_base(out) == 0
    finally:
        mp.undo()
    return out


def test_make_tiny_base_writes_a_tiny_small_vocab_qwen3(tiny_dir: Path) -> None:
    config: dict[str, Any] = json.loads((tiny_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen3"
    assert config["vocab_size"] <= 4096, "a small vocab keeps the logits tiny"
    assert config["num_hidden_layers"] <= 2
    assert config["hidden_size"] <= 64
    weights = sum(p.stat().st_size for p in tiny_dir.glob("*.safetensors"))
    assert 0 < weights < 5_000_000


def test_the_tiny_base_tokenizer_renders_training_examples(tiny_dir: Path) -> None:
    """Its chat template follows Qwen3's shape, so build_example's eval-prefix check holds."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_dir), local_files_only=True)
    assert tok.eos_token == "<|im_end|>"
    assert len(tok) <= 4096
    train = load_train_module()
    for row in scenario_rows(5):
        record = train_record(row, "Check the fuel pressure first [A-1].")
        example = train.build_example(tok, record)
        assert example["labels"][-1] == tok.eos_token_id
        assert len(example["input_ids"]) == len(example["labels"])
