"""Device control and base-model pinning for train/train_qlora.py (R5, R27, R31, GOTCHA G27, G28,
D-078).

    uv run train/train_qlora.py ... [--device auto|cpu|cuda] [--batch-size N]

- ``--device`` (default ``auto``) is the run's device; it is in the dry-run plan as ``device`` and
  is not part of the config hash. ``cpu`` never selects MPS, even on a Mac where MPS is
  available: ``TrainingArguments(use_cpu=True)`` and the model stays on the CPU. ``cuda`` without
  a CUDA GPU exits 2 before any model load. ``auto`` picks CUDA when it is available.
- The peft backend loads the base in fp32 only on the CPU, and in fp16 on CUDA (with fp16 mixed
  precision in the Trainer); bf16 is never used (T4).
- Gradient checkpointing follows ``[train] gradient_checkpointing`` on every device.
- ``--batch-size N`` overrides ``[train] per_device_train_batch_size`` (a smoke-run override; like
  ``--max-steps`` it is part of the resolved config and its hash).
- G27: ``[model] revision`` pins the configured base to a 40-hex Hub commit. Tokenizer and model
  loads of the configured base pass that revision, and ``adapter_config.json`` records it (so
  eval's HFBackend loads the same base commit). ``--base`` replaces the base and clears the
  revision (``null`` in the resolved config). ``train_log.json`` records
  ``chat_template_sha256 = groundcite.prompts.chat_template_sha256(tokenizer)`` (the SHA-256 of
  ``tokenizer.chat_template``, UTF-8) and the ``tokenizers`` version; HFBackend exposes the same
  value as ``chat_template_sha256`` once it has loaded a model.

No test here trains or runs a model: the transformers ``Trainer`` is a recording fake, and the
model loads are the tiny two-layer fixture (``tests/train_data.py``) loaded on the CPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from train_data import (
    CHAT_TEMPLATE,
    CONFIG_PATH,
    DEFAULT_BASE,
    load_train_module,
    save_tiny_base,
    tiny_tokenizer,
    train_records,
    write_dataset,
)

from groundcite import backends, hub

GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


# ---------------------------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------------------------


class FakeTrainer:
    """Records what the script hands the Trainer; ``train()`` only fakes a loss history."""

    instances: list[FakeTrainer] = []

    def __init__(self, model: Any = None, args: Any = None, **kwargs: Any) -> None:
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.state = SimpleNamespace(global_step=0, log_history=[])
        FakeTrainer.instances.append(self)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        steps = getattr(self.args, "max_steps", -1)
        steps = steps if isinstance(steps, int) and steps > 0 else 1
        self.state.global_step = steps
        self.state.log_history = [{"step": s, "loss": 1.0} for s in range(1, steps + 1)]
        return SimpleNamespace(training_loss=1.0)


class FakeTrainingArguments:
    """Stands in for TrainingArguments on a faked CUDA machine (the real one would refuse fp16
    without a GPU). Every keyword is an attribute."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


@pytest.fixture
def env(isolated_env: Path, no_network: None, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    return isolated_env


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_base(tmp_path_factory.mktemp("tiny-base-device") / "tiny-qwen3")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "data" / "subsets" / "n6.jsonl", train_records(6))


@pytest.fixture
def stack(
    env: Path, tiny_base: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Record every tokenizer and model load (any base name loads the tiny fixture, fp32, on the
    CPU) and replace the Trainer with FakeTrainer. MPS is reported available, CUDA not."""
    import torch
    import transformers

    calls: dict[str, list[tuple[str, dict[str, Any]]]] = {"tokenizer": [], "model": []}
    real_tokenizer = transformers.AutoTokenizer.from_pretrained
    real_model = transformers.AutoModelForCausalLM.from_pretrained

    def load_tokenizer(name: Any, *args: Any, **kwargs: Any) -> Any:
        calls["tokenizer"].append((str(name), dict(kwargs)))
        return real_tokenizer(str(tiny_base))

    def load_model(name: Any, *args: Any, **kwargs: Any) -> Any:
        calls["model"].append((str(name), dict(kwargs)))
        return real_model(str(tiny_base), dtype=torch.float32)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(transformers, "Trainer", FakeTrainer)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    FakeTrainer.instances = []
    return calls


def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import transformers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla T4")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(transformers, "TrainingArguments", FakeTrainingArguments)


def _main(module: ModuleType, argv: list[str]) -> int:
    try:
        return int(module.main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def _train(dataset: Path, out: Path, *extra: str, module: ModuleType | None = None) -> int:
    argv = ["--dataset", str(dataset), "--out", str(out), "--seed", "3", "--backend", "peft"]
    return _main(module or load_train_module(), [*argv, "--max-steps", "2", *extra])


def _dry_run(
    capsys: pytest.CaptureFixture[str], dataset: Path, out: Path, *extra: str
) -> dict[str, Any]:
    capsys.readouterr()
    argv = ["--dataset", str(dataset), "--out", str(out), "--seed", "3", "--dry-run", *extra]
    code = _main(load_train_module(), argv)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return dict(json.loads(captured.out))


def _trainer() -> FakeTrainer:
    assert len(FakeTrainer.instances) == 1, "the script builds exactly one Trainer"
    return FakeTrainer.instances[0]


def _log(out: Path) -> dict[str, Any]:
    return dict(json.loads((out / "train_log.json").read_text(encoding="utf-8")))


def _pinned_revision() -> str:
    return str(tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))["model"]["revision"])


# ---------------------------------------------------------------------------------------------
# (1) --device {auto,cpu,cuda}
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_dry_run_reports_the_requested_device(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    device: str,
) -> None:
    plan = _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base), "--device", device)
    assert plan["device"] == device


def test_default_device_is_auto(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base))
    assert plan["device"] == "auto"


def test_mps_is_not_a_device_choice(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--dataset", str(dataset), "--out", str(tmp_path / "o"), "--seed", "3", "--dry-run"]
    code = _main(load_train_module(), [*argv, "--device", "mps"])
    err = capsys.readouterr().err
    assert code == 2
    assert "--device" in err and "Traceback" not in err
    assert "unrecognized arguments" not in err, "--device exists; mps is not one of its choices"
    assert "cpu" in err and "cuda" in err, "the error lists the valid choices"


def test_the_device_is_not_part_of_the_config_hash(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hashes = {
        _dry_run(capsys, dataset, tmp_path / "o", "--base", str(tiny_base), "--device", d)[
            "config_sha256"
        ]
        for d in ("auto", "cpu", "cuda")
    }
    assert len(hashes) == 1


def test_cpu_device_never_selects_mps(
    stack: dict[str, Any], dataset: Path, tmp_path: Path, tiny_base: Path
) -> None:
    """On a Mac with MPS available, --device cpu keeps the Trainer and the model on the CPU."""
    import torch

    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", "cpu") == 0
    args = _trainer().args
    assert args.use_cpu is True
    assert args.device.type == "cpu", "TrainingArguments must not resolve to MPS"
    assert {p.device.type for p in _trainer().model.parameters()} == {"cpu"}
    assert args.fp16 is False and args.bf16 is False
    assert [kw.get("dtype") for _, kw in stack["model"]] == [torch.float32]
    runtime = _log(out)["runtime"]
    assert runtime["device"] == "cpu"
    assert runtime["fp16"] is False


def test_cpu_device_ignores_an_available_cuda_gpu(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", "cpu") == 0
    args = _trainer().args
    assert args.use_cpu is True and args.fp16 is False
    assert [kw.get("dtype") for _, kw in stack["model"]] == [torch.float32]
    assert "quantization_config" not in stack["model"][0][1], "no 4-bit base on the CPU"
    assert _log(out)["runtime"]["device"] == "cpu"


@pytest.mark.parametrize("device", ["cuda", "auto"])
def test_cuda_loads_the_peft_base_in_fp16_and_trains_with_fp16(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    import torch

    _fake_cuda(monkeypatch)
    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", device) == 0
    args = _trainer().args
    assert args.fp16 is True
    assert getattr(args, "bf16", False) is False
    assert not getattr(args, "use_cpu", False)
    assert [kw.get("dtype") for _, kw in stack["model"]] == [torch.float16]
    runtime = _log(out)["runtime"]
    assert runtime["device"] == "cuda"
    assert runtime["fp16"] is True


def test_cuda_device_without_a_gpu_exits_2_before_loading_anything(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    code = _train(dataset, out, "--base", str(tiny_base), "--device", "cuda")
    err = capsys.readouterr().err
    assert code == 2
    assert "cuda" in err.lower() and "Traceback" not in err
    assert "unrecognized arguments" not in err
    assert stack["model"] == []
    assert FakeTrainer.instances == []
    assert not (out / "adapter_model.safetensors").exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_checkpointing_is_on_for_every_device_when_configured(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    assert tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))["train"]["gradient_checkpointing"]
    if device == "cuda":
        _fake_cuda(monkeypatch)
    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", device) == 0
    assert _trainer().args.gradient_checkpointing is True
    assert _log(out)["runtime"]["gradient_checkpointing"] is True


def test_gradient_checkpointing_is_off_when_the_config_turns_it_off(
    stack: dict[str, Any], dataset: Path, tmp_path: Path, tiny_base: Path
) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    off = re.sub(r"(?m)^gradient_checkpointing\s*=\s*true", "gradient_checkpointing = false", text)
    assert off != text
    config = tmp_path / "config.toml"
    config.write_text(off, encoding="utf-8")
    module = load_train_module()
    module.CONFIG_PATH = config
    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", "cpu", module=module) == 0
    assert not _trainer().args.gradient_checkpointing
    assert _log(out)["runtime"]["gradient_checkpointing"] is False


# ---------------------------------------------------------------------------------------------
# (2) --batch-size: the smoke run's batch-1 override
# ---------------------------------------------------------------------------------------------


def test_batch_size_overrides_the_config_and_its_hash(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ("--base", str(tiny_base))
    default = _dry_run(capsys, dataset, tmp_path / "o", *base)
    one = _dry_run(capsys, dataset, tmp_path / "o", *base, "--batch-size", "1")
    assert one["config"]["train"]["per_device_train_batch_size"] == 1
    assert one["config_sha256"] != default["config_sha256"]


def test_batch_size_reaches_the_trainer(
    stack: dict[str, Any], dataset: Path, tmp_path: Path, tiny_base: Path
) -> None:
    out = tmp_path / "out"
    argv = ("--base", str(tiny_base), "--device", "cpu", "--batch-size", "1")
    assert _train(dataset, out, *argv) == 0
    assert _trainer().args.per_device_train_batch_size == 1


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_batch_size_must_be_a_positive_integer(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    argv = ["--dataset", str(dataset), "--out", str(tmp_path / "o"), "--seed", "3", "--dry-run"]
    assert _main(load_train_module(), [*argv, "--batch-size", value]) == 2
    err = capsys.readouterr().err
    assert "--batch-size" in err and "unrecognized arguments" not in err


# ---------------------------------------------------------------------------------------------
# (3) G27: the base revision is pinned and the chat-template hash is recorded
# ---------------------------------------------------------------------------------------------


def test_config_pins_the_base_model_revision() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["model"]["base"] == DEFAULT_BASE
    assert GIT_SHA_RE.fullmatch(str(config["model"].get("revision"))), (
        "[model] revision must be the full 40-hex Hub commit of the base"
    )


def test_dry_run_loads_the_tokenizer_at_the_pinned_revision(
    env: Path,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    seen: list[tuple[str, dict[str, Any]]] = []

    def not_cached(name: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append((str(name), dict(kwargs)))
        raise OSError("not in the local cache")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", not_cached)
    plan = _dry_run(capsys, dataset, tmp_path / "out")
    assert plan["config"]["model"]["revision"] == _pinned_revision()
    assert seen and all(name == DEFAULT_BASE for name, _ in seen)
    assert all(kwargs.get("revision") == _pinned_revision() for _, kwargs in seen)


def test_a_base_override_clears_the_pinned_revision(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base))
    assert plan["config"]["model"]["base"] == str(tiny_base)
    assert plan["config"]["model"]["revision"] is None


def test_training_loads_the_configured_base_at_the_pinned_revision(
    stack: dict[str, Any], dataset: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert _train(dataset, out, "--device", "cpu") == 0
    loads = stack["tokenizer"] + stack["model"]
    assert stack["tokenizer"] and stack["model"]
    assert all(name == DEFAULT_BASE for name, _ in loads)
    assert all(kwargs.get("revision") == _pinned_revision() for _, kwargs in loads)
    adapter = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter["base_model_name_or_path"] == DEFAULT_BASE
    assert adapter["revision"] == _pinned_revision()


def test_chat_template_sha256_is_the_hash_of_the_template_text() -> None:
    from groundcite.prompts import chat_template_sha256

    expected = hashlib.sha256(CHAT_TEMPLATE.encode("utf-8")).hexdigest()
    assert chat_template_sha256(tiny_tokenizer()) == expected
    assert chat_template_sha256(SimpleNamespace(chat_template=None)) is None


def test_training_log_records_the_chat_template_sha256_and_tokenizers_version(
    stack: dict[str, Any], dataset: Path, tmp_path: Path, tiny_base: Path
) -> None:
    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", "cpu") == 0
    log = _log(out)
    assert log["chat_template_sha256"] == hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest()
    versions = log["library_versions"]
    assert versions.get("tokenizers") not in (None, "", "not installed")


class _FakeLoadedModel:
    """What AutoModelForCausalLM returns in the eval test: echoes the prompt plus EOS."""

    device = "cpu"

    def __init__(self, eos_id: int) -> None:
        self.eos_id = eos_id

    def to(self, *args: Any, **kwargs: Any) -> _FakeLoadedModel:
        return self

    def eval(self) -> _FakeLoadedModel:
        return self

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        import torch

        return torch.cat([input_ids, torch.tensor([[self.eos_id]])], dim=1)


def test_the_template_sha_at_eval_time_equals_the_one_recorded_at_train_time(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train records the template hash; eval's HFBackend loads the adapter's base tokenizer and
    computes the same hash (GOTCHA G27)."""
    import peft
    import torch
    import transformers

    out = tmp_path / "out"
    assert _train(dataset, out, "--base", str(tiny_base), "--device", "cpu") == 0
    recorded = _log(out)["chat_template_sha256"]
    adapter = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))

    eos_id = int(tiny_tokenizer().eos_token_id)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(hub, "fetch_adapter_config", lambda repo, revision=None: dict(adapter))
    # Eval reads the pushed train_log.json and compares the hashes (tests/test_template_drift).
    log = _log(out)
    monkeypatch.setattr(hub, "fetch_train_log", lambda repo, revision=None: log, raising=False)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: _FakeLoadedModel(eos_id),
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda model, *a, **k: model)

    backend = backends.HFBackend("hf:org/groundcite-tuned")
    reply = backend.generate([{"role": "user", "content": "The engine stalls."}], "system", {})
    assert isinstance(reply, str)
    assert backend.chat_template_sha256 == recorded


# ---------------------------------------------------------------------------------------------
# (4) The Unsloth 4-bit load honors the pinned revision, and the log says what loaded (H1)
# ---------------------------------------------------------------------------------------------

MIRROR = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
MIRROR_SHA = "1" * 40


def _fake_unsloth(
    monkeypatch: pytest.MonkeyPatch, tiny_base: Path, *, always_remap: bool = False
) -> list[dict[str, Any]]:
    """Install a fake ``unsloth`` whose FastLanguageModel acts like Unsloth 2026.10.10: with
    ``load_in_4bit`` and without ``use_exact_model_name=True`` it swaps a known base for its
    pre-quantized mirror and drops ``revision`` (the mirror has no such commit). ``always_remap``
    stands for an Unsloth that can't load the exact pinned repo at all. The loaded model reports
    what it really is in ``config._name_or_path`` and ``config._commit_hash``, as transformers
    sets them."""
    import sys
    import types

    import peft
    import torch
    import transformers

    loads: list[dict[str, Any]] = []
    real_model = transformers.AutoModelForCausalLM.from_pretrained

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> tuple[Any, Any]:
            loads.append({"model_name": model_name, **kwargs})
            exact = bool(kwargs.get("use_exact_model_name")) and not always_remap
            if kwargs.get("load_in_4bit") and not exact and model_name == DEFAULT_BASE:
                name, revision = MIRROR, MIRROR_SHA
            else:
                name, revision = model_name, kwargs.get("revision")
            model = real_model(str(tiny_base), dtype=torch.float32)
            model.config._name_or_path = name
            model.config._commit_hash = revision
            return model, tiny_tokenizer()

        @staticmethod
        def get_peft_model(model: Any, **kwargs: Any) -> Any:
            config = peft.LoraConfig(
                r=kwargs["r"],
                lora_alpha=kwargs["lora_alpha"],
                lora_dropout=kwargs["lora_dropout"],
                target_modules=kwargs["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            return peft.get_peft_model(model, config)

    module = types.ModuleType("unsloth")
    module.FastLanguageModel = FastLanguageModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return loads


def _train_unsloth(dataset: Path, out: Path) -> int:
    argv = ["--dataset", str(dataset), "--out", str(out), "--seed", "3", "--backend", "unsloth"]
    return _main(load_train_module(), [*argv, "--max-steps", "2", "--device", "cuda"])


def test_unsloth_loads_the_exact_pinned_base_revision(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_exact_model_name=True`` stops Unsloth swapping in its 4-bit mirror (which has no
    such commit, so the pinned revision would be dropped); it quantizes the pinned repo itself."""
    _fake_cuda(monkeypatch)
    loads = _fake_unsloth(monkeypatch, tiny_base)
    out = tmp_path / "out"
    assert _train_unsloth(dataset, out) == 0
    assert len(loads) == 1, "one Unsloth base load"
    load = loads[0]
    assert load["model_name"] == DEFAULT_BASE
    assert load.get("revision") == _pinned_revision()
    assert load.get("load_in_4bit") is True
    assert load.get("use_exact_model_name") is True, (
        "without it Unsloth loads unsloth/...-bnb-4bit at its default branch"
    )


def test_unsloth_run_logs_the_base_it_actually_loaded(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    _fake_unsloth(monkeypatch, tiny_base)
    out = tmp_path / "out"
    assert _train_unsloth(dataset, out) == 0
    loaded = _log(out)["runtime"].get("loaded_base")
    assert loaded == {"name": DEFAULT_BASE, "revision": _pinned_revision()}, (
        "runtime.loaded_base records the repo and commit the weights came from"
    )


def test_unsloth_log_does_not_claim_the_pin_when_a_mirror_loaded(
    stack: dict[str, Any],
    dataset: Path,
    tmp_path: Path,
    tiny_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Unsloth still swaps in its mirror, the log names the mirror and its commit, not the
    configured base at the pinned revision."""
    _fake_cuda(monkeypatch)
    _fake_unsloth(monkeypatch, tiny_base, always_remap=True)
    out = tmp_path / "out"
    assert _train_unsloth(dataset, out) == 0
    loaded = _log(out)["runtime"].get("loaded_base")
    assert loaded == {"name": MIRROR, "revision": MIRROR_SHA}
