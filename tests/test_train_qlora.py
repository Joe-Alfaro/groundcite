"""QLoRA training entry point (R5, R6, R7, R8, R9, R14, R27, R31, GOTCHA G28, D-003, D-031).

    uv run train/train_qlora.py --dataset PATH --out DIR --seed S [--base Qwen/Qwen3-1.7B]
        [--max-steps N] [--epochs E] [--backend unsloth|peft] [--hub-repo R] [--dry-run]

The script is loaded in-process and driven through ``main(argv) -> int`` (exit 0 on success,
2 on a usage or input error, with a one-line message on stderr). Its public helpers:

- ``CONFIG_PATH``: ``train/config.toml``, the one versioned training config.
- ``training_text(tokenizer, record) -> str``: a train.jsonl chat record rendered for training.
  It is ``prompts.render_chat_text(tokenizer, system, turns, reply=final_reply)``, the same
  renderer eval uses, with thinking disabled (R27).
- ``build_example(tokenizer, record) -> {"input_ids": [...], "labels": [...]}``: the prompt
  (everything before the final assistant reply) is the eval string ``render_chat_text(tokenizer,
  system, turns)``, and every one of its tokens is masked with -100. Earlier assistant turns are
  scenario context, so they are masked too. The labelled tokens are the final reply followed by
  EOS; anything the template puts after EOS is dropped or masked.

Outputs in ``--out``: the adapter (``adapter_config.json``, ``adapter_model.safetensors``) and
``train_log.json``. ``config_sha256`` is the SHA-256 of the logged resolved ``config`` serialised
as ``json.dumps(config, sort_keys=True, separators=(",", ":"))``. It covers the config file plus
the CLI overrides of the hyperparameters (``--base``, ``--max-steps``, ``--epochs``), never the
dataset, output path or seed, so v1 and v2 runs share it (R7). ``--dry-run`` prints one JSON
object to stdout and writes nothing.

Tests use a tiny Qwen3-shaped tokenizer and a random two-layer Qwen3 model saved to a temp dir
(``tests/train_data.py``); nothing is downloaded and sockets are blocked.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from train_data import (
    CACHED_QWEN,
    CONFIG_PATH,
    DEFAULT_BASE,
    EOS,
    TRAIN_SCRIPT,
    load_train_module,
    p99,
    qwen_tokenizer_cached,
    run_train,
    save_tiny_base,
    scenario_rows,
    sha256_file,
    tiny_tokenizer,
    train_record,
    train_records,
    write_dataset,
)

from groundcite.evalset import parse_row
from groundcite.prompts import render_chat_text, render_messages

ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}
SHA_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
# Built at runtime so the source never holds a key-shaped string (R24).
FAKE_HF_TOKEN = "hf_" + "Zq" * 17
FAKE_COMMIT = "c0ffee" + "0" * 34


@pytest.fixture
def env(isolated_env: Path, no_network: None, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    return isolated_env


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_base(tmp_path_factory.mktemp("tiny-base") / "tiny-qwen3")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "data" / "subsets" / "n10.jsonl", train_records(10))


def _canonical_sha(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dry_run(
    capsys: pytest.CaptureFixture[str], dataset: Path, out: Path, *extra: str
) -> dict[str, Any]:
    capsys.readouterr()
    code = run_train(
        ["--dataset", str(dataset), "--out", str(out), "--seed", "7", "--dry-run", *extra]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return dict(json.loads(captured.out))


def _final_reply(record: dict[str, Any]) -> str:
    return str(record["messages"][-1]["content"])


# ---------------------------------------------------------------------------------------------
# (1) One versioned config file (R8, R31, GOTCHA G28, G29)
# ---------------------------------------------------------------------------------------------


def test_config_file_is_the_one_versioned_training_config() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config["version"], int)
    assert config["model"]["base"] == DEFAULT_BASE
    assert config["model"]["load_in_4bit"] is True

    lora = config["lora"]
    assert lora["r"] == 16
    assert isinstance(lora["alpha"], int) and lora["alpha"] > 0
    assert 0.0 <= lora["dropout"] < 1.0
    # Attention and MLP projections (ARCHITECTURE.md, Training).
    assert ATTENTION | MLP <= set(lora["target_modules"])

    train = config["train"]
    assert 0 < train["learning_rate"] < 1e-2
    assert isinstance(train["lr_scheduler_type"], str) and train["lr_scheduler_type"]
    assert train["per_device_train_batch_size"] >= 1
    assert train["gradient_accumulation_steps"] >= 1
    # The step/epoch policy is stated: fixed epochs for every curve point (R8, G29).
    assert train["step_policy"] == "epochs"
    assert train["num_train_epochs"] == 3
    assert 512 <= train["max_seq_length"] <= 8192


def test_config_targets_the_t4_fp16_not_bf16() -> None:
    """R31: the T4 (sm_75) has no bf16."""
    train = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))["train"]
    assert train["fp16"] is True
    assert train["bf16"] is False


def test_the_script_documents_its_command_line() -> None:
    text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    for flag in ("--dataset", "--out", "--seed", "--base", "--max-steps", "--epochs"):
        assert flag in text
    for flag in ("--backend", "--hub-repo", "--dry-run"):
        assert flag in text
    module = load_train_module()
    assert Path(module.CONFIG_PATH).resolve() == CONFIG_PATH.resolve()


def test_importing_the_script_does_not_import_unsloth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unsloth path is import-guarded: the module loads where unsloth isn't installed."""
    monkeypatch.setitem(sys.modules, "unsloth", None)  # any `import unsloth` now raises
    load_train_module()


# ---------------------------------------------------------------------------------------------
# (1) --dry-run: resolved config, hashes, dataset stats, p99 length; no CUDA, no writes
# ---------------------------------------------------------------------------------------------


def test_dry_run_prints_the_resolved_config_and_hashes(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    plan = _dry_run(capsys, dataset, out, "--base", str(tiny_base))

    config = plan["config"]
    assert config["train"]["fp16"] is True and config["train"]["bf16"] is False
    assert config["model"]["base"] == str(tiny_base)
    assert plan["config_sha256"] == _canonical_sha(config)
    assert plan["dataset_sha256"] == sha256_file(dataset)
    assert plan["seed"] == 7
    assert plan["max_seq_length"] == config["train"]["max_seq_length"]
    assert not out.exists() or not any(out.iterdir()), "--dry-run must write nothing"


def test_dry_run_reports_dataset_stats(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base))
    stats = plan["dataset"]
    assert stats["examples"] == 10
    categories = [r["meta"]["category"] for r in train_records(10)]
    assert stats["by_category"] == {c: categories.count(c) for c in set(categories)}


def test_dry_run_reports_the_p99_tokenized_length(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R31: the p99 is measured on the full training text with the base model's tokenizer."""
    plan = _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base))
    tok = tiny_tokenizer()
    lengths = []
    for record in train_records(10):
        system = record["messages"][0]["content"]
        text = render_chat_text(tok, system, record["messages"][1:-1], reply=_final_reply(record))
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    assert plan["p99_tokens"] == p99(lengths)
    assert plan["max_seq_length"] >= plan["p99_tokens"]


def test_dry_run_with_the_default_base_uses_the_cached_tokenizer_or_reports_none(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no network the default base's tokenizer comes from the local HF cache; if it isn't
    cached the dry run still exits 0 and reports ``p99_tokens: null``."""
    plan = _dry_run(capsys, dataset, tmp_path / "out")
    assert plan["config"]["model"]["base"] == DEFAULT_BASE
    if qwen_tokenizer_cached(DEFAULT_BASE) is None:
        assert plan["p99_tokens"] is None
    else:
        assert isinstance(plan["p99_tokens"], int) and plan["p99_tokens"] > 0


def test_dry_run_touches_neither_cuda_nor_unsloth(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    def cuda_touched(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--dry-run touched torch.cuda")

    for name in ("is_available", "init", "device_count", "current_device"):
        monkeypatch.setattr(torch.cuda, name, cuda_touched)
    monkeypatch.setitem(sys.modules, "unsloth", None)
    _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base), "--backend", "unsloth")


def test_config_sha256_is_stable_across_datasets_outputs_and_seeds(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R7: v1 and v2 runs share the config hash; only the dataset hash differs."""
    other = write_dataset(tmp_path / "v2" / "train.jsonl", train_records(12))
    a = _dry_run(capsys, dataset, tmp_path / "a", "--base", str(tiny_base))
    b = _dry_run(capsys, other, tmp_path / "b", "--base", str(tiny_base))
    capsys.readouterr()
    code = run_train(
        ["--dataset", str(dataset), "--out", str(tmp_path / "c"), "--seed", "11"]
        + ["--base", str(tiny_base), "--dry-run"]
    )
    c = json.loads(capsys.readouterr().out)
    assert code == 0
    assert a["config_sha256"] == b["config_sha256"] == c["config_sha256"]
    assert a["dataset_sha256"] != b["dataset_sha256"]
    assert c["seed"] == 11


def test_hyperparameter_overrides_change_the_config_hash(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _dry_run(capsys, dataset, tmp_path / "a", "--base", str(tiny_base))
    steps = _dry_run(capsys, dataset, tmp_path / "b", "--base", str(tiny_base), "--max-steps", "5")
    epochs = _dry_run(capsys, dataset, tmp_path / "c", "--base", str(tiny_base), "--epochs", "1")
    assert steps["config"]["train"]["max_steps"] == 5
    assert epochs["config"]["train"]["num_train_epochs"] == 1
    hashes = {base["config_sha256"], steps["config_sha256"], epochs["config_sha256"]}
    assert len(hashes) == 3


def test_dry_run_refuses_a_max_seq_length_below_the_p99(
    env: Path, tiny_base: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R31: max_seq_length must cover the p99 tokenized example; truncating targets silently
    would cut the EOS off."""
    long_reply = "Back-probe the sensor signal wire and record the reading. " * 2000
    rows = scenario_rows(4)
    records = [train_record(row, long_reply) for row in rows]
    data = write_dataset(tmp_path / "long.jsonl", records)
    capsys.readouterr()
    code = run_train(
        ["--dataset", str(data), "--out", str(tmp_path / "out"), "--seed", "7"]
        + ["--base", str(tiny_base), "--dry-run"]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "max_seq_length" in err


def test_a_missing_dataset_exits_2(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.jsonl"
    code = run_train(
        ["--dataset", str(missing), "--out", str(tmp_path / "o"), "--seed", "7", "--dry-run"]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "missing.jsonl" in err and "Traceback" not in err


def test_an_unknown_backend_exits_2(
    env: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_train(
        ["--dataset", str(dataset), "--out", str(tmp_path / "o"), "--seed", "7"]
        + ["--backend", "tpu", "--dry-run"]
    )
    assert code == 2


# ---------------------------------------------------------------------------------------------
# (2) Formatting: training text = eval renderer + reply; response-only labels ending in EOS
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 3, 4], ids=["cite", "abstain", "multi-turn"])
def test_training_text_equals_the_shared_renderer_with_thinking_off(index: int) -> None:
    """R27: the training string for a scenario is render_chat_text of the same scenario
    (rendered by render_messages, zero_shot) with the reply appended."""
    module = load_train_module()
    tok = tiny_tokenizer()
    row = scenario_rows(5)[index]
    reply = "Measure the voltage at the sensor connector [TSB-4100]."
    system, messages = render_messages(parse_row(row), "zero_shot")
    expected = render_chat_text(tok, system, messages, reply=reply)

    text = module.training_text(tok, train_record(row, reply))
    assert text == expected
    # Thinking is off: the eval prompt (with its empty think block) is a prefix.
    assert text.startswith(render_chat_text(tok, system, messages))
    assert "<think>\n\n</think>\n\n" + reply in text


@pytest.mark.parametrize("index", [0, 3, 4], ids=["cite", "abstain", "multi-turn"])
def test_labels_mask_every_prompt_token_and_end_with_eos(index: int) -> None:
    module = load_train_module()
    tok = tiny_tokenizer()
    record = train_records(5)[index]
    system = record["messages"][0]["content"]
    prompt = render_chat_text(tok, system, record["messages"][1:-1])
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

    example = module.build_example(tok, record)
    ids, labels = list(example["input_ids"]), list(example["labels"])
    assert len(ids) == len(labels)
    assert ids[: len(prompt_ids)] == prompt_ids
    assert all(label == -100 for label in labels[: len(prompt_ids)])

    target = [label for label in labels if label != -100]
    assert target, "the final reply must be trained on"
    assert target[-1] == tok.eos_token_id
    assert tok.decode(target) == _final_reply(record) + EOS
    # The labelled span is contiguous and every label equals its input id.
    first = labels.index(target[0])
    assert labels[first : first + len(target)] == target
    assert ids[first : first + len(target)] == target


def test_earlier_assistant_turns_are_context_not_targets() -> None:
    module = load_train_module()
    tok = tiny_tokenizer()
    record = train_records(5)[4]
    assert [m["role"] for m in record["messages"]].count("assistant") == 2
    example = module.build_example(tok, record)
    target = [label for label in example["labels"] if label != -100]
    assert "Which engine does the vehicle have?" not in tok.decode(target)


def test_labels_with_the_cached_qwen_tokenizer() -> None:
    """The same response-only contract on the real Qwen3 tokenizer (same template as 1.7B)."""
    tok = qwen_tokenizer_cached(CACHED_QWEN)
    if tok is None:
        pytest.skip(f"{CACHED_QWEN} tokenizer is not in the local HF cache")
    module = load_train_module()
    record = train_records(5)[4]
    system = record["messages"][0]["content"]
    prompt = render_chat_text(tok, system, record["messages"][1:-1])
    assert module.training_text(tok, record).startswith(prompt)
    example = module.build_example(tok, record)
    target = [label for label in example["labels"] if label != -100]
    assert target[-1] == tok.eos_token_id
    assert tok.decode(target) == _final_reply(record) + tok.eos_token
    prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert all(label == -100 for label in example["labels"][:prompt_len])


# ---------------------------------------------------------------------------------------------
# (3) + (4) The peft backend: a CPU/MPS smoke run (R5) and its JSON training log (R6, R14)
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One <=20-step CPU-only peft run on the tiny model, shared by the log tests (D-078: no
    MPS on the dev Mac)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("smoke")
    try:
        for name in ("connect", "connect_ex"):
            mp.setattr(socket.socket, name, _no_network)
        mp.setattr(socket, "create_connection", _no_network)
        mp.setattr(socket, "getaddrinfo", _no_network)
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("TRANSFORMERS_OFFLINE", "1")
        mp.delenv("HF_TOKEN", raising=False)
        (root / "empty.env").write_text("", encoding="utf-8")
        mp.setenv("GROUNDCITE_ENV_FILE", str(root / "empty.env"))
        base = save_tiny_base(root / "tiny-qwen3")
        data = write_dataset(root / "subsets" / "n10.jsonl", train_records(10))
        out = root / "out"
        code = run_train(
            ["--dataset", str(data), "--out", str(out), "--seed", "7", "--base", str(base)]
            + ["--backend", "peft", "--max-steps", "4", "--device", "cpu"]
        )
    finally:
        mp.undo()
    return {"code": code, "out": out, "base": base, "data": data}


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("network access is blocked in tests")


def _log(run: dict[str, Any]) -> dict[str, Any]:
    assert run["code"] == 0
    path = run["out"] / "train_log.json"
    assert path.is_file(), "train_log.json is missing"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def test_peft_smoke_run_saves_a_lora_adapter(smoke_run: dict[str, Any]) -> None:
    assert smoke_run["code"] == 0
    out = smoke_run["out"]
    assert (out / "adapter_model.safetensors").is_file()
    adapter = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter["base_model_name_or_path"] == str(smoke_run["base"])
    assert adapter["r"] == 16
    assert ATTENTION | MLP <= set(adapter["target_modules"])


def test_training_log_records_config_hashes_and_seed(smoke_run: dict[str, Any]) -> None:
    log = _log(smoke_run)
    assert log["config_sha256"] == _canonical_sha(log["config"])
    assert SHA_RE.fullmatch(log["config_sha256"])
    assert log["dataset_sha256"] == sha256_file(smoke_run["data"])
    assert log["seed"] == 7
    assert log["backend"] == "peft"
    assert log["config"]["model"]["base"] == str(smoke_run["base"])
    assert log["config"]["train"]["fp16"] is True and log["config"]["train"]["bf16"] is False


def test_training_log_records_steps_and_a_loss_curve(smoke_run: dict[str, Any]) -> None:
    log = _log(smoke_run)
    assert log["steps"] == 4
    curve = log["loss_curve"]
    assert curve and len(curve) <= 4
    assert [p["step"] for p in curve] == sorted(p["step"] for p in curve)
    assert all(1 <= p["step"] <= 4 for p in curve)
    assert all(isinstance(p["loss"], float) and math.isfinite(p["loss"]) for p in curve)


def test_training_log_records_lengths_versions_and_git_sha(smoke_run: dict[str, Any]) -> None:
    log = _log(smoke_run)
    assert log["max_seq_length"] == log["config"]["train"]["max_seq_length"]
    assert isinstance(log["p99_tokens"], int) and 0 < log["p99_tokens"] <= log["max_seq_length"]
    versions = log["library_versions"]
    for lib in ("python", "torch", "transformers", "peft"):
        assert versions[lib] and versions[lib] != "not installed"
    assert GIT_SHA_RE.fullmatch(log["git_sha"])
    assert log["dataset"]["examples"] == 10
    assert "hub" not in log or log["hub"] is None


def test_unsloth_backend_without_unsloth_fails_cleanly(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "unsloth", None)
    out = tmp_path / "out"
    code = run_train(
        ["--dataset", str(dataset), "--out", str(out), "--seed", "7", "--base", str(tiny_base)]
        + ["--backend", "unsloth", "--max-steps", "1"]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "unsloth" in err and "--backend peft" in err
    assert "Traceback" not in err
    assert not (out / "adapter_model.safetensors").exists()


def test_default_backend_is_unsloth(
    env: Path, tiny_base: Path, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _dry_run(capsys, dataset, tmp_path / "out", "--base", str(tiny_base))
    assert plan["backend"] == "unsloth"


# ---------------------------------------------------------------------------------------------
# (5) --hub-repo: adapter + model card pushed through HfApi, commit SHA logged, token hidden
# ---------------------------------------------------------------------------------------------


class FakeHfApi:
    """Records every call; ``upload_folder`` snapshots the folder it was handed."""

    instances: list[FakeHfApi] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = dict(kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.uploaded: dict[str, str] = {}
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        return SimpleNamespace(repo_id=repo_id)

    def upload_folder(self, **kwargs: Any) -> Any:
        self.calls.append(("upload_folder", dict(kwargs)))
        folder = Path(kwargs["folder_path"])
        for path in folder.rglob("*"):
            if path.is_file():
                rel = path.relative_to(folder).as_posix()
                self.uploaded[rel] = path.read_bytes().decode("utf-8", errors="replace")
        return SimpleNamespace(
            oid=FAKE_COMMIT, commit_url=f"https://huggingface.co/x/commit/{FAKE_COMMIT}"
        )


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> type[FakeHfApi]:
    import huggingface_hub

    monkeypatch.setattr(FakeHfApi, "instances", [])
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


@pytest.fixture
def scrub_terms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "scrub_terms.txt"
    path.write_text("zzqscrubterm\n", encoding="utf-8")
    monkeypatch.setenv("SCRUB_TERMS_PATH", str(path))
    return path


def _push(
    env_file: Path, base: Path, dataset: Path, out: Path, repo: str = "someone/groundcite-test"
) -> int:
    env_file.write_text(f"HF_TOKEN={FAKE_HF_TOKEN}\n", encoding="utf-8")
    return run_train(
        ["--dataset", str(dataset), "--out", str(out), "--seed", "7", "--base", str(base)]
        + ["--backend", "peft", "--max-steps", "2", "--hub-repo", repo]
    )


def test_hub_push_uploads_adapter_and_card_and_logs_the_commit(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    assert _push(env, tiny_base, dataset, out) == 0

    calls = [c for api in fake_hub.instances for c in api.calls]
    uploads = [kw for name, kw in calls if name == "upload_folder"]
    assert len(uploads) == 1
    assert uploads[0]["repo_id"] == "someone/groundcite-test"
    uploaded = {k: v for api in fake_hub.instances for k, v in api.uploaded.items()}
    assert {"adapter_config.json", "adapter_model.safetensors", "README.md"} <= set(uploaded)

    tokens_seen = [api.init_kwargs.get("token") for api in fake_hub.instances]
    tokens_seen += [kw.get("token") for _, kw in calls]
    assert FAKE_HF_TOKEN in tokens_seen

    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert log["hub"]["repo"] == "someone/groundcite-test"
    assert log["hub"]["commit_sha"] == FAKE_COMMIT


def test_model_card_says_synthetic_and_names_the_base(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
) -> None:
    assert _push(env, tiny_base, dataset, tmp_path / "out") == 0
    card = next(api.uploaded["README.md"] for api in fake_hub.instances if api.uploaded)
    assert re.search(r"\bsynthetic\b", card, re.IGNORECASE)
    assert str(tiny_base) in card
    assert sha256_file(dataset) in card


def test_hub_push_never_prints_or_writes_the_token(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    assert _push(env, tiny_base, dataset, out) == 0
    captured = capsys.readouterr()
    assert FAKE_HF_TOKEN not in captured.out + captured.err
    for path in out.rglob("*"):
        if path.is_file():
            assert FAKE_HF_TOKEN.encode() not in path.read_bytes(), path
    for api in fake_hub.instances:
        assert all(FAKE_HF_TOKEN not in text for text in api.uploaded.values())


def test_hub_push_without_a_token_fails_before_training(
    env: Path,
    tiny_base: Path,
    dataset: Path,
    tmp_path: Path,
    fake_hub: type[FakeHfApi],
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    code = run_train(
        ["--dataset", str(dataset), "--out", str(out), "--seed", "7", "--base", str(tiny_base)]
        + ["--backend", "peft", "--max-steps", "2", "--hub-repo", "someone/groundcite-test"]
    )
    assert code == 2
    assert "HF_TOKEN" in capsys.readouterr().err
    assert not (out / "adapter_model.safetensors").exists()
    assert not any(api.calls for api in fake_hub.instances)


def test_hub_push_refuses_files_that_fail_the_scrub_check(
    env: Path,
    tmp_path: Path,
    dataset: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """D-031: pushed files pass the same scrub check as tracked files. The card names the base,
    and this base path holds a scrub term, so nothing is uploaded."""
    base = save_tiny_base(tmp_path / "zzqscrubterm-base")
    code = _push(env, base, dataset, tmp_path / "out")
    err = capsys.readouterr().err
    assert code != 0
    assert "scrub" in err.lower()
    assert "zzqscrubterm" not in err, "the scrub check never prints the term"
    uploads = [c for api in fake_hub.instances for c in api.calls if c[0] == "upload_folder"]
    assert uploads == []
