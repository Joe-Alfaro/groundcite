#!/usr/bin/env python3
"""QLoRA fine-tuning on a train.jsonl dataset (R5, R6, R7, R8, R9, R14, R27, R31).

    uv run train/train_qlora.py --dataset data/datasets/v1/train.jsonl --out runs/v1 --seed 0 \\
        [--base Qwen/Qwen3-1.7B] [--max-steps N] [--epochs E] [--batch-size N] \\
        [--backend unsloth|peft] [--device auto|cpu|cuda] [--hub-repo org/name] [--dry-run]

Hyperparameters come from ``train/config.toml`` only; ``--base``, ``--max-steps``, ``--epochs``
and ``--batch-size`` override it and are part of the logged config hash. The dataset, output dir,
seed, backend and device are not, so v1 and v2 runs share one config hash (R7). The configured base
is loaded at its pinned Hub ``revision``; ``--base`` clears the pin (GOTCHA G27).

Backends: ``unsloth`` (default) is the Kaggle T4 path: a 4-bit base, fp16, Unsloth's kernels. It
needs a CUDA GPU and the ``unsloth`` package, which the Kaggle notebook installs; it is not a
project dependency. ``peft`` trains the same LoRA through transformers and peft on CUDA, MPS or
CPU (4-bit only on CUDA with bitsandbytes), for the smoke loop and small local runs.

``--device cpu`` keeps the run on the CPU even where MPS or CUDA is available (the capped local
check, D-078); ``cuda`` requires a CUDA GPU; ``auto`` (default) lets the Trainer choose.

Every run renders its examples with the eval renderer (``render_chat_text``, thinking off) and
trains on the final reply plus EOS only (R27). It writes the adapter, a model card (``README.md``)
and ``train_log.json`` into ``--out``. ``--hub-repo`` pushes them in one Hub commit after the
scrub check (D-031) and logs that commit's SHA. ``--dry-run`` prints the resolved plan as JSON
and writes nothing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from groundcite.keys import optional_key, redact
from groundcite.prompts import chat_template_sha256, render_chat_text
from groundcite.versions import git_dirty, git_sha, library_versions

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRUB_CHECK = REPO_ROOT / "scripts" / "scrub_check.py"
BACKENDS = ("unsloth", "peft")
DEVICES = ("auto", "cpu", "cuda")
IGNORE_INDEX = -100
LOG_FILE = "train_log.json"
CARD_FILE = "README.md"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# Recorded on top of groundcite.versions.LIBRARIES.
TRAIN_LIBRARIES = (
    "tokenizers",
    "accelerate",
    "huggingface-hub",
    "bitsandbytes",
    "unsloth",
    "unsloth_zoo",
)


class UsageError(RuntimeError):
    """Bad arguments or input: one line on stderr, exit 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:  # one line, no usage block
        raise UsageError(message)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1: {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="train_qlora.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="train.jsonl to train on")
    parser.add_argument("--out", type=Path, required=True, help="output dir for the adapter")
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    parser.add_argument("--base", help="base model (default: config [model] base)")
    parser.add_argument("--max-steps", type=_positive_int, help="stop after N optimizer steps")
    parser.add_argument("--epochs", type=_positive_int, help="override num_train_epochs")
    parser.add_argument(
        "--batch-size", type=_positive_int, help="override per_device_train_batch_size"
    )
    parser.add_argument("--backend", choices=BACKENDS, default="unsloth", help="training stack")
    parser.add_argument(
        "--device", choices=DEVICES, default="auto", help="cpu never uses MPS or CUDA"
    )
    parser.add_argument("--hub-repo", help="push the adapter to this Hub repo (needs HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    return parser


# ---------------------------------------------------------------------------------------------
# Config and dataset
# ---------------------------------------------------------------------------------------------


def resolve_config(
    base: str | None = None,
    max_steps: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """``config.toml`` with the hyperparameter overrides applied. Another base drops the pinned
    revision, which belongs to the configured base."""
    config = copy.deepcopy(tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    config["model"].setdefault("revision", None)
    if base is not None:
        config["model"]["base"] = base
        config["model"]["revision"] = None
    if max_steps is not None:
        config["train"]["max_steps"] = max_steps
    if epochs is not None:
        config["train"]["num_train_epochs"] = epochs
    if batch_size is not None:
        config["train"]["per_device_train_batch_size"] = batch_size
    return config


def config_sha256(config: Mapping[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """train.jsonl records, each ending in the assistant reply to train on."""
    if not path.is_file():
        raise UsageError(f"dataset not found: {path}")
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{path}:{lineno}: not JSON ({exc.msg})") from None
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or len(messages) < 2:
            raise UsageError(f"{path}:{lineno}: messages: required, a list of chat turns")
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                raise UsageError(f"{path}:{lineno}: messages: every turn needs string content")
        if messages[-1].get("role") != "assistant":
            raise UsageError(f"{path}:{lineno}: messages: the last turn must be the assistant")
        records.append(record)
    if not records:
        raise UsageError(f"dataset is empty: {path}")
    return records


def dataset_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = Counter(str((r.get("meta") or {}).get("category", "unknown")) for r in records)
    return {"examples": len(records), "by_category": dict(sorted(categories.items()))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Formatting (R27): the eval renderer, response-only labels ending in EOS
# ---------------------------------------------------------------------------------------------


def _split(record: Mapping[str, Any]) -> tuple[str | None, list[dict[str, str]], str]:
    """(system, the turns before the final reply, the final reply)."""
    messages = list(record["messages"])
    system = None
    if messages[0]["role"] == "system":
        system = str(messages[0]["content"])
        messages = messages[1:]
    turns = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages[:-1]]
    return system, turns, str(messages[-1]["content"])


def training_text(tokenizer: Any, record: Mapping[str, Any]) -> str:
    """The full training string: the eval prompt plus the final reply, thinking off."""
    system, turns, reply = _split(record)
    return render_chat_text(tokenizer, system, turns, reply=reply)


def build_example(tokenizer: Any, record: Mapping[str, Any]) -> dict[str, list[int]]:
    """Token ids and labels: the eval prompt masked, then the final reply and EOS as targets.

    The prompt and the target are tokenized separately, so the prompt ids are exactly the ids the
    model sees at eval time, and nothing the template adds after EOS is kept.
    """
    system, turns, reply = _split(record)
    prompt = render_chat_text(tokenizer, system, turns)
    full = render_chat_text(tokenizer, system, turns, reply=reply)
    if not full.startswith(prompt):
        raise ValueError("the chat template's training string does not extend its eval prompt")
    rest = full[len(prompt) :]
    eos = str(tokenizer.eos_token)
    end = rest.find(eos)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if end >= 0:
        target_ids = list(tokenizer(rest[: end + len(eos)], add_special_tokens=False)["input_ids"])
    else:
        target_ids = list(tokenizer(rest, add_special_tokens=False)["input_ids"])
    if not target_ids or target_ids[-1] != tokenizer.eos_token_id:
        target_ids.append(int(tokenizer.eos_token_id))
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def p99_tokens(tokenizer: Any, records: Sequence[Mapping[str, Any]]) -> int:
    """Nearest-rank 99th percentile of the full training-text token counts."""
    lengths = sorted(
        len(tokenizer(training_text(tokenizer, r), add_special_tokens=False)["input_ids"])
        for r in records
    )
    return lengths[max(0, math.ceil(0.99 * len(lengths)) - 1)]


def load_tokenizer(base: str, *, revision: str | None, local_only: bool) -> Any:
    """The base model's tokenizer at its pinned revision: the same one eval loads, so the template
    matches (R27, GOTCHA G27)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base, revision=revision, local_files_only=local_only)


def _check_length(p99: int | None, max_seq_length: int) -> None:
    if p99 is not None and p99 > max_seq_length:
        raise UsageError(
            f"the p99 training example is {p99} tokens, above max_seq_length {max_seq_length}; "
            "shorten the data (the config is frozen)"
        )


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------


def _import_unsloth() -> ModuleType:
    """Unsloth, imported before transformers so it can patch it."""
    try:
        import unsloth
    except (ImportError, NotImplementedError) as exc:
        reason = "not installed" if isinstance(exc, ImportError) else "needs a CUDA GPU"
        raise UsageError(
            f"the unsloth backend is unavailable (unsloth {reason}); install it on a CUDA "
            "machine (see train/kaggle_train.ipynb) or use --backend peft"
        ) from None
    return unsloth


def _load_unsloth(config: Mapping[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    import torch
    from unsloth import FastLanguageModel

    model_cfg, lora, train = config["model"], config["lora"], config["train"]
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base"],
        max_seq_length=train["max_seq_length"],
        dtype=torch.float16 if train["fp16"] else None,
        load_in_4bit=model_cfg["load_in_4bit"],
        revision=model_cfg["revision"],
        # Without it, Unsloth swaps a known base for its pre-quantized 4-bit mirror and drops
        # the revision (the mirror has no such commit); with it, it quantizes the pinned repo.
        use_exact_model_name=True,
    )
    loaded = loaded_base(model)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        bias="none",
        use_gradient_checkpointing="unsloth" if train["gradient_checkpointing"] else False,
        random_state=seed,
    )
    runtime = {
        "device": "cuda",
        "load_in_4bit": bool(model_cfg["load_in_4bit"]),
        "fp16": bool(train["fp16"]),
        "gradient_checkpointing": bool(train["gradient_checkpointing"]),
        "loaded_base": loaded,
    }
    return model, runtime


def loaded_base(model: Any) -> dict[str, Any]:
    """The repo and commit the loaded weights came from, as transformers records them on the
    model's config: a mirror swap shows up here, not as the configured base (D-089)."""
    config = model.config
    return {
        "name": getattr(config, "_name_or_path", None),
        "revision": getattr(config, "_commit_hash", None),
    }


def _load_peft(config: Mapping[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    """The base plus a LoRA on ``device`` (``cuda``, ``mps`` or ``cpu``): fp16 on CUDA (4-bit
    when bitsandbytes is installed), fp32 elsewhere."""
    import peft
    import torch
    from transformers import AutoModelForCausalLM

    model_cfg, lora, train = config["model"], config["lora"], config["train"]
    cuda = device == "cuda"
    quantize = (
        cuda and model_cfg["load_in_4bit"] and importlib.util.find_spec("bitsandbytes") is not None
    )
    fp16 = cuda and bool(train["fp16"])
    checkpointing = bool(train["gradient_checkpointing"])
    kwargs: dict[str, Any] = {
        "revision": model_cfg["revision"],
        "dtype": torch.float16 if fp16 else torch.float32,
    }
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_cfg["base"], **kwargs)
    loaded = loaded_base(model)
    if quantize:
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=checkpointing
        )
    elif checkpointing:
        # The frozen embeddings give no grads; checkpointed LoRA layers need one on their input.
        model.enable_input_require_grads()
    lora_config = peft.LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
        revision=model_cfg["revision"],
    )
    # peft keeps the trainable LoRA weights in fp32 on an fp16 base (autocast_adapter_dtype), as
    # fp16 mixed precision needs.
    model = peft.get_peft_model(model, lora_config)
    runtime = {
        "device": device,
        "load_in_4bit": bool(quantize),
        "fp16": fp16,
        "gradient_checkpointing": checkpointing,
        "loaded_base": loaded,
    }
    return model, runtime


def resolve_device(requested: str, backend: str) -> str:
    """The device a run trains on. ``cpu`` wins over any GPU; ``cuda`` needs one (exit 2)."""
    import torch

    if requested == "cpu":
        if backend == "unsloth":
            raise UsageError("the unsloth backend needs a CUDA GPU; use --backend peft on cpu")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise UsageError("--device cuda: no CUDA GPU is available; use --device cpu")
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _collate(pad_id: int) -> Any:
    import torch

    def collate(batch: Sequence[Mapping[str, list[int]]]) -> dict[str, Any]:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return collate


def train(
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    backend: str,
    seed: int,
    out: Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Train, save the adapter into ``out`` and return the run facts for the log."""
    resolved = resolve_device(device, backend)
    if backend == "unsloth":
        _import_unsloth()
    import transformers

    train_cfg = config["train"]
    base, revision = config["model"]["base"], config["model"]["revision"]
    tokenizer = load_tokenizer(base, revision=revision, local_only=False)
    p99 = p99_tokens(tokenizer, records)
    _check_length(p99, train_cfg["max_seq_length"])
    examples = [build_example(tokenizer, r) for r in records]
    kept = [e for e in examples if len(e["input_ids"]) <= train_cfg["max_seq_length"]]

    transformers.set_seed(seed)
    if backend == "unsloth":
        model, runtime = _load_unsloth(config, seed)
    else:
        model, runtime = _load_peft(config, resolved)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    with tempfile.TemporaryDirectory(prefix="train_qlora-") as scratch:
        args = transformers.TrainingArguments(
            output_dir=scratch,
            use_cpu=runtime["device"] == "cpu",
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_train_epochs=train_cfg["num_train_epochs"],
            max_steps=train_cfg.get("max_steps", -1),
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_steps=train_cfg["warmup_steps"],
            weight_decay=train_cfg["weight_decay"],
            max_grad_norm=train_cfg["max_grad_norm"],
            optim=train_cfg["optim"],
            logging_steps=train_cfg["logging_steps"],
            fp16=runtime["fp16"],
            bf16=False,
            # Unsloth sets up its own checkpointing in get_peft_model.
            gradient_checkpointing=backend == "peft" and runtime["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=seed,
            data_seed=seed,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=runtime["device"] == "cuda",
        )
        trainer = transformers.Trainer(
            model=model, args=args, train_dataset=kept, data_collator=_collate(pad_id)
        )
        trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    # Unsloth may load a pre-quantized mirror of the base; eval loads the configured base, and
    # runtime.loaded_base in the log records what training really used.
    loaded = runtime["loaded_base"]
    if loaded["name"] != base or (revision is not None and loaded["revision"] != revision):
        print(
            f"warning: trained on {loaded['name']}@{loaded['revision']}, not the configured "
            f"{base}@{revision}; see runtime.loaded_base in train_log.json",
            file=sys.stderr,
        )
    adapter_path = out / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter["base_model_name_or_path"] = base
    adapter["revision"] = revision
    adapter_path.write_text(json.dumps(adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    curve = [
        {"step": int(h["step"]), "loss": float(h["loss"])}
        for h in trainer.state.log_history
        if "loss" in h and "step" in h
    ]
    return {
        "steps": int(trainer.state.global_step),
        "loss_curve": curve,
        "p99_tokens": p99,
        "trained_examples": len(kept),
        "dropped_too_long": len(examples) - len(kept),
        "runtime": runtime,
        "chat_template_sha256": chat_template_sha256(tokenizer),
    }


# ---------------------------------------------------------------------------------------------
# Log, model card and Hub push
# ---------------------------------------------------------------------------------------------


def _library_versions() -> dict[str, str]:
    versions = library_versions()
    for name in TRAIN_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _git_state() -> tuple[str | None, bool | None]:
    try:
        return git_sha(), git_dirty()
    except Exception:  # no git checkout (e.g. an exported copy)
        return None, None


def model_card(log: Mapping[str, Any], repo: str | None) -> str:
    config = log["config"]
    base, lora, train_cfg = config["model"]["base"], config["lora"], config["train"]
    stats = log["dataset"]
    categories = ", ".join(f"{k} {v}" for k, v in stats["by_category"].items())
    return "\n".join(
        [
            "---",
            f"base_model: {base}",
            "library_name: peft",
            "tags: [lora, qlora, synthetic-data]",
            "---",
            "",
            f"# GroundCite LoRA adapter for {base}",
            "",
            f"A LoRA adapter for `{base}`. Given a vehicle complaint and ID-tagged reference "
            "snippets, it gives one next diagnostic step and cites exactly one provided snippet "
            "that supports it; if no snippet applies, it says so, cites nothing and asks for one "
            "specific missing detail.",
            "",
            "The training data is synthetic. The complaints and the TSB-style and "
            "service-manual-style snippets are made up, not real service information, so the "
            "replies are not repair guidance.",
            "",
            "## Training",
            "",
            f"- Base model: `{base}`",
            f"- Backend: {log['backend']} ({log['runtime']['device']}, "
            f"4-bit base: {log['runtime']['load_in_4bit']}, fp16: {log['runtime']['fp16']})",
            f"- Dataset: {stats['examples']} examples ({categories}), "
            f"sha256 `{log['dataset_sha256']}`",
            f"- Config: train/config.toml version {config['version']}, "
            f"sha256 `{log['config_sha256']}`",
            f"- LoRA r={lora['r']}, alpha={lora['alpha']}, dropout={lora['dropout']} on "
            f"{', '.join(lora['target_modules'])}",
            f"- {log['steps']} steps, learning rate {train_cfg['learning_rate']} "
            f"({train_cfg['lr_scheduler_type']}), max_seq_length {log['max_seq_length']}, "
            f"seed {log['seed']}",
            f"- Code: git `{log['git_sha']}`",
            "",
            "Loss is on the final reply and EOS only; prompts render with thinking disabled.",
            "",
            "## Evaluate",
            "",
            "```sh",
            f"uv run eval.py --model {repo or '<hf-repo-id>'} --eval-set <path>",
            "```",
            "",
        ]
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_scrub_terms() -> list[str]:
    spec = importlib.util.spec_from_file_location("scrub_check", SCRUB_CHECK)
    if spec is None or spec.loader is None:
        raise UsageError(f"scrub check not found: {SCRUB_CHECK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = Path(os.environ.get("SCRUB_TERMS_PATH") or module.DEFAULT_TERMS_PATH)
    if not path.is_file():
        raise UsageError(f"--hub-repo needs the scrub terms file (SCRUB_TERMS_PATH): {path}")
    terms: list[str] = module.load_terms(path)
    if not terms:
        raise UsageError(f"the scrub terms file has no terms: {path}")
    return terms


def scrub_hits(folder: Path, terms: Sequence[str]) -> list[str]:
    """``file:line`` of every scrub-term hit in the text files of ``folder``; terms are never
    returned."""
    hits = []
    for path in sorted(folder.rglob("*")):
        data = path.read_bytes() if path.is_file() else b"\0"
        if b"\0" in data:
            continue  # binary (the safetensors weights)
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if any(term in line.lower() for term in terms):
                hits.append(f"{path.relative_to(folder).as_posix()}:{number}")
    return hits


def push_to_hub(
    out: Path, repo: str, token: str, terms: Sequence[str], log: Mapping[str, Any]
) -> str:
    """Upload the adapter, card and log in one commit; return the Hub commit SHA."""
    import huggingface_hub

    with tempfile.TemporaryDirectory(prefix="train_qlora-push-") as staging:
        folder = Path(staging)
        for name in (*ADAPTER_FILES, CARD_FILE, LOG_FILE):
            shutil.copy2(out / name, folder / name)
        hits = scrub_hits(folder, terms)
        if hits:
            raise PushRefused(
                f"scrub check failed on the files to push ({', '.join(hits)}); nothing uploaded"
            )
        api = huggingface_hub.HfApi(token=token)
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=str(folder),
            commit_message=(
                f"Adapter from groundcite {log['git_sha']} (config {log['config_sha256'][:12]}, "
                f"dataset {log['dataset_sha256'][:12]}, seed {log['seed']})"
            ),
        )
    return str(commit.oid)


class PushRefused(RuntimeError):
    """The files to push failed the scrub check."""


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def _plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = resolve_config(args.base, args.max_steps, args.epochs, args.batch_size)
    records = read_dataset(args.dataset)
    plan = {
        "config": config,
        "config_sha256": config_sha256(config),
        "dataset_sha256": sha256_file(args.dataset),
        "seed": args.seed,
        "backend": args.backend,
        "device": args.device,
        "max_seq_length": config["train"]["max_seq_length"],
        "dataset": dataset_stats(records),
    }
    return plan, records


def dry_run(plan: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> int:
    try:
        model_cfg = plan["config"]["model"]
        tokenizer = load_tokenizer(
            model_cfg["base"], revision=model_cfg["revision"], local_only=True
        )
    except Exception:  # not in the local cache: no download on a dry run
        tokenizer = None
    plan["p99_tokens"] = None if tokenizer is None else p99_tokens(tokenizer, records)
    _check_length(plan["p99_tokens"], plan["max_seq_length"])
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    plan, records = _plan(args)
    if args.dry_run:
        return dry_run(plan, records)

    resolve_device(args.device, args.backend)  # an unusable --device fails before any work
    token, terms = None, None
    if args.hub_repo:
        token = optional_key("HF_TOKEN")
        if not token:
            raise UsageError("--hub-repo needs HF_TOKEN (environment or .env); nothing trained")
        terms = _load_scrub_terms()

    facts = train(plan["config"], records, args.backend, args.seed, args.out, args.device)
    sha, dirty = _git_state()
    log = {
        **plan,
        **facts,
        "library_versions": _library_versions(),
        "git_sha": sha,
        "git_dirty": dirty,
        "hub": None,
    }
    (args.out / CARD_FILE).write_text(model_card(log, args.hub_repo), encoding="utf-8")
    _write_json(args.out / LOG_FILE, log)
    print(f"trained {log['steps']} steps; adapter, model card and {LOG_FILE} in {args.out}")

    if args.hub_repo and token and terms is not None:
        try:
            commit_sha = push_to_hub(args.out, args.hub_repo, token, terms, log)
        except PushRefused as exc:
            print(f"train_qlora.py: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            message = redact(f"hub push failed: {type(exc).__name__}: {exc}", [token])
            print(f"train_qlora.py: {message}", file=sys.stderr)
            return 1
        log["hub"] = {"repo": args.hub_repo, "commit_sha": commit_sha}
        _write_json(args.out / LOG_FILE, log)
        print(f"pushed to https://huggingface.co/{args.hub_repo} at commit {commit_sha}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except UsageError as exc:
        print(f"train_qlora.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
