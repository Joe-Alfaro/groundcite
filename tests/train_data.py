"""Offline fixtures shared by the training tests: train.jsonl records, a tiny tokenizer and a
tiny Qwen3 model. Nothing here downloads anything.

The tokenizer is a byte-level BPE trained in-process on the fixture texts, with a chat template
that follows Qwen3's shape: ``<|im_start|>role\\n...<|im_end|>\\n`` turns, and with
``enable_thinking=False`` an empty ``<think>`` block opens the final assistant turn (in the
generation prompt and in a rendered final reply alike). Without ``enable_thinking=False`` the
generation prompt has no think block, so a renderer that forgets the flag breaks the
train-equals-eval prefix.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from distill_data import ABSTAIN_REPLY, cite_reply, scenario_row

from groundcite.evalset import parse_row
from groundcite.prompts import render_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
TRAIN_SCRIPT = TRAIN_DIR / "train_qlora.py"
CONFIG_PATH = TRAIN_DIR / "config.toml"
NOTEBOOK = TRAIN_DIR / "kaggle_train.ipynb"
SMOKE_LOOP = REPO_ROOT / "scripts" / "smoke_loop.sh"
DEFAULT_BASE = "Qwen/Qwen3-1.7B"
CACHED_QWEN = "Qwen/Qwen3-0.6B"

EOS = "<|im_end|>"
SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", EOS)

CHAT_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' and loop.last -%}"
    "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- else -%}"
    "{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}"
    "{{ '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif -%}"
    "{%- endif -%}"
)


# ---------------------------------------------------------------------------------------------
# train.jsonl records (the scripts/build_dataset.py format)
# ---------------------------------------------------------------------------------------------


def multi_turn_row(index: int) -> dict[str, Any]:
    """A cite scenario whose conversation already holds one assistant turn (scenario context,
    not a training target)."""
    row = scenario_row(index, category="multi_turn")
    first = row["messages"][0]["content"]
    row["messages"] = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "Which engine does the vehicle have?"},
        {"role": "user", "content": "It's the four-cylinder, and the fault comes back daily."},
    ]
    return row


def scenario_rows(n: int) -> list[dict[str, Any]]:
    """``n`` scenario rows: mostly clean cite rows, every fourth an abstain, every fifth
    multi-turn."""
    rows = []
    for i in range(n):
        if i % 5 == 4:
            rows.append(multi_turn_row(i))
        elif i % 4 == 3:
            rows.append(scenario_row(i, category="no_snippet", mode="abstain"))
        else:
            rows.append(scenario_row(i))
    return rows


def reply_for(row: Mapping[str, Any], variant: int = 0) -> str:
    supporting = row["gold"]["supporting_ids"]
    return cite_reply(supporting[0], variant) if supporting else ABSTAIN_REPLY


def train_record(row: Mapping[str, Any], reply: str) -> dict[str, Any]:
    """One train.jsonl line as ``build_dataset.train_record`` writes it."""
    system, messages = render_messages(parse_row(dict(row)), "zero_shot")
    return {
        "messages": [
            {"role": "system", "content": system},
            *messages,
            {"role": "assistant", "content": reply},
        ],
        "meta": {
            "scenario_id": row["id"],
            "category": row["category"],
            "family": row["family"],
            "teacher": "mock:compliant",
            "filter_version": "test",
            "candidate_index": 0,
            "judge": {"spec_adherence": 1, "robustness": 5},
        },
    }


def train_records(n: int) -> list[dict[str, Any]]:
    return [train_record(row, reply_for(row, i)) for i, row in enumerate(scenario_rows(n))]


def write_dataset(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p99(lengths: Sequence[int]) -> int:
    """Nearest-rank 99th percentile: the ceil(0.99 * n)-th smallest value."""
    ordered = sorted(lengths)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]


# ---------------------------------------------------------------------------------------------
# Tiny tokenizer and model
# ---------------------------------------------------------------------------------------------


def tiny_tokenizer(extra_texts: Sequence[str] = ()) -> Any:
    """A byte-level BPE (vocab ~600) trained on the fixture texts, plus the Qwen3-shaped chat
    template. EOS is ``<|im_end|>``; pad is ``<|endoftext|>``."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [m["content"] for r in train_records(12) for m in r["messages"]]
    corpus += [ABSTAIN_REPLY, *extra_texts]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(corpus, trainer=trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token=EOS,
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


def save_tiny_base(directory: Path) -> Path:
    """A randomly initialised two-layer Qwen3 model and the tiny tokenizer, saved to
    ``directory`` so ``--base <directory>`` loads them with no network."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = tiny_tokenizer()
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        bos_token_id=None,
    )
    model = Qwen3ForCausalLM(config)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tok.save_pretrained(directory)
    return directory


def qwen_tokenizer_cached(repo: str) -> Any | None:
    """The real tokenizer for ``repo`` from the local HF cache, or None if it isn't cached."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# train/train_qlora.py, loaded in-process (so the no-network fixture holds)
# ---------------------------------------------------------------------------------------------


def load_train_module() -> ModuleType:
    assert TRAIN_SCRIPT.is_file(), f"{TRAIN_SCRIPT} is missing"
    spec = importlib.util.spec_from_file_location("train_qlora", TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_train(argv: Sequence[str]) -> int:
    """``train_qlora.main(argv)`` in-process; a SystemExit becomes its exit code."""
    module = load_train_module()
    try:
        return int(module.main(list(argv)))
    except SystemExit as exc:
        return int(exc.code or 0)
