#!/usr/bin/env python3
"""Write a tiny random Qwen3 base for the local smoke loop (R5, D-078).

    uv run scripts/make_tiny_base.py --out DIR

A randomly initialised two-layer Qwen3 (hidden size 64) and a byte-level BPE tokenizer with a
4,000-token vocabulary, trained in-process on the rendered training prompts, with a chat template
of Qwen3's shape (``<|im_start|>role ... <|im_end|>`` turns, an empty think block when thinking is
off) and EOS ``<|im_end|>``. ``train/train_qlora.py --base DIR --device cpu`` trains on it in
seconds and a few hundred MB, where a real Qwen3 checkpoint would need GBs for its 152k-token
logits. Nothing is downloaded; the weights are random, so the adapter it yields is meaningless.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from groundcite.evalset import load_eval_set
from groundcite.prompts import render_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "data" / "scenarios" / "train_pool.jsonl"
VOCAB_SIZE = 4000
EOS = "<|im_end|>"
PAD = "<|endoftext|>"
SPECIAL_TOKENS = (PAD, "<|im_start|>", EOS)

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


def corpus(limit: int = 64) -> Iterator[str]:
    """The zero-shot system prompt and first ``limit`` training conversations, so the BPE
    compresses the smoke data well and examples stay far below max_seq_length."""
    for scenario in load_eval_set(CORPUS)[:limit]:
        system, messages = render_messages(scenario, "zero_shot")
        if system:
            yield system
        for message in messages:
            yield message["content"]


def tiny_tokenizer() -> object:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(corpus(), trainer=trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token=EOS,
        pad_token=PAD,
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


def make_tiny_base(out: Path) -> Path:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = tiny_tokenizer()
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tok),  # type: ignore[arg-type]
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,  # type: ignore[attr-defined]
        pad_token_id=tok.pad_token_id,  # type: ignore[attr-defined]
        bos_token_id=None,
    )
    out.mkdir(parents=True, exist_ok=True)
    Qwen3ForCausalLM(config).save_pretrained(out)
    tok.save_pretrained(out)  # type: ignore[attr-defined]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_tiny_base.py", description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory to write the model to")
    args = parser.parse_args(argv)
    make_tiny_base(args.out)
    print(f"wrote a tiny Qwen3 base to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
