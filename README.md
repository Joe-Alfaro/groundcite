# small_model_finetune

Fine-tuning a small open model to reliably hold one narrow, falsifiable behavior, with a reproducible eval harness.

## Rerunning the eval

```sh
uv run eval.py --model <hf-repo-id> --eval-set <path>
```

One command: [uv](https://docs.astral.sh/uv/) installs the pinned dependencies (torch,
transformers and peft included) on first run, and the model is downloaded from the Hub.

- **Env vars.** `OPENAI_API_KEY` is required for the pinned judge. `ANTHROPIC_API_KEY` is needed
  only for Anthropic subjects (the prompt-ceiling ablation). `HF_TOKEN` is optional, for gated or
  private repos. Keys are read from the environment or from `.env` at the repo root (override the
  path with `GROUNDCITE_ENV_FILE`). Without keys, `--judge mock` and `--dry-run` still work.
- **Hardware.** No GPU is required. A 1.7B model runs on an Apple silicon Mac (MPS), an NVIDIA
  GPU (CUDA) or, slowly, on CPU. About 8 GB of free memory is enough.
- **Reproducibility.** Local models decode greedily and the judge is a dated snapshot, but API
  sampling is not fully deterministic. The rerun tolerance is ±0.03 Spec-adherence on the same
  pins (eval code, model revision, judge, rubric and eval set, all stamped in `results.json`).

## Reproduce

One-command eval on the own eval set:

```sh
uv run eval.py --model <hf-repo-id> --eval-set data/scenarios/eval_own.jsonl
```

One prompt-ceiling ablation cell (one model, one strategy, the first 30 ablation scenarios):

```sh
uv run scripts/run_ablation.py --model <m> --strategy few_shot --limit 30
```

Where results land:

- `eval.py` writes `results/<run_id>/results.md`, `results.json` and `transcripts.jsonl`.
- `scripts/run_ablation.py` writes the same three files under `results/ablation/<run_id>/`.

### Split eval: generate on a GPU, judge anywhere

Generation and judging can run on different machines. On a Kaggle T4,
[`train/kaggle_eval.ipynb`](train/kaggle_eval.ipynb) runs

```sh
uv run eval.py --model <hf-repo-id> --eval-set <path> --generate-only --out <dir>
```

which generates every row the one-command eval would (base zero-shot, base with the best ablation
prompt, tuned) with no judge and no API key, and writes `generations.jsonl` and
`generation_pins.json` (eval set hash, model revisions, chat-template hash, decoding, library
versions). Then, anywhere with the judge key:

```sh
uv run eval.py --judge-from <dir>/<run_id>/generations.jsonl
```

judges the stored replies with no generation or Hub calls and writes the same three files as the
one-command eval; for the same replies and judge, `results.json` is identical apart from its
`run_id`. It refuses an eval set whose SHA-256 changed since generation.

Add `--dry-run` to either command to print the planned calls without making any. What exists
and what is still planned is tracked in [`docs/checklist.md`](docs/checklist.md).

## Training

QLoRA training of Qwen3-1.7B runs on a free Kaggle T4 through
[`train/kaggle_train.ipynb`](train/kaggle_train.ipynb). The notebook clones this repo at a pinned
commit, installs [`train/requirements-kaggle.txt`](train/requirements-kaggle.txt) (the locked
versions plus Unsloth, around Kaggle's own CUDA torch), reads `HF_TOKEN` from Kaggle Secrets,
trains on one of the "GPU T4 x2" GPUs, pushes the adapter to the Hub and prints a one-line summary
with the Hub commit SHA. Fill in its `# OWNER:` parameters first; it stops while they hold
placeholders. GPU-hour budgets are in [`docs/checklist.md`](docs/checklist.md#r31-gpu-hour-budget).

One training run (one data-efficiency curve point) from a published subset file:

```sh
uv run train/train_qlora.py --dataset data/datasets/<v>/subsets/n<N>.jsonl --seed <s> --out runs/n<N>
```

- **Config.** Hyperparameters live only in [`train/config.toml`](train/config.toml) and are
  frozen for v1, v2 and every curve point: LoRA r=16, alpha=32, dropout 0.05 on all attention
  and MLP projections; lr 2e-4 cosine, 5% warmup; effective batch 8; 3 epochs; max_seq_length
  2048; fp16 (the T4 has no bf16). The base is pinned to a Hub `revision`, so training and eval
load the same weights and chat template. `--base`, `--max-steps`, `--epochs` and
`--batch-size` override it.
- **Backends.** `--backend unsloth` (the default) needs a CUDA GPU and Unsloth, which only the
  notebook installs. `--backend peft` trains the same LoRA on CUDA, MPS or CPU (a 4-bit base
  only on CUDA with bitsandbytes). `--device cpu` keeps a run on the CPU even where MPS or CUDA
  exists; `--device cuda` requires a GPU; `auto` (the default) picks CUDA when it is there.
- **Outputs.** `--out` gets the adapter, a model card and `train_log.json`: the resolved config
  and its SHA-256 (the same for v1 and v2), the dataset SHA-256, seed, steps, loss curve, p99
  token length, chat-template SHA-256, library versions and git SHA. `--hub-repo <org/name>` pushes them in one Hub
  commit after the scrub check and logs the commit SHA. `--dry-run` prints the plan and writes
  nothing.

Training renders every example with the eval renderer (thinking off) and puts the loss on the
final reply and EOS only.

## Smoke loop

`scripts/smoke_loop.sh [OUT_DIR]` runs the whole loop on a smoke batch: generate (mock teacher,
32 training scenarios) → filter (mock judge) → train (peft, 20 steps, batch 1) → eval (mock model
and judge, 32 own-eval scenarios). It needs no keys. `eval.py` loads models from the Hub, so the
eval stage scores the mock model, not the smoke adapter.

- **Local (default).** A CPU-only check: it builds a tiny random small-vocab Qwen3
  (`scripts/make_tiny_base.py`) and trains on it with `--device cpu`. Every stage runs under
  `timeout 120` (or `gtimeout 120`) with CUDA hidden and the MPS fallback off. macOS has no
  `timeout` by default (`brew install coreutils` provides `gtimeout`); without one the script
  stops before any stage.
- **Kaggle.** `scripts/smoke_loop.sh --kaggle [OUT_DIR]` trains the real `Qwen/Qwen3-0.6B` with
  `--device cuda`. The training notebook runs it when `SMOKE_ONLY = True`.

## Quality gate

Run every check before committing:

```sh
scripts/check.sh
```

It runs, in order, `ruff format --check`, `ruff check`, `mypy src`, `pytest`, the leakage
check (`scripts/check_leakage.py`) and the scrub check (`scripts/scrub_check.py`), and stops at
the first failure. Requires [uv](https://docs.astral.sh/uv/).

The leakage check compares `eval_own` against `heldout`, and every committed
`data/datasets/*/train.jsonl` against all three scenario sets, and fails on shared scenario IDs,
identical or near-duplicate final user messages (word 5-gram Jaccard >= 0.6), held-out snippet
text in training, and reserved families in training or few-shot data. It rewrites a
deterministic `leakage_report.md` next to the data it checked.

The scrub check fails if any listed term appears in a tracked file, the current branch name or a
commit message reachable from `HEAD`. It reports only locations (`path:line`, `branch name` or
`commit <sha>`), never the term itself. Terms are read from the file named by `SCRUB_TERMS_PATH`,
one per line (blank lines and `#` comments ignored), defaulting to `_private/scrub_terms.txt` at
the repository root. The check fails if that file is missing or has no terms.

```sh
SCRUB_TERMS_PATH=/path/to/scrub_terms.txt scripts/check.sh
```

## Scenario sets

`data/scenarios/{ablation,eval_own,heldout}.jsonl` are generated from a hand-written, synthetic
snippet library (`data/snippets/`, `data/families.toml`, `data/phrasing.toml`). They are not
real service information. Coverage, quotas and the regeneration command are in
[`docs/scenario-coverage.md`](docs/scenario-coverage.md):

```sh
uv run scripts/gen_scenarios.py --seed 1027 --out data/scenarios --coverage-doc docs/scenario-coverage.md
```
