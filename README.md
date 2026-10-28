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

Add `--dry-run` to either command to print the planned calls without making any. What exists
and what is still planned is tracked in [`docs/checklist.md`](docs/checklist.md).

## Training

QLoRA training of Qwen3-1.7B runs on a free Kaggle T4 through
[`train/kaggle_train.ipynb`](train/kaggle_train.ipynb). The notebook clones this repo at a pinned
commit, installs the locked dependencies plus Unsloth, reads `HF_TOKEN` from Kaggle Secrets,
trains, pushes the adapter to the Hub and prints a one-line summary with the Hub commit SHA.

One training run (one data-efficiency curve point) from a published subset file:

```sh
uv run train/train_qlora.py --dataset data/datasets/<v>/subsets/n<N>.jsonl --seed <s> --out runs/n<N>
```

- **Config.** Hyperparameters live only in [`train/config.toml`](train/config.toml) and are
  frozen for v1, v2 and every curve point: LoRA r=16, alpha=32, dropout 0.05 on all attention
  and MLP projections; lr 2e-4 cosine, 5% warmup; effective batch 8; 3 epochs; max_seq_length
  2048; fp16 (the T4 has no bf16). `--base`, `--max-steps` and `--epochs` override it.
- **Backends.** `--backend unsloth` (the default) needs a CUDA GPU and Unsloth, which only the
  notebook installs. `--backend peft` trains the same LoRA on CUDA, MPS or CPU (a 4-bit base
  only on CUDA with bitsandbytes).
- **Outputs.** `--out` gets the adapter, a model card and `train_log.json`: the resolved config
  and its SHA-256 (the same for v1 and v2), the dataset SHA-256, seed, steps, loss curve, p99
  token length, library versions and git SHA. `--hub-repo <org/name>` pushes them in one Hub
  commit after the scrub check and logs the commit SHA. `--dry-run` prints the plan and writes
  nothing.

Training renders every example with the eval renderer (thinking off) and puts the loss on the
final reply and EOS only.

## Smoke loop

`scripts/smoke_loop.sh [OUT_DIR]` runs the whole loop on a smoke batch: generate (mock teacher,
32 training scenarios) → filter (mock judge) → train (peft, 20 steps) → eval (mock model and
judge, 32 own-eval scenarios). It needs no keys. Set `SMOKE_BASE` to train a different base
(default `Qwen/Qwen3-0.6B`). `eval.py` loads models from the Hub, so the eval stage scores the
mock model, not the smoke adapter.

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
