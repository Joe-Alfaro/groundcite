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
