# PARTIAL held-out run (not a result)

> Superseded: the run was resumed and completed on 2026-10-30. See `README.md` for the results.

`20261030T171626Z-e1bd6f/` is an incomplete judge pass and must not be read as base-vs-tuned
numbers. It has no `results.json` or `results.md`.

- How it happened: `eval.py --judge-from ... --dry-run` ignored `--dry-run` (the `--judge-from`
  branch returns before the dry-run check) and started real judging. It was stopped by hand.
- Judge: `openai:gpt-5.5-2026-04-23`, rubric sha256 `636a9c79...b4bf`, eval set
  `data/scenarios/heldout.jsonl` sha256 `13dae805...c564` (matches the generation pins).
- Rows judged (83 of 180):

  | System | Judged |
  | --- | --- |
  | `hf:Qwen/Qwen3-1.7B@70d244cc` zero_shot | 60 / 60 |
  | `hf:Qwen/Qwen3-1.7B@70d244cc` structured | 23 / 60 |
  | `hf:joe-alfaro/groundcite-qwen3-1.7b@b16b566d` zero_shot | 0 / 60 |

- Spend: `eval.py` did not record token usage, so actual spend for these calls is unknown from
  the run's own files. It must come from the provider's usage records.
- `inputs/` holds the generations and generation pins as downloaded (Kaggle run at repo commit
  72f24f3, adapter b16b566d), unmodified.

The complete run waits on an `eval.py` fix: honor `--dry-run` under `--judge-from`, persist judge
usage, and resume a `--judge-from` run.
