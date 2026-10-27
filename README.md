# small_model_finetune

Fine-tuning a small open model to reliably hold one narrow, falsifiable behavior, with a reproducible eval harness.

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
