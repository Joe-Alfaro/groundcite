# small_model_finetune

Fine-tuning a small open model to reliably hold one narrow, falsifiable behavior, with a reproducible eval harness.

## Quality gate

Run every check before committing:

```sh
scripts/check.sh
```

It runs, in order, `ruff format --check`, `ruff check`, `mypy src`, `pytest` and the scrub
check (`scripts/scrub_check.py`), and stops at the first failure. Requires [uv](https://docs.astral.sh/uv/).

The scrub check fails if any listed term appears in a tracked file, the current branch name or a
commit message reachable from `HEAD`. It reports only locations (`path:line`, `branch name` or
`commit <sha>`), never the term itself. Terms are read from the file named by `SCRUB_TERMS_PATH`,
one per line (blank lines and `#` comments ignored), defaulting to `_private/scrub_terms.txt` at
the repository root. The check fails if that file is missing or has no terms.

```sh
SCRUB_TERMS_PATH=/path/to/scrub_terms.txt scripts/check.sh
```
