# AI usage

Which AI models this project used, for what, and how paid calls were kept in check. Exact model
ids are pinned in [`config/models.toml`](../config/models.toml); every committed result records the
ids that produced it.

## Models in the pipeline

| Role | Model | What it did | Where it is recorded |
| --- | --- | --- | --- |
| Teacher (distillation) | `anthropic:claude-opus-5-5` | Wrote two candidate replies per training scenario (960 for dataset v1), with privileged inputs (gold mode and supporting ID) that never reach the training targets. | `data/datasets/v1/manifest.json` (`teacher_model`, `teacher_prompt_sha256`), `src/groundcite/templates/teacher_prompt_t1.md` |
| Eval judge | `openai:gpt-5.5-2026-04-23` | Scores every reply on Spec-adherence (0/1) and Robustness (1–5) with written reasoning. It scored the prompt-ceiling ablation and the long-conversation pilot, and it will score base vs tuned and the curve. A different family from the teacher. | `config/models.toml` `[judge]`, `judge_model` in every `results.json` and transcript row |
| Dataset v1 filter judge | `anthropic:claude-opus-5-5` | Judged and hint-screened the dataset v1 candidates (458 judge calls and 458 hint screens) while Opus was briefly the pinned judge. It shares the teacher's family; this is disclosed and v1 was not re-filtered. | `data/datasets/v1/manifest.json` (`judge_model`, `hint_screen_sha256`), `data/datasets/v1/NOTES.md`, write-up section 11 |
| Prompt-ceiling ablation subjects | `anthropic:claude-opus-5-5`, `openai:gpt-5.5-2026-04-23` | Each answered the 40 ablation scenarios under three prompt strategies (240 replies). The OpenAI cells share the judge's vendor and are flagged in the table. | `results/ablation/m1/` |
| Long-conversation pilot subject | `anthropic:claude-opus-5-5` (`structured` prompt) | Wrote every assistant turn of 30 scripted 8–12 turn conversations (303 turns). The user turns are scripted, not generated. | `results/rollout_pilot/pilot1/` |
| Judge-agreement candidates (rejected) | `xai:grok-4.20-0309-reasoning`, `anthropic:claude-sonnet-5-5`, `anthropic:claude-haiku-5-5` | Each re-scored the same 100 stored replies to test whether a cheaper judge could replace gpt-5.5. All three failed the bar fixed before the run, so none is used as a judge. | `config/models.toml` `[judge_candidates]`, `results/judge_agreement/`, write-up section 10 |
| Blind adjudicator | an AI agent given no judge or model names | Labeled the 19 replies the candidates disputed against P1–P8, with the verdicts shuffled and unattributed. | `results/judge_agreement/adjudication.jsonl` |
| Base and tuned models | `Qwen/Qwen3-1.7B` (and `Qwen/Qwen3-0.6B` for the smoke run) | The fine-tuning target. Generation runs on a Kaggle GPU with greedy decoding; no run is recorded yet. | `train/config.toml`, `config/models.toml` |

The scenario sets and the snippet library are hand-written and built by a seeded generator
(`scripts/gen_scenarios.py`); no model wrote them. Tests and the key-free paths use mock
backends (`mock:*`) and never call a real API.

## Building the repository

The code, tests and docs were written by AI coding agents (Claude Code), directed and reviewed by
the owner. The agents worked test-first: one agent wrote failing tests, another made them pass
without changing the tests, and a fresh agent validated the result. Design choices were logged
before merge, and every commit passed the quality gate (`scripts/check.sh`: format, lint, types,
tests, leakage, scrub and secret scan). The owner made the gating decisions: the behavior, the
models, the spend and every publish step.

## Spend controls

- **Key-free by default.** Mock backends and `--dry-run` on `eval.py`, `scripts/run_ablation.py`,
  `scripts/distill.py` and `scripts/judge_agreement.py` print the planned calls (and, where prices
  are known, an estimated cost) without making any. Tests never call real APIs.
- **Small, fixed call counts.** `--limit` caps scenarios; the ablation used 40 scenarios per cell
  and the agreement study a committed 100-reply sample.
- **Recorded spend.** Prices per million tokens are pinned in `config/models.toml` `[prices]`. The
  agreement study writes actual token usage and cost per candidate to `spend.json` (about $1.47
  for all three); dataset v1's filter used 916 calls, about $13.40 (`data/datasets/v1/NOTES.md`).
- **Hard stops.** The dataset v1 filter and the Sonnet agreement run went through a wrapper with
  a dollar cap that stopped further calls once recorded spend (plus a margin) would pass it.
- **No paid compute.** Training and model generation run on free Kaggle T4 GPUs, with a
  GPU-hour budget kept under 75% of the weekly quota ([`docs/checklist.md`](checklist.md#r31-gpu-hour-budget)).
- **Owner approval.** Paid runs happened only when assigned under the owner's approval; the larger
  ones (the v1 filter, the agreement study) had a cost estimate approved first.
