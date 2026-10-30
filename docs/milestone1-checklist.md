# Milestone 1 check-in: deliverable by deliverable

Every item due at the Milestone 1 check-in (Thu Oct 29, 11:59 PM CT), one row each, mapped to the
requirement ID in [`docs/requirements/small-model-finetune.md`](requirements/small-model-finetune.md).
It covers the Milestone 1 list, the verification requirements as they apply at Milestone 1, and
the Milestone 1 parts of the two required ablations. The finer-grained build checklist, with
Milestone 2 rows, is [`docs/checklist.md`](checklist.md).

Status values:

- **met**: the artifact is on `main` and the gate (`scripts/check.sh`) passes.
- **partial**: part of it is on `main`; the note says what is missing.
- **missing**: due now and not done; the note gives the reason and the plan.
- **owner-gated**: needs the owner's own action (a Kaggle GPU run, a Hub push, a recording).

Every model run (smoke on the real model, QLoRA training, base and tuned generation) happens on a
Kaggle GPU. The first QLoRA run on dataset v1 is done, and its adapter is public on the Hub at
[`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b), commit
`b16b566d0d36d06744c90342fe836a4e6decc4e8`; that model page is the deployed artifact for this
check-in. The adapter has not been evaluated yet, so the rows that need tuned-model numbers are
still missing or partial. The reasons are in the
[write-up's status section](writeup.md#status-at-milestone-1).

## Summary

| Status | Rows |
| --- | --- |
| met | 26 |
| partial | 5 |
| missing | 0 |
| owner-gated | 1 |
| **Total** | **32** |

## Project basics

| Req | Item | Status | Satisfied by | Note |
| --- | --- | --- | --- | --- |
| — | Public code repository link | met | provided in the submission form | The link is not repeated in tracked files; notebooks read it from a Kaggle secret. |
| R10 | README with the eval command, env vars and hardware in its first lines | met | `README.md`, `tests/test_eval_hardening.py` | Its "Project status" block links this file, the build checklist, the write-up and the AI usage notes. |
| R1 | Behavior spec, final and falsifiable, in at most two sentences | met | `SPEC.md`, requirements "Behavior Spec (R1)", `tests/test_spec.py` | Includes the citation format, reliability bar (Spec-adherence ≥ 0.90, Robustness ≥ 4.3) and the P1–P8 rubric. |
| R17 | Write-up draft (behavior thesis and evidence so far) | met | `docs/writeup.md` | A draft is what is due. Thesis, spec, scenarios, ablation, pilot, data and the judge agreement study are written; base vs tuned, v2 and the curve are marked Pending. |
| — | AI usage notes | met | [`docs/ai-usage.md`](ai-usage.md) | Teacher, judges, ablation subjects, rejected judge candidates, coding agents and spend controls. |
| — | Demo video | owner-gated | link provided in the submission form | Being recorded by the owner. The full 3–5 minute video with a live outside prompt is a Milestone 2 item (R18). |

## Eval harness

| Req | Item | Status | Satisfied by | Note |
| --- | --- | --- | --- | --- |
| R2 | LLM-as-judge scoring with written reasoning | met | `src/groundcite/judge.py`, `src/groundcite/rubrics/judge_rubric_v1.md`, `tests/test_judge.py` | Spec-adherence (0/1) and Robustness (1–5), judge `openai:gpt-5.5-2026-04-23` pinned in `config/models.toml`. |
| R2, R21 | Deterministic check for the spec's own failure modes | met | `src/groundcite/checks.py`, `tests/test_checks*.py` | HARD (structural) failures, such as an unprovided or unsupporting cited ID or a wrong citation count, force Spec-adherence to 0; SOFT lexical checks are reported. |
| R2, R10 | Base-vs-tuned comparison in one invocation | met | `eval.py`, `src/groundcite/runner.py`, `tests/test_eval_cli.py` | One `--model` adds base zero-shot and base best-prompt rows automatically. |
| R2 | Harness on `main` before any training code | met | `git log --reverse -- train/` | First `train/` commit `dd36d33` comes after the harness (`b38f644`) and the committed ablation (`2f39d0b`). |
| R10 | One-command eval | met | `uv run eval.py --model <hf-repo-id> --eval-set <path>` | Tested end to end on mock models and judge; not yet run on the tuned adapter. |
| R11 | Raw judge transcripts as JSONL (score and reasoning per example) | met | `results/ablation/m1/transcripts.jsonl`, `results/rollout_pilot/pilot1/transcripts.jsonl`, `results/judge_agreement/*/transcripts.jsonl`, `results/base_vs_tuned/*/2*/transcripts.jsonl` | Every committed run has them, including the base and tuned rows (180 held-out, 300 `eval_own`). |
| R13 | Pinned versions: eval-code commit and model commit | met | `src/groundcite/versions.py`, `uv.lock`, `uv run scripts/check_rubric.py results/`, `results/training/v1/`, `results/base_vs_tuned/` | Every `results.json` pins eval code, judge, rubric, eval set and decoding. The tuned model is pinned to Hub commit `b16b566d0d36d06744c90342fe836a4e6decc4e8`, trained from code commit `72f24f33` (both in `results/training/v1/README.md`), and both base-vs-tuned runs pin the tuned rows to that commit. `check_rubric.py results/` reports one rubric hash across 7 runs. |
| R12 | Accepts a third-party held-out set | met | `docs/eval-set-format.md`, `src/groundcite/evalset.py`, `tests/test_evalset.py`, `tests/test_eval_fixture_sets.py` | Due at Milestone 2 and done since the check-in: the loader takes JSONL, JSON array or CSV, and each fixture set is tested end to end through `eval.py` to scored output. |

## Prompt-ceiling ablation (required in full at Milestone 1)

| Req | Item | Status | Satisfied by | Note |
| --- | --- | --- | --- | --- |
| R3 | Two frontier models from different vendors | met | `results/ablation/m1/run.json` | `anthropic:claude-opus-5-5` and `openai:gpt-5.5-2026-04-23`. |
| R3 | Three prompt strategies per model | met | `src/groundcite/prompts.py`, `data/few_shot/examples.jsonl` | `zero_shot`, `few_shot` and `structured` (explicit rules plus a private checklist). |
| R3 | At least 30 scenarios per cell, same IDs, same judge rubric as later evals | met | `data/scenarios/ablation.jsonl`, `results/ablation/m1/results.md` | 40 scenarios in every cell (240 replies), one `rubric_sha256`. |
| R3 | Results table: mean Spec-adherence and Robustness per model × strategy | met | `results/ablation/README.md`, `results/ablation/m1/results.md` | With 95% CIs, per-category scores and best-vs-second-best deltas. |
| R3 | Paragraph naming the failure mode that survives the best prompt | met | `results/ablation/README.md`, `docs/writeup.md` section 4 | Gives its rate and at least two scenario IDs. |
| R3 | Prompting plateaus below the reliability bar | partial | `results/ablation/README.md`, `results/rollout_pilot/README.md`, `docs/writeup.md` sections 4 and 4a | No plateau: three cells meet the bar. The spec was hardened for long conversations; the best cell passes 0.80 [0.63, 0.93] of 8–12 turn conversations, below the bar but not decisively. Stated plainly in the write-up. |
| R14 | Prompt-ceiling script can rerun one cell | met | `uv run scripts/run_ablation.py --model <m> --strategy few_shot --limit 30` | `--dry-run` prints the plan with no calls. |

## Data, training and the data fix

| Req | Item | Status | Satisfied by | Note |
| --- | --- | --- | --- | --- |
| R4 | Distillation from a frontier teacher plus a hard quality filter | met | `scripts/distill.py`, `scripts/build_dataset.py`, `data/datasets/v1/filter_report.md` | Teacher `anthropic:claude-opus-5-5`, two candidates per scenario. |
| R6 | First real filtered dataset, and its shape | met | `data/datasets/v1/` (`train.jsonl`, `manifest.json`, `NOTES.md`, `leakage_report.md`) | 448 kept of 960 candidates, 28.1% abstain, counts by category and ID scheme in the manifest. Its filter judge was Claude Opus, disclosed in the README and write-up. |
| R5 | Full loop generate → train → eval on a smoke batch | partial | `scripts/smoke_loop.sh`, `tests/test_smoke_loop.py` | Local mode runs end to end as a CPU-only check on a tiny model. The real-model training path has now run end to end on Kaggle (`results/training/v1/`); the eval step on a real model is still pending. |
| R6 | First real QLoRA training run | met | `results/training/v1/` (`train_log.json`, `README.md`), `train/train_qlora.py`, `train/config.toml`, `train/kaggle_train.ipynb` | Qwen3-1.7B on dataset v1 with frozen config v1, Kaggle T4, code commit `72f24f33`: 168 steps (3 epochs) in 839.7 s, final step loss 0.5064, mean train loss 0.9019, no errors. |
| R6, R9 | Draft model checkpoint on the Hub with its commit hash | met | [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) at commit `b16b566d0d36d06744c90342fe836a4e6decc4e8`; `results/training/v1/README.md` | Public. The LoRA adapter, its config, the model card and the training log. This model page is the deployed artifact for Milestone 1. |
| R15 | Dataset published | met | [`joe-alfaro/groundcite-diagnostic-v1`](https://huggingface.co/datasets/joe-alfaro/groundcite-diagnostic-v1) at commit `ccb75f757844360f10fc951d7c4f3db1a57fcb36`; `data/datasets/v1/publish.json` | Public. Dataset v1 with its dataset card, published from `main` `92113c3`. |
| R6 | First base-vs-tuned numbers in the verified format | met | `results/base_vs_tuned/heldout/`, `results/base_vs_tuned/eval_own/` (README, `results.json`, `transcripts.jsonl`, `spend.json` each), `docs/writeup.md` section 7 | Held-out (headline): tuned 0.48 [0.35, 0.60] / 3.47 vs base zero_shot 0.28 / 2.97 and base structured 0.17 / 2.35. `eval_own`: tuned 0.57 / 3.73 vs 0.22 / 2.63. No system meets the bar. |
| R7 | One failure mode diagnosed from evals and fixed by a data change (v2) | partial | `results/base_vs_tuned/heldout/DIAGNOSIS.md`, `docs/writeup.md` section 8; `scripts/diff_datasets.py` (ready); `data/scenarios/train_v2_additions.jsonl`, `docs/dataset-v2-plan.md` | Diagnosis done from tuned-model transcripts: false abstain, 14 of 60 held-out replies. The v2 scenarios are ready (105 in-scope cite twins of the v1 abstain scenarios); the teacher spend awaits approval. |
| R8 | At least two data-efficiency points, or a documented reason | partial | `scripts/make_sweep.py`, `data/datasets/v1/subsets/`, `results/training/curve/README.md`, `results/curve/heldout/README.md`, `docs/writeup.md` section 9 | Since the check-in, the curve has 4 nested points trained with logs (N = 56, 112, 224 with the v1 config and seed, plus the full 448 run). The deterministic-check curve on the held-out set is in `results/curve/heldout/README.md`; the judge-scored curve is pending approval of the judge spend. |
| R14 | Data-efficiency training logs for rerunning a point | met | `results/training/curve/n{56,112,224}/train_log.json`, `results/training/curve/README.md`, `results/training/v1/train_log.json` | Every point's log has the config and dataset hashes, seed and loss curve; the README gives the rerun command. |
| R29 | Judge validity evidence | partial | `results/judge_agreement/`, `docs/writeup.md` section 10, `scripts/judge_audit.py` | The agreement study kept gpt-5.5 (it matched a blind label on 17 of 19 disputed replies). The 20-reply hand-audit tooling is done (`judge_audit.py --blind-system`); the hand audit is pending. |

## Judge API budget

Next to the [R31 GPU-hour budget](checklist.md#r31-gpu-hour-budget): the eval judge is
`openai:gpt-5.5-2026-04-23`, with a total judge spend cap of $20 actual (raised from $10 by the
owner) for base vs tuned (held-out set first, then `eval_own`), v2 and the curve points. Spent so
far: about $9.5 ($7.88 logged plus about $1.6 unlogged for the first 83 held-out judgments;
conservatively about $13.9 if those cost $6). Both base-vs-tuned sets are judged in full, and the
rest is for v2 and the curve on the held-out set. Details:
[judge API budget](checklist.md#judge-api-budget).
