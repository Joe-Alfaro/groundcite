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
Kaggle GPU. The notebooks are ready and tested, but no run had been recorded at this check-in, so
every row that needs a trained model is owner-gated or missing. The reasons are in the
[write-up's status section](writeup.md#status-at-milestone-1).

## Summary

| Status | Rows |
| --- | --- |
| met | 18 |
| partial | 7 |
| missing | 3 |
| owner-gated | 3 |
| **Total** | **31** |

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
| R10 | One-command eval | met | `uv run eval.py --model <hf-repo-id> --eval-set <path>` | Tested end to end on mock models and judge; not yet run on a tuned model, since none exists yet. |
| R11 | Raw judge transcripts as JSONL (score and reasoning per example) | partial | `results/ablation/m1/transcripts.jsonl`, `results/rollout_pilot/pilot1/transcripts.jsonl`, `results/judge_agreement/*/transcripts.jsonl` | Every committed run has them; base-vs-tuned transcripts follow the first trained model. |
| R13 | Pinned versions: eval-code commit and model commit | partial | `src/groundcite/versions.py`, `uv.lock`, `uv run scripts/check_rubric.py results/` | Every `results.json` pins eval code, judge, rubric, eval set and decoding; the tuned model's Hub commit hash waits on the checkpoint. |
| R12 | Accepts a third-party held-out set | partial | `docs/eval-set-format.md`, `src/groundcite/evalset.py`, `tests/test_evalset.py` | Due at Milestone 2. Loader takes JSONL, JSON array or CSV now; the scored-output test through `eval.py` is still planned. |

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
| R5 | Full loop generate → train → eval on a smoke batch | partial | `scripts/smoke_loop.sh`, `tests/test_smoke_loop.py` | Local mode runs end to end as a CPU-only check on a tiny model; the real-model smoke (`--kaggle`, Qwen3-0.6B) has no recorded run yet. |
| R6 | First real QLoRA training run | owner-gated | `train/train_qlora.py`, `train/config.toml`, `train/kaggle_train.ipynb` | Ready to run on a Kaggle T4; waiting on the owner's run. |
| R6, R9 | Draft model checkpoint on the Hub with its commit hash | owner-gated | Hub model repo (planned); the notebook prints the commit SHA | Pending the Kaggle training run. |
| R6 | First base-vs-tuned numbers in the verified format | missing | `results/base_vs_tuned/` (planned), `train/kaggle_eval.ipynb`, `eval.py --judge-from` | Needs the trained adapter. Target: within a day of the Kaggle run. |
| R7 | One failure mode diagnosed from evals and fixed by a data change (v2) | missing | `scripts/diff_datasets.py` (ready); diagnosis note and `data/datasets/v2/` (planned) | The failure mode must come from tuned-model transcripts, not a guess. Candidate weaknesses are noted in `data/datasets/v1/NOTES.md`. |
| R8 | At least two data-efficiency points, or a documented reason | partial | `scripts/make_sweep.py`, `docs/checklist.md`, `docs/writeup.md` (status section) | No points yet; the reason (every run is on Kaggle and v1 is not trained yet) and the plan are documented. |
| R14 | Data-efficiency training logs for rerunning a point | missing | `train/train_qlora.py` writes `train_log.json` (config and dataset hashes, seed, loss curve) | Due with the first curve point; no training run exists yet. |
| R29 | Judge validity evidence | partial | `results/judge_agreement/`, `docs/writeup.md` section 10, `scripts/judge_audit.py` | The agreement study kept gpt-5.5 (it matched a blind label on 17 of 19 disputed replies); the 20-reply hand audit waits for tuned-model replies. |
