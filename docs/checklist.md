# Milestone checklist

Maps every Milestone 1 and Milestone 2 item in
[`docs/requirements/small-model-finetune.md`](requirements/small-model-finetune.md) to the
artifact path, command or URL that shows it is done (R32).

Status values:

- **done**: the artifact is on `main` and its tests pass in `scripts/check.sh`.
- **partial**: part of it is on `main`; the rest is named in the row.
- **behind**: due at this milestone and not done; the row gives the reason and a target date.
- **not started**: nothing for it exists yet (Milestone 2 rows only). The path shown is the
  planned one.
- **owner-gated**: needs the owner's approval or action, such as running a Kaggle notebook or a
  Hub push (see "Owner-gated deliverables" in the requirements file).

Every row is filled in before each milestone. Dates are Central Time.

## Milestone 1 (Thu Oct 29, 11:59 PM CT)

Summary: the spec, the eval harness, the prompt-ceiling ablation, the long-conversation pilot,
the judge agreement study and dataset v1 (448 examples) are done. Every model run (smoke on the
real model, QLoRA training, base and tuned generation) happens on a Kaggle GPU, and the notebooks
are ready but no run has been recorded yet, so base-vs-tuned, the v2 data fix, the curve points and the Hub
checkpoint are behind or waiting on that run. The reasons are in the
[write-up's status section](writeup.md#status-at-milestone-1).

| Req | Item | Artifact, command or URL | Status |
| --- | --- | --- | --- |
| R1 | Behavior spec, citation format, reliability bar, P1–P8 rubric, worked examples | `SPEC.md`, requirements file "Behavior Spec (R1)", `tests/test_spec.py` | done |
| R1 | Judge rubric loads `SPEC.md` by path | `src/groundcite/rubrics/judge_rubric_v1.md`, `src/groundcite/judge.py`, `tests/test_judge.py` | done |
| R1 | Teacher prompt and filter load `SPEC.md` by path | `src/groundcite/templates/teacher_prompt_t1.md`, `src/groundcite/distill.py`, `src/groundcite/filter.py`, `tests/test_distill.py`, `tests/test_build_dataset.py` | done |
| R1, R2 | Spec hardening for long conversations (H1): 30 scripted 8–12 turn conversations, rollout harness | `data/scenarios/rollout_pilot.jsonl`, `scripts/run_rollouts.py`, `tests/test_rollout.py`, `tests/test_rollout_pilot.py` | done |
| R1, R2 | H1 pilot run with the best M1 cell: conversation pass 0.80 [0.63, 0.93], per-turn 0.94, no late-turn drift | `results/rollout_pilot/README.md`, `results/rollout_pilot/pilot1/`, `docs/writeup.md` section 4a | done |
| R2 | LLM judge: Spec-adherence and Robustness with reasoning, HARD failure forces 0 | `src/groundcite/judge.py`, `tests/test_judge.py` | done |
| R2 | Base and tuned in one invocation, one table | `eval.py`, `src/groundcite/cli.py`, `src/groundcite/runner.py`, `tests/test_eval_cli.py` | done |
| R2 | Harness merged to `main` before any training code | `git log --reverse -- train/` (first commit `dd36d33`) is after the harness (`b38f644`) and the committed ablation (`2f39d0b`) | done |
| R2 | Split eval: generate on a GPU (no keys), judge anywhere | `eval.py --generate-only`, `eval.py --judge-from`, `train/kaggle_eval.ipynb`, `tests/test_kaggle_notebooks.py` | done |
| R2, R11 | Judge agreement study: three cheaper judges against the pinned judge, blind adjudication; all failed the pre-registered bar, the pinned judge stays | `data/judge_agreement/`, `scripts/judge_agreement.py`, `results/judge_agreement/`, `docs/writeup.md` section 10 | done |
| R3 | Prompt-ceiling ablation runner: 2 families × 3 strategies × ≥30 scenarios, same scenario IDs | `scripts/run_ablation.py`, `src/groundcite/ablation.py`, `src/groundcite/prompts.py`, `data/few_shot/examples.jsonl`, `tests/test_ablation.py` | done |
| R3 | Real ablation run with both model families | `results/ablation/m1/` (`run.json`, `results.json`, `transcripts.jsonl`); command in `results/ablation/README.md` | done |
| R3 | Results table, per-example judge JSONL, best-vs-second-best delta, plateau check: **no plateau** (three cells meet the bar) | `results/ablation/README.md`, `results/ablation/m1/results.md`, `results/ablation/m1/transcripts.jsonl` | done |
| R3 | Surviving failure mode paragraph with rate and ≥2 scenario IDs | `results/ablation/README.md`, `docs/writeup.md` section 4 | done |
| R4 | Teacher generation (k = 2), hard quality filter, filter report (v1) | `scripts/distill.py`, `scripts/build_dataset.py`, `data/distill/v1/`, `data/datasets/v1/filter_report.md` | done |
| R5 | Smoke loop: generate → filter → train (≤20 steps) → eval on ≤32 examples | `scripts/smoke_loop.sh`, `tests/test_smoke_loop.py`; README "Smoke loop". The local mode is a CPU-only check on a tiny model; the real-model smoke (`smoke_loop.sh --kaggle`, Qwen3-0.6B) runs from `train/kaggle_train.ipynb` and has no recorded run yet | partial |
| R6 | Dataset v1 (≥300 filtered examples) with manifest: 448 kept of 960 candidates, 28.1% abstain | `data/datasets/v1/` (`train.jsonl`, `manifest.json`, `filter_report.md`, `NOTES.md`, `leakage_report.md`) | done |
| R6 | One QLoRA run of Qwen3-1.7B on v1, training log committed | `train/train_qlora.py`, `train/config.toml`, `train/kaggle_train.ipynb` (pinned to a commit that contains dataset v1), `tests/test_train_qlora.py`, `tests/test_train_notebook.py`. Ready to run; waiting on the owner's Kaggle run. Target: Fri Oct 30 | owner-gated |
| R6 | Base-vs-tuned table on the own eval set, with transcripts and pins | `results/base_vs_tuned/` (planned). Needs the trained adapter, then base and tuned generation on Kaggle (`train/kaggle_eval.ipynb`, held-out set first), then local judging with `eval.py --judge-from`. Target: within a day of the Kaggle run, Sat Oct 31 at the latest | behind |
| R6, R9 | Draft HF checkpoint and its commit hash (G14) | Hub model repo (private until the owner approves public); the commit SHA is printed by the training notebook. Pending the Kaggle training run | owner-gated |
| R7 | Diagnosis note: one v1 failure mode, rate, scenario IDs, data change | `docs/` diagnosis note (planned); candidate weaknesses already noted from reading the data in `data/datasets/v1/NOTES.md`. Needs tuned-model eval evidence first: the failure mode must come from v1 transcripts, not from guessing. Target: Sat Oct 31 | behind |
| R7 | v1 and v2 logs share a config hash, dataset diff summary committed | `scripts/diff_datasets.py`, `tests/test_diff_datasets.py`; `train_qlora.py` logs `config_sha256` and `dataset_sha256`; `data/datasets/v2/` (planned, after the diagnosis note). Target: Sat Oct 31 | behind |
| R7 | v2 eval shows the failure-mode rate going down | `results/base_vs_tuned/` (planned). Needs v2 training on Kaggle. Target: Sun Nov 1, before Milestone 2 | behind |
| R8 | ≥2 data-efficiency points on nested subsets | `scripts/make_sweep.py`, `tests/test_sweep.py`; `results/curve/` (planned). Behind because every training run happens on Kaggle and v1 itself has not been trained yet; the subsets need one Kaggle session each (about 0.6 GPU h). Target: Sat Oct 31 | behind |
| R10 | One-command eval writes `results.md`, `results.json`, `transcripts.jsonl`; base rows added automatically | `uv run eval.py --model <hf-repo-id> --eval-set <path>`, `tests/test_eval_cli.py` | done |
| R10 | README first lines name the env var and the hardware | `README.md`, `tests/test_eval_hardening.py` | done |
| R11 | Raw judge transcripts for every run, row count = scenarios × systems | `src/groundcite/report.py`, `tests/test_eval_cli.py`; committed runs: `results/ablation/m1/transcripts.jsonl`, `results/rollout_pilot/pilot1/transcripts.jsonl`, `results/judge_agreement/` | done |
| R13 | Pins in every `results.json`, dirty-tree refusal, one `rubric_sha256` | `src/groundcite/versions.py`, `uv.lock`, `tests/test_eval_cli.py`, `scripts/check_rubric.py`, `tests/test_check_rubric.py`; `uv run scripts/check_rubric.py results/` reports one `rubric_sha256` across the 5 committed runs | done |
| R17 | Write-up draft | `docs/writeup.md`: status, thesis, spec, scenarios, ablation, pilot, data and the judge agreement study are written; base vs tuned, v2 and the curve are marked Pending | partial |
| R19 | Mock judge, full eval with network blocked | `src/groundcite/backends.py` (`mock:judge`), `tests/test_eval_cli.py` | done |
| R20 | `--dry-run` on `eval.py`, `run_ablation.py`, `distill.py` | `eval.py`, `scripts/run_ablation.py`, `scripts/distill.py`; `tests/test_eval_cli.py`, `tests/test_ablation.py`, `tests/test_distill.py` | done |
| R21 | Deterministic checks with HARD/SOFT classes | `src/groundcite/checks.py`, `tests/test_checks.py`, `tests/test_checks_hard_soft.py`, `tests/test_checks_rubric.py` | done |
| R22 | Seeded scenario sets: ablation, own eval, held-out | `data/scenarios/{ablation,eval_own,heldout}.jsonl`, `docs/scenario-coverage.md`, `uv run scripts/gen_scenarios.py --seed 1027 --out data/scenarios --coverage-doc docs/scenario-coverage.md` | done |
| R23 | Leakage check in the gate, report committed | `scripts/check_leakage.py`, `data/scenarios/leakage_report.md`, `data/datasets/v1/leakage_report.md`, `tests/test_leakage.py` | done |
| R24 | Quality gate: format, lint, types, tests, leakage, scrub, secret scan | `scripts/check.sh`, `tests/test_quality_gate.py`, `tests/test_scrub_check.py`, `tests/test_secret_scan.py` | done |
| R25 | Keys from env vars only, missing key exits 2, no key values in output | `src/groundcite/keys.py`, `tests/test_backends.py` | done |
| R26 | Mock models: `compliant` scores higher than `naive` end to end | `uv run eval.py --model mock:compliant --model mock:naive --eval-set data/scenarios/eval_own.jsonl --judge mock --no-base`, `tests/test_eval_cli.py` | done |
| R27 | One prompt renderer, `enable_thinking=False`, no `<think>` in transcripts; response-only loss and EOS in training | `src/groundcite/prompts.py`, `tests/test_prompts.py`, `tests/test_eval_hardening.py`, `train/train_qlora.py`, `tests/test_train_qlora.py` | done |
| R28 | Greedy local decoding, rerun delta within ±0.03 stated in README | `config/models.toml`, `README.md`, `scripts/rerun_delta.py`, `tests/test_rerun_delta.py`; the measured delta needs a rerun of the tuned eval (after the Kaggle run) | partial |
| R29 | 20 hand-labeled replies, judge agrees on ≥18 | `scripts/judge_audit.py`, `tests/test_judge_audit.py`; labeling waits for tuned-model replies. The agreement study's blind adjudication (gpt-5.5 matched the label on 17 of 19 disputed replies) is related evidence, not this audit | partial |
| R30 | Invented-ID rate, abstain precision and recall; 20–30% abstain in training mix | `src/groundcite/report.py`, `tests/test_report.py`; `data/datasets/v1/manifest.json` (abstain share 28.1%) | done |
| R31 | T4 config (`fp16=True`), `max_seq_length` logged, GPU-hour budget table | `train/config.toml`, `train/train_qlora.py`, `tests/test_train_qlora.py`, `tests/test_train_device.py`, `tests/test_gpu_budget.py`; [GPU-hour budget](#r31-gpu-hour-budget) below (Used is 0 so far: no Kaggle run recorded yet) | done |
| R32 | Milestone checklist | `docs/checklist.md`; deliverable-by-deliverable view in `docs/milestone1-checklist.md` | done |

## Milestone 2 (Sun Nov 1, noon CT)

| Req | Item | Artifact, command or URL | Status |
| --- | --- | --- | --- |
| R8 | ≥4 nested curve points, evaluated on own and held-out sets | `results/curve/` (planned) | not started |
| R8 | Re-run every checkpoint on any eval set | `uv run scripts/run_curve.py --eval-set <path>` (planned) | not started |
| R8 | Plot and table with 95% bootstrap intervals, minimum viable N | `results/curve/`, `docs/writeup.md` (planned) | not started |
| R8 | Each curve checkpoint as its own HF revision with a recorded SHA (G19) | HF Hub model repo revisions (planned) | owner-gated |
| R9 | Public tuned model; README gives the exact HF commit hash; eval and demo load that revision | HF Hub model repo (planned), `README.md` | owner-gated |
| R12 | Third-party held-out format documented, loader accepts JSONL, JSON array, CSV | `docs/eval-set-format.md`, `src/groundcite/evalset.py`, `tests/test_evalset.py` | done |
| R12 | Three fixture sets each produce scored output | `tests/fixtures/{our_format.jsonl,prompt_only.csv,prompt_only.json,third_party.jsonl}`, `tests/test_evalset.py`; scored-output test through `eval.py` (planned) | partial |
| R13 | Pins in every committed `results.json`, one `rubric_sha256` across runs | `uv run scripts/check_rubric.py results/`, `tests/test_check_rubric.py`; rerun on the Milestone 2 results | partial |
| R14 | Rerun one ablation cell | `uv run scripts/run_ablation.py --model <m> --strategy few_shot --limit 30`, `tests/test_ablation.py` | done |
| R14 | Rerun one curve point from a published subset | `uv run train/train_qlora.py --dataset data/datasets/<v>/subsets/n<N>.jsonl --seed <s>` (planned) | not started |
| R14 | Training logs committed for every point | `results/curve/*/train_log.json` (planned) | not started |
| R15 | Public dataset with card ("synthetic, not real service information") | HF Hub dataset repo (planned) | owner-gated |
| R15 | Teacher provider's terms checked before publishing (G38) | dataset card (planned) | owner-gated |
| R16 | Base-vs-tuned demo with check results; no-snippet path abstains | `uv run scripts/demo.py` (planned) | not started |
| R16 | Public notebook link, each reply under 2 minutes, tested logged-out (G31, G32) | Colab or Kaggle link in `README.md` (planned) | owner-gated |
| R17 | Final write-up | `docs/writeup.md` (planned) | not started |
| R18 | 3–5 minute demo video, linked from README | link in `README.md` (planned) | not started |
| R18 | Live prompt from someone outside the project, run in one uncut take | video (planned) | owner-gated |
| G12 | Judge bias: the eval judge (gpt-5.5) is a different family from the teacher, so M1 and the pilot need no re-judge; the ablation table flags the cells that share the judge's family; dataset v1's Claude Opus filter judge is disclosed and pinned in its manifest; the three cheaper agreement candidates failed and are marked rejected | flag in `src/groundcite/ablation.py`, `tests/test_ablation.py`; judge pin and `[judge_candidates]` status in `config/models.toml`, `tests/test_gpt55_judge.py`; `data/datasets/v1/manifest.json` (`judge_model`); `results/judge_agreement/`; `docs/ai-usage.md` | done |
| R30 | Tuned invented-ID rate ≤ base and ≤2% on own and held-out sets | `results/base_vs_tuned/`, `results/curve/` (planned) | not started |
| — | Results tables: ablation; base vs tuned on own and held-out sets | `results/` (planned) and `docs/writeup.md` (planned) | not started |
| — | Architecture document | `ARCHITECTURE.md` | done |
| — | Canonical spec | `SPEC.md` | done |
| R32 | Milestone checklist, every row filled | `docs/checklist.md` | partial |

## R31 GPU-hour budget

Weekly Kaggle GPU quota: 30 h (T4). Every model run happens on Kaggle (D-078), and the plan keeps
at least 25% of the week free. Planned hours are estimates for one T4, including clone, install
and model download; record the hours each session actually used. As of this check-in (Thu Oct 29),
no Kaggle run has been recorded, so Used is 0 on every row.

| Run | Planned (GPU h) | Used (GPU h) |
| --- | --- | --- |
| Smoke loop (`smoke_loop.sh --kaggle`, Qwen3-0.6B, 20 steps) | 0.5 | 0 |
| v1 training (`kaggle_train.ipynb`) | 1.0 | 0 |
| v2 training | 1.0 | 0 |
| Curve: 4 nested points (N/8 to N) | 2.5 | 0 |
| Eval generation (`kaggle_eval.ipynb`): base and tuned, own and held-out sets | 4.0 | 0 |
| Eval generation for the curve checkpoints | 3.0 | 0 |
| Reruns and failed sessions | 3.0 | 0 |
| Total | 15.0 | 0 |
