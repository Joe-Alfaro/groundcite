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
real model, QLoRA training, base and tuned generation) happens on a Kaggle GPU. The first QLoRA run
on v1 is done and its adapter is public at
[`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) (commit
`b16b566d0d36d06744c90342fe836a4e6decc4e8`). Base-vs-tuned, the v2 data fix and the curve points
are still behind: they need the tuned-model eval first. The reasons are in the
[write-up's status section](writeup.md#how-the-timeline-went).

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
| R5 | Smoke loop: generate → filter → train (≤20 steps) → eval on ≤32 examples | `scripts/smoke_loop.sh`, `tests/test_smoke_loop.py`; README "Smoke loop". The local mode is a CPU-only check on a tiny model; the real-model smoke (`smoke_loop.sh --kaggle`, Qwen3-0.6B) runs from `train/kaggle_train.ipynb` and has no recorded run yet. The real-model training path has run end to end on Kaggle (`results/training/v1/`); the eval step on a real model is still pending | partial |
| R6 | Dataset v1 (≥300 filtered examples) with manifest: 448 kept of 960 candidates, 28.1% abstain | `data/datasets/v1/` (`train.jsonl`, `manifest.json`, `filter_report.md`, `NOTES.md`, `leakage_report.md`) | done |
| R6 | One QLoRA run of Qwen3-1.7B on v1, training log committed | `results/training/v1/train_log.json` and `results/training/v1/README.md`: Kaggle T4, code commit `72f24f33`, frozen config v1, 168 steps (3 epochs) in 839.7 s, final step loss 0.5064, mean train loss 0.9019. Code: `train/train_qlora.py`, `train/config.toml`, `train/kaggle_train.ipynb`, `tests/test_train_qlora.py`, `tests/test_train_notebook.py` | done |
| R6 | Base-vs-tuned table on the own eval set, with transcripts and pins | `results/base_vs_tuned/heldout/` (headline: tuned 0.48 [0.35, 0.60] / 3.47 vs base zero_shot 0.28 / 2.97, structured 0.17 / 2.35) and `results/base_vs_tuned/eval_own/` (tuned 0.57 / 3.73 vs 0.22 / 2.63, memorization caveat); `docs/writeup.md` section 7. No system meets the bar | done |
| R6, R9 | Draft HF checkpoint and its commit hash (G14) | [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) at commit `b16b566d0d36d06744c90342fe836a4e6decc4e8`, public; recorded in `results/training/v1/README.md`. The model page is the deployed artifact for Milestone 1 | done |
| R7 | Diagnosis note: one v1 failure mode, rate, scenario IDs, data change | `results/base_vs_tuned/heldout/DIAGNOSIS.md`, `docs/writeup.md` section 8: false abstain, 14 of 60 tuned held-out replies, with scenario IDs; data change: in-scope twins of the v1 abstain scenarios (105 twins, 210 teacher calls; `docs/dataset-v2-plan.md`) | done |
<!-- PENDING-B:approved -->
| R7 | v1 and v2 logs share a config hash, dataset diff summary committed | `scripts/diff_datasets.py`, `tests/test_diff_datasets.py`; `data/datasets/v2/` ({{v2_total}} examples, `dataset_sha256` `{{v2_dataset_sha256}}`); the v2 training log has the v1 `config_sha256` | done |
<!-- /PENDING-B -->
<!-- PENDING-B:not-run -->
| R7 | v1 and v2 logs share a config hash, dataset diff summary committed | `scripts/diff_datasets.py`, `tests/test_diff_datasets.py`; `train_qlora.py` logs `config_sha256` and `dataset_sha256`; v2 scenarios ready (`data/scenarios/train_v2_additions.jsonl`, 105 in-scope cite twins, `docs/dataset-v2-plan.md`). Designed, not run: the teacher step (210 calls) was not run within the budget and timeline, so `data/datasets/v2/` does not exist | partial |
<!-- /PENDING-B -->
<!-- PENDING-B:approved -->
| R7 | v2 eval shows the failure-mode rate going down | `{{v2_run_dir}}`, `docs/writeup.md` section 8: held-out false abstains v1 14 of 60, v2 {{v2_false_abstain}} of 60 | {{r7_status_build}} |
<!-- /PENDING-B -->
<!-- PENDING-B:not-run -->
| R7 | v2 eval shows the failure-mode rate going down | Not run: there is no v2 adapter to evaluate (`docs/writeup.md` section 8) | not started |
<!-- /PENDING-B -->
| R8 | ≥2 data-efficiency points on nested subsets | `scripts/make_sweep.py`, `tests/test_sweep.py`; `data/datasets/v1/subsets/`; `results/training/curve/` (4 points with logs: n56, n112, n224, plus the full 448 run in `results/training/v1/`); deterministic-check curve on the held-out set in `results/curve/heldout/README.md`. The judge-scored curve is pending approval of the judge spend | partial |
| R10 | One-command eval writes `results.md`, `results.json`, `transcripts.jsonl`; base rows added automatically | `uv run eval.py --model <hf-repo-id> --eval-set <path>`, `tests/test_eval_cli.py` | done |
| R10 | README first lines name the env var and the hardware | `README.md`, `tests/test_eval_hardening.py` | done |
| R11 | Raw judge transcripts for every run, row count = scenarios × systems | `src/groundcite/report.py`, `tests/test_eval_cli.py`; committed runs: `results/ablation/m1/transcripts.jsonl`, `results/rollout_pilot/pilot1/transcripts.jsonl`, `results/judge_agreement/`, and the base and tuned rows in `results/base_vs_tuned/heldout/` (180) and `results/base_vs_tuned/eval_own/` (300) | done |
| R13 | Pins in every `results.json`, dirty-tree refusal, one `rubric_sha256` | `src/groundcite/versions.py`, `uv.lock`, `tests/test_eval_cli.py`, `scripts/check_rubric.py`, `tests/test_check_rubric.py`; `uv run scripts/check_rubric.py results/` reports one `rubric_sha256` across the 7 committed runs, including both base-vs-tuned runs; the tuned model's Hub commit `b16b566d` and code commit `72f24f33` in `results/training/v1/README.md` | done |
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
| R29 | 20 hand-labeled replies, judge agrees on ≥18 | `results/judge_audit/` (blind sheet, key, labels, `report.json`; 19/20 agree, the one disagreement a borderline repeated citation), `scripts/judge_audit.py` (`--blind-system` hides the system behind an audit ID), `tests/test_judge_audit.py`, `tests/test_judge_audit_heldout.py`, `docs/writeup.md` section 10 | done |
| R30 | Invented-ID rate, abstain precision and recall; 20–30% abstain in training mix | `src/groundcite/report.py`, `tests/test_report.py`; `data/datasets/v1/manifest.json` (abstain share 28.1%) | done |
| R31 | T4 config (`fp16=True`), `max_seq_length` logged, GPU-hour budget table | `train/config.toml`, `train/train_qlora.py`, `tests/test_train_qlora.py`, `tests/test_train_device.py`, `tests/test_gpu_budget.py`; [GPU-hour budget](#r31-gpu-hour-budget) below (Used so far: the v1 training run) | done |
| R32 | Milestone checklist | `docs/checklist.md`; deliverable-by-deliverable view in `docs/milestone1-checklist.md` | done |

## Milestone 2 (Sun Nov 1, noon CT)

| Req | Item | Artifact, command or URL | Status |
| --- | --- | --- | --- |
<!-- PENDING-A:approved -->
| R8 | ≥4 nested curve points, evaluated on own and held-out sets | `results/training/curve/README.md`: 4 nested points trained with logs (N = 56, 112, 224, 448) with one config hash and seed; judge-scored on held-out in `{{curve_judged_run_dir}}`, deterministic checks in `results/curve/heldout/README.md`. `eval_own` for n56, n112 and n224 is not generated | partial |
<!-- /PENDING-A -->
<!-- PENDING-A:not-run -->
| R8 | ≥4 nested curve points, evaluated on own and held-out sets | `results/training/curve/README.md`: 4 nested points trained with logs (N = 56, 112, 224, 448) with one config hash and seed. Deterministic-check curve for all 4 on held-out: `results/curve/heldout/README.md`. The judge-scored curve was not run within the budget and timeline; `eval_own` for n56, n112 and n224 is not generated | partial |
<!-- /PENDING-A -->
| R8 | Re-run every checkpoint on any eval set | `uv run scripts/run_curve.py --eval-set <path>` (planned) | not started |
<!-- PENDING-A:approved -->
| R8 | Plot and table with 95% bootstrap intervals, minimum viable N | `docs/writeup.md` section 9 (judge-scored table; minimum viable N: {{min_viable_n}}), `results/curve/heldout/README.md` (deterministic-check table); no plot | partial |
<!-- /PENDING-A -->
<!-- PENDING-A:not-run -->
| R8 | Plot and table with 95% bootstrap intervals, minimum viable N | `results/curve/heldout/README.md` (deterministic-check table with 95% bootstrap intervals; no plot); `docs/writeup.md` section 9. The judge-scored table was not run, so no minimum viable N is stated | partial |
<!-- /PENDING-A -->
| R8 | Each curve checkpoint as its own HF revision with a recorded SHA (G19) | `joe-alfaro/groundcite-qwen3-1.7b-n{56,112,224}` at `51ae124` / `c049270` / `58532b3` (full SHAs in each `train_log.json`); the curve repos are private, so a third party cannot load them yet | owner-gated |
| R9 | Public tuned model; README gives the exact HF commit hash; eval and demo load that revision | [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) at commit `b16b566d0d36d06744c90342fe836a4e6decc4e8` (public), given in `README.md`; the base-vs-tuned runs and `train/demo_inference.ipynb` load that revision | done |
| R12 | Third-party held-out format documented, loader accepts JSONL, JSON array, CSV | `docs/eval-set-format.md`, `src/groundcite/evalset.py`, `tests/test_evalset.py` | done |
| R12 | Three fixture sets each produce scored output | `tests/fixtures/{our_format.jsonl,prompt_only.csv,prompt_only.json,third_party.jsonl}`, `tests/test_evalset.py`, `tests/test_eval_fixture_sets.py` (end to end through `eval.py`) | done |
| R13 | Pins in every committed `results.json`, one `rubric_sha256` across runs | `uv run scripts/check_rubric.py results/`, `tests/test_check_rubric.py`; rerun on the Milestone 2 results | partial |
| R14 | Rerun one ablation cell | `uv run scripts/run_ablation.py --model <m> --strategy few_shot --limit 30`, `tests/test_ablation.py` | done |
| R14 | Rerun one curve point from a published subset | `uv run train/train_qlora.py --dataset data/datasets/v1/subsets/n<N>.jsonl --out runs/n<N> --seed 0` (CUDA host), documented in `results/training/curve/README.md`; each log's `dataset_sha256` equals the committed subset's sha256. No independent rerun has been done | done |
| R14 | Training logs committed for every point | `results/training/curve/n{56,112,224}/train_log.json`, `results/training/curve/README.md`, `results/training/v1/train_log.json` (448) | done |
| R15 | Public dataset with card ("synthetic, not real service information") | [`joe-alfaro/groundcite-diagnostic-v1`](https://huggingface.co/datasets/joe-alfaro/groundcite-diagnostic-v1) at commit `ccb75f757844360f10fc951d7c4f3db1a57fcb36`, `data/datasets/v1/publish.json` | done |
| R15 | Teacher provider's terms checked before publishing (G38) | dataset card (planned) | owner-gated |
| R16 | Base-vs-tuned demo with check results; no-snippet path abstains | `uv run scripts/demo.py`, `tests/test_demo_cli.py`, `README.md` "Demo CLI" | done |
| R16 | Public notebook link, each reply under 2 minutes, tested logged-out (G31, G32) | Colab or Kaggle link in `README.md` (planned) | owner-gated |
| R17 | Final write-up | `docs/writeup.md`: thesis, spec, scenarios, ablation, pilot, data, base vs tuned, the v2 diagnosis and data change, the curve, the judge audit and agreement study, and limitations. Items not run are stated as not run | done |
| R18 | 3–5 minute demo video, linked from README | link in `README.md` (planned); recorded by the owner | owner-gated |
| R18 | Live prompt from someone outside the project, run in one uncut take | video (planned) | owner-gated |
| G12 | Judge bias: the eval judge (gpt-5.5) is a different family from the teacher, so M1 and the pilot need no re-judge; the ablation table flags the cells that share the judge's family; dataset v1's Claude Opus filter judge is disclosed and pinned in its manifest; the three cheaper agreement candidates failed and are marked rejected | flag in `src/groundcite/ablation.py`, `tests/test_ablation.py`; judge pin and `[judge_candidates]` status in `config/models.toml`, `tests/test_gpt55_judge.py`; `data/datasets/v1/manifest.json` (`judge_model`); `results/judge_agreement/`; `docs/ai-usage.md` | done |
| R30 | Tuned invented-ID rate ≤ base and ≤2% on own and held-out sets | `results/base_vs_tuned/`: tuned 0.02 held-out, 0.00 `eval_own` (≤2% on both, ≤ base on `eval_own`). On held-out it is one reply above base zero_shot (0.00), below base `structured` (0.40). Curve points (deterministic checks): 0.00 to 0.02 on held-out, `results/curve/heldout/README.md` | partial |
| — | Results tables: ablation; base vs tuned on own and held-out sets | `results/ablation/`, `results/base_vs_tuned/heldout/`, `results/base_vs_tuned/eval_own/`, `docs/writeup.md` sections 4 and 7 | done |
| — | Architecture document | `ARCHITECTURE.md` | done |
| — | Canonical spec | `SPEC.md` | done |
| R32 | Milestone checklist, every row filled | `docs/checklist.md`; deliverable-by-deliverable view in `docs/final-checklist.md` | partial |

## R31 GPU-hour budget

Weekly Kaggle GPU quota: 30 h (T4). Every model run happens on Kaggle (D-078), and the plan keeps
at least 25% of the week free. Planned hours are estimates for one T4, including clone, install
and model download; record the hours each session actually used. As of this check-in (Thu Oct 29),
only the v1 training run has happened; its Used figure is the training wall time (16 min 41 s), as
the session's setup time was not recorded.

| Run | Planned (GPU h) | Used (GPU h) |
| --- | --- | --- |
| Smoke loop (`smoke_loop.sh --kaggle`, Qwen3-0.6B, 20 steps) | 0.5 | 0 |
| v1 training (`kaggle_train.ipynb`) | 1.0 | 0.28 |
| v2 training | 1.0 | 0 |
| Curve: 4 nested points (N/8 to N) | 2.5 | 0 |
| Eval generation (`kaggle_eval.ipynb`): base and tuned, own and held-out sets | 4.0 | 0 |
| Eval generation for the curve checkpoints | 3.0 | 0 |
| Reruns and failed sessions | 3.0 | 0 |
| Total | 15.0 | 0.28 |

## Judge API budget

Eval judge: `openai:gpt-5.5-2026-04-23`. Total judge spend cap: $20 actual (raised from $10 by the
owner), covering base vs tuned (held-out set first, then `eval_own`), v2 and the curve points.
Spent so far: about $9.5 ($7.88 logged in `spend.json` plus about $1.6 unlogged); conservatively
about $13.9. The first 83 held-out judgments predate usage logging, so their cost is estimated
at the measured $0.0197 per call (conservatively $6).

| Run | Planned (USD) | Spent (USD) |
| --- | --- | --- |
| Base vs tuned, held-out set (3 rows, 180 calls) | about 5.8 | 1.91 logged + about 1.6 unlogged (conservatively 6) |
| Base vs tuned, `eval_own` (3 rows, 300 calls) | about 9.6 | 5.97 |
| v2 and curve points, held-out set | the rest | 0 |
| Total | 20.0 (cap) | about 9.5 (conservatively about 13.9) |
