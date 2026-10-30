# Milestone 2: deliverable by deliverable

Every deliverable due at Milestone 2 (Sun Nov 1, noon CT), from the
[Deliverables](requirements/small-model-finetune.md#deliverables) section and the Milestone 2
parts of the requirements in
[`docs/requirements/small-model-finetune.md`](requirements/small-model-finetune.md). One row each,
with the file or link that shows it and what still blocks it. The finer-grained build checklist
is [`docs/checklist.md`](checklist.md); the Milestone 1 view is
[`docs/milestone1-checklist.md`](milestone1-checklist.md).

Status values:

- **met**: the artifact is on `main` (or public on the Hub) and the gate (`scripts/check.sh`)
  passes.
- **partial**: part of it exists; the blocking dependency says what is missing.
- **missing**: due and not done.
- **owner-gated**: needs the owner's own action (a recording, a hosted link, an approval).

Status as of Friday, October 30.

## Summary

| Status | Rows |
| --- | --- |
| met | 9 |
| partial | 5 |
| missing | 0 |
| owner-gated | 1 |

## Deliverables

| Req | Deliverable | Status | File or link | Blocking dependency |
| --- | --- | --- | --- | --- |
| R15 | Public dataset with card ("synthetic, not real service information") | met | [`joe-alfaro/groundcite-diagnostic-v1`](https://huggingface.co/datasets/joe-alfaro/groundcite-diagnostic-v1) at commit `ccb75f757844360f10fc951d7c4f3db1a57fcb36`; publish record `data/datasets/v1/publish.json` | None. This is dataset v1; if v2 is built, it is a new publish. |
| R9, R13 | Public tuned model with its exact commit hash; eval and demo load that revision | met | [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) at commit `b16b566d0d36d06744c90342fe836a4e6decc4e8`, given in `README.md` | None for the v1 adapter. The three curve adapters are private (see R8). |
| R16 | Demo notebook, base vs tuned, with a public hosted link | partial | [`train/demo_inference.ipynb`](../train/demo_inference.ipynb) (pinned base and adapter side by side, with every deterministic check) | The hosted, logged-out link (Kaggle or Colab, each reply under 2 minutes) is not published; `README.md` still has the TODO-PUBLISH line. Owner action. |
| R16 | Demo CLI: `uv run scripts/demo.py` with a complaint, optional snippets or a scenario file; the no-snippet path abstains | met | [`scripts/demo.py`](../scripts/demo.py), `tests/test_demo_cli.py`; `README.md` "Demo CLI" (key-free example with the mock models) | None. Base and tuned get the same zero-shot prompt and every deterministic check runs on both replies; with no `--model` or `--base` it loads the pinned base and adapter through transformers. |
| R2, R6, R10, R11 | Eval harness and base-vs-tuned results on the own and held-out sets, with raw judge transcripts | met | `eval.py`; [`results/base_vs_tuned/heldout/`](../results/base_vs_tuned/heldout/README.md) (tuned 0.48 [0.35, 0.60] / 3.47 vs base zero_shot 0.28 / 2.97) and [`results/base_vs_tuned/eval_own/`](../results/base_vs_tuned/eval_own/README.md) (0.57 / 3.73 vs 0.22 / 2.63); `transcripts.jsonl` in each run; `docs/writeup.md` section 7 | None. No system meets the reliability bar; that is a result, stated in the write-up. |
| R3 | Prompt-ceiling ablation results table | met | [`results/ablation/README.md`](../results/ablation/README.md), `results/ablation/m1/`, `docs/writeup.md` section 4 | None. |
| R7 | One failure mode fixed by a data change (v2) | partial | [`results/base_vs_tuned/heldout/DIAGNOSIS.md`](../results/base_vs_tuned/heldout/DIAGNOSIS.md), `docs/writeup.md` section 8; [`docs/dataset-v2-plan.md`](dataset-v2-plan.md), `data/scenarios/train_v2_additions.jsonl` | Diagnosis done (false abstain, 14 of 60 held-out replies). The v2 scenarios are ready: 105 in-scope cite twins of the v1 abstain scenarios. The teacher spend (210 calls, plus the filter's judge calls) awaits approval; then v2 training on Kaggle and a held-out eval. |
| R8 | Full data-efficiency curve (≥4 nested points) with minimum viable N | partial | [`results/training/curve/README.md`](../results/training/curve/README.md) (4 nested points trained with logs: N = 56, 112, 224, plus the full 448 run in `results/training/v1/`); [`results/curve/heldout/README.md`](../results/curve/heldout/README.md) (deterministic-check curve on the held-out set, all 4 points); `docs/writeup.md` section 9 | The judge-scored curve (Spec-adherence and Robustness) is pending approval of the judge spend, and the minimum viable N is stated after judging. The curve adapters are private, so `scripts/run_curve.py` (not built) could not load them for a third party. |
| R17 | Write-up | partial | [`docs/writeup.md`](writeup.md) | The thesis, ablation, base vs tuned, v2 diagnosis and curve training facts are written. Pending: the v2 result, the curve evals with minimum viable N, and the R29 hand audit with the rerun delta (section 10). |
| R18 | 3–5 minute demo video with a live outside prompt, linked from `README.md` | owner-gated | link in `README.md` (not yet) | Recorded by the owner. The demo CLI (`scripts/demo.py`) can run it live; the hosted notebook link is not published (see R16). |
| R12 | Accept a third-party held-out set (JSONL, JSON array, CSV; three fixtures produce scored output) | met | [`docs/eval-set-format.md`](eval-set-format.md), `src/groundcite/evalset.py`, `tests/test_evalset.py`, `tests/test_eval_fixture_sets.py` (each fixture set runs end to end through `eval.py` and produces `results.json`, `results.md` and `transcripts.jsonl`) | None. |
| R29 | Judge validity: 20 hand-labeled replies, judge agrees on ≥18, reported in the write-up | partial | `scripts/judge_audit.py` (with `--blind-system`, which hides model and strategy behind an audit ID), `tests/test_judge_audit.py`, `tests/test_judge_audit_heldout.py`; related evidence in `results/judge_agreement/` and `docs/writeup.md` section 10 | The tooling is done; the 20-reply hand audit is pending. The agreement study (gpt-5.5 matched a blind label on 17 of 19 disputed replies) is related, not this audit. |
| R1 | Canonical spec | met | [`SPEC.md`](../SPEC.md) | None. |
| R32 | Milestone checklists | met | [`docs/checklist.md`](checklist.md), [`docs/milestone1-checklist.md`](milestone1-checklist.md), this file | None. |
| — | Architecture document | met | [`ARCHITECTURE.md`](../ARCHITECTURE.md) | None. |
