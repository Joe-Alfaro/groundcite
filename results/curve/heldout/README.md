# Data-efficiency curve: held-out set, deterministic checks only

<!-- PENDING-A:approved -->
> **These are deterministic-check metrics only.** The judge-scored Spec-adherence and Robustness
> curve and the minimum viable N ({{min_viable_n}}) are in `{{curve_judged_run_dir}}` and
> `docs/writeup.md` section 9. Nothing on this page meets or misses the reliability bar, which
> is defined on the judge's Spec-adherence and Robustness.
<!-- /PENDING-A -->
<!-- PENDING-A:not-run -->
> **These are deterministic-check metrics only. No judge has scored these replies.** The
> judge-scored curve was not run within the project's budget and timeline, so this table is the
> reported curve and **no minimum viable N is stated**. Nothing on this page meets or misses the
> reliability bar, which is defined on the judge's Spec-adherence and Robustness.
<!-- /PENDING-A -->

Each curve point (N = 56, 112, 224) is a QLoRA adapter trained on a nested subset of dataset v1
([`results/training/curve/`](../../training/curve/README.md)). Its replies to the 60 scenarios in
`data/scenarios/heldout.jsonl` were generated on Kaggle with the same pins as the main base-vs-tuned
run. N = 448 (full v1) and the base rows are the judged main run in
[`results/base_vs_tuned/heldout/`](../../base_vs_tuned/heldout/README.md).

Files:

- `n<N>/inputs/`: `generations.jsonl` and `generation_pins.json` of each curve point, copied
  unchanged from the generation run (run IDs: n56 `20261030T194953Z-3309ef`, n112
  `20261030T200858Z-8cd6a0`, n224 `20261030T204147Z-588841`). Each has 180 rows: the curve
  adapter plus the two base systems, 60 scenarios each.
- [`det_metrics.json`](det_metrics.json): every number below with its row count and interval.

## Input checks

For each of n56, n112 and n224 (no mismatches):

- 180 rows, no generation errors, 60 tuned rows covering all 60 scenario IDs.
- The pins name the right adapter: `joe-alfaro/groundcite-qwen3-1.7b-n56@51ae124`,
  `-n112@c049270` and `-n224@58532b3`. The rows carry the same revisions.
- The eval-set sha256 (`13dae80532ab…c564`) equals the sha256 of `data/scenarios/heldout.jsonl`.
- The base revision (`Qwen/Qwen3-1.7B@70d244cc86cc…ad5e`), chat template hash, decoding
  (greedy, 256 new tokens) and prompt hashes equal the main run's.
- The 120 base replies in each curve run are identical to the main run's, so greedy generation
  reproduced exactly. The base rows below come from the main run.

## Results

Each cell is the rate with a 95% percentile bootstrap interval (2,000 resamples, seed 0, one
seeded RNG per metric and system). n = 60 per system; 41 scenarios have gold mode cite and 19
abstain.

| N | System | HARD pass | SOFT flag | Invented-ID | Mode accuracy | Supports (cite gold) | Abstain precision | Abstain recall |
|---|---|---|---|---|---|---|---|---|
| 0 (base) | base Qwen3-1.7B zero_shot | 0.47 [0.35, 0.60] | 0.20 [0.10, 0.30] | 0.00 [0.00, 0.00] | 0.65 [0.53, 0.77] | 0.68 [0.54, 0.83] | 0.00 [0.00, 0.00] (n=2) | 0.00 [0.00, 0.00] |
| 0 (base) | base Qwen3-1.7B structured | 0.20 [0.10, 0.30] | 0.15 [0.07, 0.25] | 0.40 [0.28, 0.52] | 0.68 [0.57, 0.78] | 0.27 [0.15, 0.41] | 0.50 [0.00, 1.00] (n=4) | 0.11 [0.00, 0.26] |
| 56 | tuned, n56 | 0.35 [0.22, 0.47] | 0.08 [0.02, 0.17] | 0.00 [0.00, 0.00] | 0.70 [0.58, 0.82] | 0.49 [0.34, 0.63] | 1.00 [1.00, 1.00] (n=1) | 0.05 [0.00, 0.16] |
| 112 | tuned, n112 | 0.35 [0.23, 0.47] | 0.15 [0.07, 0.23] | 0.02 [0.00, 0.05] | 0.53 [0.40, 0.65] | 0.24 [0.12, 0.39] | 0.35 [0.19, 0.52] (n=31) | 0.58 [0.37, 0.79] |
| 224 | tuned, n224 | 0.48 [0.35, 0.62] | 0.10 [0.03, 0.18] | 0.02 [0.00, 0.05] | 0.67 [0.55, 0.78] | 0.51 [0.34, 0.66] | 0.47 [0.24, 0.71] (n=17) | 0.42 [0.16, 0.63] |
| 448 | tuned, full v1 | 0.50 [0.37, 0.63] | 0.05 [0.00, 0.12] | 0.02 [0.00, 0.05] | 0.63 [0.50, 0.75] | 0.46 [0.32, 0.61] | 0.44 [0.28, 0.64] (n=25) | 0.58 [0.37, 0.79] |

There is no plot: matplotlib is not a project dependency, so the table is the only view.

Metric definitions (all from `groundcite.checks`, the same checks the judge pipeline runs):

- **HARD pass**: no HARD check failed (citation format and count, cited ID provided and
  supporting, mode matches gold, abstain form, no `<think>` block). Equals 1 − the "HARD-check
  fail" column of the main run.
- **SOFT flag**: the lexical `one_step` or `no_fix` check flagged the reply.
- **Invented-ID**: the reply cites an ID that was not provided.
- **Mode accuracy**: the reply's mode (cite if it cites any bracketed ID, else abstain) equals
  the gold mode. Citing every time would score 41/60 = 0.68 here, so this is not a quality score
  on its own.
- **Supports (cite gold)**: on the 41 cite-gold scenarios, the reply cites only gold supporting
  IDs. An abstain on a cite-gold scenario counts as a miss.
- **Abstain precision / recall**: precision over replies that abstain (n in brackets), recall over
  the 19 abstain-gold scenarios.

## Reading

The curve is not monotone in N, and most intervals overlap, so these 60 scenarios separate few
points. Without the judge, the clear signals are:

- **The abstain behavior appears between N = 56 and 112.** n56 abstains once in 60 (recall 0.05),
  like base. From n112 on, recall is 0.42 to 0.58.
- **n112 over-abstains.** It abstains on 31 of 60 replies, mostly on cite-gold scenarios
  (precision 0.35), which pulls its mode accuracy (0.53) and supports rate (0.24) to the lowest
  of the tuned points. n224 and full v1 abstain less often (17 and 25) and more precisely.
- **HARD pass rises from 0.35 (n56, n112) to 0.48 to 0.50 (n224, full v1),** level with base
  zero_shot (0.47) and well above base structured (0.20). The fine-tuned points almost never
  invent an ID (at most 1 of 60), where base structured invents one in 40% of replies.
- **SOFT flags are at or below base** (0.05 to 0.15 tuned, 0.15 to 0.20 base).

Whether n224 matches full v1 on the judged Spec-adherence is the question that sets the minimum
viable N, and only the judge can answer it.

## Reproduce

No API keys and no model are needed:

```sh
uv run scripts/curve_det.py --scenarios data/scenarios/heldout.jsonl \
  --transcripts results/base_vs_tuned/heldout/20261030T171626Z-e1bd6f/transcripts.jsonl \
  --point 56=results/curve/heldout/n56/inputs/generations.jsonl \
  --point 112=results/curve/heldout/n112/inputs/generations.jsonl \
  --point 224=results/curve/heldout/n224/inputs/generations.jsonl \
  --out results/curve/heldout/det_metrics.json
```

The script runs the checks on each curve point's tuned replies and reads the stored checks of
the main run's transcripts. It also re-runs the checks on those transcripts and records the
mismatch count in `det_metrics.json` (`transcript_checks_mismatches`: 0).
`tests/test_det_metrics.py` regenerates the file and requires it to match exactly.
