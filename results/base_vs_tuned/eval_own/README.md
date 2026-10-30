# Base vs tuned: eval_own set

Run [`20261030T185720Z-a9795e/`](20261030T185720Z-a9795e/): 3 systems × the 100 scenarios in
`data/scenarios/eval_own.jsonl`, replies generated on Kaggle (`inputs/`), scored by the pinned
judge. Files:

- [`results.md`](20261030T185720Z-a9795e/results.md) and
  [`results.json`](20261030T185720Z-a9795e/results.json): the generated tables and the full pins.
- [`transcripts.jsonl`](20261030T185720Z-a9795e/transcripts.jsonl): one row per reply (300) with
  the reply, the deterministic checks, the judge scores and the judge's reasoning.
- [`spend.json`](20261030T185720Z-a9795e/spend.json): recorded judge usage and cost.

**The held-out set is the headline result** ([`../heldout/README.md`](../heldout/README.md)).
eval_own is the secondary set, and its numbers need the caveat below.

**Memorization caveat.** eval_own uses the same symptom families as the training data (only the
held-out families are reserved). Its near-miss distractors can be told apart by memorizing a
snippet's text rather than reading its applicability line: a text-identity probe solves 0.95 of
eval_own, against 1.00 of the ablation set. A model trained on these families can therefore score
higher on eval_own, especially on `distractor`, than its grounding skill alone would earn. eval_own
shows in-distribution performance; it is not evidence of transfer.

Command (keys read from `.env`, clean tree):

```sh
uv run eval.py --judge-from results/base_vs_tuned/eval_own/inputs/generations.jsonl \
  --out results/base_vs_tuned/eval_own --max-spend 8
```

## Results

Spec-adherence is the mean of the 0/1 verdict with a 95% bootstrap CI. Robustness is the mean
1–5 judge score. Base `structured` is the best prompt strategy from the M1 ablation.

| System | Strategy | n | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|---|---|---|---|
| base Qwen3-1.7B | zero_shot | 100 | 0.22 [0.14, 0.30] | 2.63 | 0.19 | 0.62 | 0.06 | n/a / 0.00 |
| base Qwen3-1.7B | structured (best ablation prompt) | 100 | 0.17 [0.10, 0.25] | 2.33 | 0.16 | 0.80 | 0.38 | 0.27 / 0.10 |
| tuned groundcite-qwen3-1.7b | zero_shot | 100 | 0.57 [0.47, 0.66] | 3.73 | 0.53 | 0.40 | 0.00 | 0.50 / 0.57 |

Abstain precision is n/a for base zero_shot because it never abstains.

**Spec-adherence by category** (n per category: clean 15, distractor 20, multi_turn 15,
no_snippet 20, multi_symptom 15, pressure 15):

| System | Strategy | clean | distractor | multi_turn | no_snippet | multi_symptom | pressure |
|---|---|---|---|---|---|---|---|
| base | zero_shot | 0.40 | 0.20 | 0.33 | 0.00 | 0.33 | 0.13 |
| base | structured | 0.20 | 0.15 | 0.20 | 0.05 | 0.27 | 0.20 |
| tuned | zero_shot | 0.80 | 0.35 | 0.33 | 0.70 | 0.60 | 0.67 |

## Verdict: tuned beats base, but does not hold the behavior reliably

The reliability bar is Spec-adherence ≥ 0.90 and Robustness ≥ 4.3. **No system meets it on
eval_own.** The tuned model reaches 0.57 / 3.73, and base reaches 0.22 / 2.63 (zero_shot) and
0.17 / 2.33 (structured).

## Tuned minus base

| Metric | tuned − base zero_shot | tuned − base structured |
|---|---|---|
| Spec-adherence | **+0.35** | **+0.40** |
| Robustness | +1.10 | +1.40 |
| Stressor spec | +0.34 | +0.36 |
| HARD-check fail | −0.22 | −0.40 |
| Invented-ID rate | −0.06 | −0.38 |
| Abstain precision / recall | n/a / +0.57 | +0.23 / +0.47 |
| clean / distractor / multi_turn | +0.40 / +0.15 / 0.00 | +0.60 / +0.20 / +0.13 |
| no_snippet / multi_symptom / pressure | +0.70 / +0.27 / +0.53 | +0.65 / +0.33 / +0.47 |

The CIs of tuned and both base rows do not overlap. Paired by scenario, the tuned model passes 41
scenarios base zero_shot fails and fails 6 that it passes (two-sided sign test p < 0.0001). Against
base structured it is 49 to 9 (p < 0.0001).

The gain is larger here than on the held-out set (+0.35 vs +0.20 against base zero_shot), as
expected for in-distribution families. Even so, `distractor`, the category the memorization caveat
applies to most, is only 0.35 for the tuned model. So the caveat bounds the result but does not
explain it.

## Spend

| Item | Value |
|---|---|
| Judge calls | 300 (no generation errors; 4 first attempts rejected for `temperature` and retried without it) |
| Tokens | 622,304 input, 38,035 output, 57,290 reasoning |
| Cost | **$5.9713** actual (`spend.json`), $0.0199 per call |

## Pins

- Judge: `openai:gpt-5.5-2026-04-23` (`reasoning_effort` low, `max_completion_tokens` 16000)
- Rubric sha256: `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`.
  `uv run scripts/check_rubric.py results/` reports `ok: 7 run(s), one rubric_sha256 636a9c79…b4bf`,
  the same hash as M1, the rollout pilot and the held-out run.
- Eval set: `data/scenarios/eval_own.jsonl` (sha256 `9955cd63…1ea2`), 100 scenarios
- Eval code (judge pass): git `9936bb4`. Replies generated at repo commit `72f24f3` (see
  `inputs/generation_pins.json`).
- Base: `hf:Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Tuned: `hf:joe-alfaro/groundcite-qwen3-1.7b` revision `b16b566d0d36d06744c90342fe836a4e6decc4e8`
- Chat template sha256 (both): `a55ee1b1…74d8`
- Decoding (both): greedy, `max_new_tokens` 256
- Libraries: python 3.13.2, torch 2.14.1, transformers 5.21.0, peft 0.21.2, openai 3.40.0
