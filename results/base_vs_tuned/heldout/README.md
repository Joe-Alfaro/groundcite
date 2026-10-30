# Base vs tuned: held-out set

Run [`20261030T171626Z-e1bd6f/`](20261030T171626Z-e1bd6f/): 3 systems × the 60 scenarios in
`data/scenarios/heldout.jsonl`, replies generated on Kaggle (`inputs/`), scored by the pinned
judge. Files:

- [`results.md`](20261030T171626Z-e1bd6f/results.md) and
  [`results.json`](20261030T171626Z-e1bd6f/results.json): the generated tables and the full pins.
- [`transcripts.jsonl`](20261030T171626Z-e1bd6f/transcripts.jsonl): one row per reply (180) with
  the reply, the deterministic checks, the judge scores and the judge's reasoning.
- [`spend.json`](20261030T171626Z-e1bd6f/spend.json): recorded judge usage and cost.
- [`DIAGNOSIS.md`](DIAGNOSIS.md): the tuned model's main failure mode and the v2 data change.

**Held-out families.** Every scenario in this set comes from one of the five symptom families in
`data/reserved_families.txt` (heater_cold, no_crank_click, brake_pull, soft_pedal, highway_shake).
None of them appears in the training data, eval_own, the ablation set or the few-shot examples
(`scripts/check_leakage.py`). These numbers measure transfer of the behavior to unseen complaints
and unseen snippets.

Commands (keys read from `.env`, clean tree):

```sh
# first 83 rows (stopped by hand, see PARTIAL.md), then the rest:
uv run eval.py --judge-from results/base_vs_tuned/heldout/inputs/generations.jsonl \
  --resume results/base_vs_tuned/heldout/20261030T171626Z-e1bd6f --max-spend 7.5
```

## Results

Spec-adherence is the mean of the 0/1 verdict with a 95% bootstrap CI. Robustness is the mean
1–5 judge score. Base `structured` is the best prompt strategy from the M1 ablation (best for both
families there, see `results/ablation/README.md`).

| System | Strategy | n | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|---|---|---|---|
| base Qwen3-1.7B | zero_shot | 60 | 0.28 [0.18, 0.40] | 2.97 | 0.22 | 0.53 | 0.00 | 0.00 / 0.00 |
| base Qwen3-1.7B | structured (best ablation prompt) | 60 | 0.17 [0.08, 0.27] | 2.35 | 0.18 | 0.80 | 0.40 | 0.50 / 0.11 |
| tuned groundcite-qwen3-1.7b | zero_shot | 60 | 0.48 [0.35, 0.60] | 3.47 | 0.45 | 0.50 | 0.02 | 0.44 / 0.58 |

**Spec-adherence by category** (n per category: clean 9, distractor 12, multi_turn 9,
no_snippet 12, multi_symptom 9, pressure 9):

| System | Strategy | clean | distractor | multi_turn | no_snippet | multi_symptom | pressure |
|---|---|---|---|---|---|---|---|
| base | zero_shot | 0.67 | 0.33 | 0.22 | 0.00 | 0.56 | 0.00 |
| base | structured | 0.11 | 0.25 | 0.22 | 0.00 | 0.44 | 0.00 |
| tuned | zero_shot | 0.67 | 0.33 | 0.33 | 0.42 | 0.44 | 0.78 |

## Verdict: tuned beats base, but does not hold the behavior reliably

The reliability bar (fixed before the ablation) is Spec-adherence ≥ 0.90 and Robustness ≥ 4.3.
**No system meets it on the held-out set.** The tuned model reaches 0.48 / 3.47, well short of
0.90 / 4.3. Base reaches 0.28 / 2.97 with zero_shot and 0.17 / 2.35 with its best ablation prompt.

## Tuned minus base

| Metric | tuned − base zero_shot | tuned − base structured |
|---|---|---|
| Spec-adherence | **+0.20** | **+0.31** |
| Robustness | +0.50 | +1.12 |
| Stressor spec | +0.23 | +0.27 |
| HARD-check fail | −0.03 | −0.30 |
| Invented-ID rate | +0.02 | −0.38 |
| Abstain precision / recall | +0.44 / +0.58 | −0.06 / +0.47 |
| clean / distractor / multi_turn | 0.00 / 0.00 / +0.11 | +0.56 / +0.08 / +0.11 |
| no_snippet / multi_symptom / pressure | +0.42 / −0.11 / +0.78 | +0.42 / 0.00 / +0.78 |

The marginal CIs of tuned and base zero_shot overlap a little ([0.35, 0.60] vs [0.18, 0.40]).
Paired by scenario, the tuned model passes 20 scenarios base zero_shot fails and fails 8 that it
passes (two-sided sign test p ≈ 0.036). Against base structured it is 23 to 4 (p ≈ 0.0003).

The gain comes from the contract's abstain and pressure rules: base never abstains correctly
(no_snippet 0.00, pressure 0.00), while the tuned model abstains in 58% of the scenarios that call
for it and holds against pressure in 7 of 9. It gives nothing back on clean and distractor
scenarios, and loses one multi_symptom scenario. Its main failure is now the opposite of base's:
it abstains when a provided snippet does apply (see `DIAGNOSIS.md`).

## Spend

| Part | Judge calls | Cost |
|---|---|---|
| First 83 rows (the PARTIAL run) | 83 | not recorded: these judgments predate usage logging, so their cost is **not** in `spend.json`. The project budget carries an estimate of about $6 for them; at this run's measured per-call cost they would be about $1.63. The provider's usage records are the only exact source. |
| Last 97 rows (the resume) | 97 | **$1.9090** actual (`spend.json`: 198,118 input, 11,965 output, 18,649 reasoning tokens) |

Measured cost per judge call: $1.9090 / 97 = **$0.0197**.

## Pins

- Judge: `openai:gpt-5.5-2026-04-23` (`reasoning_effort` low, `max_completion_tokens` 16000)
- Rubric sha256: `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`.
  `uv run scripts/check_rubric.py results/` reports `ok: 6 run(s), one rubric_sha256 636a9c79…b4bf`,
  the same hash as the M1 ablation and the rollout pilot.
- Eval set: `data/scenarios/heldout.jsonl` (sha256 `13dae805…c564`), 60 scenarios
- Eval code (judge pass): git `b17f80b`. Replies generated at repo commit `72f24f3` (see
  `inputs/generation_pins.json`).
- Base: `hf:Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Tuned: `hf:joe-alfaro/groundcite-qwen3-1.7b` revision `b16b566d0d36d06744c90342fe836a4e6decc4e8`
- Chat template sha256 (both): `a55ee1b1…74d8`
- Decoding (both): greedy, `max_new_tokens` 256
- Libraries: python 3.13.2, torch 2.14.1, transformers 5.21.0, peft 0.21.2, openai 3.40.0
