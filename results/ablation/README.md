# Prompt-ceiling ablation (M1, R3)

Run `m1`: 2 model families × 3 prompt strategies × the 40 scenarios in
`data/scenarios/ablation.jsonl` (the same IDs in every cell), scored by the pinned judge
`openai:gpt-5.5-2026-04-23` with rubric sha256 `636a9c79…b4bf`, eval code `3c749fe`. Files:

- [`m1/results.md`](m1/results.md): the generated tables and the full pins.
- [`m1/results.json`](m1/results.json): the same numbers, machine-readable.
- [`m1/transcripts.jsonl`](m1/transcripts.jsonl): one row per reply (240) with the reply, the
  deterministic checks, the judge scores and the judge's reasoning.
- [`m1/run.json`](m1/run.json): the run plan (models, strategies, scenario IDs, eval-set hash).

Command (keys read from `.env`, clean tree, default concurrency):

```sh
uv run scripts/run_ablation.py --model anthropic:claude-opus-5-5 \
  --model openai:gpt-5.5-2026-04-23 --eval-set data/scenarios/ablation.jsonl --run-id m1
```

API calls: 240 generations and 240 judge calls, plus 5 first attempts rejected for the
`temperature` parameter and retried without it (485 requests). No generation or judge errors.
`scripts/check_rubric.py results/` reports one rubric hash across all runs.

**Judge note.** M1 (run `m1`) and the H1 long-conversation pilot were scored by
`openai:gpt-5.5-2026-04-23`. The project judge is now `anthropic:claude-opus-5-5`, so these
numbers can't sit side by side with later (Opus-judged) results as they are. Any side-by-side
comparison uses copies re-judged by Claude Opus from the stored replies with
`scripts/rejudge.py` (no new generations); the gpt-5.5 scores here stay as the historical
cross-family reference.

## Results

Spec-adherence is the mean of the 0/1 verdict with a 95% bootstrap CI. Robustness is the mean
1–5 judge score. **Bold** cells meet the reliability bar (Spec-adherence ≥ 0.90 and Robustness
≥ 4.3).

| Model | Strategy | n | Spec-adherence (95% CI) | Robustness | HARD-check fail | Invented-ID rate | Abstain P / R | Shares judge family |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | zero_shot | 40 | 0.78 [0.65, 0.90] | 4.30 | 0.15 | 0.00 | 0.82 / 0.82 | no |
| claude-opus-5-5 | few_shot | 40 | **0.90 [0.80, 0.97]** | **4.72** | 0.07 | 0.00 | 0.91 / 0.91 | no |
| claude-opus-5-5 | structured | 40 | **0.97 [0.93, 1.00]** | **4.88** | 0.03 | 0.00 | 0.92 / 1.00 | no |
| gpt-5.5-2026-04-23 | zero_shot | 40 | 0.85 [0.72, 0.95] | 4.58 | 0.12 | 0.00 | 0.88 / 0.64 | yes |
| gpt-5.5-2026-04-23 | few_shot | 40 | 0.82 [0.70, 0.93] | 4.47 | 0.17 | 0.00 | 1.00 / 0.45 | yes |
| gpt-5.5-2026-04-23 | structured | 40 | **0.90 [0.80, 0.97]** | **4.70** | 0.10 | 0.00 | 0.90 / 0.82 | yes |

**Spec-adherence by category** (n per category: clean 6, distractor 8, no_snippet 8, pressure 6,
multi_symptom 6, multi_turn 6):

| Model | Strategy | clean | distractor | no_snippet | pressure | multi_symptom | multi_turn |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | zero_shot | 0.50 | 0.88 | 0.62 | 0.83 | 1.00 | 0.83 |
| claude-opus-5-5 | few_shot | 1.00 | 1.00 | 0.88 | 0.83 | 0.67 | 1.00 |
| claude-opus-5-5 | structured | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 |
| gpt-5.5-2026-04-23 | zero_shot | 0.83 | 1.00 | 0.75 | 0.83 | 0.83 | 0.83 |
| gpt-5.5-2026-04-23 | few_shot | 1.00 | 1.00 | 0.62 | 0.83 | 0.83 | 0.67 |
| gpt-5.5-2026-04-23 | structured | 1.00 | 0.88 | 0.75 | 0.83 | 1.00 | 1.00 |

**Best vs second-best strategy per model:**

| Model | Best | Second | Delta (Spec-adherence) |
|---|---|---|---|
| claude-opus-5-5 | structured (0.97) | few_shot (0.90) | +0.07 |
| gpt-5.5-2026-04-23 | structured (0.90) | zero_shot (0.85) | +0.05 |

Both deltas sit inside the overlapping 95% CIs, so the ranking between strategies is suggestive,
not settled, at n = 40. `structured` is best for both models.

**Judge-family flag.** The judge is `openai:gpt-5.5-2026-04-23`, the same model as the OpenAI
subject, so all three GPT cells were graded by their own model and may be scored generously. The
Claude cells were graded across vendors. The verdict below does not depend on the flagged cells:
Claude `structured` meets the bar on its own, with a CI lower bound (0.93) above 0.90.

## Plateau verdict: no plateau

The bar is Spec-adherence ≥ 0.90 and Robustness ≥ 4.3, fixed before the run. **Three cells meet
it**: Claude `structured` (0.97 / 4.88), Claude `few_shot` (0.90 / 4.72, exactly at the bar) and
GPT `structured` (0.90 / 4.70, exactly at the bar, and judge-family flagged). A prompted frontier
model therefore already holds the contract on this set. Per the write-up's plan, this result
calls for hardening the spec and the eval set before any training code is written.

## The failure mode that survives the best strategy

The one failure mode left under `structured` is **misreading a near-miss twin's applicability
scope**: given two snippets with the same procedure that differ only in model year or operating
condition, the model either cites the twin that excludes this vehicle or abstains even though the
in-scope twin applies. It accounts for all 5 failures across the two `structured` cells (5 of 80
replies, 6%; Claude 1 of 40, GPT 4 of 40), and no other failure type survives. On `eo-0056` the
user's vehicle is a 2010, and GPT cites the hot-restart snippet scoped to "model years
2011-2021" ("crank it with the accelerator pedal held to the floor … [SM-1608]"). On `eo-0071`
the user feels the pulsation "through the steering wheel rather than the seat", and GPT cites the
runout bulletin scoped to pulsation "felt through the front seatback" ([TSB-0181]). The opposite
direction appears on `eo-0039`, where GPT abstains ("None of the provided references clearly
applies unless we know the front suspension type"). The gold strut-mount bulletin matches the
complaint (a knock over bumps, quiet at full lock), while its twin is scoped to clunks "only while
you are turning slowly". The user never states the suspension type, so this abstain is the most
defensible of the five. `eo-0059`, a pressure scenario, fails in all six cells. There Claude abstains
because "the condenser fan references only cover warm air at idle", and GPT cites the idle-only
near-miss [B-22-100]. Part of that failure is a data defect: the gold snippet `SM-9330` still
has the title "Warm AC at idle", so its scope is ambiguous against a complaint of warm air "all
the time". If `eo-0059` is excluded, the surviving rate is Claude 0 of 39 and GPT 3 of 39.

## Known limitations of the sets (D-064)

- **`eval_own` near-miss distractors can be solved by memorizing text identity.** A surface probe
  that only knows which twin text tends to be cited scores 0.95 on `eval_own` distractors and
  1.00 on `ablation` distractors. `heldout` uses unseen snippet texts and is immune. That doesn't
  affect this ablation, because the prompted models never saw the training texts. It does affect
  a tuned model, so tuned-model claims lead with `heldout`, and `eval_own` is reported with this
  caveat.
- **Two weak condition twins.** In the `lean_idle` family, complaint variant 3 and, in
  `soft_pedal`, the moisture-condition twin separate gold from near-miss only weakly. Results on
  those scenarios should be read with that in mind.
- **A third weak twin found in this run.** The `ac_warm` gold `SM-9330` keeps the "at idle" title
  (see `eo-0059` above). It has not been fixed here. The sets are frozen per D-064.
