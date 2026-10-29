# Long-conversation rollout pilot (H1)

Run `pilot1`: Claude `structured` (the best M1 cell) wrote every assistant turn of the 30 scripted
8–12 turn conversations in `data/scenarios/rollout_pilot.jsonl` (303 turns; 5 held-out families,
6 conversations each). Snippets are added and withdrawn mid-conversation and the user pushes for
a guess or a repair. Each turn was scored against its own live snippets and its any-of gold
(D-068) by the pinned judge `openai:gpt-5.5-2026-04-23`, rubric sha256 `636a9c79…b4bf`, eval
code `f0b7871`. A conversation passes only if every one of its turns is on spec. Files:

- [`pilot1/results.md`](pilot1/results.md): the generated tables and the full pins.
- [`pilot1/results.json`](pilot1/results.json): the same numbers, machine-readable.
- [`pilot1/transcripts.jsonl`](pilot1/transcripts.jsonl): one row per turn (303) with the reply,
  the deterministic checks, the judge scores and the judge's reasoning.
- [`pilot1/run.json`](pilot1/run.json): the resume manifest (scripts hash, conversation IDs,
  model, strategy, prompt hashes, judge, rubric).

Command (keys read from `.env`, clean tree, default concurrency):

```sh
uv run scripts/run_rollouts.py --model anthropic:claude-opus-5-5 --strategy structured \
  --scripts data/scenarios/rollout_pilot.jsonl --run-id pilot1 --out results/rollout_pilot
```

API calls: 303 generations and 303 judge calls, plus 5 first attempts rejected for the
`temperature` parameter and retried without it (4 Anthropic, 1 OpenAI; 611 requests). The run
went through once with no interruption, no resume, and no generation or judge errors.
`scripts/check_rubric.py results/` reports `ok: 2 run(s), one rubric_sha256 636a9c79…b4bf`
across M1 and this pilot.

**Judge note.** This pilot (and M1) was scored by `openai:gpt-5.5-2026-04-23`, which is also the
project's eval judge, a different family from the Anthropic teacher. These numbers compare
directly with later eval results, with no re-judge. Dataset v1 is the one exception in the
pipeline: its filter used Claude Opus (`anthropic:claude-opus-5-5`) while that model was briefly
the judge, a disclosed difference from the gpt-5.5 eval judge.

## Results

| Metric | Value |
|---|---|
| Per-conversation pass rate (95% CI) | **0.80 [0.63, 0.93]** (24 of 30) |
| Per-turn Spec-adherence (95% CI) | 0.94 [0.92, 0.97] (286 of 303) |
| Robustness mean (1–5) | 4.83 |
| HARD-check fail rate | 0.05 |
| Invented-ID rate | 0.00 |
| Abstain precision / recall | 0.92 / 0.99 |
| First failing turn (failed conversations) | turn 0: 5, turn 2: 1 |

The conversation CI is the repo's percentile bootstrap over the 30 conversations (seed 0, 1000
resamples, the same settings `aggregate_scores` uses for the per-turn CI). A Wilson interval
gives [0.63, 0.91]. Either way the interval is wide at n = 30.

**Drift curve: per-turn Spec-adherence by turn index.**

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 25 | 6 | 2 |
| Spec-adherence | 0.83 | 0.87 | 0.93 | 0.93 | 0.97 | 1.00 | 0.93 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |

There is no late-conversation drift. Adherence is lowest at the opening turns and rises from
there. Every failing conversation first fails by turn 2, and most later failures repeat that
first mistake. Turns 9–11
rest on few conversations (25, 6, 2), so the tail says little.

By family (conversations passing out of 6): `brake_pull` 2, `soft_pedal` 5, `no_crank_click` 5,
`heater_cold` 6, `highway_shake` 6.

## Verdict: does not meet the bar

The bar is per-conversation pass rate ≥ 0.90 and Robustness ≥ 4.3, fixed before the run.
Robustness clears it (4.83). The conversation pass rate doesn't: 0.80, with the CI spanning
0.63–0.93. Per-turn Spec-adherence is 0.94, close to M1's single-turn 0.97. Requiring all ~10
turns of a conversation to be right is what pulls the conversation rate down.

## Dominant failure mode: abstaining on a scope condition the user hasn't confirmed

14 of the 17 failing turns (4.6% of all 303 turns), in 5 of the 6 failing conversations, share
one pattern. A snippet's scope carries a condition the user never states outright, and instead
of citing it the model abstains and asks the user to confirm the condition. It then usually
repeats that question every turn until the snippet is withdrawn, so one misjudgment fails
several turns. That is why the failures cluster at turns 0–4.

- **Unstated vehicle attribute (10 turns, 4 conversations, all `brake_pull`).** The slide-pin
  bulletin "Applies to vehicles with floating front calipers", and no user says what calipers
  they have. On `rp-005:t0` the model replies "the caliper slide-pin check depends on a detail
  you haven't given. Does your truck have floating (sliding) front calipers…?" and asks the
  same thing at `rp-005:t1`, `t2`, `t3` and `t6`. `rp-003:t0` and `t1`, `rp-004:t3` ("that
  hasn't been confirmed for your car") and `t4`, and `rp-001:t2` are the same.
- **Symptom read as a precondition (4 turns, `rp-023`).** The in-scope master-cylinder bulletin
  (2010–2020 hatchbacks) gives a pedal-hold test for a pedal that sinks. The model treats
  "sinks" as a condition the user must report first: "None of the provided references covers a
  soft, spongy pedal… When you hold steady pressure on the pedal at a stop, does it keep slowly
  sinking toward the floor?" (`rp-023:t0`, again at `t1`, `t6`, `t7`). That question is the
  bulletin's own test, so this is a real misreading by the model.

**How much of this is the gold.** The caliper case is a scenario-design ambiguity of the same
kind as the validator's `rp-026` watch item below: the gold (D-068) counts the bulletin as in
scope because the family's vehicle has floating calipers, but the conversation never says so.
The model isn't wrong to want that fact. It is, however, inconsistent: across the 20
`brake_pull` turns whose gold includes a slide-pin bulletin, it cited on 10 (all of `rp-006` and
`rp-002`, and three of four turns of `rp-001`) and abstained on 10. Under a stricter gold that
required the user to state the caliper type, those 10 citations would fail instead. Either way the behavior isn't
reliable. As a sensitivity check only (not the verdict): with the 10 caliper turns removed,
`rp-001`, `rp-003` and `rp-005` would pass and the rate would be 27 of 30 = 0.90, exactly at the
bar.

**The other 3 failing turns.** `rp-009:t0` and `t1` cite the battery-voltage check
`[B-21-005]` on a click-no-crank where the user says the dash "stay[s] on and look[s] normal".
The snippet ties a weak battery to dash lights that go dim, so the gold is abstain; this is a
symptom-scope miss in the opposite direction (over-citing). `rp-004:t0` cites the right snippet
but bundles three actions (match tires, set pressures, inspect wear), which fails one-step.

## Validator watch items

- **`rp-026:t0` has an unstated transmission.** The opener ("2013 hatchback, 1.6, 104k") doesn't
  name the transmission. The live runout procedure `SM 2.8.1` applies only to automatic or CVT
  cars, and the gold treats it as out of scope, leaving `SM 2.8.3` as the only valid citation.
  That depends on a fact the user never gave. The model cited `[SM 2.8.3]` and the turn passed
  (Spec 1, Robustness 5), so it didn't affect the result. It is still the same defect class as
  the caliper ambiguity above.
- **The moisture twin is a weak discriminator.** In `soft_pedal`, the brake-fluid moisture test
  and its condition twin (live together in `rp-019`, `rp-022` and `rp-024`) differ only weakly.
  The model cited a moisture snippet 17 times, and every one was the gold. Those passes are
  consistent with reading the scope but don't show it: a model picking either twin by surface
  wording could score the same.
- **All 30 conversations end on an abstain.** Every final turn's gold is abstain, because the
  builder withdraws or exhausts the applicable snippets by the end. The model abstained correctly
  on all 30 final turns. The late-turn 1.00s on the drift curve therefore measure abstain
  behavior only. They don't show that cite-mode accuracy holds late in a long conversation.
  169 of the 303 turns are abstain-gold overall.

## What this means for H1

On this set, long conversations don't make a prompted frontier model drift: per-turn accuracy
holds or improves over turns 0–11. What does break conversation-level reliability is that an
early scope misjudgment gets repeated turn after turn, plus some inconsistency on scope
conditions the user hasn't stated. The pilot misses the bar (0.80 < 0.90), but most of the
failures come from one scenario ambiguity, so the miss is fragile. Before the conversation set is
rebuilt at scale, the builder should state every scope-relevant vehicle attribute (calipers,
transmission) or make the gold abstain when one is unstated, end some conversations on a cite
turn, and replace the weak moisture twin.
