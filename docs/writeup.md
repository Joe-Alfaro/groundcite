# Write-up (draft)

> **Status: draft.** The status sections below summarize where the project stands at Milestone 1
> and since. Sections 1, 2, 3, 5 and 6 (method) describe what is built or designed in this
> repository. Section 4 reports the real prompt-ceiling ablation, section 4a the long-conversation
> pilot, section 6 dataset v1, section 7 the base-vs-tuned evals, section 8 the v2 diagnosis,
> section 9 the curve training runs (their evals are pending) and section 10 the judge agreement
> study. Every other number that needs a
> real run is marked **Pending**, along with what will go there. Nothing in the pending sections
> should be read as a result.

Requirements: [`docs/requirements/small-model-finetune.md`](requirements/small-model-finetune.md).
Architecture: [`ARCHITECTURE.md`](../ARCHITECTURE.md). Status of every item:
[`docs/checklist.md`](checklist.md); Milestone 2 deliverables: [`docs/final-checklist.md`](final-checklist.md).

Public artifacts: dataset v1 at
[`joe-alfaro/groundcite-diagnostic-v1`](https://huggingface.co/datasets/joe-alfaro/groundcite-diagnostic-v1),
commit `ccb75f757844360f10fc951d7c4f3db1a57fcb36`; the tuned adapter at
[`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b),
commit `b16b566d0d36d06744c90342fe836a4e6decc4e8`.

## Status update (Friday, October 30)

- **Base vs tuned is measured (section 7).** On the held-out set (unseen symptom families), the
  tuned model scores 0.48 [0.35, 0.60] Spec-adherence and 3.47 Robustness, against 0.28 / 2.97
  for base with the zero-shot prompt and 0.17 / 2.35 for base with `structured`. Paired by
  scenario, it wins 20 and loses 8 against base zero_shot (sign test p ≈ 0.036). On `eval_own`
  (in-distribution, with a memorization caveat) it scores 0.57 / 3.73 against 0.22 / 2.63.
  **No system meets the reliability bar on either set.**
- **The v2 diagnosis is done; v2 is designed and its generation awaits approval (section 8).**
  The tuned model's most frequent failure is a false abstain (14 of 60 held-out replies), usually
  justified by a scope exclusion that is false. The data change is in-scope twins of the v1
  abstain scenarios, with no config change.
- **Curve points trained; evals pending (section 9).** Nested subsets of v1 at N = 56, 112 and
  224 are trained with the v1 config and seed (final step losses 1.3040 / 1.0274 / 0.6386; the
  full 448 run ended at 0.5064). Their held-out evals are not scored yet, so the minimum viable N
  is not stated.
- **Judge spend:** $7.88 logged plus about $1.6 unlogged (the first 83 held-out judgments
  predate usage logging), about $9.5 of the $20 cap. With the conservative $6 estimate for the
  unlogged calls, about $13.9.

## Status at Milestone 1

Thursday, October 29. Every item, with its artifact path, is in
[`docs/checklist.md`](checklist.md).

**The headline comparison.** The claim this project tests is that fine-tuning makes Qwen3-1.7B
hold the contract better than the same base model does with the best prompt we found. Tuned 1.7B
against prompted base 1.7B is the headline. The prompted frontier models are reported separately
as a ceiling. That ceiling is high: the ablation found no plateau, and this write-up says so
plainly instead of treating the frontier numbers as something the small model has to beat.

**Done.**

- **Spec and rubric.** `SPEC.md` gives the contract, the citation format, the reliability bar
  (Spec-adherence ≥ 0.90 and Robustness ≥ 4.3) and the P1–P8 rubric. The judge rubric, the
  teacher prompt and the filter all load it by path.
- **Eval harness.** One command scores base and tuned rows on any eval set with the pinned judge,
  writes results, transcripts and pins, and has mock backends and `--dry-run` for key-free runs.
  It can also generate on a GPU with no keys and judge the stored replies elsewhere. It was merged
  before any training code.
- **Prompt-ceiling ablation (section 4).** Two frontier models from different vendors, three
  prompt strategies, 40 scenarios each. Three cells meet the bar; the best is Claude Opus with the
  `structured` prompt at 0.97 [0.93, 1.00] / 4.88. Result: **no plateau**.
- **Spec hardening pilot (section 4a).** Because of that result, the spec was hardened for long
  conversations: 30 scripted 8–12 turn conversations with snippets added and withdrawn and pressure
  to guess. The best M1 cell passed 0.80 [0.63, 0.93] of conversations and 0.94 of turns, with no
  late-turn drift. That misses the bar at the conversation level, but not clearly enough to justify
  rebuilding the sets around it, so the project went on to training.
- **Judge agreement study (section 10).** Three cheaper judges were measured against the pinned
  judge with a pre-registered bar and blind adjudication. All three failed, and the pinned judge
  (`gpt-5.5`) stays.
- **Dataset v1 (section 6).** 448 filtered examples from 960 teacher candidates (46.7% kept), 28.1%
  abstain, with a manifest, a filter report, a leakage report and a quality note from reading the
  data. It is public at [`joe-alfaro/groundcite-diagnostic-v1`](https://huggingface.co/datasets/joe-alfaro/groundcite-diagnostic-v1),
  commit `ccb75f757844360f10fc951d7c4f3db1a57fcb36`.
- **Training and smoke code.** The QLoRA script, its frozen config, the Kaggle training and eval
  notebooks and the smoke loop are written and tested. The local smoke mode is a CPU-only check on
  a tiny model; the real-model smoke runs on Kaggle.
- **First QLoRA run and the draft checkpoint.** Qwen3-1.7B was trained on dataset v1 with frozen
  config v1 on a Kaggle T4 (code commit `72f24f33`): 168 steps over 3 epochs in 839.7 s, final
  step loss 0.5064, mean train loss 0.9019, no errors. This is the first time the real-model
  training path ran end to end. The adapter is public at
  [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b), commit
  `b16b566d0d36d06744c90342fe836a4e6decc4e8`, and that model page is the deployed artifact for
  this milestone. The training log is in `results/training/v1/`.

**Behind, and why.**

- **No base-vs-tuned numbers yet.** The tuned adapter exists but has not been evaluated. Three
  things used up the time that was planned for training and eval before this milestone:
  1. *The prompt-ceiling result needed spec hardening first.* The plan (section 1) was to harden
     the spec, not train, if a prompted frontier model already met the bar. It did, so the
     long-conversation pilot came first.
  2. *The judge provider changed twice.* When the pinned judge's API credit ran out, the judge was
     switched to Claude Opus (used for the dataset v1 filter), then a cross-family replacement was
     tried. An agreement study showed none of the candidates was good enough, so the project
     returned to the original judge. Judged work paused while this was settled. The filter judge
     differing from the eval judge is disclosed in section 11.
  3. *Local inference limits moved every model run to Kaggle.* A local smoke run on the shared
     development machine took over 6 minutes and about 11 GB of its 16 GB of memory. Since then,
     only a capped CPU-only check runs locally. Training, the real-model smoke run and base and
     tuned generation all run in Kaggle notebooks, which the owner runs.
- **The v2 data fix needs eval evidence first.** The v1 → v2 change has to target a failure mode
  measured in tuned-model transcripts. Weaknesses noted while reading v1 (`data/datasets/v1/NOTES.md`)
  are candidates, not a diagnosis.
- **The data-efficiency curve points** need their own Kaggle training runs, now that v1 is
  trained.

**Next steps (targets, Central Time).**

1. Generate base and tuned replies on `heldout` first, then `eval_own`, in the Kaggle eval
   notebook, loading the adapter at Hub commit `b16b566d`, and judge them locally with `gpt-5.5`.
   Target: Sat Oct 31 at the latest. This fills section 7 and the hand audit in section 10.
2. Diagnose one v1 failure mode from those transcripts, build v2 by changing data only, train it
   with the same config and re-evaluate. Target: Sat Oct 31 to Sun Nov 1.
3. Train at least two nested data-efficiency points, then the rest of the curve for Milestone 2
   (Sun Nov 1, noon).

## 1. Behavior thesis

The project teaches Qwen3-1.7B one narrow contract: given a vehicle complaint and a few ID-tagged
reference snippets, reply with exactly one next diagnostic step and cite exactly one provided
snippet that supports it. If no snippet supports a step, say so, cite nothing and ask one question
for one missing detail. Never name a repair, and keep all of this under user pressure and across
turns.

**Why prompting is likely to fall short.** Each rule is easy to state, but every reply has to
satisfy all of them at once, and each scenario category pulls against a different rule:

- *One step.* Models trained to be helpful tend to give a short checklist ("check X, then Y").
  The contract counts any second action, including "if that's fine, check…", as a failure.
- *No repair, not even a conditional one.* "If it reads low, replace the pump" sounds responsible
  but fails. So does naming the likely cause as the answer. When the user says "just tell me what
  to replace", the default helpful move is the one the contract forbids.
- *Grounding against near-misses.* A distractor snippet covers the same system or code but a
  different engine, model year, transmission or operating condition. Citing it needs no invented
  ID, so the citation looks fine, but it is wrong. Telling the two apart means reading the
  applicability scope, not matching topic words.
- *Abstaining.* When every snippet is only topically close, the right reply admits that none
  applies and asks one question. A model that is rewarded for answering tends to cite the closest
  snippet anyway.
- *Holding over turns.* In multi-turn scenarios the earlier turns report results, add snippets or
  push back. Drift from the contract on a late turn is a failure even if the first reply was fine.

A system prompt can list these rules, but it has to compete with the model's general habits on
every reply. Whether that is enough is an empirical question, and the prompt-ceiling ablation
(section 4) is there to answer it before any training. If a prompted frontier model already meets
the reliability bar, the plan is to harden the spec, not to train.

**Why it should be learnable from data.** The contract is narrow and its output shape is
consistent: one sentence or two, one bracketed ID or a single question. Every rule is checkable
per reply, most of them by deterministic code. That makes it possible to generate many candidate
replies, throw away every one that breaks a rule, and train only on the rest. A fine-tune then
shifts the model's default toward the contract, instead of asking a prompt to override the
default each time. The claim this project tests is narrow: tuned beats base on this contract, by
a measured margin, on scenarios from families the training data never saw.

## 2. Spec and rubric

The canonical spec is one paragraph in [`SPEC.md`](../SPEC.md). The judge rubric, the teacher
prompt and the quality filter all load it by path, and a test checks that the requirements file
quotes it exactly.

**Citation format.** A citation is the exact ID of a provided snippet in square brackets, for
example `[TSB-0417]` or `[SM 6.2.1]`. IDs are opaque strings drawn from at least 3 synthetic
schemes and re-drawn for every scenario. A cited ID that was not provided is *invented*. A provided
ID that appears without brackets is a format violation.

**Pass/fail rubric (P1–P8).** A reply passes only if every line that applies holds:

| # | Rule | Applies to |
| --- | --- | --- |
| P1 | Right mode: cite when some snippet supports a sensible next step, abstain when none does. | all |
| P2 | Exactly one `[ID]`, and no other snippet IDs anywhere in the reply. | cite |
| P3 | The cited ID was provided in this conversation. | cite |
| P4 | The cited snippet supports the step given. On-topic but not supporting fails. | cite |
| P5 | Exactly one next action on one component or system. How to do it and what to expect is fine; a list or "then…" fails. | cite |
| P6 | No repair or part replacement, even conditionally, and no likely cause stated as the answer. | all |
| P7 | Clean abstain: says no reference applies, cites nothing, asks exactly one question for one specific missing detail. | abstain |
| P8 | P1–P7 still hold when the user demands a fix and on later turns. | all |

The requirements file has four worked examples (one pass, two fails, one abstain under pressure).

**Deterministic checks.** Every reply also runs through pure-function checks
(`src/groundcite/checks.py`). They come in two classes:

- **HARD (structural):** `citation_format`, `citation_count`, `cited_id_provided`,
  `cited_id_supports`, `mode_matches_gold`, `abstain_form`, `no_think`. Any HARD failure forces
  Spec-adherence to 0 and caps Robustness at 2, whatever the judge says. The forced value is
  recorded as forced.
- **SOFT (lexical):** `one_step` and `no_fix`, based on versioned word lists. Lexical checks can
  misfire, so they are reported in the table and passed to the judge as flags, but they don't
  override it.

**Judge.** A pinned, dated judge snapshot scores every reply on **Spec-adherence** (0/1, the
P1–P8 verdict) and **Robustness** (1–5, how fully the reply holds the contract against the
scenario's stressor), each with written reasoning. The judge sees every snippet with its ID, the
conversation, the reply, the check results and the gold label when there is one. The judge comes
from a different vendor than the teacher. It is, however, the same model as one ablation subject,
and the ablation table flags those cells.

**Reliability bar.** A system *holds the behavior reliably* on an eval set when Spec-adherence is
at least **0.90** and mean Robustness is at least **4.3**. The bar was fixed before the ablation
runs. The plateau check, the minimum viable N and this write-up all use it.

## 3. Scenario design

All snippets, vehicles and complaints are synthetic and hand-written. They are not real service
information. Vehicles are described generically, with no manufacturer names. Coverage tables are
generated into [`docs/scenario-coverage.md`](scenario-coverage.md).

**Scope.** Light-duty passenger vehicles, symptom to first diagnostic check, 2–5 snippets per
scenario, 8 vehicle systems and 16 symptom families. Snippets come in two styles: bulletin-like
and service-manual-like.

**Categories.** Every set meets the same quotas (±5 points), and at least 30% of every set is
`no_snippet` or `pressure`:

| Category | Share | What makes it hard |
| --- | --- | --- |
| `clean` | 15% | One snippet clearly supports one step. The baseline. |
| `distractor` | 20% | A near-miss snippet for the same system or code but the wrong engine, year or condition. |
| `no_snippet` | 20% | Nothing applies, though some snippets are topically close. Correct reply abstains. |
| `pressure` | 15% | The user demands a fix or a part name, or asks to skip diagnosis. |
| `multi_symptom` | 15% | Several complaints that tempt several steps or citations. |
| `multi_turn` | 15% | 3–6 earlier turns. Only the final reply is graded. |

The committed sets are `ablation` (40 scenarios), `eval_own` (100, a superset of `ablation`) and
`heldout` (60). They were generated by a seeded generator, so the same seed gives the same bytes,
and they were frozen before any training data exists.

**Surface cues must not predict the gold.** A distractor scenario only tests the behavior if the
correct snippet can't be picked out by something other than its applicability scope. Early
versions of the library failed this. Almost every gold citation was a manual-style snippet and
almost every near-miss was a bulletin, so "never cite a bulletin" would have solved most
distractors without reading anything. Rebalancing that one feature moved the cue somewhere else:
length, then the presence of a model-year range, then punctuation and units. A model that learns
such a cue would score well on our sets and fail on a third-party set where the cue doesn't hold.
It would also inflate both the ablation and the base-vs-tuned numbers.

Two controls address this:

- **Shortcut audit.** A test suite measures, on distractor scenarios in every set, how well
  simple surface heuristics pick the gold: snippet kind, ID scheme, a correction line, an
  "applies to all" opener, year presence, longest or shortest snippet, and length terciles. Each
  heuristic must stay at or below 0.60 accuracy in every set, and each binary feature's
  gold-rate gap must stay small. A ridge probe on all surface features together (lengths, counts
  of commas, digits and units, sentences, kind, ID scheme, opener, position, the stated year
  range and its distance from the catalog's median model year, and whether a snippet has a
  near-identical partner) is held to the same 0.60. It is scored leave-one-scenario-out within
  each set, and fit on the training pool and scored on each eval set.
- **Twin near-misses.** Rather than tuning features one at a time, each near-miss is built as a
  twin of a real gold-capable snippet from the same family. Only the applicability scope changes
  (engine, model-year range, transmission, body or operating condition). The procedure text, its
  length, punctuation and numbers stay the same. Gold and near-miss then match on every surface
  feature by construction, and the only thing that separates them is the skill the spec is about.
  Off-family snippets are drawn from other families' gold-capable snippets.

Twins still left priors that needed a second pass. Year twins sat at the ends of the catalog's
year span while the gold covered the middle, so "of two near-identical snippets, cite the one
covering the median year" beat chance without reading the user's year. Now every year pair is
split at the median with the gold on either side equally often. Two near-identical snippets
almost always meant cite, so abstain scenarios now also carry a pair with both members out of
scope. Condition twins kept the original title, which stated the very condition the edit
removed, so titles now change with the scope. Transmission-scoped twins depended on a
transmission the user never stated, so every vehicle description now states it.

A related check guards gold integrity: no gold snippet may contradict any complaint variant of its
family.

**Held-out families.** About 30% of the symptom families (5 of 16: `brake_pull`, `soft_pedal`,
`no_crank_click`, `heater_cold`, `highway_shake`) are reserved. They appear only in `heldout`, and
held-out scenarios use only snippets from reserved families. `eval_own` shares families with
training but not scenarios, so it measures in-distribution fit. `heldout` is the closer estimate
of how the model does on an unseen third-party set.

**Leakage controls.** `scripts/check_leakage.py` runs in the quality gate and fails the build on:

- a shared scenario ID;
- an identical normalized final user message (SHA-256);
- a final user message pair with word 5-gram Jaccard ≥ 0.6;
- any held-out snippet text that also appears in training;
- any training or few-shot example from a reserved family;
- any few-shot example that appears in an eval set.

Snippet IDs are re-drawn for every scenario, so the same snippet carries different IDs in
different scenarios and a model can't memorize an ID-to-symptom map. Snippet order is shuffled.
Each split uses its own seed range. The training pool is also written with its own phrase bank for
pressure and multi-turn turns, so it doesn't reuse eval phrasing templates. The report is
committed as `data/scenarios/leakage_report.md`.

## 4. Prompt-ceiling ablation

**Result: no plateau.** Run `m1` (eval code `3c749fe`) put both models through all three
strategies on the same 40 scenarios. Three cells meet the reliability bar, so a prompted frontier
model already holds the contract on this set. Full tables, pins and the 240 judged transcripts
are in [`results/ablation/`](../results/ablation/README.md).

**Design.** Two frontier models from different vendors × three strategies × the same 40 scenario
IDs from `data/scenarios/ablation.jsonl` (n ≥ 30 per cell):

- `zero_shot`: the spec as the system prompt.
- `few_shot`: the spec plus 4 fixed examples (clean, abstain, pressure, distractor) drawn from
  scenarios outside every eval set and covered by the leakage check.
- `structured`: the spec as explicit numbered rules, a private pre-answer checklist, and an
  instruction to output only the final reply.

Errors and refusals count as failures. They are not dropped. Every cell is scored by the same
pinned judge and rubric.

**Results.** Judge `openai:gpt-5.5-2026-04-23`. **Bold** cells meet the bar (Spec-adherence ≥
0.90 and Robustness ≥ 4.3).

| Model | Strategy | n | Spec-adherence (95% CI) | Robustness | HARD-check fail | Invented-ID rate | Abstain P / R | Shares judge family |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | `zero_shot` | 40 | 0.78 [0.65, 0.90] | 4.30 | 0.15 | 0.00 | 0.82 / 0.82 | no |
| claude-opus-5-5 | `few_shot` | 40 | **0.90 [0.80, 0.97]** | **4.72** | 0.07 | 0.00 | 0.91 / 0.91 | no |
| claude-opus-5-5 | `structured` | 40 | **0.97 [0.93, 1.00]** | **4.88** | 0.03 | 0.00 | 0.92 / 1.00 | no |
| gpt-5.5-2026-04-23 | `zero_shot` | 40 | 0.85 [0.72, 0.95] | 4.58 | 0.12 | 0.00 | 0.88 / 0.64 | yes |
| gpt-5.5-2026-04-23 | `few_shot` | 40 | 0.82 [0.70, 0.93] | 4.47 | 0.17 | 0.00 | 1.00 / 0.45 | yes |
| gpt-5.5-2026-04-23 | `structured` | 40 | **0.90 [0.80, 0.97]** | **4.70** | 0.10 | 0.00 | 0.90 / 0.82 | yes |

Spec-adherence by category (n: clean 6, distractor 8, no_snippet 8, pressure 6, multi_symptom 6,
multi_turn 6):

| Model | Strategy | clean | distractor | no_snippet | pressure | multi_symptom | multi_turn |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | `zero_shot` | 0.50 | 0.88 | 0.62 | 0.83 | 1.00 | 0.83 |
| claude-opus-5-5 | `few_shot` | 1.00 | 1.00 | 0.88 | 0.83 | 0.67 | 1.00 |
| claude-opus-5-5 | `structured` | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 |
| gpt-5.5-2026-04-23 | `zero_shot` | 0.83 | 1.00 | 0.75 | 0.83 | 0.83 | 0.83 |
| gpt-5.5-2026-04-23 | `few_shot` | 1.00 | 1.00 | 0.62 | 0.83 | 0.83 | 0.67 |
| gpt-5.5-2026-04-23 | `structured` | 1.00 | 0.88 | 0.75 | 0.83 | 1.00 | 1.00 |

**Best vs second-best strategy.** `structured` is best for both models: Claude +0.07 over
`few_shot` and GPT +0.05 over `zero_shot`. Both deltas fall inside overlapping 95% CIs at n = 40.

**Judge family.** The judge is the same model as the GPT subject, so the three GPT cells are
flagged. The verdict doesn't rest on them: Claude `structured` meets the bar on its own, graded
across vendors, with a CI lower bound of 0.93.

**Plateau verdict: no plateau.** Claude `structured` (0.97 / 4.88), Claude `few_shot` (0.90 /
4.72) and GPT `structured` (0.90 / 4.70) meet the bar. The last two sit exactly at 0.90. Under the
plan in section 1, the next step is to harden the spec and the eval set, not to write training
code.

**The failure mode that survives the best strategy.** The one failure mode left under
`structured` is misreading a near-miss twin's applicability scope. Given two snippets with the
same procedure that differ only in model year or operating condition, the model either cites the
twin that excludes this vehicle or abstains even though the in-scope twin applies. It accounts for
all 5 `structured` failures (5 of 80 replies, 6%; Claude 1 of 40, GPT 4 of 40). On `eo-0056` the
vehicle is a 2010, and GPT cites the hot-restart snippet scoped to "model years 2011-2021"
([SM-1608]). On `eo-0071` the pulsation is felt "through the steering wheel rather than the
seat", and GPT cites the runout bulletin scoped to pulsation "felt through the front seatback"
([TSB-0181]). On `eo-0039` GPT abstains because it doesn't know "the front suspension type",
although the gold strut-mount bulletin matches the complaint. `eo-0059` fails in all six cells. On
it Claude abstains because "the condenser fan references only cover warm air at idle", and GPT
cites the idle-only near-miss. Part of that is a data defect: the gold snippet's title still says
"at idle". If `eo-0059` is excluded, the surviving rate is Claude 0 of 39 and GPT 3 of 39.

## 4a. Hardening: long-conversation pilot (H1)

**Result: the bar is not met at the conversation level, and there is no drift.** M1 found no
plateau, so the spec was hardened along the axis the thesis names last, holding over turns. Run
`pilot1` (eval code `f0b7871`) had the best M1 cell, Claude `structured`, write every assistant
turn of 30 scripted 8–12 turn conversations (303 turns, the 5 held-out families). Snippets are
added and withdrawn mid-conversation and the user pushes for a guess or a repair. Each turn is
scored against its live snippets and an any-of gold: any in-scope snippet whose step hasn't been
done yet is a valid citation. A conversation passes only if every turn is on spec, and the bar
uses that conversation pass rate in place of per-turn Spec-adherence. Full tables, the 303 judged
turns and the watch items are in [`results/rollout_pilot/`](../results/rollout_pilot/README.md).

| Per-conversation pass (95% CI) | Per-turn Spec-adherence (95% CI) | Robustness | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|
| 0.80 [0.63, 0.93] (24/30) | 0.94 [0.92, 0.97] | 4.83 | 0.00 | 0.92 / 0.99 |

Per-turn Spec-adherence by turn index (n = 30 through turn 8, then 25, 6, 2):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.83 | 0.87 | 0.93 | 0.93 | 0.97 | 1.00 | 0.93 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |

All six failing conversations first fail at turn 0 (5) or turn 2 (1). Robustness clears the bar
(4.83 ≥ 4.3), but the conversation pass rate (0.80) doesn't reach 0.90. Per-turn accuracy is near
M1's single-turn 0.97. Requiring every turn of a ~10-turn conversation to be right is what costs
the 0.17.

**Dominant failure mode: abstaining on a scope condition the user hasn't confirmed.** 14 of the
17 failing turns (5 of 6 failing conversations) are abstains where a snippet's scope carries a
condition the user never states, and the model asks the user to confirm it instead of citing.
It then repeats the question until the snippet is withdrawn. Ten of those turns are the
`brake_pull` slide-pin bulletin ("Applies to vehicles with floating front calipers"), for example
`rp-005:t0`, `rp-003:t0` and `rp-004:t3`. The other four are `rp-023`, where the model reads the
bulletin's own pedal-sink test as a precondition the user must report (`rp-023:t0`). The caliper
case is partly a scenario defect, because no user ever states the caliper type. But the model is
also inconsistent on it, citing on 10 of the 20 slide-pin-gold turns and abstaining on the other
10. As a sensitivity check only, dropping those 10 turns would give 27/30 = 0.90.

**Caveats.** Every conversation ends on an abstain-gold turn, so the 1.00s at turns 8–11 show
abstain behavior holding late, not cite accuracy. The `soft_pedal` moisture twin discriminates
weakly, so its 17 correct citations are weak evidence of scope reading. `rp-026:t0` has an
unstated transmission, though that turn passed. n = 30 conversations gives a wide CI.

**What it means.** Long conversations don't erode the contract for a prompted frontier model on
this set. What breaks conversation-level reliability is an early scope misjudgment that gets
repeated turn after turn. The next set should state every scope-relevant vehicle attribute (or
gold it as abstain when unstated), end some conversations on a cite turn and replace the weak
moisture twin before rollouts are rebuilt at scale.

## 5. Teacher vs ablation

The teacher that writes the training data is one of the two ablation subjects. If the ablation
shows a plateau, that teacher fails the contract on some share of prompted replies. That doesn't
stop it from producing on-spec training data, for two reasons.

1. **Privileged inputs.** The teacher gets more than the student will. Its prompt carries the gold
   mode (cite or abstain) and, for cite cases, the supporting snippet ID. The hardest part of the
   task for a prompted model, deciding which snippet truly applies or whether any does, is answered
   for it. What remains is writing one compliant reply, which is a much easier job. The hint
   appears only in the teacher prompt. The student's training input is the normal rendered
   conversation, the same one used at eval time, and the training target is the teacher's reply
   alone.
2. **A hard filter.** The teacher writes k ≥ 2 candidates per scenario, and a candidate is kept
   only if it passes every stage: no generation error, no wording that leaks the hint ("gold label",
   "the hint", "supporting IDs", "as instructed", "correct citation" and similar; "a hint of
   coolant smell" and "the gold-colored connector" pass), a clean ending, agreement with the gold mode and
   supporting ID, every HARD check and the SOFT checks, a judge score of Spec-adherence 1 with
   Robustness ≥ 4, and no near-duplicate of a kept example. Even a teacher that is right only some of
   the time, filtered this way, gives a dataset with almost no off-spec targets. The cost shows up
   as a lower kept rate, not as noise in the data.

The ablation measures what a model does *without* the answer and *without* a filter. Distillation
uses it *with* both. This section was written for the plateau case. The ablation found no
plateau: the teacher meets the bar with the `structured` prompt even without the hint (section
4), so it is a stronger teacher than this argument needed.

A risk remains. If the filter's checks and the judge share a blind spot, off-spec replies with that
blind spot will pass. The judge validity audit (section 10) is the check on that.

## 6. Data

**Pipeline (method).**

1. A training pool of scenarios comes from the same generator and snippet library as the eval
   sets, with its own seed, its own phrase bank for pressure and multi-turn turns, no reserved
   families, and an abstain share of 20–30%.
2. `scripts/distill.py` asks the teacher for k candidates per scenario, each with the spec and the
   privileged hint. Errors are recorded as lines, never dropped.
3. `scripts/build_dataset.py` runs the filter in the order listed in section 5. The first stage a
   candidate fails is its rejection reason. At most one candidate is kept per scenario: the one
   with the highest Robustness, ties going to the lowest candidate index.
4. Kept examples are written as chat records rendered by the same `render_messages()` used at eval
   time, with a manifest (counts by category, ID-scheme mix, seeds, teacher, filter version,
   SHA-256), a rejections file and a filter report.
5. The leakage check runs against every committed dataset in the quality gate.

The pipeline is merged to `main` and has been run once, to build dataset v1.

**Dataset v1** (`data/datasets/v1/`; public on the HF Hub at
[`joe-alfaro/groundcite-diagnostic-v1`](https://huggingface.co/datasets/joe-alfaro/groundcite-diagnostic-v1), commit
`ccb75f757844360f10fc951d7c4f3db1a57fcb36`). 960 teacher candidates (k = 2 over 480 training scenarios)
gave 448 kept examples (46.7%), one per scenario for 448 of the 480 scenarios. That meets the R6
floor of 300. The abstain share is 28.1%. By category: clean 67, distractor 86, multi_symptom 67,
multi_turn 65, no_snippet 96, pressure 67. Rejections: 102 at the SOFT checks and 410 not selected
(the other candidate for a scenario that already had one kept). No candidate failed a HARD check
or the judge, so the judge added no signal beyond the deterministic filter. All 9 `soft_check`
rejections that were read by hand were false positives of the lexical screen, so it is over-strict. The teacher and
the filter judge were both Claude Opus (section 11). `NOTES.md` lists what a read of the data
found, including weaknesses that are candidates for v2.

## 7. Base vs tuned

One QLoRA run of Qwen3-1.7B on v1 (adapter
[`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) at
Hub commit `b16b566d`), against base Qwen3-1.7B (revision `70d244cc`) with the zero-shot prompt
and with `structured`, the best strategy in the ablation. Replies were generated greedily on
Kaggle (code commit `72f24f3`) and judged locally with the pinned judge `gpt-5.5` and rubric
`636a9c79…b4bf`, the same hash as the ablation and the pilot. Full tables, transcripts and pins:
[`results/base_vs_tuned/heldout/`](../results/base_vs_tuned/heldout/README.md) and
[`results/base_vs_tuned/eval_own/`](../results/base_vs_tuned/eval_own/README.md).

**The held-out set is the headline.** Its five symptom families are reserved: none of them
appears in training, `eval_own`, the ablation set or the few-shot examples. `eval_own` shares
families with training, and its near-miss distractors can be solved by memorizing snippet text (a
text-identity probe solves 0.95 of it), so it shows in-distribution performance and may be
inflated, especially on `distractor`.

**Held-out set** (60 scenarios):

| System | Strategy | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|---|---|---|
| base | zero_shot | 0.28 [0.18, 0.40] | 2.97 | 0.22 | 0.53 | 0.00 | 0.00 / 0.00 |
| base | structured | 0.17 [0.08, 0.27] | 2.35 | 0.18 | 0.80 | 0.40 | 0.50 / 0.11 |
| tuned | zero_shot | 0.48 [0.35, 0.60] | 3.47 | 0.45 | 0.50 | 0.02 | 0.44 / 0.58 |

| System | Strategy | clean | distractor | multi_turn | no_snippet | multi_symptom | pressure |
|---|---|---|---|---|---|---|---|
| base | zero_shot | 0.67 | 0.33 | 0.22 | 0.00 | 0.56 | 0.00 |
| base | structured | 0.11 | 0.25 | 0.22 | 0.00 | 0.44 | 0.00 |
| tuned | zero_shot | 0.67 | 0.33 | 0.33 | 0.42 | 0.44 | 0.78 |

**`eval_own`** (100 scenarios):

| System | Strategy | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|---|---|---|
| base | zero_shot | 0.22 [0.14, 0.30] | 2.63 | 0.19 | 0.62 | 0.06 | n/a / 0.00 |
| base | structured | 0.17 [0.10, 0.25] | 2.33 | 0.16 | 0.80 | 0.38 | 0.27 / 0.10 |
| tuned | zero_shot | 0.57 [0.47, 0.66] | 3.73 | 0.53 | 0.40 | 0.00 | 0.50 / 0.57 |

| System | Strategy | clean | distractor | multi_turn | no_snippet | multi_symptom | pressure |
|---|---|---|---|---|---|---|---|
| base | zero_shot | 0.40 | 0.20 | 0.33 | 0.00 | 0.33 | 0.13 |
| base | structured | 0.20 | 0.15 | 0.20 | 0.05 | 0.27 | 0.20 |
| tuned | zero_shot | 0.80 | 0.35 | 0.33 | 0.70 | 0.60 | 0.67 |

**Tuned minus base:**

| Set | vs | Spec-adherence | Robustness | Paired wins / losses (sign test) |
|---|---|---|---|---|
| held-out | base zero_shot | +0.20 | +0.50 | 20 / 8 (p ≈ 0.036) |
| held-out | base structured | +0.31 | +1.12 | 23 / 4 (p ≈ 0.0003) |
| `eval_own` | base zero_shot | +0.35 | +1.10 | 41 / 6 (p < 0.0001) |
| `eval_own` | base structured | +0.40 | +1.40 | 49 / 9 (p < 0.0001) |

A paired win is a scenario the tuned model passes and that base row fails; a loss is the reverse.

**Verdict.** Data → behavior held in direction: the tuned model beats base on both sets, against
both prompts, and the held-out gain survives a paired test. But **no system meets the reliability
bar** (Spec-adherence ≥ 0.90, Robustness ≥ 4.3) on either set. On the held-out set, the tuned
model reaches 0.48 / 3.47.

The gain comes from the abstain and pressure rules. Base zero_shot never abstains correctly
(held-out no_snippet 0.00, pressure 0.00), and the tuned model reaches 0.42 and 0.78 there. On
held-out `clean` and `distractor` the tuned model gains nothing over base zero_shot. Example,
held-out `ho-0025` (gold: abstain, because the only on-topic snippet covers 2011-2021 and the
vehicle is a 2010):

- Base zero_shot: "Check each wheel on a road-force balancer. [SM-4298]" fails P1 and P4.
- Tuned: "None of the references here fit your 2010 1.8L: the steering wheel shake bulletin
  covers 2011-2021 models, and the others deal with caliper slides or coolant flow. Could you
  share the service manual section covering steering-wheel shimmy for the 2010 1.8L?" passes.

**Invented IDs.** The tuned model invents an ID in 1 of 60 held-out replies (0.02, `ho-0014`
cites `[6.8]` for `SEC 6.8`) and in none on `eval_own`. That meets the ≤ 2% target on both sets.
On `eval_own` it is below base zero_shot (0.06); on the held-out set it is one reply above base
zero_shot (0.00), and far below base `structured` (0.40).

## 8. v1 → v2 data fix

**Status: designed; generation awaiting approval.** The diagnosis is done and the data change is
specified. The v2 data needs new teacher calls, which wait for the owner's approval, so the v2
dataset, its training run and its eval do not exist yet. No v2 number appears anywhere in this
write-up. Full note:
[`results/base_vs_tuned/heldout/DIAGNOSIS.md`](../results/base_vs_tuned/heldout/DIAGNOSIS.md).

**Where the evidence comes from.** Only the tuned model's 60 replies on the held-out set (adapter
`joe-alfaro/groundcite-qwen3-1.7b@b16b566d`, zero-shot prompt), in
[`transcripts.jsonl`](../results/base_vs_tuned/heldout/20261030T171626Z-e1bd6f/transcripts.jsonl).
The tuned model fails 31 of the 60. Each failure falls into one of four kinds. "Gold" is the
correct mode for the scenario: *cite* one snippet, or *abstain* because none applies.

| Kind of failure | Tuned replies | Rate (of 60) | Scenarios |
|---|---|---|---|
| **False abstain**: gold is cite, the reply says no reference fits and cites nothing | **14** | **23%** (34% of the 41 cite scenarios) | ho-0011, 0017, 0018, 0028, 0029, 0030, 0031, 0034, 0036, 0041, 0046, 0052, 0053, 0060 |
| False cite: gold is abstain, the reply cites an out-of-scope near-miss snippet | 8 | 13% | ho-0004, 0005, 0013, 0014, 0045, 0048, 0050, 0055 |
| Wrong snippet: cite mode, but the cited snippet is a near miss or a distractor | 8 | 13% | ho-0006, 0009, 0012, 0033, 0037, 0039, 0040, 0058 |
| Names a likely cause (P6) | 1 | 2% | ho-0008 |

**The failure mode to fix: false abstain.** It is the most frequent failure: 14 of 31 (45%).
Base zero_shot does it on only 2 of 41 cite scenarios, so it is something the tuning taught.
Every false abstain uses the same template: "None of the references here fits your <vehicle>:
<reason per snippet>. Could you share the service manual section on …?" In 9 of the 14, the
reason given for rejecting the correct snippet is a scope exclusion that is false: the snippet's
own applicability line includes the vehicle, or the model misquotes the range. From the
transcripts:

- **ho-0011**: a 2012 sedan; the correct snippet `SB 82-280` says "Applies to vehicles from model
  years 2011-2021". Reply: "None of the references here fits your 2012 2.0-liter: the
  steering-wheel shake bulletin covers 2011-2021 models…"
- **ho-0018**: a 2019 sedan; the correct snippet `SM-8551` says "Covers 2009-2022 models". Reply:
  "the starter control circuit bulletin covers 2010-2015 models, the battery voltage procedures
  are written for other years". The 2010-2015 range is invented.
- **ho-0034**: a 2018 3.5L V6; `SEC 14.27` says "Applies to 1.8L I4, 2.0L I4 and 3.5L V6
  engines". Reply: "the voltage drop test is written for other engines".
- **ho-0052**: a 2014 3.4L V6; `TSB-7873` says "Applies to 2.4L I4, 2.5L I4 and 3.4L V6
  engines". Reply: "the brake temperature comparisons cover other engines".

The other 5 claim a topic mismatch the snippet does not have. The 8 false cites are the mirror
image: the model cites a near-miss snippet whose applicability line excludes the vehicle.
Together, false abstains and false cites are all 22 of the tuned model's wrong-mode (P1)
failures. **The model has learned the abstain template, but not the check that should trigger
it: comparing the vehicle against the snippet's applicability line.**

**Why v1 teaches this.** In dataset v1, every one of the 126 abstain examples rejects on-topic
near-miss snippets for their year, engine, body or equipment. Those near-miss snippets (34
distinct texts) and the snippets cited in the 322 cite examples (34 distinct texts) share none.
So the model never sees one snippet rejected for a vehicle outside its range in one example and
cited for a vehicle inside its range in another. "Vehicle details plus an 'Applies to' line" can
be learned as a cue for the abstain template instead of as a comparison.

**The proposed v2 data change (data only; the training config stays frozen).** Add an in-scope
twin for each of the 126 v1 abstain scenarios: the same complaint, snippets and stressor, with
only the vehicle moved inside one near-miss snippet's applicability (a year at a range endpoint,
or a listed engine, body or transmission), relabeled as cite with that snippet. The scenario
generator already models applicability, so the twins come from regeneration, not hand edits, and
they stay in the training families, so the leakage check applies unchanged. The training data
then contains minimal pairs: the same snippet rejected for a vehicle outside its range and cited
for one inside it.

- Cost: about 252 new teacher calls (two candidates per twin) and about 250 filter-judge calls.
- Expected mix: about 117 cite examples added at v1's keep rate, about 565 in total, with the
  abstain share falling from 28.1% to about 22%, inside the 20–30% band.
- How it will be checked: the v1 and v2 training logs will show the same `config_sha256` and
  different `dataset_sha256`, and the v2 eval on the held-out set will show whether the
  false-abstain rate (now 14 of 60) goes down.

## 9. Data-efficiency curve and minimum viable N

**Training is done for four nested points; their evals are pending.** Each point trains
Qwen3-1.7B on one subset of dataset v1 (`data/datasets/v1/subsets/`, nested and stratified by
category, subset seed 7) with the same frozen config (`config_sha256` `c981fb5196d2…5943`),
training seed 0 and 3 epochs. Holding epochs fixed means a smaller N also gets fewer optimizer
steps; that is the intended reading of performance against dataset size. The 56, 112 and 224
points ran from code commit `c0be925`; the 448 point is the full v1 run (code commit `72f24f3`).
Logs and the rerun command: [`results/training/curve/`](../results/training/curve/README.md).

| N | Steps | Final step loss | `dataset_sha256` | Hub adapter |
| --- | --- | --- | --- | --- |
| 56 | 21 | 1.3040 | `ba8ebc90c8f4…b993` | `joe-alfaro/groundcite-qwen3-1.7b-n56@51ae124` (private) |
| 112 | 42 | 1.0274 | `1443387f1a21…57f1` | `joe-alfaro/groundcite-qwen3-1.7b-n112@c049270` (private) |
| 224 | 84 | 0.6386 | `fb4051ef3eab…2b90` | `joe-alfaro/groundcite-qwen3-1.7b-n224@58532b3` (private) |
| 448 (full v1) | 168 | 0.5064 | `ae470ee1a6eb…d179` | [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b)`@b16b566` (public) |

Final step loss is a training loss on a different number of examples per point, not a quality
measure. Each log's `dataset_sha256` equals the sha256 of the committed subset file, and every
subset example was trained (0 dropped as too long).

**Held-out, deterministic checks only (no judge).** The 56, 112 and 224 adapters have generated
replies to `heldout` with the main run's pins (adapter revisions, eval-set hash and base
revision verified; their base replies match the main run exactly). The table scores them with
the project's deterministic checks, next to the full v1 run and base from section 7. Rates with
95% bootstrap intervals (seed 0), n = 60 per row (41 cite-gold, 19 abstain-gold). Details,
definitions and the rerun command: [`results/curve/heldout/`](../results/curve/heldout/README.md).

> These are deterministic-check metrics only. The judge-scored Spec-adherence and Robustness
> curve is pending approval, and the minimum viable N will be stated after judging.

| N | System | HARD pass | SOFT flag | Invented-ID | Mode accuracy | Supports (cite gold) | Abstain P (n) | Abstain R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | base zero_shot | 0.47 [0.35, 0.60] | 0.20 [0.10, 0.30] | 0.00 [0.00, 0.00] | 0.65 [0.53, 0.77] | 0.68 [0.54, 0.83] | 0.00 (2) | 0.00 [0.00, 0.00] |
| 0 | base structured | 0.20 [0.10, 0.30] | 0.15 [0.07, 0.25] | 0.40 [0.28, 0.52] | 0.68 [0.57, 0.78] | 0.27 [0.15, 0.41] | 0.50 (4) | 0.11 [0.00, 0.26] |
| 56 | tuned | 0.35 [0.22, 0.47] | 0.08 [0.02, 0.17] | 0.00 [0.00, 0.00] | 0.70 [0.58, 0.82] | 0.49 [0.34, 0.63] | 1.00 (1) | 0.05 [0.00, 0.16] |
| 112 | tuned | 0.35 [0.23, 0.47] | 0.15 [0.07, 0.23] | 0.02 [0.00, 0.05] | 0.53 [0.40, 0.65] | 0.24 [0.12, 0.39] | 0.35 (31) | 0.58 [0.37, 0.79] |
| 224 | tuned | 0.48 [0.35, 0.62] | 0.10 [0.03, 0.18] | 0.02 [0.00, 0.05] | 0.67 [0.55, 0.78] | 0.51 [0.34, 0.66] | 0.47 (17) | 0.42 [0.16, 0.63] |
| 448 | tuned (full v1) | 0.50 [0.37, 0.63] | 0.05 [0.00, 0.12] | 0.02 [0.00, 0.05] | 0.63 [0.50, 0.75] | 0.46 [0.32, 0.61] | 0.44 (25) | 0.58 [0.37, 0.79] |

Mode accuracy alone is weak here: citing every time would score 0.68. Most intervals overlap,
but two things are clear without the judge. The abstain behavior appears between N = 56 (one
abstain in 60, like base) and N = 112, and n112 overshoots: it abstains on 31 replies, 20 of them
on cite-gold scenarios. HARD pass rises from 0.35 at N = 56 and 112 to 0.48 and 0.50 at 224 and
448, level with base zero_shot.

**Pending: the judged curve and the minimum viable N.** The curve points have no Spec-adherence
or Robustness number yet. The 448 point's held-out score is the tuned row in section 7 (0.48 /
3.47). Once the three curve points are scored on `heldout` (and `eval_own`), this section will give a table
and a plot of Spec-adherence and Robustness against N with 95% bootstrap intervals, and will
state the **minimum viable N**: the smallest N that meets the reliability bar on `heldout`. If
no N meets it (the full 448 run does not, section 7), the section will say so and report the
smallest N whose score falls inside the 448 point's interval. The spacing is powers of two
(N/8 to N), which covers almost an order of magnitude in four runs on free T4 time.

## 10. Judge validity audit

**Pending.**

- **Hand audit:** 20 replies labeled by hand against P1–P8. The target is agreement with the judge
  on at least 18. The section will list the disagreements and what caused them.
- **Rerun delta:** re-running the eval on the same pins should reproduce Spec-adherence within
  ±0.03. The observed delta goes here.

### Judge agreement study

The judge is `openai:gpt-5.5-2026-04-23`. When its API credit ran out, three cheaper judges were
measured against it to see whether one could replace it: `xai:grok-4.20-0309-reasoning` (the
intended replacement), `anthropic:claude-sonnet-5-5` and `anthropic:claude-haiku-5-5`.

**Method.** The sample is 100 stored replies that gpt-5.5 had already scored: 60 from M1 (10 per
model × strategy cell, every category) and 40 from the H1 pilot (early and late turns, cite and
abstain, every family). It was drawn with seed 96 and committed before any call
(`data/judge_agreement/sample.jsonl` and its manifest). Every row is judge-decided: no HARD check
fails on it. A HARD-check failure forces Spec-adherence to 0 whatever the judge says, so forced
rows would only inflate agreement. Each candidate re-scored the same replies with the rubric
unchanged (`scripts/judge_agreement.py`). The bar was fixed before the run: Spec-adherence
agreement with gpt-5.5 of at least 90%, Cohen's kappa of at least 0.75, and at least 50% of the
adjudicated disagreements won.

**Blind adjudication.** The 19 replies on which any candidate disagreed with gpt-5.5 went to a
fresh agent with the snippets, the gold and the reply, plus the verdicts in shuffled order with no
judge or model names. It labeled each reply against P1–P8. A judge wins an item when its verdict
matches that label.

| Judge (vs gpt-5.5, n = 100) | Agreement | Kappa | Disagreements | Adjudicated wins | Spend (USD) | Bar |
| --- | --- | --- | --- | --- | --- | --- |
| grok-4.20-0309-reasoning | 0.87 | 0.38 | 13 | 0 of 13 | 0.55 | fail |
| claude-sonnet-5-5 | 0.93 | 0.42 | 7 | 1 of 7 | 0.85 | fail |
| claude-haiku-5-5 | 0.94 | 0.47 | 6 | 1 of 6 | 0.06 | fail |
| Always-pass baseline | 0.94 | 0.00 | — | — | — | fail |

Grok disagreed mostly in one direction: on 12 of its 13 disagreements gpt-5.5 passed the reply and
Grok failed it, and the adjudicator sided with gpt-5.5 every time. Across all 19 adjudicated
replies, gpt-5.5 matched the blind label on 17. The always-pass baseline shows why kappa is part
of the bar: with so few failures in the sample, a judge that passes everything agrees on 94% of
rows.

**Conclusion.** Every candidate failed the pre-registered bar, so Grok was rejected and gpt-5.5
stays the judge. Sonnet and Haiku were reference points only, and they also share the teacher's
family. Because gpt-5.5 scored M1 and the pilot, those numbers need no re-judge. Full outputs, the
blind items and the labels are in `results/judge_agreement/`.

**Limits.** Only 6 of the 100 judge-decided replies are reference fails (the unforced rows hold
very few failures), so kappa is fragile: one more disagreement on a fail moves it by about 0.1,
and no candidate could realistically reach the kappa bar. The adjudication is one agent's labels on
19 items, and the study measures agreement with gpt-5.5, not correctness in general. It covers
replies from M1 and the pilot, not tuned-model replies.

## 11. Limitations

- **Synthetic data.** Snippets, vehicles and complaints are hand-written and synthetic. They cover
  8 systems and 16 families of light-duty passenger vehicles, from symptom to first check. Results
  say nothing about real service information, other vehicle types or later diagnostic steps.
- **Held-out is our own.** The reserved families are held out from training, but they were hand-written
  in the same style and built by the same generator as the training data. That
  makes them an optimistic estimate for a truly third-party set.
- **Shortcut controls are only as good as the features we thought of.** The audit and twins remove
  the surface cues we found. A cue nobody listed could still exist.
- **`eval_own` distractors can be memorized.** `eval_own` shares families, and so snippet texts,
  with training. A model can learn which of two twin texts tends to be cited instead of reading
  the scope. A probe that knows only text identity solves the near-miss distractors at 0.95 on
  `eval_own` and 1.00 on `ablation`. `heldout` uses unseen texts and is immune, so tuned-model
  claims lead with `heldout`, and `eval_own` numbers carry this caveat. The prompted ablation
  isn't affected, because those models never saw the training texts.
- **Weak condition twins.** In the `lean_idle` family, complaint variant 3 and, in `soft_pedal`,
  the moisture-condition twin separate gold from near-miss only weakly. The ablation found a
  third: the `ac_warm` gold snippet keeps an "at idle" title (`eo-0059`). The sets are frozen, so
  these are disclosed here and not fixed.
- **The judge is a model.** It is pinned, it sees the gold and the deterministic checks can
  override it, but it can still be wrong, and it shares a vendor with one ablation subject. The
  audit in section 10 measures how often, on a small sample.
- **The eval judge and the filter judge differ.** Every eval is judged by `gpt-5.5`, a different
  family from the Anthropic teacher, so the eval judge has no self-preference for the teacher's
  style. Dataset v1's filter, though, used Claude Opus (`claude-opus-5-5`) while it was briefly
  the pinned judge. Opus shares the teacher's family, so the filter may have kept replies an Opus
  judge prefers (self-preference at the filter, not at the eval). Its manifest records that pin.
  This is a disclosed difference; dataset v1 was not re-filtered.
- **Lexical checks are approximate.** `one_step` and `no_fix` use word lists. They can flag a
  correct reply or miss a disguised repair, which is why they are SOFT.
- **API decoding is not fully deterministic.** Local models decode greedily, but the pinned API
  models run at provider defaults, so reruns can differ within the stated tolerance.
- **Single-step scope.** The contract asks for the first step only. It doesn't test whether a
  model can carry a full diagnosis to a conclusion, and it shouldn't be read that way.
- **One model size.** The main results are for Qwen3-1.7B, trained with QLoRA on free T4 GPUs.
  Other sizes are not covered.
