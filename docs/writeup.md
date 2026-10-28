# Write-up (draft)

> **Status: draft.** Sections 1, 2, 3, 5 and 6 (method) describe what is built or designed in this
> repository. Section 4 reports the real prompt-ceiling ablation. Every other number that needs a
> real run is marked **Pending**, along with what will go there. Nothing in the pending sections
> should be read as a result.

Requirements: [`docs/requirements/small-model-finetune.md`](requirements/small-model-finetune.md).
Architecture: [`ARCHITECTURE.md`](../ARCHITECTURE.md). Status of every item:
[`docs/checklist.md`](checklist.md).

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

The pipeline is merged to `main`. It has not been run on the real teacher yet.

**Pending:**

- Dataset v1 size (target ≥ 300 kept examples) and its category and abstain mix.
- Candidates, kept and the kept rate (R4 requires 20–95%). With k = 2 and one kept example per
  scenario, the kept rate can't exceed 50%, so the rejections by reason are the more useful
  signal.
- Rejections by reason, from the filter report.

## 7. Base vs tuned

**Pending.** One QLoRA run of Qwen3-1.7B on v1, evaluated with
`uv run eval.py --model <hf-repo-id> --eval-set <path>` on `eval_own` and `heldout`. Each table
will have these rows: base with the zero-shot prompt, base with the best ablation strategy, and
tuned with the zero-shot prompt. The columns are the same as in section 4. The section will report
whether data → behavior held, with numbers, the pinned model revision, and quoted transcripts that
show base failing and tuned passing on the same scenario. It will also report the invented-ID
rate, which should be no higher than base and at most 2% on both sets.

## 8. v1 → v2 data fix

**Pending.** One failure mode found in the v1 transcripts, with its v1 rate and example scenario
IDs. The data change that targets it (examples added, removed or relabeled), with the committed
dataset diff summary. The v1 and v2 training logs will show the same config hash and different
dataset hashes, and the v2 eval on the same set will show whether the rate went down. Results are
fixed by changing data, not hyperparameters.

## 9. Data-efficiency curve and minimum viable N

**Pending.** At least 4 checkpoints trained on nested subsets (N/8 ⊂ N/4 ⊂ N/2 ⊂ N) of the final
dataset, drawn from one seeded shuffle stratified by category, with the same hyperparameters, seed
and epoch count. Holding epochs fixed means a smaller N also gets fewer optimizer steps; that is
the intended reading of performance against dataset size. A plot and a table will show
Spec-adherence and Robustness against N on `eval_own` and `heldout`, with 95% bootstrap intervals.
The **minimum viable N** is the smallest N that meets the reliability bar on `heldout`. If no N
meets it, this section will say so and report the smallest N whose score falls inside the
largest-N interval. A paragraph will justify the log spacing.

## 10. Judge validity audit

**Pending.**

- **Hand audit:** 20 replies labeled by hand against P1–P8. The target is agreement with the judge
  on at least 18. The section will list the disagreements and what caused them.
- **Second-family rescore:** a judge from the other vendor rescores 40 examples. The target is at
  least 85% agreement.
- **Rerun delta:** re-running the eval on the same pins should reproduce Spec-adherence within
  ±0.03. The observed delta goes here.

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
- **Lexical checks are approximate.** `one_step` and `no_fix` use word lists. They can flag a
  correct reply or miss a disguised repair, which is why they are SOFT.
- **API decoding is not fully deterministic.** Local models decode greedily, but the pinned API
  models run at provider defaults, so reruns can differ within the stated tolerance.
- **Single-step scope.** The contract asks for the first step only. It doesn't test whether a
  model can carry a full diagnosis to a conclusion, and it shouldn't be read that way.
- **One model size.** The main results are for Qwen3-1.7B, trained with QLoRA on free T4 GPUs.
  Other sizes are not covered.
