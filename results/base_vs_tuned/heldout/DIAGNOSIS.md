# v2 diagnosis (R7): the tuned model on the held-out set

Source: the 60 tuned rows (`hf:joe-alfaro/groundcite-qwen3-1.7b@b16b566d`, zero_shot) in
[`20261030T171626Z-e1bd6f/transcripts.jsonl`](20261030T171626Z-e1bd6f/transcripts.jsonl). Base
rows are not used here. The tuned model fails 31 of 60.

## Failures by kind

| Kind | Tuned replies | Rate (of 60) | Scenarios |
|---|---|---|---|
| **False abstain**: gold is cite, the reply says no reference fits and cites nothing | **14** | **23%** (34% of the 41 cite scenarios) | ho-0011, 0017, 0018, 0028, 0029, 0030, 0031, 0034, 0036, 0041, 0046, 0052, 0053, 0060 |
| False cite: gold is abstain, the reply cites an out-of-scope near-miss snippet | 8 | 13% | ho-0004, 0005, 0013, 0014, 0045, 0048, 0050, 0055 |
| Wrong snippet: cite mode, but the cited snippet is a near miss or a distractor | 8 | 13% | ho-0006, 0009, 0012, 0033, 0037, 0039, 0040, 0058 |
| Names a likely cause (P6) | 1 | 2% | ho-0008 |

## The most frequent failure: false abstain with an invented scope exclusion

The single most frequent failure is a **false abstain**: 14 of 60 tuned replies (23%), 14 of the
41 scenarios whose gold is cite (34%), and 14 of the 31 tuned failures (45%). Base zero_shot does
this on 2 of 41.

Every false abstain uses the same template: "None of the references here fits your <vehicle>:
<reason per snippet>. Could you share the service manual section on …?" In 9 of the 14, the reason
it gives for rejecting the gold snippet is a **scope exclusion that is false**: the snippet's own
applicability line includes the user's vehicle, or the model misquotes the range.

- **ho-0011**: 2012 sedan; gold `SB 82-280` "Applies to vehicles from model years 2011-2021". Reply:
  "None of the references here fits your 2012 2.0-liter: the steering-wheel shake bulletin covers
  2011-2021 models…"
- **ho-0018**: 2019 sedan; gold `SM-8551` "Covers 2009-2022 models". Reply: "the starter control
  circuit bulletin covers 2010-2015 models, the battery voltage procedures are written for other
  years". The 2010-2015 range is invented.
- **ho-0034**: 2018 3.5L V6; gold `SEC 14.27` "Applies to 1.8L I4, 2.0L I4 and 3.5L V6 engines".
  Reply: "the voltage drop test is written for other engines".
- **ho-0052**: 2014 3.4L V6; gold `TSB-7873` "Applies to 2.4L I4, 2.5L I4 and 3.4L V6 engines".
  Reply: "the brake temperature comparisons cover other engines".
- Also ho-0028, ho-0030, ho-0031, ho-0036, ho-0041. The other 5 (ho-0017, 0029, 0046, 0053, 0060)
  claim a topic mismatch the snippet does not have ("None of the references I have covers cabin
  heat…" when `TSB-3657` is the blend-door heat test).

The 8 false cites are the mirror image: the model cites a near-miss snippet whose applicability
line excludes the vehicle. Together, the two make up all 22 of the tuned model's P1 (wrong mode)
failures. **The model has learned the abstain template, but not the check that should trigger it:
comparing the vehicle against the snippet's applicability line.**

## Why the v1 data teaches this

In dataset v1 every one of the 126 abstain examples is a near-miss case: the snippets are on topic
and the reply rejects each one for its year, engine, body or equipment. The near-miss snippets
and the cite golds are disjoint sets: the abstain examples use 34 distinct near-miss snippet
texts, the 322 cite examples use 34 distinct gold texts, and the two share none (compared by
exact snippet text in `data/scenarios/train_pool.jsonl`). The model never sees a snippet rejected
for a vehicle in one example and cited for a vehicle inside its stated range in another. So
"vehicle details plus an 'Applies to' line" can be learned as a cue for the abstain template
instead of a comparison. On the unseen held-out families, the model falls back to that cue.

## The v2 data change (one change, no training-config change)

**Add an in-scope twin for each abstain example.** For each of the 126 v1 abstain scenarios,
regenerate a twin scenario with the same snippets, complaint and stressor. Edit only the vehicle
so it falls inside one near-miss snippet's applicability: the year at a range endpoint, or a
listed engine, body or transmission. Relabel the twin `gold.mode = cite` with that snippet as
`supporting_ids`. The scenario generator already models applicability (`Applies.matches`,
`Applies.changed_from` in `src/groundcite/scenarios/library.py`), so the twins come from
regeneration, not hand edits. They stay in the 11 training families, so the leakage check
(`scripts/check_leakage.py`) still applies unchanged.

The training data then contains minimal pairs: the same snippet rejected for a vehicle outside its
range and cited for a vehicle inside it. Nothing else changes: same teacher prompt, same filter,
same selection rule, same training config (G21).

Expected effect on the mix: at v1's per-scenario keep rate (448 / 480 = 93%), about 117 cite
examples are added, for about 565 in total. Abstain share falls from 28.1% to about 22%, still
inside the 20-30% band.

Not chosen: re-admitting the 32 cite scenarios lost to the soft-check false positives (64 existing
candidates, no teacher calls; see `data/datasets/v1/NOTES.md`). It adds cite coverage, but no
scope-boundary pairs, and needs a filter change rather than a data change.

## Teacher calls needed

**Yes: about 252 new teacher calls.** The twins are new scenarios, so no existing candidate
covers them. At v1's two candidates per scenario, that is 126 × 2 = 252 calls to the teacher
(`anthropic:claude-opus-5-5`, prompt t1). The filter adds up to about 2 judge calls per candidate
that passes the deterministic checks (hint screen and filter judge, as in v1: 916 calls for 960
candidates). That puts the filter at about 250 calls. These go to the filter's judge and do not
count against the gpt-5.5 eval-judge cap (D-116). One candidate per scenario (126 calls) would
halve the teacher cost, at the price of lower coverage after the filter.
