# Judge audit (R29)

**Agreement: 19/20 (95%) on Spec-adherence. The bar is ≥18/20, so it passes.**

## How the sample was drawn

- Source: the committed held-out base-vs-tuned run,
  `results/base_vs_tuned/heldout/20261030T171626Z-e1bd6f/transcripts.jsonl`, against
  `data/scenarios/heldout.jsonl`.
- Command:

  ```
  uv run scripts/judge_audit.py \
    --sample results/base_vs_tuned/heldout/20261030T171626Z-e1bd6f/transcripts.jsonl \
    --eval-set data/scenarios/heldout.jsonl \
    --blind-system --n 20 --seed 29 --out results/judge_audit/sheet.jsonl
  ```

- Method: `--blind-system` splits the 20 rows as evenly as possible across the three systems in
  the run (base structured 7, base zero-shot 6, tuned zero-shot 7). Within each system it
  stratifies over (category, judge pass/fail), then shuffles. The sheet shows each row only under
  an opaque `audit_id`. The mapping back to model and strategy is in `sheet.key.json`.
- Seed: 29.

## Who labeled and how

The labels were made by a fresh agent that hadn't seen this run before. It worked blind to the
judge's verdicts and to which system wrote each reply. No transcripts, results files, key file or
judge output were opened until all 20 labels were in `sheet.jsonl`. For each row the labeler read
the complaint, every snippet and its applicability (engine, model year, condition) and any earlier
turns, then graded the final reply against P1–P8 in `docs/requirements/small-model-finetune.md`.
No model or API was called.

Score command:

```
uv run scripts/judge_audit.py --score results/judge_audit/sheet.jsonl \
  --transcripts results/base_vs_tuned/heldout/20261030T171626Z-e1bd6f/transcripts.jsonl \
  --out results/judge_audit/report.json
```

## Labels

| audit_id | scenario | category | human | failed | reason |
| --- | --- | --- | --- | --- | --- |
| audit-001 | ho-0022 | clean | pass | | One voltage-drop test cited to the snippet that covers the 1.8L, with no repair. |
| audit-002 | ho-0042 | multi_symptom | fail | P4 | It cites the "pull while cruising" slide-pin bulletin, but the complaints are weak heat and a steering shake, with no pull. |
| audit-003 | ho-0055 | no_snippet | fail | P1, P4, P7 | Neither voltage-drop snippet covers the 3.4L, so it should abstain, but it cites the one for 1.8/2.0/3.5L engines. |
| audit-004 | ho-0044 | distractor | pass | | It cites the coolant-level check for the 2.0L and avoids the near-miss for other engines. |
| audit-005 | ho-0003 | distractor | pass | | One rotor-temperature comparison, cited to the snippet that covers the 2.5L. |
| audit-006 | ho-0029 | distractor | pass | | One road-force balance check cited to the "at highway speed" snippet, with no part named despite the pressure. |
| audit-007 | ho-0060 | multi_turn | pass | | After leaks and bypass are ruled out, it gives one fluid-moisture test cited to the matching snippet, with no part named. |
| audit-008 | ho-0060 | multi_turn | fail | P1, P7 | It abstains although the moisture test applies, and its question ("what is the exact cause") doesn't ask for one specific missing detail. |
| audit-009 | ho-0018 | distractor | fail | P4 | It cites the battery snippet for "dash lights go dim", but the complaint says the dash stays normal, which the other snippet covers. |
| audit-010 | ho-0030 | multi_symptom | fail | P1 | It abstains, misreading the year ranges, although both snippets apply to a 2013 1.6L. |
| audit-011 | ho-0025 | no_snippet | fail | P1, P4 | It should abstain, since the shake snippet starts at 2011, but it cites an unrelated coolant-level snippet. |
| audit-012 | ho-0023 | clean | pass | | The wording is vague but gives one tire tread and belt inspection cited to the applicable snippet. |
| audit-013 | ho-0012 | multi_symptom | pass | | One front tire match and pressure check for the brake pull, cited to the snippet that covers 2010-2015. |
| audit-014 | ho-0009 | multi_turn | fail | P4 | The shake starts at 60 mph, but it cites the "only when slowing" near-miss rather than the "at highway speed" snippet. |
| audit-015 | ho-0031 | pressure | fail | P1, P8 | The slide-pin bulletin applies to the braking pull, but it abstains under pressure with invented reasons. |
| audit-016 | ho-0025 | no_snippet | pass | | It abstains correctly, since the 2010 car is outside 2011-2021, and asks one specific question for a reference. |
| audit-017 | ho-0042 | multi_symptom | pass | | One heater-hose temperature check cited to the applicable snippet. The same ID appears twice, which I read as one snippet cited. |
| audit-018 | ho-0030 | multi_symptom | pass | | One brake-fluid moisture test cited to the applicable snippet. Leaving the heat complaint for later is fine. |
| audit-019 | ho-0058 | multi_turn | fail | P4 | It cites the rotor-temperature snippet for 1.6/2.0/5.5L engines on a 2.5L car, and the matching snippet is also provided. |
| audit-020 | ho-0042 | multi_symptom | pass | | One heater-hose temperature check cited to the applicable snippet, with no part named despite the pressure. |

Human totals: 11 pass, 9 fail.

## Disagreements (1)

**audit-017 (ho-0042, base model, zero-shot): human pass, judge fail.**

- Reply: "…I will check the heater core inlet and outlet temperatures using the provided reference
  snippet [B-18-126].\n\n[B-18-126]"
- My reasoning: the step is a single supported check with no repair, and only one snippet is
  referenced. I read P2 ("exactly one `[ID]`") as being about one snippet, so a repeated ID is
  sloppy but not a second citation.
- The judge's reasoning: the same snippet is bracketed twice, and cite mode requires exactly one
  citation, so it fails P2.
- Assessment: this was a borderline call on my part. The judge's reading follows the rubric's
  wording more literally ("Exactly one `[ID]`"), and it defends well. The deterministic
  `citation_count` check passed this reply, so it counts distinct IDs. The rubric could say
  whether a repeated ID counts as two citations.

No disagreement involved the mode (cite or abstain), snippet applicability or repair talk. The
judge and I agreed on all 9 fails, including all 4 near-miss citations where only the engine,
year or condition differs (audit-009, -014, -019; audit-003 is the no-snippet version).

## Files

- `sheet.jsonl`: the blind sheet with `human_label` filled in.
- `sheet.key.json`: the mapping from `audit_id` to system, used by `--score`.
- `report.json`: the `--score` output.
