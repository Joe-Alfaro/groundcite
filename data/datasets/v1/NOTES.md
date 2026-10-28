# Dataset v1: quality note

Built by `scripts/build_dataset.py` from `data/distill/v1/candidates.jsonl` (960 teacher
candidates, two per scenario over 480 training scenarios; sha256 `bcb8b4cf…a5ce`). The generated
numbers are in `filter_report.md` and `manifest.json`. This note covers what a read of the data
shows on top of them.

## Summary

| Item | Value |
| --- | --- |
| Kept | 448 of 960 candidates (46.7%), one per scenario; 448 of 480 scenarios |
| R6 floor (≥ 300) | met |
| Kept rate within 20-95% | yes |
| Abstain share | 28.1% (126 of 448), within 20-30% |
| Categories | clean 67, distractor 86, multi_symptom 67, multi_turn 65, no_snippet 96, pressure 67 |
| Teacher / judge | `anthropic:claude-opus-5-5` for both (prompt t1; rubric `636a9c79…`, hint screen v1 `1c3bd86e…`) |
| Judge spend | 916 calls (458 hint screens + 458 judge calls), 2.36M input + 0.20M output tokens, about $13.40 at $4/$20 per MTok |

## What was read

- 25 kept targets, at least four per category (5 no_snippet, 4 each of clean, distractor,
  multi_symptom, multi_turn, pressure).
- 15 rejections: the single `no_fix` rejection, 8 `one_step` rejections and 6 `not_selected`
  (3 judged at Robustness 4, 3 never judged).
- Automated passes over all 448 kept: numbers in cite replies that appear in neither the cited
  snippet nor the user turns; how abstain replies phrase their question; cited snippets with an
  equipment precondition the user never states.

## Kept targets: good

- Each reply gives one concrete test, cites the gold snippet and stays inside the snippet's
  procedure and limits. Pass/fail values (psi, volts, °F, mA) are carried over correctly. The only
  three numbers missing from the cited snippet are `2,000` split at its comma, so no invented
  specs were found.
- Multi-turn replies pick up the user's last result ("a clean smoke test rules out unmetered
  air…") and move to the next gold snippet without repeating the cleared one.
- Pressure replies acknowledge the push for a part and still give the test, without naming a part.
- Abstain replies say why each provided snippet does not apply (wrong engine, model-year range,
  wrong condition) and end with one question. Of the 126, 95 ask for a vehicle detail and 31 ask
  for the matching service-manual section.
- Multi-symptom replies address the symptom the snippets support and leave the others alone.

## Kept targets: weaknesses (candidates for v2)

1. **Equipment preconditions the user never states.** 61 kept cite replies cite a snippet that
   "applies to vehicles with" a piece of equipment the complaint never mentions: front disc
   brakes 24, front struts 15, clutch-driven compressor 10, electric condenser fan 7, front
   stabilizer bar 5. Examples: tr-0037 (front struts on a full-size pickup), tr-0101, tr-0009,
   tr-0071. The teacher treats the equipment as assumed. That is defensible for disc brakes and
   less so for struts or a clutch-driven compressor. Every "conventional automatic" citation (34)
   comes from a user who said "automatic".
2. **Vehicle class and transmission boundaries are fuzzy in gold.** tr-0045 abstains because
   the ball-joint snippet lists "compact crossovers" and the user has a "small crossover".
   tr-0409 abstains because the rotor snippet says "an automatic transmission" and the user has a
   CVT, and the reply paraphrases the snippet as "a conventional automatic". The reply follows
   gold, but a student could learn an arbitrary line. These boundaries should be stated in the
   snippets or kept out of near-miss pairs.
3. **Abstain wording repeats.** Most abstain replies open with "None of the references I
   have…" or "Neither reference fits…". That is fine for the contract, but the phrasing is
   narrow.

## Rejections

Only two stages rejected anything: `soft_check` 102 (one_step 101, no_fix 1) and `not_selected`
410 (the second candidate of each kept scenario, or the lower-Robustness one).

- **The soft regex screen is over-strict.** All 9 soft_check rejections read, plus 8 more `then` and
  `and inspect` hits checked in context, were false positives: one test with prep sub-steps, or
  a marker inside the recap of the complaint. Markers: `and measure` 49, `next` 22, `then` 14, `and inspect` 13, `and scan`
  2, `and check` 1, `is bad` 1. Examples: "hold the engine at 2,000 rpm … and measure voltage at
  the posts" (one test whose snippet step has the same conjunction); "on your next hot restart"
  (not sequencing); "I can't confirm the clutch is bad" (a negated `no_fix` hit). It discards good
  replies, not bad ones, so the kept set is not harmed directly. But it costs coverage:
- **32 scenarios lost both candidates**, all `cite`: battery_drain 11, hard_start_hot 9,
  dim_lights 8, brake_pulsation 3, front_clunk 1. Their procedures read naturally as "do X and
  measure Y" (the charging voltage test, rotor inspection) or "on the next hot restart" (the
  clear-flood test). Those families and snippets are under-represented in v1, and the loss lifts
  the abstain share from 26.3% of scenarios to 28.1% of kept examples.
- `not_selected` reads as expected: the three judged at Robustness 4 hedge a little more or add a
  second observation, and the selected sibling is tighter.

## Judge and hint screen

The Opus judge passed every reply it saw: spec_adherence 1 on all 448 kept, Robustness 5 on 447
and 4 on one. No `judge_spec`, `judge_robustness` or `hint_leak_semantic` rejections occurred,
and the semantic screen passed all 458 replies. The kept set is therefore the deterministic
filter's survivors plus the selection rule. It matches a mock-judge build on 448 scenarios, with
different candidates picked in a few of them.

From the 25 reads, this agrees with the replies' quality. But the teacher and judge are the same
model family (D-080), so self-preference cannot be ruled out from this build. The R29 hand-label
audit is the check on it.

## brake_pull floating-caliper snippet

**0 kept examples cite it.** brake_pull is a held-out family: the snippet appears in
`data/scenarios/heldout.jsonl` (1 scenario, where it is gold and the user never states the
caliper type) and `rollout_pilot.jsonl`, and in no training scenario. The issue is in the eval
sets, not in v1. The nearest training analogue is item 1 above (61 kept examples).
