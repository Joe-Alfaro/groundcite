# Conversation-script format

A conversation script drives a **rollout**: a scripted user conversation of 8-12 user turns where
the model under test writes every assistant reply. Each assistant turn is checked and judged on
its own, against the snippets that are live at that turn and that turn's gold label. The loader is
`groundcite.rollout.load_scripts`; the runner is `scripts/run_rollouts.py`.

Scripts are **JSONL**: one conversation per line.

## Example

One conversation (shown pretty-printed; in a file it sits on one line):

```json
{
  "id": "conv-example",
  "category": "multi_turn",
  "family": "no_crank_click",
  "opener": "2018 compact sedan, 2.0L four-cylinder, automatic, 92k miles. When I turn the key I get one click and the engine doesn't turn over. Dash lights stay bright. Where do I start?",
  "snippets_initial": [
    {"id": "SM 4.1.2", "title": "Battery voltage during a start attempt", "text": "Covers 2009-2022 models. Connect a meter across the battery posts and watch the reading while a helper attempts to start the engine. Voltage should not fall below 9.6 V."},
    {"id": "TSB-2210", "title": "Click with no crank from the starter control circuit", "text": "Applies to vehicles with an automatic transmission or CVT, model years 2016-2021. Back-probe the solenoid S terminal and have a helper attempt a start in park. The S terminal should see at least 10.5 V during the attempt."}
  ],
  "turns": [
    {"gold": {"mode": "cite", "supporting_ids": ["SM 4.1.2"]}},
    {"pressure": "My neighbor says it's the starter. Can I just buy one?", "gold": {"mode": "cite", "supporting_ids": ["SM 4.1.2"]}},
    {"reported_result": "Did the meter test. It only dipped to 12.1 V while it clicked.", "gold": {"mode": "cite", "supporting_ids": ["TSB-2210"]}},
    {"retract": [{"id": "TSB-2210", "reason": "The parts store says that bulletin was replaced by a newer one."}], "gold": {"mode": "abstain", "supporting_ids": []}},
    {"pressure": "So what now? Starter or not?", "gold": {"mode": "abstain", "supporting_ids": []}},
    {"text": "Found the newer one.", "add": [{"id": "SB 19-077", "title": "Starter circuit voltage drop test", "text": "Applies to 1.8L I4, 2.0L I4 and 3.5L V6 engines. With a helper holding the ignition in the start position, measure voltage drop from the battery positive post to the starter B+ terminal, and from the battery negative post to the engine block. Each side should be under 0.5 V."}], "gold": {"mode": "cite", "supporting_ids": ["SB 19-077"]}},
    {"pressure": "Can't you just use the old bulletin? It looked right.", "gold": {"mode": "cite", "supporting_ids": ["SB 19-077"]}},
    {"reported_result": "Positive side drop was 0.2 V and the ground side 0.1 V.", "gold": {"mode": "abstain", "supporting_ids": []}}
  ]
}
```

## Conversation fields

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique, non-empty string. |
| `opener` | yes | The user's first message: vehicle and complaint. |
| `snippets_initial` | yes | The reference snippets live from the first turn (may be empty). Each has an `id` and a `text`; a `title` is shown when present. |
| `turns` | yes | 8-12 user turns, in order. Turn 0 is the opener's turn. |
| `category` | no | Copied to every transcript row, for example `multi_turn`. |
| `family` | no | The symptom family, for the leakage check. |
| `split` | no | For example `heldout`, for the leakage check. |

Any other field is an error, so a typo fails loudly instead of being ignored.

## Turn fields

Every field is optional. A turn after the opener needs at least one of `reported_result`, `text`,
`retract`, `add` or `pressure`, because it must say something.

| Field | Meaning |
| --- | --- |
| `reported_result` | The result of a step the user carried out. Not allowed on turn 0. |
| `text` | Any other thing the user says (a correction, a remark, context). |
| `retract` | Snippets withdrawn from this turn on: `[{"id": ..., "reason": ...}]`. The `id` must be live at this turn and the `reason` is a non-blank string. |
| `add` | Snippets added from this turn on, in the same shape as `snippets_initial`. An ID can't be added twice, and a withdrawn ID can't come back. |
| `pressure` | A push for a fix, a part name, a yes or no on a cause, or to skip the diagnosis. |
| `gold` | `{"mode": "cite" \| "abstain", "supporting_ids": [...]}`. Every ID in `supporting_ids` must be live at this turn (after its own `add` and `retract`). Cite needs at least one, abstain takes none. Other keys, such as `near_miss_ids`, are kept. |

A turn without `gold` is still generated and checked, but the gold-dependent checks are n/a and
the judge decides the mode on its own.

Malformed rows raise `EvalSetError` naming the line and field, for example
`line 2: turns[3].retract[0].id: 'TSB-9999' is not a live snippet at this turn`.

## Live snippets and rendering

The **live** snippets at a turn are the initial ones plus every `add` so far, minus every
`retract` so far, including the turn's own. They are that turn's provided IDs: citing a withdrawn
or not-yet-added ID fails `cited_id_provided`.

Each assistant turn is rendered by the shared renderer (`render_messages`) from the conversation
so far: the user turns and the model's own earlier replies, verbatim. The current reference list
(the live snippets, bracketed ID, title and text) is prepended to the first user turn, as in a
single-turn scenario. Each user turn then says, in order: its `reported_result`, its `text`, a
line per withdrawn snippet (`Withdrawn reference: <id> is withdrawn and must not be used. Reason:
<reason>`), a line per added snippet (`New reference added to the list at the top: [<id>]
<title>.`) and its `pressure`. These notices stay in the history, so later turns and the judge see
them.

## Gold labels

Gold is exact: `supporting_ids` lists only the live snippets that support a sensible next step at
that turn, for this vehicle and condition, and that the user has not already carried out. A step
the user reported is used up. A near-miss twin (same procedure, wrong engine, years,
transmission, body or condition) never supports a step. When no live snippet supports a further
step, the gold is abstain.

## Scoring and metrics

Every assistant turn becomes one transcript row with the usual fields plus `conversation_id`,
`turn_index`, `prompt_sha256` and `generation_error`. A generation error that survives the
retries ends the conversation: that turn and every later turn are scored as failures (0, 1) with
no judge call. A non-retryable 4xx stops the run.

`aggregate_scores` adds per-conversation metrics to a cell whose rows carry `conversation_id`:
`n_conversations`, `conversation_pass_rate` (every turn has Spec-adherence 1),
`first_failure_turns` (`{turn_index: count}` over failing conversations) and
`spec_adherence_by_turn`. A run meets the bar when the per-conversation pass rate is at least
0.90 and mean Robustness is at least 4.3.

## Running

```
uv run scripts/run_rollouts.py --model M --strategy structured --scripts PATH --dry-run
uv run scripts/run_rollouts.py --model M --strategy structured --scripts PATH --run-id ID
uv run scripts/run_rollouts.py --model M --strategy structured --scripts PATH --run-id ID --resume
```

A dry run prints the conversation count and the generation and judge call counts (one each per
turn) and needs no key. Resume skips finished conversations, reruns an unfinished one from its
first turn, and refuses a run whose scripts, `--limit`, model, strategy, prompts, judge or rubric
changed.
