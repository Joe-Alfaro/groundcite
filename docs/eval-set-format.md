# Eval-set format

The harness accepts a held-out eval set as **JSONL**, a **JSON array** or **CSV**. The format is
chosen from the file extension (`.jsonl`, `.json`, `.csv`). The loader is
`groundcite.evalset.load_eval_set`.

## Rows

Each row is one scenario: a conversation that ends with a user turn, which the model answers.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique, non-empty string. |
| `messages` | one of `messages` / `prompt` | List of `{"role", "content"}` turns. Roles are `system`, `user` or `assistant`. The last turn must be `user`. |
| `prompt` | one of `messages` / `prompt` | A plain string. It becomes a single user turn. |
| `snippets` | no | List of reference snippets, each with an `id` and a `text`. Other snippet fields are kept but not required. |
| `gold` | no | `{"mode": "cite" \| "abstain", "supporting_ids": [...]}`. |
| `category` | no | Scenario category, for example `distractor` or `no_snippet`. |

Any other fields are ignored.

### Snippet IDs

- An ID is 2–24 characters from `A-Z a-z 0-9 . _ - space`, with no brackets, for example
  `TSB-0417`, `SM 6.2.1` or `ref_q7.22`. A snippet whose ID is outside this grammar is rejected,
  because a citation of it could never be parsed.
- A reply cites a snippet by writing its ID in square brackets: `[TSB-0417]`.
- The **provided IDs** of a row are the `id`s of its `snippets`, in order. A row without
  `snippets` (a **prompt-only** row) takes its provided IDs from the bracketed IDs in its user and
  system text. IDs that appear only in assistant turns are never provided.
- Because a prompt-only row's provided IDs can include bracketed trouble codes or UI words, an
  unbracketed mention of a provided ID in a reply is only a SOFT `citation_format` flag on
  prompt-only rows; on rows with `snippets` it stays HARD.

### Gold

- `cite` needs at least one `supporting_ids` entry; `abstain` takes none (an empty list or no
  key).
- Every supporting ID must be one of the row's provided IDs.
- Without `gold`, checks that need it (`cited_id_supports`, `mode_matches_gold`) are reported as
  `n/a` and the judge decides the mode.

## Minimal examples

JSONL with structured snippets and gold (one row per line):

```json
{"id": "ex-1", "snippets": [{"id": "ref_q9.04", "text": "Coolant temperature sensor: compare the scan-tool reading with an infrared reading at the thermostat housing."}, {"id": "ref_q7.22", "text": "Cooling fan relay: probe the relay control circuit for a ground command."}], "messages": [{"role": "user", "content": "Temperature gauge reads cold on the highway and the heater blows lukewarm."}], "gold": {"mode": "cite", "supporting_ids": ["ref_q9.04"]}}
```

JSON array of prompt-only rows, gold optional:

```json
[
  {
    "id": "ex-2",
    "prompt": "References:\n[SM-3105] Alternator output test.\n[SM-2210] Fuel pressure test.\n\nComplaint: battery light flickers at idle.",
    "gold": {"mode": "cite", "supporting_ids": ["SM-3105"]}
  },
  {"id": "ex-3", "prompt": "Notes: [ref_q3.11] Wiper motor park switch test.\nThe car pulls right under braking."}
]
```

CSV with an `id` and a `prompt` column (quoted fields may span lines; an optional `category`
column is read too; gold needs JSONL or JSON):

```csv
id,prompt
ex-4,"References: [B-19-045] Rotor thickness variation check. Complaint: the wheel shakes when braking from highway speed."
```

## Errors

A malformed set fails on the first bad row with an `EvalSetError` (a `ValueError`) that names the
location and field, for example `line 3: messages: the last turn must be a user turn`.

- JSONL and CSV report the physical file line (blank lines count; the CSV header is line 1; a
  multi-line CSV record reports the line it starts on).
- A JSON array reports the 1-based item index, for example `item 2: messages: ...`.
- Duplicate `id`s are reported after every row has validated.
