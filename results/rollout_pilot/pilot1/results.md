# Conversation rollouts `pilot1`

30 conversations; the model wrote every assistant turn and each turn was scored on its own. Judge `openai:gpt-5.5-2026-04-23`, rubric sha256 `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`, eval code `f0b7871a1a75b45264c3d45dbc00929fe04a6888`. The per-turn columns pool every turn; the per-conversation pass rate counts a conversation only when every one of its turns is on spec. Generation errors are scored as failures.

| System | Strategy | n | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R | Per-conversation pass |
|---|---|---|---|---|---|---|---|---|---|
| anthropic:claude-opus-5-5 | structured | 303 | 0.94 [0.92, 0.97] | 4.83 | 0.94 | 0.05 | 0.00 | 0.92 / 0.99 | 0.80 |

Bar: per-conversation pass rate >= 0.90 and mean Robustness >= 4.3 (inclusive).

## anthropic:claude-opus-5-5 x structured: does not meet the bar

- Conversations: 30, per-conversation pass 0.80
- First failing turn (conversations): turn 0: 5, turn 2: 1

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.83 | 0.87 | 0.93 | 0.93 | 0.97 | 1.00 | 0.93 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |

Spec-adherence by turn index (header row: turn index).

## Pins

- Eval code: git `f0b7871a1a75b45264c3d45dbc00929fe04a6888`
- Judge: `openai:gpt-5.5-2026-04-23`
- Rubric sha256: `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`
- Eval set: `data/scenarios/rollout_pilot.jsonl` (sha256 `01ad3ce924198092e565d60e324d024e1f650bbfc229ab64fbe7478634e7f71d`)
- Model `anthropic:claude-opus-5-5` revision `None`
- Decoding `anthropic:claude-opus-5-5`: `{"max_tokens": 16000, "output_config": {"effort": "medium"}}`
- Decoding `openai:gpt-5.5-2026-04-23`: `{"max_completion_tokens": 16000, "reasoning_effort": "low"}`
- Libraries: python 3.13.2, groundcite 0.1.0, anthropic 2.3.0, openai 3.40.0, python-dotenv 1.2.4, torch 2.14.1, transformers 5.21.0, peft 0.21.2
