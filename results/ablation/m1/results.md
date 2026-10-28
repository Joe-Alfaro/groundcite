# Prompt-ceiling ablation `m1`

40 scenarios, the same IDs in every cell. Judge `openai:gpt-5.5-2026-04-23`, rubric sha256 `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`, eval code `3c749fe71bf4f475f9da0b1fb47a65a7fdd5a9fe`. Spec-adherence is the mean with a 95% bootstrap CI; Robustness is the mean 1-5 judge score. Errors and refusals are scored as failures.

| System | Strategy | n | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R | Shares judge family |
|---|---|---|---|---|---|---|---|---|---|
| anthropic:claude-opus-5-5 | zero_shot | 40 | 0.78 [0.65, 0.90] | 4.30 | 0.82 | 0.15 | 0.00 | 0.82 / 0.82 | no |
| anthropic:claude-opus-5-5 | few_shot | 40 | 0.90 [0.80, 0.97] | 4.72 | 0.88 | 0.07 | 0.00 | 0.91 / 0.91 | no |
| anthropic:claude-opus-5-5 | structured | 40 | 0.97 [0.93, 1.00] | 4.88 | 0.97 | 0.03 | 0.00 | 0.92 / 1.00 | no |
| openai:gpt-5.5-2026-04-23 | zero_shot | 40 | 0.85 [0.72, 0.95] | 4.58 | 0.85 | 0.12 | 0.00 | 0.88 / 0.64 | yes |
| openai:gpt-5.5-2026-04-23 | few_shot | 40 | 0.82 [0.70, 0.93] | 4.47 | 0.79 | 0.17 | 0.00 | 1.00 / 0.45 | yes |
| openai:gpt-5.5-2026-04-23 | structured | 40 | 0.90 [0.80, 0.97] | 4.70 | 0.88 | 0.10 | 0.00 | 0.90 / 0.82 | yes |

A cell marked as sharing the judge family was graded by a model from its own vendor; read it with that bias in mind (GOTCHA G12).

## Spec-adherence by category

| System | Strategy | multi_symptom | pressure | distractor | multi_turn | no_snippet | clean |
|---|---|---|---|---|---|---|---|
| anthropic:claude-opus-5-5 | zero_shot | 1.00 | 0.83 | 0.88 | 0.83 | 0.62 | 0.50 |
| anthropic:claude-opus-5-5 | few_shot | 0.67 | 0.83 | 1.00 | 1.00 | 0.88 | 1.00 |
| anthropic:claude-opus-5-5 | structured | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| openai:gpt-5.5-2026-04-23 | zero_shot | 0.83 | 0.83 | 1.00 | 0.83 | 0.75 | 0.83 |
| openai:gpt-5.5-2026-04-23 | few_shot | 0.83 | 0.83 | 1.00 | 0.67 | 0.62 | 1.00 |
| openai:gpt-5.5-2026-04-23 | structured | 1.00 | 0.83 | 0.88 | 1.00 | 0.75 | 1.00 |

## Best vs second-best strategy per model

| Model | Best | Second | Delta (Spec-adherence) |
|---|---|---|---|
| anthropic:claude-opus-5-5 | structured | few_shot | 0.07 |
| openai:gpt-5.5-2026-04-23 | structured | zero_shot | 0.05 |

## Plateau check

Reliability bar: Spec-adherence >= 0.90 and mean Robustness >= 4.3 (inclusive). Verdict: **no plateau**: anthropic:claude-opus-5-5 x few_shot, anthropic:claude-opus-5-5 x structured, openai:gpt-5.5-2026-04-23 x structured meet(s) the reliability bar.

## Pins

- Eval code: git `3c749fe71bf4f475f9da0b1fb47a65a7fdd5a9fe`
- Judge: `openai:gpt-5.5-2026-04-23`
- Rubric sha256: `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`
- Eval set: `data/scenarios/ablation.jsonl` (sha256 `4a5e8b50ec425828cb5bec5f43f6c41596cddad192c8bf4867de48eae3e8233b`)
- Model `anthropic:claude-opus-5-5` revision `None`
- Model `openai:gpt-5.5-2026-04-23` revision `None`
- Decoding `anthropic:claude-opus-5-5`: `{"max_tokens": 16000, "output_config": {"effort": "medium"}}`
- Decoding `openai:gpt-5.5-2026-04-23`: `{"max_completion_tokens": 16000, "reasoning_effort": "low"}`
- Libraries: python 3.13.2, groundcite 0.1.0, anthropic 2.3.0, openai 3.40.0, python-dotenv 1.2.4, torch 2.14.1, transformers 5.21.0, peft 0.21.2
