# Eval results `20261030T185720Z-a9795e`

Judge `openai:gpt-5.5-2026-04-23`, rubric sha256 `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`, eval code `9936bb4641172792903c25566d60d27d092b3868`. Spec-adherence is the mean with a 95% bootstrap CI; Robustness is the mean 1-5 judge score.

| System | Strategy | n | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|---|---|---|---|
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | zero_shot | 100 | 0.22 [0.14, 0.30] | 2.63 | 0.19 | 0.62 | 0.06 | n/a / 0.00 |
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | structured | 100 | 0.17 [0.10, 0.25] | 2.33 | 0.16 | 0.80 | 0.38 | 0.27 / 0.10 |
| hf:joe-alfaro/groundcite-qwen3-1.7b | zero_shot | 100 | 0.57 [0.47, 0.66] | 3.73 | 0.53 | 0.40 | 0.00 | 0.50 / 0.57 |

## Spec-adherence by category

| System | Strategy | multi_symptom | pressure | distractor | multi_turn | clean | no_snippet |
|---|---|---|---|---|---|---|---|
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | zero_shot | 0.33 | 0.13 | 0.20 | 0.33 | 0.40 | 0.00 |
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | structured | 0.27 | 0.20 | 0.15 | 0.20 | 0.20 | 0.05 |
| hf:joe-alfaro/groundcite-qwen3-1.7b | zero_shot | 0.60 | 0.67 | 0.35 | 0.33 | 0.80 | 0.70 |

## Pins

- Eval code: git `9936bb4641172792903c25566d60d27d092b3868`
- Judge: `openai:gpt-5.5-2026-04-23`
- Rubric sha256: `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`
- Eval set: `data/scenarios/eval_own.jsonl` (sha256 `9955cd63e8d80a96f9bfa40326fd883cce83f76d9a2a2306d15f557d0a701ea2`)
- Model `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, chat template sha256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- Model `hf:joe-alfaro/groundcite-qwen3-1.7b` revision `b16b566d0d36d06744c90342fe836a4e6decc4e8`, chat template sha256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- Decoding `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`: `{"do_sample": false, "max_new_tokens": 256}`
- Decoding `hf:joe-alfaro/groundcite-qwen3-1.7b`: `{"do_sample": false, "max_new_tokens": 256}`
- Decoding `openai:gpt-5.5-2026-04-23`: `{"max_completion_tokens": 16000, "reasoning_effort": "low"}`
- Libraries: python 3.13.2, groundcite 0.1.0, anthropic 2.3.0, openai 3.40.0, python-dotenv 1.2.4, torch 2.14.1, transformers 5.21.0, peft 0.21.2
