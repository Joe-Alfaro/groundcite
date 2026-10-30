# Eval results `20261030T171626Z-e1bd6f`

Judge `openai:gpt-5.5-2026-04-23`, rubric sha256 `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`, eval code `b17f80bdbfdfc05c1346187242156d56b0a7d233`. Spec-adherence is the mean with a 95% bootstrap CI; Robustness is the mean 1-5 judge score.

| System | Strategy | n | Spec-adherence (95% CI) | Robustness | Stressor spec | HARD-check fail | Invented-ID rate | Abstain P / R |
|---|---|---|---|---|---|---|---|---|
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | zero_shot | 60 | 0.28 [0.18, 0.40] | 2.97 | 0.22 | 0.53 | 0.00 | 0.00 / 0.00 |
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | structured | 60 | 0.17 [0.08, 0.27] | 2.35 | 0.18 | 0.80 | 0.40 | 0.50 / 0.11 |
| hf:joe-alfaro/groundcite-qwen3-1.7b | zero_shot | 60 | 0.48 [0.35, 0.60] | 3.47 | 0.45 | 0.50 | 0.02 | 0.44 / 0.58 |

## Spec-adherence by category

| System | Strategy | clean | distractor | multi_turn | no_snippet | multi_symptom | pressure |
|---|---|---|---|---|---|---|---|
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | zero_shot | 0.67 | 0.33 | 0.22 | 0.00 | 0.56 | 0.00 |
| hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | structured | 0.11 | 0.25 | 0.22 | 0.00 | 0.44 | 0.00 |
| hf:joe-alfaro/groundcite-qwen3-1.7b | zero_shot | 0.67 | 0.33 | 0.33 | 0.42 | 0.44 | 0.78 |

## Pins

- Eval code: git `b17f80bdbfdfc05c1346187242156d56b0a7d233`
- Judge: `openai:gpt-5.5-2026-04-23`
- Rubric sha256: `636a9c79b9ba0ad02b082351bd25e385c0963c9d8edffedad3919e56b556b4bf`
- Eval set: `data/scenarios/heldout.jsonl` (sha256 `13dae80532ab13f3d8e0290374e1fd1abb81df05a23b36e749ca347ae4b3c564`)
- Model `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, chat template sha256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- Model `hf:joe-alfaro/groundcite-qwen3-1.7b` revision `b16b566d0d36d06744c90342fe836a4e6decc4e8`, chat template sha256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- Decoding `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`: `{"do_sample": false, "max_new_tokens": 256}`
- Decoding `hf:joe-alfaro/groundcite-qwen3-1.7b`: `{"do_sample": false, "max_new_tokens": 256}`
- Decoding `openai:gpt-5.5-2026-04-23`: `{"max_completion_tokens": 16000, "reasoning_effort": "low"}`
- Libraries: python 3.13.2, groundcite 0.1.0, anthropic 2.3.0, openai 3.40.0, python-dotenv 1.2.4, torch 2.14.1, transformers 5.21.0, peft 0.21.2
