# First QLoRA run: Qwen3-1.7B on dataset v1

The training log of the first real QLoRA run (R6, R14), copied unchanged from the Hub model repo.

| Field | Value |
| --- | --- |
| Hub model | [`joe-alfaro/groundcite-qwen3-1.7b`](https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b) (public) |
| Hub commit | `b16b566d0d36d06744c90342fe836a4e6decc4e8` |
| Code commit | `72f24f331402c9f846e382a251744e29f49c0dab` (`git_dirty: false`) |
| Base model | `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, 4-bit |
| Dataset | `data/datasets/v1/` (448 examples, `dataset_sha256` `ae470ee1…d179`) |
| Config | `train/config.toml` version 1 (frozen; `config_sha256` `c981fb51…5943`), seed 0 |
| Hardware | Kaggle T4, fp16 |
| Steps | 168 (3 epochs) |
| Train time | 839.7 s (16 min 41 s wall) |
| Final step loss | 0.5064 |
| Mean train loss | 0.9019 |

The Hub commit holds `README.md`, `adapter_config.json`, `adapter_model.safetensors` and
`train_log.json`. `train_log.json` here is the file from that commit, fetched with:

```sh
uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('joe-alfaro/groundcite-qwen3-1.7b', 'train_log.json', revision='b16b566d0d36d06744c90342fe836a4e6decc4e8'))"
```

It records the config and dataset hashes, the seed, library versions, the chat template hash and
the per-step loss curve. The train time and mean train loss come from the run's trainer output,
not from this file.

No eval of this adapter has run yet: base-vs-tuned numbers are pending.
