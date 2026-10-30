# Data-efficiency curve: QLoRA runs on dataset v1 subsets

The training logs of the three curve points (R14), copied unchanged from the runs. Each point
trains Qwen3-1.7B on one stratified subset of dataset v1 from `data/datasets/v1/subsets/`
(`sweep_manifest.json`, subset seed 7) with the same frozen config, training seed and code commit.
The full v1 run (trained from code commit `72f24f3`) is in the last row for reference
([`results/training/v1/`](../v1/README.md)).

| N | Steps | Final step loss | `dataset_sha256` | `config_sha256` | Hub adapter |
| --- | --- | --- | --- | --- | --- |
| 56 | 21 | 1.3040 | `ba8ebc90c8f4…b993` | `c981fb5196d2…5943` | `joe-alfaro/groundcite-qwen3-1.7b-n56@51ae124` (private) |
| 112 | 42 | 1.0274 | `1443387f1a21…57f1` | `c981fb5196d2…5943` | `joe-alfaro/groundcite-qwen3-1.7b-n112@c049270` (private) |
| 224 | 84 | 0.6386 | `fb4051ef3eab…2b90` | `c981fb5196d2…5943` | `joe-alfaro/groundcite-qwen3-1.7b-n224@58532b3` (private) |
| 448 (full v1) | 168 | 0.5064 | `ae470ee1a6eb…d179` | `c981fb5196d2…5943` | `joe-alfaro/groundcite-qwen3-1.7b@b16b566` (public) |

Every curve log records:

- `config_sha256` equal to the v1 run's (`train/config.toml` version 1), seed 0, 3 epochs;
- `dataset_sha256` equal to the sha256 of the committed `data/datasets/v1/subsets/n<N>.jsonl`;
- 0 examples dropped as too long, so every subset example was trained;
- code commit `c0be9258c80eff7c0f6d73515a7b71797ddd7fd6` (`git_dirty: false`), base
  `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` in 4-bit on CUDA
  with fp16, and the same chat template hash and library versions as the v1 run;
- the full Hub commit SHA under `hub.commit_sha`, and the per-step loss curve.

Final step loss is the last entry of `loss_curve`. It is a training loss on a different number
of examples per point, not a quality measure.

To rerun a point, train on its subset with the same config and seed on a CUDA host (Kaggle):

```sh
uv run train/train_qlora.py --dataset data/datasets/v1/subsets/n<N>.jsonl --out runs/n<N> --seed 0
```

The held-out evals of the curve adapters are pending: no quality numbers per point yet.
