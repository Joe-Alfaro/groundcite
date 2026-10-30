# Dataset v2 plan: in-scope cite twins of the v1 abstain scenarios

Why: the held-out diagnosis (`results/base_vs_tuned/heldout/DIAGNOSIS.md`) found that the most
frequent tuned failure is a false abstain with an invented scope exclusion. In v1, every abstain
example rejects an on-topic snippet for its scope, and no example cites that same snippet for a
vehicle inside its scope. v2 adds those minimal pairs. It changes the data only.

## What v2 is

dataset v2 = dataset v1 (448 examples, unchanged) + one kept example per in-scope twin.

- **The twins:** `data/scenarios/train_v2_additions.jsonl`, 105 scenarios, written by
  `uv run scripts/gen_v2_additions.py`. The defaults (`--seed 2027`) reproduce the committed
  file byte for byte. Each row is a training scenario plus `source_id`, the v1 abstain scenario
  in `data/scenarios/train_pool.jsonl` it is the twin of.
- **A twin** keeps its source's family, its set of snippet texts and its conversation shape. The
  vehicle and/or model year moves, so exactly one provided on-topic snippet now applies: the
  gold, cited alone, and the same snippet text the source rejected. Every other provided
  near-miss stays ruled out by its own scope change, and a provided twin of the gold is listed as
  a near-miss. Messages, snippet IDs and snippet order are regenerated from the twin's own seed,
  so no final user message is a near-duplicate of an eval final, a training-pool final or another
  twin's final.
- **Categories:** a `no_snippet` source becomes a `distractor` twin (92), and `pressure` (8) and
  `multi_symptom` (5) keep their category.

### Coverage: 105 of the 126 v1 abstain scenarios

The 21 sources without a twin, and why:

| Sources | Count | Why no twin |
|---|---|---|
| multi_turn: tr-0048, tr-0099, tr-0113, tr-0137, tr-0143, tr-0157, tr-0220, tr-0268, tr-0301, tr-0356, tr-0357, tr-0392, tr-0467, tr-0473 | 14 | The earlier turns already cite the chain steps, and a near-miss is never the next chain step, so no provided snippet can be the next unchecked step. |
| tr-0025, tr-0153, tr-0367, tr-0394: only condition near-misses on topic | 4 | A condition near-miss is ruled out by the complaint for every vehicle. |
| tr-0058, tr-0205: only other-family snippets could apply | 2 | The only chain snippets come from a related family's "close" list, which is the wrong procedure for the complaint. |
| tr-0064: two near-misses with the same scope change | 1 | Both near-misses change the same year range, so whenever one applies the other can't be ruled out. |

## Teacher step (paid, on hold)

The twins are new scenarios, so no existing candidate covers them. At v1's two candidates per
scenario:

    uv run scripts/distill.py --scenarios data/scenarios/train_v2_additions.jsonl \
        --k 2 --out data/distill/v2_additions --dry-run

plans **210 teacher calls** (105 scenarios, k=2) and makes no request. The filter then adds up
to about 2 judge calls per candidate that passes the deterministic checks, as in v1. Running it
without `--dry-run` needs owner approval for the spend.

`scripts/build_v2.sh` runs this whole plan. By default it is a dry run: it prints the plan and
the teacher call count and writes nothing. After approval,
`scripts/build_v2.sh --approved --filter opus|gpt-5.5` runs the teacher step (stopping if it
plans more than 210 calls), the filter with the chosen judge into `data/datasets/v2/additions/`,
`scripts/assemble_dataset.py` (v1 bytes unchanged, then the kept twins, with a summed manifest),
the diff below, the leakage, secret and scrub checks, and a commit. Training v2 then needs only
two notebook parameter edits (`DATASET = "data/datasets/v2/train.jsonl"` and a `HUB_REPO` ending
in `-v2`), and the notebook stops unless the config still hashes to v1's.

## Build and compare

After the teacher step, the candidates go through the same filter (`scripts/build_dataset.py`,
same filter version and selection rule). The kept twin examples are appended to the 448 v1
lines to form `data/datasets/v2/train.jsonl`. Then:

    uv run scripts/diff_datasets.py data/datasets/v1 data/datasets/v2

must report only added scenarios (the kept twins), with nothing removed or relabeled and all 448
v1 examples unchanged.

- v1 `train.jsonl` sha256: `ae470ee1a6eb771bae3e7f1b4dcaa5764e457e101b4023ad101711d7afcbd179`
- Abstain share: 126 / (448 + kept). With all 105 kept, it is 22.8%. At a 50% keep rate (52
  kept), it is 25.2%. Both are inside the 20-30% band.

## Training config: frozen

v2 changes the data only (G21). The training config still resolves to the v1 run's hash,
`c981fb5196d25647ea69c6ee091cfb92a51fc73b6de0aa05576a2618aaf55943`
(`results/training/v1/train_log.json`). `tests/test_v2_additions.py` guards it.

## Gates the twins pass

- Scenario gates: gold consistency, category vs mode, 2-5 snippets with valid unique IDs, and
  alternating roles.
- Near-miss scope gates: every near-miss is ruled out by its scope change, the facts that decide
  the gold are stated, and so is the transmission.
- None of the 15 trivial heuristics, the median-year pair rule or the surface probe solves the
  twins, or train_pool plus the twins. A probe fit on that pool doesn't transfer to the eval
  distractors, and the gold kind stays balanced.
- Leakage: `scripts/check_leakage.py` passes against the three eval sets and the few-shot file.
