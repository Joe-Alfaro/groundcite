# Architecture

Requirements: [`docs/requirements/small-model-finetune.md`](docs/requirements/small-model-finetune.md)
(R1–R32; gotcha-derived requirements are marked GOTCHA there).

## Problem

Small instruct models, and prompted frontier models too, drift off narrow contracts. They give
three steps instead of one, cite references they weren't given, cite a near-miss snippet, or give
in and name a repair when the user pushes. The goal is to teach Qwen3-1.7B one grounded-citation
contract through data alone: one next diagnostic step, one supporting `[ID]` from the provided
snippets, or a clean abstain with one question. Re-runnable numbers have to prove three things:
prompting plateaus below the bar, the tuned model beats the base model, and fewer examples than
expected might be enough. The eval harness is built first, the dataset is the main artifact, and
the training run comes last.

## Pipeline

```
 snippet library (synthetic, hand-written)        symptom families (train | reserved)
              \                                         /
               v                                       v
        +-------------------------------------------------------+
        | scenario generator (seeded, templated)  R22           |
        |   -> train pool scenarios   (train families, seed A)  |
        |   -> eval_own / ablation    (train families, seed B)  |
        |   -> heldout                (reserved families, seed C)|
        +-------------------------------------------------------+
               |  train pool                     | eval sets
               v                                 |
   +-------------------------+                   |
   | teacher distillation R4 |  privileged gold  |
   | (frontier, k cands/scn) |  hint in teacher  |
   +-------------------------+  prompt only      |
               v                                 |
   +---------------------------------------+     |
   | hard quality filter R4                |     |
   |  1 deterministic checks (R21)         |     |
   |  2 gold agreement (mode, ID)          |     |
   |  3 judge: spec=1, robustness>=4       |     |
   |  4 near-dup removal (R23 rules)       |     |
   +---------------------------------------+     |
               v                                 |
   dataset vN (chat JSONL + manifest)  --leakage check R23-->
               |                                 |
               v                                 |
   Unsloth QLoRA on Kaggle T4 (train/)           |
               v                                 |
   HF Hub (adapter + merged, commit hash) R9     |
               v                                 v
        eval.py --model base --model tuned --eval-set ...   R2 R10
          render_messages -> generate -> checks -> judge -> results + transcripts
```

The prompt-ceiling ablation (R3) uses the same `eval.py` internals, with API models and three
prompting strategies.

## Components

| Module (`src/groundcite/`) | Responsibility |
| --- | --- |
| `spec.py` | Loads `SPEC.md`, the citation and ID parsing, and `RUBRIC_VERSION` plus `rubric_sha256`. The single source for R1. |
| `schema.py` | Dataclasses and validation for snippet, scenario, chat record and transcript. Loaders for JSONL, JSON array and CSV, including prompt-only rows. Line-numbered errors (R12). |
| `prompts.py` | `render_messages(scenario, strategy)` for `zero_shot`, `few_shot` and `structured`, plus the single chat-template call with `enable_thinking=False`. Training, eval and demo all use it (R27). |
| `checks.py` | Deterministic checks (R21), plus invented-ID and abstain precision/recall metrics (R30). Pure functions returning `CheckResult(name, passed, detail)`. |
| `judge/` | `rubric.py` (prompt plus output parser), `backends.py` (`anthropic`, `openai`, `gemini`, `mock`). |
| `models/` | Generation backends: `hf_local` (transformers, optional 4-bit), `api` (anthropic/openai/gemini), `mock` (`compliant`, `naive`, scripted). |
| `scenarios/` | Snippet library loader, family registry (train vs reserved), seeded generator. |
| `distill.py`, `filter.py` | Teacher generation and the quality filter with a rejection report. |
| `dataset.py` | Build vN, manifest, nested sweep subsets. |
| `leakage.py` | Overlap checks (R23). |
| `runner.py`, `report.py` | Eval loop, automatic base rows (R10), aggregation, bootstrap intervals, markdown and JSON tables. |
| `versions.py` | Git commit and dirty flag, HF revision resolution, file hashes, library versions (R13). |
| `keys.py` | Env-only key lookup with fail-fast errors. Values are never logged (R25). |

## Judge design

- **One rubric, versioned** (`RUBRIC_VERSION`), used unchanged for the ablation, the quality
  filter and base-vs-tuned. The rubric embeds the R1 spec text and the P1–P8 checklist.
- **Inputs:** every snippet with its ID, the conversation, the reply, the deterministic check
  results and, when present, the gold mode and supporting IDs. A judge that can't see the snippets
  can't grade a citation.
- **Output (strict JSON):**
  ```json
  {"spec_adherence": 0, "robustness": 3, "failed_checks": ["P5"], "reasoning": "..."}
  ```
  - **Spec-adherence: 0/1.** The binary stranger-gradable verdict on P1–P8. A deterministic
    failure forces 0 after the judge runs, and the forced value is recorded as such.
  - **Robustness: 1–5**, anchored as defined in the requirements file (G5), and aggregated by the
  single shared `aggregate_scores()`:
    - 5: fully holds the contract against the scenario's stressor (pressure, distractor,
      ambiguity, turn drift).
    - 4: holds, with minor hedging or verbosity.
    - 3: partly gives way. Some hedged repair talk, or a weak or mismatched citation.
    - 2: mostly gives way.
    - 1: drops the contract.
- **Pinned judge:** one model id, pinned to a dated version and never an alias, at temperature 0.
  It is set in `config/models.toml` and recorded in every transcript alongside `rubric_sha256`.
  The default is an Anthropic model (`claude-sonnet-5-5`), confirmed when keys arrive.
- **Bias controls:**
  - The judge model is never itself one of the ablation subjects. The ablation table flags the
    cell that shares the judge's family.
  - The judge and the teacher share a family. R29 requires a hand audit of 20 replies. A
    second-family cross-check on 40 examples is owner-gated (it needs that key) and listed as a
    risk.
  - `scripts/check_rubric.py` asserts one rubric hash across all committed runs.
- **Backends:** `anthropic`, `openai`, `gemini` and `mock`. `mock` derives scores from the
  deterministic checks (spec = all pass; robustness = 5 if all pass, 2 with one failure, 1 with more) and is
  deterministic (R19).
- **Unparseable judge output:** one retry with a format reminder, then `judge_error` is recorded
  and the example counts as a failure. It is never silently dropped.

## File formats

**Snippet** (`data/snippets/*.jsonl`):
```json
{"id": "TSB-0417", "kind": "tsb", "system": "intake", "applies_to": "2.0L I4, 2013-2016 compact sedan",
 "title": "Lean codes at idle from cracked PCV hose", "text": "...", "family": "lean_idle"}
```
- Vehicles are described generically (body style, engine, year range, mileage). No manufacturer
  names appear.
- IDs are opaque strings drawn from ≥3 visibly synthetic schemes (for example `TSB-0417`,
  `SM 6.2.1`, `B-19-045`) and are re-assigned per scenario, so a model can't memorize an
  ID→symptom map.
- Snippet order is shuffled per scenario.

**Scenario / eval-set line** (`data/scenarios/*.jsonl`; documented for third parties in
`docs/eval-set-format.md`):
```json
{"id": "eo-0042", "snippets": [ {snippet}, ... ],
 "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}],
 "gold": {"mode": "cite", "supporting_ids": ["TSB-0417"]},
 "category": "distractor", "tags": ["pressure"], "family": "lean_idle", "split": "eval_own", "seed": 1002}
```
- Required: `id`, plus either `messages` (ending with a user turn) or `prompt`. Everything else
  is optional.
- Without `snippets`, the provided IDs come from the bracketed IDs in the prompt text.
- CSV with an `id` and a `prompt` column, and a JSON array, are also accepted (R12).

**Training chat record** (`data/datasets/vN/train.jsonl`):
```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, ..., {"role": "assistant", "content": "<target reply>"}],
 "meta": {"scenario_id": "tr-0913", "category": "pressure", "family": "misfire", "teacher": "<model id>",
          "filter_version": "f1", "judge": {"spec_adherence": 1, "robustness": 5}}}
```
It sits next to a `manifest.json` (counts, hashes, versions) and `rejections.jsonl`.

**Judge transcript** (`results/<run_id>/transcripts.jsonl`, one line per model × scenario):
```json
{"run_id": "...", "scenario_id": "eo-0042", "model": "hf:org/repo@<sha>", "strategy": "zero_shot",
 "reply": "...", "had_think": false, "checks": {"one_step": true, "no_fix": false, ...},
 "score": {"spec_adherence": 0, "robustness": 2}, "reasoning": "...",
 "judge": {"model": "...", "rubric_version": "r1", "raw": "...", "forced_by_checks": true}}
```

**Results** (`results/<run_id>/results.json` plus `results.md`):
- One row per model × strategy: `n`, `spec_adherence` (mean with 95% bootstrap CI),
  `robustness_mean`, `det_fail_rate`, and Spec-adherence by category.
- A header with the R13 pins.
- With two or more models, a delta row (tuned − base).
- Invented-ID rate and abstain precision/recall (R30).

## Repo layout

```
SPEC.md                  # canonical behavior spec (R1), loaded by path
eval.py                  # thin CLI entry -> groundcite.cli:main (R10)
pyproject.toml uv.lock
config/models.toml       # pinned judge/teacher/ablation model ids, decoding params
src/groundcite/          # package (see Components)
scripts/
  check.sh               # quality gate (R24)
  gen_scenarios.py  run_ablation.py  distill.py  build_dataset.py
  make_sweep.py  run_curve.py  check_leakage.py  check_rubric.py  judge_audit.py  demo.py
data/
  snippets/  scenarios/{ablation,eval_own,heldout}.jsonl  datasets/v1/ v2/ ...
train/
  train_qlora.py         # Unsloth QLoRA; args: --dataset --out --seed --hub-repo
  kaggle_train.ipynb     # thin wrapper: clone repo @commit, run train_qlora.py
results/                 # committed runs: ablation/, base_vs_tuned/, curve/
docs/  requirements/  eval-set-format.md  writeup.md  checklist.md
tests/                   # pytest; fixtures/ incl. hand-written third-party-style set
```

## Model backends for eval

- **`hf:<repo>[@rev]`** loads through transformers locally. It defaults to the pinned revision.
  For an adapter repo, the base comes from `adapter_config.json`, and eval.py adds base rows
  automatically: base with `zero_shot`, base with the best ablation strategy, and tuned with
  `zero_shot` (R10). On CUDA it uses 4-bit; on Apple
  Silicon it uses MPS in fp16; otherwise CPU.
  - LoRA adapters are detected and loaded on top of the base model.
  - The revision resolves to a commit sha, recorded in results.
  - Greedy decoding, `max_new_tokens=256`, the chat template with `enable_thinking=False`, and any
    `<think>` blocks stripped (R27).
- **`anthropic:` / `openai:` / `gemini:`** are API models for the ablation, at temperature 0. The
  three strategies come from `prompts.py`.
- **`mock:compliant` / `mock:naive`** are scripted, deterministic backends for tests and dry
  pipelines (R26).

## Prompt-ceiling ablation runner

`scripts/run_ablation.py --models anthropic:<m1> openai:<m2> --strategies zero_shot few_shot structured --eval-set data/scenarios/ablation.jsonl`

- It runs a grid of models × strategies × scenarios (≥2 × 3 × ≥40) through `runner.py` with the
  pinned judge.
- Strategies:
  - `zero_shot`: the spec as a system prompt.
  - `few_shot`: the spec plus 4 fixed examples (one each for clean, abstain, pressure and
    distractor), drawn from scenarios outside every eval set.
  - `structured`: the spec as explicit numbered rules, a private pre-answer checklist, and an
    instruction to output only the final reply.
- `--dry-run` prints the grid and the call counts. `--limit k` runs one point for reproduction
  (R14). Responses are cached on disk by (model, strategy, scenario, prompt hash), so a crash can
  resume without re-paying.
- Output:
  - `results/ablation/<run_id>/`: the table, transcripts and per-category breakdown.
  - The plateau check (R3), computed and printed.

## Data-efficiency sweep

- `scripts/make_sweep.py --pool data/datasets/vK/train.jsonl --sizes N N/2 N/4 N/8 --seed 7`
  shuffles the pool once with a fixed seed. Subsets are prefixes, nested and stratified by
  category, so each smaller set is contained in the larger.
- Log spacing (powers of 2) covers an order of magnitude with 4 runs. That suits the budget
  (~30 free T4 hours a week) and the expected saturating curve.
- If N/8 already saturates, add N/16. If N doesn't saturate, extend the pool.
- Every point uses identical hyperparameters, seed and epochs (3), and only the dataset path
  changes. Holding epochs fixed means smaller N gets fewer optimizer steps. That is the intended
  reading of "performance vs dataset size", and the write-up states it.
- Each checkpoint is pushed to the HF Hub as its own revision with a recorded SHA.
  `scripts/run_curve.py --eval-set <path>` re-evaluates every checkpoint on any set, including a
  third-party one.
  Each point logs its config, dataset hash, loss curve and wall time to
  `results/curve/<N>/train_log.json` (R14).
- Every checkpoint is evaluated on `eval_own` and `heldout`. The minimum viable N rule is in R8.

## Leakage controls

- **Families:** the symptom families are split. Roughly 70% are training families (train pool and
  `eval_own`) and roughly 30% are **reserved** (`heldout` only). This estimates generalization to
  an unseen third-party set.
- **Seeds:** the train pool, `eval_own` and `heldout` each use disjoint seed ranges. `eval_own`
  shares families with training but not specific scenarios.
- **Hash and overlap checks (R23):** scenario ids, final user message SHA-256, 5-gram Jaccard, and
  snippet text hashes for reserved families. IDs are re-assigned per scenario, so ID overlap alone
  is not a signal. Few-shot examples and the training pool are
  also checked against every eval set.
- The gate runs the check on every committed dataset, so a leak fails the build.

## Training

- Unsloth QLoRA, Qwen3-1.7B, 4-bit, LoRA r=16 on attention and MLP projections.
- `fp16=True, bf16=False`. The T4 (sm_75) has no bf16.
- `max_seq_length` covers the p99 tokenized example, and the value is logged.
- Loss on the assistant turn only, with EOS-terminated targets, and the chat template from
  `prompts.py` with thinking disabled.
- A GPU-hour budget table is kept in `docs/checklist.md` (R31).
- One config file, frozen at v1. Bad results are fixed by changing the data (R7), never the
  hyperparameters.
- A Kaggle notebook clones the repo at a pinned commit and runs `train/train_qlora.py`. It pushes
  the adapter to the HF Hub when `HF_TOKEN` is set (from Kaggle Secrets), with no notebook
  output showing it.

## Requirement → component map

| Req | Components |
| --- | --- |
| R1 | `SPEC.md`, `spec.py`. The judge rubric, teacher prompt and filter load it by path. |
| R2 | `runner.py`, `checks.py`, `judge/`, `report.py` |
| R3 | `scripts/run_ablation.py`, `prompts.py`, `models/api`, `report.py` |
| R4 | `distill.py`, `filter.py`, `judge/`, `checks.py` |
| R5 | `scripts/` chain plus `train/train_qlora.py` (`--max-steps`) |
| R6, R7 | `dataset.py`, `build_dataset.py`, `train/`, `results/`, diagnosis note in `docs/` |
| R8 | `make_sweep.py`, `run_curve.py`, `train/`, `report.py` (bootstrap CI, plot) |
| R9, R15 | `train_qlora.py --hub-repo`, dataset card, HF Hub |
| R10, R11 | `eval.py`, `cli.py`, `runner.py`, `report.py` |
| R12 | `schema.py`, `docs/eval-set-format.md`, `tests/fixtures/third_party.jsonl` |
| R13 | `versions.py`, `scripts/check_rubric.py`, `uv.lock` |
| R14 | `run_ablation.py --limit`, `results/curve/*/train_log.json` |
| R16 | `scripts/demo.py`, public notebook |
| R17, R18 | `docs/writeup.md`, video |
| R19 | `judge/backends.py` (mock) |
| R20 | `cli.py`, `run_ablation.py`, `distill.py` (`--dry-run`) |
| R21 | `checks.py` |
| R22 | `scenarios/`, `gen_scenarios.py`, `data/scenarios/` |
| R23 | `leakage.py`, `check_leakage.py`, `check.sh` |
| R24 | `scripts/check.sh` (incl. scrub + secret scan) |
| R25 | `keys.py` |
| R26 | `models/mock.py` |
| R27 | `prompts.py`, `models/hf_local.py`, `train/train_qlora.py` |
| R28 | `runner.py` (decoding), README tolerance |
| R29 | `scripts/judge_audit.py` |
| R30 | `checks.py`, `report.py`, dataset manifest |
| R31 | `train/train_qlora.py`, `docs/checklist.md` |
| R32 | `docs/checklist.md` |

## Test strategy

- **Unit (pytest):**
  - Every deterministic check, with pass and fail fixtures for the R1 worked examples.
  - Schema validation, including the line-numbered error.
  - Judge output parsing: valid, malformed and retry.
  - Prompt rendering, plus the R27 train-equals-eval assertion and no `<think>` in transcripts.
  - Leakage rules on small synthetic pairs.
  - Bootstrap CI, sweep nesting and stratification.
- **Contract:**
  - The spec text is identical across `spec.py`, the rubric and the teacher prompt.
  - The scenario-mix quotas hold for the committed sets.
  - The generator is byte-identical for a given seed.
  - Three third-party fixture shapes run: our format, prompt-only, and an unfamiliar ID scheme.
  - `SPEC.md` is loaded by path everywhere, and the requirements quote matches it.
- **End to end (no network):**
  - `eval.py` on `mock:compliant` and `mock:naive` with `--judge mock` and sockets blocked. It
    asserts the output files, the transcript fields and that `compliant` beats `naive` (R19, R26).
  - `--dry-run` for all three CLIs.
  - The missing-key exit code and that no key leaks (R25).
- **Not unit-tested:** real API calls and GPU training. They're verified by committed run
  artifacts and the smoke loop (R5).
- **Gate:** `scripts/check.sh` (R24) must pass before any merge.

## Risks

| Risk | Mitigation |
| --- | --- |
| **No plateau.** Frontier models plus `structured` pass ≥90%. | The scenario mix already weights `no_snippet`, `distractor`, `pressure`, `multi_symptom` and `multi_turn`, and P6 bans even conditional repairs. If R3 still shows no plateau, the spec gets hardened before training: longer multi-turn drift (6–10 turns), more near-miss snippets, and stronger pressure. The change is logged and reported. Nothing gets trained on an unproven edge. |
| **T4 limits** (16 GB, ~30 h a week, notebook sessions) | 1.7B in 4-bit fits easily. The sweep is nested and small. 0.6B is the fallback point. Runs need the owner's Kaggle account. |
| **Judge bias** (self-preference, leniency) | Deterministic checks override the judge and cap Robustness at 2. A second-family cross-check is owner-gated (needs that key). The judge is a different model from the ablation subjects. The rubric is fixed and versioned, temperature 0. Spot-audit 20 transcripts by hand against P1–P8 and report the agreement rate. |
| **No API keys yet** | Everything in M0 is key-free (mock backends, `--dry-run`, committed scenario sets). M1 and later stages wait for keys. |
| **Heuristic checks** (`one_step`, `no_fix`) misfire | Versioned lexicons with fixtures. False positives are reviewed in the filter report. A check that disagrees with a hand label on more than 5% of a 50-item audit sample gets revised, with a version bump. |
| **Synthetic snippets read as unrealistic** | A hand-written library across ~16 families in two styles (bulletin and manual). Never real text. |
| **Qwen3 thinking output and template mismatch** | One renderer, `enable_thinking=False`, a test that train and eval strings are equal, and a `no_think` check (R27). |
| **Publishing needed by Milestone 1** (draft public checkpoint for pinned numbers) | It needs the owner's approval and HF account. Ask early. Until then, pins are recorded and the push is a single command. |
| **Third-party held-out format unknown** | A tolerant loader (JSONL, JSON, CSV, prompt-only, any ID scheme) with three fixture shapes (R12). |

## Deployment

There is no service. The deployment is the public HF Hub model (adapter plus merged weights, with
its commit hash) and dataset, plus the local inference demo (`scripts/demo.py`), base vs tuned.
