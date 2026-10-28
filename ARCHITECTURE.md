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
| `judge.py`, `rubrics/` | The versioned rubric (built from `SPEC.md`), judge input, output parser, retry and HARD-check forcing. |
| `backends.py`, `config.py`, `keys.py` | One `Backend` registry for subjects and the judge: `hf:` (transformers, loaded lazily), `anthropic:`, `openai:` and `mock:` (`compliant`, `naive`, `judge`). Pinned model ids and decoding params from `config/models.toml`. |
| `scenarios/` | Snippet library loader, family registry (train vs reserved), seeded generator. |
| `distill.py`, `filter.py`, `hint_screen.py` | Teacher generation, the quality filter with a rejection report, and the semantic hint-leak screen (its own prompt, `rubrics/hint_screen_v1.md`, separate from the eval rubric). |
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
- **Pinned judge:** one exact model id, never a moving alias: `anthropic:claude-opus-5-5`. It
  is set in `config/models.toml` and recorded in every transcript alongside `rubric_sha256`. It
  replaced `openai:gpt-5.5-2026-04-23`, which scored the M1 ablation and the H1 pilot; the rubric
  and its hash are unchanged.
- **Decoding:** the pinned judge rejects `temperature`, so it runs at the provider default
  temperature with a pinned effort (`[decoding.anthropic]`). The backend drops a rejected sampling param, retries
  once and records the params that actually ran (`effective_params`), which go into
  `results.json`. Reproducibility rests on the pinned snapshot plus the rerun tolerance in R28.
- **Bias controls:**
  - The judge is the same model as the teacher, so it may prefer teacher-style replies
    (self-preference) and inflate the base-vs-tuned gap. The docs state this risk; the
    deterministic checks still override the judge, and the R29 hand audit measures agreement.
  - The judge is the same model as the Anthropic ablation subject. The ablation table flags every
    cell that shares the judge's family.
  - A second-family judge (`[cross_check]`, OpenAI) rescores a 40-example subset, with a
    target agreement of ≥85%. R29 also requires a hand audit of 20 replies.
  - `scripts/check_rubric.py` asserts one rubric hash across all committed runs and fails a run
    whose rows come from more than one judge.
  - Side-by-side numbers come from one judge. `scripts/rejudge.py RUN_DIR --judge <spec> --out
    NEW_DIR` re-scores a run's stored replies (no generation) with new judge pins and a
    `rejudged_from` pointer; M1 and the H1 pilot are re-judged this way before comparison.
- **Backends:** `anthropic`, `openai` and `mock`. `mock` derives scores from the
  deterministic checks (spec = all pass; robustness = 5 if all pass, 2 with one failure, 1 with more) and is
  deterministic (R19).
- **Unparseable judge output:** one retry with a format reminder, then `judge_error` is recorded
  and the example counts as a failure. It is never silently dropped.

## File formats

**Snippet** (`data/snippets/*.jsonl`):
```json
{"id": "lean_idle.pcv_elbow", "kind": "tsb", "system": "fuel_air", "family": "lean_idle",
 "title": "Lean code at idle from cracked PCV hose", "text": "Applies to 2.0L I4 and 2.5L I4 engines, ...",
 "applies": {"engines": ["2.0L I4", "2.5L I4"], "years": [2012, 2020]},
 "requires": {"rough_idle": "cold"}, "step": "...", "results": ["..."]}
{"id": "lean_idle.nm_pcv_elbow", ..., "text": "... worst on a warm engine ...",
 "requires": {"rough_idle": "warm"}, "mismatch": "condition", "twin_of": "lean_idle.pcv_elbow",
 "scope_edits": [["worst on a cold engine", "worst on a warm engine"]]}
```
- Library ids are internal. Scenarios carry only `id` (re-assigned), `kind`, `system`, `title`
  and `text`.
- Chain snippets (`step`, `results`) are gold-capable. Each states its scope explicitly: a
  structured `applies` (engines, years, bodies, transmissions, all spelled out in the text) and a
  `requires` table of operating conditions. Every complaint of the family lists its facts in
  `complaint_facts` (`data/families.toml`), and each required condition must be stated by every
  complaint with an allowed value, so a gold snippet never contradicts the complaint.
- Near-misses are **twins** (D-056): a chain snippet's text with only `scope_edits` applied
  (at most 10 word or number tokens: an engine list, a year range, a transmission, a body or one
  condition phrase). Procedure, punctuation, numbers and opener stay the same, and every scope
  edit keeps the character and word counts (all twins but one held-out transmission twin), so
  gold and near-miss match on every surface feature and only the scope separates them. A
  condition twin's title is edited with its scope, so it never states the condition the edit
  removed. A vehicle twin is placed only where a scope field it changed excludes the scenario
  vehicle. A condition twin `requires` a condition every complaint of its family contradicts. In
  a distractor scenario the first near-miss is the gold's own twin.
- **No prior picks the gold within a pair** (D-061). Every year twin pair uses the same two
  ranges, 2010-2015 and 2016-2021, split at the catalog median model year (2016), with the gold
  on the earlier side in half the pairs and on the later side in the other half. The generator
  keeps two running balances per split (and separately for the ablation subset): which side of
  the median the gold of a provided pair covers, steering the next gold choice toward the other
  side, and whether the gold or its twin comes first in the snippet list. Abstain scenarios
  often hold a twin pair with both members out of scope (a chain snippet whose `applies` fails
  for the vehicle, plus its twin), so "two near-identical snippets" doesn't mean cite.
  `tests/test_twins.py` and `tests/test_twin_priors.py` check the construction. They hold a
  ridge probe on surface, year-range and pair features, and every single surface heuristic, to
  at most 0.60 accuracy on distractors.
- Vehicles are described generically (body style, engine, transmission, model year, mileage).
  Every vehicle intro states the transmission, so a transmission-scoped snippet never rests on a
  guess, and every applicability field and required condition of a gold or near-miss is stated
  by the scenario. No manufacturer names appear.
- IDs are opaque strings drawn from ≥3 visibly synthetic schemes (for example `TSB-0417`,
  `SM 6.2.1`, `B-19-045`) and are re-assigned per scenario, so a model can't memorize an
  ID→symptom map.
- Snippet order is shuffled per scenario, apart from the gold-before-twin balance above.

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
- Gold with two or more `supporting_ids` is any-of: citing any one of them is correct, and the
  reply still cites exactly one. Our sets use this for multi_symptom rows where both complaints
  have a supporting snippet.
- CSV with an `id` and a `prompt` column, and a JSON array, are also accepted (R12).

**Training chat record** (`data/datasets/vN/train.jsonl`):
```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, ..., {"role": "assistant", "content": "<target reply>"}],
 "meta": {"scenario_id": "tr-0913", "category": "pressure", "family": "misfire", "teacher": "<model id>",
          "filter_version": "f2", "judge": {"spec_adherence": 1, "robustness": 5}}}
```
It sits next to a `manifest.json` (counts, hashes, versions) and `rejections.jsonl`.

**Judge transcript** (`results/<run_id>/transcripts.jsonl`, one line per model × scenario):
```json
{"run_id": "...", "scenario_id": "eo-0042", "model": "hf:org/repo@<sha>", "strategy": "zero_shot",
 "prompt_sha256": "...", "reply": "...", "had_think": false, "checks": {"one_step": true, "no_fix": false, ...},
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
  `zero_shot` (R10). It runs on MPS (Apple silicon) or CUDA in bf16, otherwise on CPU in fp32.
  torch, transformers and peft are core dependencies, so `uv run eval.py` needs no extra.
  - LoRA adapters are detected (`adapter_config.json`) and loaded on top of the base model
    through peft.
  - The prompt is rendered by `render_chat_text` and that string is tokenized (R27).
  - The revision resolves to a commit sha, recorded in results.
  - Greedy decoding, `max_new_tokens=256`, the chat template with `enable_thinking=False`, and any
    `<think>` blocks stripped (R27).
- **`anthropic:` / `openai:`** are API models for the ablation, at the pinned decoding params
  (provider defaults where a param is rejected; the effective params are recorded). The
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
  resume without re-paying. The run directory's `run.json` records the eval-set sha256, the
  scenario IDs, the per-strategy prompt hashes, the judge and the rubric; `--resume` refuses any
  mismatch. A non-retryable 4xx (bad key, no access) stops the run with exit 2 instead of scoring
  failures.
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
| **Judge bias** (self-preference, leniency) | Deterministic checks override the judge and cap Robustness at 2. The judge shares the teacher's family (a stated self-preference risk). A second-family (OpenAI) judge rescores 40 examples, target agreement ≥85%. Side-by-side numbers use one judge (re-judged runs). The ablation table flags the cell that shares the judge's family. The rubric is fixed and versioned, and the judge snapshot is pinned. Spot-audit 20 transcripts by hand against P1–P8 and report the agreement rate. |
| **No API keys yet** | Everything in M0 is key-free (mock backends, `--dry-run`, committed scenario sets). M1 and later stages wait for keys. |
| **Heuristic checks** (`one_step`, `no_fix`) misfire | Versioned lexicons with fixtures. False positives are reviewed in the filter report. A check that disagrees with a hand label on more than 5% of a 50-item audit sample gets revised, with a version bump. |
| **Synthetic snippets read as unrealistic** | A hand-written library across ~16 families in two styles (bulletin and manual). Never real text. |
| **Qwen3 thinking output and template mismatch** | One renderer, `enable_thinking=False`, a test that train and eval strings are equal, and a `no_think` check (R27). |
| **Publishing needed by Milestone 1** (draft public checkpoint for pinned numbers) | It needs the owner's approval and HF account. Ask early. Until then, pins are recorded and the push is a single command. |
| **Third-party held-out format unknown** | A tolerant loader (JSONL, JSON, CSV, prompt-only, any ID scheme) with three fixture shapes (R12). |

## Deployment

There is no service. The deployment is the public HF Hub model (adapter plus merged weights, with
its commit hash) and dataset, plus the local inference demo (`scripts/demo.py`), base vs tuned.
