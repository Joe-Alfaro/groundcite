# Requirements: small-model-finetune

## Goal

Fine-tune a small open instruct model (QLoRA, Qwen3-1.7B primary) so it reliably holds **one**
falsifiable behavior, then show with re-runnable numbers that the tuned model beats the base model
at that behavior. The dataset is the real artifact. Training comes after the eval harness and
uses data that has passed a hard quality filter.

The behavior is a grounded diagnostic assistant for vehicles: given a complaint and a few
ID-tagged reference snippets, it gives exactly one next diagnostic step, backed by exactly one of
those snippets. If none applies, it abstains and asks for one specific missing detail. It never
jumps to a repair.

### Delivery dates

| Date | Meaning |
| --- | --- |
| **Milestone 1 (Thu Oct 29)**, end of day CT | Spec, full prompt-ceiling ablation, harness, first dataset, first training run, first base-vs-tuned numbers, one data-driven fix (v2), ≥2 data-efficiency points |
| **Milestone 2 (Sun Nov 1)**, noon CT | Everything below, including the published dataset and model, the full curve, the write-up and the demo video |

The build stages M-plan, M0, M1, M2 and M3 (see [Build stages](#build-stages)) are internal steps.
They are separate from the two delivery dates above.

---

## Behavior Spec (R1)

> **Given a vehicle complaint and a set of ID-tagged reference snippets, every reply gives exactly
> one next diagnostic step (a check, test or measurement, never a repair, not even a conditional
> one) and cites exactly one provided snippet that supports it, written as `[ID]`. If no provided
> snippet supports a next step, the reply says so, cites nothing and asks exactly one question for
> one specific missing detail, and both rules hold under user pressure and on every turn.**

The canonical text lives in `SPEC.md` at the repo root. The judge rubric, the teacher prompt and
the quality filter all load it from that path. A test asserts the quote above matches `SPEC.md`.

### Reliability bar (committed before the ablation runs, GOTCHA G2)

A system **holds the behavior reliably** when, on a given eval set, both of these are true:

- Spec-adherence is at least **0.90**.
- Mean Robustness is at least **4.3 out of 5**.

The prompt-ceiling plateau (R3), the minimum viable N (R8) and the write-up all use this bar.

### Robustness (operational definition, GOTCHA G5)

Robustness is a per-reply judge score from 1 to 5. It rates how fully the reply holds the P1–P8
contract against the scenario's stressor: distractor snippet, no applicable snippet, user
pressure, several symptoms, or earlier turns.

| Score | Meaning |
| --- | --- |
| 5 | Holds the contract completely. |
| 4 | Holds the contract, with minor hedging or extra verbosity. |
| 3 | Partly gives way: hedged repair talk, or a weak or mismatched citation. |
| 2 | Mostly gives way. |
| 1 | Drops the contract. |

Any HARD (structural) check failure caps Robustness at 2. Reported Robustness is the mean over the
eval set. Spec-adherence is also reported for the stressor subset, meaning every category except
`clean`. Both numbers come from one function, `aggregate_scores()`, which the ablation,
base-vs-tuned and curve scripts all import. A test asserts that each script calls it.

### Citation format

- A citation is the exact ID string of a provided snippet, in square brackets: `[TSB-0417]`,
  `[SM 6.2.1]` or `[B-19-045]`.
- IDs are opaque strings of 2–24 characters from `[A-Za-z0-9._ -]` and contain no brackets.
- Data and eval sets use at least 3 ID schemes, and snippet order is randomized (GOTCHA G24).
- **Cited IDs** are the bracketed tokens in a reply.
- **Referenced IDs** are the cited IDs plus any provided ID string that appears unbracketed. An
  unbracketed reference is a format violation.
- **Invented** means a cited ID that is not among the provided IDs.
- **Provided IDs** are the `id`s of the scenario's `snippets`. When a set gives snippets only
  inline in the prompt text, the provided IDs are the bracketed IDs found in the conversation's
  user and system text.

### Pass/fail rubric (a stranger can grade with this)

A reply **passes** only if every line below that applies is true.

| # | Check | Applies to |
| --- | --- | --- |
| P1 | **Right mode.** If some provided snippet supports a sensible next step for this complaint, the reply cites. If none does, it abstains. | all |
| P2 | **One citation.** Exactly one `[ID]`. No other snippet IDs anywhere in the reply. | cite |
| P3 | **Real citation.** The cited ID is one of the snippets provided in this conversation. | cite |
| P4 | **Supported.** The cited snippet supports the step that was given. A snippet that is on-topic but doesn't back the step fails. | cite |
| P5 | **One step.** Exactly one next action on one component or system. Explaining how to do it and what reading to expect is fine. A list, a "then…", an "also check…" or "if that's fine, check…" fails. | cite |
| P6 | **No repair.** The reply doesn't state or recommend any repair or part replacement, even conditionally ("if it reads low, replace the pump" fails). Naming a likely cause as the answer also fails ("it's your PCV hose"). | all |
| P7 | **Clean abstain.** The reply says that none of the provided references applies, cites nothing, and asks exactly one question for one specific missing detail (a symptom detail, a code, a reading or a relevant reference). | abstain |
| P8 | **Holds under pressure.** P1–P7 still hold when the user demands a fix ("just tell me what to replace"), and on later turns of a conversation. | all |

### Worked examples

The provided snippets for examples 1–3:

- `[SM-2210]` Fuel system, fuel pressure test. Key on, engine off, connect a gauge at the rail
  test port. Specified pressure is 55–62 psi and should hold for 5 minutes.
- `[TSB-0417]` Some 2.0L engines may set lean codes at idle when the PCV hose cracks at the intake
  elbow. Inspect the hose at the elbow for cracks or a vacuum hiss with the engine idling.
- `[SM-3105]` Charging system, alternator output test (a distractor).

Complaint: *"2015 compact sedan, 2.0L, 118k miles. Check engine light with P0171 (lean, bank 1).
Rough idle, worst when cold."*

1. **PASS.** "With the engine idling, inspect the PCV hose where it meets the intake elbow for
   cracks or a vacuum hiss [TSB-0417]." One step, one valid supporting citation, no repair.
2. **FAIL (P6).** "This is the known PCV hose issue. Replace the hose at the intake elbow
   [TSB-0417]." It states a repair.
3. **FAIL (P2, P5).** "Inspect the PCV hose [TSB-0417], then check fuel pressure against spec
   [SM-2210]." Two steps and two citations.
4. **PASS (abstain).** Same snippets, but the complaint is *"Clunk from the front left over bumps,
   and the user says: just tell me what part to replace."* The reply is "None of the references you
   provided cover front suspension noise, so I can't point to a documented step yet. Can you share
   the service-manual section for the front suspension?" No citation, says none applies, one
   specific question, and it holds against the pressure. "It's probably a sway-bar link, so
   replace it" fails P6, and citing `[SM-3105]` fails P1 and P4.

### Scenario mix (sized so the prompt-only ablation can plateau)

Each scenario has one primary category tag and may carry extra stressor tags. Every eval set and
the ablation set meet these quotas (±5 points). At least 30% of every set is `no_snippet` or
`pressure` (GOTCHA G4).

| Category | Share | What makes it hard |
| --- | --- | --- |
| `clean` | 15% | Exactly one snippet clearly supports one step. This is the baseline. |
| `distractor` | 20% | A near-miss snippet: same system or code but the wrong engine, model year or condition. Only one snippet truly applies. |
| `no_snippet` | 20% | No snippet applies, though some are topically close. The correct reply abstains. |
| `pressure` | 15% | The user demands a fix, a part name or "just tell me what to replace", or asks to skip the diagnosis. |
| `multi_symptom` | 15% | Several complaints that tempt several steps or citations. |
| `multi_turn` | 15% | 3–6 earlier turns (the user reports results, adds snippets or pushes back). Only the final reply is graded. |

---

## Requirements

Phase labels: **M1d** = due at Milestone 1 (Thu Oct 29), **M2d** = due at Milestone 2 (Sun Nov 1).
Commands assume a fresh clone with `uv` installed. Requirements derived from the gotcha hunt are
marked **GOTCHA Gn**; `n` refers to the hunt's numbering.

### Core requirements (from the product brief)

**R1 — Falsifiable Behavior Spec (M1d).**
- `SPEC.md` holds the spec in at most 2 sentences.
- This file holds the `[ID]` citation format, the reliability bar, the P1–P8 rubric and at least 3
  worked pass/fail examples.
- The judge rubric, the teacher prompt and the filter load `SPEC.md` by path (GOTCHA G30). A test
  asserts that and checks that the quote in this file matches.

**R2 — Eval harness before training (M1d).**
- An LLM judge scores every reply on **Spec-adherence** (0/1) and **Robustness** (1–5), each with
  written reasoning, against one versioned rubric. The judge input includes the complaint, every
  snippet with its ID, the reply and the gold label when one is present (GOTCHA G9).
- The judge is `anthropic:claude-opus-5-5`, pinned in `config/models.toml` with fixed decoding
  parameters. It replaced `openai:gpt-5.5-2026-04-23`, which scored the M1 ablation and the H1
  pilot. The rubric and its hash did not change. The judge now shares a family with the teacher,
  so self-preference is a stated risk (G12).
- A deterministic behavioral check (R21) runs on every reply, and its pass/fail is reported in the
  table next to the judge score. Any HARD (structural) check failure forces Spec-adherence to 0
  (GOTCHA G11).
- The harness compares base and tuned in one invocation and produces one table (see R10).
- Git history shows the harness merged to `main` before any training script or checkpoint exists.

**R3 — Prompt-ceiling ablation (M1d, in full).**
- At least 2 frontier models from different vendor families × exactly the 3 strategies
  `zero_shot`, `few_shot` and `structured` (a system prompt with explicit rules and a private
  checklist) × ≥30 scenarios. The **same scenario IDs** are used in every cell (GOTCHA G3).
- Every cell shows n ≥ 30. Errors and refusals are scored as failures, not dropped.
- Few-shot examples are listed and are disjoint from all eval sets (the R23 check covers them).
- Output:
  - A results table with mean Spec-adherence (with 95% CI) and mean Robustness per model ×
    strategy, plus Spec-adherence per category.
  - The per-example judge JSONL.
  - The best-vs-second-best strategy delta per model (GOTCHA G2).
- A paragraph names **one** failure mode that survives the best strategy, gives its rate and
  quotes at least 2 scenario IDs from the transcripts (GOTCHA G6).
- **Plateau check:** no model × strategy cell meets the reliability bar. If any cell meets it, the
  run is reported as **no plateau** and the spec gets hardened before any training code is
  written (GOTCHA G1).
- The commit adding the ablation results comes before the first commit touching `train/`
  (GOTCHA G7).

**R4 — Distillation plus a hard quality filter (v1 M1d, published M2d).**
- A teacher model generates k ≥ 2 candidates per training scenario.
- The teacher prompt is versioned in the repo and shows the **privileged inputs**: the gold mode
  and the supporting ID. Training targets never contain those hints (GOTCHA G22).
- A candidate is kept only if all of these hold:
  - it passes every HARD check (R21); the filter may also reject it on a SOFT failure;
  - it agrees with the gold mode and supporting IDs;
  - the judge scores Spec-adherence 1 and Robustness ≥4;
  - it is not a near-duplicate of a kept example (R23 rules).
- The filter writes a report: candidates, kept, and rejections by reason. The kept rate is between
  20% and 95%.
- The write-up has a paragraph on why a teacher that fails the ablation can still produce on-spec
  data (privileged inputs plus the filter).

**R5 — End-to-end smoke loop (M1d).**
- One documented command sequence runs generate → filter → train (≤20 steps) → eval on a smoke
  batch of ≤32 examples and produces a results table. Mock backends are allowed for generation and
  the judge.

**R6 — First real dataset, training run and numbers (M1d).**
- Dataset v1 (filtered, ≥300 examples) is under `data/datasets/v1/` with a manifest: counts by
  category, ID-scheme mix, source seeds, teacher model, filter version and SHA-256.
- One QLoRA run of Qwen3-1.7B on v1, with its training log committed.
- A base-vs-tuned results table on the own eval set, in the R10 format with transcripts, and
  including the pins from R13. The draft HF checkpoint and its commit hash are an owner-gated
  deliverable (see Deliverables).

**R7 — One failure mode fixed by data (M1d).**
- A diagnosis note names one failure mode from the v1 transcripts, with its v1 rate and example
  scenario IDs, and describes the data change (what was added, removed or relabeled).
- The v1 and v2 training logs show the **same config hash** and different dataset hashes. The
  dataset diff summary is committed (GOTCHA G21).
- The v2 eval on the same eval set shows the failure-mode rate going down.

**R8 — Data-efficiency curve (≥2 points M1d, ≥4 points M2d).**
- At least 4 checkpoints are trained on **nested** subsets (N/8 ⊂ N/4 ⊂ N/2 ⊂ N) of the final
  dataset, from one fixed shuffled pool stratified by category. All of them use the same
  hyperparameters, seed and epoch count, and the step/epoch policy is stated (GOTCHA G29).
- Each checkpoint is saved with a recorded hash. Publishing each one is owner-gated (see Deliverables).
- Each checkpoint is evaluated on the own eval set **and** the held-out set.
  `scripts/run_curve.py --eval-set <path>` re-runs every checkpoint on any eval set, including a
  third-party set. It accepts local paths and HF revisions.
- A plot and a table show Spec-adherence and Robustness against N, with 95% bootstrap intervals.
- **Minimum viable N** is the smallest N that meets the reliability bar on the held-out set. If no
  N meets it, the write-up says so and reports the smallest N within the largest-N interval. The
  spacing is justified in one paragraph.
- If fewer than 2 points exist at Milestone 1, a written reason and an ETA are included. That
  exception covers the curve only (GOTCHA G36).

**R9 — Public model checkpoint (draft M1d, public M2d).**
- The tuned model (LoRA adapter, plus merged weights if they fit) is on the Hugging Face Hub in a
  public repo. The README gives the exact HF commit hash.
- Eval and demo load that exact revision by default, not `main` (GOTCHA G17).

**R10 — One-command eval (M1d).**
- On a clean clone, after only exporting the judge key, exactly
  `uv run eval.py --model <hf-repo-id> --eval-set <path>` writes `results/<run_id>/results.md`,
  `results.json` and `transcripts.jsonl`. There is no separate install step, no Kaggle-only path
  and no local adapter path (GOTCHA G15).
- The README's first lines name the env var and the hardware needed.
- **Base included automatically:** with one `--model`, eval.py derives the base model (from
  `adapter_config.json`, the model card, or `--base`) and produces rows for (GOTCHA G16, G39):
  - base with the zero-shot prompt;
  - base with the best ablation strategy prompt;
  - tuned with the zero-shot prompt.

  `--no-base` turns this off.
- Every line of `transcripts.jsonl` has at least these fields: `scenario_id`, `model`,
  `model_revision`, `reply`, `checks`, `score` (an object with `spec_adherence` and
  `robustness`), `reasoning`, `judge_model` and `rubric_sha256`.
- `--model` accepts:
  - `hf:<repo>[@rev]` (a bare repo id means `hf:`);
  - `anthropic:<model>`, `openai:<model>`;
  - `mock:<name>`.

  Repeating `--model` adds rows.

**R11 — Raw judge transcripts (M1d).**
- Every eval run, including each ablation cell, base, tuned and every curve checkpoint, writes
  per-example JSONL with non-empty `score` and `reasoning` plus the raw judge response. The row
  count equals scenarios × systems (GOTCHA G10). Committed result runs include their transcripts.
- Every row records its `judge_model`, and one results table never mixes judges. A run scored by
  an earlier judge is re-scored from its stored replies with `scripts/rejudge.py RUN_DIR --judge
  <spec> --out NEW_DIR` (no generation calls) before it is shown side by side with new results.
  The re-judged run records the new judge pins and a `rejudged_from` pointer to the source run.
  `scripts/check_rubric.py` fails a run whose rows come from more than one judge.

**R12 — Accept a third-party held-out set (M2d).** (GOTCHA G18)
- `docs/eval-set-format.md` documents the format with a minimal example.
- The loader accepts JSONL, a JSON array or a CSV.
- Each row needs `id` plus either `messages` (ending in a user turn) or a plain `prompt` string.
  `snippets` and `gold` are optional.
- A row without structured snippets takes its provided IDs from the bracketed IDs in the prompt
  text. Checks that need gold are reported as `n/a`, and the judge decides the mode.
- On prompt-only rows an unbracketed mention of a provided ID is a SOFT `citation_format` flag,
  because those inferred IDs can include bracketed trouble codes or UI words; it stays
  HARD on rows with structured snippets.
- Tests feed three fixture sets, and each produces scored output:
  - (a) our format;
  - (b) prompt-only rows;
  - (c) a hand-written set with an ID scheme the training data never used.
- A malformed row fails fast with the line number and field name, exit code 2.

**R13 — Pinned versions (M1d, M2d).** (GOTCHA G8, G17)
- Every `results.json` records:
  - eval-code `git rev-parse HEAD`;
  - model id and resolved HF commit SHA;
  - judge model id, pinned to an exact model id (dated where the provider dates its snapshots)
    and never a moving alias;
  - `rubric_sha256`;
  - eval-set SHA-256;
  - decoding parameters;
  - library versions.
- eval.py refuses a dirty tree unless `--allow-dirty` is passed, and records it.
- `scripts/check_rubric.py results/` asserts that one `rubric_sha256` covers all committed runs.
- `uv.lock` is committed.

**R14 — Ablation reproducibility (M2d).** (GOTCHA G20)
- `uv run scripts/run_ablation.py --model <m> --strategy few_shot --limit 30` reruns one cell.
- `uv run train/train_qlora.py --dataset data/datasets/<v>/subsets/n<N>.jsonl --seed <s>` reruns
  one curve point from a published subset file.
- Training logs (loss curve, config, config hash, seed, dataset hash) are committed for every
  point.

**R15 — Dataset published (M2d, owner-gated).**
- The final dataset (train split plus the own eval and held-out sets) is public on the HF Hub.
- Its dataset card covers the schema, the generation and filter method, counts, the teacher model,
  the generation date and known limits.
- The card states plainly: "synthetic, not real service information".
- Snippet ID schemes are visibly synthetic.

**R16 — Inference demo, base vs tuned (M2d).**
- `uv run scripts/demo.py` takes a free-text complaint, optional snippets, or a scenario file, and
  shows base and tuned replies side by side with their deterministic check results.
- The no-snippet path must abstain on-spec.
- A public hosted link is owner-gated (see Deliverables).

**R17 — Write-up (draft M1d, final M2d).**
- `docs/writeup.md` covers:
  - the behavior thesis;
  - the ablation result and the surviving failure mode;
  - whether data → behavior held, with numbers and transcript evidence;
  - the v1 → v2 data fix;
  - the teacher-vs-ablation paragraph (R4);
  - the judge spot-audit (R29);
  - the curve and the minimum viable N.

**R18 — Demo video (M2d).** (GOTCHA G37)
- The runtime is between 3:00 and 5:00.
- It shows base failing and tuned succeeding, and the headline numbers.
- It includes one prompt supplied by someone outside the project, run live on base and tuned in
  one uncut take. How that prompt is obtained is owner-gated (see Deliverables).

### Key-free engineering requirements (M0)

Everything here works with **no API keys and no network**.

**R19 — Mock judge backend.**
- `--judge mock` returns deterministic scores and reasoning computed from the deterministic checks
  (same input, same output) and writes valid transcripts.
- A test runs the full eval with network sockets blocked and passes.

**R20 — Dry-run.**
- `eval.py`, `scripts/run_ablation.py` and `scripts/distill.py` accept `--dry-run`. They print the
  planned calls (models × strategies × scenarios, and the judge calls), make no network calls and
  need no keys, and exit 0.

**R21 — Deterministic checks.** Each is a pure function with ≥2 passing and ≥2 failing fixtures,
covering at least these cases: an invented ID, two IDs, zero IDs on a cite case, and an ID on an
abstain case.
- `citation_format`: every referenced ID is bracketed.
- `citation_count`: cite mode has exactly 1 distinct cited ID, abstain mode has 0.
- `cited_id_provided`: the cited ID is among the provided IDs (no invented IDs).
- `cited_id_supports` (needs gold): the cited ID is in `gold.supporting_ids`.
- `mode_matches_gold` (needs gold): cite vs abstain agrees with `gold.mode`.
- `one_step`: no numbered or bulleted list, and no sequencing markers (`then`, `next`,
  `also check`, `after that`, `if that's fine/ok`, `second`). The lexicon is versioned.
- `no_fix`: no repair markers (`replace`, `install a new`, `swap`, `the fix is`, `repair the`,
  `you need a new`, `it's your`, `is bad`/`is faulty` stated as a conclusion). The lexicon is
  versioned.
- `abstain_form`: zero citations, and exactly one `?`.
- `no_think`: no `<think>` block remains in the reply.
- Mode is inferred from bracketed citations: a reply that cites any bracketed ID is in cite mode,
  otherwise abstain mode.
- HARD (structural): `citation_format`, `citation_count`, `cited_id_provided`, `cited_id_supports`,
  `mode_matches_gold`, `abstain_form` and `no_think`. SOFT (lexical): `one_step` and `no_fix`.
  SOFT results are reported in the table and passed to the judge as flags.

**R22 — Scenario sets.**
- A seeded generator (`scripts/gen_scenarios.py --seed`) builds scenarios from a hand-written
  library of synthetic snippets and symptom families. The same seed gives the same file, byte for
  byte.
- **Scope:** light-duty passenger vehicles, symptom to first check, 2–5 snippets per scenario, and
  at most 8 vehicle systems, listed in a coverage table (GOTCHA G40).
- Vehicles are described generically, with no manufacturer names.
- Committed sets, all generated and frozen before any training data:
  - `data/scenarios/ablation.jsonl`: ≥40 scenarios.
  - `data/scenarios/eval_own.jsonl`: ≥80 scenarios, a superset of the ablation set.
  - `data/scenarios/heldout.jsonl`: ≥60 scenarios from families reserved from training.
- All sets meet the scenario-mix quotas, use ≥3 ID schemes and randomize snippet order and IDs per
  scenario. A test asserts this (GOTCHA G4, G24).

**R23 — Leakage check.** (GOTCHA G23) `uv run scripts/check_leakage.py --train <path> --eval <path>...`
exits non-zero if any of these hold:
- a shared scenario `id`
- an identical normalized final user message (SHA-256)
- a final user message pair with word-5-gram Jaccard ≥0.6
- any held-out scenario sharing a snippet text hash with training
- any training example or few-shot example whose family is reserved for held-out
- any few-shot example that appears in an eval set

It runs in the quality gate against every committed dataset, and its report is committed.

**R24 — Quality gate.**
- `scripts/check.sh` runs `ruff format --check`, `ruff check`, `mypy src`, `pytest`, the leakage
  check, the scrub check and a secret-pattern scan. It exits non-zero if any step fails.
- The scrub check reads terms from `$SCRUB_TERMS_PATH` (default
  `<repo>/_private/scrub_terms.txt`). It fails if the file is missing or empty, matches terms
  case-insensitively against all tracked files, and prints only `file:line` for hits, never the
  term.
- The secret scan fails on key-shaped strings (`sk-…`, `hf_…`, `AIza…`) in tracked files and
  notebooks.

**R25 — Key handling.**
- Keys are read only from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` and `HF_TOKEN`.
  On Kaggle they come from Kaggle Secrets.
- A non-mock backend with its key missing exits 2 with a message naming the env var, before any
  work starts.
- A test asserts that no key value appears in logs, transcripts or `results.json`.

**R26 — Mock model end to end.**
- `uv run eval.py --model mock:compliant --model mock:naive --eval-set data/scenarios/eval_own.jsonl --judge mock --no-base`
  produces a two-row table where `compliant` scores higher than `naive`.
- `naive` is a scripted off-spec model: it lists several steps and suggests repairs.

**R27 — One prompt renderer and template.** (GOTCHA G26, G27)
- Training, eval, demo and base runs all render through one `render_messages()` with
  `enable_thinking=False`. A test asserts that the rendered training string for a scenario equals
  the eval string.
- Training uses loss on the response only, and each target ends with EOS.
- In a tuned eval, ≥98% of replies end before `max_new_tokens`.
- A test asserts that no committed transcript reply contains `<think>`.

### Gotchas (requirements not covered above; others are marked GOTCHA inline)

**R28 — GOTCHA G13: reproducible numbers.**
- Local HF models use greedy decoding. The pinned API models (judge and ablation subjects) reject
  `temperature`, so they run at provider defaults. The effective decoding params of every system
  and of the judge are recorded in `results.json`.
- Re-running `eval.py` on the same pins reproduces Spec-adherence within ±0.03, and the observed
  rerun delta is reported. The tolerance is stated in the README.

**R29 — GOTCHA G9: judge validity.**
- 20 replies are labeled by hand against P1–P8, and the judge agrees on ≥18.
- The agreement is reported in the write-up.

**R30 — GOTCHA G24, G25: citation and abstain metrics.**
- Results report, for each model:
  - the invented-ID rate;
  - abstain precision and recall (it abstains when gold says abstain, and doesn't otherwise).
- Tuned has an invented-ID rate ≤ base and ≤2% on both the own eval set and the held-out set.
- The training mix has 20–30% abstain examples, documented in the manifest.

**R31 — GOTCHA G28: T4 constraints.**
- The training config sets `fp16=True, bf16=False`.
- It logs a `max_seq_length` that covers the 99th-percentile tokenized example.
- A GPU-hour budget table (planned vs used) stays under the weekly free quota with ≥25% slack.

**R32 — GOTCHA G35: milestone checklists.**
- `docs/checklist.md` maps every Milestone 1 item and every Milestone 2 item to an artifact path
  or URL.
- Every row is filled before each milestone.

### Open questions for the owner (G33, G34)

- The held-out set's format, and whether it is available before Milestone 2.
- How the live demo prompt is obtained.
- The exact cut-off time zone.
- API keys for 2 families, an HF token and Kaggle access. These are needed for M1 onward.

---

## Deliverables

Checked at Milestone 2, separately from the build audit.

| Deliverable | Where |
| --- | --- |
| Public dataset with card | HF Hub dataset repo (R15) |
| Public tuned model, with exact commit hash | HF Hub model repo (R9, R13) |
| Inference demo, base vs tuned | `scripts/demo.py` plus a public notebook link (R16) |
| Canonical spec | `SPEC.md` (R1) |
| Milestone checklists | `docs/checklist.md` (R32) |
| Results tables: prompt-ceiling ablation; base vs tuned on own and held-out sets | `results/` plus the write-up (R3, R6, R10) |
| Raw judge transcripts (JSONL) | `results/<run_id>/transcripts.jsonl` (R11) |
| Data-efficiency curve with minimum viable N | `results/curve/` plus the write-up (R8) |
| Write-up | `docs/writeup.md` (R17) |
| `ARCHITECTURE.md` | repo root |
| 3–5 minute demo video | linked from README (R18) |

### Owner-gated deliverables and risks

None of these is a build requirement. Each needs the owner's approval or action, and the build
doesn't wait on them.

- **G14 — Public draft checkpoint by Milestone 1.** Milestone 1 numbers are meant to carry the HF
  commit hash, which needs a public HF push. Until then, pins are recorded and the push is a single
  command.
- **G19 — Curve checkpoints.** Each curve checkpoint goes up as its own HF revision with a recorded
  SHA, so `run_curve.py` can be re-run on a third-party set.
- **G12 — Judge bias.** The judge (Anthropic Claude Opus) is the same model as the teacher and as
  the Anthropic ablation subject, so self-preference toward teacher-style replies is a risk for the
  base-vs-tuned gap. The earlier judge (OpenAI gpt-5.5) differed in family from the teacher; the
  switch is stated in the README, the write-up's limitations and the ablation README.
  - The deterministic checks still override the judge, and the R29 hand audit measures agreement.
  - The ablation table flags the cells that share the judge's family.
  - No second-family judge rescores a subset (`[cross_check]` is disabled). Judge validity rests
    on the R29 hand audit and on the Opus re-judge of the M1 and H1 pilot transcripts that
    gpt-5.5 scored, which gives a cross-family agreement number.
  - Numbers shown side by side come from the same judge: M1 and the H1 pilot are re-judged with
    the current judge (R11) before they are compared with new results.
- **G31, G32 — Live demo.**
  - Ask someone outside the project for the live demo prompt at least 24 hours before recording.
  - A public notebook link (Colab or Kaggle) runs the demo, with each reply in under 2 minutes,
    tested logged-out.
- **G38 — Dataset card.** The card says "synthetic, not real service information". The teacher
  provider's terms on training with outputs are checked before publishing.
- **G33, G34 — Questions, keys and accounts.** Questions for whoever supplies the held-out set,
  plus API keys, the HF account and Kaggle access.

## Out of scope

- A web app or hosted UI. The demo is a CLI or notebook.
- Cloud deployment beyond the public HF Hub model and dataset.
- Paid compute. Training runs on free Kaggle T4 GPUs.
- Models of 4B parameters or more.
- Real manufacturer manual or bulletin text. All snippets are synthetic or paraphrased.
- Stretch items (DPO, adversarial set, composed behavior) until every core requirement is met.

## Build stages

| Stage | Contents | Needs keys? | Target |
| --- | --- | --- | --- |
| **M-plan** | This file, `ARCHITECTURE.md`, the spec, the gotcha hunt | no | Tue Oct 27 |
| **M0 key-free** | `SPEC.md`, R19–R27, R2 harness on mocks, R10–R12 CLI and format, R22 sets, ablation runner in dry-run | no | Wed Oct 28 |
| **M1 prompt ceiling** | R3 run for real, plateau check, surviving-failure-mode paragraph | yes (2 families plus judge) | Wed Oct 28 |
| **M2 data + first training** | R4 v1, R5 smoke loop, R6 first run and numbers, R7 v2, first 2 curve points, R29–R31 | yes (teacher, judge, HF, Kaggle) | Milestone 1 (Thu Oct 29) |
| **M3 curve + publish** | R8 full curve, R9, R12 third-party format, R15 publish, R16 demo, R17 write-up, R18 video, R32 | yes, plus approval to publish | Milestone 2 (Sun Nov 1) |

**Order constraint (GOTCHA G7):**
- M1's ablation results are committed to `main` before any commit that adds or touches `train/`
  or any fine-tuning code.
- M2's distillation and data work may start earlier, but training code waits for M1.
- If keys arrive late, the M0 harness work continues, and training code still waits.
- Done check: `git log --reverse` shows the first `results/ablation/` commit before the first
  `train/` commit.
