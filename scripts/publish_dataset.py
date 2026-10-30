#!/usr/bin/env python3
"""Publish a dataset version, its eval sets and a dataset card to the HF Hub (R15, G38, D-118).

    uv run scripts/publish_dataset.py --dataset data/datasets/vN --repo OWNER/NAME
        [--dry-run] [--private | --public] [--license ID] [--generation-date YYYY-MM-DD]
        [--stage-dir DIR] [--scenarios DIR] [--few-shot PATH]

Stages, flat in one folder: ``train.jsonl``, ``manifest.json`` and ``filter_report.md`` from
``--dataset`` (byte copies), ``eval_own.jsonl`` and ``heldout.jsonl`` from ``--scenarios``,
``few_shot.jsonl`` from ``--few-shot`` and a generated ``README.md`` dataset card. Every staged
file passes the scrub check (terms from ``$SCRUB_TERMS_PATH``) and the secret scan before any Hub
call; hits are reported as ``file:line`` only. ``--dry-run`` stops after the checks: no token, no
network. Otherwise the dataset repo is created (private unless ``--public``; the visibility is set
explicitly, so ``--public`` also flips an existing private repo), the folder is uploaded in one
commit and the commit SHA is printed and recorded in ``<dataset>/publish.json``.

Exit codes: 0 ok, 1 scrub/secret/upload failure, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from groundcite.config import load_models_config  # noqa: E402
from groundcite.keys import optional_key, redact  # noqa: E402

DEFAULT_SCENARIOS = REPO_ROOT / "data" / "scenarios"
DEFAULT_FEW_SHOT = REPO_ROOT / "data" / "few_shot" / "examples.jsonl"
DEFAULT_LICENSE = "cc-by-4.0"
DATASET_FILES = ("train.jsonl", "manifest.json", "filter_report.md")
SCENARIO_FILES = ("eval_own.jsonl", "heldout.jsonl")
CARD = "README.md"
RECORD = "publish.json"
REPO_RE = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# One Hub config per JSONL file: (config name, file, what it holds).
CONFIGS = (
    ("train", "train.jsonl", "training conversations, the teacher's reply as the last turn"),
    ("eval_own", "eval_own.jsonl", "the own eval set (scenarios with gold labels)"),
    ("heldout", "heldout.jsonl", "the held-out eval set (reserved symptom families)"),
    ("few_shot", "few_shot.jsonl", "the few-shot prompt examples (hand-written replies)"),
)


class UsageError(ValueError):
    """Bad arguments or inputs: exit 2."""


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_publish_{name}", SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scrub_check = _sibling("scrub_check")
secret_scan = _sibling("secret_scan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="publish_dataset.py", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--dataset", type=Path, required=True, help="data/datasets/vN")
    parser.add_argument("--repo", required=True, help="the Hub dataset repo, OWNER/NAME")
    parser.add_argument("--dry-run", action="store_true", help="stage and check; no Hub call")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true", help="private repo (the default)")
    visibility.add_argument("--public", action="store_true", help="make the repo public")
    parser.add_argument("--license", default=DEFAULT_LICENSE, help="HF license id")
    parser.add_argument("--generation-date", help="YYYY-MM-DD (default: from the manifest)")
    parser.add_argument("--stage-dir", type=Path, help="keep the staged folder here")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--few-shot", type=Path, default=DEFAULT_FEW_SHOT)
    return parser


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def _sources(args: argparse.Namespace) -> dict[str, Path]:
    sources = {name: args.dataset / name for name in DATASET_FILES}
    sources |= {name: args.scenarios / name for name in SCENARIO_FILES}
    sources["few_shot.jsonl"] = args.few_shot
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise UsageError(f"missing input file(s): {', '.join(missing)}")
    return sources


def _commit_date(path: Path) -> str | None:
    """The date of the last commit touching ``path``, or None outside git or when untracked."""
    done = subprocess.run(
        ["git", "-C", str(path.parent), "log", "-1", "--format=%cs", "--", path.name],
        capture_output=True,
        text=True,
        check=False,
    )
    date = done.stdout.strip() if done.returncode == 0 else ""
    return date if DATE_RE.fullmatch(date) else None


def generation_date(flag: str | None, manifest: dict[str, Any], manifest_path: Path) -> str:
    """``--generation-date``, else the manifest's ``generated_at``, else the manifest's last
    commit date."""
    if flag is not None:
        if not DATE_RE.fullmatch(flag):
            raise UsageError(f"--generation-date must be YYYY-MM-DD, got {flag!r}")
        return flag
    recorded = manifest.get("generated_at")
    if isinstance(recorded, str) and DATE_RE.match(recorded):
        return recorded[:10]
    date = _commit_date(manifest_path)
    if date is None:
        raise UsageError(
            "no generation date: the manifest has no generated_at and no commit; "
            "pass --generation-date YYYY-MM-DD"
        )
    return date


def _rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------------------------------------------------------------------------------------------
# The dataset card
# ---------------------------------------------------------------------------------------------


def _front_matter(license_id: str, pretty_name: str) -> str:
    lines = [
        "---",
        f"license: {license_id}",
        f"pretty_name: {json.dumps(pretty_name)}",
        "language:",
        "- en",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- synthetic",
        "- automotive",
        "- citations",
        "- grounded-generation",
        "- distillation",
        "configs:",
    ]
    for name, file, _ in CONFIGS:
        lines += [
            f"- config_name: {name}",
            "  data_files:",
            "  - split: train",
            f"    path: {file}",
        ]
    lines.append("---")
    return "\n".join(lines)


def build_card(
    manifest: dict[str, Any],
    rows: dict[str, int],
    *,
    license_id: str,
    date: str,
    eval_judge: str,
) -> str:
    """The dataset card: HF front matter, then a body built from the manifest and row counts."""
    version = str(manifest.get("version", "v?"))
    teacher = str(manifest["teacher_model"])
    provider = teacher.split(":", 1)[0]
    provider_name = {"anthropic": "Anthropic", "openai": "OpenAI"}.get(provider, provider)
    filter_judge = str(manifest.get("judge_model", "unknown"))
    pretty_name = f"GroundCite diagnostic citations {version}"
    by_category = manifest.get("counts_by_category", {})
    schemes = manifest.get("id_scheme_mix", {})
    abstain = manifest.get("abstain_share")

    category_rows = "\n".join(f"| {c} | {n} |" for c, n in sorted(by_category.items()))
    scheme_rows = "\n".join(f"| `{s}` | {n} |" for s, n in sorted(schemes.items()))
    config_rows = "\n".join(
        f"| `{name}` | `{file}` | {rows[file]} | {what} |" for name, file, what in CONFIGS
    )
    abstain_line = (
        f" Abstain share: {abstain:.1%} of the training rows." if isinstance(abstain, float) else ""
    )
    eval_family = eval_judge.split(":", 1)[0]
    eval_judge_note = (
        "a different model family from the teacher"
        if eval_family != provider
        else "the same model family as the teacher"
    )
    filter_judge_note = (
        "The filter judge is the teacher model itself, a disclosed difference from the eval "
        "judge: the filter may favor the teacher's own style."
        if filter_judge == teacher
        else "The filter judge is a different model from the teacher."
    )
    return f"""{_front_matter(license_id, pretty_name)}

# {pretty_name}

Training and eval data for a small instruct model fine-tuned to act as an automotive diagnostic
assistant under one contract: given a vehicle complaint and a set of ID-tagged reference
snippets, every reply gives exactly one next diagnostic step and cites exactly one provided
snippet that supports it, written as `[ID]`. If no provided snippet supports a step, the reply
says so, cites nothing and asks for one specific missing detail. It never states a repair as the
answer and never cites an ID that was not provided.

**This data is synthetic, not real service information.** The vehicles, complaints, reference
snippets, bulletin and manual IDs are invented or paraphrased for this dataset. Do not use any of
it to diagnose or repair a real vehicle.

## Teacher model and terms

The assistant replies in `train.jsonl` were written by the teacher model `{teacher}`
({provider_name}); the few-shot replies were written by hand. Use of this data is
subject to the teacher provider's terms ({provider_name}'s terms for model outputs),
in addition to the license below; check them before using the data to train a model.

## License

`{license_id}`. The license covers this compilation; the teacher-written replies are also
subject to {provider_name}'s terms, as above.

## Files and counts

| Config | File | Rows | Contents |
| --- | --- | --- | --- |
{config_rows}

`manifest.json` records the build (seeds, hashes, model pins, counts) and `filter_report.md` the
filter's per-stage results.

Training rows: {manifest.get("count")} in total.{abstain_line}

| Category | Rows |
| --- | --- |
{category_rows}

## Schema

- `train.jsonl`: `messages` (a chat: the system prompt, then user and assistant turns; the last
  assistant turn is the teacher's reply) and `meta` (`scenario_id`, `category`, `family`,
  `teacher`, `filter_version`, `candidate_index`, `judge` scores). When a scenario has snippets,
  they open the first user turn as a `Reference snippets:` block of `[ID] Title. Text` lines.
- `eval_own.jsonl`, `heldout.jsonl`: `id`, `snippets` (a list of `id`, `kind`, `system`,
  `title`, `text`), `messages` (user and assistant turns, ending in the user
  turn to answer), `gold` (`mode`: `cite` or `abstain`, `supporting_ids`, `near_miss_ids`),
  `category`, `tags`, `family`, `split` and `seed`.
- `few_shot.jsonl`: the same fields as the eval sets plus `reply`, the example answer.

## Generation and filter method

1. Scenarios were generated from seeded templates over hand-written symptom families, each with
   a set of synthetic reference snippets in mixed ID schemes, some of them near-miss distractors.
2. The teacher `{teacher}` (prompt version `{manifest.get("teacher_prompt_version", "?")}`)
   wrote {manifest.get("candidates", "the")} candidate replies.
3. Filter version `{manifest.get("filter_version")}` kept {manifest.get("kept")} of them: hard
   deterministic checks (citation format and count, cited ID provided and supporting, one step,
   no repair, abstain form), soft checks, a hint-leak screen, an LLM judge for spec adherence and
   robustness, and near-duplicate removal. Selection rule: {manifest.get("selection_rule")}.
4. The build's leakage check fails on shared scenario IDs, near-duplicate final messages,
   held-out snippet text or reserved held-out families in the training rows.

Generation date: {date}.

## Judges

- Filter judge (scored candidates while building this dataset): `{filter_judge}`.
- Eval judge (scores models on `eval_own` and `heldout`): `{eval_judge}`, {eval_judge_note}.

{filter_judge_note}

## Snippet ID schemes

The snippet IDs are synthetic and mix several shapes (digits shown as `#`), so a model cannot
learn one fixed format. Counts are snippet occurrences in the training rows.

| Scheme | Count |
| --- | --- |
{scheme_rows}

## Known limits

- Synthetic and English only: the snippets are short paraphrases written for this dataset, not
  real bulletins or manuals, and the vehicles are generic.
- One teacher model wrote every training reply, so the replies share its phrasing and any blind
  spots.
- The symptom families are a small, hand-picked set; the held-out set uses reserved families,
  but every scenario follows the same snippet-and-complaint layout.
- The checks and the judge measure the citation contract, not whether a step is the best
  diagnostic choice for a real vehicle.
"""


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def _scrub_terms() -> list[str]:
    path = Path(os.environ.get("SCRUB_TERMS_PATH") or scrub_check.DEFAULT_TERMS_PATH)
    if not path.is_file():
        raise UsageError(f"scrub terms file not found (set SCRUB_TERMS_PATH): {path}")
    terms: list[str] = scrub_check.load_terms(path)
    if not terms:
        raise UsageError(f"scrub terms file has no terms: {path}")
    return terms


def scan(folder: Path, terms: list[str]) -> tuple[list[str], list[str]]:
    """``(scrub hits, secret hits)`` as ``file:line`` for every staged file; never the match."""
    scrub_hits: list[str] = []
    secret_hits: list[str] = []
    for path in sorted(folder.iterdir()):
        text = path.read_bytes().decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if any(term in lowered for term in terms):
                scrub_hits.append(f"{path.name}:{number}")
            if any(pattern.search(line) for pattern in secret_scan.KEY_PATTERNS):
                secret_hits.append(f"{path.name}:{number}")
    return scrub_hits, secret_hits


# ---------------------------------------------------------------------------------------------
# Staging and upload
# ---------------------------------------------------------------------------------------------


def stage(args: argparse.Namespace, folder: Path) -> list[str]:
    """Copy the inputs and write the card into ``folder``; return the staged names."""
    sources = _sources(args)
    manifest_path = args.dataset / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UsageError(f"{manifest_path} is not valid JSON: {exc}") from None
    if not isinstance(manifest, dict) or "teacher_model" not in manifest:
        raise UsageError(f"{manifest_path} has no teacher_model")
    date = generation_date(args.generation_date, manifest, manifest_path)
    eval_judge = str(load_models_config()["judge"]["model"])

    folder.mkdir(parents=True, exist_ok=True)
    # A kept stage dir is re-used on a re-run, but only ever holds the staged files.
    expected = {*sources, CARD}
    foreign = sorted(p.name for p in folder.iterdir() if p.name not in expected)
    if foreign:
        raise UsageError(f"--stage-dir {folder} holds other files: {', '.join(foreign)}")
    for name, src in sources.items():
        shutil.copyfile(src, folder / name)
    rows = {file: _rows(folder / file) for _, file, _ in CONFIGS}
    card = build_card(manifest, rows, license_id=args.license, date=date, eval_judge=eval_judge)
    (folder / CARD).write_text(card, encoding="utf-8")
    return sorted(p.name for p in folder.iterdir())


def upload(args: argparse.Namespace, folder: Path, token: str) -> str:
    """Create the repo, set its visibility, upload ``folder`` in one commit; return the SHA."""
    import huggingface_hub

    private = not args.public
    api = huggingface_hub.HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=private, exist_ok=True)
    # create_repo leaves an existing repo's visibility alone, so set it explicitly.
    api.update_repo_settings(repo_id=args.repo, repo_type="dataset", private=private)
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(folder),
        commit_message=f"Dataset {args.dataset.name} with its eval sets and card",
    )
    return str(commit.oid)


def _publish(args: argparse.Namespace, folder: Path) -> int:
    terms = _scrub_terms()
    staged = stage(args, folder)
    scrub_hits, secret_hits = scan(folder, terms)
    if scrub_hits:
        print(f"error: scrub check failed: {', '.join(scrub_hits)}", file=sys.stderr)
    if secret_hits:
        print(f"error: secret scan failed: {', '.join(secret_hits)}", file=sys.stderr)
    if scrub_hits or secret_hits:
        print("nothing uploaded", file=sys.stderr)
        return 1
    print(f"staged in {folder} (scrub check and secret scan clean):")
    for name in staged:
        print(f"  {name}")
    visibility = "public" if args.public else "private"
    if args.dry_run:
        print(f"dry run: would upload to dataset repo {args.repo} ({visibility})")
        return 0

    token = optional_key("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN is not set (environment or .env)", file=sys.stderr)
        return 2
    try:
        sha = upload(args, folder, token)
    except Exception as exc:  # any Hub or network error; the token is redacted
        print(f"error: upload failed: {redact(str(exc), [token])}", file=sys.stderr)
        return 1
    record = {"repo": args.repo, "commit_sha": sha, "private": not args.public}
    (args.dataset / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"uploaded to dataset repo {args.repo} ({visibility}), commit {sha}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not REPO_RE.fullmatch(args.repo):
            raise UsageError(f"--repo must be OWNER/NAME, got {args.repo!r}")
        if not args.dataset.is_dir():
            raise UsageError(f"--dataset is not a directory: {args.dataset}")
        if args.stage_dir is not None:
            return _publish(args, args.stage_dir)
        with tempfile.TemporaryDirectory(prefix="publish_dataset-") as tmp:
            return _publish(args, Path(tmp))
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
