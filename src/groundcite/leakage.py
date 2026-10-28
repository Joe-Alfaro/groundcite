"""Train/eval overlap checks (R23).

Rows come from JSONL files holding scenario lines (``id``, ``messages``, ``snippets``,
``family``, ``split``), rollout conversation scripts (``opener`` and ``turns`` instead of
``messages``) or training chat records (``messages`` plus ``meta.scenario_id`` and
``meta.family``). Only the final user message is compared, because every multi-turn
conversation in a family opens the same way by design.

Each rule has a stable code that the CLI prints, so a report says exactly which rule fired.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHARED_ID = "shared_id"
SAME_FINAL_MESSAGE = "same_final_message"
NGRAM_JACCARD = "ngram_jaccard"
HELDOUT_SNIPPET_HASH = "heldout_snippet_hash"
RESERVED_FAMILY = "reserved_family"
FEWSHOT_IN_EVAL = "fewshot_in_eval"
# reserved_family can only fire on a row that names a family, so a training or few-shot row
# without one is itself a finding. Eval rows (third-party sets) need no family.
MISSING_FAMILY = "missing_family"
RULES = (
    SHARED_ID,
    SAME_FINAL_MESSAGE,
    NGRAM_JACCARD,
    HELDOUT_SNIPPET_HASH,
    RESERVED_FAMILY,
    FEWSHOT_IN_EVAL,
    MISSING_FAMILY,
)

NGRAM_N = 5
JACCARD_THRESHOLD = 0.6


class LeakageInputError(ValueError):
    """A row can't be read. The message names the file and line."""


@dataclass(frozen=True)
class Row:
    id: str
    family: str | None
    split: str | None
    final_user: str
    snippet_texts: tuple[str, ...]
    # User and system text, for finding inline snippets in chat records without ``snippets``.
    prompt_text: str
    where: str


@dataclass(frozen=True)
class Finding:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"LEAK {self.rule}: {self.detail}"


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int = NGRAM_N) -> frozenset[tuple[str, ...]]:
    """Word n-grams of the normalized text. Texts shorter than ``n`` words give one gram."""
    words = normalize(text).split()
    if not words:
        return frozenset()
    if len(words) < n:
        return frozenset({tuple(words)})
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class NgramIndex:
    """Finds stored texts whose word-n-gram Jaccard with a query meets a threshold."""

    def __init__(self) -> None:
        self._grams: list[frozenset[tuple[str, ...]]] = []
        self._postings: dict[tuple[str, ...], list[int]] = defaultdict(list)

    def add(self, text: str) -> int:
        grams = word_ngrams(text)
        index = len(self._grams)
        self._grams.append(grams)
        for gram in grams:
            self._postings[gram].append(index)
        return index

    def matches(self, text: str, threshold: float = JACCARD_THRESHOLD) -> list[tuple[int, float]]:
        grams = word_ngrams(text)
        shared: dict[int, int] = defaultdict(int)
        for gram in grams:
            for index in self._postings.get(gram, ()):
                shared[index] += 1
        hits = []
        for index, inter in sorted(shared.items()):
            score = inter / (len(grams) + len(self._grams[index]) - inter)
            if score >= threshold:
                hits.append((index, score))
        return hits


def row_from_record(record: Mapping[str, Any], where: str) -> Row:
    """A comparable row from a scenario line or a training chat record."""
    meta = record.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    row_id = record.get("id", meta.get("scenario_id"))
    if not isinstance(row_id, str) or not row_id:
        raise LeakageInputError(f"{where}: missing 'id' (or 'meta.scenario_id')")
    messages = record.get("messages")
    snippets = record.get("snippets") or []
    if isinstance(messages, list):
        turns = [m for m in messages if isinstance(m, Mapping)]
    elif isinstance(record.get("opener"), str) and isinstance(record.get("turns"), list):
        turns, snippets = _conversation_script(record)
    elif isinstance(record.get("prompt"), str):
        turns = [{"role": "user", "content": record["prompt"]}]
    else:
        raise LeakageInputError(f"{where}: missing 'messages' (or 'prompt')")
    user_turns = [str(m.get("content", "")) for m in turns if m.get("role") == "user"]
    if not user_turns:
        raise LeakageInputError(f"{where}: no user message")
    family = record.get("family", meta.get("family"))
    return Row(
        id=row_id,
        family=family if isinstance(family, str) else None,
        split=record.get("split") if isinstance(record.get("split"), str) else None,
        final_user=user_turns[-1],
        snippet_texts=tuple(
            str(s["text"]) for s in snippets if isinstance(s, Mapping) and s.get("text")
        ),
        prompt_text="\n".join(
            str(m.get("content", "")) for m in turns if m.get("role") in ("user", "system")
        ),
        where=where,
    )


def _conversation_script(record: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Any]]:
    """A rollout conversation script (docs/conversation-format.md) as user turns and snippets:
    the opener, then each turn's own words (result, text, pressure); the initial snippets plus
    every added one."""
    turns: list[Mapping[str, Any]] = [{"role": "user", "content": str(record["opener"])}]
    snippets = list(record.get("snippets_initial") or [])
    for turn in record["turns"]:
        if not isinstance(turn, Mapping):
            continue
        words = [turn.get(k) for k in ("reported_result", "text", "pressure")]
        if any(isinstance(w, str) and w for w in words):
            turns.append({"role": "user", "content": "\n\n".join(w for w in words if w)})
        snippets += turn.get("add") or []
    return turns, snippets


def load_rows(path: Path) -> list[Row]:
    """Rows from a JSONL file. Locations are reported as ``<file name>:<line>``."""
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LeakageInputError(f"{where}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, Mapping):
            raise LeakageInputError(f"{where}: expected a JSON object")
        rows.append(row_from_record(record, where))
    return rows


def is_heldout(row: Row, reserved: Iterable[str]) -> bool:
    return row.split == "heldout" or (row.family is not None and row.family in set(reserved))


def find_leaks(
    train: Sequence[Row],
    evals: Sequence[Row],
    few_shot: Sequence[Row] = (),
    reserved: frozenset[str] = frozenset(),
) -> list[Finding]:
    """Every R23 violation between training rows, eval rows and few-shot rows."""
    findings: list[Finding] = []
    eval_by_id: dict[str, list[Row]] = defaultdict(list)
    eval_by_hash: dict[str, list[Row]] = defaultdict(list)
    index = NgramIndex()
    for row in evals:
        eval_by_id[row.id].append(row)
        eval_by_hash[text_hash(row.final_user)].append(row)
        index.add(row.final_user)

    for row in train:
        for other in eval_by_id.get(row.id, ()):
            findings.append(Finding(SHARED_ID, f"{row.id!r} at {row.where} and {other.where}"))
        for other in eval_by_hash.get(text_hash(row.final_user), ()):
            findings.append(
                Finding(SAME_FINAL_MESSAGE, f"{row.where} ({row.id}) = {other.where} ({other.id})")
            )
        for i, score in index.matches(row.final_user):
            other = evals[i]
            findings.append(
                Finding(
                    NGRAM_JACCARD,
                    f"{row.where} ({row.id}) ~ {other.where} ({other.id}), jaccard {score:.2f}",
                )
            )
        if not row.family:
            findings.append(
                Finding(MISSING_FAMILY, f"training {row.where} ({row.id}) has no family")
            )
        elif row.family in reserved:
            findings.append(
                Finding(RESERVED_FAMILY, f"training {row.where} ({row.id}) uses {row.family}")
            )

    findings.extend(_heldout_snippet_leaks(train, [r for r in evals if is_heldout(r, reserved)]))

    for row in few_shot:
        if not row.family:
            findings.append(
                Finding(MISSING_FAMILY, f"few-shot {row.where} ({row.id}) has no family")
            )
        elif row.family in reserved:
            findings.append(
                Finding(RESERVED_FAMILY, f"few-shot {row.where} ({row.id}) uses {row.family}")
            )
        hits = [(other, "same id") for other in eval_by_id.get(row.id, ())]
        hits += [
            (other, "same final message")
            for other in eval_by_hash.get(text_hash(row.final_user), ())
        ]
        hits += [(evals[i], f"jaccard {score:.2f}") for i, score in index.matches(row.final_user)]
        for other, why in hits:
            findings.append(
                Finding(
                    FEWSHOT_IN_EVAL,
                    f"{row.where} ({row.id}) matches {other.where} ({other.id}): {why}",
                )
            )
    return findings


def _heldout_snippet_leaks(train: Sequence[Row], heldout: Sequence[Row]) -> list[Finding]:
    findings = []
    train_hashes: dict[str, Row] = {}
    inline_texts = []
    for row in train:
        for text in row.snippet_texts:
            train_hashes.setdefault(text_hash(text), row)
        if not row.snippet_texts:
            inline_texts.append((normalize(row.prompt_text), row))
    for row in heldout:
        for text in row.snippet_texts:
            match = train_hashes.get(text_hash(text))
            if match is None:
                # Chat records carry snippets inline in the prompt text.
                needle = normalize(text)
                match = next((r for hay, r in inline_texts if needle in hay), None)
            if match is not None:
                findings.append(
                    Finding(
                        HELDOUT_SNIPPET_HASH,
                        f"held-out {row.where} ({row.id}) shares snippet text with "
                        f"{match.where} ({match.id})",
                    )
                )
    return findings


def render_report(
    findings: Sequence[Finding],
    inputs: Mapping[str, Sequence[Path]],
    counts: Mapping[str, int],
) -> str:
    """A deterministic markdown report (no timestamps), suitable for committing."""
    lines = ["# Leakage check report", "", "Generated by `scripts/check_leakage.py` (R23).", ""]
    lines += ["| Input | Files | Rows |", "| --- | --- | --- |"]
    for role, paths in inputs.items():
        names = ", ".join(f"`{p.as_posix()}` (sha256 `{_file_sha(p)[:12]}`)" for p in paths) or "-"
        lines.append(f"| {role} | {names} | {counts.get(role, 0)} |")
    lines += ["", "| Rule | Findings |", "| --- | --- |"]
    by_rule: dict[str, int] = defaultdict(int)
    for finding in findings:
        by_rule[finding.rule] += 1
    for rule in RULES:
        lines.append(f"| `{rule}` | {by_rule[rule]} |")
    lines += ["", f"Result: {'LEAK' if findings else 'clean'}.", ""]
    if findings:
        lines += [f"- {finding}" for finding in findings] + [""]
    return "\n".join(lines)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
