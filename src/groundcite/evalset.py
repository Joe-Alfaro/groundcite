"""Eval-set loader (R12): JSONL, JSON array or CSV, including prompt-only rows.

The format is documented for third parties in ``docs/eval-set-format.md``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groundcite.spec import is_valid_id, parse_cited_ids

ROLES = ("system", "user", "assistant")
GOLD_MODES = ("cite", "abstain")
# Roles whose text can introduce snippet IDs. Assistant turns never provide IDs.
PROVIDING_ROLES = ("system", "user")


class EvalSetError(ValueError):
    """A malformed eval set. ``line`` and ``field`` locate the problem when known."""

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        field: str | None = None,
        unit: str = "line",
    ) -> None:
        self.line = line
        self.field = field
        where = f"{unit} {line}: " if line is not None else ""
        what = f"{field}: " if field is not None else ""
        super().__init__(f"{where}{what}{message}")


@dataclass(frozen=True)
class Scenario:
    id: str
    messages: list[dict[str, str]]
    snippets: list[dict[str, Any]] = field(default_factory=list)
    provided_ids: list[str] = field(default_factory=list)
    gold: Mapping[str, Any] | None = None
    category: str | None = None


class _RowError(Exception):
    """Raised inside row parsing; the caller adds the location."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


def load_eval_set(path: str | Path) -> list[Scenario]:
    """Load an eval set, choosing the format from the file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows, unit = _jsonl_rows(path), "line"
    elif suffix == ".json":
        rows, unit = _json_array_rows(path), "item"
    elif suffix == ".csv":
        rows, unit = _csv_rows(path), "line"
    else:
        raise EvalSetError(f"unsupported eval-set extension {suffix!r} (use .jsonl, .json or .csv)")

    located: list[tuple[int, Scenario]] = []
    for line, row in rows:
        try:
            located.append((line, parse_row(row)))
        except _RowError as exc:
            raise EvalSetError(exc.message, line=line, field=exc.field, unit=unit) from None

    # Set-level check, after every row is valid: results are keyed by scenario id.
    seen: set[str] = set()
    for line, scenario in located:
        if scenario.id in seen:
            raise EvalSetError(f"duplicate id {scenario.id!r}", line=line, field="id", unit=unit)
        seen.add(scenario.id)
    return [scenario for _, scenario in located]


def _read_text(path: Path) -> str:
    # utf-8-sig tolerates the byte-order mark spreadsheet tools add to CSV exports.
    return path.read_text(encoding="utf-8-sig")


def _jsonl_rows(path: Path) -> Iterator[tuple[int, Any]]:
    for line_no, text in enumerate(_read_text(path).splitlines(), start=1):
        if not text.strip():
            continue
        try:
            yield line_no, json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"invalid JSON: {exc.msg}", line=line_no) from None


def _json_array_rows(path: Path) -> Iterator[tuple[int, Any]]:
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise EvalSetError(f"invalid JSON: {exc.msg}", line=exc.lineno) from None
    if not isinstance(data, list):
        raise EvalSetError("a .json eval set must be a JSON array of rows")
    yield from enumerate(data, start=1)


def _csv_rows(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise EvalSetError("empty CSV file", line=1)
        header = [name.strip() for name in header]
        for required in ("id", "prompt"):
            if required not in header:
                raise EvalSetError(f"CSV header needs an {required!r} column", line=1)
        end = reader.line_num
        for record in reader:
            # A quoted field may span several physical lines; report where the record starts.
            start, end = end + 1, reader.line_num
            if not record:
                continue
            if len(record) != len(header):
                raise EvalSetError(f"expected {len(header)} columns, got {len(record)}", line=start)
            # CSV has no nulls: an empty optional cell means the column is absent.
            row = {k: v for k, v in zip(header, record, strict=True) if v or k in ("id", "prompt")}
            yield start, row


def parse_row(row: Any) -> Scenario:
    """Validate one decoded row and build its Scenario. Raises _RowError."""
    if not isinstance(row, dict):
        raise _RowError("row", "each row must be an object")

    scenario_id = row.get("id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise _RowError("id", "required: a non-empty string")

    messages = _parse_messages(row)
    snippets = _parse_snippets(row.get("snippets"))
    if snippets:
        provided_ids = [s["id"] for s in snippets]
    else:
        provided_text = "\n".join(m["content"] for m in messages if m["role"] in PROVIDING_ROLES)
        # Every bracketed token is provided, all-letter IDs included (R12, D-033). An inferred list
        # can hold codes or UI words, so checks treat an unbracketed mention as SOFT on these rows.
        provided_ids = parse_cited_ids(provided_text)

    category = row.get("category")
    if category is not None and not isinstance(category, str):
        raise _RowError("category", "must be a string")

    return Scenario(
        id=scenario_id,
        messages=messages,
        snippets=snippets,
        provided_ids=provided_ids,
        gold=_parse_gold(row.get("gold"), provided_ids),
        category=category or None,
    )


def _parse_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    has_messages = row.get("messages") is not None
    has_prompt = row.get("prompt") is not None
    if has_messages and has_prompt:
        raise _RowError("prompt", "give either messages or prompt, not both")
    if has_prompt:
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise _RowError("prompt", "must be a non-empty string")
        return [{"role": "user", "content": prompt}]
    if not has_messages:
        raise _RowError("messages", "required: messages (ending in a user turn) or prompt")

    raw = row["messages"]
    if not isinstance(raw, list) or not raw:
        raise _RowError("messages", "must be a non-empty list")
    messages: list[dict[str, str]] = []
    for i, message in enumerate(raw):
        if not isinstance(message, dict):
            raise _RowError(f"messages[{i}]", "must be an object with role and content")
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            raise _RowError(f"messages[{i}].role", f"must be one of {', '.join(ROLES)}")
        if not isinstance(content, str):
            raise _RowError(f"messages[{i}].content", "must be a string")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        raise _RowError("messages", "the last turn must be a user turn")
    return messages


def _parse_snippets(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _RowError("snippets", "must be a list")
    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, snippet in enumerate(raw):
        if not isinstance(snippet, dict):
            raise _RowError(f"snippets[{i}]", "must be an object with id and text")
        snippet_id = snippet.get("id")
        if not isinstance(snippet_id, str):
            raise _RowError(f"snippets[{i}].id", "required: a string")
        if not is_valid_id(snippet_id):
            raise _RowError(
                f"snippets[{i}].id",
                f"{snippet_id!r} is outside the ID grammar (2-24 chars of A-Z a-z 0-9 . _ space -)",
            )
        if snippet_id in seen:
            raise _RowError(f"snippets[{i}].id", f"duplicate snippet id {snippet_id!r}")
        if not isinstance(snippet.get("text"), str):
            raise _RowError(f"snippets[{i}].text", "required: a string")
        seen.add(snippet_id)
        snippets.append(dict(snippet))
    return snippets


def _parse_gold(raw: Any, provided_ids: list[str]) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _RowError("gold", "must be an object with mode and supporting_ids")
    mode = raw.get("mode")
    if mode not in GOLD_MODES:
        raise _RowError("gold.mode", f"must be one of {', '.join(GOLD_MODES)}")
    supporting = raw.get("supporting_ids", [])
    if not isinstance(supporting, list) or not all(isinstance(s, str) for s in supporting):
        raise _RowError("gold.supporting_ids", "must be a list of strings")
    if mode == "cite" and not supporting:
        raise _RowError("gold.supporting_ids", "cite mode needs at least one supporting id")
    if mode == "abstain" and supporting:
        raise _RowError("gold.supporting_ids", "abstain mode takes no supporting ids")
    unknown = [s for s in supporting if s not in provided_ids]
    if unknown:
        raise _RowError("gold.supporting_ids", f"not among the provided ids: {unknown}")
    return {**raw, "mode": mode, "supporting_ids": list(supporting)}
