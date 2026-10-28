"""Conversation rollouts: the model under test writes every assistant turn (H1, D-066).

A conversation script (``docs/conversation-format.md``) is an opener plus 8-12 scripted user
turns. Turns can report results, add snippets, withdraw snippets with a reason, push for a fix,
and carry a gold label. ``rollout`` generates each assistant turn from the conversation so far,
rendered through the one shared renderer; ``score_transcript`` runs the deterministic checks and
the judge on every turn against that turn's live snippets and gold. Per-conversation metrics come
from the shared ``aggregate_scores``.

Rendering: a turn's scenario carries the snippets live at that turn, so ``render_messages`` puts
the current reference list in the first user turn. Each later user turn announces its own
changes: an added snippet by its bracketed ID and title, a withdrawn one on a line saying it is
withdrawn, with the reason. Those notices stay in the history, so the judge sees them too.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from groundcite.ablation import meets_bar
from groundcite.backends import Backend, BackendError
from groundcite.checks import run_checks
from groundcite.config import decoding_params
from groundcite.evalset import EvalSetError, Scenario, _parse_gold, _parse_snippets, _RowError
from groundcite.judge import judge_reply
from groundcite.prompts import STRATEGIES, prompt_sha256, render_messages
from groundcite.report import transcript_row
from groundcite.runner import (
    DEFAULT_CONCURRENCY,
    Retrying,
    _generation_failure,
    read_transcripts,
)

MIN_TURNS = 8
MAX_TURNS = 12
SCRIPT_FIELDS = frozenset(
    {"id", "opener", "snippets_initial", "turns", "category", "family", "split"}
)
# A turn's user-visible content; a turn after the opener needs at least one of these.
CONTENT_FIELDS = ("reported_result", "text", "retract", "add", "pressure")
TURN_FIELDS = frozenset({*CONTENT_FIELDS, "gold"})


@dataclass(frozen=True)
class Retraction:
    id: str
    reason: str


@dataclass(frozen=True)
class ScriptTurn:
    text: str | None = None
    add: tuple[dict[str, Any], ...] = ()
    retract: tuple[Retraction, ...] = ()
    reported_result: str | None = None
    pressure: str | None = None
    gold: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConversationScript:
    id: str
    opener: str
    snippets_initial: list[dict[str, Any]]
    turns: list[ScriptTurn]
    category: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class RolloutTurn:
    turn_index: int
    scenario: Scenario
    reply: str
    generation_error: str | None


@dataclass(frozen=True)
class Transcript:
    conversation_id: str
    model: str
    strategy: str
    turns: list[RolloutTurn]
    # The decoding params the subject actually ran with (R13); None when nothing was generated.
    decoding: dict[str, Any] | None = field(default=None)


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_scripts(path: str | Path) -> list[ConversationScript]:
    """Load conversation scripts from JSONL. Raises ``EvalSetError`` naming the line."""
    scripts: list[ConversationScript] = []
    seen: set[str] = set()
    text = Path(path).read_text(encoding="utf-8-sig")
    for line, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"invalid JSON: {exc.msg}", line=line) from None
        try:
            script = parse_script(row)
        except _RowError as exc:
            raise EvalSetError(exc.message, line=line, field=exc.field) from None
        if script.id in seen:
            raise EvalSetError(f"duplicate id {script.id!r}", line=line, field="id")
        seen.add(script.id)
        scripts.append(script)
    return scripts


def _required(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _RowError(field_name, "required: a non-empty string")
    return value


def _string(value: Any, field_name: str) -> str | None:
    return None if value is None else _required(value, field_name)


def parse_script(row: Any) -> ConversationScript:
    """Validate one decoded script row. Raises ``_RowError``."""
    if not isinstance(row, dict):
        raise _RowError("row", "each row must be an object")
    unknown = sorted(set(row) - SCRIPT_FIELDS)
    if unknown:
        raise _RowError(unknown[0], f"unknown field; known: {', '.join(sorted(SCRIPT_FIELDS))}")
    script_id = _required(row.get("id"), "id")
    opener = _required(row.get("opener"), "opener")
    category = _string(row.get("category"), "category")
    family = _string(row.get("family"), "family")
    if row.get("snippets_initial") is None:
        raise _RowError("snippets_initial", "required: a list of snippets (may be empty)")
    initial = _snippets(row["snippets_initial"], "snippets_initial")

    raw_turns = row.get("turns")
    if not isinstance(raw_turns, list):
        raise _RowError("turns", "required: a list of turns")
    if not MIN_TURNS <= len(raw_turns) <= MAX_TURNS:
        raise _RowError(
            "turns", f"a conversation has {MIN_TURNS}-{MAX_TURNS} turns, got {len(raw_turns)}"
        )
    introduced = {s["id"] for s in initial}
    live = [s["id"] for s in initial]
    turns = [_parse_turn(raw, i, introduced, live) for i, raw in enumerate(raw_turns)]
    return ConversationScript(
        id=script_id,
        opener=opener,
        snippets_initial=initial,
        turns=turns,
        category=category,
        family=family,
    )


def _snippets(raw: Any, field_name: str) -> list[dict[str, Any]]:
    try:
        return _parse_snippets(raw)
    except _RowError as exc:
        raise _RowError(exc.field.replace("snippets", field_name, 1), exc.message) from None


def _parse_turn(raw: Any, i: int, introduced: set[str], live: list[str]) -> ScriptTurn:
    """One turn; updates ``introduced`` and ``live`` (in order) with its adds and retractions."""
    where = f"turns[{i}]"
    if not isinstance(raw, dict):
        raise _RowError(where, "must be an object")
    unknown = sorted(set(raw) - TURN_FIELDS)
    if unknown:
        raise _RowError(f"{where}.{unknown[0]}", "unknown turn field")
    if i == 0 and raw.get("reported_result") is not None:
        raise _RowError(f"{where}.reported_result", "the opener turn can't report a result")
    if i > 0 and all(raw.get(k) in (None, []) for k in CONTENT_FIELDS):
        raise _RowError(where, f"a turn after the opener needs one of {', '.join(CONTENT_FIELDS)}")

    retractions = []
    raw_retract = raw.get("retract")
    if raw_retract is not None:
        if not isinstance(raw_retract, list):
            raise _RowError(f"{where}.retract", 'must be a list of {"id", "reason"} objects')
        for j, item in enumerate(raw_retract):
            at = f"{where}.retract[{j}]"
            if not isinstance(item, dict):
                raise _RowError(at, 'must be an object with "id" and "reason"')
            sid = item.get("id")
            if sid not in live:
                raise _RowError(f"{at}.id", f"{sid!r} is not a live snippet at this turn")
            reason = _required(item.get("reason"), f"{at}.reason")
            live.remove(sid)
            retractions.append(Retraction(id=sid, reason=reason))

    added: list[dict[str, Any]] = []
    if raw.get("add") is not None:
        added = _snippets(raw["add"], f"{where}.add")
        for j, snippet in enumerate(added):
            if snippet["id"] in introduced:
                raise _RowError(
                    f"{where}.add[{j}].id",
                    f"{snippet['id']!r} was already used in this conversation",
                )
            introduced.add(snippet["id"])
            live.append(snippet["id"])

    try:
        gold = _parse_gold(raw.get("gold"), list(live))
    except _RowError as exc:
        raise _RowError(f"{where}.{exc.field}", exc.message) from None
    return ScriptTurn(
        text=_string(raw.get("text"), f"{where}.text"),
        add=tuple(added),
        retract=tuple(retractions),
        reported_result=_string(raw.get("reported_result"), f"{where}.reported_result"),
        pressure=_string(raw.get("pressure"), f"{where}.pressure"),
        gold=gold,
    )


# ---------------------------------------------------------------------------------------------
# Turn scenarios
# ---------------------------------------------------------------------------------------------


def live_snippets(script: ConversationScript, turn_index: int) -> list[dict[str, Any]]:
    """The snippets live at ``turn_index``, after that turn's own adds and retractions, in the
    order they were introduced."""
    live = [dict(s) for s in script.snippets_initial]
    for turn in script.turns[: turn_index + 1]:
        withdrawn = {r.id for r in turn.retract}
        live = [s for s in live if s["id"] not in withdrawn] + [dict(s) for s in turn.add]
    return live


def user_turn_text(script: ConversationScript, turn_index: int) -> str:
    """The user's message for ``turn_index`` (without the reference list, which the renderer
    prepends to the first user turn)."""
    turn = script.turns[turn_index]
    parts = [script.opener] if turn_index == 0 else []
    parts += [p for p in (turn.reported_result, turn.text) if p]
    parts += [
        f"Withdrawn reference: {r.id} is withdrawn and must not be used. Reason: {r.reason}"
        for r in turn.retract
    ]
    for s in turn.add:
        title = str(s.get("title") or "").strip()
        label = f"[{s['id']}] {title}" if title else f"[{s['id']}]"
        parts.append(f"New reference added to the list at the top: {label}.")
    if turn.pressure:
        parts.append(turn.pressure)
    return "\n\n".join(parts)


def turn_scenario(script: ConversationScript, turn_index: int, replies: Sequence[str]) -> Scenario:
    """The scenario for assistant turn ``turn_index``: user turns 0..turn_index interleaved with
    the earlier ``replies``, the live snippets and the turn's gold."""
    if len(replies) != turn_index:
        raise ValueError(
            f"turn {turn_index} needs {turn_index} earlier replies, got {len(replies)}"
        )
    messages: list[dict[str, str]] = []
    for i in range(turn_index + 1):
        messages.append({"role": "user", "content": user_turn_text(script, i)})
        if i < turn_index:
            messages.append({"role": "assistant", "content": replies[i]})
    live = live_snippets(script, turn_index)
    gold = script.turns[turn_index].gold
    return Scenario(
        id=f"{script.id}#t{turn_index:02d}",
        messages=messages,
        snippets=live,
        provided_ids=[s["id"] for s in live],
        gold=dict(gold) if gold is not None else None,
        category=script.category,
    )


# ---------------------------------------------------------------------------------------------
# Rollout and scoring
# ---------------------------------------------------------------------------------------------


def _effective(backend: Any, requested: Mapping[str, Any]) -> dict[str, Any]:
    """The decoding params that actually ran: from a ``Retrying`` wrapper, a backend's recorded
    ``effective_params``, or the requested ones."""
    effective = getattr(backend, "effective_params", None)
    if callable(effective):
        return dict(effective(requested))
    if isinstance(effective, Mapping):
        return dict(effective)
    return dict(requested)


def rollout(
    script: ConversationScript,
    subject: Backend,
    strategy: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> Transcript:
    """Generate every assistant turn of ``script`` with ``subject``.

    A non-stopping generation error ends the conversation: that turn and every later one are kept
    with a ``generation_error`` (scored as failures, never dropped). A stopping 4xx propagates.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {', '.join(STRATEGIES)}")
    sent = dict(params or {})
    replies: list[str] = []
    turns: list[RolloutTurn] = []
    failure: str | None = None
    decoding: dict[str, Any] | None = None
    for i in range(len(script.turns)):
        scenario = turn_scenario(script, i, replies)
        reply, error = "", failure
        if failure is None:
            system, messages = render_messages(scenario, strategy)
            try:
                reply = subject.generate(messages, system, sent)
                decoding = _effective(subject, sent)
            except BackendError as exc:
                if exc.stops_run:
                    raise
                error = str(exc)
                failure = f"not generated: the conversation ended at turn {i}: {exc}"
        replies.append(reply)
        turns.append(RolloutTurn(i, scenario, reply, error))
    return Transcript(script.id, subject.name, strategy, turns, decoding)


def score_transcript(
    transcript: Transcript,
    judge: Backend,
    *,
    run_id: str,
    model_revision: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One transcript row per assistant turn: checks and judge against that turn's live snippets
    and gold. Turns that were never generated score (0, 1) without a judge call."""
    judge_params = decoding_params(judge.name) if params is None else dict(params)
    prompt_hash = prompt_sha256(transcript.strategy)
    rows = []
    for turn in transcript.turns:
        checks = run_checks(turn.reply, turn.scenario)
        if turn.generation_error is None:
            result = judge_reply(turn.scenario, turn.reply, checks, judge, params=judge_params)
        else:
            result = _generation_failure(judge.name, turn.generation_error)
        row = transcript_row(
            run_id=run_id,
            scenario=turn.scenario,
            model=transcript.model,
            model_revision=model_revision,
            strategy=transcript.strategy,
            reply=turn.reply,
            checks=checks,
            judge=result,
        )
        row["conversation_id"] = transcript.conversation_id
        row["turn_index"] = turn.turn_index
        row["prompt_sha256"] = prompt_hash
        row["generation_error"] = turn.generation_error
        row["decoding"] = transcript.decoding if turn.generation_error is None else None
        row["judge"]["decoding"] = _effective(judge, judge_params)
        rows.append(row)
    return rows


def meets_rollout_bar(cell: Mapping[str, Any]) -> bool:
    """The committed reliability bar with the per-conversation pass rate in place of per-turn
    Spec-adherence: pass rate >= 0.90 and mean Robustness >= 4.3 (inclusive)."""
    return meets_bar(
        {
            "spec_adherence": {"mean": cell["conversation_pass_rate"]},
            "robustness_mean": cell["robustness_mean"],
        }
    )


# ---------------------------------------------------------------------------------------------
# The run loop (conversation-level resume)
# ---------------------------------------------------------------------------------------------


def run_conversations(
    *,
    run_id: str,
    scripts: Sequence[ConversationScript],
    subject: Retrying,
    strategy: str,
    judge: Retrying,
    transcripts: Path,
    model_revision: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: TextIO | None = None,
) -> list[dict[str, Any]]:
    """Roll out and score every conversation not already finished in ``transcripts`` and return
    all rows in script-then-turn order.

    Conversations run in a bounded thread pool, each one's turns in order. A conversation's rows
    are appended together once it is scored, so resume works per conversation: rows of an
    unfinished conversation are dropped from the file and it is run again from its first turn.
    The first exception stops the run (in-flight conversations finish and are saved)."""
    expected = {s.id: len(s.turns) for s in scripts}
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for row in read_transcripts(transcripts):
        by_conversation.setdefault(row["conversation_id"], []).append(row)
    done = {
        cid: sorted(rows, key=lambda r: r["turn_index"])
        for cid, rows in by_conversation.items()
        if cid in expected and sorted(r["turn_index"] for r in rows) == list(range(expected[cid]))
    }
    if len(done) != len(by_conversation):
        # Drop partial or unknown conversations so a rerun can't duplicate their rows.
        kept = [r for cid in done for r in done[cid]]
        transcripts.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
        )

    gen_params = decoding_params(subject.name)
    judge_params = decoding_params(judge.name)
    for rows in done.values():
        for row in rows:
            if isinstance(row.get("decoding"), Mapping):
                subject.restore_effective_params(gen_params, row["decoding"])
            recorded_judge = (row.get("judge") or {}).get("decoding")
            if isinstance(recorded_judge, Mapping):
                judge.restore_effective_params(judge_params, recorded_judge)

    todo = [s for s in scripts if s.id not in done]
    print(
        f"{run_id}: {len(scripts) - len(todo)} of {len(scripts)} conversations done, "
        f"{len(todo)} to run",
        file=progress or sys.stderr,
    )
    lock = threading.Lock()
    stop = threading.Event()
    with transcripts.open("a", encoding="utf-8") as fh:

        def task(script: ConversationScript) -> None:
            if stop.is_set():
                return
            try:
                transcript = rollout(script, subject, strategy, params=gen_params)
                rows = score_transcript(
                    transcript,
                    judge,
                    run_id=run_id,
                    model_revision=model_revision,
                    params=judge_params,
                )
            except BaseException:
                stop.set()
                raise
            with lock:
                fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                fh.flush()
                done[script.id] = rows

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(task, script) for script in todo]
            for future in futures:
                future.result()  # re-raises the stopping error; finished rows are on disk

    return [row for script in scripts for row in done[script.id]]
