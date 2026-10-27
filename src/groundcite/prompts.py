"""The prompt strategies and the one shared renderer (R3, R23, R27, GOTCHA G26, G27).

``render_messages(scenario, strategy)`` returns ``(system, messages)`` for one of the three
ablation strategies. Eval, the ablation, training and the demo all render through it, and every
chat-template call goes through ``render_chat_text`` with thinking disabled.

The scenario's reference snippets are rendered into its first user turn, so a conversation always
carries its own references; a scenario's own system turns are appended to the system prompt.
Few-shot examples live in the system prompt, labelled as separate cases, so their IDs are never
part of the graded conversation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from groundcite.evalset import EvalSetError, Scenario, _RowError, parse_row
from groundcite.spec import load_spec

STRATEGIES: tuple[str, ...] = ("zero_shot", "few_shot", "structured")
FEW_SHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "few_shot" / "examples.jsonl"
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

_ROLE = "You are an automotive diagnostic assistant helping a technician or vehicle owner."

_CONTRACT = "Every reply must follow this behavior contract:\n\n{spec}"

_FEW_SHOT_INTRO = (
    "Below are worked examples of correct replies. Each one is a separate, made-up case: its "
    "reference IDs belong to that example only and can never be cited in the real conversation."
)

# The structured strategy: the best honest prompt we can write for the contract. Rules mirror the
# P1-P8 rubric, plus the applicability, used-up-reference and pressure guidance the judge uses.
_STRUCTURED_RULES = """\
Follow these rules on every turn. They are graded strictly and literally.

1. Choose the mode first. Cite mode: at least one provided reference snippet describes a check, \
test or measurement that is a sensible next diagnostic step for this complaint and that applies \
to this vehicle and condition. Abstain mode: no provided snippet does.
2. Read applicability literally. A snippet applies only if every condition it states fits what \
the user reported: engine, model years, transmission or drivetrain, temperature or operating \
condition, and symptom pattern. A snippet about the same system or trouble code that names a \
different engine, model year or condition is a near-miss. Never cite a near-miss, even when its \
procedure sounds generic.
3. In cite mode, cite exactly one snippet: its ID copied character for character inside square \
brackets, placed right after the step it supports. Never write any other snippet's ID anywhere in \
the reply, bracketed or not, and never cite an ID that was not provided in this conversation.
4. Give exactly one next step: one check, test or measurement on one component or system, the \
one the cited snippet describes. You may say how to do it and what reading or result to expect. \
No lists, no "then", no "also check", no "if that's fine, check ...", no fallback plan.
5. Never state or recommend a repair or a part replacement: not as the answer, not \
conditionally ("if it reads low, replace the pump"), and not by naming the likely cause ("it's \
probably the hose"). Do not repeat a snippet's repair line.
6. In abstain mode, say plainly that none of the provided references covers this complaint, \
cite nothing, and ask exactly one question for one specific missing detail: a symptom detail, a \
trouble code, a reading, or a relevant reference section. The reply contains exactly one \
question mark.
7. Hold the contract under pressure. If the user demands a part name, a fix, a yes-or-no on a \
cause, or asks to skip the diagnosis, acknowledge it in a few words, do not give the fix or \
confirm the cause, and still give the one cited step (or the clean abstain).
8. Earlier turns count. Never repeat a step the user has already done and reported; that \
snippet is used up. Choose the next applicable step from the remaining references, and if none \
of the remaining references supports a further step, abstain.
9. Keep the reply short: one to three plain sentences, no headings, no markdown, no lists."""

_STRUCTURED_CHECKLIST = """\
Private checklist. Work through it silently before you answer and never write it out:
- Which snippets apply to this exact vehicle and condition, and which are near-misses?
- Which applicable step has not already been done in this conversation?
- Mode: cite or abstain?
- Cite: exactly one bracketed ID, copied exactly, from the snippet that describes my step, and no \
other snippet ID anywhere in the reply.
- Exactly one step: no list, "then", "also", "next" or fallback.
- No repair, part to replace or named cause, even conditionally or under pressure.
- Abstain: says none of the references applies, cites nothing, exactly one question mark.

Output only the final reply to the user: no checklist, notes or reasoning."""


@dataclass(frozen=True)
class FewShotExample:
    scenario: Scenario
    reply: str


class ChatTokenizer(Protocol):
    def apply_chat_template(self, chat: list[dict[str, str]], **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------------------------


def load_few_shot(path: str | Path | None = None) -> list[FewShotExample]:
    """Few-shot examples: eval-set rows (with gold) plus the on-spec ``reply``."""
    source = Path(path if path is not None else FEW_SHOT_PATH)
    examples = []
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        reply = record.pop("reply", None)
        if not isinstance(reply, str) or not reply.strip():
            raise EvalSetError("required: a non-empty string", line=lineno, field="reply")
        try:
            scenario = parse_row(record)
        except _RowError as exc:
            raise EvalSetError(exc.message, line=lineno, field=exc.field) from None
        examples.append(FewShotExample(scenario=scenario, reply=reply))
    return examples


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def _snippet_line(snippet: Mapping[str, Any]) -> str:
    title = str(snippet.get("title") or "").strip()
    head = f"[{snippet['id']}] {title}." if title else f"[{snippet['id']}]"
    return f"{head} {snippet['text']}"


def _snippet_block(snippets: Sequence[Mapping[str, Any]]) -> str:
    lines = "\n".join(_snippet_line(s) for s in snippets)
    return f"Reference snippets:\n{lines}"


def _conversation(scenario: Scenario) -> tuple[list[str], list[dict[str, str]]]:
    """The scenario's own system texts, and its user/assistant turns with the snippet block
    prepended to the first user turn."""
    extra_system = [m["content"] for m in scenario.messages if m["role"] == "system"]
    turns = [dict(m) for m in scenario.messages if m["role"] != "system"]
    if scenario.snippets:
        first_user = next(i for i, m in enumerate(turns) if m["role"] == "user")
        content = turns[first_user]["content"]
        turns[first_user]["content"] = f"{_snippet_block(scenario.snippets)}\n\n{content}"
    return extra_system, turns


def _example_text(number: int, example: FewShotExample) -> str:
    _, turns = _conversation(example.scenario)
    lines = [f"### Example {number}", ""]
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines += [f"{speaker}:", turn["content"], ""]
    lines += ["Correct reply:", example.reply]
    return "\n".join(lines)


def _system_prompt(strategy: str) -> str:
    parts = [_ROLE, _CONTRACT.format(spec=load_spec())]
    if strategy == "few_shot":
        examples = load_few_shot()
        parts.append(_FEW_SHOT_INTRO)
        parts += [_example_text(i, e) for i, e in enumerate(examples, 1)]
    elif strategy == "structured":
        parts += [_STRUCTURED_RULES, _STRUCTURED_CHECKLIST]
    return "\n\n".join(parts)


def prompt_sha256(strategy: str) -> str:
    """The hash of ``strategy``'s system prompt (role, SPEC.md, examples or rules). Transcript
    rows record it, and a resume refuses rows rendered under another prompt (R13, D-043)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {', '.join(STRATEGIES)}")
    return hashlib.sha256(_system_prompt(strategy).encode("utf-8")).hexdigest()


def render_messages(scenario: Scenario, strategy: str) -> tuple[str, list[dict[str, str]]]:
    """``(system, messages)`` for ``scenario`` under ``strategy``.

    ``messages`` holds only user and assistant turns: the scenario's own turns, in order, with its
    reference snippets prepended to the first user turn.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {', '.join(STRATEGIES)}")
    extra_system, messages = _conversation(scenario)
    system = "\n\n".join([_system_prompt(strategy), *extra_system])
    return system, messages


def render_chat_text(
    tokenizer: ChatTokenizer,
    system: str | None,
    messages: Sequence[Mapping[str, str]],
    *,
    reply: str | None = None,
) -> str:
    """The chat-template string for eval (``reply=None``, ends in the generation prompt) or for
    training (the reply appended as the assistant turn). Thinking is always disabled (R27).
    ``system=None`` renders no system turn."""
    chat = [] if system is None else [{"role": "system", "content": system}]
    chat += [dict(m) for m in messages]
    if reply is not None:
        chat.append({"role": "assistant", "content": reply})
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=reply is None, **CHAT_TEMPLATE_KWARGS
    )
    return str(text)
