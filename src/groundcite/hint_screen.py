"""The semantic hint-leak screen (R4, D-062): a short classification prompt to the judge model.

The filter's ``_HINT_WORDS`` regex catches the obvious ways a teacher reply can show that it saw
the privileged inputs (the gold label, the supporting IDs or the hint note); paraphrases such as
"the intended reference here is ..." get through it. Every candidate the rubric judge would see is
screened first with this prompt, which asks only whether the reply's wording refers to privileged
information. It is a separate versioned file with its own hash, so the eval rubric, and with it the
one ``rubric_sha256`` shared by eval, ablation and the dataset judge, stays untouched (R13).

The screen input holds the conversation, the snippets and the reply, never the gold label: the
screen judges wording, not correctness.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundcite.config import decoding_params
from groundcite.evalset import Scenario
from groundcite.judge import JudgeBackend, JudgeParseError, context_sections, load_json_object

log = logging.getLogger(__name__)

HINT_SCREEN_VERSION = "v1"
HINT_SCREEN_PATH = (
    Path(__file__).resolve().parent / "rubrics" / f"hint_screen_{HINT_SCREEN_VERSION}.md"
)

_FORMAT_REMINDER = (
    "Your previous answer could not be parsed ({error}). Screen the same reply again and answer "
    'with only one JSON object: {{"meta_leak": true or false, "reason": "..."}}. No prose and no '
    "code fences."
)


@dataclass(frozen=True)
class ScreenVerdict:
    meta_leak: bool
    reason: str


@dataclass(frozen=True)
class ScreenResult:
    """The screen's verdict for one reply, after retries.

    When every attempt is unparseable, ``error`` is set and ``meta_leak`` is True: callers that
    look only at the flag still fail closed.
    """

    meta_leak: bool
    reason: str
    raw: str
    error: str | None
    attempts: int
    model: str
    prompt_sha256: str


def render_hint_screen(path: str | Path | None = None) -> str:
    """The screen system prompt."""
    return Path(path if path is not None else HINT_SCREEN_PATH).read_text(encoding="utf-8")


def hint_screen_sha256(path: str | Path | None = None) -> str:
    return hashlib.sha256(render_hint_screen(path).encode("utf-8")).hexdigest()


def build_hint_screen_messages(scenario: Scenario, reply: str) -> tuple[str, list[dict[str, str]]]:
    """Return ``(system, messages)`` for screening ``reply`` to ``scenario``."""
    sections = [
        *context_sections(scenario),
        f"## Reply to screen\n\n<<<\n{reply}\n>>>",
        "Answer with only the JSON object.",
    ]
    return render_hint_screen(), [{"role": "user", "content": "\n\n".join(sections)}]


def parse_hint_screen_output(text: str) -> ScreenVerdict:
    """Parse a screen answer: a JSON object with a bool ``meta_leak`` and a non-blank ``reason``."""
    obj = load_json_object(text)
    if not isinstance(obj, dict):
        raise JudgeParseError(f"expected a JSON object, got {type(obj).__name__}")
    if "meta_leak" not in obj:
        raise JudgeParseError("missing 'meta_leak'")
    leak = obj["meta_leak"]
    if not isinstance(leak, bool):
        raise JudgeParseError(f"'meta_leak' must be true or false, got {leak!r}")
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeParseError("'reason' must be a non-blank string")
    return ScreenVerdict(leak, reason.strip())


def _with_reminder(messages: list[dict[str, str]], error: str) -> list[dict[str, str]]:
    *head, last = messages
    reminder = _FORMAT_REMINDER.format(error=error)
    return [*head, {"role": last["role"], "content": f"{last['content']}\n\n{reminder}"}]


def screen_reply(
    scenario: Scenario,
    reply: str,
    backend: JudgeBackend,
    *,
    max_retries: int = 1,
    params: Mapping[str, Any] | None = None,
) -> ScreenResult:
    """Screen ``reply`` with ``backend``.

    Unparseable output is retried up to ``max_retries`` times with a format reminder; after that
    the result carries an ``error``. Backend exceptions propagate, as in ``judge_reply``.
    """
    system, messages = build_hint_screen_messages(scenario, reply)
    params = decoding_params(backend.name) if params is None else params
    raws: list[str] = []
    verdict: ScreenVerdict | None = None
    error = ""
    for attempt in range(max_retries + 1):
        sent = messages if attempt == 0 else _with_reminder(messages, error)
        text = backend.generate(sent, system, params)
        raws.append(text)
        try:
            verdict = parse_hint_screen_output(text)
            break
        except JudgeParseError as exc:
            error = str(exc)
            log.warning(
                "hint screen %s output unparseable for %s (attempt %d): %s",
                backend.name,
                scenario.id,
                attempt + 1,
                error,
            )

    sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    if verdict is not None:
        return ScreenResult(
            verdict.meta_leak, verdict.reason, raws[-1], None, len(raws), backend.name, sha
        )
    screen_error = f"unparseable hint screen output after {len(raws)} attempt(s): {error}"
    raw = "\n\n--- retry ---\n\n".join(raws)
    return ScreenResult(True, screen_error, raw, screen_error, len(raws), backend.name, sha)
