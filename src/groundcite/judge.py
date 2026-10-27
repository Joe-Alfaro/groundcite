"""The LLM judge (R2, R11, R19): one versioned rubric built from SPEC.md, the judge input, strict
output parsing with a bounded retry, and HARD-check forcing (D-028).

The rubric template lives in ``rubrics/judge_rubric_<version>.md``; ``SPEC.md`` is substituted at
render time, and ``rubric_sha256`` hashes the rendered text, so a change to either is visible in
every transcript.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from groundcite.checks import HARD_CHECKS, CheckResult, forces_zero
from groundcite.config import decoding_params
from groundcite.evalset import Scenario
from groundcite.spec import load_spec

log = logging.getLogger(__name__)

RUBRIC_VERSION = "v1"
RUBRIC_PATH = Path(__file__).resolve().parent / "rubrics" / f"judge_rubric_{RUBRIC_VERSION}.md"
_SPEC_PLACEHOLDER = "{{SPEC}}"

# One line per deterministic check in the judge input; the mock judge reads these back.
CHECK_LINE_RE = re.compile(r"^CHECK (\S+) \| (HARD|SOFT) \| (PASS|FAIL|N/A)\b", re.MULTILINE)
_STATUS = {True: "PASS", False: "FAIL", None: "N/A"}

_FORMAT_REMINDER = (
    "Your previous answer could not be parsed ({error}). Grade the same reply again and answer "
    'with only one JSON object: {{"spec_adherence": 0 or 1, "robustness": an integer from 1 to 5, '
    '"failed_checks": [...], "reasoning": "..."}}. No prose and no code fences.'
)


class JudgeBackend(Protocol):
    name: str

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        system: str | None,
        params: Mapping[str, Any],
    ) -> str: ...


class JudgeParseError(ValueError):
    """The judge's answer is not a valid verdict object."""


@dataclass(frozen=True)
class JudgeVerdict:
    spec_adherence: int
    robustness: int
    reasoning: str
    failed_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgeResult:
    """The final score for one reply, after retries and HARD-check forcing."""

    spec_adherence: int
    robustness: int
    reasoning: str
    failed_checks: tuple[str, ...]
    raw: str
    forced_by_checks: bool
    error: str | None
    attempts: int
    model: str
    rubric_sha256: str

    def scores(self) -> dict[str, int]:
        """The two rubric scores, as recorded per candidate in dataset and filter reports."""
        return {"spec_adherence": self.spec_adherence, "robustness": self.robustness}


# ---------------------------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------------------------


def render_rubric(
    spec_path: str | Path | None = None, rubric_path: str | Path | None = None
) -> str:
    """The judge system prompt: the rubric template with the spec text substituted."""
    template = Path(rubric_path if rubric_path is not None else RUBRIC_PATH).read_text(
        encoding="utf-8"
    )
    return template.replace(_SPEC_PLACEHOLDER, load_spec(spec_path))


def rubric_sha256(
    spec_path: str | Path | None = None, rubric_path: str | Path | None = None
) -> str:
    return hashlib.sha256(render_rubric(spec_path, rubric_path).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Judge input
# ---------------------------------------------------------------------------------------------


def _check_line(result: CheckResult) -> str:
    cls = "HARD" if result.name in HARD_CHECKS else "SOFT"
    line = f"CHECK {result.name} | {cls} | {_STATUS[result.passed]}"
    # Details only on failures: they say what tripped the check; passes need no explanation.
    return f"{line} | {result.detail}" if result.passed is False else line


def _gold_section(gold: Mapping[str, Any] | None) -> str:
    if gold is None or gold.get("mode") is None:
        return "No gold label for this scenario. Decide the mode and support from the snippets."
    supporting = ", ".join(gold.get("supporting_ids") or []) or "none"
    return f"Gold mode: {gold['mode']}; supporting snippet ID(s): {supporting}"


def build_judge_messages(
    scenario: Scenario, reply: str, checks: Sequence[CheckResult]
) -> tuple[str, list[dict[str, str]]]:
    """Return ``(system, messages)`` for judging ``reply`` to ``scenario``."""
    turns = "\n\n".join(
        f"[turn {i}] {m['role']}:\n{m['content']}" for i, m in enumerate(scenario.messages, 1)
    )
    if scenario.snippets:
        snippets = "\n".join(f"[{s['id']}] {s.get('text', '')}" for s in scenario.snippets)
    else:
        snippets = "(No structured snippets; any references appear inline in the conversation.)"
    provided = ", ".join(scenario.provided_ids) or "none"
    sections = [
        f"## Conversation\n\n{turns}",
        f"## Reference snippets\n\nProvided IDs: {provided}\n\n{snippets}",
        f"## Reply to grade\n\n<<<\n{reply}\n>>>",
        f"## Gold label\n\n{_gold_section(scenario.gold)}",
        "## Deterministic checks\n\n" + "\n".join(_check_line(c) for c in checks),
        "Answer with only the JSON object.",
    ]
    content = "\n\n".join(sections)
    return render_rubric(), [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _load_object(text: str) -> Any:
    candidates = [text.strip()]
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise JudgeParseError("no JSON object found in the judge output")


def _int_field(obj: Mapping[str, Any], key: str, allowed: range) -> int:
    if key not in obj:
        raise JudgeParseError(f"missing {key!r}")
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        raise JudgeParseError(
            f"{key!r} must be an integer from {allowed.start} to {allowed.stop - 1}, got {value!r}"
        )
    return value


def parse_judge_output(text: str) -> JudgeVerdict:
    """Parse a judge answer: a JSON object, optionally in a ```json fence."""
    obj = _load_object(text)
    if not isinstance(obj, dict):
        raise JudgeParseError(f"expected a JSON object, got {type(obj).__name__}")
    spec = _int_field(obj, "spec_adherence", range(0, 2))
    robustness = _int_field(obj, "robustness", range(1, 6))
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise JudgeParseError("'reasoning' must be a non-blank string")
    failed = obj.get("failed_checks", [])
    if not isinstance(failed, list) or not all(isinstance(f, str) for f in failed):
        raise JudgeParseError("'failed_checks' must be a list of strings")
    return JudgeVerdict(spec, robustness, reasoning.strip(), tuple(failed))


# ---------------------------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------------------------


def _with_reminder(messages: list[dict[str, str]], error: str) -> list[dict[str, str]]:
    *head, last = messages
    reminder = _FORMAT_REMINDER.format(error=error)
    return [*head, {"role": last["role"], "content": f"{last['content']}\n\n{reminder}"}]


def judge_reply(
    scenario: Scenario,
    reply: str,
    checks: Sequence[CheckResult],
    backend: JudgeBackend,
    *,
    max_retries: int = 1,
    params: Mapping[str, Any] | None = None,
) -> JudgeResult:
    """Score ``reply`` with ``backend``, then apply HARD-check forcing.

    Unparseable output is retried up to ``max_retries`` times with a format reminder; after that
    the reply is recorded as a judge error scoring (0, 1), never dropped. Backend exceptions
    propagate, so an API outage stops the run instead of zeroing it.
    """
    system, messages = build_judge_messages(scenario, reply, checks)
    params = decoding_params(backend.name) if params is None else params
    raws: list[str] = []
    verdict: JudgeVerdict | None = None
    error = ""
    for attempt in range(max_retries + 1):
        sent = messages if attempt == 0 else _with_reminder(messages, error)
        text = backend.generate(sent, system, params)
        raws.append(text)
        try:
            verdict = parse_judge_output(text)
            break
        except JudgeParseError as exc:
            error = str(exc)
            log.warning(
                "judge %s output unparseable for %s (attempt %d): %s",
                backend.name,
                scenario.id,
                attempt + 1,
                error,
            )

    if verdict is not None:
        spec, robustness, reasoning = verdict.spec_adherence, verdict.robustness, verdict.reasoning
        failed, raw, judge_error = verdict.failed_checks, raws[-1], None
    else:
        spec, robustness, failed = 0, 1, ()
        judge_error = f"unparseable judge output after {len(raws)} attempt(s): {error}"
        reasoning = f"Judge error, scored as a failure: {judge_error}."
        raw = "\n\n--- retry ---\n\n".join(raws)

    forced = forces_zero(checks)
    if forced:
        spec, robustness = 0, min(robustness, 2)
    return JudgeResult(
        spec_adherence=spec,
        robustness=robustness,
        reasoning=reasoning,
        failed_checks=tuple(failed),
        raw=raw,
        forced_by_checks=forced,
        error=judge_error,
        attempts=len(raws),
        model=backend.name,
        rubric_sha256=hashlib.sha256(system.encode("utf-8")).hexdigest(),
    )
