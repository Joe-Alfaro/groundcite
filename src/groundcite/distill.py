"""The teacher prompt for distillation (R1, R4, GOTCHA G22).

The teacher sees what the student will see (the scenario rendered by the shared
``render_messages(..., "zero_shot")``) plus two things the student never does: a versioned
instruction template built around SPEC.md (loaded by path, never copied) and a privileged note
with the gold mode and supporting ID. Training targets are filtered so the note can't leak
(``groundcite.filter``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from groundcite.evalset import Scenario
from groundcite.prompts import render_messages
from groundcite.spec import load_spec

TEACHER_PROMPT_VERSION = "t1"
TEACHER_PROMPT_PATH = (
    Path(__file__).resolve().parent / "templates" / f"teacher_prompt_{TEACHER_PROMPT_VERSION}.md"
)
HINT_HEADER = "PRIVILEGED NOTE FOR THE TEACHER"
_SPEC_PLACEHOLDER = "{{SPEC}}"

# One opening style per candidate index, so the k candidates for a scenario differ in shape and
# the dataset doesn't teach one stock opening. Part of the prompt hash.
STYLE_NUDGES: tuple[str, ...] = (
    "For this reply, go straight to the step in the first sentence.",
    "For this reply, open with a brief acknowledgement of what the user just said.",
    "For this reply, say in a few words why this check comes first, before or after the step.",
    "For this reply, use the user's own wording for the symptom where it fits.",
)


def render_template(
    spec_path: str | Path | None = None, prompt_path: str | Path | None = None
) -> str:
    """The teacher instructions with the spec text substituted."""
    template = Path(prompt_path if prompt_path is not None else TEACHER_PROMPT_PATH).read_text(
        encoding="utf-8"
    )
    return template.replace(_SPEC_PLACEHOLDER, load_spec(spec_path)).strip()


def teacher_prompt_sha256(
    spec_path: str | Path | None = None, prompt_path: str | Path | None = None
) -> str:
    """Hash of everything scenario-independent the teacher sees: the rendered template (so a spec
    change shows) and the style nudges."""
    parts = [render_template(spec_path, prompt_path), *STYLE_NUDGES]
    return hashlib.sha256("\n\0".join(parts).encode("utf-8")).hexdigest()


def _bracketed(ids: Sequence[str]) -> str:
    return ", ".join(f"[{i}]" for i in ids)


def render_hint(scenario: Scenario) -> str:
    """The privileged note: the gold mode and, in cite mode, the supporting snippet ID(s)."""
    gold = scenario.gold or {}
    lines = [f"## {HINT_HEADER}"]
    if gold.get("mode") == "cite":
        supporting = list(gold.get("supporting_ids") or [])
        if len(supporting) == 1:
            lines.append(
                f"Correct mode: cite. The snippet that supports the next step is "
                f"{_bracketed(supporting)}. Give the step it describes and cite only that ID."
            )
        else:
            lines.append(
                f"Correct mode: cite. Each of these snippets supports a next step: "
                f"{_bracketed(supporting)}. Pick one, give the step it describes and cite only "
                "that ID."
            )
        lines.append("Every other snippet is a near-miss, unrelated or already used up.")
    else:
        lines.append(
            "Correct mode: abstain. No provided snippet supports a next step for this complaint "
            "(they are near-misses, unrelated or already used up). Cite nothing, write no snippet "
            "ID, say no provided reference covers it, and ask one question for one missing detail."
        )
    return "\n".join(lines)


def render_teacher_prompt(
    scenario: Scenario,
    *,
    spec_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    variant: int | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """``(system, messages)`` for the teacher. ``messages`` are exactly the student's rendered
    turns; the system prompt holds the instructions, the scenario's own system text, an optional
    style nudge (``variant``, e.g. the candidate index) and the privileged note last."""
    extra_system = [m["content"] for m in scenario.messages if m["role"] == "system"]
    _, messages = render_messages(scenario, "zero_shot")
    parts = [render_template(spec_path, prompt_path)]
    if extra_system:
        parts.append("## System text in this conversation\n\n" + "\n\n".join(extra_system))
    if variant is not None:
        parts.append(STYLE_NUDGES[variant % len(STYLE_NUDGES)])
    parts.append(render_hint(scenario))
    return "\n\n".join(parts), messages
