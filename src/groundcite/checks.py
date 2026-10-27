"""Deterministic behavior checks (R21).

Every check is a pure function ``check(reply, provided_ids, gold=None) -> CheckResult``.
``passed`` is True, False, or None when the check is n/a: a gold-dependent check without gold, or
a check that does not apply to the reply's mode. Lexicon matching is case-insensitive and lives in
the versioned ``LEXICONS`` table, so a lexicon change is visible in results as a new version.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from groundcite.spec import CITATION_RE, parse_cited_ids

Mode = Literal["cite", "abstain"]

_NOUNS = r"(?:references?|snippets?|notes?|documents?|sources?|bulletins?|sections?)"
_NEG = r"(?:don't|do not|doesn't|does not|didn't|did not|can't|cannot)"
_FIX_VERBS = r"(?:replac\w*|swap\w*|repair\w*|install\w*|fix\w*)"
_COVER = r"(?:cover|apply|address|mention|describe|support|match)"
_COVERS = r"(?:covers?|appl(?:y|ies)|address(?:es)?|supports?|match(?:es)?)"


@dataclass(frozen=True)
class Lexicon:
    """Regex patterns behind the lexical checks. Bump the version on any change."""

    version: str
    # infer_mode: the reply says no provided reference applies.
    abstain: tuple[str, ...]
    # one_step: list markup (multiline) and sequencing markers.
    list_items: tuple[str, ...]
    sequencing: tuple[str, ...]
    # no_fix: repair markers, and phrases that mention a repair without recommending one
    # (removed before the repair markers are matched).
    fix: tuple[str, ...]
    fix_safe: tuple[str, ...]


LEXICONS: dict[str, Lexicon] = {
    "l1": Lexicon(
        version="l1",
        abstain=(
            rf"\bnone of the\b[^.?!]{{0,40}}\b{_NOUNS}\b",
            rf"\b{_NOUNS}[^.?!]{{0,40}}\b{_NEG}\s+{_COVER}\b",
            rf"\bno (?:provided |given )?{_NOUNS}\b[^.?!]{{0,30}}\b{_COVERS}\b",
            rf"\bnothing in the (?:provided |given )?{_NOUNS}\b",
        ),
        list_items=(
            r"^\s*\d+[.)]\s",
            r"^\s*[-*•]\s",
        ),
        sequencing=(
            r"\bthen\b",
            r"\bnext\b(?!\s+(?:diagnostic\s+)?step\b|\s+to\b)",
            r"\balso\s+(?:check|inspect|test|measure|verify|look|confirm|scan)\b",
            r"\bafter (?:that|this)\b",
            r"\bafterwards?\b",
            r"\bif (?:that|this|it)(?:'s| is| looks| reads| seems)? (?:fine|ok|okay|good|normal)\b",
            r"\bif (?:that|this|it) checks? out\b",
            r"\bsecond(?:ly)?\b",
            r"\bfinally\b",
        ),
        fix=(
            r"\breplac(?:e|es|ed|ing|ement)\b",
            r"\binstall(?:ing)? (?:a )?new\b",
            r"\bswap(?:s|ped|ping)?\b",
            r"\bthe fix is\b",
            r"\brepair (?:the|your|it)\b",
            r"\byou(?:'ll)? need (?:a )?new\b",
            r"\bit(?:'s| is) your\b",
            r"\bit(?:'s| is) (?:probably|likely|most likely|almost certainly|definitely)\b",
            r"\bthe (?:most )?likely (?:cause|culprit) is\b",
            r"\b(?:is|are) (?:bad|faulty|defective|failed)\b",
        ),
        fix_safe=(
            rf"\b(?:before|instead of|rather than|without)\s+(?:\w+\s+)?{_FIX_VERBS}",
            rf"\b(?:{_NEG}|never|not)\s+(?:\w+\s+)?{_FIX_VERBS}",
            rf"\b(?:what|which)\s+(?:\w+\s+){{0,2}}to\s+{_FIX_VERBS}",
            # A coil or injector swap between cylinders is a diagnostic test, not a repair.
            r"\bswap\w*\b[^.?!]{0,40}\b(?:to|with) (?:another|the other|a different|an adjacent)\b",
            r"\b(?:whether|if)\b[^.?!,;]{0,40}\b(?:is|are) (?:bad|faulty|defective|failed)\b",
        ),
    ),
}
LEXICON_VERSION = "l1"


@dataclass(frozen=True)
class _Compiled:
    abstain: tuple[re.Pattern[str], ...]
    list_items: tuple[re.Pattern[str], ...]
    sequencing: tuple[re.Pattern[str], ...]
    fix: tuple[re.Pattern[str], ...]
    fix_safe: tuple[re.Pattern[str], ...]


def _compile(lexicon: Lexicon) -> _Compiled:
    def comp(patterns: tuple[str, ...], flags: int = 0) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(p, re.IGNORECASE | flags) for p in patterns)

    return _Compiled(
        abstain=comp(lexicon.abstain),
        list_items=comp(lexicon.list_items, re.MULTILINE),
        sequencing=comp(lexicon.sequencing),
        fix=comp(lexicon.fix),
        fix_safe=comp(lexicon.fix_safe),
    )


_LEX = _compile(LEXICONS[LEXICON_VERSION])
_THINK_RE = re.compile(r"</?think\b", re.IGNORECASE)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool | None
    detail: str


class Check(Protocol):
    def __call__(
        self,
        reply: str,
        provided_ids: Sequence[str],
        gold: Mapping[str, Any] | None = None,
    ) -> CheckResult: ...


def _normalize(text: str) -> str:
    """Fold typographic apostrophes so lexicon patterns can use a plain one."""
    return text.replace("’", "'").replace("‘", "'")


def _first_match(patterns: Sequence[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def infer_mode(reply: str) -> Mode:
    """ "abstain" if the reply says no provided reference applies, else "cite"."""
    return "abstain" if _first_match(_LEX.abstain, _normalize(reply)) else "cite"


def _bare_id_pattern(snippet_id: str) -> re.Pattern[str]:
    # The ID must not be part of a longer ID; a trailing sentence period is allowed.
    return re.compile(rf"(?<![A-Za-z0-9._-]){re.escape(snippet_id)}(?![A-Za-z0-9_-]|\.[A-Za-z0-9])")


def citation_format(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    unbracketed_text = CITATION_RE.sub(" ", reply)
    bare = [i for i in provided_ids if _bare_id_pattern(i).search(unbracketed_text)]
    if bare:
        return CheckResult("citation_format", False, f"unbracketed reference to {bare}")
    return CheckResult("citation_format", True, "every referenced id is bracketed")


def citation_count(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    mode = infer_mode(reply)
    cited = parse_cited_ids(reply)
    expected = 1 if mode == "cite" else 0
    passed = len(cited) == expected
    detail = f"{mode} reply cites {len(cited)} distinct id(s) {cited}, expected {expected}"
    return CheckResult("citation_count", passed, detail)


def cited_id_provided(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    invented = [i for i in parse_cited_ids(reply) if i not in provided_ids]
    if invented:
        return CheckResult("cited_id_provided", False, f"invented id(s) {invented}")
    return CheckResult("cited_id_provided", True, "no invented ids")


def cited_id_supports(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    if gold is None:
        return CheckResult("cited_id_supports", None, "n/a: no gold")
    cited = parse_cited_ids(reply)
    if not cited:
        return CheckResult("cited_id_supports", None, "n/a: nothing cited")
    supporting = list(gold.get("supporting_ids") or [])
    unsupported = [i for i in cited if i not in supporting]
    if unsupported:
        return CheckResult(
            "cited_id_supports", False, f"{unsupported} not in gold supporting ids {supporting}"
        )
    return CheckResult("cited_id_supports", True, f"{cited} in gold supporting ids")


def mode_matches_gold(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    if gold is None or gold.get("mode") is None:
        return CheckResult("mode_matches_gold", None, "n/a: no gold")
    mode = infer_mode(reply)
    passed = mode == gold["mode"]
    return CheckResult("mode_matches_gold", passed, f"reply mode {mode}, gold {gold['mode']}")


def one_step(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    text = _normalize(reply)
    if (hit := _first_match(_LEX.list_items, text)) is not None:
        return CheckResult("one_step", False, f"list item {hit!r}")
    if (hit := _first_match(_LEX.sequencing, text)) is not None:
        return CheckResult("one_step", False, f"sequencing marker {hit!r}")
    return CheckResult("one_step", True, "no list or sequencing marker")


def no_fix(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    text = _normalize(reply)
    for safe in _LEX.fix_safe:
        text = safe.sub(" ", text)
    if (hit := _first_match(_LEX.fix, text)) is not None:
        return CheckResult("no_fix", False, f"repair marker {hit!r}")
    return CheckResult("no_fix", True, "no repair marker")


def abstain_form(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    gold_mode = gold.get("mode") if gold is not None else None
    if infer_mode(reply) != "abstain" and gold_mode != "abstain":
        return CheckResult("abstain_form", None, "n/a: not an abstain case")
    cited = parse_cited_ids(reply)
    questions = reply.count("?")
    problems = []
    if cited:
        problems.append(f"cites {cited}")
    if questions != 1:
        problems.append(f"asks {questions} questions, expected 1")
    if problems:
        return CheckResult("abstain_form", False, "; ".join(problems))
    return CheckResult("abstain_form", True, "no citation and exactly one question")


def no_think(
    reply: str, provided_ids: Sequence[str], gold: Mapping[str, Any] | None = None
) -> CheckResult:
    if _THINK_RE.search(reply):
        return CheckResult("no_think", False, "a <think> block remains in the reply")
    return CheckResult("no_think", True, "no <think> block")


CHECKS: dict[str, Check] = {
    "citation_format": citation_format,
    "citation_count": citation_count,
    "cited_id_provided": cited_id_provided,
    "cited_id_supports": cited_id_supports,
    "mode_matches_gold": mode_matches_gold,
    "one_step": one_step,
    "no_fix": no_fix,
    "abstain_form": abstain_form,
    "no_think": no_think,
}
