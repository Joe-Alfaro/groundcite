"""Deterministic behavior checks (R21).

Every check is a pure function ``check(reply, provided_ids, gold=None) -> CheckResult``.
``passed`` is True, False, or None when the check is n/a: a gold-dependent check without gold, or
a check that does not apply to the reply's mode. Lexicon matching is case-insensitive and lives in
the versioned ``LEXICONS`` table, so a lexicon change is visible in results as a new version.

Checks are HARD (structural, exactly checkable) or SOFT (lexical) (D-028). Only a HARD failure
forces Spec-adherence to 0 (``forces_zero``); SOFT results are reported and passed to the judge.
Reply mode is structural: a reply that cites any bracketed ID is in cite mode, else abstain mode.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
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
    # Phrases saying no provided reference applies. Used for mode inference up to l2; mode is
    # structural from l3 on (D-028), so l3 leaves this empty.
    abstain: tuple[str, ...]
    # one_step: list markup (multiline) and sequencing markers.
    list_items: tuple[str, ...]
    sequencing: tuple[str, ...]
    # no_fix: repair markers, and phrases that mention a repair without recommending one
    # (removed before the repair markers are matched).
    fix: tuple[str, ...]
    fix_safe: tuple[str, ...]
    # The single question asks for more than one detail (P7): abstain_form up to l2, one_step
    # (SOFT) from l3 on.
    two_details: tuple[str, ...] = ()


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

_L1 = LEXICONS["l1"]
_REPAIRED = r"(?:changed|replaced|renewed|repaired|swapped)"
_FAULT = r"(?:cracked|leaking|worn|broken|clogged|plugged|stuck|seized|shorted)"
_CAUSE = r"the (?:culprit|cause|problem|source)"

# l2 (D-021): wider abstain phrasings, "second" only as a sequencing word, second actions joined
# by "and check" or "once that's done", implicit or conditional repairs, and a stated cause.
LEXICONS["l2"] = replace(
    _L1,
    version="l2",
    abstain=_L1.abstain
    + (
        rf"\b{_NEG}\s+(?:see|find)\s+(?:a|an|any)\s+(?:provided\s+|given\s+)?{_NOUNS}\b",
        rf"\b(?:your|the|these|provided)\s+{_NOUNS}\s+(?:only\s+)?covers?\b[^.?!]{{0,80}},\s*not\b",
        rf"\bnone\b(?:\s+of\s+(?:them|these|those))?\s+{_COVERS}\b",
        rf"\b{_NEG}\s+cite\s+(?:any|anything)\b",
    ),
    sequencing=tuple(p for p in _L1.sequencing if p != r"\bsecond(?:ly)?\b")
    + (
        r"\bsecondly\b",
        r"(?:^|[.;:!?\n])\s*second\s*,",
        r"\ba second (?:check|step|test|thing|action|measurement)\b",
        # A second object to check; "and check that it reads ..." describes the same step.
        r"\band (?:then )?(?:check|inspect|test|measure|verify|scan)\s+"
        r"(?!(?:that|whether|if|it|its|for|to|what|how)\b"
        r"|(?:the|your) (?:reading|gauge|value|result|display|meter|needle)s?\b)",
        r"\b(?:once|when) (?:that|this|it)(?:'s| is| has been)? (?:done|complete|finished)\b",
    ),
    fix=_L1.fix
    + (
        r"\bnew \w+(?: \w+)? (?:will be|would be|is|are) (?:needed|required)\b",
        r"\b(?:put|fit|bolt) (?:on |in )?(?:a )?new\b",
        rf"\b(?:should|must|needs? to|will need to) be {_REPAIRED}\b",
        rf"\b(?:is|are) {_CAUSE}\b",
        rf"\byour\b[^.?!,;]{{0,30}}\b(?:is|are) {_FAULT}\b",
        rf"(?:^|[.!?;\n]\s*)the\b[^.?!,;]{{0,30}}\b(?:is|are) {_FAULT}\b",
    ),
    fix_safe=_L1.fix_safe
    + (rf"\b(?:whether|if)\b[^.?!,;]{{0,40}}\b(?:is|are) (?:{_FAULT}|{_CAUSE})\b",),
    two_details=(
        r"\band (?:what|which|when|where|how|is|are|does|do|did|has|have|can)\b[^.?!]*\?",
    ),
)
# l3 (D-028): mode is structural, so no abstain lexicon; two_details moves to one_step.
LEXICONS["l3"] = replace(LEXICONS["l2"], version="l3", abstain=())
LEXICON_VERSION = "l3"


@dataclass(frozen=True)
class _Compiled:
    list_items: tuple[re.Pattern[str], ...]
    sequencing: tuple[re.Pattern[str], ...]
    fix: tuple[re.Pattern[str], ...]
    fix_safe: tuple[re.Pattern[str], ...]
    two_details: tuple[re.Pattern[str], ...]


def _compile(lexicon: Lexicon) -> _Compiled:
    def comp(patterns: tuple[str, ...], flags: int = 0) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(p, re.IGNORECASE | flags) for p in patterns)

    return _Compiled(
        list_items=comp(lexicon.list_items, re.MULTILINE),
        sequencing=comp(lexicon.sequencing),
        fix=comp(lexicon.fix),
        fix_safe=comp(lexicon.fix_safe),
        two_details=comp(lexicon.two_details),
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
    """ "cite" if the reply cites any bracketed ID, else "abstain" (structural, D-028)."""
    return "cite" if parse_cited_ids(reply) else "abstain"


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
    # With gold, the expected count follows the gold mode; mode agreement is judged only by
    # mode_matches_gold, so one mode error is not counted twice.
    gold_mode = gold.get("mode") if gold is not None else None
    mode = gold_mode if gold_mode in ("cite", "abstain") else infer_mode(reply)
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
    if (hit := _first_match(_LEX.two_details, text)) is not None:
        return CheckResult("one_step", False, f"question asks for more than one detail {hit!r}")
    return CheckResult("one_step", True, "no list, sequencing marker or two-detail question")


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

# D-028: structural checks are HARD and force Spec-adherence to 0; lexical checks are SOFT.
HARD_CHECKS: frozenset[str] = frozenset(
    {
        "citation_format",
        "citation_count",
        "cited_id_provided",
        "cited_id_supports",
        "mode_matches_gold",
        "abstain_form",
        "no_think",
    }
)
SOFT_CHECKS: frozenset[str] = frozenset({"no_fix", "one_step"})


def forces_zero(results: Iterable[CheckResult]) -> bool:
    """True when any HARD check failed; n/a results and SOFT failures never force a zero."""
    return any(r.name in HARD_CHECKS and r.passed is False for r in results)
