"""The R4 quality filter: deterministic stages, the judge, near-duplicates and selection.

A candidate passes through the stages in ``REJECTION_REASONS`` order and is rejected at the first
it fails. The deterministic stages run before the judge, so the judge is never paid for a reply a
rule already rejects. HARD and SOFT checks both reject here (D-028): training data favors
precision, so a lexical false alarm costs one candidate, while a lexical miss would teach the
student the wrong behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from groundcite.checks import CheckResult, run_checks
from groundcite.distill import HINT_HEADER
from groundcite.evalset import Scenario
from groundcite.judge import JudgeResult
from groundcite.leakage import JACCARD_THRESHOLD, NgramIndex, text_hash

FILTER_VERSION = "f1"

GENERATION_ERROR = "generation_error"
HINT_LEAK = "hint_leak"
UNCLEAN_END = "unclean_end"
GOLD_MISMATCH = "gold_mismatch"
HARD_CHECK = "hard_check"
SOFT_CHECK = "soft_check"
JUDGE_ERROR = "judge_error"
JUDGE_SPEC = "judge_spec"
JUDGE_ROBUSTNESS = "judge_robustness"
NEAR_DUPLICATE = "near_duplicate"
NOT_SELECTED = "not_selected"
REJECTION_REASONS = (
    GENERATION_ERROR,
    HINT_LEAK,
    UNCLEAN_END,
    GOLD_MISMATCH,
    HARD_CHECK,
    SOFT_CHECK,
    JUDGE_ERROR,
    JUDGE_SPEC,
    JUDGE_ROBUSTNESS,
    NEAR_DUPLICATE,
    NOT_SELECTED,
)

MIN_ROBUSTNESS = 4
MAX_ROBUSTNESS = 5
SELECTION_RULE = (
    "one example per scenario: among candidates that pass every stage, the highest judge "
    "Robustness wins, ties go to the lowest candidate_index; scenarios in candidates-file order"
)
# Words that only make sense if the reply saw the privileged inputs (the gold label, the supporting
# IDs or the hint note), including the hint note's own wording ("near-miss", "already used up",
# "supports the next step"). "the note" is matched only as a phrase, so "a service manual note" and
# "note the idle speed" still pass; "supports the next step" only after "that", so "your reading
# supports the next step" still passes. "labeled", "designated" and "instructions" count only
# with a reference noun or "my", so "the labeled vacuum port" and "the service manual
# instructions" still pass. "gold", "hint" and "the note" count only in their reference senses
# (D-061): "gold label/answer/ID", "the hint" but not "a hint of", "the note says" but not "the
# note card" or "the note in the bulletin", so "gold-colored connector" and "any hint of a vacuum
# hiss" pass. The paraphrases ("as instructed", "meant to use", "correct citation", "teacher",
# "decoys", ...) come from validation 3.
_HINT_WORDS = re.compile(
    r"\b(?:gold (?:labels?|answers?|citations?|ids?|snippets?|references?|mode)|"
    r"supporting[_ ]ids?|privileged|hints?(?! of\b)|answer key|"
    r"the note(?! (?:card|in|on|from|about|of|pad)\b)|teacher|"
    r"grading|graded|reference answer|i was told|i(?:['\u2019]ve| have) been told|correct mode|"
    r"correct (?:ids?|citations?|references?|snippets?|answers?)|expected answer|"
    r"near[- ]miss(?:es)?|already used up|"
    r"that supports the next step|intended answer|my instructions|grader|"
    r"label(?:l)?ed (?:reference|snippet|answer|id|citation)s?|"
    r"designated (?:reference|snippet|answer|id|citation)s?|supporting reference|"
    r"(?:reference|snippet|answer|id)s? i (?:was|were|have been|got) given|"
    r"i (?:was|were|have been|got) (?:given|handed)|guidance i|"
    r"i(?:['\u2019]m| am| was) (?:meant|supposed) to|as instructed|my notes?|decoys?|"
    r"(?:flagged|marked|tagged) as (?:relevant|applicable|correct|the right|the one)|"
    r"the key (?:points to|says|shows|lists|marks))\b|" + re.escape(HINT_HEADER),
    re.IGNORECASE,
)
_GOLD_CHECKS = ("mode_matches_gold", "cited_id_supports")


@dataclass(frozen=True)
class Rejection:
    reason: str
    detail: str
    failed_checks: tuple[str, ...] = ()


@dataclass
class Candidate:
    scenario_id: str
    candidate_index: int
    reply: str
    error: str | None
    scenario: Scenario
    row: Mapping[str, Any]
    checks: list[CheckResult] = field(default_factory=list)


@dataclass
class Outcome:
    """A candidate's way through the filter: its rejection (None while it survives) and verdict."""

    candidate: Candidate
    teacher: str
    rejection: Rejection | None = None
    judge: JudgeResult | None = None


def _failed(checks: Sequence[CheckResult], severity: str | None = None) -> list[CheckResult]:
    return [c for c in checks if c.passed is False and (severity is None or c.severity == severity)]


def _describe(failed: Sequence[CheckResult]) -> str:
    return "; ".join(f"{c.name}: {c.detail}" for c in failed)


def deterministic_stage(candidate: Candidate) -> Rejection | None:
    """The first pre-judge stage the candidate fails, or None. Fills ``candidate.checks``."""
    reply = candidate.reply
    if candidate.error or not reply.strip():
        return Rejection(GENERATION_ERROR, candidate.error or "empty reply")
    if match := _HINT_WORDS.search(reply):
        return Rejection(HINT_LEAK, f"reply mentions {match.group(0)!r}")
    stripped = reply.strip()
    if stripped[-1] not in ".?!":
        return Rejection(UNCLEAN_END, f"reply ends with {stripped[-20:]!r}")
    candidate.checks = run_checks(stripped, candidate.scenario)
    failed = _failed(candidate.checks)
    names = tuple(c.name for c in failed)
    gold = [c for c in failed if c.name in _GOLD_CHECKS]
    if gold:
        return Rejection(GOLD_MISMATCH, _describe(gold), names)
    if hard := _failed(candidate.checks, "HARD"):
        return Rejection(HARD_CHECK, _describe(hard), names)
    if soft := _failed(candidate.checks, "SOFT"):
        return Rejection(SOFT_CHECK, _describe(soft), names)
    return None


def judge_stage(result: JudgeResult) -> Rejection | None:
    if result.error:
        return Rejection(JUDGE_ERROR, result.error)
    scores = f"spec_adherence {result.spec_adherence}, robustness {result.robustness}"
    if result.spec_adherence != 1:
        return Rejection(JUDGE_SPEC, f"{scores}: {result.reasoning}")
    if result.robustness < MIN_ROBUSTNESS:
        return Rejection(JUDGE_ROBUSTNESS, f"{scores} < {MIN_ROBUSTNESS}: {result.reasoning}")
    return None


def final_user_message(scenario: Scenario) -> str:
    return [m["content"] for m in scenario.messages if m["role"] == "user"][-1]


class NearDuplicateIndex:
    """Final user messages of kept examples, matched with the R23 rules."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}
        self._ngrams = NgramIndex()
        self._ids: list[str] = []

    def match(self, text: str) -> str | None:
        """A description of the kept example ``text`` duplicates, or None."""
        if (other := self._hashes.get(text_hash(text))) is not None:
            return f"same final user message as kept {other}"
        hits = self._ngrams.matches(text, JACCARD_THRESHOLD)
        if hits:
            index, score = max(hits, key=lambda h: h[1])
            return f"final user message ~ kept {self._ids[index]} (jaccard {score:.2f})"
        return None

    def add(self, scenario_id: str, text: str) -> None:
        self._hashes.setdefault(text_hash(text), scenario_id)
        self._ngrams.add(text)
        self._ids.append(scenario_id)
