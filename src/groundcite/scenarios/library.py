"""Load and validate the hand-written scenario library under ``data/``.

The library is four files:

- ``data/snippets/*.jsonl``: synthetic reference snippets (one JSON object per line).
- ``data/families.toml``: the generic vehicle catalog, symptom families and related systems.
- ``data/phrasing.toml``: phrasing pools for user and assistant turns.
- ``data/reserved_families.txt``: families reserved for the held-out set.

Validation fails fast with every problem listed, so a bad edit to the data can't silently
produce scenarios with wrong gold labels.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

KINDS = ("tsb", "manual")
MISMATCH_KINDS = ("vehicle", "condition")
MAX_SYSTEMS = 8
# A near-miss twin changes only its scope: an engine list, a year range, a transmission, a body
# or one condition phrase (D-056).
MAX_TWIN_TOKEN_EDITS = 10
SCOPE_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")

# Words an on-spec reply must not contain: the R21 sequencing and repair markers, plus the
# wider deterministic-check lexicon (D-021). Steps and assistant templates are spliced into
# replies, so they are held to the same lexicon. Kept local until the checks module is on main.
REPLY_BANNED = re.compile(
    r"\b(then|next|also (check|inspect|test|measure|verify|look|confirm|scan)|after (that|this)|"
    r"afterwards?|second(ly)?|finally|and check|once that's done|"
    r"if (that|this|it)('s| is| looks| reads| seems)? (fine|ok|okay|good|normal)|"
    r"if (that|this|it) checks? out|replac\w*|install(ing)? (a )?new|swap\w*|the fix is|"
    r"repair (the|your|it)|you('ll)? need (a )?new|it('s| is) your|"
    r"it('s| is) (probably|likely|most likely|almost certainly|definitely)|"
    r"the (most )?likely (cause|culprit) is|(is|are) (bad|faulty|defective|failed))\b",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"\{(\w+)\}")
COMPLAINT_PLACEHOLDERS = frozenset({"cyl", "side"})


class LibraryError(ValueError):
    """The scenario library is inconsistent. The message lists every problem found."""


@dataclass(frozen=True)
class Vehicle:
    body: str
    engine: str
    names: tuple[str, ...]
    trans: str
    years: tuple[int, int]


@dataclass(frozen=True)
class Applies:
    """Structured applicability of a snippet. ``None`` fields don't constrain."""

    engines: tuple[str, ...] | None = None
    years: tuple[int, int] | None = None
    bodies: tuple[str, ...] | None = None
    trans: tuple[str, ...] | None = None

    def matches(self, vehicle: Vehicle, year: int) -> bool:
        """True when no set field excludes ``vehicle`` in ``year``."""
        if self.engines is not None and vehicle.engine not in self.engines:
            return False
        if self.years is not None and not self.years[0] <= year <= self.years[1]:
            return False
        if self.bodies is not None and vehicle.body not in self.bodies:
            return False
        return self.trans is None or vehicle.trans in self.trans

    def changed_from(self, other: Applies) -> Applies:
        """Only the fields that differ from ``other`` (the rest don't constrain)."""
        return Applies(
            engines=self.engines if self.engines != other.engines else None,
            years=self.years if self.years != other.years else None,
            bodies=self.bodies if self.bodies != other.bodies else None,
            trans=self.trans if self.trans != other.trans else None,
        )


# A ``requires`` value: one allowed string or a list of them.
Requirement = str | tuple[str, ...]


def allowed(value: Requirement) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


@dataclass(frozen=True)
class Snippet:
    id: str
    kind: str
    system: str
    family: str
    title: str
    text: str
    applies: Applies
    mismatch: str | None
    step: str | None
    results: tuple[str, ...]
    # Operating conditions the text states, as structured facts the complaint must match.
    requires: Mapping[str, Requirement] = field(default_factory=dict, hash=False, compare=False)
    # Near-misses are twins of a gold-capable snippet: its text with only ``scope_edits`` applied.
    twin_of: str | None = None
    scope_edits: tuple[tuple[str, str], ...] = ()
    # The applicability fields the twin changed. A vehicle twin is ruled out for a vehicle only
    # when one of these excludes it, so the scope change is what makes it a near-miss.
    twin_scope: Applies = Applies()

    def as_scenario_snippet(self, snippet_id: str) -> dict[str, str]:
        """The snippet as it appears in a scenario: re-assigned ID, no generator metadata."""
        return {
            "id": snippet_id,
            "kind": self.kind,
            "system": self.system,
            "title": self.title,
            "text": self.text,
        }


@dataclass(frozen=True)
class Family:
    name: str
    system: str
    symptom: str
    chain: tuple[str, ...]
    near_miss: tuple[str, ...]
    close: tuple[str, ...]
    suspects: tuple[str, ...]
    complaints: tuple[str, ...]
    # One table of stated facts per complaint (same order), matched against ``requires``.
    complaint_facts: tuple[Mapping[str, str], ...] = ()
    # Ordered stages of chain snippet ids. Steps in one stage are independent; a later-stage
    # step only makes sense once earlier stages are done, and its reported result makes every
    # earlier-stage step moot.
    chain_order: tuple[tuple[str, ...], ...] = ()
    # Transmissions the complaint makes sense on (``None``: any). A harsh-shift complaint, for
    # example, only fits an automatic, even in a scenario no chain snippet constrains.
    trans: tuple[str, ...] | None = None

    def fits(self, vehicle: Vehicle) -> bool:
        return self.trans is None or vehicle.trans in self.trans

    def stage(self, snippet_id: str) -> int:
        return next(i for i, stage in enumerate(self.chain_order) if snippet_id in stage)

    def order_chain(self, picked: list[str]) -> list[str]:
        """``picked`` sorted by ``chain_order`` stage; ties keep their picked order."""
        return sorted(picked, key=self.stage)

    def contradicts(self, requires: Mapping[str, Requirement], complaint: int) -> list[str]:
        """Required keys that ``complaint`` states with a value ``requires`` doesn't allow."""
        facts = self.complaint_facts[complaint]
        return [k for k, v in requires.items() if k in facts and facts[k] not in allowed(v)]

    def unmet(self, requires: Mapping[str, Requirement], complaint: int) -> list[str]:
        """Required keys ``complaint`` doesn't state with an allowed value (unstated counts)."""
        facts = self.complaint_facts[complaint]
        return [k for k, v in requires.items() if facts.get(k) not in allowed(v)]


@dataclass(frozen=True)
class Library:
    snippets: Mapping[str, Snippet]
    families: Mapping[str, Family]
    vehicles: tuple[Vehicle, ...]
    related: frozenset[frozenset[str]]
    reserved: frozenset[str]
    phrasing: Mapping[str, tuple[str, ...]]

    def related_systems(self, a: str, b: str) -> bool:
        return a == b or frozenset((a, b)) in self.related

    def systems(self) -> list[str]:
        return sorted({s.system for s in self.snippets.values()})

    def pool(self, reserved: bool) -> list[Family]:
        """Training families (``reserved=False``) or held-out families, in name order."""
        return [
            f for name, f in sorted(self.families.items()) if (name in self.reserved) == reserved
        ]

    def phrases(self, name: str) -> tuple[str, ...]:
        return self.phrasing[name]


def load_reserved(path: Path) -> frozenset[str]:
    """Family names from a reserved-families file: one per line, ``#`` comments ignored."""
    names = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return frozenset(names)


def load_library(data_dir: Path = DATA_DIR) -> Library:
    """Load and validate the library under ``data_dir``. Raises ``LibraryError``."""
    problems: list[str] = []
    snippets = _with_twin_scopes(
        _load_snippets(sorted((data_dir / "snippets").glob("*.jsonl")), problems)
    )
    families_doc = tomllib.loads((data_dir / "families.toml").read_text(encoding="utf-8"))
    phrasing_doc = tomllib.loads((data_dir / "phrasing.toml").read_text(encoding="utf-8"))
    library = Library(
        snippets=snippets,
        families={f.name: f for f in (_family(raw) for raw in families_doc["family"])},
        vehicles=tuple(_vehicle(raw) for raw in families_doc["vehicle"]),
        related=frozenset(frozenset(pair) for pair in families_doc["related_systems"]),
        reserved=load_reserved(data_dir / "reserved_families.txt"),
        phrasing={key: tuple(value) for key, value in phrasing_doc.items()},
    )
    problems.extend(validate(library))
    if problems:
        raise LibraryError("scenario library is invalid:\n  " + "\n  ".join(problems))
    return library


def load_train_phrasing(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """The training phrase bank (``data/train_phrasing.toml``): pools that replace the library's
    eval pools key by key. Raises ``LibraryError`` if an assistant template uses a banned word."""
    doc = tomllib.loads((path or DATA_DIR / "train_phrasing.toml").read_text(encoding="utf-8"))
    phrasing = {key: tuple(value) for key, value in doc.items()}
    problems = [
        f"train phrasing {key}: reply template uses a banned word: {template!r}"
        for key in ("assistant_first", "assistant_followup", "assistant_hold")
        for template in phrasing.get(key, ())
        if REPLY_BANNED.search(template)
    ]
    if problems:
        raise LibraryError("training phrase bank is invalid:\n  " + "\n  ".join(problems))
    return phrasing


def validate(lib: Library) -> list[str]:
    """Every consistency problem in ``lib`` (empty when valid)."""
    problems: list[str] = []
    for snippet in lib.snippets.values():
        problems.extend(_validate_snippet(snippet))
        problems.extend(_validate_twin(lib, snippet))
    if len(lib.systems()) > MAX_SYSTEMS:
        problems.append(f"{len(lib.systems())} systems, at most {MAX_SYSTEMS}: {lib.systems()}")
    unknown_reserved = lib.reserved - set(lib.families)
    if unknown_reserved:
        problems.append(f"reserved families not in families.toml: {sorted(unknown_reserved)}")
    for family in lib.families.values():
        problems.extend(_validate_family(lib, family))
    for name in sorted(lib.snippets):
        if lib.snippets[name].family not in lib.families:
            problems.append(f"{name}: unknown family {lib.snippets[name].family!r}")
    for key in ("assistant_first", "assistant_followup", "assistant_hold"):
        for template in lib.phrasing.get(key, ()):
            if REPLY_BANNED.search(template):
                problems.append(f"phrasing {key}: reply template uses a banned word: {template!r}")
    return problems


def _validate_snippet(s: Snippet) -> list[str]:
    problems = []
    if s.kind not in KINDS:
        problems.append(f"{s.id}: kind {s.kind!r} not in {KINDS}")
    if s.mismatch is not None and s.mismatch not in MISMATCH_KINDS:
        problems.append(f"{s.id}: mismatch {s.mismatch!r} not in {MISMATCH_KINDS}")
    if s.mismatch == "vehicle" and s.applies == Applies():
        problems.append(f"{s.id}: a vehicle-mismatch near-miss needs structured 'applies'")
    # The structured applicability must be visible in the text the model reads.
    for engine in s.applies.engines or ():
        if engine not in s.text:
            problems.append(f"{s.id}: applies engine {engine!r} not stated in the text")
    if s.applies.years is not None:
        span = f"{s.applies.years[0]}-{s.applies.years[1]}"
        if span not in s.text:
            problems.append(f"{s.id}: applies years {span} not stated in the text")
    for body in s.applies.bodies or ():
        if body not in s.text:
            problems.append(f"{s.id}: applies body {body!r} not stated in the text")
    for trans in s.applies.trans or ():
        if trans not in s.text:
            problems.append(f"{s.id}: applies transmission {trans!r} not stated in the text")
    if s.step is not None and REPLY_BANNED.search(s.step):
        problems.append(f"{s.id}: step uses a banned reply word: {s.step!r}")
    return problems


def _validate_family(lib: Library, family: Family) -> list[str]:
    problems = []
    name = family.name
    if len(family.chain) < 3:
        problems.append(f"{name}: needs at least 3 chain snippets for multi_turn")
    if len(family.near_miss) < 2:
        problems.append(f"{name}: needs at least 2 near-miss snippets")
    if not family.complaints or not family.suspects:
        problems.append(f"{name}: needs complaints and suspects")
    reserved = name in lib.reserved
    for role, ids in (("chain", family.chain), ("near_miss", family.near_miss)):
        for snippet_id in ids:
            snippet = lib.snippets.get(snippet_id)
            if snippet is None:
                problems.append(f"{name}: unknown {role} snippet {snippet_id!r}")
                continue
            if snippet.family != name:
                problems.append(f"{name}: {role} snippet {snippet_id} belongs to {snippet.family}")
            if role == "chain" and (snippet.step is None or not snippet.results):
                problems.append(f"{name}: chain snippet {snippet_id} needs 'step' and 'results'")
            if role == "near_miss" and snippet.mismatch is None:
                problems.append(f"{name}: near-miss {snippet_id} needs 'mismatch'")
    for snippet_id in family.close:
        snippet = lib.snippets.get(snippet_id)
        if snippet is None:
            problems.append(f"{name}: unknown close snippet {snippet_id!r}")
        elif snippet.family == name:
            problems.append(f"{name}: close snippet {snippet_id} is from the same family")
        elif (snippet.family in lib.reserved) != reserved:
            # Held-out scenarios must not share snippet text with training-family scenarios.
            problems.append(f"{name}: close snippet {snippet_id} crosses the reserved split")
    flat = [snippet_id for stage in family.chain_order for snippet_id in stage]
    if len(family.chain_order) < 2 or not all(family.chain_order):
        problems.append(f"{name}: chain_order needs at least 2 non-empty stages")
    if len(flat) != len(set(flat)) or set(flat) != set(family.chain):
        problems.append(f"{name}: chain_order must list every chain snippet exactly once")
    if family.trans is not None and not any(family.fits(v) for v in lib.vehicles):
        problems.append(f"{name}: no catalog vehicle has a transmission in {family.trans}")
    for complaint in family.complaints:
        extra = set(PLACEHOLDER.findall(complaint)) - COMPLAINT_PLACEHOLDERS
        if extra:
            problems.append(f"{name}: complaint has unknown placeholders {sorted(extra)}")
    if len(family.complaint_facts) != len(family.complaints) or not all(family.complaint_facts):
        problems.append(f"{name}: needs one non-empty complaint_facts table per complaint")
        return problems
    # Gold integrity: a chain snippet's conditions hold for every complaint, and a condition
    # near-miss's conditions fail for every complaint.
    for i in range(len(family.complaints)):
        for snippet_id in family.chain:
            snippet = lib.snippets.get(snippet_id)
            if snippet is not None and (unmet := family.unmet(snippet.requires, i)):
                problems.append(f"{name}: complaint {i} doesn't state {unmet} for {snippet_id}")
        for snippet_id in family.near_miss:
            snippet = lib.snippets.get(snippet_id)
            if (
                snippet is not None
                and snippet.mismatch == "condition"
                and not family.contradicts(snippet.requires, i)
            ):
                problems.append(f"{name}: complaint {i} doesn't contradict near-miss {snippet_id}")
    return problems


def _validate_twin(lib: Library, s: Snippet) -> list[str]:
    """A near-miss is its twin's text with only the declared scope edits applied (D-056)."""
    if s.mismatch is None:
        return []
    twin = lib.snippets.get(s.twin_of or "")
    if twin is None or twin.step is None or twin.family != s.family:
        return [f"{s.id}: twin_of {s.twin_of!r} is not a chain snippet of {s.family}"]
    problems = []
    text = twin.text
    for old, new in s.scope_edits:
        if old == new or text.count(old) != 1:
            problems.append(f"{s.id}: scope edit {old!r} -> {new!r} doesn't apply exactly once")
            return problems
        text = text.replace(old, new)
    if text != s.text:
        problems.append(f"{s.id}: text is not {twin.id} with its scope edits applied")
    a, b = SCOPE_TOKEN.findall(twin.text.lower()), SCOPE_TOKEN.findall(s.text.lower())
    if not 0 < _edit_distance(a, b) <= MAX_TWIN_TOKEN_EDITS:
        problems.append(f"{s.id}: more than {MAX_TWIN_TOKEN_EDITS} token edits from {twin.id}")
    if s.mismatch == "vehicle" and s.twin_scope == Applies():
        problems.append(f"{s.id}: a vehicle twin must change engines, years, bodies or trans")
    if s.mismatch == "condition" and dict(s.requires) == dict(twin.requires):
        problems.append(f"{s.id}: a condition twin must change a required condition")
    return problems


def _edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _with_twin_scopes(snippets: dict[str, Snippet]) -> dict[str, Snippet]:
    out = {}
    for snippet_id, s in snippets.items():
        twin = snippets.get(s.twin_of or "")
        if twin is not None:
            s = replace(s, twin_scope=s.applies.changed_from(twin.applies))
        out[snippet_id] = s
    return out


def _load_snippets(paths: Iterable[Path], problems: list[str]) -> dict[str, Snippet]:
    snippets: dict[str, Snippet] = {}
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                snippet = _snippet(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                problems.append(f"{where}: invalid snippet: {exc!r}")
                continue
            if snippet.id in snippets:
                problems.append(f"{where}: duplicate snippet id {snippet.id!r}")
            snippets[snippet.id] = snippet
    return snippets


def _snippet(raw: dict[str, Any]) -> Snippet:
    applies = raw.get("applies", {})
    years = applies.get("years")
    return Snippet(
        id=str(raw["id"]),
        kind=str(raw["kind"]),
        system=str(raw["system"]),
        family=str(raw["family"]),
        title=str(raw["title"]),
        text=str(raw["text"]),
        applies=Applies(
            engines=_opt_tuple(applies.get("engines")),
            years=(int(years[0]), int(years[1])) if years else None,
            bodies=_opt_tuple(applies.get("bodies")),
            trans=_opt_tuple(applies.get("trans")),
        ),
        mismatch=raw.get("mismatch"),
        step=raw.get("step"),
        results=tuple(raw.get("results", ())),
        requires={
            str(k): v if isinstance(v, str) else tuple(v)
            for k, v in (raw.get("requires") or {}).items()
        },
        twin_of=raw.get("twin_of"),
        scope_edits=tuple((str(old), str(new)) for old, new in raw.get("scope_edits") or ()),
    )


def _opt_tuple(value: list[str] | None) -> tuple[str, ...] | None:
    return tuple(value) if value else None


def _vehicle(raw: dict[str, Any]) -> Vehicle:
    return Vehicle(
        body=raw["body"],
        engine=raw["engine"],
        names=tuple(raw["names"]),
        trans=raw["trans"],
        years=(int(raw["years"][0]), int(raw["years"][1])),
    )


def _family(raw: dict[str, Any]) -> Family:
    return Family(
        name=raw["name"],
        system=raw["system"],
        symptom=raw["symptom"],
        chain=tuple(raw["chain"]),
        near_miss=tuple(raw["near_miss"]),
        close=tuple(raw["close"]),
        suspects=tuple(raw["suspects"]),
        complaints=tuple(raw["complaints"]),
        complaint_facts=tuple(dict(facts) for facts in raw.get("complaint_facts", ())),
        chain_order=tuple(tuple(str(i) for i in stage) for stage in raw.get("chain_order", ())),
        trans=_opt_tuple(raw.get("trans")),
    )
