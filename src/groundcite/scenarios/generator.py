"""Seeded scenario generator (R22).

One master seed fixes everything. Each split gets a disjoint seed range
(``master * 1_000_000 + SPLIT_OFFSETS[split] + index``), and every scenario is built from its
own ``random.Random(seed)``, so a scenario's content depends only on the library and its seed.
The split-level plan (which category and family each scenario gets) uses the split's base seed.

Gold labels are correct by construction:

- a cite scenario contains chain snippets that apply to the stated vehicle, and every one of
  them is listed in ``gold.supporting_ids``;
- near-miss snippets are twins of chain snippets (D-056): the same text with only the scope
  changed. A vehicle twin is ruled out by a scope field it changed (checked structurally against
  the vehicle and year), and a condition twin requires a condition every complaint of its family
  contradicts. Where the gold has a twin, it is the first near-miss, so gold and near-miss match
  on every surface feature and only the scope separates them;
- filler snippets come from unrelated systems, and "close" snippets are hand-curated per family;
- an abstain scenario often holds an out-of-scope twin pair: a chain snippet whose ``applies``
  excludes the vehicle, plus its own twin, so a pair doesn't imply cite (D-061).

Within a split, two running balances (kept apart for the ablation subset) steer the next scenario
so that no prior picks the gold out of a gold-twin pair: which side of the catalog median year
the gold's stated range lies on, and whether the gold or its twin comes first (D-061).

Final user messages are kept pairwise below the R23 near-duplicate threshold across every set
produced in one run, so the sets don't leak into each other.
"""

from __future__ import annotations

import itertools
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from groundcite.leakage import JACCARD_THRESHOLD, NgramIndex, text_hash
from groundcite.scenarios.library import Family, Library, Snippet, Vehicle

CATEGORIES = ("clean", "distractor", "no_snippet", "pressure", "multi_symptom", "multi_turn")
# Scenario mix from the requirements file, in percent.
QUOTAS = {
    "clean": 15,
    "distractor": 20,
    "no_snippet": 20,
    "pressure": 15,
    "multi_symptom": 15,
    "multi_turn": 15,
}
SPLIT_OFFSETS = {"eval_own": 100_000, "heldout": 200_000, "train": 500_000}
SET_SIZES = {"eval_own": 100, "heldout": 60}
ABLATION_SIZE = 40
ID_PREFIX = {"eval_own": "eo", "heldout": "ho", "train": "tr"}
# The training pool's master seed: never an eval master seed, so no seed range overlaps (R22).
TRAIN_SEED = 2027
# R6 needs >=300 kept examples and keeps at most one per scenario, so the pool leaves headroom.
TRAIN_POOL_SIZE = 480

MAX_SNIPPETS = 5
MAX_ATTEMPTS = 60

# Visibly synthetic ID schemes in two styles. Digits are random per scenario. The style is drawn
# independently of the snippet's kind (D-046), so a bulletin-style ID says nothing about whether
# the snippet is a bulletin, or whether it is the one to cite.
ID_SCHEMES: dict[str, tuple[Callable[[random.Random], str], ...]] = {
    "bulletin": (
        lambda r: f"TSB-{r.randint(100, 9999):04d}",
        lambda r: f"B-{r.randint(10, 24):02d}-{r.randint(1, 299):03d}",
        lambda r: f"SB {r.randint(10, 99):02d}-{r.randint(1, 999):03d}",
    ),
    "manual": (
        lambda r: f"SM {r.randint(1, 9)}.{r.randint(1, 9)}.{r.randint(1, 9)}",
        lambda r: f"SM-{r.randint(100, 9999):04d}",
        lambda r: f"SEC {r.randint(1, 19)}.{r.randint(1, 30)}",
    ),
}
ID_STYLES = ("bulletin", "manual")
SNIPPET_KINDS = ("tsb", "manual")


T = TypeVar("T")


class GenerationError(RuntimeError):
    """The library can't satisfy a scenario's constraints."""


def split_seed(master_seed: int, split: str) -> int:
    return master_seed * 1_000_000 + SPLIT_OFFSETS[split]


def category_counts(total: int) -> dict[str, int]:
    """Per-category counts for ``total`` scenarios, by largest remainder on the quotas."""
    exact = {c: total * QUOTAS[c] / 100 for c in CATEGORIES}
    counts = {c: int(exact[c]) for c in CATEGORIES}
    by_remainder = sorted(CATEGORIES, key=lambda c: (-(exact[c] - counts[c]), CATEGORIES.index(c)))
    for c in by_remainder[: total - sum(counts.values())]:
        counts[c] += 1
    return counts


@dataclass
class _Draft:
    """A scenario before IDs are assigned."""

    supporting: list[Snippet]
    others: list[Snippet]
    near_miss: list[Snippet]
    vehicle: Vehicle
    year: int
    tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _PlanItem:
    """One planned scenario. The three preferences are balanced across each category (and its
    ablation subset) so that snippet kind and ID style never predict the gold citation (D-046):
    ``gold_kind`` is the preferred kind of the (first) supporting snippet, ``near_kind`` the
    preferred kind of the first near-miss and ``gold_style`` the ID style of the first supporting
    snippet. A builder falls back to any snippet when the family has none of the preferred kind."""

    category: str
    family: Family
    ablation: bool
    gold_kind: str = "manual"
    near_kind: str = "manual"
    gold_style: str = "manual"


@dataclass(frozen=True)
class Mix:
    """Within-category gold-mode probabilities. Each is compared as ``rng.random() < p``."""

    pressure_abstain: float = 0.35
    # multi_symptom draws one variant: both supported below the first bound, one supported below
    # the second, neither above it.
    multi_symptom_both: float = 0.45
    multi_symptom_partial: float = 0.85
    multi_turn_cite: float = 0.65
    # An abstain draft holds an out-of-scope twin pair (a chain snippet and its twin, neither
    # applying) below this bound, so "two near-identical snippets" doesn't mean cite (D-061).
    abstain_pair: float = 0.9


EVAL_MIX = Mix()
# Training needs a 20-30% abstain share (R30); the eval mix runs about 31%, so training scenarios
# outside no_snippet abstain less often (expected share about 26%).
TRAIN_MIX = Mix(
    pressure_abstain=0.15, multi_symptom_partial=0.90, multi_turn_cite=0.80, abstain_pair=0.7
)


class _Generator:
    def __init__(
        self,
        lib: Library,
        *,
        mix: Mix = EVAL_MIX,
        phrasing: Mapping[str, tuple[str, ...]] | None = None,
        avoid: Iterable[str] = (),
    ) -> None:
        self.lib = lib
        self.mix = mix
        # Phrase pools that replace the library's pools key by key (training uses its own bank).
        self.phrasing = dict(phrasing or {})
        self.seen = NgramIndex()
        self.seen_hashes: set[str] = set()
        self.by_text = {s.text: s for s in lib.snippets.values()}
        self.year_side = _year_sides(lib)
        # Running sum of ``year_side`` over the informative twin pairs built so far, kept apart for
        # ablation and other items so the ablation subset is balanced on its own (D-061).
        self.year_balance: dict[bool, int] = {}
        # The side the current scenario's gold should take: +1 or -1, or 0 for no preference.
        self.want_side = 0
        # Running (gold before its twin) minus (twin before the gold), kept the same way, so the
        # order within a pair says nothing about which one is gold (D-061).
        self.order_balance: dict[bool, int] = {}
        self.gold_first = True
        # Final user messages of other sets: new finals must stay clear of them too (R23).
        for text in avoid:
            self.seen.add(text)
            self.seen_hashes.add(text_hash(text))

    def phrases(self, name: str) -> tuple[str, ...]:
        return self.phrasing.get(name) or self.lib.phrases(name)

    # --- planning ------------------------------------------------------------------------

    def plan(
        self, rng: random.Random, families: list[Family], total: int, ablation_total: int
    ) -> list[_PlanItem]:
        ablation = category_counts(ablation_total) if ablation_total else {}
        items = []
        for category, count in category_counts(total).items():
            # Cycle through the families in a per-category shuffled order so each category
            # covers every family before repeating one; the first entries form the ablation set.
            order = rng.sample(families, len(families))
            gold_kinds = _alternating(rng, SNIPPET_KINDS, count)
            near_kinds = _alternating(rng, SNIPPET_KINDS, count)
            gold_styles = _alternating(rng, ID_STYLES, count)
            for i in range(count):
                items.append(
                    _PlanItem(
                        category,
                        order[i % len(order)],
                        i < ablation.get(category, 0),
                        gold_kind=gold_kinds[i],
                        near_kind=near_kinds[i],
                        gold_style=gold_styles[i],
                    )
                )
        rng.shuffle(items)
        return items

    def build_split(
        self,
        split: str,
        master_seed: int,
        reserved: bool,
        ablation_total: int = 0,
        size: int | None = None,
    ) -> list[dict[str, Any]]:
        base = split_seed(master_seed, split)
        families = self.lib.pool(reserved=reserved)
        total = SET_SIZES[split] if size is None else size
        plan = self.plan(random.Random(base), families, total, ablation_total)
        self.year_balance = {False: 0, True: 0}
        self.order_balance = {False: 0, True: 0}
        rows = []
        for index, item in enumerate(plan, 1):
            seed = base + index
            row = self.build(item, random.Random(seed), families)
            row["id"] = f"{ID_PREFIX[split]}-{index:04d}"
            row["split"] = split
            row["seed"] = seed
            if item.ablation:
                row["_ablation"] = True
            rows.append(row)
        return rows

    # --- one scenario --------------------------------------------------------------------

    def build(self, item: _PlanItem, rng: random.Random, pool: list[Family]) -> dict[str, Any]:
        builder = getattr(self, f"_build_{item.category}")
        balance = self.year_balance.get(item.ablation, 0)
        self.want_side = -1 if balance > 0 else 1 if balance < 0 else 0
        self.gold_first = self.order_balance.get(item.ablation, 0) <= 0
        for _ in range(MAX_ATTEMPTS):
            row: dict[str, Any] | None = builder(item, rng, pool)
            if row is None:
                continue
            final = row["messages"][-1]["content"]
            if text_hash(final) in self.seen_hashes or self.seen.matches(final, JACCARD_THRESHOLD):
                continue
            self.seen.add(final)
            self.seen_hashes.add(text_hash(final))
            self._count_year_pairs(row, item.ablation)
            ordered = {
                "id": "",
                "snippets": row["snippets"],
                "messages": row["messages"],
                "gold": row["gold"],
                "category": item.category,
                "tags": sorted(row["tags"]),
                "family": item.family.name,
            }
            return ordered
        raise GenerationError(f"could not build a distinct {item.category} for {item.family.name}")

    def _build_clean(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        fam = item.family
        support = [self._gold(rng, fam, item.gold_kind)]
        others = self._sample(rng, self._unrelated([fam], pool), rng.randint(1, 3))
        draft = self._draft(rng, [fam], support, others, [])
        if draft is None:
            return None
        return self._single_turn(rng, item, draft, pressure=False)

    def _build_distractor(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        fam = item.family
        support = [self._gold(rng, fam, item.gold_kind)]
        near = self._near(rng, fam, item.near_kind, rng.randint(1, 2), twin_of=support[0])
        extra = self._sample(
            rng, self._snips(fam.close) + self._unrelated([fam], pool), rng.randint(0, 1)
        )
        draft = self._draft(rng, [fam], support, near + extra, near)
        if draft is None:
            return None
        pressure = rng.random() < 0.15
        return self._single_turn(rng, item, draft, pressure=pressure)

    def _build_no_snippet(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        draft = self._abstain_draft(item, rng, pool)
        if draft is None:
            return None
        pressure = rng.random() < 0.15
        return self._single_turn(rng, item, draft, pressure=pressure)

    def _build_pressure(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        fam = item.family
        if rng.random() < self.mix.pressure_abstain:
            draft = self._abstain_draft(item, rng, pool)
            if draft is not None:
                draft.tags.add("no_snippet")
        else:
            support = [self._gold(rng, fam, item.gold_kind)]
            near = self._near(rng, fam, item.near_kind, rng.choice((0, 1)), twin_of=support[0])
            others = self._sample(rng, self._unrelated([fam], pool), rng.randint(1, 2))
            draft = self._draft(rng, [fam], support, near + others, near)
            if draft is not None and near:
                draft.tags.add("distractor")
        if draft is None:
            return None
        return self._single_turn(rng, item, draft, pressure=True)

    def _build_multi_symptom(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        fam = item.family
        candidates = [f for f in pool if not self.lib.related_systems(f.system, fam.system)]
        second = rng.choice(candidates)
        variant = rng.random()
        if (
            variant < self.mix.multi_symptom_both
        ):  # both complaints have a supporting snippet; either is a valid citation
            # One supporting snippet of each kind, so the pair never favors a kind.
            support = [
                self._gold(rng, fam, item.gold_kind),
                self._gold(rng, second, _other(SNIPPET_KINDS, item.gold_kind)),
            ]
            near: list[Snippet] = []
        elif variant < self.mix.multi_symptom_partial:  # only one complaint is supported
            supported, missed = (fam, second) if rng.random() < 0.5 else (second, fam)
            support = [self._gold(rng, supported, item.gold_kind)]
            near = self._near(rng, missed, item.near_kind, 1)
        else:  # neither complaint is supported
            support = []
            near = self._near(rng, fam, item.near_kind, 1) + self._near(
                rng, second, _other(SNIPPET_KINDS, item.near_kind), 1
            )
        close = [
            s
            for s in self._snips(fam.close)
            if not self.lib.related_systems(s.system, second.system)
        ]
        filler_pool = close + self._unrelated([fam, second], pool)
        others = self._sample(rng, filler_pool, rng.randint(0 if support else 1, 1))
        draft = self._draft(rng, [fam, second], support, near + others, near)
        if draft is None:
            return None
        if near and support:
            draft.tags.add("distractor")
        if not support:
            draft.tags.add("no_snippet")
        if rng.random() < 0.15:
            draft.tags.add("pressure")

        complaints = [self._complaint(rng, fam), self._complaint(rng, second)]
        rng.shuffle(complaints)
        joined = (
            complaints[0]
            + " "
            + _fill(
                rng.choice(self.phrases("symptom_connector")),
                complaint=_lower_first(complaints[1]),
                Complaint=complaints[1],
            )
        )
        if rng.random() < 0.5:
            joined += " " + rng.choice(self.phrases("minor_symptom"))
        ids = self._assign_ids(rng, draft, item.gold_style)
        tail = (
            self._pressure(rng, fam)
            if "pressure" in draft.tags
            else rng.choice(self.phrases("question"))
        )
        message = _join(self._intro(rng, draft), joined, tail)
        return self._row(rng, draft, ids, [{"role": "user", "content": message}])

    def _build_multi_turn(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        fam = item.family
        picked = self._chain_walk(rng, fam, item.gold_kind)
        first, second, third = (self._snip(i) for i in picked)
        cite = rng.random() < self.mix.multi_turn_cite
        near = self._near(
            rng, fam, item.near_kind, 1 if not cite else rng.choice((0, 1)), twin_of=third
        )
        others = self._sample(
            rng, self._snips(fam.close) + self._unrelated([fam], pool), rng.choice((0, 1))
        )
        # The two chain snippets already used earlier in the conversation stay provided, but
        # they no longer support a *next* step, so only the third one is gold.
        support = [third] if cite else []
        draft = self._draft(
            rng, [fam], support, [first, second] + near + others, near, also_apply=[first, second]
        )
        if draft is None:
            return None
        if near and cite:
            draft.tags.add("distractor")
        if not cite:
            draft.tags.add("no_snippet")
        ids = self._assign_ids(rng, draft, item.gold_style)
        phrases = self.phrases
        part = rng.choice(fam.suspects)

        opening = _join(
            self._intro(rng, draft), self._complaint(rng, fam), rng.choice(phrases("question"))
        )
        messages = [_user(opening), _assistant(rng.choice(phrases("assistant_first")), first, ids)]
        if rng.random() < 0.5:  # 6 earlier turns: the user pushes back before doing the check
            messages += [
                _user(_fill(rng.choice(phrases("user_pushback")), part=part)),
                _assistant(rng.choice(phrases("assistant_hold")), first, ids),
                _user(_join(rng.choice(phrases("user_relent")), rng.choice(first.results))),
            ]
        else:
            messages.append(
                _user(_join(rng.choice(phrases("user_result_prefix")), rng.choice(first.results)))
            )
        messages.append(_assistant(rng.choice(phrases("assistant_followup")), second, ids))

        # One snippet list holds for the whole conversation, so the user never "adds" a reference.
        final = [rng.choice(second.results)]
        if rng.random() < 0.45:
            draft.tags.add("pressure")
            final.append(_fill(rng.choice(phrases("final_pressure")), part=part))
        else:
            final.append(rng.choice(phrases("final_ask")))
        messages.append(_user(_join(*final)))
        return self._row(rng, draft, ids, messages)

    # --- pieces --------------------------------------------------------------------------

    def _abstain_draft(
        self, item: _PlanItem, rng: random.Random, pool: list[Family]
    ) -> _Draft | None:
        fam = item.family
        pairs = self._out_of_scope_pairs(fam)
        if pairs and rng.random() < self.mix.abstain_pair:
            # A chain snippet and its own twin, both out of scope for the vehicle, plus at most
            # one more near-miss. The chain snippet is neither gold nor a near-miss.
            source, twin = rng.choice(pairs)
            rest = [s for s in self._snips(fam.near_miss) if s != twin]
            near = [twin, *self._sample(rng, rest, rng.randint(0, 1))]
            close = self._sample(rng, self._snips(fam.close), rng.randint(0, 1))
            others = [source, *near, *close]
            if rng.random() < 0.3:
                others += self._sample(rng, self._unrelated([fam], pool), 1)
            return self._draft(rng, [fam], [], others[:MAX_SNIPPETS], near, must_fail=[source])
        near = self._near(rng, fam, item.near_kind, rng.randint(1, min(3, len(fam.near_miss))))
        close = self._sample(rng, self._snips(fam.close), rng.randint(0, 2))
        others = near + close
        if len(others) < 2 or rng.random() < 0.3:
            others += self._sample(rng, self._unrelated([fam], pool), 1)
        return self._draft(rng, [fam], [], others[:MAX_SNIPPETS], near)

    def _out_of_scope_pairs(self, fam: Family) -> list[tuple[Snippet, Snippet]]:
        """(chain snippet, its twin) pairs of ``fam`` that some catalog vehicle and year the
        family fits puts both out of scope: the chain snippet's ``applies`` fails, and so does
        a scope field a vehicle twin changed (a condition twin shares the chain snippet's
        ``applies`` and is ruled out by its condition too)."""
        pairs = []
        for twin in self._snips(fam.near_miss):
            source = self.lib.snippets.get(twin.twin_of or "")
            if source is None:
                continue
            if any(
                not source.applies.matches(vehicle, year)
                and not (twin.mismatch == "vehicle" and twin.twin_scope.matches(vehicle, year))
                for vehicle in self.lib.vehicles
                if fam.fits(vehicle)
                for year in range(vehicle.years[0], vehicle.years[1] + 1)
            ):
                pairs.append((source, twin))
        return pairs

    def _draft(
        self,
        rng: random.Random,
        families: Sequence[Family],
        support: list[Snippet],
        others: list[Snippet],
        near: list[Snippet],
        also_apply: Sequence[Snippet] = (),
        must_fail: Sequence[Snippet] = (),
    ) -> _Draft | None:
        """Pick a vehicle and year that every complaint's family fits and every supporting (and
        ``also_apply``) snippet applies to, where a scope field each vehicle-mismatch twin
        changed excludes it and no ``must_fail`` snippet applies. ``None`` if the combination
        is impossible."""
        snippets = support + [s for s in others if s not in support]
        if not 2 <= len(snippets) <= MAX_SNIPPETS or len(set(snippets)) != len(snippets):
            return None
        must = [*support, *also_apply]
        must_not = [s for s in snippets if s.mismatch == "vehicle" and s not in must]
        options = [
            (vehicle, year)
            for vehicle in self.lib.vehicles
            if all(f.fits(vehicle) for f in families)
            for year in range(vehicle.years[0], vehicle.years[1] + 1)
            if all(s.applies.matches(vehicle, year) for s in must)
            and not any(s.twin_scope.matches(vehicle, year) for s in must_not)
            and not any(s.applies.matches(vehicle, year) for s in must_fail)
        ]
        if not options:
            return None
        vehicle, year = rng.choice(options)
        return _Draft(
            list(support), [s for s in snippets if s not in support], list(near), vehicle, year
        )

    def _single_turn(
        self, rng: random.Random, item: _PlanItem, draft: _Draft, pressure: bool
    ) -> dict[str, Any]:
        fam = item.family
        if pressure:
            draft.tags.add("pressure")
        ids = self._assign_ids(rng, draft, item.gold_style)
        intro, complaint = self._intro(rng, draft), self._complaint(rng, fam)
        body = [intro, complaint] if rng.random() < 0.7 else [complaint, intro]
        if not pressure:
            message = _join(*body, rng.choice(self.phrases("question")))
        elif rng.random() < 0.25:
            message = _join(
                rng.choice(self.phrases("pressure_lead")), *body, self._pressure(rng, fam)
            )
        else:
            message = _join(*body, self._pressure(rng, fam))
        return self._row(rng, draft, ids, [{"role": "user", "content": message}])

    def _row(
        self, rng: random.Random, draft: _Draft, ids: dict[str, str], messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        order = draft.supporting + draft.others
        rng.shuffle(order)
        for gold in draft.supporting:
            twin = next((s for s in order if s.twin_of == gold.id), None)
            if twin is not None:
                i, j = order.index(gold), order.index(twin)
                if (i < j) != self.gold_first:
                    order[i], order[j] = twin, gold
        return {
            "snippets": [s.as_scenario_snippet(ids[s.id]) for s in order],
            "messages": messages,
            "gold": {
                "mode": "cite" if draft.supporting else "abstain",
                "supporting_ids": [ids[s.id] for s in draft.supporting],
                "near_miss_ids": [ids[s.id] for s in draft.near_miss],
            },
            "tags": draft.tags,
        }

    def _assign_ids(self, rng: random.Random, draft: _Draft, gold_style: str) -> dict[str, str]:
        """IDs whose style ignores the snippet's kind. Half the snippets (a random one more when
        the count is odd) get a bulletin-style ID; the first supporting snippet gets
        ``gold_style``, which the plan balances, and the rest are spread at random."""
        snippets = draft.supporting + draft.others
        bulletin = len(snippets) // 2 + (len(snippets) % 2 and rng.random() < 0.5)
        styles = []
        if draft.supporting:
            styles.append(gold_style)
            bulletin -= gold_style == "bulletin"
        rest = ["bulletin"] * bulletin + ["manual"] * (len(snippets) - len(styles) - bulletin)
        rng.shuffle(rest)
        styles += rest
        ids: dict[str, str] = {}
        for snippet, style in zip(snippets, styles, strict=True):
            while True:
                candidate = rng.choice(ID_SCHEMES[style])(rng)
                if candidate not in ids.values():
                    ids[snippet.id] = candidate
                    break
        return ids

    def _intro(self, rng: random.Random, draft: _Draft) -> str:
        miles = rng.randint(25, 190)
        return _fill(
            rng.choice(self.phrases("vehicle_intro")),
            year=str(draft.year),
            body=draft.vehicle.body,
            engine=rng.choice(draft.vehicle.names),
            trans=rng.choice(self.phrases(f"trans_{draft.vehicle.trans}")),
            miles=f"{miles}k" if rng.random() < 0.5 else f"{miles},{rng.randint(0, 999):03d}",
        )

    def _complaint(self, rng: random.Random, fam: Family) -> str:
        return _fill(
            rng.choice(fam.complaints),
            cyl=str(rng.randint(1, 4)),
            side=rng.choice(("left", "right")),
        )

    def _pressure(self, rng: random.Random, fam: Family) -> str:
        return _fill(rng.choice(self.phrases("pressure")), part=rng.choice(fam.suspects))

    def _snip(self, snippet_id: str) -> Snippet:
        return self.lib.snippets[snippet_id]

    def _gold(self, rng: random.Random, fam: Family, kind: str) -> Snippet:
        """A chain snippet of ``kind`` when the family has one, else any chain snippet, picked
        by ``_by_side``."""
        chain = self._snips(fam.chain)
        of_kind = [s for s in chain if s.kind == kind] or chain
        return rng.choice(self._by_side(of_kind, lambda s: s))

    def _by_side(self, items: list[T], gold: Callable[[T], Snippet]) -> list[T]:
        """The items whose ``gold`` snippet has year side ``want_side``, else those whose gold has
        no year side, else all of them. With no wanted side, all of them."""
        if not self.want_side:
            return items
        side = [i for i in items if self.year_side.get(gold(i).id, 0) == self.want_side]
        neutral = [i for i in items if not self.year_side.get(gold(i).id, 0)]
        return side or neutral or items

    def _count_year_pairs(self, row: Mapping[str, Any], ablation: bool) -> None:
        """Add the year side and the order of every gold whose twin the scenario provides."""
        position = {s["text"]: i for i, s in enumerate(row["snippets"])}
        by_id = {s["id"]: s["text"] for s in row["snippets"]}
        for gold_id in row["gold"]["supporting_ids"]:
            gold = self.by_text[by_id[gold_id]]
            twins = [t for t in self.lib.snippets.values() if t.twin_of == gold.id]
            here = [position[t.text] for t in twins if t.text in position]
            if not here:
                continue
            side = self.year_side.get(gold.id, 0)
            self.year_balance[ablation] = self.year_balance.get(ablation, 0) + side
            first = 1 if position[gold.text] < here[0] else -1
            self.order_balance[ablation] = self.order_balance.get(ablation, 0) + first

    def _near(
        self,
        rng: random.Random,
        fam: Family,
        kind: str,
        k: int,
        twin_of: Snippet | None = None,
    ) -> list[Snippet]:
        """``k`` near-misses. The first is the twin of ``twin_of`` when ``fam`` has one, else
        one of ``kind`` when the family has one."""
        if k <= 0:
            return []
        snippets = self._snips(fam.near_miss)
        twins = [s for s in snippets if twin_of is not None and s.twin_of == twin_of.id]
        first = rng.choice(twins or [s for s in snippets if s.kind == kind] or snippets)
        return [first, *self._sample(rng, [s for s in snippets if s != first], k - 1)]

    def _chain_walk(self, rng: random.Random, fam: Family, kind: str) -> list[str]:
        """Three chain snippet ids in ``chain_order``, preferring walks whose last step (the
        multi-turn gold) is of ``kind``."""
        walks = [
            walk
            for p in itertools.permutations(fam.chain, 3)
            if self._some_vehicle(fam, walk := fam.order_chain(list(p)))
        ]
        preferred = [w for w in walks if self._snip(w[-1]).kind == kind] or walks
        return rng.choice(self._by_side(preferred, lambda w: self._snip(w[-1])))

    def _some_vehicle(self, fam: Family, ids: Sequence[str]) -> bool:
        """True when some catalog vehicle and year the family fits has every snippet apply."""
        snippets = self._snips(ids)
        return any(
            all(s.applies.matches(vehicle, year) for s in snippets)
            for vehicle in self.lib.vehicles
            if fam.fits(vehicle)
            for year in range(vehicle.years[0], vehicle.years[1] + 1)
        )

    def _snips(self, ids: Sequence[str]) -> list[Snippet]:
        return [self.lib.snippets[i] for i in ids]

    def _unrelated(self, families: Sequence[Family], pool: Sequence[Family]) -> list[Snippet]:
        """Snippets from the pool whose system is unrelated to every given family's system."""
        names = {f.name for f in pool}
        excluded = {f.name for f in families}
        return [
            s
            for _, s in sorted(self.lib.snippets.items())
            if s.family in names
            and s.family not in excluded
            and not any(self.lib.related_systems(s.system, f.system) for f in families)
        ]

    @staticmethod
    def _sample(rng: random.Random, items: Sequence[Snippet], k: int) -> list[Snippet]:
        return rng.sample(list(items), min(k, len(items)))


def generate(lib: Library, master_seed: int) -> dict[str, list[dict[str, Any]]]:
    """The ``ablation``, ``eval_own`` and ``heldout`` sets for ``master_seed``.

    ``eval_own`` and ``ablation`` use training families; ``heldout`` uses only reserved
    families and only snippets from reserved families. ``ablation`` is a quota-balanced subset
    of ``eval_own`` whose rows are byte-identical to their ``eval_own`` rows.
    """
    gen = _Generator(lib)
    eval_own = gen.build_split(
        "eval_own", master_seed, reserved=False, ablation_total=ABLATION_SIZE
    )
    heldout = gen.build_split("heldout", master_seed, reserved=True)
    ablation = [row for row in eval_own if row.get("_ablation")]
    for row in eval_own + heldout:
        row.pop("_ablation", None)
    return {"ablation": ablation, "eval_own": eval_own, "heldout": heldout}


def generate_train(
    lib: Library,
    master_seed: int,
    size: int,
    phrasing: Mapping[str, tuple[str, ...]],
    avoid: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """The training scenario pool: ``size`` scenarios from training families only.

    ``phrasing`` replaces the library's phrase pools key by key, so training user turns use their
    own pressure and multi-turn phrasing instead of the eval templates. ``avoid`` holds the eval
    sets' final user messages; no training final is a near-duplicate of one (R23).
    """
    gen = _Generator(lib, mix=TRAIN_MIX, phrasing=phrasing, avoid=avoid)
    return gen.build_split("train", master_seed, reserved=False, size=size)


def catalog_median_year(lib: Library) -> int:
    """The median of every catalog vehicle's model years (the lower one on a tie)."""
    years = [y for v in lib.vehicles for y in range(v.years[0], v.years[1] + 1)]
    return int(statistics.median_low(years))


def _year_sides(lib: Library) -> dict[str, int]:
    """For each chain snippet with a twin where exactly one of the two stated year ranges
    covers the catalog median year: +1 when the chain snippet (the gold) covers it, -1 when the
    twin does. A reader could otherwise learn "of a pair, cite the one covering the middle
    years" without reading the user's year, so the generator keeps the two sides balanced."""
    median = catalog_median_year(lib)

    def covers(s: Snippet) -> bool:
        return s.applies.years is None or s.applies.years[0] <= median <= s.applies.years[1]

    sides = {}
    for twin in lib.snippets.values():
        gold = lib.snippets.get(twin.twin_of or "")
        if gold is not None and covers(gold) != covers(twin):
            sides[gold.id] = 1 if covers(gold) else -1
    return sides


def _alternating(rng: random.Random, values: tuple[str, str], count: int) -> list[str]:
    """``count`` values that alternate in pairs (each pair in random order), so every prefix is
    balanced to within one."""
    out: list[str] = []
    while len(out) < count:
        out += rng.sample(values, 2)
    return out[:count]


def _other(values: tuple[str, str], value: str) -> str:
    return values[1] if value == values[0] else values[0]


def _fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def _lower_first(text: str) -> str:
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant(template: str, snippet: Snippet, ids: dict[str, str]) -> dict[str, str]:
    step = snippet.step or ""
    content = _fill(template, Step=step, step=_lower_first(step), id=ids[snippet.id])
    return {"role": "assistant", "content": content}
