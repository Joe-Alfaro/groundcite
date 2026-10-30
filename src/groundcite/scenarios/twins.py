"""In-scope twins of abstain scenarios: the dataset v2 additions (R7, D-056, D-061).

An abstain training scenario rejects every on-topic snippet it provides for the stated vehicle.
Its in-scope twin keeps the same family, snippet texts and conversation shape, and moves the
vehicle and year so that exactly one of those snippets now applies. That snippet is the gold,
cited alone, so the training data holds minimal pairs: one snippet rejected for a vehicle
outside its stated scope and cited for a vehicle inside it.

The twin's messages, snippet IDs and snippet order are regenerated with the scenario generator,
from the twin's own seed, so its final user message is not a near-duplicate of its source's.
A ``no_snippet`` source becomes a ``distractor`` twin; other categories keep theirs.

Not every source has a twin. A multi-turn source has no valid gold: its chain snippets were
already checked in earlier turns, and a near-miss is never a next chain step. A gold must not be
a near-miss whose own chain snippet is also provided (that snippet would have to be a near-miss
of its own twin). Some single-turn sources have no catalog vehicle that brings exactly one
provided snippet into scope while every other near-miss stays ruled out by its scope change.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from groundcite.scenarios.generator import (
    ID_STYLES,
    SNIPPET_KINDS,
    TRAIN_MIX,
    TRAIN_SEED,
    _alternating,
    _Draft,
    _Generator,
    _PlanItem,
)
from groundcite.scenarios.library import Family, Library, Snippet, Vehicle

# The additions' master seed defaults to the training pool's; their seed range sits above every
# split offset the generator uses, so no seed collides with a training or eval scenario (R22).
TWIN_SEED = TRAIN_SEED
TWIN_OFFSET = 600_000
ID_PREFIX = "v2"
# Categories whose abstain scenarios can have an in-scope twin with the same snippet set.
TWIN_CATEGORIES = ("no_snippet", "pressure", "multi_symptom")
NO_SNIPPET_PRESSURE = 0.15


@dataclass(frozen=True)
class _Option:
    gold: Snippet
    near: tuple[Snippet, ...]
    vehicle: Vehicle
    year: int


class _TwinGenerator(_Generator):
    def twin(
        self, source: Mapping[str, Any], item: _PlanItem, rng: random.Random
    ) -> dict[str, Any] | None:
        """The in-scope twin of ``source``, or ``None`` when it has none."""
        snippets = [self.by_text[s["text"]] for s in source["snippets"]]
        families = self._complaint_families(source, item.family, snippets)
        if source["category"] == "multi_symptom" and len(families) != 2:
            return None
        options = self._options(source, families, snippets)
        if not options:
            return None
        balance = self.year_balance.get(False, 0)
        self.want_side = -1 if balance > 0 else 1 if balance < 0 else 0
        self.gold_first = self.order_balance.get(False, 0) <= 0
        golds = sorted({o.gold for o in options}, key=lambda s: s.id)
        of_kind = [g for g in golds if g.kind == item.gold_kind] or golds
        gold = rng.choice(self._by_side(of_kind, lambda s: s))
        picked = rng.choice([o for o in options if o.gold == gold])
        for _ in range(60):
            draft = _Draft(
                [gold],
                [s for s in snippets if s != gold],
                list(picked.near),
                picked.vehicle,
                picked.year,
            )
            row = self._messages(source, item, families, draft, rng)
            if self._distinct(row):
                self._count_year_pairs(row, False)
                return {
                    "id": "",
                    "snippets": row["snippets"],
                    "messages": row["messages"],
                    "gold": row["gold"],
                    "category": item.category,
                    "tags": sorted(row["tags"]),
                    "family": item.family.name,
                }
        return None

    def _complaint_families(
        self, source: Mapping[str, Any], fam: Family, snippets: Sequence[Snippet]
    ) -> list[Family]:
        """The families whose complaints the user states: the row family, plus for a
        multi-symptom source the family of its other near-miss."""
        if source["category"] != "multi_symptom":
            return [fam]
        by_id = {s["id"]: self.by_text[s["text"]] for s in source["snippets"]}
        others = sorted(
            {by_id[i].family for i in source["gold"]["near_miss_ids"]} - {fam.name},
        )
        return [fam, *(self.lib.families[name] for name in others)]

    def _options(
        self,
        source: Mapping[str, Any],
        families: Sequence[Family],
        snippets: Sequence[Snippet],
    ) -> list[_Option]:
        """Every (gold, vehicle, year) where the gold is the one provided on-topic snippet that
        applies and every other on-topic near-miss is ruled out by its own scope change."""
        names = {f.name for f in families}
        on_topic = [s for s in snippets if s.family in names]
        old = (self._source_vehicle(source), _first_year(source))
        options = []
        for gold in on_topic:
            if gold.mismatch == "condition" or any(s.id == gold.twin_of for s in snippets):
                continue
            near = tuple(s for s in on_topic if s != gold and s.twin_of is not None)
            rivals = [s for s in on_topic if s != gold and s.mismatch != "condition"]
            for vehicle in self.lib.vehicles:
                if not all(f.fits(vehicle) for f in families):
                    continue
                for year in range(vehicle.years[0], vehicle.years[1] + 1):
                    if (
                        (vehicle, year) != old
                        and gold.applies.matches(vehicle, year)
                        and not any(s.applies.matches(vehicle, year) for s in rivals)
                        and not any(
                            s.mismatch == "vehicle" and s.twin_scope.matches(vehicle, year)
                            for s in near
                        )
                    ):
                        options.append(_Option(gold, near, vehicle, year))
        return options

    def _source_vehicle(self, source: Mapping[str, Any]) -> Vehicle | None:
        first = next(m["content"] for m in source["messages"] if m["role"] == "user")
        named = [
            v for v in self.lib.vehicles if v.body in first and any(n in first for n in v.names)
        ]
        named.sort(key=lambda v: -max(len(n) for n in v.names if n in first))
        return named[0] if named else None

    def _messages(
        self,
        source: Mapping[str, Any],
        item: _PlanItem,
        families: Sequence[Family],
        draft: _Draft,
        rng: random.Random,
    ) -> dict[str, Any]:
        tags = set(source["tags"]) - {"no_snippet", "distractor"}
        draft.tags = tags
        if item.category == "multi_symptom":
            if draft.near_miss:
                draft.tags.add("distractor")
            return self._multi_symptom_row(rng, item, draft, families[0], families[1])
        if item.category == "pressure":
            if draft.near_miss:
                draft.tags.add("distractor")
            return self._single_turn(rng, item, draft, pressure=True)
        pressure = "pressure" in tags or rng.random() < NO_SNIPPET_PRESSURE
        return self._single_turn(rng, item, draft, pressure=pressure)

    def _distinct(self, row: Mapping[str, Any]) -> bool:
        """Record the final user message, unless it nears one already seen (R23)."""
        return self._remember(row["messages"][-1]["content"])


def abstain_sources(
    pool: Iterable[Mapping[str, Any]], kept_seeds: Iterable[int]
) -> list[Mapping[str, Any]]:
    """The abstain scenarios of ``pool`` whose seed a dataset kept, in pool order."""
    kept = set(kept_seeds)
    return [row for row in pool if row["gold"]["mode"] == "abstain" and row["seed"] in kept]


def generate_twins(
    lib: Library,
    sources: Sequence[Mapping[str, Any]],
    master_seed: int,
    phrasing: Mapping[str, tuple[str, ...]],
    avoid: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """One in-scope twin per source that has one, each with ``source_id`` set.

    ``avoid`` holds every final user message no twin may near-duplicate (the eval sets' and the
    training pool's). Twin ``i`` (1-based, over all sources) gets seed
    ``master_seed * 1_000_000 + TWIN_OFFSET + i``, and the plan preferences (gold kind and ID
    style) are drawn from the base seed, so the output depends only on the seed and the inputs.
    """
    gen = _TwinGenerator(lib, mix=TRAIN_MIX, phrasing=phrasing, avoid=avoid)
    gen.year_balance = {False: 0}
    gen.order_balance = {False: 0}
    base = master_seed * 1_000_000 + TWIN_OFFSET
    plan_rng = random.Random(base)
    kinds = _alternating(plan_rng, SNIPPET_KINDS, len(sources))
    styles = _alternating(plan_rng, ID_STYLES, len(sources))
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources, 1):
        if source["category"] not in TWIN_CATEGORIES:
            continue
        category = "distractor" if source["category"] == "no_snippet" else source["category"]
        item = _PlanItem(
            category,
            lib.families[source["family"]],
            ablation=False,
            gold_kind=kinds[index - 1],
            gold_style=styles[index - 1],
        )
        seed = base + index
        row = gen.twin(source, item, random.Random(seed))
        if row is None:
            continue
        row["id"] = f"{ID_PREFIX}-{len(rows) + 1:04d}"
        row["split"] = "train"
        row["seed"] = seed
        row["source_id"] = source["id"]
        rows.append(row)
    return rows


def _first_year(row: Mapping[str, Any]) -> int | None:
    first = next(m["content"] for m in row["messages"] if m["role"] == "user")
    for token in first.replace(",", " ").split():
        if len(token) == 4 and token.isdigit() and token.startswith(("19", "20")):
            return int(token)
    return None
