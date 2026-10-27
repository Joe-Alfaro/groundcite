"""Seeded scenario generator (R22).

One master seed fixes everything. Each split gets a disjoint seed range
(``master * 1_000_000 + SPLIT_OFFSETS[split] + index``), and every scenario is built from its
own ``random.Random(seed)``, so a scenario's content depends only on the library and its seed.
The split-level plan (which category and family each scenario gets) uses the split's base seed.

Gold labels are correct by construction:

- a cite scenario contains chain snippets that apply to the stated vehicle, and every one of
  them is listed in ``gold.supporting_ids``;
- near-miss snippets either don't apply to the stated vehicle (checked structurally against
  the snippet's ``applies``) or describe a condition the family's complaints contradict;
- filler snippets come from unrelated systems, and "close" snippets are hand-curated per family.

Final user messages are kept pairwise below the R23 near-duplicate threshold across every set
produced in one run, so the sets don't leak into each other.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

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

MAX_SNIPPETS = 5
MAX_ATTEMPTS = 60

# Visibly synthetic ID schemes per snippet kind. Digits are random per scenario.
ID_SCHEMES: dict[str, tuple[Callable[[random.Random], str], ...]] = {
    "tsb": (
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
    category: str
    family: Family
    ablation: bool


class _Generator:
    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.seen = NgramIndex()
        self.seen_hashes: set[str] = set()

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
            for i in range(count):
                items.append(
                    _PlanItem(category, order[i % len(order)], i < ablation.get(category, 0))
                )
        rng.shuffle(items)
        return items

    def build_split(
        self, split: str, master_seed: int, reserved: bool, ablation_total: int = 0
    ) -> list[dict[str, Any]]:
        base = split_seed(master_seed, split)
        families = self.lib.pool(reserved=reserved)
        plan = self.plan(random.Random(base), families, SET_SIZES[split], ablation_total)
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
        for _ in range(MAX_ATTEMPTS):
            row: dict[str, Any] | None = builder(item.family, rng, pool)
            if row is None:
                continue
            final = row["messages"][-1]["content"]
            if text_hash(final) in self.seen_hashes or self.seen.matches(final, JACCARD_THRESHOLD):
                continue
            self.seen.add(final)
            self.seen_hashes.add(text_hash(final))
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
        self, fam: Family, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        support = [self._snip(rng.choice(fam.chain))]
        others = self._sample(rng, self._unrelated([fam], pool), rng.randint(1, 3))
        draft = self._draft(rng, [fam], support, others, [])
        if draft is None:
            return None
        return self._single_turn(rng, fam, draft, pressure=False)

    def _build_distractor(
        self, fam: Family, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        support = [self._snip(rng.choice(fam.chain))]
        near = self._sample(rng, self._snips(fam.near_miss), rng.randint(1, 2))
        extra = self._sample(
            rng, self._snips(fam.close) + self._unrelated([fam], pool), rng.randint(0, 1)
        )
        draft = self._draft(rng, [fam], support, near + extra, near)
        if draft is None:
            return None
        pressure = rng.random() < 0.15
        return self._single_turn(rng, fam, draft, pressure=pressure)

    def _build_no_snippet(
        self, fam: Family, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        draft = self._abstain_draft(fam, rng, pool)
        if draft is None:
            return None
        pressure = rng.random() < 0.15
        return self._single_turn(rng, fam, draft, pressure=pressure)

    def _build_pressure(
        self, fam: Family, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        if rng.random() < 0.35:
            draft = self._abstain_draft(fam, rng, pool)
            if draft is not None:
                draft.tags.add("no_snippet")
        else:
            support = [self._snip(rng.choice(fam.chain))]
            near = self._sample(rng, self._snips(fam.near_miss), rng.choice((0, 1)))
            others = self._sample(rng, self._unrelated([fam], pool), rng.randint(1, 2))
            draft = self._draft(rng, [fam], support, near + others, near)
            if draft is not None and near:
                draft.tags.add("distractor")
        if draft is None:
            return None
        return self._single_turn(rng, fam, draft, pressure=True)

    def _build_multi_symptom(
        self, fam: Family, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        candidates = [f for f in pool if not self.lib.related_systems(f.system, fam.system)]
        second = rng.choice(candidates)
        variant = rng.random()
        if variant < 0.45:  # both complaints have a supporting snippet; either is a valid citation
            support = [self._snip(rng.choice(fam.chain)), self._snip(rng.choice(second.chain))]
            near: list[Snippet] = []
        elif variant < 0.85:  # only one complaint is supported; the other gets a near-miss
            supported, missed = (fam, second) if rng.random() < 0.5 else (second, fam)
            support = [self._snip(rng.choice(supported.chain))]
            near = [self._snip(rng.choice(missed.near_miss))]
        else:  # neither complaint is supported
            support = []
            near = [self._snip(rng.choice(fam.near_miss)), self._snip(rng.choice(second.near_miss))]
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
                rng.choice(self.lib.phrases("symptom_connector")),
                complaint=_lower_first(complaints[1]),
                Complaint=complaints[1],
            )
        )
        if rng.random() < 0.5:
            joined += " " + rng.choice(self.lib.phrases("minor_symptom"))
        ids = self._assign_ids(rng, draft)
        tail = (
            self._pressure(rng, fam)
            if "pressure" in draft.tags
            else rng.choice(self.lib.phrases("question"))
        )
        message = _join(self._intro(rng, draft), joined, tail)
        return self._row(rng, draft, ids, [{"role": "user", "content": message}])

    def _build_multi_turn(
        self, fam: Family, rng: random.Random, pool: list[Family]
    ) -> dict[str, Any] | None:
        picked = fam.order_chain(rng.sample(fam.chain, 3))
        first, second, third = (self._snip(i) for i in picked)
        cite = rng.random() < 0.65
        near = self._sample(rng, self._snips(fam.near_miss), 1 if not cite else rng.choice((0, 1)))
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
        ids = self._assign_ids(rng, draft)
        phrases = self.lib.phrases
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

    def _abstain_draft(self, fam: Family, rng: random.Random, pool: list[Family]) -> _Draft | None:
        near = self._sample(
            rng, self._snips(fam.near_miss), rng.randint(1, min(3, len(fam.near_miss)))
        )
        close = self._sample(rng, self._snips(fam.close), rng.randint(0, 2))
        others = near + close
        if len(others) < 2 or rng.random() < 0.3:
            others += self._sample(rng, self._unrelated([fam], pool), 1)
        return self._draft(rng, [fam], [], others[:MAX_SNIPPETS], near)

    def _draft(
        self,
        rng: random.Random,
        families: Sequence[Family],
        support: list[Snippet],
        others: list[Snippet],
        near: list[Snippet],
        also_apply: Sequence[Snippet] = (),
    ) -> _Draft | None:
        """Pick a vehicle and year that every complaint's family fits, every supporting (and
        ``also_apply``) snippet applies to and no vehicle-mismatch snippet does. ``None`` if the
        combination is impossible."""
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
            and not any(s.applies.matches(vehicle, year) for s in must_not)
        ]
        if not options:
            return None
        vehicle, year = rng.choice(options)
        return _Draft(
            list(support), [s for s in snippets if s not in support], list(near), vehicle, year
        )

    def _single_turn(
        self, rng: random.Random, fam: Family, draft: _Draft, pressure: bool
    ) -> dict[str, Any]:
        if pressure:
            draft.tags.add("pressure")
        ids = self._assign_ids(rng, draft)
        intro, complaint = self._intro(rng, draft), self._complaint(rng, fam)
        body = [intro, complaint] if rng.random() < 0.7 else [complaint, intro]
        if not pressure:
            message = _join(*body, rng.choice(self.lib.phrases("question")))
        elif rng.random() < 0.25:
            message = _join(
                rng.choice(self.lib.phrases("pressure_lead")), *body, self._pressure(rng, fam)
            )
        else:
            message = _join(*body, self._pressure(rng, fam))
        return self._row(rng, draft, ids, [{"role": "user", "content": message}])

    def _row(
        self, rng: random.Random, draft: _Draft, ids: dict[str, str], messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        order = draft.supporting + draft.others
        rng.shuffle(order)
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

    def _assign_ids(self, rng: random.Random, draft: _Draft) -> dict[str, str]:
        ids: dict[str, str] = {}
        for snippet in draft.supporting + draft.others:
            while True:
                candidate = rng.choice(ID_SCHEMES[snippet.kind])(rng)
                if candidate not in ids.values():
                    ids[snippet.id] = candidate
                    break
        return ids

    def _intro(self, rng: random.Random, draft: _Draft) -> str:
        miles = rng.randint(25, 190)
        return _fill(
            rng.choice(self.lib.phrases("vehicle_intro")),
            year=str(draft.year),
            body=draft.vehicle.body,
            engine=rng.choice(draft.vehicle.names),
            miles=f"{miles}k" if rng.random() < 0.5 else f"{miles},{rng.randint(0, 999):03d}",
        )

    def _complaint(self, rng: random.Random, fam: Family) -> str:
        return _fill(
            rng.choice(fam.complaints),
            cyl=str(rng.randint(1, 4)),
            side=rng.choice(("left", "right")),
        )

    def _pressure(self, rng: random.Random, fam: Family) -> str:
        return _fill(rng.choice(self.lib.phrases("pressure")), part=rng.choice(fam.suspects))

    def _snip(self, snippet_id: str) -> Snippet:
        return self.lib.snippets[snippet_id]

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
