"""Hand-built scenarios, candidates and fake backends shared by the distillation tests.

Nothing here touches the network. The scenarios use training families, synthetic snippet texts
that appear nowhere in the library, and final user messages written to stay clear of every
committed eval set, so a dataset built from them passes the R23 leakage check.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from groundcite.judge import CHECK_LINE_RE

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SCENARIOS = REPO_ROOT / "data" / "scenarios"
EVAL_SETS = (
    SCENARIOS / "ablation.jsonl",
    SCENARIOS / "eval_own.jsonl",
    SCENARIOS / "heldout.jsonl",
)
FEW_SHOT = REPO_ROOT / "data" / "few_shot" / "examples.jsonl"
RESERVED_FAMILIES = REPO_ROOT / "data" / "reserved_families.txt"

TRAIN_FAMILIES = (
    "lean_idle",
    "hard_start_hot",
    "misfire_load",
    "overheat_idle",
    "coolant_loss",
    "battery_drain",
    "dim_lights",
    "brake_pulsation",
    "front_clunk",
    "ac_warm",
    "harsh_shift",
)

# Distinct final user messages, far from each other and from the committed eval sets.
FINAL_MESSAGES = (
    "My hatchback bucks gently whenever cruise control holds sixty on flat interstate pavement.",
    "A faint chirp comes from under the dash only after the blower runs on high for an hour.",
    "The temperature needle creeps toward the top while parked in a drive-through queue.",
    "Every morning the starter spins for ages before the engine finally catches and stumbles.",
    "Headlamps flicker in rhythm with the turn signal when the stereo bass is turned up.",
    "Steering wheel shimmies on downhill braking after the mountain pass near our cabin.",
    "Cabin vents blow lukewarm air on humid afternoons though the compressor clutch clicks.",
    "Shifts from second into third feel like a shove after the car sits in the sun.",
    "The battery is flat after a long weekend at the airport parking garage every time.",
    "Coolant reservoir drops an inch weekly with no puddle under the driveway at all.",
    "A knock from the front right corner shows up crossing railroad tracks at low speed.",
    "Idle surges up and down by a few hundred rpm with the headlights and defroster on.",
    "It hesitates and pings climbing the steep ramp at the grocery store with a full trunk.",
    "Hot restarts after fueling take three tries, yet cold starts at dawn are instant.",
    "Dash lights dim noticeably whenever the rear window defogger is switched on at night.",
    "Brake pedal pulses only during gentle stops from highway speed, not hard stops.",
)

CATEGORIES = ("clean", "distractor", "no_snippet", "pressure", "multi_symptom", "multi_turn")


def scenario_row(
    index: int,
    *,
    category: str = "clean",
    mode: str = "cite",
    family: str | None = None,
    final: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """A training scenario line with three snippets in three ID schemes.

    In cite mode the first snippet (``TSB-…``) is gold; the second is a near-miss.
    """
    ids = [f"TSB-{4100 + index:04d}", f"SM {index % 9 + 1}.{index % 7 + 1}.4", f"B-21-{index:03d}"]
    texts = [
        f"Synthetic test reference {index}a: back-probe the signal circuit and record the reading.",
        f"Synthetic test reference {index}b: applies to a different engine than the stated one.",
        f"Synthetic test reference {index}c: an unrelated body-electrical procedure for wipers.",
    ]
    snippets = [
        {"id": sid, "kind": "tsb" if n != 1 else "manual", "text": text}
        for n, (sid, text) in enumerate(zip(ids, texts, strict=True))
    ]
    supporting = [ids[0]] if mode == "cite" else []
    return {
        "id": f"tr-t{index:03d}",
        "snippets": snippets,
        "messages": [
            {"role": "user", "content": final or FINAL_MESSAGES[index % len(FINAL_MESSAGES)]}
        ],
        "gold": {"mode": mode, "supporting_ids": supporting, "near_miss_ids": [ids[1]]},
        "category": category,
        "tags": [],
        "family": family or TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)],
        "split": "train",
        "seed": seed if seed is not None else 2027500000 + index,
    }


def cite_reply(snippet_id: str, variant: int = 0) -> str:
    """An on-spec cite reply: one step, one bracketed citation, no repair wording."""
    steps = (
        "With the engine idling, inspect the hose at the intake elbow for cracks or a vacuum hiss",
        "Key on, engine off, measure the voltage at the sensor connector against the reference",
        "Back-probe the sensor signal wire and compare the reading with the specified range",
    )
    return f"{steps[variant % len(steps)]} [{snippet_id}]."


ABSTAIN_REPLY = (
    "None of the references you provided cover this complaint, so I can't point to a documented "
    "step yet. Which trouble code is stored?"
)


def candidate_row(
    scenario: Mapping[str, Any],
    index: int,
    reply: str,
    *,
    teacher: str = "mock:teacher_probe",
    error: str | None = None,
) -> dict[str, Any]:
    """A candidates.jsonl line in the format scripts/distill.py writes."""
    from groundcite.distill import TEACHER_PROMPT_VERSION, teacher_prompt_sha256

    return {
        "scenario_id": scenario["id"],
        "candidate_index": index,
        "teacher": teacher,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "teacher_prompt_sha256": teacher_prompt_sha256(),
        "reply": reply,
        "error": error,
        "scenario": dict(scenario),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_script(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None and path.is_file(), f"{path} is missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(name: str, argv: Sequence[str]) -> int:
    """Run ``scripts/<name>.py``'s ``main(argv)`` in-process (so sockets stay blocked)."""
    module = load_script(name)
    try:
        return int(module.main(list(argv)))
    except SystemExit as exc:
        return int(exc.code or 0)


def id_scheme(snippet_id: str) -> str:
    """The shape of an ID with every digit run collapsed, e.g. 'SM 6.2.1' -> 'SM #.#.#'."""
    return re.sub(r"[0-9]+", "#", snippet_id)


# ---------------------------------------------------------------------------------------------
# Fake backends (registered under backends.MOCKS by the tests)
# ---------------------------------------------------------------------------------------------


class RecordingTeacher:
    """Answers on-spec from the hint: cites the first gold ID it finds, else abstains.

    It records every (system, messages) pair it receives.
    """

    name = "mock:teacher_probe"
    calls: list[tuple[str | None, list[dict[str, str]]]] = []
    gold_ids: dict[str, str] = {}

    def generate(
        self, messages: list[dict[str, str]], system: str | None, params: Mapping[str, Any]
    ) -> str:
        type(self).calls.append((system, [dict(m) for m in messages]))
        text = "\n".join([system or ""] + [m["content"] for m in messages])
        for gold_id in type(self).gold_ids.values():
            if f"[{gold_id}]" in text:
                return cite_reply(gold_id, len(type(self).calls))
        return ABSTAIN_REPLY


class ScriptedJudge:
    """A judge whose verdict is chosen by a phrase found in the reply under review.

    ``verdicts`` maps a reply phrase to (spec_adherence, robustness); any HARD check marked FAIL
    in the judge input scores (0, 1), and everything else scores (1, 5).

    It also answers the filter-only hint-leak screen, recognized by the ``"meta_leak"`` output key
    in its system prompt (the eval rubric never names that key): a request whose input contains
    a phrase from ``leaks`` is flagged. ``calls`` counts rubric requests only; ``leak_calls``
    counts screen requests.
    """

    name = "mock:judge_probe"
    verdicts: dict[str, tuple[int, int]] = {}
    calls: int = 0
    leaks: tuple[str, ...] = ()
    leak_calls: int = 0

    def generate(
        self, messages: list[dict[str, str]], system: str | None, params: Mapping[str, Any]
    ) -> str:
        text = "\n".join(m["content"] for m in messages)
        if system is not None and '"meta_leak"' in system:
            type(self).leak_calls += 1
            leak = any(phrase in text for phrase in type(self).leaks)
            reason = "Scripted: refers to privileged inputs." if leak else "Scripted: no leak."
            return json.dumps({"meta_leak": leak, "reason": reason})
        type(self).calls += 1
        spec, robustness = 1, 5
        if any(status == "FAIL" for _, cls, status in CHECK_LINE_RE.findall(text) if cls == "HARD"):
            spec, robustness = 0, 1
        for phrase, verdict in type(self).verdicts.items():
            if phrase in text:
                spec, robustness = verdict
        return json.dumps(
            {
                "spec_adherence": spec,
                "robustness": robustness,
                "failed_checks": [],
                "reasoning": f"Scripted verdict ({spec}, {robustness}).",
            }
        )
