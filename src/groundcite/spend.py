"""Judge spend: dry-run estimates, recorded per-call usage, ``spend.json`` and the ``--max-spend``
cap, shared by every judging entry point (R10, R11, R20, D-095, D-123).

A run's judge is wrapped in a ``MeteredJudge``. Each call's usage, as the API reported it, goes
into the transcript row that made it and into the invocation's total, which ``record_spend`` adds
to ``RUN_DIR/spend.json``. With a limit, a call is made only if the spend already recorded plus
the worst case of every call in flight, this one included, stays within it. The worst case is an
upper bound, not an estimate: every prompt byte counted as a token (byte-level tokenizers emit at
most one token per byte) and the full output-token cap from the decoding params. So the recorded
spend can't pass the limit, under any concurrency.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO

from groundcite.checks import run_checks
from groundcite.config import cost_usd, decoding_params, price
from groundcite.evalset import Scenario
from groundcite.judge import build_judge_messages

SPEND = "spend.json"
USAGE_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens")
# Dry-run estimate, as in judge_agreement.py: about 4 characters per input token and a fixed
# output allowance per call (a ~200-token verdict plus reasoning). An assumption, not a quote;
# spend.json records what a run actually used.
CHARS_PER_TOKEN = 4
EST_OUTPUT_TOKENS_PER_CALL = 2000
# The --max-spend bound on the tokens a provider adds around each message (role markers).
MESSAGE_OVERHEAD_TOKENS = 16
# Decoding params that cap a call's output tokens, reasoning included, per provider.
OUTPUT_CAP_PARAMS = ("max_completion_tokens", "max_tokens", "max_output_tokens", "max_new_tokens")

Messages = Sequence[Mapping[str, str]]
Prompt = tuple[str | None, Messages]
Usage = dict[str, Any]


class SpendCapReached(RuntimeError):
    """The next judge call could push the recorded spend past ``--max-spend``."""


class _Judge(Protocol):
    name: str
    inner: Any

    def generate(
        self, messages: Messages, system: str | None, params: Mapping[str, Any]
    ) -> str: ...


def is_free(spec: str) -> bool:
    """Mock backends make no paid calls."""
    return spec.startswith("mock:")


def judge_prompt(scenario: Scenario, reply: str) -> Prompt:
    """The ``(system, messages)`` the judge is sent for ``reply``."""
    return build_judge_messages(scenario, reply, run_checks(reply, scenario))


def _chars(prompt: Prompt) -> int:
    system, messages = prompt
    return len(system or "") + sum(len(m["content"]) for m in messages)


def estimate_cost(spec: str, prompts: Iterable[Prompt]) -> float | None:
    """Estimated dollars for one judge call per prompt; None without a ``[prices]`` entry."""
    if is_free(spec):
        return 0.0
    usages = [
        {
            "input_tokens": math.ceil(_chars(p) / CHARS_PER_TOKEN),
            "output_tokens": EST_OUTPUT_TOKENS_PER_CALL,
            "reasoning_tokens": None,
        }
        for p in prompts
    ]
    try:
        return cost_usd(spec, usages)
    except KeyError:
        return None


def print_estimate(spec: str, cost: float | None, out: TextIO | None = None) -> None:
    """The dry run's ``estimated cost: $X.XX`` line."""
    if cost is None:
        print(f"estimated cost: unknown (no [prices] entry for {spec})", file=out or sys.stdout)
    else:
        print(f"estimated cost: ${cost:.2f}", file=out or sys.stdout)


def _output_cap(params: Mapping[str, Any]) -> int | None:
    for name in OUTPUT_CAP_PARAMS:
        if isinstance(params.get(name), int):
            return int(params[name])
    return None


def check_limit(spec: str, limit: float | None) -> None:
    """Refuse a ``--max-spend`` that can't be enforced for ``spec``, before any work."""
    if limit is None or is_free(spec):
        return
    if limit < 0:
        raise ValueError("--max-spend must be zero or more")
    try:
        price(spec)
    except KeyError:
        raise ValueError(
            f"--max-spend needs a [prices] entry for {spec} in config/models.toml"
        ) from None
    if _output_cap(decoding_params(spec)) is None:
        raise ValueError(
            f"--max-spend needs an output-token cap in the decoding params for {spec} "
            f"(one of {', '.join(OUTPUT_CAP_PARAMS)})"
        )


def worst_case_usage(prompt: Prompt, params: Mapping[str, Any]) -> Usage:
    """An upper bound on one call's usage: every prompt byte a token, and the full output cap."""
    system, messages = prompt
    size = len((system or "").encode("utf-8"))
    size += sum(len(m["content"].encode("utf-8")) for m in messages)
    return {
        "input_tokens": size + MESSAGE_OVERHEAD_TOKENS * (len(messages) + 1),
        "output_tokens": _output_cap(params) or 0,
        "reasoning_tokens": 0,
    }


class MeteredJudge:
    """The judge with its spend recorded and, given ``limit``, capped.

    ``spent`` is the cost already recorded for the run (a resumed run's ``spend.json``). ``usage``
    holds this invocation's calls. Use ``for_row()`` per transcript row to collect that row's
    calls."""

    def __init__(self, judge: _Judge, *, limit: float | None = None, spent: float = 0.0) -> None:
        check_limit(judge.name, limit)
        self.judge = judge
        self.name = judge.name
        self.limit = None if is_free(judge.name) else limit
        self.spent = spent
        self.usage: list[Usage] = []
        self._reserved = 0.0
        self._lock = threading.Lock()

    def _take_usage(self) -> Usage | None:
        take = getattr(self.judge.inner, "take_last_usage", None)
        return take() if callable(take) else None

    def _reserve(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> float:
        if self.limit is None:
            return 0.0
        worst = cost_usd(self.name, [worst_case_usage((system, messages), params)])
        with self._lock:
            if self.spent + self._reserved + worst > self.limit:
                raise SpendCapReached(
                    f"--max-spend ${self.limit:g} reached: ${self.spent:.4f} recorded, and the "
                    f"next judge call could cost up to ${worst:.4f}. Finished rows and "
                    f"{SPEND} are kept"
                )
            self._reserved += worst
        return worst

    def call(
        self, messages: Messages, system: str | None, params: Mapping[str, Any]
    ) -> tuple[str, Usage | None]:
        """One judge call: its text and recorded usage (None for a backend that reports none).
        A call that fails after the API answered still has its usage recorded."""
        reserved = self._reserve(messages, system, params)
        self._take_usage()  # drop anything an unmetered call on this thread left behind
        usage: Usage | None = None
        try:
            text = self.judge.generate(messages, system, params)
        finally:
            usage = self._take_usage()
            with self._lock:
                self._reserved -= reserved
                if usage is not None:
                    self.usage.append(usage)
                    if not is_free(self.name):
                        try:
                            self.spent += cost_usd(self.name, [usage])
                        except KeyError:
                            pass  # no price: nothing to cap (check_limit refused a limit)
        return text, usage

    def for_row(self) -> RowCalls:
        return RowCalls(self)


class RowCalls:
    """The judge as one transcript row sees it: ``usage`` lists that row's calls."""

    def __init__(self, meter: MeteredJudge) -> None:
        self.name = meter.name
        self.usage: list[Usage] = []
        self._meter = meter

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        text, usage = self._meter.call(messages, system, params)
        if usage is not None:
            self.usage.append(usage)
        return text


def read_spend(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / SPEND
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def recorded_cost(run_dir: Path) -> float:
    """The spend already recorded in ``run_dir`` (0 for none). An unknown cost is refused: a cap
    can't count it."""
    spend = read_spend(run_dir)
    if spend is None:
        return 0.0
    cost = spend.get("cost_usd")
    if not isinstance(cost, int | float):
        raise ValueError(f"{run_dir / SPEND} has no known cost_usd; --max-spend can't count it")
    return float(cost)


def record_spend(
    run_dir: Path, judge_name: str, usages: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Add this invocation's usage to ``run_dir/spend.json`` (judge_agreement.py's fields)."""
    spend: dict[str, Any] = {
        "judge_model": judge_name,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    spend.update(read_spend(run_dir) or {})
    spend["calls"] += len(usages)
    for field in USAGE_FIELDS:
        spend[field] += sum(u.get(field) or 0 for u in usages)
    if usages and not is_free(judge_name):
        try:
            added = cost_usd(judge_name, usages)
        except KeyError:
            print(
                f"warning: no [prices] entry for {judge_name}; cost not recorded", file=sys.stderr
            )
            spend["cost_usd"] = None
        else:
            if spend["cost_usd"] is not None:
                spend["cost_usd"] += added
    (run_dir / SPEND).write_text(json.dumps(spend, indent=2) + "\n", encoding="utf-8")
    return spend


def spend_note(spend: Mapping[str, Any]) -> str:
    """``N judge calls, spend $X`` for a run's closing line."""
    cost = spend.get("cost_usd")
    shown = "unknown" if cost is None else f"${cost:.4f}"
    return f"{spend.get('calls', 0)} judge calls in all, spend {shown}"
