#!/usr/bin/env python3
"""Show the base and the tuned model's replies to one complaint, side by side, with checks (R16).

    uv run scripts/demo.py --complaint TEXT [--snippets FILE | --snippet "ID: text" ...]
                           [--model SPEC] [--base SPEC] [--json]
    uv run scripts/demo.py --scenario EVAL_SET --id SCENARIO_ID [--model SPEC] [--base SPEC]

Both models get the same zero-shot prompt and every deterministic check runs on both replies.
A complaint with no snippets is scored as an abstain case: no snippet can apply, so the on-spec
reply cites nothing and asks for one detail. Model specs use eval.py's syntax (a bare
``org/repo`` means ``hf:``); ``mock:compliant`` and ``mock:naive`` run key-free with no model.
Exit 0 after printing (failed checks are output, not errors), 2 on bad usage or input.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from groundcite.backends import Backend, get_backend
from groundcite.checks import CheckResult, run_checks
from groundcite.config import decoding_params
from groundcite.evalset import EvalSetError, Scenario, _RowError, load_eval_set, parse_row
from groundcite.keys import MissingKeyError
from groundcite.prompts import render_messages
from groundcite.spec import is_valid_id

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "train" / "config.toml"
# The published adapter at the commit train/demo_inference.ipynb pins; keep the two in step.
ADAPTER_REPO = "joe-alfaro/groundcite-qwen3-1.7b"
ADAPTER_REVISION = "b16b566d0d36d06744c90342fe836a4e6decc4e8"
DEFAULT_TUNED = f"hf:{ADAPTER_REPO}@{ADAPTER_REVISION}"
# With no snippets nothing can be cited, so the on-spec reply abstains.
ABSTAIN_GOLD = {"mode": "abstain", "supporting_ids": []}
STRATEGY = "zero_shot"
VERDICTS = {True: "PASS", False: "FAIL", None: "n/a"}


class DemoError(ValueError):
    """Bad usage or input; the message says what is wrong."""


def default_base() -> str:
    """The base model at train/config.toml's pin, the weights the adapter was trained on."""
    with TRAIN_CONFIG.open("rb") as fh:
        model = tomllib.load(fh)["model"]
    return f"hf:{model['base']}@{model['revision']}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--complaint", help="the customer's complaint, one user turn")
    snippets = parser.add_mutually_exclusive_group()
    snippets.add_argument(
        "--snippets", type=Path, metavar="FILE", help='JSON array of {"id", "text", "title"?}'
    )
    snippets.add_argument(
        "--snippet",
        action="append",
        metavar='"ID: text"',
        help="one reference snippet (repeatable); the ID is the text before the first ': '",
    )
    parser.add_argument("--scenario", type=Path, metavar="EVAL_SET", help="take a row from a set")
    parser.add_argument("--id", dest="scenario_id", metavar="ID", help="--scenario: the row's id")
    parser.add_argument(
        "--model", default=DEFAULT_TUNED, help=f"tuned model spec (default {DEFAULT_TUNED})"
    )
    parser.add_argument("--base", help="base model spec (default: train/config.toml's pinned base)")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead")
    return parser


def _flag_snippets(values: list[str]) -> list[dict[str, Any]]:
    snippets = []
    for value in values:
        snippet_id, sep, text = value.partition(": ")
        if not sep:
            raise DemoError(f'--snippet {value!r}: use "ID: text"')
        if not is_valid_id(snippet_id):
            raise DemoError(
                f"--snippet {value!r}: {snippet_id!r} is outside the ID grammar "
                "(2-24 chars of A-Z a-z 0-9 . _ space -)"
            )
        snippets.append({"id": snippet_id, "text": text})
    return snippets


def _file_snippets(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DemoError(f"{path}: {exc.strerror or exc}") from None
    except json.JSONDecodeError as exc:
        raise DemoError(f"{path}: invalid JSON at line {exc.lineno}: {exc.msg}") from None
    if not isinstance(data, list):
        raise DemoError(f"{path}: must be a JSON array of snippets")
    return data


def _free_text_scenario(complaint: str, snippets: list[dict[str, Any]], where: str) -> Scenario:
    row: dict[str, Any] = {"id": "demo", "messages": [{"role": "user", "content": complaint}]}
    if snippets:
        row["snippets"] = snippets
    else:
        row["gold"] = ABSTAIN_GOLD
    try:
        return parse_row(row)
    except _RowError as exc:
        raise DemoError(f"{where}: {exc.field}: {exc.message}") from None


def _scenario_row(path: Path, scenario_id: str) -> Scenario:
    try:
        scenarios = load_eval_set(path)
    except OSError as exc:
        raise DemoError(f"{path}: {exc.strerror or exc}") from None
    except EvalSetError as exc:
        raise DemoError(f"{path}: {exc}") from None
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario
    raise DemoError(f"{path}: no scenario with id {scenario_id!r}")


def build_scenario(args: argparse.Namespace) -> Scenario:
    has_snippets = args.snippets is not None or args.snippet
    if args.scenario is not None:
        if args.complaint is not None or has_snippets:
            raise DemoError("--scenario excludes --complaint, --snippets and --snippet")
        if not args.scenario_id:
            raise DemoError("--scenario needs --id")
        return _scenario_row(args.scenario, args.scenario_id)
    if args.scenario_id is not None:
        raise DemoError("--id needs --scenario")
    if not args.complaint or not args.complaint.strip():
        raise DemoError("give --complaint (with optional snippets) or --scenario with --id")
    if args.snippets is not None:
        return _free_text_scenario(
            args.complaint, _file_snippets(args.snippets), str(args.snippets)
        )
    return _free_text_scenario(args.complaint, _flag_snippets(args.snippet or []), "--snippet")


def _backend(spec: str) -> Backend:
    try:
        return get_backend(spec)
    except (ValueError, MissingKeyError) as exc:
        raise DemoError(str(exc)) from None


def _result(backend: Backend, reply: str, scenario: Scenario) -> dict[str, Any]:
    checks: list[CheckResult] = run_checks(reply, scenario)
    return {
        "model": backend.name,
        "reply": reply,
        "checks": [
            {"name": r.name, "severity": r.severity, "passed": r.passed, "detail": r.detail}
            for r in checks
        ],
    }


def run_demo(scenario: Scenario, tuned_spec: str, base_spec: str) -> dict[str, Any]:
    """Generate both replies to the same zero-shot prompt and check each."""
    # Both backends are built before either generates, so a bad spec fails before a model loads.
    models = {"base": _backend(base_spec), "tuned": _backend(tuned_spec)}
    system, messages = render_messages(scenario, STRATEGY)
    replies = {}
    for role, backend in models.items():
        reply = backend.generate(messages, system, decoding_params(backend.name))
        replies[role] = _result(backend, reply, scenario)
    return {
        "scenario_id": scenario.id,
        "snippet_ids": list(scenario.provided_ids),
        "gold": dict(scenario.gold) if scenario.gold is not None else None,
        "replies": replies,
    }


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines() or [""])


def format_text(scenario: Scenario, result: dict[str, Any]) -> str:
    replies = result["replies"]
    gold = result["gold"]
    lines = [
        f"scenario: {result['scenario_id']}",
        f"snippets: {', '.join(result['snippet_ids']) or '(none)'}",
        f"gold:     {json.dumps(gold) if gold is not None else '(none)'}",
        "",
        "complaint:",
        _indent(scenario.messages[-1]["content"]),
        "",
    ]
    for role in ("base", "tuned"):
        lines += [
            f"{role.upper():<6} {replies[role]['model']}",
            _indent(replies[role]["reply"]),
            "",
        ]

    base_checks, tuned_checks = replies["base"]["checks"], replies["tuned"]["checks"]
    width = max(len(c["name"]) for c in base_checks)
    lines.append(f"{'check':<{width}}  severity   base  tuned")
    for b, t in zip(base_checks, tuned_checks, strict=True):
        # citation_format is downgraded to SOFT per reply, so the two can differ.
        severity = (
            b["severity"] if b["severity"] == t["severity"] else f"{b['severity']}/{t['severity']}"
        )
        lines.append(
            f"{b['name']:<{width}}  {severity:<9}  {VERDICTS[b['passed']]:<4}  "
            f"{VERDICTS[t['passed']]}"
        )

    failures = [
        f"  {role}: {c['name']}: {c['detail']}"
        for role in ("base", "tuned")
        for c in replies[role]["checks"]
        if c["passed"] is False and c["detail"]
    ]
    if failures:
        lines += ["", "failed checks:", *failures]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scenario = build_scenario(args)
        result = run_demo(scenario, args.model, args.base or default_base())
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_text(scenario, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
