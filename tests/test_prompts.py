"""Tests for the prompt strategies and the one shared renderer (R3, R23, R27, GOTCHA G26, G27).

`groundcite.prompts` renders a scenario for one of the three ablation strategies. Eval, the
ablation, training and the demo all call the same `render_messages()`, and the HF chat template is
always applied with `enable_thinking=False`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from groundcite.checks import CHECKS
from groundcite.evalset import Scenario, load_eval_set
from groundcite.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "data" / "scenarios"
EVAL_SETS = [
    SCENARIOS / "ablation.jsonl",
    SCENARIOS / "eval_own.jsonl",
    SCENARIOS / "heldout.jsonl",
]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _ablation() -> list[Scenario]:
    return load_eval_set(SCENARIOS / "ablation.jsonl")


def _by_category(category: str) -> Scenario:
    return next(s for s in _ablation() if s.category == category)


def _prompt_text(system: str, messages: list[dict[str, str]]) -> str:
    return "\n".join([system] + [m["content"] for m in messages])


# ---------------------------------------------------------------------------------------------
# Strategy names
# ---------------------------------------------------------------------------------------------


def test_strategies_are_exactly_the_three_named_ones() -> None:
    from groundcite.prompts import STRATEGIES

    assert tuple(STRATEGIES) == ("zero_shot", "few_shot", "structured")


def test_unknown_strategy_is_rejected_by_name() -> None:
    from groundcite.prompts import render_messages

    with pytest.raises(ValueError, match="chain_of_thought"):
        render_messages(_by_category("clean"), "chain_of_thought")


# ---------------------------------------------------------------------------------------------
# render_messages(scenario, strategy) -> (system, messages)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["zero_shot", "few_shot", "structured"])
def test_render_returns_a_system_prompt_and_a_chat_ending_in_the_scenario_user_turn(
    strategy: str,
) -> None:
    from groundcite.prompts import render_messages

    scenario = _by_category("multi_turn")
    system, messages = render_messages(scenario, strategy)
    assert isinstance(system, str) and system.strip()
    assert all(m["role"] in ("user", "assistant") for m in messages), "system goes in `system`"
    assert messages[-1]["role"] == "user"
    # The scenario's own turns are all present, in order, at the end of the chat.
    tail = messages[-len(scenario.messages) :]
    assert [m["role"] for m in tail] == [m["role"] for m in scenario.messages]
    for rendered, original in zip(tail, scenario.messages, strict=True):
        assert original["content"] in rendered["content"]


@pytest.mark.parametrize("strategy", ["zero_shot", "few_shot", "structured"])
def test_every_snippet_is_rendered_with_its_bracketed_id_and_text(strategy: str) -> None:
    from groundcite.prompts import render_messages

    scenario = _by_category("distractor")
    system, messages = render_messages(scenario, strategy)
    text = _prompt_text(system, messages)
    for snippet in scenario.snippets:
        assert f"[{snippet['id']}]" in text
        assert snippet["text"] in text


@pytest.mark.parametrize("strategy", ["zero_shot", "few_shot", "structured"])
def test_every_strategy_carries_the_spec_text_loaded_by_path(strategy: str) -> None:
    from groundcite.prompts import render_messages

    system, _ = render_messages(_by_category("clean"), strategy)
    assert load_spec().strip() in system


@pytest.mark.parametrize("strategy", ["zero_shot", "few_shot", "structured"])
def test_rendering_never_leaks_the_gold_label(strategy: str) -> None:
    from groundcite.prompts import render_messages

    scenario = _by_category("clean")
    system, messages = render_messages(scenario, strategy)
    text = _prompt_text(system, messages).lower()
    assert "supporting_ids" not in text
    assert "gold" not in text


def test_strategies_render_three_different_prompts() -> None:
    from groundcite.prompts import STRATEGIES, render_messages

    scenario = _by_category("pressure")
    rendered = {s: render_messages(scenario, s) for s in STRATEGIES}
    texts = {s: _prompt_text(*r) for s, r in rendered.items()}
    assert len(set(texts.values())) == 3


def test_rendering_is_deterministic() -> None:
    from groundcite.prompts import STRATEGIES, render_messages

    scenario = _by_category("multi_symptom")
    for strategy in STRATEGIES:
        assert render_messages(scenario, strategy) == render_messages(scenario, strategy)


def test_structured_has_explicit_numbered_rules_and_a_private_checklist() -> None:
    from groundcite.prompts import render_messages

    system, _ = render_messages(_by_category("clean"), "structured")
    numbered = re.findall(r"(?m)^\s*\d+[.)]\s+\S", system)
    assert len(numbered) >= 3, "structured needs explicit numbered rules"
    assert "checklist" in system.lower()
    # The checklist is private: the model is told to output only the final reply.
    assert re.search(r"only the final reply", system, re.IGNORECASE)


def test_zero_shot_has_no_examples_and_no_checklist() -> None:
    from groundcite.prompts import load_few_shot, render_messages

    scenario = _by_category("clean")
    system, messages = render_messages(scenario, "zero_shot")
    text = _prompt_text(system, messages)
    assert len(messages) == len(scenario.messages)
    assert "checklist" not in system.lower()
    for example in load_few_shot():
        assert example.reply not in text


def test_prompt_only_rows_render_without_snippets() -> None:
    from groundcite.prompts import STRATEGIES, render_messages

    scenario = load_eval_set(FIXTURES / "prompt_only.json")[0]
    assert not scenario.snippets
    for strategy in STRATEGIES:
        system, messages = render_messages(scenario, strategy)
        assert messages[-1]["role"] == "user"
        assert scenario.messages[-1]["content"] in messages[-1]["content"]


# ---------------------------------------------------------------------------------------------
# Few-shot examples: a committed file, disjoint from every eval set
# ---------------------------------------------------------------------------------------------


def test_few_shot_file_is_committed_under_data() -> None:
    from groundcite.prompts import FEW_SHOT_PATH

    assert Path(FEW_SHOT_PATH) == REPO_ROOT / "data" / "few_shot" / "examples.jsonl"
    assert Path(FEW_SHOT_PATH).is_file()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(FEW_SHOT_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, "the few-shot file must be tracked by git"


def test_few_shot_examples_cover_cite_abstain_pressure_and_distractor() -> None:
    from groundcite.prompts import load_few_shot

    examples = load_few_shot()
    assert len(examples) >= 4
    categories = {e.scenario.category for e in examples}
    assert {"clean", "no_snippet", "pressure", "distractor"} <= categories
    modes = {(e.scenario.gold or {}).get("mode") for e in examples}
    assert modes == {"cite", "abstain"}


def test_few_shot_replies_pass_every_deterministic_check() -> None:
    from groundcite.prompts import load_few_shot

    for example in load_few_shot():
        s = example.scenario
        assert s.gold is not None, f"{s.id}: few-shot examples need gold"
        for name, check in CHECKS.items():
            result = check(example.reply, s.provided_ids, s.gold)
            assert result.passed is not False, f"{s.id}: {name} fails: {result.detail}"


def test_few_shot_ids_are_disjoint_from_every_eval_set() -> None:
    from groundcite.prompts import load_few_shot

    few_ids = {e.scenario.id for e in load_few_shot()}
    for path in EVAL_SETS:
        eval_ids = {s.id for s in load_eval_set(path)}
        assert not few_ids & eval_ids, f"few-shot ids overlap {path.name}"


def test_few_shot_examples_appear_in_the_few_shot_prompt_only() -> None:
    from groundcite.prompts import load_few_shot, render_messages

    scenario = _by_category("clean")
    few = _prompt_text(*render_messages(scenario, "few_shot"))
    structured = _prompt_text(*render_messages(scenario, "structured"))
    for example in load_few_shot():
        assert example.reply in few
        assert example.reply not in structured


def test_load_few_shot_accepts_an_explicit_path(tmp_path: Path) -> None:
    from groundcite.prompts import FEW_SHOT_PATH, load_few_shot

    copy = tmp_path / "fs.jsonl"
    copy.write_text(Path(FEW_SHOT_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    assert [e.scenario.id for e in load_few_shot(copy)] == [e.scenario.id for e in load_few_shot()]


def test_check_leakage_passes_with_the_committed_few_shot_file(tmp_path: Path) -> None:
    from groundcite.prompts import FEW_SHOT_PATH

    empty_train = tmp_path / "train.jsonl"
    empty_train.write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_leakage.py"),
            "--train",
            str(empty_train),
            "--eval",
            *[str(p) for p in EVAL_SETS],
            "--few-shot",
            str(FEW_SHOT_PATH),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"(\d+) few-shot", result.stdout)
    assert match and int(match.group(1)) >= 4


def test_quality_gate_checks_the_few_shot_file() -> None:
    check_sh = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    assert "--few-shot" in check_sh
    assert "data/few_shot/examples.jsonl" in check_sh


# ---------------------------------------------------------------------------------------------
# One chat-template call with thinking disabled (R27)
# ---------------------------------------------------------------------------------------------


class FakeTokenizer:
    """A stand-in chat template: `<|role|>content<|end|>`, generation prompt `<|assistant|>`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, chat: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"chat": [dict(m) for m in chat], **kwargs})
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in chat)
        if kwargs.get("add_generation_prompt"):
            text += "<|assistant|>"
        return text


def test_chat_template_kwargs_disable_thinking() -> None:
    from groundcite.prompts import CHAT_TEMPLATE_KWARGS

    assert CHAT_TEMPLATE_KWARGS["enable_thinking"] is False


def test_render_chat_text_puts_the_system_first_and_disables_thinking() -> None:
    from groundcite.prompts import render_chat_text, render_messages

    tok = FakeTokenizer()
    system, messages = render_messages(_by_category("clean"), "zero_shot")
    text = render_chat_text(tok, system, messages)
    call = tok.calls[-1]
    assert call["enable_thinking"] is False
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["chat"][0] == {"role": "system", "content": system}
    assert call["chat"][1:] == messages
    assert text.endswith("<|assistant|>")


def test_training_string_extends_the_eval_string() -> None:
    """R27: the rendered training example starts with exactly the eval prompt."""
    from groundcite.prompts import render_chat_text, render_messages

    tok = FakeTokenizer()
    system, messages = render_messages(_by_category("pressure"), "zero_shot")
    eval_text = render_chat_text(tok, system, messages)
    train_text = render_chat_text(tok, system, messages, reply="Check the hose [X-1].")
    train_call = tok.calls[-1]
    assert train_call["enable_thinking"] is False
    assert train_call["add_generation_prompt"] is False
    assert train_call["chat"][-1] == {"role": "assistant", "content": "Check the hose [X-1]."}
    assert train_text.startswith(eval_text.removesuffix("<|assistant|>"))
    assert "Check the hose [X-1]." in train_text
