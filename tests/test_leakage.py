"""Tests for scripts/check_leakage.py (R23), driven through the CLI on small JSONL fixtures.

CLI: check_leakage.py --train PATH --eval PATH [PATH ...] [--few-shot PATH] [--reserved PATH]
Exit 0 when clean, 1 when any rule fires. Each fired rule is named in the output by its code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_leakage.py"
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"

RULE_CODES = (
    "shared_id",
    "same_final_message",
    "ngram_jaccard",
    "heldout_snippet_hash",
    "reserved_family",
    "fewshot_in_eval",
    "missing_family",
)

RESERVED = "fam_reserved"

TRAIN_MSG = "front brakes squeal loudly at low speed when stopping after the car sits in rain"
EVAL_MSG = "rough idle when cold with a lean code on bank one and a hesitation on takeoff"
HELDOUT_MSG = "battery light flickers at night with the headlights on and the radio dims too"
FEWSHOT_MSG = "steering wheel shakes above sixty miles per hour on the highway but not below"
# Differs from EVAL_MSG only in the last word: word-5-gram Jaccard is 11/13, about 0.85.
EVAL_MSG_NEAR = "rough idle when cold with a lean code on bank one and a hesitation on launch"


def _snippet(snippet_id: str, text: str) -> dict[str, Any]:
    return {"id": snippet_id, "kind": "tsb", "system": "engine", "text": text}


def _scenario(
    scenario_id: str,
    message: str,
    family: str,
    split: str,
    texts: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "snippets": [_snippet(f"TSB-{i:04d}", text) for i, text in enumerate(texts)],
        "messages": [{"role": "user", "content": message}],
        "gold": {"mode": "abstain", "supporting_ids": []},
        "category": "no_snippet",
        "tags": [],
        "family": family,
        "split": split,
    }


def _train(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "scenario_id": "tr-0001",
        "message": TRAIN_MSG,
        "family": "fam_brakes",
        "split": "train",
        "texts": ("Brake pad wear indicator check text.", "Rotor runout measurement text."),
    }
    fields.update(overrides)
    return _scenario(**fields)


def _eval(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "scenario_id": "eo-0001",
        "message": EVAL_MSG,
        "family": "fam_lean",
        "split": "eval_own",
        "texts": ("PCV hose crack inspection text.", "Fuel rail pressure test text."),
    }
    fields.update(overrides)
    return _scenario(**fields)


def _heldout(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "scenario_id": "ho-0001",
        "message": HELDOUT_MSG,
        "family": RESERVED,
        "split": "heldout",
        "texts": ("Alternator ripple measurement text.", "Ground strap voltage drop text."),
    }
    fields.update(overrides)
    return _scenario(**fields)


def _fewshot(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "scenario_id": "fs-0001",
        "message": FEWSHOT_MSG,
        "family": "fam_vibration",
        "split": "few_shot",
        "texts": ("Wheel balance check text.", "Tire radial runout text."),
    }
    fields.update(overrides)
    return _scenario(**fields)


def _chat_record(scenario_id: str, message: str, family: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Inspect the pads for wear [TSB-0000]."},
        ],
        "meta": {"scenario_id": scenario_id, "category": "clean", "family": family},
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _run(
    tmp_path: Path,
    *,
    train: list[dict[str, Any]] | None = None,
    eval_own: list[dict[str, Any]] | None = None,
    heldout: list[dict[str, Any]] | None = None,
    few_shot: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    assert SCRIPT.is_file(), f"{SCRIPT.relative_to(REPO_ROOT)} is missing"
    reserved = tmp_path / "reserved_families.txt"
    reserved.write_text(f"# reserved for held-out\n{RESERVED}\n")
    args = [
        sys.executable,
        str(SCRIPT),
        "--train",
        str(_write(tmp_path / "train.jsonl", train if train is not None else [_train()])),
        "--eval",
        str(_write(tmp_path / "eval_own.jsonl", eval_own if eval_own is not None else [_eval()])),
        str(_write(tmp_path / "heldout.jsonl", heldout if heldout is not None else [_heldout()])),
        "--few-shot",
        str(
            _write(tmp_path / "few_shot.jsonl", few_shot if few_shot is not None else [_fewshot()])
        ),
        "--reserved",
        str(reserved),
    ]
    result = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60)
    return result.returncode, result.stdout + result.stderr


def _assert_flags(code: int, output: str, expected: str, allowed: tuple[str, ...] = ()) -> None:
    assert code == 1, f"expected exit 1 for {expected}, got {code}:\n{output}"
    assert expected in output, f"rule {expected!r} not reported:\n{output}"
    unexpected = [c for c in RULE_CODES if c in output and c != expected and c not in allowed]
    assert not unexpected, f"fixture for {expected!r} also fired {unexpected}:\n{output}"


def test_clean_pair_exits_zero(tmp_path: Path) -> None:
    code, output = _run(tmp_path)
    assert code == 0, output
    for rule in RULE_CODES:
        assert rule not in output


def test_shared_scenario_id(tmp_path: Path) -> None:
    code, output = _run(tmp_path, eval_own=[_eval(scenario_id="tr-0001")])
    _assert_flags(code, output, "shared_id")


def test_identical_normalized_final_user_message(tmp_path: Path) -> None:
    shouty = "  " + TRAIN_MSG.upper().replace(" ", "   ") + "\n"
    code, output = _run(tmp_path, eval_own=[_eval(message=shouty)])
    # An identical message also has Jaccard 1.0, so the n-gram rule may fire too.
    _assert_flags(code, output, "same_final_message", allowed=("ngram_jaccard",))


def test_only_the_final_user_message_is_compared(tmp_path: Path) -> None:
    multi = _eval()
    multi["messages"] = [
        {"role": "user", "content": TRAIN_MSG},
        {"role": "assistant", "content": "Measure rotor runout with a dial indicator [TSB-0001]."},
        {"role": "user", "content": EVAL_MSG},
    ]
    code, output = _run(tmp_path, eval_own=[multi])
    assert code == 0, output


def test_word_five_gram_jaccard_at_threshold(tmp_path: Path) -> None:
    code, output = _run(
        tmp_path,
        train=[_train(message=EVAL_MSG_NEAR)],
        eval_own=[_eval()],
    )
    _assert_flags(code, output, "ngram_jaccard")


def test_heldout_snippet_text_shared_with_training(tmp_path: Path) -> None:
    shared = "Brake pad wear indicator check text."
    code, output = _run(
        tmp_path,
        heldout=[_heldout(texts=(shared, "Ground strap voltage drop text."))],
    )
    _assert_flags(code, output, "heldout_snippet_hash")


def test_eval_own_may_share_snippet_text_with_training(tmp_path: Path) -> None:
    # IDs are re-assigned per scenario and eval_own shares the training families and library,
    # so shared snippet text only counts as leakage for held-out scenarios.
    shared = "Brake pad wear indicator check text."
    code, output = _run(tmp_path, eval_own=[_eval(texts=(shared, "Fuel rail pressure test text."))])
    assert code == 0, output


def test_training_example_from_reserved_family(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_train(family=RESERVED)])
    _assert_flags(code, output, "reserved_family")


def test_few_shot_example_from_reserved_family(tmp_path: Path) -> None:
    code, output = _run(tmp_path, few_shot=[_fewshot(family=RESERVED)])
    _assert_flags(code, output, "reserved_family")


def test_few_shot_example_with_eval_id(tmp_path: Path) -> None:
    code, output = _run(tmp_path, few_shot=[_fewshot(scenario_id="eo-0001")])
    _assert_flags(code, output, "fewshot_in_eval")


def test_few_shot_example_with_eval_message(tmp_path: Path) -> None:
    code, output = _run(tmp_path, few_shot=[_fewshot(message=HELDOUT_MSG)])
    _assert_flags(code, output, "fewshot_in_eval")


def test_train_file_may_hold_chat_records(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_chat_record("tr-0001", TRAIN_MSG, "fam_brakes")])
    assert code == 0, output


def test_chat_record_shared_scenario_id(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_chat_record("eo-0001", TRAIN_MSG, "fam_brakes")])
    _assert_flags(code, output, "shared_id")


def test_chat_record_from_reserved_family(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_chat_record("tr-0001", TRAIN_MSG, RESERVED)])
    _assert_flags(code, output, "reserved_family")


# --- missing family (validator note Q8) -------------------------------------------------------
# reserved_family can only fire on rows that name a family, so a training or few-shot row
# without one is itself a finding.


def _without_family(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.pop("family", None)
    if "meta" in row:
        row["meta"] = {k: v for k, v in row["meta"].items() if k != "family"}
    return row


def test_training_row_without_family(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_without_family(_train())])
    _assert_flags(code, output, "missing_family")


def test_training_row_with_empty_family(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_train(family="")])
    _assert_flags(code, output, "missing_family")


def test_training_chat_record_without_meta_family(tmp_path: Path) -> None:
    record = _without_family(_chat_record("tr-0001", TRAIN_MSG, "fam_brakes"))
    code, output = _run(tmp_path, train=[record])
    _assert_flags(code, output, "missing_family")


def test_few_shot_row_without_family(tmp_path: Path) -> None:
    code, output = _run(tmp_path, few_shot=[_without_family(_fewshot())])
    _assert_flags(code, output, "missing_family")


def test_missing_family_names_the_row(tmp_path: Path) -> None:
    code, output = _run(tmp_path, train=[_train(), _without_family(_train(scenario_id="tr-0002"))])
    assert code == 1, output
    assert "tr-0002" in output
    assert "tr-0001" not in output


def test_eval_rows_without_family_are_not_flagged(tmp_path: Path) -> None:
    # Third-party eval sets need not carry a family; only training and few-shot rows must.
    code, output = _run(tmp_path, eval_own=[_without_family(_eval())])
    assert code == 0, output
    assert "missing_family" not in output


def test_missing_family_is_a_reported_rule() -> None:
    from groundcite.leakage import RULES

    assert "missing_family" in RULES


def test_check_sh_runs_the_leakage_check() -> None:
    assert CHECK_SH.is_file(), f"{CHECK_SH} is missing"
    lines = [
        line.strip()
        for line in CHECK_SH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any("check_leakage" in line and "echo" not in line for line in lines), (
        "check.sh does not invoke scripts/check_leakage.py"
    )
