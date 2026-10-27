"""R12: eval-set loader for JSONL, JSON array and CSV, including prompt-only rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(path: Path) -> list[Any]:
    from groundcite.evalset import load_eval_set

    return load_eval_set(path)


def _by_id(path: Path) -> dict[str, Any]:
    return {s.id: s for s in _load(path)}


def _write_jsonl(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


GOOD_ROW = json.dumps({"id": "ok-1", "messages": [{"role": "user", "content": "Rough idle."}]})


# --- (a) our format -------------------------------------------------------------------------


def test_our_format_loads_every_row_in_order() -> None:
    scenarios = _load(FIXTURES / "our_format.jsonl")
    assert [s.id for s in scenarios] == ["fx-a-001", "fx-a-002", "fx-a-003"]


def test_our_format_provided_ids_come_from_snippets_in_order() -> None:
    s = _by_id(FIXTURES / "our_format.jsonl")
    assert s["fx-a-001"].provided_ids == ["SM-3105", "TSB-0417", "SM-2210"]
    assert s["fx-a-002"].provided_ids == ["B-19-045", "SM 6.2.1"]
    assert len(s["fx-a-001"].snippets) == 3


def test_our_format_keeps_multi_turn_messages_and_gold() -> None:
    s = _by_id(FIXTURES / "our_format.jsonl")["fx-a-002"]
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user", "assistant", "user"]
    assert s.messages[-1]["content"].startswith("The pedal pulses too.")
    assert s.gold["mode"] == "cite"
    assert list(s.gold["supporting_ids"]) == ["B-19-045"]
    assert s.category == "multi_turn"


def test_our_format_abstain_gold() -> None:
    s = _by_id(FIXTURES / "our_format.jsonl")["fx-a-003"]
    assert s.gold["mode"] == "abstain"
    assert list(s.gold["supporting_ids"]) == []
    assert s.category == "no_snippet"


# --- (b) prompt-only rows -------------------------------------------------------------------


def test_prompt_only_csv_becomes_single_user_message() -> None:
    s = _by_id(FIXTURES / "prompt_only.csv")
    assert list(s) == ["po-csv-001", "po-csv-002"]
    first = s["po-csv-001"]
    assert len(first.messages) == 1
    assert first.messages[0]["role"] == "user"
    # The quoted multi-line CSV field survives intact.
    assert first.messages[0]["content"].startswith("References:\n[TSB-0417]")
    assert first.messages[0]["content"].endswith("rough idle when cold.")


def test_prompt_only_csv_derives_provided_ids_from_prompt_text() -> None:
    s = _by_id(FIXTURES / "prompt_only.csv")
    assert s["po-csv-001"].provided_ids == ["TSB-0417", "SM-2210"]
    assert s["po-csv-002"].provided_ids == ["B-19-045", "SM 6.2.1"]


def test_prompt_only_rows_have_no_gold_or_snippets_by_default() -> None:
    s = _by_id(FIXTURES / "prompt_only.csv")["po-csv-001"]
    assert s.gold is None
    assert not s.snippets


def test_prompt_only_json_array_loads_with_optional_gold() -> None:
    s = _by_id(FIXTURES / "prompt_only.json")
    assert list(s) == ["po-json-001", "po-json-002"]
    assert s["po-json-001"].provided_ids == ["SM-3105", "SM-2210"]
    assert s["po-json-001"].gold["mode"] == "cite"
    assert list(s["po-json-001"].gold["supporting_ids"]) == ["SM-3105"]
    assert s["po-json-002"].provided_ids == ["TSB-0417"]
    assert s["po-json-002"].gold is None


# --- (c) third-party set with an unfamiliar ID scheme ---------------------------------------


def test_third_party_structured_snippets_need_only_id_and_text() -> None:
    s = _by_id(FIXTURES / "third_party.jsonl")["tp-7"]
    assert s.provided_ids == ["ref_q9.04", "ref_q7.22"]
    assert list(s.gold["supporting_ids"]) == ["ref_q9.04"]
    assert s.category is None


def test_third_party_prompt_only_ids_with_spaces() -> None:
    s = _by_id(FIXTURES / "third_party.jsonl")["tp-8"]
    assert s.provided_ids == ["ref_q3.11", "Ref Q12.5"]
    assert s.gold["mode"] == "abstain"


def test_without_snippets_provided_ids_come_from_user_and_system_text_only() -> None:
    s = _by_id(FIXTURES / "third_party.jsonl")["tp-9"]
    # [ref_q1.01] appears only in an assistant turn, so it was never provided.
    assert s.provided_ids == ["ref_q2.40"]
    assert s.gold is None
    assert s.messages[0]["role"] == "system"
    assert s.messages[-1]["role"] == "user"


# --- malformed rows fail fast with line number and field name -------------------------------


def _expect_error(path: Path) -> Any:
    from groundcite.evalset import EvalSetError

    with pytest.raises(EvalSetError) as info:
        _load(path)
    return info.value


def test_eval_set_error_is_a_value_error() -> None:
    from groundcite.evalset import EvalSetError

    assert issubclass(EvalSetError, ValueError)


def test_missing_id_reports_line_and_field(tmp_path: Path) -> None:
    bad = json.dumps({"messages": [{"role": "user", "content": "Rough idle."}]})
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [GOOD_ROW, bad]))
    assert err.line == 2
    assert err.field == "id"
    assert "line 2" in str(err)
    assert "id" in str(err)


def test_line_numbers_count_blank_lines(tmp_path: Path) -> None:
    bad = json.dumps({"id": "no-content"})
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [GOOD_ROW, "", "", bad]))
    assert err.line == 4
    assert "line 4" in str(err)


def test_row_without_messages_or_prompt_is_rejected(tmp_path: Path) -> None:
    bad = json.dumps({"id": "no-content", "gold": {"mode": "cite", "supporting_ids": ["A1"]}})
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [bad]))
    assert err.line == 1
    assert err.field == "messages"


def test_messages_must_end_with_a_user_turn(tmp_path: Path) -> None:
    bad = json.dumps(
        {
            "id": "ends-assistant",
            "messages": [
                {"role": "user", "content": "Rough idle."},
                {"role": "assistant", "content": "Check the hose [TSB-0417]."},
            ],
        }
    )
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [GOOD_ROW, GOOD_ROW, bad]))
    assert err.line == 3
    assert err.field == "messages"
    assert "line 3" in str(err)
    assert "messages" in str(err)


def test_empty_prompt_is_rejected(tmp_path: Path) -> None:
    bad = json.dumps({"id": "blank", "prompt": "   "})
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [bad]))
    assert err.field == "prompt"


def test_invalid_gold_mode_is_rejected(tmp_path: Path) -> None:
    bad = json.dumps({"id": "g", "prompt": "Rough idle [A1].", "gold": {"mode": "maybe"}})
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [GOOD_ROW, bad]))
    assert err.line == 2
    assert err.field == "gold.mode"


def test_snippet_without_id_is_rejected(tmp_path: Path) -> None:
    bad = json.dumps(
        {"id": "s", "prompt": "Rough idle.", "snippets": [{"id": "A1", "text": "x"}, {"text": "y"}]}
    )
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [bad]))
    assert err.line == 1
    assert err.field is not None and err.field.startswith("snippets")


def test_snippet_id_outside_the_id_grammar_is_rejected(tmp_path: Path) -> None:
    bad = json.dumps({"id": "s", "prompt": "Rough idle.", "snippets": [{"id": "Ref#Q7/22"}]})
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [bad]))
    assert err.field is not None and err.field.startswith("snippets")


def test_invalid_json_line_reports_line_number(tmp_path: Path) -> None:
    err = _expect_error(_write_jsonl(tmp_path / "set.jsonl", [GOOD_ROW, "{not json"]))
    assert err.line == 2
    assert "line 2" in str(err)


def test_csv_errors_use_physical_file_line(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    path.write_text('id,prompt\nok-1,"Rough idle [A1]."\n,"Missing id [A1]."\n', encoding="utf-8")
    err = _expect_error(path)
    assert err.line == 3
    assert err.field == "id"


def test_json_array_errors_use_one_based_row_index(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    path.write_text(json.dumps([{"id": "ok", "prompt": "Rough idle."}, {"id": "bad"}]))
    err = _expect_error(path)
    assert err.line == 2
    assert err.field == "messages"
