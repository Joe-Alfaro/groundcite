"""Generation/judge backends behind one interface, key handling (R25), pinned models (D-019),
mock models (R26). Real HTTP never happens: API backends take an injected transport, and
sockets are blocked.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from groundcite.checks import CHECKS, HARD_CHECKS
from groundcite.evalset import Scenario
from groundcite.spec import parse_cited_ids

pytestmark = pytest.mark.usefixtures("no_network", "isolated_env")

REPO = Path(__file__).resolve().parents[1]

FAKE_ANTHROPIC = "FAKE-ANTHROPIC-0000-TESTVALUE"
FAKE_OPENAI = "FAKE-OPENAI-1111-TESTVALUE"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM = "You are a grounded diagnostic assistant."
MESSAGES = [
    {
        "role": "user",
        "content": (
            "References:\n"
            "[SM-2210] Fuel pressure test at the rail test port, 55-62 psi.\n"
            "[TSB-0417] Inspect the PCV hose at the intake elbow for cracks.\n"
            "[SM-3105] Alternator output test.\n\n"
            "2015 compact sedan, 2.0L. P0171, rough idle when cold."
        ),
    }
]
PROVIDED = ["SM-2210", "TSB-0417", "SM-3105"]
NO_REF_MESSAGES = [
    {"role": "user", "content": "Clunk from the front left over bumps. What should I check?"}
]


class FakeTransport:
    """Stands in for HTTP: records (url, headers, body) and returns a canned JSON response."""

    def __init__(self, response: Mapping[str, Any] | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _anthropic_response(text: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _openai_response(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-5.5-2026-04-23",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


# ---------------------------------------------------------------------------------------------
# config/models.toml (D-019)
# ---------------------------------------------------------------------------------------------


def test_models_config_pins_the_d019_models() -> None:
    from groundcite.config import MODELS_CONFIG_PATH, load_models_config

    assert Path(MODELS_CONFIG_PATH) == REPO / "config" / "models.toml"
    cfg = load_models_config()
    # D-107: the judge is back on the D-019 pin, a different family from the Anthropic teacher.
    assert cfg["judge"]["model"] == "openai:gpt-5.5-2026-04-23"
    assert cfg["teacher"]["model"] == "anthropic:claude-opus-5-5"
    assert cfg["cross_check"]["model"] == "anthropic:claude-opus-5-5"
    assert set(cfg["ablation"]["models"]) == {
        "anthropic:claude-opus-5-5",
        "openai:gpt-5.5-2026-04-23",
    }


def test_judge_model_is_a_dated_snapshot_not_an_alias() -> None:
    from groundcite.config import load_models_config

    judge = load_models_config()["judge"]["model"]
    assert judge.startswith("openai:")
    assert "latest" not in judge
    assert re.search(r"-\d{4}-\d{2}-\d{2}$", judge)


def test_judge_decoding_params_are_fixed() -> None:
    """The judge runs with the pinned [decoding.openai] table: a positive max_completion_tokens
    and a fixed reasoning effort, the settings M1 and the H1 pilot were scored with."""
    from groundcite.config import decoding_params, load_models_config

    cfg = load_models_config()
    judge = cfg["judge"]["model"]
    params = decoding_params(judge)
    assert params == cfg["decoding"]["openai"]
    assert params.get("max_completion_tokens", 0) > 0
    assert params.get("reasoning_effort") in {"minimal", "low", "medium", "high"}, params


def test_decoding_params_are_fixed_per_provider() -> None:
    from groundcite.config import decoding_params, load_models_config

    cfg = load_models_config()
    anth = decoding_params("anthropic:claude-opus-5-5")
    oai = decoding_params("openai:gpt-5.5-2026-04-23")
    assert anth == cfg["decoding"]["anthropic"]
    assert oai == cfg["decoding"]["openai"]
    assert anth["temperature"] == 0  # R28
    assert anth.get("max_tokens", 0) > 0
    assert oai, "openai decoding params must be pinned"
    if "temperature" in oai:
        assert oai["temperature"] == 0


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------


def test_registry_is_keyed_by_provider_prefix() -> None:
    from groundcite.backends import BACKENDS

    assert {"anthropic:", "openai:", "mock:", "hf:"} <= set(BACKENDS)


def test_unknown_prefix_is_rejected_by_name() -> None:
    from groundcite.backends import get_backend

    with pytest.raises(ValueError, match="nosuch"):
        get_backend("nosuch:model")


def test_backends_share_one_generate_interface(isolated_env: Path) -> None:
    from groundcite.backends import get_backend

    isolated_env.write_text(
        f"ANTHROPIC_API_KEY={FAKE_ANTHROPIC}\nOPENAI_API_KEY={FAKE_OPENAI}\n", encoding="utf-8"
    )
    specs = {
        "anthropic:claude-opus-5-5": FakeTransport(_anthropic_response("Inspect [TSB-0417].")),
        "openai:gpt-5.5-2026-04-23": FakeTransport(_openai_response("Inspect [TSB-0417].")),
    }
    for spec, transport in specs.items():
        backend = get_backend(spec, transport=transport)
        assert backend.name == spec
        assert backend.generate(MESSAGES, SYSTEM, {}) == "Inspect [TSB-0417]."
    for spec in ("mock:compliant", "mock:naive"):
        backend = get_backend(spec)
        assert backend.name == spec
        assert isinstance(backend.generate(MESSAGES, SYSTEM, {}), str)


def test_hf_backend_resolves_lazily_without_loading_a_model() -> None:
    from groundcite.backends import get_backend

    before = set(sys.modules)
    backend = get_backend("hf:org/some-model@abc1234")
    assert backend.name == "hf:org/some-model@abc1234"
    assert callable(backend.generate)
    assert not {"transformers", "torch"} & (set(sys.modules) - before)


def test_bare_repo_id_means_hf() -> None:
    from groundcite.backends import get_backend

    assert get_backend("org/some-model").name == "hf:org/some-model"


# ---------------------------------------------------------------------------------------------
# Keys (R25)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "var"),
    [
        ("anthropic:claude-opus-5-5", "ANTHROPIC_API_KEY"),
        ("openai:gpt-5.5-2026-04-23", "OPENAI_API_KEY"),
    ],
)
def test_missing_key_fails_before_any_work_naming_the_var(spec: str, var: str) -> None:
    from groundcite.backends import get_backend
    from groundcite.keys import MissingKeyError

    transport = FakeTransport(_anthropic_response("x"))
    with pytest.raises(MissingKeyError) as exc:
        get_backend(spec, transport=transport)
    assert var in str(exc.value)
    assert exc.value.env_var == var
    assert exc.value.exit_code == 2
    assert transport.calls == []


def test_missing_key_message_never_carries_another_key_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groundcite.backends import get_backend
    from groundcite.keys import MissingKeyError

    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    with pytest.raises(MissingKeyError) as exc:
        get_backend("anthropic:claude-opus-5-5", transport=FakeTransport({}))
    assert FAKE_OPENAI not in str(exc.value)
    assert FAKE_OPENAI not in repr(exc.value)


def test_empty_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from groundcite.keys import MissingKeyError, get_key

    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(MissingKeyError):
        get_key("OPENAI_API_KEY")


def test_mock_and_hf_backends_need_no_keys() -> None:
    from groundcite.backends import get_backend

    get_backend("mock:compliant")
    get_backend("mock:naive")
    get_backend("mock:judge")
    get_backend("hf:org/some-model")


def test_keys_load_from_the_env_file_override(isolated_env: Path) -> None:
    import os

    from groundcite.keys import get_key

    isolated_env.write_text(f"OPENAI_API_KEY={FAKE_OPENAI}\n", encoding="utf-8")
    assert get_key("OPENAI_API_KEY") == FAKE_OPENAI
    # Reading the file never exports keys into the process environment.
    assert "OPENAI_API_KEY" not in os.environ


def test_real_env_vars_take_precedence_over_the_env_file(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from groundcite.keys import get_key

    isolated_env.write_text(f"OPENAI_API_KEY={FAKE_OPENAI}\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE-FROMENV-2222")
    assert get_key("OPENAI_API_KEY") == "FAKE-FROMENV-2222"


def test_missing_env_file_is_not_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from groundcite.keys import get_key

    monkeypatch.setenv("GROUNDCITE_ENV_FILE", str(tmp_path / "does-not-exist.env"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    assert get_key("ANTHROPIC_API_KEY") == FAKE_ANTHROPIC


def test_default_env_file_is_the_repo_root_dotenv() -> None:
    from groundcite.keys import DEFAULT_ENV_FILE, ENV_FILE_VAR

    assert ENV_FILE_VAR == "GROUNDCITE_ENV_FILE"
    assert Path(DEFAULT_ENV_FILE) == REPO / ".env"


def test_backend_picks_up_keys_from_the_env_file(isolated_env: Path) -> None:
    from groundcite.backends import get_backend

    isolated_env.write_text(f"OPENAI_API_KEY={FAKE_OPENAI}\n", encoding="utf-8")
    transport = FakeTransport(_openai_response("ok"))
    get_backend("openai:gpt-5.5-2026-04-23", transport=transport).generate(MESSAGES, SYSTEM, {})
    _, headers, _ = transport.calls[0]
    assert headers.get("Authorization") == f"Bearer {FAKE_OPENAI}"


# ---------------------------------------------------------------------------------------------
# API backends against fake HTTP
# ---------------------------------------------------------------------------------------------


def test_anthropic_request_carries_pinned_model_and_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groundcite.backends import get_backend
    from groundcite.config import decoding_params, load_models_config

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    spec = load_models_config()["teacher"]["model"]
    params = decoding_params(spec)
    transport = FakeTransport(_anthropic_response("Inspect the hose [TSB-0417]."))
    out = get_backend(spec, transport=transport).generate(MESSAGES, SYSTEM, params)

    assert out == "Inspect the hose [TSB-0417]."
    assert len(transport.calls) == 1
    url, headers, body = transport.calls[0]
    assert url == ANTHROPIC_URL
    assert headers.get("x-api-key") == FAKE_ANTHROPIC
    assert "anthropic-version" in headers
    assert body["model"] == spec.split(":", 1)[1] == "claude-opus-5-5"
    for key, value in params.items():
        assert body[key] == value, key
    assert body["system"] == SYSTEM
    assert body["messages"] == MESSAGES


def test_openai_request_carries_pinned_ablation_model_and_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groundcite.backends import get_backend
    from groundcite.config import decoding_params, load_models_config

    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    # The OpenAI model is both the judge and the second ablation family.
    [spec] = [m for m in load_models_config()["ablation"]["models"] if m.startswith("openai:")]
    params = decoding_params(spec)
    transport = FakeTransport(_openai_response('{"spec_adherence": 1}'))
    out = get_backend(spec, transport=transport).generate(MESSAGES, SYSTEM, params)

    assert out == '{"spec_adherence": 1}'
    url, headers, body = transport.calls[0]
    assert url == OPENAI_URL
    assert headers.get("Authorization") == f"Bearer {FAKE_OPENAI}"
    assert body["model"] == spec.split(":", 1)[1] == "gpt-5.5-2026-04-23"
    for key, value in params.items():
        assert body[key] == value, key
    assert body["messages"][0]["role"] in ("system", "developer")
    assert body["messages"][0]["content"] == SYSTEM
    assert body["messages"][1:] == MESSAGES


def test_judge_reply_uses_config_decoding_for_the_pinned_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groundcite.backends import get_backend
    from groundcite.config import decoding_params, load_models_config
    from groundcite.judge import judge_reply

    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    spec = load_models_config()["judge"]["model"]
    verdict = '{"spec_adherence": 1, "robustness": 5, "reasoning": "ok"}'
    transport = FakeTransport(_openai_response(verdict))
    scenario = Scenario(id="t-1", messages=MESSAGES, provided_ids=PROVIDED)
    reply = "Inspect the PCV hose at the intake elbow for cracks [TSB-0417]."
    checks = [fn(reply, PROVIDED) for fn in CHECKS.values()]
    res = judge_reply(scenario, reply, checks, get_backend(spec, transport=transport))

    url, _, body = transport.calls[0]
    assert url == OPENAI_URL
    assert body["model"] == "gpt-5.5-2026-04-23"
    for key, value in decoding_params(spec).items():
        assert body[key] == value, key
    assert res.model == spec


# ---------------------------------------------------------------------------------------------
# Key values never leak (R25)
# ---------------------------------------------------------------------------------------------


def test_backend_errors_and_repr_never_show_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from groundcite.backends import get_backend

    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    # A transport error that echoes the request headers, as some HTTP errors do.
    boom = RuntimeError(f"401 Unauthorized; sent Authorization: Bearer {FAKE_OPENAI}")
    backend = get_backend("openai:gpt-5.5-2026-04-23", transport=FakeTransport(boom))
    assert FAKE_OPENAI not in repr(backend)
    with pytest.raises(Exception) as exc:
        backend.generate(MESSAGES, SYSTEM, {})
    assert FAKE_OPENAI not in str(exc.value)
    assert FAKE_OPENAI not in repr(exc.value)


def test_key_values_never_appear_in_logs_transcripts_or_results(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from groundcite.backends import get_backend
    from groundcite.judge import judge_reply
    from groundcite.report import aggregate_scores, transcript_row

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    caplog.set_level(logging.DEBUG)

    model_spec = "anthropic:claude-opus-5-5"
    judge_spec = "openai:gpt-5.5-2026-04-23"
    model = get_backend(
        model_spec, transport=FakeTransport(_anthropic_response("Inspect the hose [TSB-0417]."))
    )
    # First judge answer is malformed so the retry path (and its logging) runs too.
    judge_transport = FakeTransport(_openai_response("not json"))
    judge = get_backend(judge_spec, transport=judge_transport)

    scenario = Scenario(
        id="t-1",
        messages=MESSAGES,
        provided_ids=PROVIDED,
        gold={"mode": "cite", "supporting_ids": ["TSB-0417"]},
        category="clean",
    )
    reply = model.generate(scenario.messages, SYSTEM, {})
    checks = [fn(reply, PROVIDED, gold=scenario.gold) for fn in CHECKS.values()]
    result = judge_reply(scenario, reply, checks, judge)
    row = transcript_row(
        run_id="r-test",
        scenario=scenario,
        model=model_spec,
        model_revision=None,
        strategy="zero_shot",
        reply=reply,
        checks=checks,
        judge=result,
    )
    results = aggregate_scores([row], seed=0)

    out, err = capsys.readouterr()
    blob = "\n".join(
        [
            caplog.text,
            out,
            err,
            json.dumps(row),
            json.dumps(results),
            repr(result),
        ]
    )
    for value in (FAKE_ANTHROPIC, FAKE_OPENAI):
        assert value not in blob


# ---------------------------------------------------------------------------------------------
# Mock models (R26)
# ---------------------------------------------------------------------------------------------


def _check_map(reply: str, provided: list[str]) -> dict[str, bool | None]:
    return {name: fn(reply, provided).passed for name, fn in CHECKS.items()}


def test_mock_compliant_cites_exactly_one_provided_id_in_one_clean_step() -> None:
    from groundcite.backends import get_backend

    reply = get_backend("mock:compliant").generate(MESSAGES, SYSTEM, {})
    cited = parse_cited_ids(reply)
    assert len(cited) == 1 and cited[0] in PROVIDED
    results = _check_map(reply, PROVIDED)
    assert all(v is not False for v in results.values()), results


def test_mock_compliant_abstains_cleanly_when_nothing_is_provided() -> None:
    from groundcite.backends import get_backend

    reply = get_backend("mock:compliant").generate(NO_REF_MESSAGES, SYSTEM, {})
    assert parse_cited_ids(reply) == []
    assert reply.count("?") == 1
    results = _check_map(reply, [])
    assert all(v is not False for v in results.values()), results


def test_mock_naive_lists_steps_cites_several_ids_and_suggests_repairs() -> None:
    from groundcite.backends import get_backend

    reply = get_backend("mock:naive").generate(MESSAGES, SYSTEM, {})
    results = _check_map(reply, PROVIDED)
    assert results["one_step"] is False
    assert results["no_fix"] is False
    assert len(parse_cited_ids(reply)) >= 2
    assert any(results[name] is False for name in HARD_CHECKS)


@pytest.mark.parametrize("spec", ["mock:compliant", "mock:naive"])
def test_mock_models_are_deterministic(spec: str) -> None:
    from groundcite.backends import get_backend

    a = get_backend(spec).generate(MESSAGES, SYSTEM, {})
    b = get_backend(spec).generate(MESSAGES, SYSTEM, {})
    assert a == b
    other = get_backend(spec).generate(NO_REF_MESSAGES, SYSTEM, {})
    assert isinstance(other, str) and other


def test_unknown_mock_name_is_rejected() -> None:
    from groundcite.backends import get_backend

    with pytest.raises(ValueError, match="nosuch"):
        get_backend("mock:nosuch")
