"""The ``xai:`` backend, judge-candidate pins and the price table (D-095, D-096).

``xai:<model>`` reaches xAI through the official ``openai`` SDK client pointed at
``https://api.x.ai/v1`` with the key from ``XAI_API_KEY`` (env file as for the other keys; real
env vars win). It speaks either the Responses API or Chat Completions; the primary path is set by
``[xai] api`` in ``config/models.toml`` (``"responses"`` or ``"chat"``) and can be overridden per
backend with ``XAIBackend(spec, transport, api=...)``. In both paths:

- the request carries the pinned model ID exactly and the given decoding params verbatim;
- a rejected droppable param is dropped, retried and left out of ``effective_params``, as for
  the openai backend;
- a non-retryable 4xx raises a redacted ``BackendError`` that stops the run; 429 and 5xx are
  retryable;
- reasoning content (``reasoning_content`` on a chat message, ``reasoning`` output items in a
  response) is never returned as the reply;
- every successful call appends ``{"input_tokens", "output_tokens", "reasoning_tokens"}`` to
  ``backend.usage``. ``output_tokens`` is the visible output only: when the provider's output
  count already includes reasoning (``total_tokens == input + output``), the reasoning tokens are
  subtracted. ``reasoning_tokens`` is None when the provider reports none.

The Anthropic backend records usage the same way (``reasoning_tokens`` None: Anthropic bills
thinking inside ``output_tokens``), so Sonnet and Haiku candidate runs can be costed too.

``config.price(spec)`` reads ``[prices]`` and ``config.cost_usd(spec, usages)`` turns recorded
usage into dollars: (input x input price + (output + reasoning) x output price) / 1e6.

The active judge is NOT switched: Grok only becomes the judge after the agreement check passes.

Sockets are blocked and every key is fake; no test calls a real API.
"""

from __future__ import annotations

import json
import logging
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from config_helpers import patched_config

from groundcite import backends, config
from groundcite.checks import CHECKS
from groundcite.evalset import Scenario

pytestmark = pytest.mark.usefixtures("no_network", "isolated_env")

XAI_SPEC = "xai:grok-4.20-0309-reasoning"
XAI_MODEL = "grok-4.20-0309-reasoning"
SONNET = "anthropic:claude-sonnet-5-5"
HAIKU = "anthropic:claude-haiku-5-5"
REFERENCE = "openai:gpt-5.5-2026-04-23"
ACTIVE_JUDGE = "anthropic:claude-opus-5-5"

XAI_BASE = "https://api.x.ai/v1"
CHAT_URL = f"{XAI_BASE}/chat/completions"
RESPONSES_URL = f"{XAI_BASE}/responses"

FAKE_XAI = "xai-FAKE-3333-TESTVALUE-do-not-use"
FAKE_OPENAI = "FAKE-OPENAI-1111-TESTVALUE"

SYSTEM = "You are a grounded diagnostic assistant."
MESSAGES = [
    {
        "role": "user",
        "content": (
            "References:\n"
            "[SM-2210] Fuel pressure test at the rail test port, 55-62 psi.\n"
            "[TSB-0417] Inspect the PCV hose at the intake elbow for cracks.\n\n"
            "2015 compact sedan, 2.0L. P0171, rough idle when cold."
        ),
    }
]
PROVIDED = ["SM-2210", "TSB-0417"]
REPLY = "Inspect the PCV hose at the intake elbow for cracks [TSB-0417]."
THINKING = "SECRET-REASONING: maybe it is the fuel pump [SM-2210]"
APIS = ("responses", "chat")
TOKEN_CAP = {"responses": "max_output_tokens", "chat": "max_completion_tokens"}


@pytest.fixture(autouse=True)
def fresh_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh process: no model has rejected a parameter yet."""
    monkeypatch.setattr(backends, "_REJECTED", {})


@pytest.fixture
def xai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI)
    return FAKE_XAI


# ---------------------------------------------------------------------------------------------
# Fake xAI HTTP
# ---------------------------------------------------------------------------------------------


class FakeHTTPError(Exception):
    """An SDK-style HTTP error: ``status_code`` and, for 400s, the offending ``param``."""

    def __init__(self, status: int, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status
        self.param = param


def chat_response(
    text: str | None,
    *,
    reasoning: str | None = THINKING,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-xai-test",
        "object": "chat.completion",
        "model": XAI_MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": dict(
            usage
            or {
                # xAI style: completion_tokens excludes reasoning (total = prompt + both).
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1500,
                "completion_tokens_details": {"reasoning_tokens": 400},
            }
        ),
    }


def responses_response(
    text: str | None,
    *,
    reasoning: str | None = THINKING,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if reasoning is not None:
        output.append(
            {
                "type": "reasoning",
                "id": "rs_test",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    if text is not None:
        output.append(
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    return {
        "id": "resp_test",
        "object": "response",
        "model": XAI_MODEL,
        "status": "completed",
        "output": output,
        "usage": dict(
            usage
            or {
                # Responses style: output_tokens includes reasoning (total = input + output).
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "output_tokens_details": {"reasoning_tokens": 400},
            }
        ),
    }


def response_for(api: str, text: str | None, **kwargs: Any) -> dict[str, Any]:
    return (responses_response if api == "responses" else chat_response)(text, **kwargs)


class FakeTransport:
    """Stands in for HTTP: records (url, headers, body); answers from a list, one per call.

    An entry may be a response dict, an exception to raise, or a callable(body) returning either.
    The last entry repeats once the list runs out.
    """

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if callable(answer) and not isinstance(answer, Exception):
            answer = answer(body)
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)


def xai(transport: FakeTransport, api: str) -> Any:
    from groundcite.backends import XAIBackend

    return XAIBackend(XAI_SPEC, transport, api=api)


def user_items(api: str, body: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = body["input"] if api == "responses" else body["messages"]
    return [dict(m) for m in items if m.get("role") == "user"]


def system_text(api: str, body: Mapping[str, Any]) -> str | None:
    items = body["input"] if api == "responses" else body["messages"]
    if api == "responses" and body.get("instructions"):
        return str(body["instructions"])
    first = items[0]
    return str(first["content"]) if first.get("role") in ("system", "developer") else None


# ---------------------------------------------------------------------------------------------
# Registry, URLs and the config switch
# ---------------------------------------------------------------------------------------------


def test_xai_prefix_is_registered() -> None:
    assert "xai:" in backends.BACKENDS


def test_xai_base_url_and_endpoints() -> None:
    assert backends.XAI_BASE_URL.rstrip("/") == XAI_BASE


@pytest.mark.parametrize("api", APIS)
def test_each_api_posts_to_its_xai_endpoint(xai_key: str, api: str) -> None:
    transport = FakeTransport(response_for(api, REPLY))
    xai(transport, api).generate(MESSAGES, SYSTEM, {})
    url, headers, _ = transport.calls[0]
    assert url == (RESPONSES_URL if api == "responses" else CHAT_URL)
    assert headers.get("Authorization") == f"Bearer {FAKE_XAI}"


def test_get_backend_builds_the_xai_backend(xai_key: str) -> None:
    from groundcite.backends import XAIBackend, get_backend

    backend = get_backend(XAI_SPEC)
    assert isinstance(backend, XAIBackend)
    assert backend.name == XAI_SPEC
    assert FAKE_XAI not in repr(backend)


def test_an_unknown_api_is_refused(xai_key: str) -> None:
    from groundcite.backends import XAIBackend

    with pytest.raises(ValueError, match="api"):
        XAIBackend(XAI_SPEC, FakeTransport({}), api="completions")


@pytest.mark.parametrize("api", APIS)
def test_the_default_api_comes_from_the_config_switch(
    xai_key: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api: str
) -> None:
    from groundcite.backends import XAIBackend

    def edit(cfg: dict[str, Any]) -> None:
        cfg.setdefault("xai", {})["api"] = api

    patched_config(monkeypatch, tmp_path, edit)
    transport = FakeTransport(response_for(api, REPLY))
    assert XAIBackend(XAI_SPEC, transport).generate(MESSAGES, SYSTEM, {}) == REPLY
    assert transport.calls[0][0] == (RESPONSES_URL if api == "responses" else CHAT_URL)


# ---------------------------------------------------------------------------------------------
# Requests: pinned model, messages, params
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
def test_request_carries_the_pinned_model_messages_and_params(xai_key: str, api: str) -> None:
    params = {"temperature": 0, TOKEN_CAP[api]: 123}
    transport = FakeTransport(response_for(api, REPLY))
    out = xai(transport, api).generate(MESSAGES, SYSTEM, params)

    assert out == REPLY
    assert len(transport.calls) == 1
    _, _, body = transport.calls[0]
    assert body["model"] == XAI_MODEL
    for key, val in params.items():
        assert body[key] == val, key
    assert system_text(api, body) == SYSTEM
    assert user_items(api, body) == MESSAGES


@pytest.mark.parametrize("api", APIS)
def test_no_system_prompt_sends_only_the_conversation(xai_key: str, api: str) -> None:
    transport = FakeTransport(response_for(api, REPLY))
    xai(transport, api).generate(MESSAGES, None, {})
    _, _, body = transport.calls[0]
    assert system_text(api, body) is None
    assert user_items(api, body) == MESSAGES


# ---------------------------------------------------------------------------------------------
# The official openai SDK client, pointed at xAI
# ---------------------------------------------------------------------------------------------


class _Dumpable:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump()


class FakeOpenAIClient:
    instances: list[FakeOpenAIClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.created: list[tuple[str, dict[str, Any]]] = []
        FakeOpenAIClient.instances.append(self)
        client = self

        class _Completions:
            def create(self, **body: Any) -> _Dumpable:
                client.created.append(("chat", body))
                return _Dumpable(chat_response(REPLY))

        class _Responses:
            def create(self, **body: Any) -> _Dumpable:
                client.created.append(("responses", body))
                return _Dumpable(responses_response(REPLY))

        self.chat = types.SimpleNamespace(completions=_Completions())
        self.responses = _Responses()


@pytest.mark.parametrize("api", APIS)
def test_sdk_path_uses_the_openai_client_with_the_xai_base_url(
    xai_key: str, monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    import openai

    from groundcite.backends import XAIBackend

    monkeypatch.setattr(FakeOpenAIClient, "instances", [])
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    backend = XAIBackend(XAI_SPEC, None, api=api)
    assert backend.generate(MESSAGES, SYSTEM, {"temperature": 0}) == REPLY

    [client] = FakeOpenAIClient.instances
    assert str(client.kwargs.get("base_url")).rstrip("/") == XAI_BASE
    assert client.kwargs.get("api_key") == FAKE_XAI
    assert client.kwargs.get("max_retries") == 0, "callers retry; the SDK must not stack retries"
    [(endpoint, body)] = client.created
    assert endpoint == api
    assert body["model"] == XAI_MODEL
    assert body["temperature"] == 0


# ---------------------------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------------------------


def test_missing_key_fails_before_any_work_naming_xai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groundcite.backends import get_backend
    from groundcite.keys import MissingKeyError

    # Another provider's key is set; it is never borrowed for xAI or shown in the error.
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    transport = FakeTransport(chat_response(REPLY))
    with pytest.raises(MissingKeyError) as exc:
        get_backend(XAI_SPEC, transport=transport)
    assert exc.value.env_var == "XAI_API_KEY"
    assert "XAI_API_KEY" in str(exc.value)
    assert exc.value.exit_code == 2
    assert FAKE_OPENAI not in str(exc.value)
    assert transport.calls == []


def test_empty_xai_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from groundcite.backends import get_backend
    from groundcite.keys import MissingKeyError

    monkeypatch.setenv("XAI_API_KEY", "")
    with pytest.raises(MissingKeyError):
        get_backend(XAI_SPEC, transport=FakeTransport({}))


@pytest.mark.parametrize("api", APIS)
def test_key_loads_from_the_env_file(isolated_env: Path, api: str) -> None:
    import os

    isolated_env.write_text(f"XAI_API_KEY={FAKE_XAI}\n", encoding="utf-8")
    transport = FakeTransport(response_for(api, REPLY))
    xai(transport, api).generate(MESSAGES, SYSTEM, {})
    assert transport.calls[0][1].get("Authorization") == f"Bearer {FAKE_XAI}"
    assert "XAI_API_KEY" not in os.environ


def test_real_env_var_wins_over_the_env_file(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_env.write_text(f"XAI_API_KEY={FAKE_XAI}\n", encoding="utf-8")
    monkeypatch.setenv("XAI_API_KEY", "xai-FAKE-FROMENV-4444")
    transport = FakeTransport(chat_response(REPLY))
    xai(transport, "chat").generate(MESSAGES, SYSTEM, {})
    assert transport.calls[0][1].get("Authorization") == "Bearer xai-FAKE-FROMENV-4444"


# ---------------------------------------------------------------------------------------------
# Rejected params, 4xx stops and retryable errors
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
def test_a_rejected_param_is_dropped_and_recorded_in_effective_params(
    xai_key: str, api: str
) -> None:
    params = {"temperature": 0, TOKEN_CAP[api]: 123}

    def reject_temperature(body: Mapping[str, Any]) -> Any:
        if "temperature" in body:
            return FakeHTTPError(
                400, "Model does not support parameter: temperature", "temperature"
            )
        return response_for(api, REPLY)

    transport = FakeTransport(reject_temperature)
    backend = xai(transport, api)
    assert backend.generate(MESSAGES, SYSTEM, params) == REPLY
    assert len(transport.calls) == 2
    assert "temperature" not in transport.calls[1][2]
    assert backend.effective_params == {TOKEN_CAP[api]: 123}

    # Later calls in the process skip the rejected param straight away.
    backend.generate(MESSAGES, SYSTEM, params)
    assert len(transport.calls) == 3
    assert "temperature" not in transport.calls[2][2]


@pytest.mark.parametrize("api", APIS)
def test_restored_effective_params_skip_a_param_an_earlier_process_dropped(
    xai_key: str, api: str
) -> None:
    params = {"temperature": 0, TOKEN_CAP[api]: 123}
    transport = FakeTransport(response_for(api, REPLY))
    backend = xai(transport, api)
    backend.restore_effective_params(params, {TOKEN_CAP[api]: 123})
    backend.generate(MESSAGES, SYSTEM, params)
    assert "temperature" not in transport.calls[0][2]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_non_retryable_4xx_stops_the_run_with_a_redacted_message(
    xai_key: str, api: str, status: int
) -> None:
    from groundcite.backends import BackendError

    # Some HTTP errors echo the request headers, key included.
    boom = FakeHTTPError(status, f"Error code: {status} (Authorization: Bearer {FAKE_XAI})")
    transport = FakeTransport(boom)
    backend = xai(transport, api)
    with pytest.raises(BackendError) as exc:
        backend.generate(MESSAGES, SYSTEM, {TOKEN_CAP[api]: 123})
    assert exc.value.status == status
    assert exc.value.stops_run
    assert not exc.value.retryable
    assert FAKE_XAI not in str(exc.value)
    assert FAKE_XAI not in repr(exc.value)
    assert exc.value.__cause__ is None or FAKE_XAI not in str(exc.value.__cause__)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_429_and_5xx_are_retried(xai_key: str, api: str, status: int) -> None:
    from groundcite.backends import BackendError
    from groundcite.runner import Retrying

    transient = FakeHTTPError(status, f"Error code: {status} (Bearer {FAKE_XAI})")
    transport = FakeTransport(transient, response_for(api, REPLY))
    backend = xai(transport, api)
    with pytest.raises(BackendError) as exc:
        backend.generate(MESSAGES, SYSTEM, {})
    assert exc.value.retryable and not exc.value.stops_run
    assert FAKE_XAI not in str(exc.value)

    transport.calls.clear()
    sleeps: list[float] = []
    retrying = Retrying(xai(transport, api), sleep=sleeps.append)
    assert retrying.generate(MESSAGES, SYSTEM, {}) == REPLY
    assert len(transport.calls) == 2
    assert len(sleeps) == 1


# ---------------------------------------------------------------------------------------------
# Reasoning content is never the reply
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
def test_reasoning_content_is_not_part_of_the_reply(xai_key: str, api: str) -> None:
    transport = FakeTransport(response_for(api, REPLY, reasoning=THINKING))
    out = xai(transport, api).generate(MESSAGES, SYSTEM, {})
    assert out == REPLY
    assert "SECRET-REASONING" not in out


@pytest.mark.parametrize("api", APIS)
def test_reasoning_without_a_reply_is_an_error_not_a_reply(xai_key: str, api: str) -> None:
    from groundcite.backends import BackendError

    empty = None if api == "responses" else ""
    transport = FakeTransport(response_for(api, empty, reasoning=THINKING))
    with pytest.raises(BackendError) as exc:
        xai(transport, api).generate(MESSAGES, SYSTEM, {})
    assert "SECRET-REASONING" not in str(exc.value)


# ---------------------------------------------------------------------------------------------
# Token usage per call
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "usage", "expected"),
    [
        # xAI chat style: completion excludes reasoning (total = prompt + completion + reasoning).
        (
            "chat",
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1500,
                "completion_tokens_details": {"reasoning_tokens": 400},
            },
            {"input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 400},
        ),
        # OpenAI chat style: completion includes reasoning (total = prompt + completion).
        (
            "chat",
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "completion_tokens_details": {"reasoning_tokens": 400},
            },
            {"input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 400},
        ),
        # No reasoning reported.
        (
            "chat",
            {"prompt_tokens": 900, "completion_tokens": 80, "total_tokens": 980},
            {"input_tokens": 900, "output_tokens": 80, "reasoning_tokens": None},
        ),
        # Responses style: output includes reasoning (total = input + output).
        (
            "responses",
            {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "output_tokens_details": {"reasoning_tokens": 400},
            },
            {"input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 400},
        ),
        (
            "responses",
            {"input_tokens": 700, "output_tokens": 60, "total_tokens": 760},
            {"input_tokens": 700, "output_tokens": 60, "reasoning_tokens": None},
        ),
    ],
)
def test_usage_is_captured_per_call(
    xai_key: str, api: str, usage: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    transport = FakeTransport(response_for(api, REPLY, usage=usage))
    backend = xai(transport, api)
    assert backend.usage == []
    backend.generate(MESSAGES, SYSTEM, {})
    assert backend.usage == [expected]
    backend.generate(MESSAGES, SYSTEM, {})
    assert backend.usage == [expected, expected], "one entry per successful call"


@pytest.mark.parametrize("api", APIS)
def test_failed_calls_record_no_usage(xai_key: str, api: str) -> None:
    from groundcite.backends import BackendError

    backend = xai(FakeTransport(FakeHTTPError(500, "server error")), api)
    with pytest.raises(BackendError):
        backend.generate(MESSAGES, SYSTEM, {})
    assert backend.usage == []


def test_anthropic_backend_records_usage_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sonnet and Haiku candidate runs are costed from recorded usage as well."""
    from groundcite.backends import AnthropicBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE-ANTHROPIC-0000-TESTVALUE")
    response = {
        "content": [{"type": "text", "text": REPLY}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1200, "output_tokens": 300},
    }
    backend = AnthropicBackend(SONNET, FakeTransport(response))
    backend.generate(MESSAGES, SYSTEM, {})
    assert backend.usage == [{"input_tokens": 1200, "output_tokens": 300, "reasoning_tokens": None}]


# ---------------------------------------------------------------------------------------------
# The key never leaks
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
def test_key_never_appears_in_logs_transcripts_results_or_exceptions(
    xai_key: str,
    api: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from groundcite.backends import BackendError
    from groundcite.judge import judge_reply
    from groundcite.report import aggregate_scores, transcript_row

    caplog.set_level(logging.DEBUG)
    verdict = '{"spec_adherence": 1, "robustness": 5, "reasoning": "ok"}'

    def reject_temperature(body: Mapping[str, Any]) -> Any:
        if "temperature" in body:
            return FakeHTTPError(400, f"temperature unsupported (Bearer {FAKE_XAI})", "temperature")
        return response_for(api, verdict)

    # The first judge answer is malformed so the retry path (and its logging) runs too.
    transport = FakeTransport(response_for(api, "not json"), reject_temperature)
    judge = xai(transport, api)
    scenario = Scenario(
        id="x-1",
        messages=MESSAGES,
        provided_ids=PROVIDED,
        gold={"mode": "cite", "supporting_ids": ["TSB-0417"]},
        category="clean",
    )
    checks = [fn(REPLY, PROVIDED, gold=scenario.gold) for fn in CHECKS.values()]
    result = judge_reply(scenario, REPLY, checks, judge, params={"temperature": 0})
    row = transcript_row(
        run_id="r-xai",
        scenario=scenario,
        model="mock:compliant",
        model_revision=None,
        strategy="zero_shot",
        reply=REPLY,
        checks=checks,
        judge=result,
    )
    results = aggregate_scores([row], seed=0)

    failing = xai(FakeTransport(RuntimeError(f"connection reset; key={FAKE_XAI}")), api)
    with pytest.raises(BackendError) as exc:
        failing.generate(MESSAGES, SYSTEM, {})

    out, err = capsys.readouterr()
    blob = "\n".join(
        [
            caplog.text,
            out,
            err,
            json.dumps(row),
            json.dumps(results),
            repr(result),
            repr(judge),
            json.dumps(judge.usage),
            json.dumps(judge.effective_params),
            str(exc.value),
            repr(exc.value),
        ]
    )
    assert FAKE_XAI not in blob


# ---------------------------------------------------------------------------------------------
# config/models.toml: candidates, the unchanged active judge, decoding, prices
# ---------------------------------------------------------------------------------------------


def test_judge_candidates_list_the_pinned_ids() -> None:
    cfg = config.load_models_config()
    candidates = cfg["judge_candidates"]
    assert candidates["models"] == [XAI_SPEC, SONNET, HAIKU]
    assert candidates["reference"] == REFERENCE
    for spec in candidates["models"]:
        assert "latest" not in spec


def test_the_active_judge_is_not_switched_before_the_agreement_check() -> None:
    cfg = config.load_models_config()
    assert cfg["judge"]["model"] == ACTIVE_JUDGE


def test_xai_api_switch_and_decoding_are_consistent() -> None:
    cfg = config.load_models_config()
    api = cfg["xai"]["api"]
    assert api in APIS
    params = config.decoding_params(XAI_SPEC)
    assert params == cfg["decoding"]["xai"]
    assert params.get(TOKEN_CAP[api], 0) > 0, f"the {api} API's output-token cap is pinned"
    other = TOKEN_CAP["chat" if api == "responses" else "responses"]
    assert other not in params, "the token-cap name must match the configured API"
    if "temperature" in params:
        assert params["temperature"] == 0


def test_every_candidate_has_a_price() -> None:
    cfg = config.load_models_config()
    for spec in cfg["judge_candidates"]["models"]:
        price = config.price(spec)
        assert price["input_per_mtok"] > 0, spec
        assert price["output_per_mtok"] > 0, spec
        assert price == cfg["prices"][spec]
    # D-095: $1.25 in / $2.50 out per million tokens.
    assert config.price(XAI_SPEC) == {"input_per_mtok": 1.25, "output_per_mtok": 2.5}


def test_a_spec_without_a_price_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="mock:judge"):
        config.price("mock:judge")


def test_cost_comes_from_recorded_usage_and_configured_prices() -> None:
    usages = [
        {"input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 400},
        {"input_tokens": 2000, "output_tokens": 50, "reasoning_tokens": None},
    ]
    # (3000 x 1.25 + (150 + 400) x 2.50) / 1e6
    assert config.cost_usd(XAI_SPEC, usages) == pytest.approx(0.005125)
    assert config.cost_usd(XAI_SPEC, []) == 0


def test_xai_key_is_a_known_key_var() -> None:
    from groundcite import keys

    assert "XAI_API_KEY" in keys.KEY_VARS
