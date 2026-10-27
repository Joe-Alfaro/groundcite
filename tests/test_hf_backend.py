"""HFBackend passes HF_TOKEN to the Hub loaders when it is set (D-039), without real torch.

Fake `torch` and `transformers` modules stand in for the real ones, so nothing is downloaded.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

FAKE_TOKEN = "fake-hub-token-123"


class _Encoded(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> _Encoded:
        return self


class _Ids:
    shape = (1, 3)


class _FakeTokenizer:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    def apply_chat_template(self, chat: Any, **kwargs: Any) -> _Encoded:
        self.calls.append({"apply_chat_template": kwargs})
        return _Encoded(input_ids=_Ids())

    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
        return "Check the hose [A-1]."


class _FakeModel:
    device = "cpu"

    def to(self, device: Any) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def generate(self, **kwargs: Any) -> list[list[int]]:
        return [[0, 0, 0, 1, 2]]


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]

    def tok_from_pretrained(repo: str, **kwargs: Any) -> _FakeTokenizer:
        calls.append({"tokenizer": repo, **kwargs})
        return _FakeTokenizer(calls)

    def model_from_pretrained(repo: str, **kwargs: Any) -> _FakeModel:
        calls.append({"model": repo, **kwargs})
        return _FakeModel()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=tok_from_pretrained
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=model_from_pretrained
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def _loader_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in calls if "tokenizer" in c or "model" in c]


def test_hf_backend_passes_hf_token_when_set(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    isolated_env.write_text(f"HF_TOKEN={FAKE_TOKEN}\n", encoding="utf-8")
    calls = _install_fakes(monkeypatch)
    backend = get_backend("hf:org/model@abc123")
    backend.generate([{"role": "user", "content": "hi"}], "sys", {})
    loaders = _loader_calls(calls)
    assert len(loaders) >= 2
    for call in loaders:
        assert call.get("token") == FAKE_TOKEN, call.keys()
        assert call.get("revision") == "abc123"


def test_hf_backend_works_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    calls = _install_fakes(monkeypatch)
    backend = get_backend("hf:org/model")
    reply = backend.generate([{"role": "user", "content": "hi"}], "sys", {})
    assert reply == "Check the hose [A-1]."
    for call in _loader_calls(calls):
        assert call.get("token") is None


def test_hf_backend_token_never_appears_in_repr(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    isolated_env.write_text(f"HF_TOKEN={FAKE_TOKEN}\n", encoding="utf-8")
    _install_fakes(monkeypatch)
    backend = get_backend("hf:org/model")
    backend.generate([{"role": "user", "content": "hi"}], "sys", {})
    assert FAKE_TOKEN not in repr(backend)


def test_hf_backend_applies_the_template_with_thinking_disabled(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    calls = _install_fakes(monkeypatch)
    get_backend("hf:org/model").generate([{"role": "user", "content": "hi"}], "sys", {})
    template_calls = [c["apply_chat_template"] for c in calls if "apply_chat_template" in c]
    assert template_calls and all(c["enable_thinking"] is False for c in template_calls)
