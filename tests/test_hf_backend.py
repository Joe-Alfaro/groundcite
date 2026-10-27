"""HFBackend without real torch (R10, R27, D-039, D-043): HF_TOKEN, LoRA adapters, one renderer.

Fake `torch`, `transformers` and `peft` modules stand in for the real ones, so nothing is
downloaded. The backend finds out whether a repo is a LoRA adapter through
`groundcite.hub.fetch_adapter_config(repo, revision)` (the same lookup eval.py uses for the base
rows); the tests replace it. With an adapter, the base named by `base_model_name_or_path` is
loaded with `AutoModelForCausalLM.from_pretrained` and the adapter goes on top with
`peft.PeftModel.from_pretrained(model, repo, ...)`. The prompt is rendered by
`prompts.render_chat_text` (one chat-template call with thinking disabled) and the resulting
string is what gets tokenized.
"""

from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

from groundcite import backends, hub

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
FAKE_TOKEN = "fake-hub-token-123"
BASE_REPO = "Qwen/Qwen3-1.7B"


class _Encoded(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> _Encoded:
        return self


class _Ids:
    shape = (1, 3)


def _encoded() -> _Encoded:
    return _Encoded(input_ids=_Ids())


class _FakeTokenizer:
    """A chat template `<|role|>content<|end|>` with generation prompt `<|assistant|>`.

    Records every template call and every string handed to the tokenizer itself.
    """

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.tokenized: list[str] = []

    def apply_chat_template(self, chat: Any, **kwargs: Any) -> Any:
        self.calls.append({"apply_chat_template": kwargs})
        if kwargs.get("tokenize", True) is False:
            text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in chat)
            if kwargs.get("add_generation_prompt"):
                text += "<|assistant|>"
            return text
        return _encoded()

    def __call__(self, text: Any, **kwargs: Any) -> _Encoded:
        self.tokenized.append(text if isinstance(text, str) else text[0])
        return _encoded()

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.tokenized.append(text)
        return [0, 0, 0]

    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
        return "Check the hose [A-1]."


class _FakeModel:
    device = "cpu"

    def __init__(self, label: str) -> None:
        self.label = label

    def to(self, device: Any) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def merge_and_unload(self) -> _FakeModel:
        return self

    def generate(self, *args: Any, **kwargs: Any) -> list[list[int]]:
        return [[0, 0, 0, 1, 2]]


class Fakes:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tokenizers: list[_FakeTokenizer] = []
        self.adapter_lookups: list[tuple[str, str | None]] = []


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, adapter_config: dict[str, Any] | None = None
) -> Fakes:
    fakes = Fakes()
    calls = fakes.calls

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.no_grad = lambda: _NullContext()  # type: ignore[attr-defined]
    torch.inference_mode = lambda: _NullContext()  # type: ignore[attr-defined]

    def tok_from_pretrained(repo: str, **kwargs: Any) -> _FakeTokenizer:
        calls.append({"tokenizer": repo, **kwargs})
        tok = _FakeTokenizer(calls)
        fakes.tokenizers.append(tok)
        return tok

    def model_from_pretrained(repo: str, **kwargs: Any) -> _FakeModel:
        calls.append({"model": repo, **kwargs})
        return _FakeModel(f"base:{repo}")

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=tok_from_pretrained
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=model_from_pretrained
    )

    def peft_from_pretrained(model: Any, repo: str, **kwargs: Any) -> _FakeModel:
        calls.append({"peft": repo, "base": getattr(model, "label", None), **kwargs})
        return _FakeModel(f"peft:{repo}")

    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=peft_from_pretrained
    )

    def fetch(repo: str, revision: str | None = None) -> dict[str, Any] | None:
        fakes.adapter_lookups.append((repo, revision))
        return adapter_config

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(hub, "fetch_adapter_config", fetch)
    monkeypatch.setattr(backends, "fetch_adapter_config", fetch, raising=False)
    return fakes


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _loader_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in calls if "tokenizer" in c or "model" in c or "peft" in c]


def _adapter() -> dict[str, Any]:
    return {"base_model_name_or_path": BASE_REPO, "peft_type": "LORA"}


# ---------------------------------------------------------------------------------------------
# HF_TOKEN (D-039)
# ---------------------------------------------------------------------------------------------


def test_hf_backend_passes_hf_token_when_set(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    isolated_env.write_text(f"HF_TOKEN={FAKE_TOKEN}\n", encoding="utf-8")
    fakes = _install_fakes(monkeypatch)
    backend = get_backend("hf:org/model@abc123")
    backend.generate([{"role": "user", "content": "hi"}], "sys", {})
    loaders = _loader_calls(fakes.calls)
    assert len(loaders) >= 2
    for call in loaders:
        assert call.get("token") == FAKE_TOKEN, call.keys()
        assert call.get("revision") == "abc123"


def test_hf_backend_works_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    fakes = _install_fakes(monkeypatch)
    backend = get_backend("hf:org/model")
    reply = backend.generate([{"role": "user", "content": "hi"}], "sys", {})
    assert reply == "Check the hose [A-1]."
    for call in _loader_calls(fakes.calls):
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

    fakes = _install_fakes(monkeypatch)
    get_backend("hf:org/model").generate([{"role": "user", "content": "hi"}], "sys", {})
    template_calls = [c["apply_chat_template"] for c in fakes.calls if "apply_chat_template" in c]
    assert template_calls and all(c["enable_thinking"] is False for c in template_calls)


# ---------------------------------------------------------------------------------------------
# One renderer for training and eval (R27, GOTCHA G26)
# ---------------------------------------------------------------------------------------------


def test_hf_backend_tokenizes_exactly_the_render_chat_text_string(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    """The string the HF backend feeds the model is the shared renderer's eval string for the
    scenario, byte for byte, rendered with thinking disabled."""
    from groundcite.backends import get_backend
    from groundcite.evalset import load_eval_set
    from groundcite.prompts import render_chat_text, render_messages

    fakes = _install_fakes(monkeypatch)
    scenario = load_eval_set(ABLATION)[0]
    system, messages = render_messages(scenario, "zero_shot")

    get_backend("hf:org/model").generate(messages, system, {})

    expected = render_chat_text(_FakeTokenizer([]), system, messages)
    assert expected.endswith("<|assistant|>")
    tokenized = [text for tok in fakes.tokenizers for text in tok.tokenized]
    assert tokenized == [expected], "the backend must tokenize the render_chat_text() string"
    template_calls = [c["apply_chat_template"] for c in fakes.calls if "apply_chat_template" in c]
    assert template_calls and all(c["enable_thinking"] is False for c in template_calls)

    train = render_chat_text(_FakeTokenizer([]), system, messages, reply="Check [A-1].")
    assert train.startswith(expected.removesuffix("<|assistant|>"))


# ---------------------------------------------------------------------------------------------
# LoRA adapters through peft (R10, D-043 item 4)
# ---------------------------------------------------------------------------------------------


def test_hf_backend_loads_a_lora_adapter_on_its_base(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    fakes = _install_fakes(monkeypatch, adapter_config=_adapter())
    reply = get_backend("hf:org/tuned@abc123").generate(
        [{"role": "user", "content": "hi"}], "sys", {}
    )
    assert reply == "Check the hose [A-1]."
    assert fakes.adapter_lookups and fakes.adapter_lookups[0][0] == "org/tuned"

    models = [c for c in fakes.calls if "model" in c]
    assert [c["model"] for c in models] == [BASE_REPO], "load the base named by adapter_config"
    adapters = [c for c in fakes.calls if "peft" in c]
    assert len(adapters) == 1
    assert adapters[0]["peft"] == "org/tuned"
    assert adapters[0]["base"] == f"base:{BASE_REPO}"
    assert adapters[0].get("revision") == "abc123"


def test_hf_backend_passes_hf_token_to_the_adapter_and_base_loads(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    isolated_env.write_text(f"HF_TOKEN={FAKE_TOKEN}\n", encoding="utf-8")
    fakes = _install_fakes(monkeypatch, adapter_config=_adapter())
    get_backend("hf:org/tuned@abc123").generate([{"role": "user", "content": "hi"}], "sys", {})
    loaders = _loader_calls(fakes.calls)
    assert {"tokenizer", "model", "peft"} <= {k for c in loaders for k in c}
    for call in loaders:
        assert call.get("token") == FAKE_TOKEN, call


def test_hf_backend_loads_a_merged_model_without_peft(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, no_network: None
) -> None:
    from groundcite.backends import get_backend

    fakes = _install_fakes(monkeypatch, adapter_config=None)
    get_backend("hf:org/merged@abc123").generate([{"role": "user", "content": "hi"}], "sys", {})
    assert fakes.adapter_lookups, "the backend must check for adapter_config.json"
    assert [c["model"] for c in fakes.calls if "model" in c] == ["org/merged"]
    assert not [c for c in fakes.calls if "peft" in c]


# ---------------------------------------------------------------------------------------------
# hf: works on a plain install: torch, transformers and peft are core dependencies (R10)
# ---------------------------------------------------------------------------------------------


def _dep_name(requirement: str) -> str:
    name = requirement.split(";")[0]
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", " "):
        name = name.split(sep)[0]
    return name.strip().lower()


def test_torch_transformers_and_peft_are_core_dependencies() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = {_dep_name(r) for r in project["dependencies"]}
    missing = {"torch", "transformers", "peft"} - core
    assert not missing, f"not core dependencies (an extra is not one plain command): {missing}"
