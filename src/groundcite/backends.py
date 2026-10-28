"""Generation and judge backends behind one interface (R10, R19, R25, R26).

``get_backend(spec)`` resolves a spec by its provider prefix: ``anthropic:``, ``openai:``,
``hf:`` (a bare ``org/repo`` means ``hf:``) or ``mock:``. Every backend has ``.name`` (the full
spec) and ``.generate(messages, system, params) -> str``.

API backends call the official SDKs by default. Tests inject ``transport(url, headers, body)``
instead, which receives the exact JSON request body. Keys are resolved when the backend is built,
so a missing key fails before any work; key values are redacted from every error.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from groundcite import hub
from groundcite.judge import CHECK_LINE_RE
from groundcite.keys import get_key, optional_key, redact
from groundcite.prompts import chat_template_sha256, render_chat_text
from groundcite.spec import parse_cited_ids

log = logging.getLogger(__name__)

Messages = Sequence[Mapping[str, str]]
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]]

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Sampling parameters a model may reject (Claude Opus 5.5 rejects temperature; some OpenAI
# reasoning snapshots do too). Only these are dropped on a rejection; anything else fails loudly.
DROPPABLE_PARAMS = frozenset({"temperature", "top_p", "top_k", "seed"})
# Parameters each model has rejected in this process, so later calls skip the failed request.
_REJECTED: dict[str, set[str]] = {}


class Backend(Protocol):
    name: str

    def generate(
        self, messages: Messages, system: str | None, params: Mapping[str, Any]
    ) -> str: ...


# Transient failures worth retrying with backoff: rate limits, timeouts and server errors.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
# The callers already retry transient failures with backoff, so the SDK clients don't retry on
# their own: stacked retries would turn one failing row into many paid attempts.
SDK_MAX_RETRIES = 0


class BackendError(RuntimeError):
    """A backend call failed. The message never contains a key value.

    ``status`` is the HTTP status when there was one; ``retryable`` marks transient failures
    (rate limits, server errors, dropped connections) that a caller may retry.
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable or status in RETRYABLE_STATUS

    @property
    def stops_run(self) -> bool:
        """A non-retryable 4xx (bad key, no access, malformed request): every later request would
        fail the same way, so the run stops instead of scoring failures (D-043)."""
        return self.status is not None and 400 <= self.status < 500 and not self.retryable


def _is_transient(exc: Exception) -> bool:
    """Connection and timeout errors (the SDKs raise these without an HTTP status)."""
    name = type(exc).__name__
    return (
        isinstance(exc, (TimeoutError, ConnectionError))
        or "Connection" in name
        or ("Timeout" in name)
    )


# ---------------------------------------------------------------------------------------------
# API backends
# ---------------------------------------------------------------------------------------------


def _rejected_param(exc: Exception, body: Mapping[str, Any]) -> str | None:
    """The droppable body parameter a 400 response (or the SDK, client-side) rejects, if any."""
    if isinstance(exc, TypeError):
        # Newer SDKs drop removed parameters from their signatures, e.g. anthropic 2.x has no
        # `temperature` because current Claude models reject it.
        match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
        if match and match.group(1) in DROPPABLE_PARAMS and match.group(1) in body:
            return match.group(1)
        return None
    if getattr(exc, "status_code", None) != 400:
        return None
    param = getattr(exc, "param", None)
    if param in DROPPABLE_PARAMS and param in body:
        return str(param)
    named = [p for p in DROPPABLE_PARAMS & set(body) if re.search(rf"\b{p}\b", str(exc))]
    return named[0] if len(named) == 1 else None


class _ApiBackend:
    url: str
    key_var: str

    def __init__(self, spec: str, transport: Transport | None) -> None:
        self.name = spec
        self.model = spec.split(":", 1)[1]
        self._key = get_key(self.key_var)
        self._transport = transport
        self._client: Any = None
        # The decoding parameters the last successful call actually ran with (R13).
        self.effective_params: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        rejected = _REJECTED.setdefault(self.model, set())
        sent = {k: v for k, v in params.items() if k not in rejected}
        while True:
            body = self._body(messages, system, sent)
            try:
                response = self._post(body)
                break
            except Exception as exc:
                param = _rejected_param(exc, sent)
                if param is None:
                    message = redact(f"{self.name} request failed: {exc}", [self._key])
                    status = getattr(exc, "status_code", None)
                    raise BackendError(
                        message,
                        status=status if isinstance(status, int) else None,
                        retryable=_is_transient(exc),
                    ) from None
                log.warning("%s rejected %r; retrying without it", self.name, param)
                rejected.add(param)
                del sent[param]
        self.effective_params = dict(sent)
        return self._text(response)

    def restore_effective_params(
        self, requested: Mapping[str, Any], recorded: Mapping[str, Any]
    ) -> None:
        """Adopt params an earlier process recorded: what it dropped counts as rejected."""
        dropped = {k for k in requested if k not in recorded and k in DROPPABLE_PARAMS}
        _REJECTED.setdefault(self.model, set()).update(dropped)
        self.effective_params = dict(recorded)

    def _post(self, body: dict[str, Any]) -> Mapping[str, Any]:
        if self._transport is not None:
            return self._transport(self.url, self._headers(), body)
        return self._sdk_call(body)

    def _body(
        self, messages: Messages, system: str | None, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _sdk_call(self, body: dict[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    def _text(self, response: Mapping[str, Any]) -> str:
        raise NotImplementedError


class AnthropicBackend(_ApiBackend):
    url = ANTHROPIC_URL
    key_var = "ANTHROPIC_API_KEY"

    def _body(
        self, messages: Messages, system: str | None, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages]}
        if system is not None:
            body["system"] = system
        return {**body, **params}

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _sdk_call(self, body: dict[str, Any]) -> Mapping[str, Any]:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._key, max_retries=SDK_MAX_RETRIES)
        result: Mapping[str, Any] = self._client.messages.create(**body).to_dict()
        return result

    def _text(self, response: Mapping[str, Any]) -> str:
        if response.get("stop_reason") == "refusal":
            raise BackendError(f"{self.name} refused the request")
        # Thinking blocks may precede the answer; the reply is the text blocks.
        texts = [b["text"] for b in response.get("content", []) if b.get("type") == "text"]
        if not texts:
            raise BackendError(f"{self.name} returned no text content")
        return "".join(texts)


class OpenAIBackend(_ApiBackend):
    url = OPENAI_URL
    key_var = "OPENAI_API_KEY"

    def _body(
        self, messages: Messages, system: str | None, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        msgs = [dict(m) for m in messages]
        if system is not None:
            msgs.insert(0, {"role": "developer", "content": system})
        return {"model": self.model, "messages": msgs, **params}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "content-type": "application/json"}

    def _sdk_call(self, body: dict[str, Any]) -> Mapping[str, Any]:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._key, max_retries=SDK_MAX_RETRIES)
        result: Mapping[str, Any] = self._client.chat.completions.create(**body).model_dump()
        return result

    def _text(self, response: Mapping[str, Any]) -> str:
        choice = response["choices"][0]
        content = choice["message"].get("content")
        if not content:
            reason = choice.get("finish_reason")
            raise BackendError(f"{self.name} returned no content (finish_reason={reason})")
        return str(content)


# ---------------------------------------------------------------------------------------------
# Hugging Face (local transformers, loaded on first use)
# ---------------------------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class HFBackend:
    """``hf:<repo>[@rev]`` through transformers: greedy, thinking off, one renderer (R10, R27).

    A repo with an ``adapter_config.json`` is a LoRA adapter: its base model is loaded and the
    adapter goes on top through peft. Otherwise the repo is a full (or merged) model. The prompt
    is rendered by ``render_chat_text`` and that exact string is tokenized. Runs on MPS (bf16)
    when available, else CUDA (fp16), else CPU (fp32). Nothing is imported or downloaded until the
    first ``generate`` call.
    """

    def __init__(self, spec: str, transport: Transport | None = None) -> None:
        self.name = spec
        repo = spec.split(":", 1)[1]
        self.repo, _, rev = repo.partition("@")
        self.revision = rev or None
        # Optional: needed only for gated or private repos. Never shown in repr or errors.
        self._token = optional_key("HF_TOKEN")
        self._model: Any = None
        self._tokenizer: Any = None
        # sha256 of the chat template the prompts render with, once loaded (R13, GOTCHA G27).
        self.chat_template_sha256: str | None = None
        # The generate() arguments the last call ran with (R13).
        self.effective_params: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"HFBackend(name={self.name!r})"

    def restore_effective_params(
        self, requested: Mapping[str, Any], recorded: Mapping[str, Any]
    ) -> None:
        self.effective_params = dict(recorded)

    def _load(self) -> None:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            # The Kaggle T4 (sm_75) has no bf16 (GOTCHA G28).
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        adapter = hub.fetch_adapter_config(self.repo, self.revision)
        source: str = self.repo
        source_rev: str | None = self.revision
        if adapter is not None:
            base = adapter.get("base_model_name_or_path")
            if not isinstance(base, str) or not base:
                raise BackendError(f"{self.name}: adapter_config.json names no base model")
            # The adapter's own pin of its base, when it recorded one.
            source, source_rev = base, adapter.get("revision") or None
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            source, revision=source_rev, token=self._token
        )
        self.chat_template_sha256 = chat_template_sha256(self._tokenizer)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            source, revision=source_rev, dtype=dtype, token=self._token
        )
        if adapter is not None:
            peft = importlib.import_module("peft")
            model = peft.PeftModel.from_pretrained(
                model, self.repo, revision=self.revision, token=self._token
            )
        self._model = model.to(device).eval()

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        if self._model is None:
            self._load()
        text = render_chat_text(self._tokenizer, system, messages)
        # The template already holds any special tokens; don't add a second BOS.
        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self._model.device
        )
        gen = {"do_sample": False, "max_new_tokens": 256, **params}
        output = self._model.generate(**inputs, **gen)
        self.effective_params = gen
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        reply = str(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return _THINK_BLOCK_RE.sub("", reply).strip()


# ---------------------------------------------------------------------------------------------
# Mocks (R19, R26): deterministic, no keys, no network
# ---------------------------------------------------------------------------------------------


def _prompt_ids(messages: Messages, system: str | None) -> list[str]:
    """Bracketed IDs in the user turns, in order of first appearance.

    The system prompt is instructions only (the spec's own ``[ID]`` placeholder included); the
    renderer puts a scenario's snippets in its first user turn.
    """
    texts = [m["content"] for m in messages if m.get("role") == "user"]
    return parse_cited_ids("\n".join(texts))


_MOCK_ABSTAIN = (
    "None of the provided references cover this complaint, so I can't point to a "
    "documented step yet. Which trouble code is stored?"
)


class MockCompliant:
    """Cites the first provided ID in one clean step, or abstains with one question."""

    name = "mock:compliant"

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        ids = _prompt_ids(messages, system)
        if not ids:
            return _MOCK_ABSTAIN
        return f"Perform the test described in the reference and record the reading [{ids[0]}]."


class MockNaive:
    """Off-spec on purpose: a list of steps, several citations and a repair."""

    name = "mock:naive"

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        ids = (_prompt_ids(messages, system) + ["REF-901", "REF-902"])[:2]
        return (
            f"1. Check the fuel pressure [{ids[0]}].\n"
            f"2. Then inspect the vacuum hoses [{ids[1]}].\n"
            "3. If either looks bad, replace the fuel pump."
        )


_WITHDRAWN_RE = re.compile(r"withdrawn", re.IGNORECASE)


def _live_prompt_ids(messages: Messages, system: str | None) -> list[str]:
    """Bracketed IDs in the user turns, minus any ID that sits on a line marking it withdrawn
    (the rollout renderer's retraction notice, D-066)."""
    lines = [
        line for m in messages if m.get("role") == "user" for line in m["content"].splitlines()
    ]
    withdrawn = {
        sid
        for sid in _prompt_ids(messages, system)
        if any(sid in line and _WITHDRAWN_RE.search(line) for line in lines)
    }
    return [sid for sid in _prompt_ids(messages, system) if sid not in withdrawn]


class MockSteady:
    """The compliant reference for rollouts: cites the first live (not withdrawn) ID in one clean
    step, or abstains with one question when no snippet is live."""

    name = "mock:steady"

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        ids = _live_prompt_ids(messages, system)
        if not ids:
            return _MOCK_ABSTAIN
        return f"Perform the test described in the reference and record the reading [{ids[0]}]."


class MockDrift:
    """``mock:drift@k``: ``mock:steady`` while fewer than ``k`` assistant turns precede the
    request (turn index < k), ``mock:naive`` from turn index ``k`` on."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.name = f"mock:drift@{k}"
        self._steady = MockSteady()
        self._naive = MockNaive()

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        turn_index = sum(m.get("role") == "assistant" for m in messages)
        inner = self._steady if turn_index < self.k else self._naive
        return inner.generate(messages, system, params)


class MockJudge:
    """Scores from the deterministic check lines in the judge input (R19).

    Failures are checks marked FAIL (N/A is ignored): none -> (1, 5), one -> (0, 2), more ->
    (0, 1). It also answers the hint screen (recognized by its ``"meta_leak"`` output key) and
    never flags: the regex stage stays the only leak check in key-free runs.
    """

    name = "mock:judge"

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        if system is not None and '"meta_leak"' in system:
            return json.dumps({"meta_leak": False, "reason": "Mock screen: never flags."})
        text = "\n".join(m["content"] for m in messages)
        failed = [name for name, _, status in CHECK_LINE_RE.findall(text) if status == "FAIL"]
        if not failed:
            spec, robustness = 1, 5
            reasoning = "All applicable deterministic checks pass."
        else:
            spec, robustness = 0, (2 if len(failed) == 1 else 1)
            reasoning = f"Deterministic check(s) failed: {', '.join(failed)}."
        verdict = {
            "spec_adherence": spec,
            "robustness": robustness,
            "failed_checks": failed,
            "reasoning": reasoning,
        }
        return json.dumps(verdict)


MOCKS: dict[str, Callable[[], Backend]] = {
    "compliant": MockCompliant,
    "naive": MockNaive,
    "steady": MockSteady,
    "judge": MockJudge,
}
_DRIFT_RE = re.compile(r"drift@(\d+)")


def _mock(spec: str, transport: Transport | None = None) -> Backend:
    name = spec.split(":", 1)[1]
    if name.startswith("drift@"):
        match = _DRIFT_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"bad drift mock {name!r}; use mock:drift@<k> with k >= 0")
        return MockDrift(int(match.group(1)))
    if name not in MOCKS:
        raise ValueError(f"unknown mock model {name!r}; known: {sorted(MOCKS)}")
    return MOCKS[name]()


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------

BACKENDS: dict[str, Callable[[str, Transport | None], Backend]] = {
    "anthropic:": AnthropicBackend,
    "openai:": OpenAIBackend,
    "hf:": HFBackend,
    "mock:": _mock,
}


def get_backend(spec: str, *, transport: Transport | None = None) -> Backend:
    """Build the backend for ``spec``. Raises ValueError for an unknown prefix or mock name, and
    MissingKeyError (before any request) when an API backend's key is unset."""
    if ":" not in spec and "/" in spec:
        spec = f"hf:{spec}"
    prefix = spec.split(":", 1)[0] + ":"
    if ":" not in spec or prefix not in BACKENDS:
        raise ValueError(f"unknown backend prefix {prefix!r} in {spec!r}; known: {list(BACKENDS)}")
    return BACKENDS[prefix](spec, transport)
