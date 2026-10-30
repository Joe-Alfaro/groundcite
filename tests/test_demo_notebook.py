"""The public base-vs-tuned inference demo notebook (R16, G32, D-118).

``train/demo_inference.ipynb`` runs from a public link with no secrets, logged out of the Hub, on
a free GPU (Kaggle or Colab T4):

- it is committed as valid nbformat-4 JSON with no outputs and no execution counts;
- its one ``parameters`` cell holds ``REPO_URL`` (default ``""``: no code-repo URL is ever
  tracked, D-083; the hosted public copy fills it in), ``REPO_COMMIT`` (a full SHA, the same pin
  as the Kaggle notebooks, so ``scripts/repin_notebooks.py`` re-pins all three), ``BASE_MODEL``
  (``Qwen/Qwen3-1.7B``) at ``BASE_REVISION`` (the ``train/config.toml`` pin), ``ADAPTER_REPO``
  (the public ``joe-alfaro/groundcite-qwen3-1.7b``) at ``ADAPTER_REVISION`` (a full Hub commit
  SHA, the one the README records), a free-text ``COMPLAINT`` and an optional ``SNIPPETS`` list of
  ``{"id", "title" (optional), "text"}`` objects (``[]`` means no snippets);
- with ``REPO_URL`` empty it stops before any shell command or download, naming ``REPO_URL``;
- otherwise it clones ``REPO_URL``, checks out ``REPO_COMMIT``, installs the clone with pip
  (never torch: the platform's CUDA torch stays) and imports ``groundcite`` from it in-process;
- it never reads a secret or a token: no Kaggle or Colab secrets, no ``HF_TOKEN``, no login, and
  no ``from_pretrained`` call passes a token;
- it loads the base at ``BASE_REVISION`` and the adapter with ``PeftModel.from_pretrained`` at
  ``ADAPTER_REVISION``, renders the complaint and snippets with the shared renderer
  (``render_messages(scenario, "zero_shot")`` and ``render_chat_text``, thinking off), decodes
  greedily with the ``[decoding.hf]`` cap, and generates one base reply (adapter disabled, or a
  separate unwrapped base) and one tuned reply;
- it leaves ``REPLIES = {"base": ..., "tuned": ...}`` and ``CHECK_RESULTS = {"base": [...],
  "tuned": [...]}`` (``groundcite.checks.run_checks`` on the scenario built from ``COMPLAINT`` and
  ``SNIPPETS``) in its globals and prints both replies side by side with every check result.

The notebook is executed in-process by ``tests/notebook_sandbox.py`` with fake ``transformers``
and ``peft`` modules: nothing is downloaded and nothing runs on a GPU.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest
from notebook_sandbox import Sandbox, parameters_cell, run_notebook, source

from groundcite.checks import CHECKS, run_checks
from groundcite.evalset import parse_row
from groundcite.prompts import render_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "train" / "demo_inference.ipynb"
EVAL_NOTEBOOK = REPO_ROOT / "train" / "kaggle_eval.ipynb"
TRAIN_CONFIG = REPO_ROOT / "train" / "config.toml"
MODELS_CONFIG = REPO_ROOT / "config" / "models.toml"
README = REPO_ROOT / "README.md"
REPIN = REPO_ROOT / "scripts" / "repin_notebooks.py"
ADAPTER_REPO = "joe-alfaro/groundcite-qwen3-1.7b"
FAKE_REPO_URL = "https://example.invalid/someone/project.git"
SHA = re.compile(r"[0-9a-f]{40}")
BASE_REPLY = "It's probably the fuel pump, replace it."
TUNED_REPLY = "Check the fuel pressure at the rail with a gauge, key on engine off [TSB-12]."
SNIPPETS = [
    {"id": "TSB-12", "title": "Fuel rail pressure", "text": "Measure rail pressure key on."},
    {"id": "SM 3.1", "text": "Check the coil primary resistance."},
]
COMPLAINT = "2014 sedan cranks but won't start after sitting overnight."
SECRET_WORDS = re.compile(
    r"kaggle_secrets|UserSecretsClient|google\.colab|userdata|getpass|HF_TOKEN|"
    r"notebook_login|\blogin\(|auth login|huggingface-cli login",
)


# ---------------------------------------------------------------------------------------------
# Fake transformers and peft
# ---------------------------------------------------------------------------------------------


class Ids:
    """A generated or encoded tensor: slicing and moving keep its tag (which model made it)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.shape = (1, 7)

    def __getitem__(self, key: Any) -> Ids:
        return self

    def to(self, *args: Any, **kwargs: Any) -> Ids:
        return self

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[Ids]:
        return iter([self])


class Inputs(dict):  # type: ignore[type-arg]
    def __init__(self) -> None:
        super().__init__(input_ids=Ids("prompt"), attention_mask=Ids("prompt"))
        self.input_ids = self["input_ids"]
        self.attention_mask = self["attention_mask"]

    def to(self, *args: Any, **kwargs: Any) -> Inputs:
        return self


class Hub:
    """Records every load and generate call across the fake libraries."""

    def __init__(self) -> None:
        self.loads: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.generates: list[dict[str, Any]] = []
        self.chats: list[tuple[list[dict[str, str]], dict[str, Any]]] = []


class FakeTokenizer:
    chat_template = "{{ fake }}"
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<eos>"

    def __init__(self, hub: Hub) -> None:
        self.hub = hub

    def apply_chat_template(self, chat: list[dict[str, str]], **kwargs: Any) -> Any:
        self.hub.chats.append(([dict(m) for m in chat], dict(kwargs)))
        if kwargs.get("tokenize", True) is False:
            return "<rendered prompt>"
        return Inputs() if kwargs.get("return_dict") else Ids("prompt")

    def __call__(self, *args: Any, **kwargs: Any) -> Inputs:
        return Inputs()

    def decode(self, ids: Any, *args: Any, **kwargs: Any) -> str:
        return {"base": BASE_REPLY, "tuned": TUNED_REPLY}[ids.tag]

    def batch_decode(self, ids: Any, *args: Any, **kwargs: Any) -> list[str]:
        return [self.decode(ids)]


class FakeModel:
    """A base model; wrapping it in a PeftModel injects the adapter into it in place, as real
    peft does, so generating with it afterwards gives the tuned reply."""

    device = "cuda"

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.adapter = False
        self.adapter_disabled = False
        self.generation_config = mock.MagicMock()
        self.config = mock.MagicMock()

    def generate(self, *args: Any, **kwargs: Any) -> Ids:
        self.hub.generates.append(dict(kwargs))
        return Ids("tuned" if self.adapter and not self.adapter_disabled else "base")

    def eval(self) -> FakeModel:
        return self

    def to(self, *args: Any, **kwargs: Any) -> FakeModel:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generate replies with .generate()")


class FakePeftModel:
    def __init__(self, base: FakeModel) -> None:
        self.base_model = base
        base.adapter = True
        self.device = base.device
        self.generation_config = base.generation_config
        self.config = base.config

    def generate(self, *args: Any, **kwargs: Any) -> Ids:
        return self.base_model.generate(*args, **kwargs)

    def disable_adapter(self) -> Any:
        base = self.base_model

        class _Off:
            def __enter__(self) -> None:
                base.adapter_disabled = True

            def __exit__(self, *exc: Any) -> None:
                base.adapter_disabled = False

        return _Off()

    def eval(self) -> FakePeftModel:
        return self

    def to(self, *args: Any, **kwargs: Any) -> FakePeftModel:
        return self


def _fake_libraries(hub: Hub) -> dict[str, ModuleType]:
    def loader(kind: str, make: Any) -> Any:
        class Auto:
            @staticmethod
            def from_pretrained(*args: Any, **kwargs: Any) -> Any:
                hub.loads.append((kind, args, dict(kwargs)))
                return make(*args, **kwargs)

        return Auto

    def peft_load(model: FakeModel, *args: Any, **kwargs: Any) -> FakePeftModel:
        hub.loads.append(("peft", args, dict(kwargs)))
        return FakePeftModel(model)

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = loader("tokenizer", lambda *a, **k: FakeTokenizer(hub))  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = loader("model", lambda *a, **k: FakeModel(hub))  # type: ignore[attr-defined]
    transformers.__version__ = "0.0.0"  # type: ignore[attr-defined]
    transformers.set_seed = lambda *a, **k: None  # type: ignore[attr-defined]
    peft = ModuleType("peft")
    peft.PeftModel = type(  # type: ignore[attr-defined]
        "PeftModel", (), {"from_pretrained": staticmethod(peft_load)}
    )
    peft.__version__ = "0.0.0"  # type: ignore[attr-defined]
    return {"transformers": transformers, "peft": peft}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _load() -> dict[str, Any]:
    assert DEMO.is_file(), f"{DEMO.relative_to(REPO_ROOT)} is missing"
    return dict(json.loads(DEMO.read_text(encoding="utf-8")))


@pytest.fixture
def notebook() -> dict[str, Any]:
    return _load()


def _code(notebook: dict[str, Any]) -> str:
    return "\n".join(source(c) for c in notebook["cells"] if c["cell_type"] == "code")


def _params(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(source(parameters_cell(notebook)), namespace)
    return namespace


def _with_inputs(notebook: dict[str, Any], **values: Any) -> dict[str, Any]:
    """A copy whose parameters cell ends by reassigning ``values`` (any Python literal)."""
    edited = copy.deepcopy(notebook)
    cell = parameters_cell(edited)
    extra = "".join(f"\n{name} = {value!r}" for name, value in values.items())
    cell["source"] = source(cell) + extra  # type: ignore[index]
    return edited


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, no_network: None) -> None:
    """Logged out: no token anywhere, sys.path restored after the notebook edits it."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))


def _run(notebook: dict[str, Any], **params: str) -> tuple[Sandbox, Hub]:
    hub = Hub()
    with mock.patch.dict(sys.modules, _fake_libraries(hub)):
        run = run_notebook(
            notebook,
            params={"REPO_URL": FAKE_REPO_URL, **params},
            stop_at=lambda cmd: False,
            secrets={"HF_TOKEN": None, "REPO_URL": None},
        )
    return run, hub


def _full_run(notebook: dict[str, Any]) -> tuple[Sandbox, Hub]:
    run, hub = _run(notebook)
    assert run.error is None, f"the demo notebook failed: {run.error!r}"
    return run, hub


# ---------------------------------------------------------------------------------------------
# File shape and parameters
# ---------------------------------------------------------------------------------------------


def test_the_notebook_is_valid_json_with_no_outputs(notebook: dict[str, Any]) -> None:
    assert notebook.get("nbformat") == 4
    assert notebook["cells"], "the notebook has cells"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], "commit the notebook with no outputs"
            assert cell.get("execution_count") is None, "clear the execution counts"
    assert notebook["cells"][0]["cell_type"] == "markdown", "open with what the demo does"


def test_the_parameters_pin_the_base_the_adapter_and_the_code(notebook: dict[str, Any]) -> None:
    params = _params(notebook)
    base = tomllib.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))["model"]
    assert params["BASE_MODEL"] == "Qwen/Qwen3-1.7B" == base["base"]
    assert params["BASE_REVISION"] == base["revision"], "the train/config.toml base pin"
    assert params["ADAPTER_REPO"] == ADAPTER_REPO
    assert SHA.fullmatch(params["ADAPTER_REVISION"]), "pin the adapter to a full Hub commit SHA"
    assert params["ADAPTER_REVISION"] in README.read_text(encoding="utf-8"), (
        "the README records the adapter commit the demo loads"
    )
    assert SHA.fullmatch(params["REPO_COMMIT"])
    eval_params = _params(json.loads(EVAL_NOTEBOOK.read_text(encoding="utf-8")))
    assert params["REPO_COMMIT"] == eval_params["REPO_COMMIT"], "one pin for every notebook"


def test_the_committed_repo_url_is_empty(notebook: dict[str, Any]) -> None:
    """D-083: no tracked file holds the code repo's URL; the hosted public copy fills it in."""
    params = _params(notebook)
    assert params["REPO_URL"] == ""
    assert "://" not in source(parameters_cell(notebook))


def test_the_default_inputs_are_a_complaint_and_valid_snippets(notebook: dict[str, Any]) -> None:
    params = _params(notebook)
    assert isinstance(params["COMPLAINT"], str) and params["COMPLAINT"].strip()
    assert isinstance(params["SNIPPETS"], list) and params["SNIPPETS"], "ship an example"
    parse_row(
        {
            "id": "demo",
            "messages": [{"role": "user", "content": params["COMPLAINT"]}],
            "snippets": params["SNIPPETS"],
        }
    )


def test_repin_covers_the_demo_notebook() -> None:
    text = REPIN.read_text(encoding="utf-8")
    assert "demo_inference.ipynb" in text, "scripts/repin_notebooks.py re-pins the demo too"


# ---------------------------------------------------------------------------------------------
# No secrets, logged out
# ---------------------------------------------------------------------------------------------


def test_the_notebook_reads_no_secret_and_no_token(notebook: dict[str, Any]) -> None:
    code = _code(notebook)
    hits = sorted({m.group(0) for m in SECRET_WORDS.finditer(code)})
    assert not hits, f"the public demo runs logged out with no secrets, found {hits}"


def test_no_model_load_passes_a_token(notebook: dict[str, Any], clean_env: None) -> None:
    _, hub = _full_run(notebook)
    assert hub.loads, "the notebook loads models"
    for kind, _, kwargs in hub.loads:
        assert not kwargs.get("token"), f"{kind} load passes a token"
        assert not kwargs.get("use_auth_token"), f"{kind} load passes a token"


def test_no_shell_command_logs_in(notebook: dict[str, Any], clean_env: None) -> None:
    run, _ = _full_run(notebook)
    assert not [c for c in run.commands if re.search(r"\blogin\b", c)]


# ---------------------------------------------------------------------------------------------
# Install from the pinned repo commit
# ---------------------------------------------------------------------------------------------


def test_an_empty_repo_url_stops_before_any_command(
    notebook: dict[str, Any], clean_env: None
) -> None:
    run, hub = _run(notebook, REPO_URL="")
    assert run.error is not None, "stop when REPO_URL is not filled in"
    assert "REPO_URL" in str(run.error), "name the parameter to fill in"
    assert run.commands == [], "no clone, no install"
    assert hub.loads == [], "no download"


def test_the_repo_is_cloned_at_the_pinned_commit_and_installed(
    notebook: dict[str, Any], clean_env: None
) -> None:
    run, _ = _full_run(notebook)
    commit = _params(notebook)["REPO_COMMIT"]
    clones = [i for i, c in enumerate(run.commands) if "git clone" in c and FAKE_REPO_URL in c]
    checkouts = [i for i, c in enumerate(run.commands) if "checkout" in c and commit in c]
    installs = [i for i, c in enumerate(run.commands) if re.search(r"\bpip\b.*\binstall\b", c)]
    assert clones, "clone REPO_URL"
    assert checkouts and checkouts[0] > clones[0], "check out REPO_COMMIT after the clone"
    assert installs and max(installs) > checkouts[0], "install the checked-out code"


def test_no_pip_command_touches_torch(notebook: dict[str, Any], clean_env: None) -> None:
    run, _ = _full_run(notebook)
    for command in run.commands:
        if re.search(r"\bpip\b", command):
            assert not re.search(r"(?<![\w-])(torch|triton|nvidia-[\w-]+)(?![\w-])", command), (
                f"keep the platform's CUDA torch: {command}"
            )


# ---------------------------------------------------------------------------------------------
# Base and tuned replies with their deterministic checks
# ---------------------------------------------------------------------------------------------


def test_the_base_and_the_adapter_load_at_their_pins(
    notebook: dict[str, Any], clean_env: None
) -> None:
    params = _params(notebook)
    _, hub = _full_run(notebook)
    by_kind: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {}
    for kind, args, kwargs in hub.loads:
        by_kind.setdefault(kind, []).append((args, kwargs))

    for kind in ("tokenizer", "model"):
        assert by_kind.get(kind), f"load the base {kind}"
        for args, kwargs in by_kind[kind]:
            name = args[0] if args else kwargs.get("pretrained_model_name_or_path")
            assert name == params["BASE_MODEL"], f"the {kind} is the base model"
            assert kwargs.get("revision") == params["BASE_REVISION"], f"pin the base {kind}"

    assert len(by_kind.get("peft", [])) == 1, "load the adapter once with PeftModel"
    args, kwargs = by_kind["peft"][0]
    name = args[0] if args else kwargs.get("model_id")
    assert name == params["ADAPTER_REPO"]
    assert kwargs.get("revision") == params["ADAPTER_REVISION"], "pin the adapter commit"


def test_replies_decode_greedily_within_the_hf_cap(
    notebook: dict[str, Any], clean_env: None
) -> None:
    decoding = tomllib.loads(MODELS_CONFIG.read_text(encoding="utf-8"))["decoding"]["hf"]
    _, hub = _full_run(notebook)
    assert len(hub.generates) >= 2, "generate a base and a tuned reply"
    for kwargs in hub.generates:
        assert kwargs.get("do_sample") is False, "greedy decoding"
        assert kwargs.get("max_new_tokens") == decoding["max_new_tokens"]


def test_the_prompt_is_the_shared_zero_shot_render_with_thinking_off(
    notebook: dict[str, Any], clean_env: None
) -> None:
    edited = _with_inputs(notebook, COMPLAINT=COMPLAINT, SNIPPETS=SNIPPETS)
    run, hub = _run(edited)
    assert run.error is None, f"the demo notebook failed: {run.error!r}"
    scenario = parse_row(
        {"id": "demo", "messages": [{"role": "user", "content": COMPLAINT}], "snippets": SNIPPETS}
    )
    system, messages = render_messages(scenario, "zero_shot")
    expected = [{"role": "system", "content": system}, *messages]
    assert hub.chats, "render with the tokenizer's chat template"
    for chat, kwargs in hub.chats:
        assert chat == expected, "use groundcite.prompts.render_messages(…, 'zero_shot')"
        assert kwargs.get("enable_thinking") is False, "thinking off (R27)"


def test_the_base_and_tuned_replies_are_kept_apart(
    notebook: dict[str, Any], clean_env: None
) -> None:
    """The base reply comes from the base weights: with the adapter disabled, or from a base the
    adapter was never injected into."""
    run, _ = _full_run(notebook)
    assert run.namespace.get("REPLIES") == {"base": BASE_REPLY, "tuned": TUNED_REPLY}


@pytest.mark.parametrize("snippets", [SNIPPETS, []], ids=["snippets", "no-snippets"])
def test_check_results_are_run_checks_on_the_demo_scenario(
    notebook: dict[str, Any], clean_env: None, snippets: list[dict[str, str]]
) -> None:
    edited = _with_inputs(notebook, COMPLAINT=COMPLAINT, SNIPPETS=snippets)
    run, hub = _run(edited)
    assert run.error is None, f"the demo notebook failed: {run.error!r}"
    row: dict[str, Any] = {"id": "demo", "messages": [{"role": "user", "content": COMPLAINT}]}
    if snippets:
        row["snippets"] = snippets
    scenario = parse_row(row)
    results = run.namespace.get("CHECK_RESULTS")
    assert results == {
        "base": run_checks(BASE_REPLY, scenario),
        "tuned": run_checks(TUNED_REPLY, scenario),
    }
    if not snippets:
        user = [m for chat, _ in hub.chats for m in chat if m["role"] == "user"]
        assert user and all(m["content"] == COMPLAINT for m in user), "no snippet block"


def test_both_replies_and_every_check_are_printed_side_by_side(
    notebook: dict[str, Any], clean_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _full_run(notebook)
    out = capsys.readouterr().out
    assert BASE_REPLY in out and TUNED_REPLY in out
    assert re.search(r"\bbase\b", out, re.IGNORECASE) and re.search(r"\btuned\b", out, re.I)
    for name in CHECKS:
        assert out.count(name) >= 1, f"show the {name} result"
    assert re.search(r"\bfail", out, re.IGNORECASE), "show failing checks as failures"
    assert re.search(r"\bpass", out, re.IGNORECASE), "show passing checks as passes"


def test_the_notebook_uses_no_display_only_helpers(notebook: dict[str, Any]) -> None:
    """The side-by-side view is printed, so it reads the same in Kaggle, Colab and a log."""
    assert not re.search(r"(?m)^\s*display\(", _code(notebook))
