"""Eval refuses an adapter whose train-time chat template differs from the eval-time one (M2,
GOTCHA G27, D-079).

train_qlora.py records ``chat_template_sha256`` in ``train_log.json`` and pushes that file with
the adapter. At load time, HFBackend compares it with the hash of the tokenizer it just loaded:

- ``groundcite.hub.fetch_train_log(repo, revision=None)`` returns the repo's ``train_log.json``
  at that revision (``GET {HUB_URL}/{repo}/resolve/{revision or main}/train_log.json``), or None
  on 404;
- for an adapter repo, HFBackend fetches it at the adapter's own revision. When it records a
  ``chat_template_sha256`` that differs from the loaded tokenizer's, loading stops with a
  ``BackendError`` naming the chat template and both hashes (at least their first 12 hex
  characters), before any model weights load. A missing log, or a log without the hash (an
  older adapter), loads as before;
- in eval.py the mismatch stops the run (non-zero exit, the mismatch on stderr) instead of
  becoming a per-row generation error, and ``--generate-only`` writes no generations.jsonl.

Sockets are blocked and transformers is faked; nothing is downloaded or run.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from groundcite import backends, hub
from groundcite.backends import BackendError

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION = REPO_ROOT / "data" / "scenarios" / "ablation.jsonl"
BASE_REPO = "Qwen/Qwen3-1.7B"
TUNED_REPO = "org/groundcite-tuned"
TUNED = f"hf:{TUNED_REPO}@rev1"
TEMPLATE = "{{ messages }}"
EVAL_SHA = hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()
TRAIN_SHA = "ab" * 32


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


# ---------------------------------------------------------------------------------------------
# hub.fetch_train_log
# ---------------------------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def test_fetch_train_log_reads_train_log_json_at_the_revision(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(request: Any, timeout: float | None = None) -> _Response:
        urls.append(request.full_url)
        return _Response(json.dumps({"chat_template_sha256": TRAIN_SHA}).encode())

    monkeypatch.setattr(hub.urllib.request, "urlopen", urlopen)
    fetch = getattr(hub, "fetch_train_log", None)
    assert fetch is not None, "groundcite.hub.fetch_train_log is missing"
    assert fetch(TUNED_REPO, "rev1") == {"chat_template_sha256": TRAIN_SHA}
    assert urls == [f"{hub.HUB_URL}/{TUNED_REPO}/resolve/rev1/train_log.json"]
    fetch(TUNED_REPO)
    assert urls[-1] == f"{hub.HUB_URL}/{TUNED_REPO}/resolve/main/train_log.json"


def test_fetch_train_log_is_none_without_a_log(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(hub.urllib.request, "urlopen", urlopen)
    fetch = getattr(hub, "fetch_train_log", None)
    assert fetch is not None, "groundcite.hub.fetch_train_log is missing"
    assert fetch(TUNED_REPO, "rev1") is None


# ---------------------------------------------------------------------------------------------
# HFBackend: compare at load time
# ---------------------------------------------------------------------------------------------


class _Model:
    device = "cpu"

    def to(self, *args: Any, **kwargs: Any) -> _Model:
        return self

    def eval(self) -> _Model:
        return self

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        return input_ids


class _Tok:
    chat_template = TEMPLATE

    def apply_chat_template(self, chat: Any, **kwargs: Any) -> str:
        return "prompt"

    def __call__(self, text: str, **kwargs: Any) -> Any:
        import torch

        class _Enc(dict):  # type: ignore[type-arg]
            def to(self, device: Any) -> Any:
                return self

        return _Enc(input_ids=torch.tensor([[1, 2]]))

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        return "Check the hose [A-1]."


def _fake_stack(
    monkeypatch: pytest.MonkeyPatch, train_log: dict[str, Any] | None
) -> dict[str, list[Any]]:
    """Fake the Hub lookups and the transformers/peft loads; return what each was asked for."""
    import peft
    import torch
    import transformers

    seen: dict[str, list[Any]] = {"train_log": [], "model": [], "peft": []}

    def adapter_config(repo: str, revision: str | None = None) -> dict[str, Any] | None:
        if repo == TUNED_REPO:
            return {"base_model_name_or_path": BASE_REPO, "peft_type": "LORA"}
        return None

    def fetch_train_log(repo: str, revision: str | None = None) -> dict[str, Any] | None:
        seen["train_log"].append((repo, revision))
        return dict(train_log) if train_log is not None and repo == TUNED_REPO else None

    def load_model(*args: Any, **kwargs: Any) -> _Model:
        seen["model"].append(args)
        return _Model()

    def load_peft(model: Any, *args: Any, **kwargs: Any) -> Any:
        seen["peft"].append(args)
        return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(hub, "fetch_adapter_config", adapter_config)
    monkeypatch.setattr(hub, "fetch_train_log", fetch_train_log, raising=False)
    monkeypatch.setattr(
        hub,
        "resolve_revision",
        lambda repo, revision=None: hashlib.sha1(f"{repo}@{revision}".encode()).hexdigest(),
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", load_peft)
    return seen


def _generate() -> str:
    backend = backends.HFBackend(TUNED)
    return backend.generate([{"role": "user", "content": "It stalls."}], None, {})


def test_a_template_mismatch_refuses_to_load_the_adapter(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_stack(monkeypatch, {"chat_template_sha256": TRAIN_SHA})
    with pytest.raises(BackendError) as caught:
        _generate()
    message = str(caught.value)
    assert "chat template" in message.lower()
    assert TRAIN_SHA[:12] in message and EVAL_SHA[:12] in message, message
    assert seen["model"] == [] and seen["peft"] == [], "refuse before loading any weights"


def test_the_train_log_is_read_at_the_adapter_revision(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_stack(monkeypatch, {"chat_template_sha256": EVAL_SHA})
    assert _generate() == "Check the hose [A-1]."
    assert seen["train_log"] == [(TUNED_REPO, "rev1")]


@pytest.mark.parametrize(
    "train_log",
    [None, {"seed": 0}, {"chat_template_sha256": None}],
    ids=["no-log", "no-key", "null"],
)
def test_an_adapter_without_a_recorded_template_still_loads(
    env: Path, monkeypatch: pytest.MonkeyPatch, train_log: dict[str, Any] | None
) -> None:
    seen = _fake_stack(monkeypatch, train_log)
    assert _generate() == "Check the hose [A-1]."
    assert seen["model"] and seen["peft"]


# ---------------------------------------------------------------------------------------------
# eval.py: the mismatch stops the run
# ---------------------------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    from groundcite.cli import main

    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def _generate_only(out: Path) -> int:
    argv = ["--model", TUNED, "--no-base", "--eval-set", str(ABLATION), "--limit", "2"]
    return _run_cli([*argv, "--allow-dirty", "--generate-only", "--out", str(out)])


def test_eval_stops_on_a_template_mismatch(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_stack(monkeypatch, {"chat_template_sha256": TRAIN_SHA})
    out = tmp_path / "gen"
    code = _generate_only(out)
    err = capsys.readouterr().err
    assert code != 0, "a template mismatch must stop the run, not become per-row errors"
    assert "chat template" in err.lower(), err
    assert not list(out.rglob("generations.jsonl")), "no generations from a mismatched template"


def test_eval_generates_when_the_templates_match(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_stack(monkeypatch, {"chat_template_sha256": EVAL_SHA})
    out = tmp_path / "gen"
    assert _generate_only(out) == 0
    rows = [
        json.loads(line)
        for path in out.rglob("generations.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows and all(r["generation_error"] is None for r in rows)
