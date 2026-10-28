"""Hugging Face Hub lookups over plain HTTPS (R10, R13, GOTCHA G16).

Three lookups: a repo's ``adapter_config.json`` (to derive the base model), an adapter's
``train_log.json`` (its train-time chat-template hash) and the commit SHA a revision resolves
to. ``HF_TOKEN`` is sent when set and never logged.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from groundcite.keys import optional_key, redact

HUB_URL = "https://huggingface.co"
TIMEOUT_S = 30


class HubError(RuntimeError):
    """A Hub lookup failed. The message never contains the token."""


def _get(url: str) -> bytes | None:
    """The response body, or None on 404."""
    token = optional_key("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise HubError(redact(f"GET {url} failed: HTTP {exc.code}", [token or ""])) from None
    except OSError as exc:
        raise HubError(redact(f"GET {url} failed: {exc}", [token or ""])) from None


def _quote(repo: str) -> str:
    return urllib.parse.quote(repo, safe="/")


def fetch_adapter_config(repo: str, revision: str | None = None) -> dict[str, Any] | None:
    """The repo's ``adapter_config.json``, or None when it has none (a full or merged model)."""
    rev = urllib.parse.quote(revision or "main", safe="")
    body = _get(f"{HUB_URL}/{_quote(repo)}/resolve/{rev}/adapter_config.json")
    if body is None:
        return None
    config = json.loads(body)
    if not isinstance(config, dict):
        raise HubError(f"{repo}: adapter_config.json is not a JSON object")
    return config


def fetch_train_log(repo: str, revision: str | None = None) -> dict[str, Any] | None:
    """The ``train_log.json`` train/train_qlora.py pushed with an adapter, or None without one."""
    rev = urllib.parse.quote(revision or "main", safe="")
    body = _get(f"{HUB_URL}/{_quote(repo)}/resolve/{rev}/train_log.json")
    if body is None:
        return None
    log = json.loads(body)
    if not isinstance(log, dict):
        raise HubError(f"{repo}: train_log.json is not a JSON object")
    return log


def resolve_revision(repo: str, revision: str | None = None) -> str:
    """The commit SHA ``revision`` (default: the main branch) points to."""
    rev = urllib.parse.quote(revision or "main", safe="")
    body = _get(f"{HUB_URL}/api/models/{_quote(repo)}/revision/{rev}")
    if body is None:
        raise HubError(f"{repo}@{revision or 'main'} was not found on the Hub")
    sha = json.loads(body).get("sha")
    if not isinstance(sha, str) or not sha:
        raise HubError(f"{repo}@{revision or 'main'}: the Hub returned no commit sha")
    return sha
