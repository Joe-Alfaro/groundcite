"""API keys (R25). Real environment variables win over the env file; values are never logged.

The env file defaults to ``<repo>/.env`` and can be moved with ``GROUNDCITE_ENV_FILE``. It is read
with python-dotenv but never exported into ``os.environ``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from dotenv import dotenv_values

ENV_FILE_VAR = "GROUNDCITE_ENV_FILE"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN")
REDACTED = "[REDACTED]"


class MissingKeyError(RuntimeError):
    """A required key is unset or empty. The CLI exits with ``exit_code``."""

    exit_code = 2

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(
            f"{env_var} is not set. Export it, or add it to the env file "
            f"(default {DEFAULT_ENV_FILE}, override with {ENV_FILE_VAR})."
        )


def env_file_path() -> Path:
    override = os.environ.get(ENV_FILE_VAR)
    return Path(override) if override else DEFAULT_ENV_FILE


def _from_env_file(name: str) -> str | None:
    path = env_file_path()
    if not path.is_file():
        return None
    return dotenv_values(path).get(name)


def get_key(name: str) -> str:
    """Return the key ``name``: the real env var if non-empty, else the env file's value."""
    value = os.environ.get(name) or _from_env_file(name)
    if not value:
        raise MissingKeyError(name)
    return value


def optional_key(name: str) -> str | None:
    """Like ``get_key``, but None when unset (for keys such as ``HF_TOKEN`` that are optional)."""
    return os.environ.get(name) or _from_env_file(name) or None


def redact(text: str, secrets: Iterable[str]) -> str:
    """Replace every secret value in ``text`` with a placeholder."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text
