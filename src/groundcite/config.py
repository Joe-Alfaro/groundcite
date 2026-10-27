"""Pinned model ids and decoding parameters from ``config/models.toml`` (D-019, R13, R28)."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

MODELS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.toml"


def load_models_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parse the models config (default: ``config/models.toml`` in the repo)."""
    with Path(path if path is not None else MODELS_CONFIG_PATH).open("rb") as fh:
        return tomllib.load(fh)


def decoding_params(spec: str, path: str | Path | None = None) -> dict[str, Any]:
    """The fixed decoding parameters for a backend spec's provider; empty when none are pinned.

    ``spec`` is a backend spec such as ``openai:gpt-5.5-2026-04-23``. Mock and unknown providers
    get ``{}``. The result is a copy, so callers may change it freely.
    """
    provider = spec.split(":", 1)[0] if ":" in spec else "hf"
    table = load_models_config(path).get("decoding", {}).get(provider, {})
    return copy.deepcopy(dict(table))
