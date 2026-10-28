"""Pinned model ids and decoding parameters from ``config/models.toml`` (D-019, R13, R28)."""

from __future__ import annotations

import copy
import tomllib
from collections.abc import Iterable, Mapping
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


def price(spec: str, path: str | Path | None = None) -> dict[str, float]:
    """The ``[prices]`` entry for ``spec``: ``{input_per_mtok, output_per_mtok}`` in USD per
    million tokens. Raises KeyError naming the spec when none is configured."""
    table = load_models_config(path).get("prices", {})
    if spec not in table:
        raise KeyError(f"no [prices] entry for {spec} in the models config")
    return dict(table[spec])


def cost_usd(
    spec: str, usages: Iterable[Mapping[str, Any]], path: str | Path | None = None
) -> float:
    """Dollars for recorded per-call usage: reasoning tokens bill at the output price."""
    rates = price(spec, path)
    total = 0.0
    for usage in usages:
        output = (usage.get("output_tokens") or 0) + (usage.get("reasoning_tokens") or 0)
        total += (usage.get("input_tokens") or 0) * rates["input_per_mtok"]
        total += output * rates["output_per_mtok"]
    return total / 1e6
