"""Test helper: point ``groundcite.config`` at an edited copy of ``config/models.toml``."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from groundcite import config


def write_toml(path: Path, data: Mapping[str, Any]) -> None:
    """A tiny TOML writer for the config shapes used here (tables, scalars, scalar lists and
    inline tables)."""

    def value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int | float):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        if isinstance(v, Mapping):
            return "{ " + ", ".join(f"{json.dumps(k)} = {value(x)}" for k, x in v.items()) + " }"
        raise TypeError(type(v))

    lines: list[str] = []

    def table(name: str, t: Mapping[str, Any], inline: bool) -> None:
        scalars = {k: v for k, v in t.items() if inline or not isinstance(v, Mapping)}
        nested = {k: v for k, v in t.items() if k not in scalars}
        if name:
            lines.append(f"[{name}]")
        lines.extend(f"{json.dumps(k)} = {value(v)}" for k, v in scalars.items())
        for k, v in nested.items():
            # Price entries stay inline tables under [prices].
            table(f"{name}.{json.dumps(k)}" if name else json.dumps(k), v, k == "prices")

    table("", data, False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patched_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> Path:
    """A copy of config/models.toml with ``edit`` applied, made the default config."""
    cfg = config.load_models_config()
    edit(cfg)
    path = tmp_path / "models.toml"
    write_toml(path, cfg)
    assert config.load_models_config(path) == cfg
    monkeypatch.setattr(config, "MODELS_CONFIG_PATH", path)
    return path
