"""Shared fixtures: no real network and no real keys in tests that opt in."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN")


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args: Any, **kwargs: Any) -> Any:
    raise NetworkBlocked("network access is blocked in tests")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any attempt to resolve or connect raises NetworkBlocked."""
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Unset every key and point GROUNDCITE_ENV_FILE at an empty temp file.

    This keeps the real `.env` of the main checkout out of every test. Returns the temp env-file
    path so a test can write fake keys into it.
    """
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    env_file = tmp_path / "test.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("GROUNDCITE_ENV_FILE", str(env_file))
    return env_file


# Release-gate tests (tests/test_release_gate.py) hold only after the notebooks are re-pinned to
# a pushed commit, so a plain `pytest` (and scripts/check.sh) collects them but deselects them.
# scripts/release_check.sh runs them with --release-gate.
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--release-gate",
        action="store_true",
        default=False,
        help="run only the release_gate tests (deselected by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "release_gate: holds only once the notebooks pin a pushed, current commit"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gate = bool(config.getoption("--release-gate"))
    keep, drop = [], []
    for item in items:
        (keep if (item.get_closest_marker("release_gate") is not None) == gate else drop).append(
            item
        )
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
