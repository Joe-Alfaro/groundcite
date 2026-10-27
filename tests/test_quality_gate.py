"""Tests for scripts/check.sh, the local quality gate."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"

# Each step and a pattern that identifies it in the script, in required order.
STEPS: list[tuple[str, str]] = [
    ("format", "ruff format --check"),
    ("lint", "ruff check"),
    ("types", "mypy src"),
    ("tests", "pytest"),
    ("scrub", "scrub_check"),
]


def _command_lines() -> list[str]:
    assert CHECK_SH.is_file(), f"{CHECK_SH} is missing"
    lines = []
    for raw in CHECK_SH.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def test_check_sh_exists_and_is_executable() -> None:
    assert CHECK_SH.is_file(), f"{CHECK_SH} is missing"
    assert CHECK_SH.stat().st_mode & stat.S_IXUSR


def test_check_sh_runs_all_steps_in_order() -> None:
    lines = _command_lines()
    positions = []
    for name, pattern in STEPS:
        hits = [i for i, line in enumerate(lines) if pattern in line]
        assert hits, f"step {name!r} ({pattern!r}) not found in check.sh"
        positions.append(hits[0])
    assert positions == sorted(positions), "steps are not in the required order"


def _stub_bin(tmp_path: Path, failing: str | None) -> Path:
    """Create stub ruff/mypy/pytest/python/uv executables that log their invocation.

    The stub named ``failing`` exits 1 (``python`` covers ``python3`` too); the rest
    exit 0. This assumes check.sh invokes the tools by name via PATH, directly or
    through ``uv run``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    for name in ("ruff", "mypy", "pytest", "python", "python3"):
        code = 1 if failing and name.startswith(failing) else 0
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit {code}\n')
        script.chmod(0o755)
    # uv stub: drop "run" and any flags, then exec the remaining command via PATH.
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "run" ] && shift\n'
        'while [ "${1#-}" != "$1" ]; do shift; done\n'
        'exec "$@"\n'
    )
    uv.chmod(0o755)
    return bin_dir


def _run_check(tmp_path: Path, failing: str | None) -> tuple[int, str]:
    bin_dir = _stub_bin(tmp_path, failing)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin"
    env["SCRUB_TERMS_PATH"] = str(tmp_path / "terms.txt")
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["bash", str(CHECK_SH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )
    log = tmp_path / "calls.log"
    return result.returncode, log.read_text() if log.exists() else ""


def test_check_sh_passes_when_all_steps_pass(tmp_path: Path) -> None:
    assert CHECK_SH.is_file(), f"{CHECK_SH} is missing"
    code, calls = _run_check(tmp_path, failing=None)
    assert code == 0, calls
    for tool in ("ruff", "mypy", "pytest"):
        assert tool in calls


@pytest.mark.parametrize("failing", ["ruff", "mypy", "pytest", "python"])
def test_check_sh_fails_when_a_step_fails(tmp_path: Path, failing: str) -> None:
    assert CHECK_SH.is_file(), f"{CHECK_SH} is missing"
    code, calls = _run_check(tmp_path, failing=failing)
    assert code != 0, f"check.sh exited 0 although {failing} failed; calls:\n{calls}"
