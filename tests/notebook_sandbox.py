"""Run a Kaggle notebook's code cells in-process with every side effect faked.

Shell commands (``subprocess``, ``os.system`` and ``!`` lines) are recorded, not run.
``kaggle_secrets`` returns a dummy value for every secret (``secrets`` overrides that: a label
mapped to None is missing, and reading it raises the way Kaggle does when a secret is not
attached), ``torch`` reports one T4, and file
system checks and directory changes are no-ops that succeed. Parameter values are set the way a
user edits the parameters cell: each ``NAME = ...`` line in the tagged cell is rewritten. Execution
stops at the first command that ``stop_at`` matches, which is recorded with the environment it
would have run with.
"""

from __future__ import annotations

import builtins
import os
import re
import shlex
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock


class NotebookStopped(Exception):
    """Raised by the fake shell at the command the test stops at."""


@dataclass
class Sandbox:
    commands: list[str] = field(default_factory=list)
    stopped_at: str | None = None
    stop_env: dict[str, str] = field(default_factory=dict)
    error: BaseException | None = None


def source(cell: Mapping[str, Any]) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def parameters_cell(notebook: Mapping[str, Any]) -> Mapping[str, Any]:
    tagged = [
        c
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "parameters" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(tagged) == 1, "exactly one code cell tagged 'parameters'"
    return tagged[0]


def set_parameters(params: str, values: Mapping[str, str]) -> str:
    for name, value in values.items():
        pattern = rf"(?m)^{re.escape(name)}\s*(?::[^=]*)?=.*$"
        assert re.search(pattern, params), f"the parameters cell has no {name}"
        line = f"{name} = {value!r}"
        params = re.sub(pattern, lambda _, line=line: line, params)
    return params


def _shell_magic(code: str) -> str:
    """``!cmd`` lines become recorded shell calls (IPython expands ``{expr}``); ``%`` magics are
    dropped."""
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("!"):
            lines.append(f"{indent}__shell__(f{stripped[1:]!r})")
        elif stripped.startswith("%"):
            lines.append(f"{indent}pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def _command_text(cmd: Any) -> str:
    if isinstance(cmd, str | bytes):
        return cmd if isinstance(cmd, str) else cmd.decode()
    return shlex.join(str(c) for c in cmd)


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda *a, **k: "Tesla T4",
        get_device_capability=lambda *a, **k: (7, 5),
    )
    torch.cuda = cuda  # type: ignore[attr-defined]
    torch.__version__ = "2.0.0+cu121"  # type: ignore[attr-defined]
    return torch


class SecretNotAttached(Exception):
    """What the fake raises for a missing secret (Kaggle raises a generic backend error)."""


def _fake_secrets(overrides: Mapping[str, str | None] | None = None) -> types.ModuleType:
    module = types.ModuleType("kaggle_secrets")
    overrides = dict(overrides or {})

    class UserSecretsClient:
        def get_secret(self, label: str) -> str:
            if label in overrides:
                value = overrides[label]
                if value is None:
                    raise SecretNotAttached(
                        "Unexpected response from the service. Response: {'errors': ['No user "
                        "secrets exist for kernel id 1 and label " + label + "']}."
                    )
                return value
            return f"dummy-{label.lower()}"

    module.UserSecretsClient = UserSecretsClient  # type: ignore[attr-defined]
    return module


@contextmanager
def _faked(
    sandbox: Sandbox,
    stop_at: Callable[[str], bool],
    secrets: Mapping[str, str | None] | None = None,
) -> Iterator[Callable[..., Any]]:
    def shell(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        text = _command_text(cmd)
        sandbox.commands.append(text)
        if stop_at(text):
            sandbox.stopped_at = text
            env = dict(os.environ)
            env.update(kwargs.get("env") or {})
            sandbox.stop_env = env
            raise NotebookStopped(text)
        return subprocess.CompletedProcess(text, 0, stdout="", stderr="")

    def check_output(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        shell(cmd, *args, **kwargs)
        return "" if kwargs.get("text") or kwargs.get("universal_newlines") else b""

    def system(cmd: str) -> int:
        shell(cmd)
        return 0

    true = mock.Mock(return_value=True)
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(subprocess, "run", shell),
        mock.patch.object(subprocess, "call", lambda *a, **k: shell(*a, **k).returncode),
        mock.patch.object(subprocess, "check_call", lambda *a, **k: shell(*a, **k).returncode),
        mock.patch.object(subprocess, "check_output", check_output),
        mock.patch.object(os, "system", system),
        mock.patch.object(os, "chdir", lambda *a, **k: None),
        mock.patch.object(os, "makedirs", lambda *a, **k: None),
        mock.patch.object(os.path, "exists", true),
        mock.patch.object(os.path, "isfile", true),
        mock.patch.object(os.path, "isdir", true),
        mock.patch.object(Path, "exists", lambda self, *a, **k: True),
        mock.patch.object(Path, "is_file", lambda self, *a, **k: True),
        mock.patch.object(Path, "is_dir", lambda self, *a, **k: True),
        mock.patch.object(Path, "mkdir", lambda self, *a, **k: None),
        mock.patch.dict(
            sys.modules, {"kaggle_secrets": _fake_secrets(secrets), "torch": _fake_torch()}
        ),
    ):
        yield shell


def run_notebook(
    notebook: Mapping[str, Any],
    *,
    params: Mapping[str, str],
    stop_at: Callable[[str], bool],
    secrets: Mapping[str, str | None] | None = None,
) -> Sandbox:
    """Execute the code cells in order until ``stop_at`` matches a command or a cell raises."""
    tagged = parameters_cell(notebook)
    sandbox = Sandbox()
    namespace: dict[str, Any] = {"__name__": "__main__", "__builtins__": builtins}
    with _faked(sandbox, stop_at, secrets) as shell:
        namespace["__shell__"] = shell
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = source(cell)
            if cell is tagged:
                code = set_parameters(code, params)
            try:
                exec(compile(_shell_magic(code), "<notebook cell>", "exec"), namespace)
            except NotebookStopped:
                return sandbox
            except (Exception, SystemExit) as exc:  # the notebook's own failure
                sandbox.error = exc
                return sandbox
    return sandbox


def flag(command: str, name: str) -> str | None:
    """The value after ``name`` in a shell command (``--name value`` or ``--name=value``)."""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for i, part in enumerate(parts):
        if part == name and i + 1 < len(parts):
            return parts[i + 1]
        if part.startswith(name + "="):
            return part.split("=", 1)[1]
    return None
