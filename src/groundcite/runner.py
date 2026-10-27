"""The run loop shared by ``eval.py`` and ``scripts/run_ablation.py`` (R3, R10, R11, R13, R20).

A run is a list of systems (model x strategy) over the same scenarios. Each (system, scenario) is
rendered, generated, checked and judged, and its transcript row is appended to
``transcripts.jsonl`` as soon as it is done, so an interrupted run can resume without re-paying.
Requests run in a bounded thread pool; transient API failures (429, 5xx, dropped connections) are
retried with exponential backoff. A subject refusal, empty reply or transient error that survives
the retries is scored as a failure (0, 1) and kept. A non-retryable 4xx (bad key, no access,
malformed request) from any model, or a judge error that survives the retries, stops the run.

Each row records the prompt hash it was rendered under and the decoding parameters that actually
ran; a run directory's ``run.json`` records the eval set, scenario IDs, prompt hashes, judge and
rubric, and a resume refuses to continue a run whose inputs differ (R11, R13, D-043).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from groundcite import versions
from groundcite.backends import Backend, BackendError, Messages, get_backend
from groundcite.checks import run_checks
from groundcite.config import decoding_params
from groundcite.evalset import Scenario
from groundcite.judge import JudgeResult, judge_reply, rubric_sha256
from groundcite.prompts import STRATEGIES, prompt_sha256, render_messages
from groundcite.report import transcript_row

log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4
MAX_RETRIES = 4
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
BOOTSTRAP_SEED = 0
MOCK_JUDGE = "mock:judge"


class UsageError(RuntimeError):
    """A bad invocation or input, reported on stderr with exit code 2 before any work."""


@dataclass(frozen=True)
class System:
    """One row of the results table: a model under one prompt strategy."""

    model: str
    strategy: str


# ---------------------------------------------------------------------------------------------
# Specs and backends
# ---------------------------------------------------------------------------------------------


def normalize_spec(spec: str) -> str:
    """A bare ``org/repo`` means ``hf:org/repo``; ``mock`` alone means the mock judge."""
    if spec == "mock":
        return MOCK_JUDGE
    if ":" not in spec and "/" in spec:
        return f"hf:{spec}"
    return spec


def split_hf(spec: str) -> tuple[str, str | None]:
    """``hf:org/repo@rev`` -> ``("org/repo", "rev")``."""
    repo, _, rev = spec.split(":", 1)[1].partition("@")
    return repo, rev or None


class Retrying:
    """Wraps a backend: retries transient ``BackendError``s with jittered exponential backoff.

    Backends that are not thread-safe (local HF models) are serialized with a lock.
    """

    def __init__(
        self,
        inner: Backend,
        *,
        max_retries: int = MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self._max_retries = max_retries
        self._sleep = sleep
        self._lock = threading.Lock() if inner.name.startswith("hf:") else None

    def generate(self, messages: Messages, system: str | None, params: Mapping[str, Any]) -> str:
        attempt = 0
        while True:
            try:
                if self._lock is None:
                    return self.inner.generate(messages, system, params)
                with self._lock:
                    return self.inner.generate(messages, system, params)
            except BackendError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2**attempt) * random.uniform(0.5, 1)
                log.warning("%s: %s; retry %d in %.1fs", self.name, exc, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1

    def effective_params(self, requested: Mapping[str, Any]) -> dict[str, Any]:
        """The decoding params that actually ran (API backends drop rejected ones)."""
        effective = getattr(self.inner, "effective_params", None)
        return dict(effective) if isinstance(effective, Mapping) else dict(requested)

    def restore_effective_params(
        self, requested: Mapping[str, Any], recorded: Mapping[str, Any]
    ) -> None:
        """Adopt the params an earlier attempt of this run recorded, so a resume pins what ran
        even when it makes no calls, and doesn't re-send a parameter the model rejected."""
        restore = getattr(self.inner, "restore_effective_params", None)
        if callable(restore):
            restore(requested, recorded)


def build_backend(spec: str, *, revision: str | None = None) -> Retrying:
    """Build (and key-check) one backend. ``revision`` pins an ``hf:`` model to a commit."""
    build_spec = spec
    if revision is not None and spec.startswith("hf:"):
        repo, _ = split_hf(spec)
        build_spec = f"hf:{repo}@{revision}"
    return Retrying(get_backend(build_spec))


# ---------------------------------------------------------------------------------------------
# Planning and run directories
# ---------------------------------------------------------------------------------------------


def print_plan(
    systems: Sequence[System],
    scenarios: Sequence[Scenario],
    judge: str,
    eval_set: Path,
    out: TextIO | None = None,
) -> None:
    """The R20 dry-run plan: systems, judge and call counts. No network, no keys."""
    out = out or sys.stdout
    n = len(scenarios)
    print(f"eval set: {eval_set} ({n} scenarios)", file=out)
    print("systems (model x strategy):", file=out)
    for s in systems:
        print(f"  - {s.model} x {s.strategy}: {n} scenarios", file=out)
    print(f"judge: {judge}", file=out)
    print(f"generation calls: {len(systems) * n}", file=out)
    print(f"judge calls: {len(systems) * n}", file=out)
    print("dry run: no requests made, nothing written", file=out)


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{random.randbytes(3).hex()}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    """``path`` relative to the repo when it is inside it, so results carry no home path."""
    try:
        return str(path.resolve().relative_to(versions.REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------------------------
# The run manifest (resume safety)
# ---------------------------------------------------------------------------------------------

MANIFEST = "run.json"


def run_manifest(eval_set: Path, scenarios: Sequence[Scenario], judge: str) -> dict[str, Any]:
    """What a run directory's rows depend on besides the models: a resume must match it."""
    return {
        "eval_set_sha256": sha256_file(eval_set),
        "scenario_ids": [s.id for s in scenarios],
        "prompt_sha256": {strategy: prompt_sha256(strategy) for strategy in STRATEGIES},
        "judge_model": judge,
        "rubric_sha256": rubric_sha256(),
    }


def check_resumable(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Raise UsageError unless ``run_dir`` was started with the same inputs as ``manifest`` and
    every row on disk was rendered, judged and scored under them."""
    path = run_dir / MANIFEST
    if not path.is_file():
        raise UsageError(f"{run_dir} has no {MANIFEST}; it can't be resumed safely")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    labels = {
        "eval_set_sha256": "eval set (sha256)",
        "scenario_ids": "scenario list (--eval-set / --limit)",
        "prompt_sha256": "prompt hashes",
        "judge_model": "judge model",
        "rubric_sha256": "judge rubric",
    }
    for key, label in labels.items():
        if recorded.get(key) != manifest[key]:
            raise UsageError(
                f"{run_dir} was started with a different {label}; resume only continues the "
                "same run (start a new --run-id instead)"
            )
    prompts = manifest["prompt_sha256"]
    for row in read_transcripts(run_dir / "transcripts.jsonl"):
        if (
            row.get("judge_model") != manifest["judge_model"]
            or row.get("rubric_sha256") != manifest["rubric_sha256"]
        ):
            raise UsageError(
                f"{run_dir} has rows scored by {row.get('judge_model')} with rubric "
                f"{row.get('rubric_sha256')}; this run uses {manifest['judge_model']} with "
                f"{manifest['rubric_sha256']}"
            )
        if row.get("prompt_sha256") != prompts.get(row.get("strategy")):
            raise UsageError(
                f"{run_dir} has a {row.get('strategy')} row for {row.get('scenario_id')} rendered "
                "under another prompt; start a new --run-id instead"
            )


# ---------------------------------------------------------------------------------------------
# The run loop
# ---------------------------------------------------------------------------------------------


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str | None]:
    """The transcript cache key: a row is reused only under the same prompt (R13)."""
    return (row["model"], row["strategy"], row["scenario_id"], row.get("prompt_sha256"))


def read_transcripts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _generation_failure(judge: str, error: str) -> JudgeResult:
    """The score for a reply that never arrived: a failure, never dropped (R3, R11)."""
    return JudgeResult(
        spec_adherence=0,
        robustness=1,
        reasoning=f"Generation failed after retries, scored as a failure: {error}",
        failed_checks=(),
        raw="",
        forced_by_checks=False,
        error=f"generation failed: {error}",
        attempts=0,
        model=judge,
        rubric_sha256=rubric_sha256(),
    )


def _one(
    *,
    run_id: str,
    system: System,
    scenario: Scenario,
    backend: Retrying,
    revision: str | None,
    judge: Retrying,
    gen_params: Mapping[str, Any],
    judge_params: Mapping[str, Any],
    prompt_hash: str,
) -> dict[str, Any]:
    prompt_system, messages = render_messages(scenario, system.strategy)
    error: str | None = None
    try:
        reply = backend.generate(messages, prompt_system, gen_params)
    except BackendError as exc:
        if exc.stops_run:
            raise
        reply, error = "", str(exc)
    checks = run_checks(reply, scenario)
    if error is None:
        result = judge_reply(scenario, reply, checks, judge, params=judge_params)
    else:
        result = _generation_failure(judge.name, error)
    row = transcript_row(
        run_id=run_id,
        scenario=scenario,
        model=system.model,
        model_revision=revision,
        strategy=system.strategy,
        reply=reply,
        checks=checks,
        judge=result,
    )
    row["prompt_sha256"] = prompt_hash
    row["generation_error"] = error
    # What actually ran, so pins survive a resume that makes no calls (R13, R28).
    row["decoding"] = backend.effective_params(gen_params) if error is None else None
    row["judge"]["decoding"] = judge.effective_params(judge_params)
    return row


def run_grid(
    *,
    run_id: str,
    systems: Sequence[System],
    scenarios: Sequence[Scenario],
    backends: Mapping[str, Retrying],
    revisions: Mapping[str, str | None],
    judge: Retrying,
    transcripts: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: TextIO | None = None,
) -> list[dict[str, Any]]:
    """Run every (system, scenario) not already in ``transcripts`` and return all grid rows in
    system-then-scenario order. New rows are appended as they finish.

    The first exception stops the run: tasks not yet started are skipped, in-flight ones finish
    and are saved, and the exception is re-raised."""
    existing = read_transcripts(transcripts)
    done = {row_key(r): r for r in existing}
    hashes = {s.strategy: prompt_sha256(s.strategy) for s in systems}

    def key(system: System, scenario: Scenario) -> tuple[str, str, str, str]:
        return (system.model, system.strategy, scenario.id, hashes[system.strategy])

    todo = [
        (system, scenario)
        for system in systems
        for scenario in scenarios
        if key(system, scenario) not in done
    ]
    gen_params = {s.model: decoding_params(s.model) for s in systems}
    judge_params = decoding_params(judge.name)
    for row in existing:
        if isinstance(row.get("decoding"), Mapping) and row["model"] in backends:
            backends[row["model"]].restore_effective_params(
                gen_params.get(row["model"], {}), row["decoding"]
            )
        recorded_judge = (row.get("judge") or {}).get("decoding")
        if isinstance(recorded_judge, Mapping):
            judge.restore_effective_params(judge_params, recorded_judge)
    lock = threading.Lock()
    stop = threading.Event()
    total = len(systems) * len(scenarios)
    print(
        f"{run_id}: {total - len(todo)} of {total} rows done, {len(todo)} to run",
        file=progress or sys.stderr,
    )

    with transcripts.open("a", encoding="utf-8") as fh:

        def task(system: System, scenario: Scenario) -> None:
            if stop.is_set():
                return
            try:
                row = _one(
                    run_id=run_id,
                    system=system,
                    scenario=scenario,
                    backend=backends[system.model],
                    revision=revisions.get(system.model),
                    judge=judge,
                    gen_params=gen_params[system.model],
                    judge_params=judge_params,
                    prompt_hash=hashes[system.strategy],
                )
            except BaseException:
                stop.set()
                raise
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[row_key(row)] = row

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(task, system, scenario) for system, scenario in todo]
            for future in futures:
                future.result()  # re-raises the stopping error; finished rows are on disk

    return [done[key(system, scenario)] for system in systems for scenario in scenarios]


# ---------------------------------------------------------------------------------------------
# Pins and output
# ---------------------------------------------------------------------------------------------


def pins(
    *,
    eval_set: Path,
    judge: Retrying,
    systems: Sequence[System],
    backends: Mapping[str, Retrying],
    revisions: Mapping[str, str | None],
    git_dirty: bool,
) -> dict[str, Any]:
    """The R13 pins for results.json."""
    models = list(dict.fromkeys(s.model for s in systems))
    decoding = {m: backends[m].effective_params(decoding_params(m)) for m in models}
    decoding[judge.name] = judge.effective_params(decoding_params(judge.name))
    return {
        "git_sha": versions.git_sha(),
        "git_dirty": git_dirty,
        "judge_model": judge.name,
        "rubric_sha256": rubric_sha256(),
        "eval_set": {"path": display_path(eval_set), "sha256": sha256_file(eval_set)},
        "models": [{"model": m, "model_revision": revisions.get(m)} for m in models],
        "decoding": decoding,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "library_versions": versions.library_versions(),
    }


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def results_table(cells: Iterable[Mapping[str, Any]], *, extra: str | None = None) -> list[str]:
    """The markdown results table, one line per cell (R3, R10, R30)."""
    head = [
        "System",
        "Strategy",
        "n",
        "Spec-adherence (95% CI)",
        "Robustness",
        "Stressor spec",
        "HARD-check fail",
        "Invented-ID rate",
        "Abstain P / R",
    ]
    if extra:
        head.append(extra)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in cells:
        spec = c["spec_adherence"]
        values = [
            c["model"],
            c["strategy"],
            str(c["n"]),
            f"{fmt(spec['mean'])} [{fmt(spec['ci_low'])}, {fmt(spec['ci_high'])}]",
            fmt(c["robustness_mean"]),
            fmt(c["stressor_spec_adherence"]),
            fmt(c["det_fail_rate"]),
            fmt(c["invented_id_rate"]),
            f"{fmt(c['abstain_precision'])} / {fmt(c['abstain_recall'])}",
        ]
        if extra:
            values.append(str(c.get("_extra", "")))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def category_table(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    categories = list(dict.fromkeys(k for c in cells for k in c["spec_adherence_by_category"]))
    head = ["System", "Strategy", *categories]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in cells:
        by = c["spec_adherence_by_category"]
        values = [c["model"], c["strategy"], *[fmt(by.get(k)) for k in categories]]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def pins_section(p: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Pins",
        "",
        f"- Eval code: git `{p['git_sha']}`" + (" (dirty tree)" if p["git_dirty"] else ""),
        f"- Judge: `{p['judge_model']}`",
        f"- Rubric sha256: `{p['rubric_sha256']}`",
        f"- Eval set: `{p['eval_set']['path']}` (sha256 `{p['eval_set']['sha256']}`)",
    ]
    for m in p["models"]:
        lines.append(f"- Model `{m['model']}` revision `{m['model_revision']}`")
    for spec, params in p["decoding"].items():
        lines.append(f"- Decoding `{spec}`: `{json.dumps(params, sort_keys=True)}`")
    libs = ", ".join(f"{k} {v}" for k, v in p["library_versions"].items())
    lines.append(f"- Libraries: {libs}")
    return lines


def stop_message(exc: BackendError) -> str:
    """The stderr line for a run stopped by a non-retryable 4xx (the message is redacted)."""
    return (
        f"error: run stopped by a non-retryable HTTP {exc.status}: {exc}. Check the API key, "
        "model access and request, then rerun"
    )


def discard_if_empty(run_dir: Path) -> None:
    """Remove a run directory this invocation created when no row was written to it."""
    if not read_transcripts(run_dir / "transcripts.jsonl"):
        shutil.rmtree(run_dir, ignore_errors=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
