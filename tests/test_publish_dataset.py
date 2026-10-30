"""Publishing a dataset version to the HF Hub (R15, G38, D-031, D-118).

    uv run scripts/publish_dataset.py --dataset data/datasets/vN --repo OWNER/NAME
        [--dry-run] [--private | --public] [--license ID] [--generation-date YYYY-MM-DD]
        [--stage-dir DIR] [--scenarios DIR] [--few-shot PATH]

The script is loaded in-process and driven through ``main(argv) -> int``: 0 on success, 1 when
the scrub check, the secret scan or the upload fails, 2 on a usage or input error (one line on
stderr).

It stages exactly these files, flat, in one folder: ``train.jsonl``, ``manifest.json`` and
``filter_report.md`` from ``--dataset`` (byte-for-byte copies), ``eval_own.jsonl`` and
``heldout.jsonl`` from ``--scenarios`` (default ``data/scenarios``), ``few_shot.jsonl`` from
``--few-shot`` (default ``data/few_shot/examples.jsonl``), and a generated ``README.md`` dataset
card. ``--stage-dir`` keeps the staged folder for review; otherwise it is a temp dir.

The card opens with HF YAML front matter (``license``, default ``cc-by-4.0``, set with
``--license``; ``tags`` including ``synthetic``; ``pretty_name``; one ``configs`` entry per JSONL
file). The body states "synthetic, not real service information" and covers the schema, the
generation and filter method, counts, the teacher model, the generation date, the license, known
limits, the snippet ID schemes and the eval judge note (the filter judge from the manifest and
the eval judge from ``config/models.toml``). The generation date is ``--generation-date``, else the
manifest's ``generated_at``, else the date of the last commit touching ``manifest.json``; with
none of them the script exits 2.

Every staged file passes the scrub check (terms from ``$SCRUB_TERMS_PATH``) and the secret scan
before anything is uploaded; hits are reported as ``file:line`` and never print the match.
``--dry-run`` runs both checks and lists the staged files, needs no token and makes no network
call. The upload reads ``HF_TOKEN`` (environment or ``.env``), never prints it, creates the
dataset repo (``create_repo(repo_type="dataset", exist_ok=True)``), private unless ``--public``
is passed, uploads the folder in one commit, prints the commit SHA and records it in
``<dataset>/publish.json``.

``huggingface_hub.HfApi`` is a fake; sockets are blocked and keys unset.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from distill_data import load_script, run_script

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data" / "datasets" / "v1"
SCENARIOS = REPO_ROOT / "data" / "scenarios"
FEW_SHOT = REPO_ROOT / "data" / "few_shot" / "examples.jsonl"
MODELS_CONFIG = REPO_ROOT / "config" / "models.toml"
REPO = "someone/groundcite-test-data"
FAKE_HF_TOKEN = "hf_" + "FakeTokenForPublishTests0123456789"
FAKE_COMMIT = "c0ffee" + "0" * 34
SCRUB_TERM = "zzqscrubterm"
STAGED = {
    "README.md",
    "train.jsonl",
    "manifest.json",
    "filter_report.md",
    "eval_own.jsonl",
    "heldout.jsonl",
    "few_shot.jsonl",
}
SOURCES = {
    "train.jsonl": V1 / "train.jsonl",
    "manifest.json": V1 / "manifest.json",
    "filter_report.md": V1 / "filter_report.md",
    "eval_own.jsonl": SCENARIOS / "eval_own.jsonl",
    "heldout.jsonl": SCENARIOS / "heldout.jsonl",
    "few_shot.jsonl": FEW_SHOT,
}
GENERATED_AT = "2026-10-28"


class FakeHfApi:
    """Records every call; tracks the repo's visibility; snapshots each uploaded folder."""

    instances: list[FakeHfApi] = []
    private: bool | None = None
    fail_with: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = dict(kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.uploaded: dict[str, bytes] = {}
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        if FakeHfApi.fail_with:
            raise RuntimeError(FakeHfApi.fail_with)
        if FakeHfApi.private is None:
            # A new repo: real create_repo makes it public unless private=True.
            FakeHfApi.private = bool(kwargs.get("private", False))
        return SimpleNamespace(repo_id=repo_id)

    def _set_private(self, name: str, repo_id: str, **kwargs: Any) -> None:
        self.calls.append((name, {"repo_id": repo_id, **kwargs}))
        if "private" in kwargs and kwargs["private"] is not None:
            FakeHfApi.private = bool(kwargs["private"])

    def update_repo_settings(self, repo_id: str, **kwargs: Any) -> None:
        self._set_private("update_repo_settings", repo_id, **kwargs)

    def update_repo_visibility(self, repo_id: str, **kwargs: Any) -> None:
        self._set_private("update_repo_visibility", repo_id, **kwargs)

    def upload_folder(self, **kwargs: Any) -> Any:
        self.calls.append(("upload_folder", dict(kwargs)))
        folder = Path(kwargs["folder_path"])
        for path in folder.rglob("*"):
            if path.is_file():
                self.uploaded[path.relative_to(folder).as_posix()] = path.read_bytes()
        return SimpleNamespace(
            oid=FAKE_COMMIT, commit_url=f"https://huggingface.co/datasets/x/commit/{FAKE_COMMIT}"
        )


# ---------------------------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def env(isolated_env: Path, no_network: None) -> Path:
    return isolated_env


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> type[FakeHfApi]:
    import huggingface_hub

    monkeypatch.setattr(FakeHfApi, "instances", [])
    monkeypatch.setattr(FakeHfApi, "private", None)
    monkeypatch.setattr(FakeHfApi, "fail_with", None)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


@pytest.fixture
def scrub_terms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "scrub_terms.txt"
    path.write_text(f"{SCRUB_TERM}\n", encoding="utf-8")
    monkeypatch.setenv("SCRUB_TERMS_PATH", str(path))
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A copy of v1 outside git, with a recorded generation date."""
    out = tmp_path / "datasets" / "v9"
    shutil.copytree(V1, out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["generated_at"] = GENERATED_AT
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def _publish(*argv: str) -> int:
    return run_script("publish_dataset", list(argv))


def _token(env_file: Path) -> None:
    env_file.write_text(f"HF_TOKEN={FAKE_HF_TOKEN}\n", encoding="utf-8")


def _stage(tmp_path: Path, dataset: Path, *extra: str) -> Path:
    stage = tmp_path / "stage"
    argv = ["--dataset", str(dataset), "--repo", REPO, "--dry-run", "--stage-dir", str(stage)]
    code = _publish(*argv, *extra)
    assert code == 0, "the dry run stages the files"
    return stage


def _card(stage: Path) -> tuple[dict[str, Any], str]:
    text = (stage / "README.md").read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    assert match, "the card opens with YAML front matter between --- lines"
    front = yaml.safe_load(match.group(1))
    assert isinstance(front, dict)
    return front, match.group(2)


def _calls(fake: type[FakeHfApi], name: str) -> list[dict[str, Any]]:
    return [kw for api in fake.instances for n, kw in api.calls if n == name]


def _uploaded(fake: type[FakeHfApi]) -> dict[str, bytes]:
    return {k: v for api in fake.instances for k, v in api.uploaded.items()}


def _data_files(config: dict[str, Any]) -> list[str]:
    raw = config.get("data_files")
    items = raw if isinstance(raw, list) else [raw]
    paths: list[str] = []
    for item in items:
        if isinstance(item, dict):
            path = item.get("path")
            paths += path if isinstance(path, list) else [path]
        else:
            paths.append(item)
    return [str(p) for p in paths]


def _jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------------------------------------------------------------------------------------------
# Staging and the dry run
# ---------------------------------------------------------------------------------------------


def test_the_script_exposes_main() -> None:
    assert callable(load_script("publish_dataset").main)


def test_dry_run_lists_the_staged_files_and_calls_nothing(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _publish("--dataset", str(dataset), "--repo", REPO, "--dry-run") == 0
    out = capsys.readouterr().out
    for name in STAGED:
        assert name in out, f"the dry run lists {name}"
    assert fake_hub.instances == [], "no HfApi on a dry run"
    assert not (dataset / "publish.json").exists(), "a dry run records nothing"


def test_exactly_the_listed_files_are_staged_byte_for_byte(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    stage = _stage(tmp_path, dataset)
    staged = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    assert staged == STAGED, "rejections, notes and leakage reports stay out"
    for name, src in SOURCES.items():
        if src.parent == V1:
            src = dataset / src.name
        assert (stage / name).read_bytes() == src.read_bytes(), f"{name} is copied unchanged"


def test_the_scenario_and_few_shot_sources_can_be_overridden(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    scenarios = tmp_path / "scen"
    scenarios.mkdir()
    for name in ("eval_own.jsonl", "heldout.jsonl"):
        lines = (SCENARIOS / name).read_text(encoding="utf-8").splitlines()[:3]
        (scenarios / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    few_shot = tmp_path / "fs.jsonl"
    few_shot.write_text(FEW_SHOT.read_text(encoding="utf-8").splitlines()[0] + "\n", "utf-8")
    stage = _stage(tmp_path, dataset, "--scenarios", str(scenarios), "--few-shot", str(few_shot))
    assert (stage / "eval_own.jsonl").read_bytes() == (scenarios / "eval_own.jsonl").read_bytes()
    assert (stage / "few_shot.jsonl").read_bytes() == few_shot.read_bytes()


@pytest.mark.parametrize(
    "argv",
    [
        ["--repo", "no-owner"],
        ["--repo", "a/b/c"],
        ["--repo", REPO, "--public", "--private"],
    ],
    ids=["no-owner", "too-deep", "public-and-private"],
)
def test_bad_arguments_exit_2(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, argv: list[str]
) -> None:
    assert _publish("--dataset", str(dataset), *argv, "--dry-run") == 2
    assert fake_hub.instances == []


def test_a_dataset_without_a_manifest_exits_2(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path
) -> None:
    (dataset / "manifest.json").unlink()
    assert _publish("--dataset", str(dataset), "--repo", REPO, "--dry-run") == 2


# ---------------------------------------------------------------------------------------------
# The dataset card
# ---------------------------------------------------------------------------------------------


def test_the_card_front_matter_has_license_tags_and_configs(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    front, _ = _card(_stage(tmp_path, dataset))
    assert front.get("license") == "cc-by-4.0", "the default license (D-118 decision)"
    assert "synthetic" in front.get("tags", [])
    assert front.get("pretty_name")
    configs = front.get("configs")
    assert isinstance(configs, list) and configs
    files = [f for c in configs for f in _data_files(c)]
    for name in ("train.jsonl", "eval_own.jsonl", "heldout.jsonl", "few_shot.jsonl"):
        assert name in files, f"a config for {name}"
    assert all(c.get("config_name") for c in configs)


def test_the_license_can_be_set(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    front, body = _card(_stage(tmp_path, dataset, "--license", "mit"))
    assert front["license"] == "mit"
    assert "mit" in body.lower(), "the body states the license too"


def test_the_card_says_synthetic_not_real_service_information(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    _, body = _card(_stage(tmp_path, dataset))
    assert "synthetic, not real service information" in body.lower()


def test_the_card_covers_schema_method_counts_teacher_limits_and_ids(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    _, body = _card(_stage(tmp_path, dataset))
    headings = [h.lower() for h in re.findall(r"(?m)^#{1,4}\s+(.+?)\s*$", body)]

    assert any("schema" in h for h in headings), "a schema section"
    for field in ("messages", "snippets", "gold"):
        assert f"`{field}`" in body, f"the schema names `{field}`"
    assert any("generation" in h or "method" in h for h in headings), "a method section"
    assert "filter" in body.lower() and manifest["filter_version"] in body
    assert any("limit" in h for h in headings), "a known-limits section"

    assert str(manifest["count"]) in body
    for category, n in manifest["counts_by_category"].items():
        assert re.search(rf"{re.escape(category)}\W+(?:\S+\W+)?{n}\b", body), (
            f"the {category} count"
        )
    for name in ("eval_own.jsonl", "heldout.jsonl", "few_shot.jsonl"):
        assert str(_jsonl_rows(SOURCES[name])) in body, f"the {name} row count"

    assert manifest["teacher_model"] in body
    assert GENERATED_AT in body, "the generation date"
    for scheme in manifest["id_scheme_mix"]:
        assert scheme in body, "show the synthetic ID schemes"


def test_the_card_notes_the_filter_judge_and_the_eval_judge(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    eval_judge = tomllib.loads(MODELS_CONFIG.read_text(encoding="utf-8"))["judge"]["model"]
    _, body = _card(_stage(tmp_path, dataset))
    assert manifest["judge_model"] in body, "the judge that filtered the data"
    assert eval_judge in body, "the judge that scores eval"


def test_the_generation_date_flag_wins(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, tmp_path: Path
) -> None:
    _, body = _card(_stage(tmp_path, dataset, "--generation-date", "2026-01-02"))
    assert "2026-01-02" in body


def test_the_generation_date_falls_back_to_the_manifest_commit_date(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, tmp_path: Path
) -> None:
    """Committed v1 records no generated_at: the card uses the manifest's last commit date."""
    log = ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%cs", "--"]
    manifest = "data/datasets/v1/manifest.json"
    done = subprocess.run([*log, manifest], check=True, capture_output=True, text=True)
    date = done.stdout.strip()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
    _, body = _card(_stage(tmp_path, V1))
    assert date in body


def test_no_generation_date_at_all_exits_2(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    del manifest["generated_at"]
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _publish("--dataset", str(dataset), "--repo", REPO, "--dry-run") == 2
    assert "--generation-date" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# Scrub check and secret scan before any upload
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "upload"])
def test_a_scrub_hit_in_a_staged_file_stops_everything(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    _token(env)
    report = dataset / "filter_report.md"
    report.write_text(report.read_text(encoding="utf-8") + f"\nmade for {SCRUB_TERM}\n", "utf-8")
    argv = ["--dataset", str(dataset), "--repo", REPO] + (["--dry-run"] if dry_run else [])
    assert _publish(*argv) == 1
    err = capsys.readouterr().err
    assert "scrub" in err.lower()
    assert "filter_report.md" in err, "report the file:line"
    assert SCRUB_TERM not in err, "never print the term"
    assert _calls(fake_hub, "create_repo") == [] and _calls(fake_hub, "upload_folder") == []
    assert not (dataset / "publish.json").exists()


def test_a_scrub_hit_in_the_generated_card_stops_the_upload(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path
) -> None:
    """The card is checked too: a scrub term reaching it through the manifest is caught."""
    _token(env)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["teacher_model"] = f"anthropic:{SCRUB_TERM}"
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _publish("--dataset", str(dataset), "--repo", REPO) == 1
    assert _calls(fake_hub, "upload_folder") == []


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "upload"])
def test_a_key_shaped_string_in_a_staged_file_stops_everything(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    _token(env)
    leaked = "sk-" + "a1B2" * 10
    train = dataset / "train.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + json.dumps({"x": leaked}) + "\n", "utf-8")
    argv = ["--dataset", str(dataset), "--repo", REPO] + (["--dry-run"] if dry_run else [])
    assert _publish(*argv) == 1
    captured = capsys.readouterr()
    assert "secret" in captured.err.lower()
    assert "train.jsonl" in captured.err
    assert leaked not in captured.out + captured.err, "never print the match"
    assert _calls(fake_hub, "create_repo") == [] and _calls(fake_hub, "upload_folder") == []


def test_a_missing_scrub_terms_file_exits_2(
    env: Path,
    fake_hub: type[FakeHfApi],
    dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _token(env)
    monkeypatch.setenv("SCRUB_TERMS_PATH", str(tmp_path / "missing.txt"))
    assert _publish("--dataset", str(dataset), "--repo", REPO) == 2
    assert _calls(fake_hub, "create_repo") == [] and _calls(fake_hub, "upload_folder") == []


# ---------------------------------------------------------------------------------------------
# The upload
# ---------------------------------------------------------------------------------------------


def test_the_upload_creates_a_private_dataset_repo_and_records_the_commit(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _token(env)
    assert _publish("--dataset", str(dataset), "--repo", REPO) == 0

    creates = _calls(fake_hub, "create_repo")
    assert len(creates) == 1
    assert creates[0]["repo_id"] == REPO
    assert creates[0].get("repo_type") == "dataset"
    assert creates[0].get("private") is True, "private by default"
    assert creates[0].get("exist_ok") is True

    uploads = _calls(fake_hub, "upload_folder")
    assert len(uploads) == 1, "one commit"
    assert uploads[0]["repo_id"] == REPO and uploads[0].get("repo_type") == "dataset"
    assert set(_uploaded(fake_hub)) == STAGED

    assert FAKE_COMMIT in capsys.readouterr().out, "print the commit SHA"
    record = json.loads((dataset / "publish.json").read_text(encoding="utf-8"))
    assert record["repo"] == REPO
    assert record["commit_sha"] == FAKE_COMMIT
    assert record["private"] is True


def test_the_token_comes_from_hf_token_and_is_never_shown(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _token(env)
    assert _publish("--dataset", str(dataset), "--repo", REPO) == 0
    seen = [api.init_kwargs.get("token") for api in fake_hub.instances]
    seen += [kw.get("token") for api in fake_hub.instances for _, kw in api.calls]
    assert FAKE_HF_TOKEN in seen, "authenticate with HF_TOKEN"
    captured = capsys.readouterr()
    assert FAKE_HF_TOKEN not in captured.out + captured.err
    assert FAKE_HF_TOKEN.encode() not in (dataset / "publish.json").read_bytes()
    assert all(FAKE_HF_TOKEN.encode() not in data for data in _uploaded(fake_hub).values())


def test_an_upload_error_is_redacted(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _token(env)
    monkeypatch.setattr(FakeHfApi, "fail_with", f"401 bad token {FAKE_HF_TOKEN}")
    assert _publish("--dataset", str(dataset), "--repo", REPO) == 1
    captured = capsys.readouterr()
    assert FAKE_HF_TOKEN not in captured.out + captured.err
    assert "401" in captured.err, "keep the useful part of the error"
    assert not (dataset / "publish.json").exists()


def test_no_token_exits_2_before_any_hub_call(
    env: Path,
    fake_hub: type[FakeHfApi],
    scrub_terms: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _publish("--dataset", str(dataset), "--repo", REPO) == 2
    assert "HF_TOKEN" in capsys.readouterr().err
    assert all(not api.calls for api in fake_hub.instances)


@pytest.mark.parametrize("flags", [[], ["--private"]], ids=["default", "private"])
def test_the_repo_is_never_made_public_without_the_flag(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path, flags: list[str]
) -> None:
    _token(env)
    assert _publish("--dataset", str(dataset), "--repo", REPO, *flags) == 0
    assert FakeHfApi.private is True
    for name in ("update_repo_settings", "update_repo_visibility"):
        assert all(kw.get("private") is not False for kw in _calls(fake_hub, name))


def test_public_makes_the_repo_public_even_if_it_exists_private(
    env: Path, fake_hub: type[FakeHfApi], scrub_terms: Path, dataset: Path
) -> None:
    _token(env)
    FakeHfApi.private = True  # an earlier private publish
    assert _publish("--dataset", str(dataset), "--repo", REPO, "--public") == 0
    assert FakeHfApi.private is False
    record = json.loads((dataset / "publish.json").read_text(encoding="utf-8"))
    assert record["private"] is False
