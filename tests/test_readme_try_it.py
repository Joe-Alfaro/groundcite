"""The README's "Try it" section (R15, R16, G32, D-118).

A ``## Try it`` section links the public adapter on the HF Hub, the demo notebook
``train/demo_inference.ipynb`` and the public dataset. Until the publish step runs, the dataset
link and the hosted notebook link may each be a placeholder line marked ``TODO-PUBLISH``. Every
URL host stays on the allowlist in ``tests/test_no_repo_url.py`` (no code-host URL).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_no_repo_url import ALLOWED_HOSTS, URL

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
MODEL_URL = "https://huggingface.co/joe-alfaro/groundcite-qwen3-1.7b"
DATASET_URL = re.compile(re.escape(MODEL_URL.rsplit("/", 2)[0]) + r"/datasets/joe-alfaro/[\w.-]+")
# Written as "/{2}" so the no-repo-URL scan does not read the pattern as a link.
KAGGLE_URL = re.compile(r"https:/{2}(?:www\.)?kaggle\.com/code/[\w.-]+/[\w.-]+")
PLACEHOLDER = "TODO-PUBLISH"


@pytest.fixture(scope="module")
def section() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^## Try it[^\n]*\n(.*?)(?=^## |\Z)", text)
    assert match, "the README has a '## Try it' section"
    return match.group(1)


def test_try_it_links_the_public_model(section: str) -> None:
    assert MODEL_URL in section


def test_try_it_links_the_demo_notebook(section: str) -> None:
    assert "train/demo_inference.ipynb" in section


def test_try_it_links_the_dataset_or_marks_it_todo(section: str) -> None:
    lines = section.splitlines()
    linked = DATASET_URL.search(section)
    pending = any(PLACEHOLDER in line and "dataset" in line.lower() for line in lines)
    assert linked or pending, f"link the HF dataset, or a '{PLACEHOLDER}' dataset line"


def test_try_it_links_the_hosted_notebook_or_marks_it_todo(section: str) -> None:
    lines = section.splitlines()
    linked = KAGGLE_URL.search(section)
    pending = any(PLACEHOLDER in line and "notebook" in line.lower() for line in lines)
    assert linked or pending, f"link the public notebook, or a '{PLACEHOLDER}' notebook line"


def test_try_it_says_the_demo_needs_no_token(section: str) -> None:
    assert re.search(r"no (?:token|secrets?|login)|logged[- ]out", section, re.IGNORECASE)


def test_every_try_it_url_is_on_the_allowlist(section: str) -> None:
    hosts = {m.group(1).lower().rstrip(".") for m in URL.finditer(section)}
    assert hosts, "the section has links"
    assert hosts <= ALLOWED_HOSTS, f"hosts outside the allowlist: {sorted(hosts - ALLOWED_HOSTS)}"
