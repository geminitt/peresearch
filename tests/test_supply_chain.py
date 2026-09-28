"""Models and packages are pinned: a moved Hugging Face revision or an unlocked dependency is a different
program than the one that was tested."""
import re
from pathlib import Path

from peresearch.zetokrag import models

ROOT = Path(__file__).parent.parent


def test_every_model_is_pinned_to_a_commit():
    for spec in [*models.EMBEDDERS.values(), models.RERANKER]:
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), spec["repo"]


def test_dependencies_are_locked():
    lock = (ROOT / "pixi.lock").read_text()
    for package in ["bm25s", "pymupdf", "torch", "sentence-transformers"]:
        assert re.search(rf"/{package}-\d", lock) or f"name: {package}" in lock, package
