"""eval/retrieval.py and eval/speed.py end to end on a tiny corpus, with the fake models (CPU). Runs where torch
is installed (the default environment), so every line the Kaggle run takes is executed before it reaches a GPU."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import FakeEmbedder, FakeReranker

pytest.importorskip("torch")

EVAL = Path(__file__).parent.parent / "eval"
TOPICS = ["học máy", "nấu phở", "bóng đá", "lập trình python", "du lịch đà lạt"]


def fake_load(name):
    """40 documents on 5 topics, one query per topic; in ArguAna the first query's own document is in the corpus."""
    doc_ids = [f"d{i}" for i in range(40)]
    if name == "arguana":
        doc_ids[0] = "q0"
    texts = [f"{TOPICS[i % 5]} bài số {i} nói về {TOPICS[i % 5]}" for i in range(40)]
    qs = [f"q{k}" for k in range(5)]
    rel = {q: {doc_ids[j]: 1 for j in range(40) if j % 5 == k and doc_ids[j] != q} for k, q in enumerate(qs)}
    return doc_ids, texts, qs, [f"tìm hiểu {t}" for t in TOPICS], rel


class Embedder(FakeEmbedder):
    def __init__(self, name="bge-m3", batch_size=32):
        self.name = name
        self.model = SimpleNamespace(tokenizer=lambda t, add_special_tokens=True: {"input_ids": t.split()})


class Reranker(FakeReranker):
    def __init__(self, batch_size=32):
        pass


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    monkeypatch.setenv("PERESEARCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("PERESEARCH_RESULTS", str(tmp_path / "results"))
    monkeypatch.syspath_prepend(str(EVAL))
    for m in ("retrieval", "speed"):
        sys.modules.pop(m, None)
    import retrieval
    import speed

    from peresearch.zetokrag import models
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)   # fake models: the laptop's GPU is not touched,
    monkeypatch.setattr(models, "Embedder", Embedder)                 # and a chat holding it does not fail the run
    monkeypatch.setattr(models, "Reranker", Reranker)
    monkeypatch.setattr(retrieval, "load", fake_load)
    monkeypatch.setattr(speed, "load", fake_load)
    yield retrieval, speed
    for m in ("retrieval", "speed"):
        sys.modules.pop(m, None)


def test_benchmark_and_speed_run_end_to_end(scripts):
    retrieval, speed = scripts
    assert retrieval.run(["scifact-vn", "arguana"]) == []
    for name, setting in [("scifact-vn", "as-typed"), ("scifact-vn", "no-diacritics"), ("arguana", "as-typed")]:
        m = json.loads((retrieval.RUNS / name / setting / "metrics.json").read_text())
        assert set(m["per_query"]) == set(retrieval.METHODS) and len(m["answerable"]) == 5
        assert all(0 <= q["ndcg@10"] <= 1 for v in m["per_query"].values() for q in v)
    z = np.load(retrieval.RUNS / "arguana" / "as-typed" / "first_stage.npz")
    assert all(0 not in z[k][0] for k in z.files if k.endswith(":ids"))     # q0's own document never ranked
    text = retrieval.report()
    assert "zetokrag" in text and not re.search(r"\bnan\b", text)
    speed.main(["scifact-vn"])
    r = json.loads((speed.OUT / "scifact-vn.json").read_text())
    assert set(r["index"]) == set(r["query"]) == set(speed.MODELS)
    assert not re.search(r"\bnan\b", speed.report())


def test_a_failing_corpus_does_not_stop_the_others(scripts, monkeypatch):
    retrieval, _ = scripts

    def broken(name):
        if name == "arguana":
            raise ConnectionError("429")
        return fake_load(name)

    monkeypatch.setattr(retrieval, "load", broken)
    assert retrieval.run(["arguana", "scifact-vn"]) == ["arguana/as-typed"]
    assert (retrieval.RUNS / "scifact-vn" / "no-diacritics" / "metrics.json").exists()
