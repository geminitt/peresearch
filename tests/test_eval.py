"""The benchmark's metrics against hand-computed values."""
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("retrieval", Path(__file__).parent.parent / "eval" / "retrieval.py")
retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval)


def test_ranking_metrics():
    m = retrieval.metrics(["x", "a", "y", "b"], {"a": 2, "b": 1, "c": 1})
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(m["ndcg@10"], dcg / ideal)
    assert m["recall@10"] == 2 / 3 and m["mrr@10"] == 1 / 2
    assert retrieval.metrics(["z"], {"a": 1})["ndcg@10"] == 0


def test_auroc():
    assert retrieval.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert retrieval.auroc([0.5], [0.5]) == 0.5
    assert retrieval.auroc([0.1], [0.9]) == 0.0


def test_own_document_is_excluded_only_where_the_query_is_in_the_corpus():
    import numpy as np
    assert retrieval.own_documents("scifact", ["a", "q1"], ["q1"]) == [None]
    assert retrieval.own_documents("arguana", ["a", "q1"], ["q1", "q9"]) == [1, None]
    scores = retrieval.exclude([np.array([0.2, 0.9])], [1])[0]
    assert scores[1] == -np.inf and int(np.argmax(scores)) == 0


def test_exact_binomial_interval():
    spec2 = importlib.util.spec_from_file_location("personal", Path(__file__).parent.parent / "eval" / "personal.py")
    personal = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(personal)
    lo, hi = personal.exact_interval(25, 25)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 25)) < 1e-6          # 0.863: the rule of three's exact form
    lo, hi = personal.exact_interval(0, 60)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 60))) < 1e-6
    lo, hi = personal.exact_interval(5, 10)
    assert abs(lo - 0.1871) < 1e-3 and abs(hi - 0.8129) < 1e-3      # textbook Clopper-Pearson for 5/10


def test_every_corpus_is_pinned_and_shards_cover_all_once():
    assert set(retrieval.REVISIONS) == set(retrieval.DATASETS)
    assert all(len(r) == 40 for r in retrieval.REVISIONS.values())
    shards = [retrieval.shard(k, 4) for k in range(4)]
    assert sorted(c for s in shards for c in s) == sorted(retrieval.DATASETS)
    assert len(retrieval.jobs()) == 29


def test_retry_gives_up_only_after_the_last_try():
    import pytest

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("429")
        return "ok"

    assert retrieval.retry(flaky, tries=3, wait=0) == "ok" and len(calls) == 3
    with pytest.raises(ConnectionError):
        retrieval.retry(lambda: (_ for _ in ()).throw(ConnectionError("429")), tries=2, wait=0)
