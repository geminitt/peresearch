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
