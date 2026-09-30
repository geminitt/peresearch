"""eval/multilingual.py: fusion from saved top lists, reranking from cached scores, the lexical terms, and the whole
lexical -> scores -> analyze pipeline on a synthetic set with fake models, including a resume after a crash."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
import multilingual  # noqa: E402

from peresearch.zetokrag import core  # noqa: E402


def test_fusion_from_saved_top_lists_equals_zetokrag_fusion():
    rng = np.random.default_rng(1)
    for share in (0.05, 0.3, 0.0, 1.0):                  # few matches (zeros fill the top 100), many, none, all
        for _ in range(10):
            lex = np.where(rng.random(500) < share, rng.random(500) * 12, 0.0).astype(np.float32)
            dense = rng.random(500).astype(np.float32)
            ids, sc = multilingual.fuse(multilingual._top(lex), multilingual._top(dense))
            ref_ids, ref_sc = core.fuse_minmax(lex, dense, 0.5, 100)
            assert np.allclose(sc, ref_sc)
            assert ids[sc > 0].tolist() == ref_ids[ref_sc > 0].tolist()   # ties at 0 may be ordered otherwise


def test_bm25_matching_nothing_pushes_arbitrary_documents_up_unless_guarded():
    """ZetokRAG's min-max maps an all-zero BM25 list to 1 (core.minmax: hi == lo), so documents BM25 did not
    match at all enter the fusion with rho; the guard drops such a list."""
    lex = (np.arange(100, dtype=np.int32), np.zeros(100, np.float32))
    dense = multilingual._top(np.linspace(0, 1, 500).astype(np.float32))
    ids, _ = multilingual.fuse(lex, dense)
    assert set(ids[:10].tolist()) & set(range(100))                    # junk from the empty BM25 list on top
    ids, _ = multilingual.fuse(lex, dense, guard=True)
    assert ids[:10].tolist() == list(range(499, 489, -1))              # the dense order


def test_an_empty_lexical_list_leaves_the_dense_order():
    dense = multilingual._top(np.array([0.1, 0.9, 0.5], dtype=np.float32))
    ids, _ = multilingual.fuse((np.zeros(0, np.int32), np.zeros(0, np.float32)), dense)
    assert ids.tolist() == [1, 2, 0]


def test_reranking_from_cached_scores_matches_core_and_refuses_gaps():
    order = np.array([5, 3, 9, 1])
    cached = {5: 0.1, 3: 0.9, 9: 0.5}
    assert multilingual.reranked(order, cached, 3) == [3, 9, 5, 1]
    ids, _ = core.rerank(order, np.array([0.1, 0.9, 0.5]))
    assert ids.tolist() == [3, 9, 5, 1]
    assert multilingual.reranked(order, cached, 4) is None           # 1 was never scored


def test_folded_terms_add_the_bare_form_as_zetokrag_does():
    text = "Học máy SuperBPE 机器学习"
    assert multilingual.lexical_terms("current", [text], True)[0] == core.tokenize(text, folded=True)
    assert multilingual.lexical_terms("unspaced-bigrams", [text], True)[0] == \
        ["học", "hoc", "máy", "may", "superbpe", "机器", "器学", "学习"]


# --- the pipeline on a synthetic set with fake models --------------------------------------------------------------

WORDS = ["alpha", "beta", "gamma", "delta", "học", "máy", "dữ", "liệu", "mạng", "nơron", "kernel", "graph"]


def _vec(text: str, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return v / (np.linalg.norm(v) or 1)


class FakeEmbedder:
    def __init__(self, name="x", batch_size=32):
        self.name = name

    def documents(self, texts):
        return np.stack([_vec(t) for t in texts]).astype(np.float16)

    def queries(self, texts):
        return self.documents(texts)


class FakeReranker:
    calls = 0
    fail_after = None

    def scores(self, query, texts):
        FakeReranker.calls += 1
        if FakeReranker.fail_after is not None and FakeReranker.calls > FakeReranker.fail_after:
            raise RuntimeError("simulated crash")
        q = set(query.lower().split())
        return np.array([len(q & set(t.lower().split())) + 1e-3 * len(t) for t in texts], dtype=np.float32)


class FakeBgeM3:
    def __init__(self, batch_size=32):
        pass

    def encode(self, texts):
        import scipy.sparse as sp

        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            for w in set(t.lower().split()):
                rows.append(r)
                cols.append(int(hashlib.md5(w.encode()).hexdigest(), 16) % 997)
                vals.append(0.1 + len(w) / 10)
        return (np.stack([_vec(t[::-1]) for t in texts]).astype(np.float16),
                sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), 997), dtype=np.float32))


def _synthetic(n_docs=2100, n_queries=130, seed=0):
    rng = np.random.default_rng(seed)
    texts = [" ".join(rng.choice(WORDS, 6)) + f" doc{i}" for i in range(n_docs)]
    ids = [f"d{i}" for i in range(n_docs)]
    qtexts, rels = [], []
    for k in range(n_queries):
        target = int(rng.integers(n_docs))
        qtexts.append(" ".join(texts[target].split()[:3]) + f" doc{target}")
        rels.append({ids[target]: 1})
    return ids, texts, [f"q{k}" for k in range(n_queries)], qtexts, rels


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import peresearch.zetokrag.models as models

    data = {"synth-en": _synthetic(seed=0), "synth-vi": _synthetic(seed=1)}
    monkeypatch.setattr(multilingual, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(multilingual, "ROOT", tmp_path)
    monkeypatch.setattr(multilingual, "SETS", {"synth-en": ("en", "synthetic", (), "test"),
                                                "synth-vi": ("vi", "synthetic", (), "test")})
    monkeypatch.setattr(multilingual, "load", lambda name: data[name])
    monkeypatch.setattr(multilingual, "BgeM3", FakeBgeM3)
    monkeypatch.setattr(multilingual, "LEXICAL", ["current", "unspaced-bigrams"])
    monkeypatch.setattr(multilingual, "CANDIDATE", "unspaced-bigrams")
    monkeypatch.setattr(multilingual, "CORES", {"qwen3": 1, "bge-m3-hybrid": 1, "qwen3 + rerank": 2,
                                                 "fusion current rho=0.5": 2, "zetokrag (as used)": 3,
                                                 "zetokrag, rerank top 50": 3,
                                                 "zetokrag with bm25 unspaced-bigrams": 3})
    monkeypatch.setattr(multilingual.retrieval, "preflight", lambda: None)
    monkeypatch.setattr(models, "Embedder", FakeEmbedder)
    monkeypatch.setattr(models, "Reranker", FakeReranker)
    monkeypatch.setenv("PERESEARCH_DEVICE", "cpu")
    FakeReranker.calls, FakeReranker.fail_after = 0, None
    return tmp_path


def test_the_pipeline_runs_end_to_end_and_resumes_after_a_crash(pipeline):
    pytest.importorskip("torch")                                    # the GPU stage (CI has no PyTorch)
    names = ["synth-en", "synth-vi"]
    assert multilingual.run_lexical(names) == []
    for n in names:
        for st in multilingual.settings(n):
            z = np.load(pipeline / "runs" / n / st / "lexical.npz")
            assert z["current:ids"].shape == (130, 100)
    assert multilingual.settings("synth-vi") == ["as-typed", "no-diacritics"]

    FakeReranker.fail_after = 110                                   # crash at the first set's 111th query
    assert multilingual.run_scores(names) == names                  # both fail: the counter keeps failing
    part = pipeline / "runs" / "synth-en" / "as-typed" / "rerank.part.npz"
    assert part.exists() and int(np.load(part)["queries_done"]) == 100
    FakeReranker.fail_after = None
    FakeReranker.calls = 0
    assert multilingual.run_scores(names) == []
    assert FakeReranker.calls == 30 + 2 * 130                       # the first set resumed at query 100
    for n in names:
        for st in multilingual.settings(n):
            assert (pipeline / "runs" / n / st / "rerank.npz").exists()
            assert not (pipeline / "runs" / n / st / "rerank.part.npz").exists()

    resumed = {(n, st): np.load(pipeline / "runs" / n / st / "rerank.npz")["scores"].copy()
               for n in names for st in multilingual.settings(n)}
    report = multilingual.analyze(names, n_boot=200)
    assert "## Decision by the rule set beforehand" in report and "**pick:" in report
    assert "zetokrag (as used)" in report and "## Diagnostics" in report
    m = json.loads((pipeline / "runs" / "synth-en" / "as-typed" / "metrics.json").read_text())
    for v in ("bm25 current", "qwen3", "bge-m3-hybrid", "zetokrag (as used)", "zetokrag, rerank top 50"):
        assert len(m["per_query"][v]["ndcg@10"]) == 130, v
    assert 0 <= m["fusion_recall_by_depth"]["@30"] <= m["fusion_recall_by_depth"]["@100"] <= 1

    # an uninterrupted run from scratch gives the same reranker scores
    import shutil
    for n in names:
        for st in multilingual.settings(n):
            (pipeline / "runs" / n / st / "rerank.npz").unlink()
            (pipeline / "runs" / n / st / "metrics.json").unlink()
    assert multilingual.run_scores(names) == []
    for key, scores in resumed.items():
        assert np.array_equal(np.load(pipeline / "runs" / key[0] / key[1] / "rerank.npz")["scores"], scores), key
    shutil.rmtree(pipeline / "runs" / "synth-en")
    assert multilingual.run_lexical(["synth-en"]) == [] and multilingual.run_scores(["synth-en"]) == []


def test_the_gate1_cross_check_reads_gate1_metrics(pipeline, monkeypatch):
    monkeypatch.setattr(multilingual, "SETS", {"scifact": ("en", "gate1", ("scifact",), "test")})
    d = pipeline / "runs" / "kaggle" / "x" / "runs" / "retrieval" / "scifact" / "as-typed"
    d.mkdir(parents=True)
    d.joinpath("metrics.json").write_text(json.dumps({"queries": ["a", "b"], "per_query": {
        "zetokrag": [{"ndcg@10": 1.0}, {"ndcg@10": 0.5}]}}))
    r = {"set": "scifact", "setting": "as-typed", "queries": ["a", "b"],
         "per_query": {"zetokrag (as used)": {"ndcg@10": [1.0, 0.25]}}}
    assert multilingual.gate1_check([r]) == ["| scifact | as-typed | 75.00 | 62.50 | 1/2 |"]


def test_bge_m3_sparse_weights_match_the_model_card():
    pytest.importorskip("torch")
    try:
        enc = multilingual.BgeM3()
    except Exception as e:                                      # model files not available offline
        pytest.skip(f"BGE-M3 not available: {e}")
    dense, sparse = enc.encode(["What is BGE M3?", "BGE M3 is an embedding model supporting dense retrieval, "
                                "lexical matching and multi-vector interaction."])
    row = sparse.getrow(0)
    weights = {enc.tok.convert_ids_to_tokens(int(i)).lstrip("▁"): float(v) for i, v in zip(row.indices, row.data)}
    card = {"What": 0.08356, "is": 0.0814, "B": 0.1296, "GE": 0.252, "M": 0.1702, "3": 0.2695, "?": 0.04092}
    assert weights.keys() == card.keys()
    assert all(abs(weights[k] - card[k]) < 2e-3 for k in card), weights
    assert abs(float(sparse.getrow(0).multiply(sparse.getrow(1)).sum()) - 0.19554901123046875) < 2e-3
    assert abs(float(np.linalg.norm(dense[0].astype(np.float32))) - 1) < 1e-2


def test_arguana_never_ranks_the_querys_own_text(pipeline, monkeypatch):
    """As in gate 1 (and BEIR): an ArguAna query is itself a document of the corpus and is excluded from every list."""
    pytest.importorskip("torch")                                    # the GPU stage (CI has no PyTorch)
    ids, texts, _, _, _ = _synthetic(n_docs=400, n_queries=0)
    qs = [f"d{i}" for i in range(0, 400, 10)]                      # the query ids are document ids
    qtexts = [texts[i] for i in range(0, 400, 10)]                 # each query is its own document's text
    rels = [{f"d{(i + 1) % 400}": 1} for i in range(0, 400, 10)]
    monkeypatch.setattr(multilingual, "SETS", {"arguana": ("en", "synthetic", (), "test")})
    monkeypatch.setattr(multilingual, "load", lambda name: (ids, texts, qs, qtexts, rels))
    assert multilingual.run_lexical(["arguana"]) == [] and multilingual.run_scores(["arguana"]) == []
    views = multilingual.load_views("arguana", "as-typed")
    own = [ids.index(q) for q in qs]
    for name, lists in views.items():
        assert all(o not in lst[0].tolist() for o, lst in zip(own, lists)), name
        assert all(len(lst[0]) == 100 for lst in lists) or name == "bge-m3-sparse", name
    z = np.load(pipeline / "runs" / "arguana" / "as-typed" / "rerank.npz")
    per_query = np.split(z["ids"], z["offsets"][1:-1])
    assert len(per_query) == len(qs) and all(o not in r.tolist() for o, r in zip(own, per_query))


def test_shards_balance_the_two_gpus_and_cover_every_set():
    names = [n for n in multilingual.SETS if not n.startswith("mldr")]
    assert set(names) == set(multilingual.COST)                     # every set has a measured cost
    a, b = multilingual.shards(names, 2)
    assert sorted(a + b) == sorted(names)
    load = [sum(multilingual.COST[n] for n in g) for g in (a, b)]
    assert abs(load[0] - load[1]) <= max(multilingual.COST.values())
