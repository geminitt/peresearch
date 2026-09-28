import numpy as np

from peresearch.zetokrag import core


def test_tokenize_adds_bare_forms_for_vietnamese():
    assert core.fold("Đà Nẵng học máy") == "Da Nang hoc may"
    assert core.tokenize("Học MÁY") == ["học", "hoc", "máy", "may"]
    assert core.tokenize("Học máy", folded=False) == ["học", "máy"]


def test_bm25_matches_across_diacritics():
    bm = core.BM25(["ghi chú về học máy", "công thức nấu phở", "tokenizer BPE"])
    assert int(np.argmax(bm.scores("hoc may"))) == 0          # typed without accents
    assert bm.scores("hoc may")[1] == 0
    plain = core.BM25(["ghi chú về học máy", "công thức nấu phở"], folded=False)
    assert plain.scores("hoc may").max() == 0                 # plain BM25 misses it


def test_top_and_minmax():
    s = np.array([0.1, 0.9, 0.5, 0.9])
    assert core.top(s, 2).tolist() == [1, 3]
    assert core.minmax(np.array([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert core.minmax(np.array([7.0, 7.0])).tolist() == [1.0, 1.0]


def test_minmax_fusion_weights_and_absent_views():
    lexical = np.array([10.0, 0.0, 5.0, 0.0])
    dense = np.array([0.0, 0.9, 0.1, 0.5])
    ids, scores = core.fuse_minmax(lexical, dense, rho=0.5, n=2)
    # lexical top 2: {0: 1.0, 2: 0.0}; dense top 2: {1: 1.0, 3: 0.0}; each weighted 0.5
    assert dict(zip(ids.tolist(), scores.tolist())) == {0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0}
    ids, _ = core.fuse_minmax(lexical, dense, rho=1.0, n=4)
    assert ids[0] == 0


def test_rrf_and_rerank():
    ids, scores = core.fuse_rrf(np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0]), k=60, n=3)
    assert ids[0] == 1 and np.isclose(scores[0], 1 / 62 + 1 / 61)   # ranks 2 and 1 beat ranks 1 and 3
    order, s = core.rerank(np.array([7, 8, 9, 10]), np.array([0.1, 0.9, 0.5]))
    assert order.tolist() == [8, 9, 7, 10] and s.tolist() == [np.float32(0.9), np.float32(0.5), np.float32(0.1)]


def test_accented_detects_typed_diacritics():
    assert core.accented("học máy") and core.accented("Đà Nẵng")
    assert not core.accented("hoc may") and not core.accented("BPE tokenizer")


def test_jax_is_kept_off_the_gpu():
    # bm25s imports JAX when it is installed; on the GPU, JAX would claim 75% of the memory at that import
    import os

    assert os.environ["JAX_PLATFORMS"] == "cpu"
