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


def test_excluded_texts_never_reach_a_ranking():
    s = np.array([0.5, -np.inf, 0.9, 0.1], dtype=np.float32)
    assert core.top(s, 10).tolist() == [2, 0, 3]
    ids, fused = core.fuse_minmax(s, np.array([0.2, -np.inf, 0.3, 0.8], dtype=np.float32), 0.5, 10)
    assert 1 not in ids and np.isfinite(fused).all()
    assert len(core.top(np.full(3, -np.inf), 5)) == 0


# --- the multilingual cut (2026-10-01, results/multilingual.md) ---------------------------------------------------

def test_bm25_finds_chinese_and_japanese_words_inside_unspaced_text():
    """`\\w+` kept a whole Chinese or Japanese sentence as one term, so BM25 matched nothing for 60-97% of the
    Chinese and Japanese queries of MIRACL, DuRetrieval, CmedqaRetrieval and JMTEB."""
    docs = ["机器学习模型的训练方法", "今天天气很好", "機械学習のモデルを訓練する", "học máy và mô hình"]
    bm = core.BM25(docs)
    zh, ja = bm.scores("机器学习"), bm.scores("機械学習")
    assert zh.max() > 0 and int(np.argmax(zh)) == 0
    assert ja.max() > 0 and int(np.argmax(ja)) == 2


def test_words_keep_their_vowel_signs():
    """Hindi vowel signs are combining marks: `\\w+` cut every word at them (मशीन -> मश, न)."""
    assert core.tokenize("मशीन लर्निंग क्या है", folded=False) == ["मशीन", "लर्निंग", "क्या", "है"]


def test_spaced_text_is_cut_as_before():
    """English, Vietnamese and Korean words are unchanged, so gate 1's numbers stand."""
    assert core.tokenize("Học MÁY, SuperBPE và Qwen3.6-35B-A3B-FP8") == \
        ["học", "hoc", "máy", "may", "superbpe", "và", "va", "qwen3", "6", "35b", "a3b", "fp8"]
    assert core.tokenize("학습을 시작했다", folded=False) == ["학습을", "시작했다"]
    assert core.tokenize("hoc may la gi", folded=False) == ["hoc", "may", "la", "gi"]


def test_fusion_ignores_a_lexical_view_that_matched_nothing():
    """All-zero BM25 scores were min-max normalized to 1, so 100 documents BM25 had not matched each got rho and
    pushed the dense ranking down (NLP Journal, Japanese: 25.1 nDCG@10 against 94.9 without them)."""
    dense = np.linspace(0, 1, 500).astype(np.float32)
    ids, _ = core.fuse_minmax(np.zeros(500, np.float32), dense, rho=0.5, n=100)
    assert ids[:10].tolist() == list(range(499, 489, -1))
    lexical = np.zeros(500, np.float32)
    lexical[3] = 2.0                                     # one match still counts
    ids, _ = core.fuse_minmax(lexical, dense, rho=0.5, n=100)
    assert 3 in ids[:5].tolist()
