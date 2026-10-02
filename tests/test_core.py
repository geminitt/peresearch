"""Contracts of zetokrag.core: the BM25 cut, BM25 scores, and the ranking arithmetic (top, min-max, fusion, rerank).

Each contract is one property checked on generated inputs (Hypothesis); a past bug is an `@example` of the
property it broke, not a test of its own.
"""
import os
import unicodedata

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from peresearch.zetokrag import core

# Letters of the six target languages and the scripts the cut treats apart, plus digits, marks and punctuation.
LATIN = "abcdeghiklmnopqrstuvxyABCDEGHIKLMNOPQRSTUVXYđĐàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
HAN, KANA, THAI = "机器学习模型训练方法今天天气", "あいうえおかがきぎアイウエオカガ", "ภาษาไทยดีมากเรียน"
HANGUL, DEVANAGARI = "학습을시작했다한국어", "मशीनलर्निंगक्याहै"
ALPHABET = LATIN + HAN + KANA + THAI + HANGUL + DEVANAGARI + "0123456789_ \n\t.,;:!?()[]-'\"。、！？"
texts = st.text(alphabet=ALPHABET, max_size=80)
UNSPACED = core._UNSPACED_RUN


def graphemes(s: str) -> list[str]:
    return core._GRAPHEME.findall(s)


# --- the cut ------------------------------------------------------------------------------------------------------

@given(texts)
@example("मशीन लर्निंग क्या है")          # `\w+` cut Hindi words at their vowel signs (मशीन -> मश, न)
@example("机器学习模型的训练方法")         # `\w+` kept a whole Chinese sentence as one term
def test_terms_are_lowercase_words_and_ignore_unicode_form(text):
    terms = core.words(text)
    assert all(t and t == t.lower() and core._WORD.fullmatch(t) for t in terms)
    assert core.words(unicodedata.normalize("NFD", text)) == terms
    assert core.words(text.lower()) == terms


@given(st.lists(st.text(alphabet=LATIN + HANGUL + DEVANAGARI + "0123456789", min_size=1, max_size=12), max_size=8))
@example(["Học", "MÁY", "SuperBPE", "và"])
@example(["학습을", "시작했다"])
def test_spaced_scripts_keep_whole_words(ws):
    """English, Vietnamese, Korean and Hindi words are terms as typed (lower-cased), so gate 1's numbers stand."""
    text = " ".join(ws)
    assert core.words(text) == [unicodedata.normalize("NFC", w).lower() for w in ws]


@given(st.text(alphabet=HAN + KANA + THAI, min_size=1, max_size=30))
@example("机器学习")
@example("ภาษาไทย")
@example("学")
def test_unspaced_scripts_are_cut_into_overlapping_pairs(run):
    """A run of n grapheme clusters gives the n - 1 pairs of neighbours (one cluster alone stays itself); a Thai
    vowel stays with its consonant because pairs are of clusters, not code points."""
    g = graphemes(unicodedata.normalize("NFC", run))
    expected = g if len(g) == 1 else [g[i] + g[i + 1] for i in range(len(g) - 1)]
    assert core.words(run) == expected


def test_a_word_mixing_scripts_is_cut_only_where_the_script_is_unspaced():
    assert core.words("Qwen3.6模型") == ["qwen3", "6", "模型"]
    assert core.words("GPT机器学习x") == ["gpt", "机器", "器学", "学习", "x"]
    assert core.words("机器学习model") == ["机器", "器学", "学习", "model"]
    assert core.words("x机器y机器学习z") == ["x", "机器", "y", "机器", "器学", "学习", "z"]


@given(texts)
def test_folding_only_adds_bare_forms(text):
    """The folded cut (kept for the benchmarks) is the plain cut with each word's bare form after it."""
    plain = core.tokenize(text, folded=False)
    assert plain == core.words(text)
    expected = []
    for w in plain:
        expected += [w] + ([core.fold(w)] if core.fold(w) != w else [])
    assert core.tokenize(text) == expected
    assert core.accented(text) == (core.fold(text) != unicodedata.normalize("NFC", text))


def test_fold_drops_vietnamese_diacritics():
    assert core.fold("Đà Nẵng học máy đường") == "Da Nang hoc may duong"
    folded = core.BM25(["học máy", "nấu phở"])                              # the benchmarks' folded BM25
    assert folded.folded and (folded.scores("hoc may") > 0).tolist() == [True, False]


# --- BM25 ---------------------------------------------------------------------------------------------------------

@given(st.lists(texts, min_size=1, max_size=8), texts)
@settings(deadline=None)
def test_bm25_scores_every_text_and_only_shared_terms_count(docs, query):
    bm = core.BM25(docs, folded=False)
    s = bm.scores(query)
    assert s.shape == (len(docs),) and s.dtype == np.float32 and (s >= 0).all()
    q = set(core.words(query))
    for doc, score in zip(docs, s):
        if not q & set(core.words(doc)):
            assert score == 0


def test_bm25_ranks_the_text_holding_the_rare_term_first():
    docs = ["机器学习模型的训练方法", "今天天气很好", "機械学習のモデルを訓練する", "học máy và mô hình"]
    bm = core.BM25(docs, folded=False)
    assert int(np.argmax(bm.scores("机器学习"))) == 0 and int(np.argmax(bm.scores("機械学習"))) == 2
    assert bm.scores("hoc may").max() == 0                   # no folding: words as typed


# --- ranking arithmetic: each checked against a plain reference ---------------------------------------------------

scores = st.lists(st.one_of(st.floats(-5, 5, allow_nan=False, width=32), st.just(-np.inf)), max_size=30)


def ref_top(s, n):
    finite = [i for i in range(len(s)) if s[i] > -np.inf]
    return sorted(finite, key=lambda i: (-s[i], i))[:n]


@given(scores, st.integers(0, 40))
@example([0.5, -np.inf, 0.9, 0.1], 10)                   # an excluded text (-inf) was still ranked
def test_top_is_the_n_best_finite_scores_ties_by_position(s, n):
    assert core.top(np.array(s, dtype=np.float32), n).tolist() == ref_top(np.array(s, dtype=np.float32), n)


@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False, width=32), min_size=1, max_size=30))
def test_minmax_maps_the_range_onto_0_1(v):
    v = np.array(v, dtype=np.float32)
    m = core.minmax(v)
    if v.max() == v.min():
        assert (m == 1).all()
    else:
        assert m.min() == 0 and m.max() == 1 and np.all((m >= 0) & (m <= 1))
        assert m[int(np.argmax(v))] == 1 and m[int(np.argmin(v))] == 0


def ref_minmax_fusion(lex, den, rho, n):
    fused = {}
    for weight, s in ((rho, lex), (1 - rho, den)):
        idx = ref_top(s, n)
        if not idx or (s is lex and s[idx[0]] <= 0):              # a view with nothing to give is left out
            continue
        vals = np.array([s[i] for i in idx], dtype=np.float32)
        norm = np.ones_like(vals) if len(vals) and vals.max() == vals.min() else (vals - vals.min()) / (vals.max() - vals.min())
        for i, x in zip(idx, norm):
            fused[i] = fused.get(i, 0.0) + weight * float(x)
    return fused


pairs = st.integers(1, 25).flatmap(lambda k: st.tuples(
    st.lists(st.floats(0, 10, allow_nan=False, width=32), min_size=k, max_size=k),
    st.lists(st.one_of(st.floats(-1, 1, allow_nan=False, width=32), st.just(-np.inf)), min_size=k, max_size=k)))


@given(pairs, st.sampled_from([0.0, 0.3, 0.5, 1.0]), st.integers(1, 30))
@example(([0.0] * 5, [0.1, 0.2, 0.3, 0.4, 0.5]), 0.5, 3)   # all-zero BM25 got rho for every text (results/multilingual.md)
@example(([0.0], [-np.inf]), 0.0, 1)                        # every dense score excluded: fusion used to raise
def test_minmax_fusion_matches_its_definition(pair, rho, n):
    lex, den = (np.array(x, dtype=np.float32) for x in pair)
    ids, fused = core.fuse_minmax(lex, den, rho, n)
    ref = ref_minmax_fusion(lex, den, rho, n)
    assert sorted(ids.tolist()) == sorted(ref) and len(set(ids.tolist())) == len(ids)
    assert np.allclose([ref[i] for i in ids.tolist()], fused, atol=1e-6)
    assert all(fused[i] >= fused[i + 1] for i in range(len(fused) - 1))


@given(pairs, st.integers(1, 30), st.integers(1, 100))
def test_rrf_matches_its_definition(pair, n, k):
    lex, den = (np.array(x, dtype=np.float32) for x in pair)
    ids, fused = core.fuse_rrf(lex, den, k=k, n=n)
    ref = {}
    for s in (lex, den):
        for rank, i in enumerate(ref_top(s, n), start=1):
            ref[i] = ref.get(i, 0.0) + 1.0 / (k + rank)
    assert sorted(ids.tolist()) == sorted(ref)
    assert np.allclose([ref[i] for i in ids.tolist()], fused, atol=1e-6)
    assert all(fused[i] >= fused[i + 1] for i in range(len(fused) - 1))


@given(st.integers(0, 20).flatmap(lambda n: st.tuples(
    st.permutations(list(range(100, 100 + n))),
    st.integers(0, n).flatmap(lambda h: st.lists(st.floats(0, 1, allow_nan=False, width=32), min_size=h, max_size=h)))))
def test_rerank_reorders_the_head_and_keeps_the_tail(case):
    ids, head = np.array(case[0], dtype=np.int64), np.array(case[1], dtype=np.float32)
    out, s = core.rerank(ids, head)
    h = len(head)
    order = sorted(range(h), key=lambda i: (-head[i], i))
    assert out.tolist() == [int(ids[i]) for i in order] + ids[h:].tolist()
    assert s.tolist() == [float(head[i]) for i in order]


def test_jax_is_kept_off_the_gpu():
    """bm25s imports JAX when it is installed; on a GPU, JAX claimed 75% of the memory at that import."""
    assert os.environ["JAX_PLATFORMS"] == "cpu"
