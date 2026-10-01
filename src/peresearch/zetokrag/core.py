"""The retrieval core of ZetokRAG, the same code for a folder of notes and for a benchmark corpus.

Two first-stage views score every text: BM25 and a dense embedding. ZetokRAG's BM25 matches words as typed
(owner, 2026-10-02: notes and questions typed without Vietnamese diacritics are rare, so the folded index and
the rule choosing it by the query's diacritics were removed). `fold`, `accented` and `BM25(folded=True)` stay
for the benchmarks that measured that rule (results/retrieval.md, results/multilingual.md).
Each view keeps its top candidates, min-max normalizes their scores (a text missing from a view gets 0),
and the two are mixed with weight rho, the fusion Zero-Mem uses. A cross-encoder then reranks the best
fused candidates. No step calls an LLM, hence "zero-token".
"""

import os
import unicodedata

import numpy as np
import regex

# bm25s imports JAX when it is installed (Kaggle's image has it) and runs one JAX op at import; JAX then claims
# 75% of the GPU for itself, and the embedders run out of memory. Nothing here needs JAX on the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# A word: letters, combining marks (Hindi and Thai vowel signs belong to their word; `\w+` cut words at them) and
# digits. A stretch in a script written without spaces (Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar) is cut
# further by `_cut_unspaced`; spaced scripts (Latin, Hangul, Devanagari…) keep their words as before. Measured on six
# languages in results/multilingual.md (Chinese 8.0 → 63.7, Japanese 27.6 → 83.8 nDCG@10; English, Vietnamese,
# Korean unchanged).
_WORD = regex.compile(r"[\p{L}\p{M}\p{N}_]+")
_UNSPACED = (r"\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Thai}\p{scx=Lao}\p{scx=Khmer}\p{scx=Myanmar}")
_UNSPACED_RUN = regex.compile(rf"((?:[{_UNSPACED}]\p{{M}}*)+)")
_GRAPHEME = regex.compile(r"\X")


def _cut_unspaced(run: str) -> list[str]:
    """Overlapping pairs of characters (grapheme clusters, so a Thai vowel stays with its consonant)."""
    g = _GRAPHEME.findall(run)
    return g if len(g) == 1 else [g[i] + g[i + 1] for i in range(len(g) - 1)]


def words(text: str) -> list[str]:
    """Lower-cased words; each stretch of an unspaced script inside a word is cut by `_cut_unspaced`."""
    out = []
    for word in _WORD.findall(unicodedata.normalize("NFC", text).lower()):
        for k, piece in enumerate(_UNSPACED_RUN.split(word)):
            if piece:
                out.extend(_cut_unspaced(piece) if k % 2 else [piece])
    return out


def fold(text: str) -> str:
    """Drop Vietnamese diacritics: 'học máy' -> 'hoc may', 'đ' -> 'd'."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str, folded: bool = True) -> list[str]:
    """Lower-cased words; with `folded`, every word with diacritics also yields its bare form, so a query
    typed without accents ('hoc may') still meets a note written with them ('học máy') and vice versa."""
    terms = words(text)
    if not folded:
        return terms
    out = []
    for w in terms:
        out.append(w)
        bare = fold(w)
        if bare != w:
            out.append(bare)
    return out


def accented(text: str) -> bool:
    """True when the text carries diacritics (typed with Vietnamese accents)."""
    return fold(text) != unicodedata.normalize("NFC", text)


class BM25:
    """Okapi BM25 (bm25s) over `tokenize`; `scores` returns one score per text."""

    def __init__(self, texts: list[str], folded: bool = True):
        import bm25s

        self.folded = folded
        self.model = bm25s.BM25()
        self.model.index([tokenize(t, folded) or ["∅"] for t in texts], show_progress=False)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.model.get_scores(tokenize(query, self.folded)), dtype=np.float32)


def top(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest scores, best first; a score of -inf marks an excluded text, never returned."""
    n = min(n, int((scores > -np.inf).sum()))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")]


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return np.ones_like(values) if hi == lo else (values - lo) / (hi - lo)


def fuse_minmax(lexical: np.ndarray, dense: np.ndarray, rho: float = 0.5, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """rho * lexical + (1 - rho) * dense over each view's top n, both min-max normalized over that top n.

    A lexical view that matched nothing (best score 0) is left out: min-max maps equal scores to 1, so 100 documents
    BM25 never matched used to get rho each and push the dense ranking down (results/multilingual.md)."""
    fused: dict[int, float] = {}
    for weight, scores in ((rho, lexical), (1 - rho, dense)):
        idx = top(scores, n)
        if scores is lexical and (len(idx) == 0 or float(scores[idx[0]]) <= 0):
            continue
        for i, s in zip(idx, minmax(scores[idx])):
            fused[int(i)] = fused.get(int(i), 0.0) + weight * float(s)
    ids = np.array(sorted(fused, key=lambda i: -fused[i]))
    return ids, np.array([fused[i] for i in ids], dtype=np.float32)


def fuse_rrf(lexical: np.ndarray, dense: np.ndarray, k: int = 60, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal rank fusion, sum of 1 / (k + rank) over the views (the usual baseline, k = 60)."""
    fused: dict[int, float] = {}
    for scores in (lexical, dense):
        for rank, i in enumerate(top(scores, n), start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (k + rank)
    ids = np.array(sorted(fused, key=lambda i: -fused[i]))
    return ids, np.array([fused[i] for i in ids], dtype=np.float32)


def rerank(ids: np.ndarray, rerank_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder the reranked head by the cross-encoder; `ids` beyond the head keep their order after it."""
    head = len(rerank_scores)
    order = np.argsort(-rerank_scores, kind="stable")
    return np.concatenate([ids[:head][order], ids[head:]]), rerank_scores[order]
