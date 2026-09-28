"""The retrieval core of ZetokRAG, the same code for a folder of notes and for a benchmark corpus.

Two first-stage views score every text: BM25 and a dense embedding. BM25 matches words as typed, except when
the query itself has no diacritics ("hoc may"): then it matches on diacritic-free forms, so the notes written
with accents are still found. (Matching every query on bare forms merges too many Vietnamese words: ma, má,
mà, mả, mã, mạ; the benchmark in results/variants.md chose this "auto" rule.)
Each view keeps its top candidates, min-max normalizes their scores (a text missing from a view gets 0),
and the two are mixed with weight rho, the fusion Zero-Mem uses. A cross-encoder then reranks the best
fused candidates. No step calls an LLM, hence "zero-token".
"""

import os
import re
import unicodedata

import numpy as np

# bm25s imports JAX when it is installed (Kaggle's image has it) and runs one JAX op at import; JAX then claims
# 75% of the GPU for itself, and the embedders run out of memory. Nothing here needs JAX on the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_WORD = re.compile(r"\w+")


def fold(text: str) -> str:
    """Drop Vietnamese diacritics: 'học máy' -> 'hoc may', 'đ' -> 'd'."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str, folded: bool = True) -> list[str]:
    """Lower-cased words; with `folded`, every word with diacritics also yields its bare form, so a query
    typed without accents ('hoc may') still meets a note written with them ('học máy') and vice versa."""
    words = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    if not folded:
        return words
    out = []
    for w in words:
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
    """rho * lexical + (1 - rho) * dense over each view's top n, both min-max normalized over that top n."""
    fused: dict[int, float] = {}
    for weight, scores in ((rho, lexical), (1 - rho, dense)):
        idx = top(scores, n)
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
