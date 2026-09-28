"""`find`: the ZetokRAG query over the local index.

BM25 (diacritic-folded only for queries typed without accents) and the dense embedder each propose their
top candidates, min-max fusion (rho) merges them, bge-reranker-v2-m3
reranks the best N_RERANK, and the top k are returned with their source and neighbouring chunks. Before an
excerpt is shown, its file is checked again: if the file changed and the excerpt is no longer in it, the
excerpt is dropped as stale instead of being quoted wrongly.

The verdict "does the index hold an answer?" compares the best reranker score with two thresholds,
calibrated on the user's own question set (eval/personal.py) and the public benchmark (results/retrieval.md).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from peresearch.zetokrag import core
from peresearch.zetokrag.index import Index, embed_text, sha256

RHO, N_FIRST, N_RERANK = 0.5, 100, 30
# Reranker-score thresholds, set after seeing the user's own 85-question set: every unanswerable question scored
# at most 0.16 and every answerable one at least 0.56; on the public corpora 86% of answerable queries reach 0.25.
ENOUGH, PARTIAL = 0.5, 0.25


@dataclass
class Hit:
    id: int
    path: str
    section: str
    unit: str
    start: int
    end: int
    text: str
    score: float
    neighbours: list[str] = field(default_factory=list)
    file_changed: bool = False

    @property
    def where(self) -> str:
        span = f"{self.start}" if self.start == self.end else f"{self.start}-{self.end}"
        prefix = {"line": "L", "page": "p.", "cell": "cell "}[self.unit]
        return f"{self.path}:{prefix}{span}"


def verdict(best: float) -> str:
    return "enough" if best >= ENOUGH else "partial" if best >= PARTIAL else "none"


class Searcher:
    def __init__(self, index: Index, reranker=None):
        self.index = index
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:
            from peresearch.zetokrag.models import Reranker

            self._reranker = Reranker()
        return self._reranker

    def rank(self, query: str) -> tuple[list[tuple], np.ndarray]:
        """The reranked head: chunk rows and reranker scores, best first."""
        bm_ids, bm = self.index.bm25(folded=not core.accented(query))
        dense_ids, vecs = self.index.dense32()
        if bm is None or len(dense_ids) == 0:
            return [], np.zeros(0, dtype=np.float32)
        assert np.array_equal(bm_ids, dense_ids), "index out of sync: run `peresearch index`"
        lexical = bm.scores(query)
        q = self.index.embedder.queries([query])[0].astype(np.float32)
        semantic = vecs @ q
        cand, _ = core.fuse_minmax(lexical, semantic, RHO, N_FIRST)
        rows = self.index.rows(bm_ids[cand[:N_RERANK]])
        scores = self.reranker.scores(query, [embed_text(r[2], r[6]) for r in rows])
        order = np.argsort(-scores, kind="stable")
        return [rows[i] for i in order], scores[order]

    def find(self, query: str, k: int = 5) -> tuple[list[Hit], str, int]:
        """(hits, verdict, number of stale excerpts dropped)."""
        rows, scores = self.rank(query)
        hits, stale, files = [], 0, {}
        for row, score in zip(rows, scores):
            if len(hits) == k or score < PARTIAL:   # below PARTIAL an excerpt is noise, not evidence
                break
            cid, path, section, unit, start, end, text, sha = row
            if path not in files:
                p = Path(path)
                files[path] = None if not p.exists() else (sha256(p), p.read_text(errors="replace")
                                                           if unit == "line" else None)
            current = files[path]
            changed = current is None or current[0] != sha
            present = current is not None and current[1] is not None and all(
                part in current[1] for part in text.split("\n\n"))       # paragraphs may be re-spaced
            if changed and not present:
                stale += 1
                continue
            hits.append(Hit(cid, path, section, unit, start, end, text, float(score),
                            self.neighbours(cid, path, section), changed))
        best = float(scores[0]) if len(scores) else 0.0
        return hits, verdict(best), stale

    def neighbours(self, cid: int, path: str, section: str) -> list[str]:
        """The chunks just before and after, if they belong to the same file and section."""
        rows = self.index.db.execute("SELECT id, section, text FROM chunks WHERE path=? AND id IN (?, ?)",
                                     (path, cid - 1, cid + 1)).fetchall()
        return [r[2] for r in sorted(rows) if r[1] == section]
