"""`find`: the ZetokRAG query over the local index.

BM25 (on words as typed) and the dense embedder each propose their top candidates, min-max fusion (rho) merges
them, bge-reranker-v2-m3 reranks the best N_RERANK (or the first PERESEARCH_RERANK_DEPTH of them, below), and they
are returned best first, with their scores, sources and neighbouring chunks: all of them to the agent (which shows
the model what fits its context and keeps the rest to read on), a page of k at a time to `peresearch find`. Before
an excerpt is shown, its file is checked again: if the file changed and the excerpt is no longer in it, the excerpt
is dropped as stale instead of being quoted wrongly.

No score decides for the model (decided 2026-10-04, measured on the gold dev split): there is no verdict and no
threshold. The best reranker score told answerable questions from unanswerable ones no better than chance (AUROC
0.56 [0.37, 0.75]; 11 of 12 unanswerable questions scored over the old 0.25), and the old cut (top 5, stop below
0.25) showed the model 0.60 of the answering passages against 0.71 in the top 10 and 0.78 in the top 30.

Rerank depth (owner's choice, 2026-10-04): the reranker reorders the first PERESEARCH_RERANK_DEPTH (default all 30)
fused candidates; the rest follow in fused order, without a score. 30 on every machine by default, because fewer
costs quality (nDCG@10 against 30 on the gold dev split, paired bootstrap 95%: 10 gives -0.066 [-0.104, -0.032],
0 gives -0.133 [-0.187, -0.080]); a machine without a GPU may trade it for time (reranker time per search on an
Intel Core Ultra 7 165H CPU, median of 3 gold questions: 38 s for 30, 17 s for 10), and `peresearch setup` shows it
there (runs/gold/cpu-reranker-2026-10-04.py).
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from peresearch import settings
from peresearch.zetokrag import core, parse
from peresearch.zetokrag.index import FROM_SETTINGS, Index, sha256

RHO, N_FIRST, N_RERANK = 0.5, 100, 30
PAGE = 10                    # candidates a page of `peresearch find`; the reranked N_RERANK make 3 pages
# A near-copy (README.template beside README, a notebook beside its .txt source: 4.9% of the top-10 places on the gold
# set, 2026-10-04) is folded into the better-ranked candidate: Jaccard of 8-character shingles (any script) at least
COPY_JACCARD, SHINGLE = 0.8, 8
OUTLINE_HEADINGS = 8         # headings an outline names before "… and N more"


def rerank_depth(value: str | int | None = None) -> int:
    """How many of the N_RERANK fused candidates the reranker reorders: `value`, else the PERESEARCH_RERANK_DEPTH
    setting, else N_RERANK."""
    raw = (settings.get("PERESEARCH_RERANK_DEPTH") or str(N_RERANK)) if value is None else value
    try:
        depth = int(raw)
    except ValueError:
        depth = -1
    if not 0 <= depth <= N_RERANK:
        raise ValueError(f"PERESEARCH_RERANK_DEPTH must be a whole number from 0 to {N_RERANK} (got {str(raw)!r})")
    return depth


def _shingles(text: str) -> set[str]:
    flat = " ".join(text.casefold().split())
    return {flat[i:i + SHINGLE] for i in range(max(1, len(flat) - SHINGLE + 1))}


def _alike(x: set, y: set) -> bool:
    return len(x & y) >= COPY_JACCARD * len(x | y)


def near_copy(a: str, b: str) -> bool:
    return _alike(_shingles(a), _shingles(b))


def place(path: str, unit: str, start: int, end: int) -> str:
    """How a citation names a span: path:L3 or path:L3-5, path:p.2, path:cell 4."""
    span = f"{start}" if start == end else f"{start}-{end}"
    return f"{path}:{ {'line': 'L', 'page': 'p.', 'cell': 'cell '}[unit] }{span}"


@dataclass
class Context:
    """A chunk just before or after a hit, in the same file and section."""
    start: int
    end: int
    text: str


@dataclass
class Hit:
    id: int
    path: str
    section: str
    unit: str
    start: int
    end: int
    text: str
    score: float | None      # None: past the rerank depth, in fused order
    before: Context | None = None
    after: Context | None = None
    file_changed: bool = False
    copies: list[str] = field(default_factory=list)     # where its near-copies are (folded into it)

    def _place(self, start: int, end: int) -> str:
        return place(self.path, self.unit, start, end)

    @property
    def where(self) -> str:
        """The hit's own place."""
        return self._place(self.start, self.end)

    @property
    def span(self) -> str:
        """The place of the hit with its context."""
        return self._place(self.before.start if self.before else self.start, self.after.end if self.after else self.end)


@dataclass
class Document:
    """A file as overview ranks it: its best fused chunk (its place and text) and how many of its chunks are
    candidates."""
    path: str
    section: str
    unit: str
    start: int
    end: int
    text: str
    sha: str
    score: float | None      # None: past the rerank depth, in fused order
    matches: int
    copies: list[str] = field(default_factory=list)     # files folded into it: their best chunk a near-copy of its

    @property
    def where(self) -> str:
        return place(self.path, self.unit, self.start, self.end)


def _fused(lexical: np.ndarray, dense: np.ndarray | None) -> np.ndarray:
    """The N_FIRST best candidates: BM25 and meaning fused (min-max, RHO); with no meaning, BM25 alone (a chunk no
    query word is in is no candidate)."""
    if dense is None:
        order = np.argsort(-lexical, kind="stable")
        return order[lexical[order] > 0][:N_FIRST]
    return core.fuse_minmax(lexical, dense, RHO, N_FIRST)[0]


class Searcher:
    def __init__(self, index: Index, reranker=FROM_SETTINGS, depth: int | None = None):
        """`reranker`: the user's rerank endpoint by default (PERESEARCH_RERANK_URL…), or one given; None: the fused
        order, without a relevance score."""
        self.index = index
        if reranker is FROM_SETTINGS:
            from peresearch.zetokrag.embed import from_settings

            reranker = from_settings()[1]
        self.reranker = reranker
        self.depth = rerank_depth(depth)                # a bad setting is said, a rerank endpoint set or not
        if reranker is None:
            self.depth = 0
        self.words_only = ""                    # why the last search ranked by words alone ("": it used meaning too)
        self.unranked = ""                      # why the last search's rerank failed ("": it did not, or none is set)
        self.allowed = None                    # optional live access policy, applied before external reranking

    def _scores(self, query: str, texts: list[str]) -> np.ndarray | None:
        """The reranker's scores, or None when its endpoint failed (unranked says so)."""
        from peresearch.zetokrag.embed import EndpointError

        try:
            got = np.asarray(self.reranker.scores(query, texts), dtype=np.float32)
        except EndpointError as e:
            self.unranked = f"the rerank endpoint did not answer ({e})"
            return None
        self.unranked = ""
        return got

    def _dense(self, query: str, vecs) -> np.ndarray | None:
        """The query's similarity to every chunk (a chunk without a vector yet: the lowest), or None — then the search
        ranks by words alone, and words_only says why."""
        if vecs is None or self.index.embedder is None:
            stored = self.index.read("SELECT value FROM meta WHERE key='embedder'")
            self.words_only = ("no embedding endpoint is set" if self.index.embedder is None else
                               "the stored vectors are another embedder's; the next index update makes them again"
                               if stored and stored[0][0] != self.index.signature() else "no passage has a vector yet")
            return None
        from peresearch.zetokrag.embed import EndpointError

        if not np.isfinite(vecs).all(axis=1).any():
            self.words_only = "no permitted passage has a vector yet"
            return None
        try:
            q = self.index.embedder.queries([query])[0].astype(np.float32)
        except EndpointError as e:
            self.words_only = f"the embedding endpoint did not answer ({e})"
            return None
        if q.shape[0] != vecs.shape[1]:          # another model behind the same name: Index.embed makes them again
            self.words_only = "the stored vectors are another model's; the next index update makes them again"
            return None
        dense = vecs @ q
        have = ~np.isnan(dense)
        if not have.any():
            self.words_only = "the embedding endpoint returned no finite similarities"
            return None
        dense[~have] = dense[have].min()
        self.words_only = ""
        return dense

    def rank(self, query: str) -> tuple[list[tuple], np.ndarray]:
        """The N_RERANK best fused candidates: the first `depth` reordered by the reranker, best first, with their
        scores; the rest after them in fused order, with NaN scores."""
        ids, bm, vecs = self.index.snapshot()          # one read: ids, BM25 and vectors line up
        if bm is None:
            return [], np.zeros(0, dtype=np.float32)
        lexical = bm.scores(query)
        ids, lexical, vecs = self._permitted(ids, lexical, vecs)
        if not len(ids):
            return [], np.zeros(0, dtype=np.float32)
        cand = _fused(lexical, self._dense(query, vecs))
        rows = self.index.rows(ids[cand[:N_RERANK]])   # fewer if an update dropped some meanwhile
        if self.allowed is not None:
            rows = [r for r in rows if self.allowed(r[1])]
        head, tail = rows[:self.depth], rows[self.depth:]
        if not head:                                    # depth 0: the reranker is not even loaded
            return tail, np.full(len(tail), np.nan, dtype=np.float32)
        scores = self._scores(query, [self.index.ranking_text(r) for r in head])
        if scores is None:                              # the endpoint failed: the fused order, said
            return head + tail, np.full(len(head) + len(tail), np.nan, dtype=np.float32)
        assert len(scores) == len(head), f"{len(scores)} reranker scores for {len(head)} chunks"
        # shown to the model as a relevance from 0 to 1: a reranker giving raw logits would mislead it
        assert bool(np.all((scores >= 0) & (scores <= 1))), \
            f"reranker scores outside [0, 1] (from {scores.min():.3f} to {scores.max():.3f})"
        order = np.argsort(-scores, kind="stable")
        return [head[i] for i in order] + tail, np.concatenate([scores[order], np.full(len(tail), np.nan, np.float32)])

    def _permitted(self, ids, lexical, vecs):
        """Keep the live scope before fusion, so removed files neither leave nor crowd out permitted evidence."""
        if self.allowed is None:
            return ids, lexical, vecs
        paths = dict(self.index.read("SELECT id, path FROM chunks"))
        sel = np.array([i for i, cid in enumerate(ids) if int(cid) in paths and self.allowed(paths[int(cid)])],
                       dtype=np.int64)
        return ids[sel], lexical[sel], None if vecs is None else vecs[sel]

    def documents(self, query: str, folder: str | None = None) -> list[Document]:
        """Files ranked for an overview question (runs/gold/overview-documents-2026-10-06.py: ranking files rather
        than excerpts found +0.11 [+0.02, +0.21] more of the relevant files in 20 lines): the search's own fusion of
        every chunk (inside `folder` when given, kept before the fusion), each file once, standing for it its best
        chunk; the first `depth` reordered by the reranker on that chunk, the rest in fused order, without a score; a
        file whose best chunk is a near-copy of a better-ranked file's folded into it (+0.04 [+0.01, +0.10] more,
        none worse: the -folded.log beside the script)."""
        ids, bm, vecs = self.index.snapshot()
        if bm is None:
            return []
        lexical = bm.scores(query)
        ids, lexical, vecs = self._permitted(ids, lexical, vecs)
        if not len(ids):
            return []
        dense = self._dense(query, vecs)
        if folder is not None:
            inside = folder.rstrip("/") + "/"
            paths = dict(self.index.read("SELECT id, path FROM chunks"))
            sel = np.array([i for i, c in enumerate(ids) if paths.get(int(c), "").startswith(inside)], dtype=np.int64)
            if len(sel) == 0:
                return []
            ids, lexical, dense = ids[sel], lexical[sel], None if dense is None else dense[sel]
        cand = _fused(lexical, dense)
        order: list[str] = []
        best: dict[str, tuple] = {}
        matches: dict[str, int] = {}
        for row in self.index.rows(ids[cand]):              # best first: a file's first row is its best chunk
            path = row[1]
            if self.allowed is not None and not self.allowed(path):
                continue
            matches[path] = matches.get(path, 0) + 1
            if path not in best:
                best[path] = row
                order.append(path)
        head, tail = order[:self.depth], order[self.depth:]
        scores: dict[str, float] = {}
        got = None
        if head:
            got = self._scores(query, [self.index.ranking_text(best[p]) for p in head])
        if head and got is not None:
            assert len(got) == len(head), f"{len(got)} reranker scores for {len(head)} documents"
            assert bool(np.all((got >= 0) & (got <= 1))), \
                f"reranker scores outside [0, 1] (from {got.min():.3f} to {got.max():.3f})"
            scores = {p: float(x) for p, x in zip(head, got)}
            head = [head[i] for i in np.argsort(-got, kind="stable")]
        docs, seen = [], []
        for p in head + tail:            # a near-copy folded into the better-ranked file, as candidates folds excerpts
            mine = _shingles(best[p][6])
            twin = next((d for d, other in zip(docs, seen) if _alike(mine, other)), None)
            if twin is not None:
                twin.copies.append(p)
                continue
            docs.append(Document(p, best[p][2], best[p][3], best[p][4], best[p][5], best[p][6], best[p][7],
                                 scores.get(p), matches[p]))
            seen.append(mine)
        assert len({d.path for d in docs}) == len(docs) and all(
            folder is None or d.path.startswith(folder.rstrip("/") + "/") for d in docs), "a document twice or outside"
        return docs

    def outline(self, path: str) -> tuple[list[str], str]:
        """(headings, first line): the last part of each heading path in the file's order, each once (at most
        OUTLINE_HEADINGS, then "… and N more"), and its first line of text that is not a heading."""
        heads: list[str] = []
        lead = ""
        for section, text in self.index.read("SELECT section, text FROM chunks WHERE path=? ORDER BY id", (path,)):
            leaf = section.rsplit(" > ", 1)[-1].strip()
            if leaf and leaf not in heads:
                heads.append(leaf)
            if not lead:
                lead = next((ln.strip() for ln in text.split("\n") if ln.strip() and not ln.lstrip().startswith("#")),
                            "")
        more = len(heads) - OUTLINE_HEADINGS
        return heads[:OUTLINE_HEADINGS] + ([f"… and {more} more"] if more > 0 else []), lead

    def candidates(self, query: str) -> tuple[list[tuple], np.ndarray, dict[int, list[tuple]]]:
        """The reranked head with each near-copy folded into the better-ranked candidate: (rows, scores, the rows of
        the copies folded into the candidate at each position)."""
        rows, scores = self.rank(query)
        kept, shingles, copies = [], [], {}
        for i, row in enumerate(rows):
            mine = _shingles(row[6])
            twin = next((j for j, other in enumerate(shingles) if _alike(mine, other)), None)
            if twin is None:
                kept.append(i)
                shingles.append(mine)
            else:
                copies.setdefault(twin, []).append(row)
        assert len(kept) + sum(map(len, copies.values())) == len(rows), "a candidate lost when folding copies"
        return [rows[i] for i in kept], scores[kept] if len(kept) else scores[:0], copies

    def find(self, query: str, k: int | None = PAGE, page: int = 1) -> tuple[list[Hit], int, int]:
        """(hits, stale excerpts dropped, ranked candidates in all): page `page` of the ranked candidates, k a page,
        best first; k None: all of them. A stale excerpt is dropped from its page, not replaced, so that each
        candidate has one page."""
        if (k is not None and k < 1) or page < 1:
            raise ValueError(f"k and page start at 1 (got k={k}, page={page})")
        rows, scores, copies = self.candidates(query)
        k, page = (len(rows) or 1, 1) if k is None else (k, page)
        hits, stale, files = [], 0, {}
        def current_file(path, unit):
            if path not in files:
                p = Path(path)
                try:
                    if unit == "line" and p.stat().st_size > parse.MAX_BYTES:
                        files[path] = None
                        return None
                    digest = sha256(p)
                    if unit == "line":
                        # A file can grow after stat/hash; bound the actual allocation too.
                        with p.open("rb") as stream:
                            raw = stream.read(parse.MAX_BYTES + 1)
                        if len(raw) > parse.MAX_BYTES:
                            files[path] = None
                            return None
                        # Hash and decoded evidence must describe the same read, even if a writer raced the hash.
                        digest = hashlib.sha256(raw).hexdigest()
                        files[path] = (digest, parse.decode(raw)[0].replace("\r\n", "\n"))
                    else:
                        files[path] = (digest, None)
                except OSError:
                    files[path] = None
            return files[path]

        for at, (row, score) in list(enumerate(zip(rows, scores)))[(page - 1) * k:page * k]:
            cid, path, section, unit, start, end, text, sha, _ = row
            current = current_file(path, unit)
            changed = current is None or current[0] != sha
            if changed:
                located = self._locate(text, current[1]) if current is not None and current[1] is not None else None
                if located is None:
                    stale += 1
                    continue
                start, end, text = located
                section = ""                      # a former heading path may no longer describe this passage
            verified_copies = []
            for c in copies.get(at, []):
                if self.allowed is not None and not self.allowed(c[1]):
                    continue
                other = current_file(c[1], c[3])
                if other is None:
                    continue
                bounds = (c[4], c[5]) if other[0] == c[7] else self._locate(c[6], other[1]) if other[1] is not None else None
                if bounds is not None:
                    verified_copies.append(place(c[1], c[3], bounds[0], bounds[1]))
            hits.append(Hit(cid, path, section, unit, start, end, text, None if np.isnan(score) else float(score),
                            *((None, None) if changed else self.context(cid, path, section)), changed,
                            verified_copies))
        assert len(hits) <= k and all(b.score is None or (a.score is not None and a.score >= b.score)
                                      for a, b in zip(hits, hits[1:])), "hits beyond k or out of order"
        return hits, stale, len(rows)

    @staticmethod
    def _locate(text: str, current: str) -> tuple[int, int, str] | None:
        """An unambiguous verbatim span in the current file; only blank paragraph spacing may differ.
        Separate paragraphs must still occur together and in order, never assembled from unrelated locations."""
        pattern = r"\n(?:[ \t]*\n)+".join(re.escape(part) for part in text.split("\n\n"))
        matches = list(re.finditer(pattern, current))
        if len(matches) != 1:
            return None
        match = matches[0]
        start = current.count("\n", 0, match.start()) + 1
        end = start + match.group().count("\n")
        return start, end, match.group()

    def context(self, cid: int, path: str, section: str) -> tuple[Context | None, Context | None]:
        """(before, after): the chunks just before and after, each if it belongs to the same file and section (a
        file's chunks get consecutive ids)."""
        rows = {r[0]: r for r in self.index.read(
            "SELECT id, section, start, end, text FROM chunks WHERE path=? AND id IN (?, ?)", (path, cid - 1, cid + 1))}
        before, after = (Context(*rows[i][2:]) if i in rows and rows[i][1] == section else None
                         for i in (cid - 1, cid + 1))
        return before, after
