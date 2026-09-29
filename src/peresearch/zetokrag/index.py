"""The local index of the declared folders: SQLite for chunks and their sources, a matrix of embeddings.

`update` walks the declared folders and only re-reads files whose size or modification time changed (and
only re-chunks them when their content hash changed). Files that disappeared or fell out of the declared
folders are dropped. Another process may search while an update runs: the new chunks are embedded and the
vectors written (one file, replaced atomically) before the chunks are committed, so a reader always sees
vectors for every committed chunk; extra vectors of not-yet-committed chunks are ignored by the search. Paragraphs that look like they hold a credential are withheld: they are never stored, and
the report names the file, not the content.
"""

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from peresearch import guard
from peresearch.zetokrag import chunk, parse

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, sha TEXT, status TEXT,
                                  note TEXT, withheld INTEGER);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, path TEXT, section TEXT, unit TEXT, start INTEGER,
                                   end INTEGER, text TEXT, sha TEXT);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""
EMBEDDER = "qwen3-embedding-0.6b"   # chosen over BGE-M3 by a rule fixed before the benchmark (results/retrieval.md)
MAX_FILES_PER_FOLDER = 500   # more files than this side by side is a dataset (e.g. 12,500 reviews), not notes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embed_text(section: str, text: str) -> str:
    return f"{section}\n{text}" if section else text


@dataclass
class Report:
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    denied: int = 0
    problems: dict[str, list[str]] = field(default_factory=dict)   # status -> paths
    withheld: dict[str, int] = field(default_factory=dict)         # path -> chunks withheld as possible secrets


class Index:
    def __init__(self, home: Path | None = None, embedder=None):
        self.home = Path(home or guard.home())
        self.home.mkdir(parents=True, exist_ok=True)
        # The TUI updates and searches from worker threads, one at a time (never both at once), so the connection
        # may move between threads; SQLite itself is compiled thread-safe.
        self.db = sqlite3.connect(self.home / "index.sqlite", check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._embedder = embedder
        self._bm25 = self._bm25_version = None
        self._dense32 = None

    # --- storage ---

    @property
    def embedder(self):
        if self._embedder is None:
            from peresearch.zetokrag.models import Embedder

            self._embedder = Embedder(EMBEDDER)
        return self._embedder

    def dense(self) -> tuple[np.ndarray, np.ndarray]:
        """(chunk ids, fp16 vectors), ids ascending. Read from one file, so ids and vectors always match."""
        path = self.home / "dense.npz"
        if path.exists():
            with np.load(path) as z:
                return z["ids"], z["vecs"]
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float16)

    def _save_dense(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        """Write to a temporary file, then rename over the old one: a reader sees the old file or the new one."""
        tmp = self.home / "dense.tmp.npz"
        np.savez(tmp, ids=ids, vecs=vecs)
        os.replace(tmp, self.home / "dense.npz")

    def dense32(self) -> tuple[np.ndarray, np.ndarray]:
        """The stored vectors as float32, converted once and kept until the file on disk changes (an update by
        this or another process), instead of reading and converting them for every query."""
        path = self.home / "dense.npz"
        stamp = path.stat().st_mtime_ns if path.exists() else None
        if self._dense32 is None or self._dense32[0] != stamp:
            ids, vecs = self.dense()
            self._dense32 = (stamp, ids, vecs.astype(np.float32))
        return self._dense32[1], self._dense32[2]

    def rows(self, ids=None) -> list[tuple]:
        q = "SELECT id, path, section, unit, start, end, text, sha FROM chunks"
        if ids is None:
            return self.db.execute(q + " ORDER BY id").fetchall()
        got = {r[0]: r for r in self.db.execute(q + f" WHERE id IN ({','.join('?' * len(ids))})", list(map(int, ids)))}
        return [got[int(i)] for i in ids if int(i) in got]

    def bm25(self, folded: bool):
        """(ids, BM25) over all chunks, plain or diacritic-folded; built on first use after an update, whether this
        process made it or another one (SQLite's data_version changes when another connection commits)."""
        version = self.db.execute("PRAGMA data_version").fetchone()[0]
        if self._bm25 is None or self._bm25_version != version:
            self._bm25, self._bm25_version = {}, version
        if folded not in self._bm25:
            from peresearch.zetokrag.core import BM25

            rows = self.db.execute("SELECT id, section, text FROM chunks ORDER BY id").fetchall()
            self._bm25[folded] = (np.array([r[0] for r in rows], dtype=np.int64),
                                  BM25([embed_text(r[1], r[2]) for r in rows], folded=folded) if rows else None)
        return self._bm25[folded]

    # --- update ---

    def _files(self, roots: list[Path], report: Report):
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                keep = []
                for d in dirnames:
                    p = Path(dirpath) / d
                    if d in guard.SKIP_DIRS or os.path.islink(p):
                        continue
                    if guard.denied(p):
                        report.denied += 1
                        continue
                    keep.append(d)
                dirnames[:] = keep
                if len(filenames) > MAX_FILES_PER_FOLDER:     # a dataset, not notes: skipped and reported
                    report.problems.setdefault("large_folder", []).append(f"{dirpath} ({len(filenames)} files)")
                    continue
                for f in filenames:
                    p = Path(dirpath) / f
                    if not guard.allowed(p, roots):
                        report.denied += 1
                        continue
                    if parse.kind(p) != "unsupported":
                        yield p

    def update(self, roots: list[Path] | None = None) -> Report:
        roots = [Path(r) for r in (guard.roots() if roots is None else roots)]
        report = Report()
        known = {r[0]: r for r in self.db.execute("SELECT path, size, mtime, sha, status FROM files")}
        seen = set()
        for path in self._files(roots, report):
            key = str(path)
            seen.add(key)
            st = path.stat()
            old = known.get(key)
            if old and old[1] == st.st_size and old[2] == st.st_mtime:
                report.unchanged += 1
                continue
            digest = sha256(path)
            if old and old[3] == digest:
                self.db.execute("UPDATE files SET size=?, mtime=? WHERE path=?", (st.st_size, st.st_mtime, key))
                report.unchanged += 1
                continue
            doc = parse.read(path)
            safe = lambda section, text: not (guard.find_secrets(text) or guard.find_secrets(section))
            chunks, withheld = chunk.chunk(doc, keep=safe) if doc.status == "ok" else ([], 0)
            self.db.execute("DELETE FROM chunks WHERE path=?", (key,))
            self.db.executemany("INSERT INTO chunks (path, section, unit, start, end, text, sha) VALUES (?,?,?,?,?,?,?)",
                                [(key, c.section, c.unit, c.start, c.end, c.text, digest) for c in chunks])
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?)",
                            (key, st.st_size, st.st_mtime, digest, doc.status, doc.note, withheld))
            if doc.status != "ok":
                report.problems.setdefault(doc.status, []).append(key)
            if withheld:
                report.withheld[key] = withheld
            report.changed += bool(old)
            report.added += not old
        for key in set(known) - seen:
            self.db.execute("DELETE FROM chunks WHERE path=?", (key,))
            self.db.execute("DELETE FROM files WHERE path=?", (key,))
            report.removed += 1
        self._sync_dense()      # vectors first, then the chunks become visible to other processes
        self.db.commit()
        self._bm25 = None
        report.chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        guard.audit("index", added=report.added, changed=report.changed, removed=report.removed,
                    chunks=report.chunks, withheld=sum(report.withheld.values()))
        return report

    def _sync_dense(self) -> None:
        """Embed chunks that have no vector yet and drop vectors of deleted chunks, as this update's uncommitted
        transaction sees them. Readers meanwhile see the old chunks: the few just deleted lose their vectors a
        moment early, which the search tolerates."""
        old_ids, old_vecs = self.dense()
        rows = self.db.execute("SELECT id, section, text FROM chunks ORDER BY id").fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        keep = np.isin(old_ids, ids)
        old_ids, old_vecs = old_ids[keep], old_vecs[keep] if len(old_vecs) else old_vecs
        have = set(old_ids.tolist())
        new = [r for r in rows if r[0] not in have]
        if new:
            vecs = self.embedder.documents([embed_text(r[1], r[2]) for r in new])
            old_ids = np.concatenate([old_ids, [r[0] for r in new]]).astype(np.int64)
            old_vecs = np.concatenate([old_vecs, vecs]) if len(old_vecs) else vecs
        order = np.argsort(old_ids)
        self._save_dense(old_ids[order], old_vecs[order] if len(old_vecs) else old_vecs)
