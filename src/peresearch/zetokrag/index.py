"""The local index of the declared folders: one SQLite file holds the chunks, their sources and their vectors.

`update` walks the declared folders and only re-reads files whose size or modification time changed (and
only re-chunks them when their content hash changed). Files that disappeared or fell out of the declared
folders are dropped. Chunks and their vectors are written in one transaction, so a reader (another thread or
process) sees both or neither; the database runs in WAL mode, so readers are never blocked by an update.
Paragraphs that look like they hold a credential are withheld: they are never stored, and the report names the
file, not the content.

What is not searched, and why (each is listed in the report, never dropped silently):
- a copy of another file (same content hash) is indexed once, under the first path in sorted order; when that
  file goes, the copy takes its place (five homework folders held the same vendored stb_image.h);
- a data file (a text file whose lines are nearly all URLs or numbers, e.g. IMDB's urls_*.txt) is not chunked;
  the agent can still read and grep it, only retrieval leaves it out;
- a folder with more than MAX_FILES_PER_FOLDER files side by side is a dataset, not notes.
"""

import hashlib
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from peresearch import guard
from peresearch.zetokrag import chunk, parse

VERSION = 2   # 2: vectors moved from dense.npz into the database, chunk ids never reused
SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, sha TEXT, status TEXT,
                                  note TEXT, withheld INTEGER);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, section TEXT, unit TEXT,
                                   start INTEGER, end INTEGER, text TEXT, sha TEXT);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, vec BLOB);   -- id = chunks.id, fp16 bytes
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
OLD_FILES = ("dense.npz", "dense.tmp.npz", "dense.npy", "dense_ids.npy")   # vector files of earlier layouts
EMBEDDER = "qwen3-embedding-0.6b"   # chosen over BGE-M3 by a rule fixed before the benchmark (results/retrieval.md)
MAX_FILES_PER_FOLDER = 500   # more files than this side by side is a dataset (e.g. 12,500 reviews), not notes
DATA_MIN_LINES, DATA_SHARE = 50, 0.9   # a text file of at least 50 lines, 90% of them a URL or numbers alone
_URL_LINE = re.compile(r"^\S+://\S+$")
_NUMBER_LINE = re.compile(r"^[-+.,:;/|eE\d\s]*\d[-+.,:;/|eE\d\s]*$")
_PLAIN_TEXT = parse.TEXT - parse.MARKDOWN      # .txt, .rst…: where data files hide; notes and code are never data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embed_text(section: str, text: str) -> str:
    return f"{section}\n{text}" if section else text


def is_data(doc: parse.Doc) -> bool:
    """A plain-text file whose lines are nearly all a URL or numbers alone: a list of data, not something to read."""
    if doc.path.suffix.lower() not in _PLAIN_TEXT or not doc.blocks or doc.blocks[0].unit != "line":
        return False
    lines = [line.strip() for line in doc.blocks[0].text.splitlines() if line.strip()]
    if len(lines) < DATA_MIN_LINES:
        return False
    return sum(bool(_URL_LINE.match(x) or _NUMBER_LINE.match(x)) for x in lines) >= DATA_SHARE * len(lines)


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


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)   # transactions are explicit
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    return db


class Index:
    def __init__(self, home: Path | None = None, embedder=None):
        self.home = Path(home or guard.home())
        self.home.mkdir(parents=True, exist_ok=True)
        # The TUI searches (a question) and updates (a folder check) from worker threads, possibly at once. One
        # sqlite3 connection used by two threads at once fails ("bad parameter or other API misuse"), so searches
        # read through `db` one at a time (under _lock) and an update writes through a connection of its own:
        # readers then see only committed data, as another process would.
        self.db = _connect(self.home / "index.sqlite")
        self._migrate()
        self._writer = None
        self._write_lock = threading.Lock()            # one update at a time
        self._embedder = embedder
        self._snapshot = None                           # (data_version, ids, BM25, float32 vectors)
        self._lock = threading.RLock()                 # the reading connection and the snapshot

    def _migrate(self) -> None:
        """An index from an earlier layout is emptied (it is rebuilt by the next update) and its files removed."""
        if self.db.execute("PRAGMA user_version").fetchone()[0] < VERSION:
            self.db.executescript("DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS chunks; "
                                  "DROP TABLE IF EXISTS vectors; DROP TABLE IF EXISTS meta;")
            for name in OLD_FILES:
                (self.home / name).unlink(missing_ok=True)
        self.db.executescript(SCHEMA)
        self.db.execute(f"PRAGMA user_version={VERSION}")

    # --- storage ---

    @property
    def embedder(self):
        if self._embedder is None:
            from peresearch.zetokrag.models import Embedder

            # 16 texts at a time: whole chunks reach ~1,100 tokens, and 32 of them peaked at 5.8 of the laptop's
            # 6 GiB; 16 peaked at 3.5 GiB with the same speed (measured 2026-10-02)
            self._embedder = Embedder(EMBEDDER, batch_size=16)
        return self._embedder

    def signature(self) -> str:
        """The embedder the stored vectors must come from (without loading it)."""
        if self._embedder is not None:
            return getattr(self._embedder, "signature", type(self._embedder).__name__)
        from peresearch.zetokrag.models import signature

        return signature(EMBEDDER)

    def read(self, sql: str, params=()) -> list[tuple]:
        """A query on the reading connection, one thread at a time."""
        with self._lock:
            return self.db.execute(sql, params).fetchall()

    def rows(self, ids=None) -> list[tuple]:
        q = "SELECT id, path, section, unit, start, end, text, sha FROM chunks"
        if ids is None:
            return self.read(q + " ORDER BY id")
        got = {r[0]: r for r in self.read(q + f" WHERE id IN ({','.join('?' * len(ids))})", list(map(int, ids)))}
        return [got[int(i)] for i in ids if int(i) in got]

    def dense(self) -> tuple[np.ndarray, np.ndarray]:
        """(chunk ids, fp16 vectors), ids ascending."""
        rows = self.read("SELECT id, vec FROM vectors ORDER BY id")
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float16)
        return (np.array([r[0] for r in rows], dtype=np.int64),
                np.stack([np.frombuffer(r[1], dtype=np.float16) for r in rows]))

    def snapshot(self):
        """(ids, BM25, float32 vectors) over every chunk, read in one transaction so that the three line up, and
        kept until another connection commits (SQLite's data_version changes then), instead of being rebuilt
        for every query. BM25 is None for an empty index."""
        from peresearch.zetokrag.core import BM25

        with self._lock:
            version = self.db.execute("PRAGMA data_version").fetchone()[0]
            if self._snapshot is None or self._snapshot[0] != version:
                self.db.execute("BEGIN")
                try:
                    rows = self.db.execute("SELECT c.id, c.section, c.text, v.vec FROM chunks c "
                                           "JOIN vectors v ON v.id = c.id ORDER BY c.id").fetchall()
                finally:
                    self.db.execute("COMMIT")
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                bm = BM25([embed_text(r[1], r[2]) for r in rows], folded=False) if rows else None
                vecs = (np.stack([np.frombuffer(r[3], dtype=np.float16) for r in rows]).astype(np.float32)
                        if rows else np.zeros((0, 0), dtype=np.float32))
                self._snapshot = (version, ids, bm, vecs)
            return self._snapshot[1:]

    def bm25(self):
        """(ids, BM25) over every chunk, on words as typed."""
        ids, bm, _ = self.snapshot()
        return ids, bm

    def dense32(self) -> tuple[np.ndarray, np.ndarray]:
        ids, _, vecs = self.snapshot()
        return ids, vecs

    # --- update ---

    def _files(self, roots: list[Path], report: Report):
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                keep = []
                for d in sorted(dirnames):
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
                for f in sorted(filenames):
                    p = Path(dirpath) / f
                    if not guard.allowed(p, roots):
                        report.denied += 1
                        continue
                    if parse.kind(p) != "unsupported":
                        yield p

    def update(self, roots: list[Path] | None = None) -> Report:
        with self._write_lock:
            if self._writer is None:
                self._writer = _connect(self.home / "index.sqlite")
            db = self._writer
            db.execute("BEGIN IMMEDIATE")
            try:
                report = self._update(db, roots)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        report.chunks = self.read("SELECT COUNT(*) FROM chunks")[0][0]
        guard.audit("index", added=report.added, changed=report.changed, removed=report.removed,
                    chunks=report.chunks, withheld=sum(report.withheld.values()))
        return report

    @staticmethod
    def _drop(db, key: str) -> None:
        db.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM chunks WHERE path=?)", (key,))
        db.execute("DELETE FROM chunks WHERE path=?", (key,))

    def _update(self, db, roots) -> Report:
        # a folder declared inside another is walked once, with it (the owner's config lists ~/projects and
        # projects inside it, whose files were read twice)
        roots = guard.outermost([Path(r) for r in (guard.roots() if roots is None else roots)])
        report = Report()
        signature = self.signature()
        stored = db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        if stored is None or stored[0] != signature:          # vectors of another embedder cannot be mixed in
            db.execute("DELETE FROM vectors")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (signature,))
        known = {r[0]: r for r in db.execute("SELECT path, size, mtime, sha, status FROM files")}
        found = []
        for path in self._files(roots, report):
            key, st = str(path), path.stat()
            old = known.get(key)
            same = bool(old and old[1] == st.st_size and old[2] == st.st_mtime)
            found.append((path, key, st, old, same, old[3] if same else sha256(path)))
        first = {}                                            # content hash -> the path indexed for it
        for _, key, _, _, _, digest in found:
            first.setdefault(digest, key)
        for path, key, st, old, same, digest in found:
            if first[digest] != key:                          # a copy: indexed once, under the first path
                self._drop(db, key)
                db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?)",
                           (key, st.st_size, st.st_mtime, digest, "duplicate", first[digest], 0))
                report.problems.setdefault("duplicate", []).append(f"{key} (same as {first[digest]})")
                continue
            if old and old[3] == digest and old[4] != "duplicate":
                if not same:
                    db.execute("UPDATE files SET size=?, mtime=? WHERE path=?", (st.st_size, st.st_mtime, key))
                if old[4] == "data":
                    report.problems.setdefault("data_file", []).append(key)
                report.unchanged += 1
                continue
            doc = parse.read(path)
            status, chunks, withheld = doc.status, [], 0
            if status == "ok" and is_data(doc):
                status = "data"
            elif status == "ok":
                safe = lambda section, text: not (guard.find_secrets(text) or guard.find_secrets(section))
                chunks, withheld = chunk.chunk(doc, keep=safe)
            self._drop(db, key)
            db.executemany("INSERT INTO chunks (path, section, unit, start, end, text, sha) VALUES (?,?,?,?,?,?,?)",
                           [(key, c.section, c.unit, c.start, c.end, c.text, digest) for c in chunks])
            db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?)",
                       (key, st.st_size, st.st_mtime, digest, status, doc.note, withheld))
            if status == "data":
                report.problems.setdefault("data_file", []).append(key)
            elif status != "ok":
                report.problems.setdefault(status, []).append(key)
            if withheld:
                report.withheld[key] = withheld
            report.changed += bool(old)
            report.added += not old
        for key in set(known) - {key for _, key, *_ in found}:
            self._drop(db, key)
            db.execute("DELETE FROM files WHERE path=?", (key,))
            report.removed += 1
        self._embed_missing(db)
        return report

    def _embed_missing(self, db, batch: int = 256) -> None:
        """Embed every chunk that has no vector yet, inside the update's transaction."""
        rows = db.execute("SELECT c.id, c.section, c.text FROM chunks c LEFT JOIN vectors v ON v.id = c.id "
                          "WHERE v.id IS NULL ORDER BY c.id").fetchall()
        for a in range(0, len(rows), batch):
            part = rows[a:a + batch]
            vecs = self.embedder.documents([embed_text(r[1], r[2]) for r in part]).astype(np.float16)
            db.executemany("INSERT INTO vectors VALUES (?, ?)", [(r[0], v.tobytes()) for r, v in zip(part, vecs)])
