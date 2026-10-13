"""The local index of the declared folders: one SQLite file holds the chunks, their sources and their vectors.

`update` walks the declared folders and only re-reads files whose size or modification time changed (and
only re-chunks them when their content hash changed). Files that disappeared or fell out of the declared
folders are dropped. An update commits as it goes (every COMMIT_SECONDS), each commit whole: a file's chunks with
their vectors and its extracted text, so a reader (another thread or process) sees each file as it was or as it
is; a stopped update keeps what it committed and the next one goes on. The database runs in WAL mode, so readers
are never blocked by an update.
Paragraphs that look like they hold a credential are withheld: they are never stored, and the report names the
file, not the content.

The chunking (chunk.DEFAULT) and the ranking context (CONTEXT) are named in the index: when one changes, every file
is read and cut again; until it is reached, a file keeps its old chunks (searchable), and once the update ends no
two chunkings are mixed.

What is not searched, and why (each is listed in the report, never dropped silently):
- a copy of another file (same content hash) is indexed once, under the first path in sorted order; when that
  file goes, the copy takes its place (five homework folders held the same vendored stb_image.h);
- a data file (a plain-text file the data filter calls a log, a table, a word list, JSON, numbers and codes, or
  URLs or numbers alone; datafilter.py) is not chunked, its reason kept; the agent can still read and grep it, only
  retrieval leaves it out; when the filter's VERSION changes, plain text and code are judged again;
- what files.py leaves out (folders a tool made, hidden names, symlinked folders, large folders, the user's patterns)
  and files of a type the parser does not read are not walked or read.
Everything left out is kept in the table `skipped` with its reason (a folder once, not each file in it), so
`why_not` can say why a path is missing; a protected path (guard.DENY) is only counted, never named.

The text of PDFs, notebooks and text files not in UTF-8 is also kept in the extracted-text store (extracted.py),
in step with the index, for plain-text search (`extracted(path)`).
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from peresearch import files, guard
from peresearch.zetokrag import chunk, datafilter, extracted, parse

VERSION = 2   # 2: vectors moved from dense.npz into the database, chunk ids never reused
SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, sha TEXT, status TEXT,
                                  note TEXT, withheld INTEGER, reason TEXT);   -- reason: what a data file is
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, section TEXT, unit TEXT,
                                   start INTEGER, end INTEGER, text TEXT, sha TEXT, place TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, vec BLOB);   -- id = chunks.id, fp16 bytes
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS skipped (path TEXT PRIMARY KEY, reason TEXT, note TEXT);   -- rewritten by every update
CREATE TABLE IF NOT EXISTS texts (sha TEXT PRIMARY KEY, units TEXT);   -- the extracted-text store, JSON line map
"""
OLD_FILES = ("dense.npz", "dense.tmp.npz", "dense.npy", "dense_ids.npy")   # vector files of earlier layouts
COMMIT_SECONDS = 60          # an update commits at least this often: a stopped one loses about this much work at most
MARKED = -1.0                # files.mtime of a file to read and cut again (a new chunking, context or embedder)
EMBED_BATCH = 64             # chunks a request to the embedding endpoint, and a commit of their vectors
FROM_SETTINGS = object()     # Index's default embedder: the user's endpoint (embed.from_settings), or none: lexical
LARGE_FILE_CHUNKS, LARGEST = 100, 5   # a file this big is named in the report (a 50-page PDF makes about 70)
EMBEDDER = "qwen3-embedding-0.6b"   # the test tools' local embedder (zetokrag/models.py; eval), never the product's
# What is put before a chunk's text when it is ranked (BM25, embedder, reranker): "section", its heading path; or
# "place", the file's place (its declared folder's name and the path inside it) and then the heading path. Kept in
# the index (meta.context): a change re-reads and re-embeds every file. "place" chosen 2026-10-03 (owner: "nếu place là
# tốt nhất thì dùng nó"): on the gold dev split nDCG@10 0.650 against 0.626 for "section" on the same chunking,
# +0.024 [-0.003, +0.050] paired bootstrap 95%, recall@30 -0.007 [-0.030, +0.015].
CONTEXTS = ("section", "place")
CONTEXT = "place"


class Stopped(Exception):
    """An update asked to stop (the interface quitting): what it did is committed, and the next one goes on."""


@contextmanager
def _transaction(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embed_text(section: str, text: str, place: str = "") -> str:
    head = "\n".join(x for x in (place, section) if x)
    return f"{head}\n{text}" if head else text


def chunking() -> str:
    """The chunking an index records: its options' name and the cutter's version (a fix to how pieces are cut changes
    the chunks as surely as other options do)."""
    return f"{chunk.DEFAULT.name} v{chunk.VERSION}"


def place_of(path: Path, roots: list[Path]) -> str:
    """Where a file sits for a reader: its declared folder's name and the path inside it ("projects/a/b.md")."""
    for r in roots:
        try:
            return (Path(r.name) / Path(path).relative_to(r)).as_posix()
        except ValueError:
            continue
    raise AssertionError(f"{path} is under no declared folder")


def is_data(doc: parse.Doc) -> str | None:
    """What kind of data a file holds, or what tool wrote its code (datafilter.reason); None when it is to be read."""
    if not doc.blocks or doc.blocks[0].unit != "line":
        return None
    if doc.markdown and parse.kind(doc.path) == "html":    # a page's main text is prose, not its markup
        return None
    return datafilter.reason(doc.path.name, doc.blocks[0].text)


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
    largest: list[tuple[str, int]] = field(default_factory=list)  # (path, chunks), the biggest at or over the mark
    embedded: int = 0          # chunks given a vector by this update
    unembedded: int = 0        # chunks still without one (an embedder is set): searched by their words until then
    embed_error: str = ""      # why the embedding endpoint did not answer, when it did not
    halved: int = 0            # chunks embedded by their head: the endpoint's model takes less than a chunk


def _connect(path: Path) -> sqlite3.Connection:
    # Transactions are explicit, and a lock held by another connection is waited for (30 s). Switching to WAL is
    # retried: when two connections open a new file at once, SQLite answers the switch "database is locked" without
    # waiting, and a file left in the old journal mode fails writes the same way later (both reproduced, 2026-10-04).
    os.close(guard.open_private(path, os.O_RDWR))      # 0600 before SQLite opens it; its -wal and -shm follow it
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
    deadline = time.monotonic() + 30
    while True:
        try:
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal":
                break
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() > deadline:
                raise
        if time.monotonic() > deadline:                 # a file system without WAL (some network drives)
            raise sqlite3.OperationalError(f"{path}: the index needs SQLite's WAL mode, which this drive refuses")
        time.sleep(0.05)
    return db


class Index:
    def __init__(self, home: Path | None = None, embedder=FROM_SETTINGS):
        """`embedder`: the user's embedding endpoint by default (PERESEARCH_EMBED_URL…), or one given; None: lexical."""
        self.home = Path(home or guard.home())
        guard.private_folder(self.home)
        # The TUI searches (a question) and updates (a folder check) from worker threads, possibly at once. One
        # sqlite3 connection used by two threads at once fails ("bad parameter or other API misuse"), so searches
        # read through `db` one at a time (under _lock) and an update writes through a connection of its own:
        # readers then see only committed data, as another process would.
        self.db = _connect(self.home / "index.sqlite")
        self._migrate()
        self._writer = None
        self._write_lock = threading.Lock()            # one update at a time
        self.from_settings = embedder is FROM_SETTINGS
        self.embedding_settings = None
        if self.from_settings:
            from peresearch.zetokrag.embed import from_settings
            from peresearch import settings

            self.embedding_settings = tuple(settings.get("PERESEARCH_EMBED_" + name) for name in ("URL", "MODEL", "KEY"))
            embedder = from_settings()[0]
        self._embedder = embedder
        self._snapshot = None                           # (data_version, ids, BM25, float32 vectors)
        self._lock = threading.RLock()                 # the reading connection and the snapshot

    def _migrate(self) -> None:
        """An index from an earlier layout is emptied (it is rebuilt by the next update) and its files removed.
        The version is read and the layout set up inside one write transaction: a second connection opening the
        same new file waits for it, then finds it set up (two that read version 0 at once each emptied what the other
        had set up and written: "no such table: vectors", reproduced 2026-10-05). No executescript here: it commits
        whatever transaction is open first."""
        db = self.db
        db.execute("BEGIN IMMEDIATE")
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] < VERSION:
                for table in ("files", "chunks", "vectors", "meta"):
                    db.execute(f"DROP TABLE IF EXISTS {table}")
                for name in OLD_FILES:
                    (self.home / name).unlink(missing_ok=True)
            for statement in SCHEMA.split(";"):
                if statement.strip():                    # a comment alone is an empty statement: harmless
                    db.execute(statement)
            if "reason" not in {r[1] for r in db.execute("PRAGMA table_info(files)")}:    # added 2026-10-03
                db.execute("ALTER TABLE files ADD COLUMN reason TEXT DEFAULT ''")
            if "place" not in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}:    # added 2026-10-03
                db.execute("ALTER TABLE chunks ADD COLUMN place TEXT DEFAULT ''")
            db.execute(f"PRAGMA user_version={VERSION}")
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise

    # --- storage ---

    @property
    def embedder(self):
        """The embedder (the user's endpoint), or None: the index is searched by words alone."""
        return self._embedder

    def signature(self) -> str:
        """The embedder the stored vectors must come from: its model and where it runs; "" with none."""
        if self._embedder is None:
            return ""
        return getattr(self._embedder, "signature", type(self._embedder).__name__)

    def read(self, sql: str, params=()) -> list[tuple]:
        """A query on the reading connection, one thread at a time."""
        with self._lock:
            return self.db.execute(sql, params).fetchall()

    def context(self) -> str:
        """What the stored vectors were embedded with before each chunk (CONTEXTS), as searches must rank with it."""
        row = self.read("SELECT value FROM meta WHERE key='context'")
        return row[0][0] if row else "section"

    def ranking_text(self, row: tuple) -> str:
        """The text a chunk (a row of `rows`) is ranked by: what was embedded for it."""
        return embed_text(row[2], row[6], row[8] if self.context() == "place" else "")

    def rows(self, ids=None) -> list[tuple]:
        q = "SELECT id, path, section, unit, start, end, text, sha, place FROM chunks"
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
        for every query. BM25 is None for an empty index; the vectors None when none can be searched (no embedder, or
        the stored ones are another embedder's), a chunk without one yet a row of NaN."""
        from peresearch.zetokrag.core import BM25

        with self._lock:
            version = self.db.execute("PRAGMA data_version").fetchone()[0]
            if self._snapshot is None or self._snapshot[0] != version:
                self.db.execute("BEGIN")
                try:
                    rows = self.db.execute("SELECT c.id, c.section, c.text, c.place, v.vec FROM chunks c "
                                           "LEFT JOIN vectors v ON v.id = c.id ORDER BY c.id").fetchall()
                    place = self.context() == "place"
                    stored = self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
                finally:
                    self.db.execute("COMMIT")
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                bm = (BM25([embed_text(r[1], r[2], r[3] if place else "") for r in rows], folded=False)
                      if rows else None)
                usable = self._embedder is not None and stored is not None and stored[0] == self.signature()
                found = [np.frombuffer(r[4], dtype=np.float16) for r in rows if r[4] is not None] if usable else []
                dims = {len(v) for v in found}
                vecs = None
                if found and len(dims) == 1:
                    vecs = np.full((len(rows), dims.pop()), np.nan, dtype=np.float32)
                    for i, r in enumerate(rows):
                        if r[4] is not None:
                            vecs[i] = np.frombuffer(r[4], dtype=np.float16)
                assert (vecs is None or vecs.shape[0] == len(ids)) and \
                    (bm is None or bm.model.scores["num_docs"] == len(ids)), "snapshot: ids, BM25 and vectors do not line up"
                assert bool(np.all(np.diff(ids) > 0)), "snapshot: chunk ids not strictly increasing"
                self._snapshot = (version, ids, bm, vecs)
            return self._snapshot[1:]

    def why_not(self, path) -> str | None:
        """Why `path` is not in the index: the reason kept for it or for the nearest skipped folder above it, or
        its file status when it was read but not chunked (duplicate, data, too_large…). None when it is indexed,
        protected, or was never seen."""
        p = Path(path)
        for q in (p, *p.parents):
            row = self.read("SELECT reason, note FROM skipped WHERE path=?", (str(q),))
            if row:
                reason, note = row[0]
                return f"{reason}: {note}" if note else reason
        row = self.read("SELECT status, reason FROM files WHERE path=?", (str(p),))
        if not row or row[0][0] == "ok":
            return None
        status, reason = row[0]
        return f"{status}: {reason}" if reason else status

    def coverage(self) -> dict[str, int]:
        """What the index holds no chunk of, counted by reason: what the walk left out (a folder counts once) and
        files read but not chunked (no text layer, too large, data…). Copies are not counted: their content is in."""
        out = dict(self.read("SELECT reason, COUNT(*) FROM skipped GROUP BY reason"))
        for status, n in self.read("SELECT status, COUNT(*) FROM files WHERE status NOT IN ('ok', 'duplicate') "
                                   "GROUP BY status"):
            out[status] = out.get(status, 0) + n
        return out

    def held(self) -> tuple[int, int]:
        """(files searched by meaning, their chunks)."""
        return (self.read("SELECT COUNT(*) FROM files WHERE status='ok'")[0][0],
                self.read("SELECT COUNT(*) FROM chunks")[0][0])

    def left_out(self, reason: str, limit: int = 50) -> tuple[list[str], int]:
        """The paths `coverage()` counts under `reason` (the first `limit`, sorted) and how many there are."""
        query = ("SELECT path FROM skipped WHERE reason=? UNION SELECT path FROM files WHERE status=? "
                 "AND status NOT IN ('ok', 'duplicate')")
        total = self.read(f"SELECT COUNT(*) FROM ({query})", (reason, reason))[0][0]
        return [p for (p,) in self.read(f"{query} ORDER BY path LIMIT ?", (reason, reason, limit))], total

    def bm25(self):
        """(ids, BM25) over every chunk, on words as typed."""
        ids, bm, _ = self.snapshot()
        return ids, bm

    # --- update ---

    def _files(self, roots: list[Path], report: Report, skipped: list[tuple[str, str, str]]):
        """The files to index, by the rules of files.py and the types parse reads; everything else left out goes to
        `skipped` (path, reason, note) and the report, except protected paths, which are only counted."""
        def skip(s: files.Skipped):
            skipped.append((s.path, s.reason, s.note))
            shown = f"{s.path} ({s.note})" if s.reason == "large_folder" else s.path
            report.problems.setdefault(s.reason, []).append(shown)

        def denied():
            report.denied += 1

        for p in files.walk(roots, guard.config(), skip, denied):
            kind = parse.kind(p)
            if kind == "unsupported":
                skip(files.Skipped(str(p), "unsupported", p.suffix.lower() or "no extension"))
            elif kind in ("data", "spreadsheet"):       # tables and logs: read by the tools, not ranked
                skip(files.Skipped(str(p), "data_type", p.suffix.lower()))
            else:
                yield p

    def update(self, roots: list[Path] | None = None, stop=None, progress=None) -> Report:
        """`stop()`, checked before each file: true, the update commits what it did and raises Stopped.
        `progress(done, total)` after each file: how many of the files to look at it has reached."""
        with self._write_lock:
            if self._writer is None:
                self._writer = _connect(self.home / "index.sqlite")
            report = self._update(self._writer, roots, stop, progress)
        if self._embedder is not None:                       # the chunks are committed: searchable by their words
            from peresearch.zetokrag.embed import EndpointError

            before = getattr(self._embedder, "halved", 0)
            try:
                report.embedded = self.embed(stop=stop)
            except EndpointError as e:                       # down or wrong: said; the next update goes on
                report.embed_error = str(e)
            report.halved = getattr(self._embedder, "halved", 0) - before
            report.unembedded = self.read("SELECT COUNT(*) FROM chunks c LEFT JOIN vectors v ON v.id = c.id "
                                          "WHERE v.id IS NULL")[0][0]
        report.chunks = self.read("SELECT COUNT(*) FROM chunks")[0][0]
        # Hand-copied third-party code cannot be told from one's own (stb_image.h: 206 chunks): the biggest files are
        # named, for the user to exclude what is not theirs
        report.largest = [(path, n) for path, n in self.read(
            "SELECT path, COUNT(*) AS n FROM chunks GROUP BY path ORDER BY n DESC, path LIMIT ?", (LARGEST,))
            if n >= LARGE_FILE_CHUNKS]
        guard.audit("index", added=report.added, changed=report.changed, removed=report.removed,
                    chunks=report.chunks, withheld=sum(report.withheld.values()))
        return report

    @staticmethod
    def _drop(db, key: str) -> None:
        db.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM chunks WHERE path=?)", (key,))
        db.execute("DELETE FROM chunks WHERE path=?", (key,))

    def _update(self, db, roots, stop=None, progress=None) -> Report:
        """Bring the index in line with the declared folders, committing as it goes (at least every COMMIT_SECONDS
        of work): an update that is stopped (Ctrl+C, a power cut, an error) keeps what it committed, and the next one
        goes on from there. Every commit leaves the index whole: each file's chunks with their vectors and its
        extracted text, no vector without its chunk; only files not reached yet are as they were."""
        # a folder declared inside another is walked once, with it (the owner's config lists ~/projects and
        # projects inside it, whose files were read twice)
        roots = guard.outermost([Path(r) for r in (guard.roots() if roots is None else roots)])
        report = Report()
        assert CONTEXT in CONTEXTS, f"context {CONTEXT!r}"
        with _transaction(db):
            stored = db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
            cut = db.execute("SELECT value FROM meta WHERE key='chunking'").fetchone()
            context = db.execute("SELECT value FROM meta WHERE key='context'").fetchone()
            secrets = db.execute("SELECT value FROM meta WHERE key='secrets'").fetchone()
            # Vectors of another embedder cannot be searched with this one's queries: they go (the chunks stay; Index.embed
            # makes them again); with no embedder they wait, unused. Another chunking or context (an index from before they were
            # recorded holds the baseline cut, ranked with the section only): every file is read and cut again, each
            # keeping its old chunks, still searchable, until it is reached.
            # A file to read again is marked by mtime = MARKED; its content hash stays, so its extracted text
            # stays found until it is reached (a stopped re-cut once lost them: reproduced 2026-10-04).
            if self._embedder is not None and (stored is None or stored[0] != self.signature()):
                db.execute("DELETE FROM vectors")
                db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (self.signature(),))
            if ((cut[0] if cut else "baseline") != chunking()
                    or (context[0] if context else "section") != CONTEXT
                    or (secrets[0] if secrets else "") != guard.fingerprint()):    # judged by other secret rules
                db.execute("UPDATE files SET size = -1, mtime = ?", (MARKED,))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('chunking', ?)", (chunking(),))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('context', ?)", (CONTEXT,))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('secrets', ?)", (guard.fingerprint(),))
            # A kind whose reader changed (parse.READERS; an index from before they were recorded read every kind
            # with its first reader) has its files read again, and only those
            readers = db.execute("SELECT value FROM meta WHERE key='readers'").fetchone()
            before = json.loads(readers[0]) if readers else {}
            changed = {k for k, v in parse.READERS.items() if before.get(k, 1) != v}
            for (key,) in db.execute("SELECT path FROM files").fetchall() if changed else []:
                if parse.kind(Path(key)) in changed:
                    db.execute("UPDATE files SET size = -1, mtime = ? WHERE path = ?", (MARKED, key))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('readers', ?)", (json.dumps(parse.READERS, sort_keys=True),))
        known = {r[0]: r for r in db.execute("SELECT path, size, mtime, sha, status FROM files")}
        judged = db.execute("SELECT value FROM meta WHERE key='datafilter'").fetchone()
        rejudge = judged is None or judged[0] != str(datafilter.VERSION)   # the filter changed: judge plain text again
        found, skipped, parsed = [], [], {}
        for path in self._files(roots, report, skipped):
            key = str(path)
            try:
                st = path.stat()
                old = known.get(key)
                digest = sha256(path)
                same = bool(old and old[1] == st.st_size and old[2] == st.st_mtime and old[3] == digest)
                found.append((path, key, st, old, same, digest))
            except FileNotFoundError:          # gone since the walk listed it (Word's ~$ owner file): next update
                continue
            except PermissionError as e:       # a file the user may not read
                skipped.append((key, "unreadable", type(e).__name__))
                report.problems.setdefault("unreadable", []).append(key)
        first = {}                                            # content hash -> the path indexed for it
        for _, key, _, _, _, digest in found:
            first.setdefault(digest, key)

        def commit():
            orphans = db.execute("SELECT COUNT(*) FROM vectors v LEFT JOIN chunks c ON c.id = v.id "
                                 "WHERE c.id IS NULL").fetchone()[0]
            assert orphans == 0, f"index: {orphans} vectors without a chunk"
            self._sync_texts(db, parsed)
            db.execute("COMMIT")
            parsed.clear()

        opened, since = False, 0.0
        try:
            for done, entry in enumerate(found, 1):
                if stop is not None and stop():
                    if opened:                                # kept: what it did, whole
                        opened = False
                        commit()
                    raise Stopped("the index update was asked to stop; the next one goes on from here")
                if not opened:
                    db.execute("BEGIN IMMEDIATE")
                    opened, since = True, time.monotonic()
                self._index_file(db, entry, roots, first, rejudge, report, parsed)
                if time.monotonic() - since >= COMMIT_SECONDS:
                    opened = False
                    commit()
                if progress is not None:
                    progress(done, len(found))
            if not opened:
                db.execute("BEGIN IMMEDIATE")
                opened = True
            # last, once every file was reached: what is gone, what was left out, and the filter the files were judged by
            for key in set(known) - {key for _, key, *_ in found}:
                self._drop(db, key)
                db.execute("DELETE FROM files WHERE path=?", (key,))
                report.removed += 1
            db.execute("DELETE FROM skipped")
            db.executemany("INSERT INTO skipped VALUES (?, ?, ?)", skipped)   # a path skipped twice is a bug: it raises
            db.execute("INSERT OR REPLACE INTO meta VALUES ('datafilter', ?)", (str(datafilter.VERSION),))
            opened = False
            commit()
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        return report

    def _index_file(self, db, entry, roots, first, rejudge, report, parsed) -> int:
        """Bring one file up to date inside the open transaction; the number of chunks it now waits to embed."""
        path, key, st, old, same, digest = entry
        if first[digest] != key:                              # a copy: indexed once, under the first path
            self._drop(db, key)
            db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)",
                       (key, st.st_size, st.st_mtime, digest, "duplicate", first[digest], 0, ""))
            report.problems.setdefault("duplicate", []).append(f"{key} (same as {first[digest]})")
            return 0
        doc, was_known = None, old is not None
        if old and old[3] == digest and old[4] in ("ok", "data") and rejudge and datafilter.judges(path.name):
            doc = parse.read(path)                            # judged again; read again only if its verdict moved
            why = is_data(doc) if doc.status == "ok" else None
            if ("data" if why else doc.status) != old[4]:
                old = None
            elif why:                                         # still data, perhaps for another reason: said anew
                db.execute("UPDATE files SET reason=? WHERE path=?", (why, key))
        if old and old[3] == digest and old[4] != "duplicate" and old[2] != MARKED:
            if not same:
                db.execute("UPDATE files SET size=?, mtime=? WHERE path=?", (st.st_size, st.st_mtime, key))
            if old[4] == "data":
                report.problems.setdefault("data_file", []).append(key)
            report.unchanged += 1
            # its declared folder changed (a parent declared instead): a new place, and with it a new vector
            moved = db.execute("UPDATE chunks SET place=? WHERE path=? AND place IS NOT ?",
                               (place_of(path, roots), key, place_of(path, roots))).rowcount
            if moved and CONTEXT == "place":
                db.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM chunks WHERE path=?)", (key,))
                return moved
            return 0
        doc = doc or parse.read(path)
        parsed[digest] = doc
        status, chunks, withheld, reason = doc.status, [], 0, ""
        if status == "ok" and (why := is_data(doc)):
            status, reason = "data", why
        elif status == "ok":
            code = guard.quoted(path)                        # in code an assigned secret is a string literal
            safe = lambda section, text: not (guard.find_secrets(text, code) or guard.find_secrets(section, code))
            chunks, withheld = chunk.chunk(doc, keep=safe)
        assert status in ("ok", "data", "too_large", "binary", "needs_ocr", "error"), f"{key}: status {status!r}"
        self._drop(db, key)
        db.executemany("INSERT INTO chunks (path, section, unit, start, end, text, sha, place) VALUES (?,?,?,?,?,?,?,?)",
                       [(key, c.section, c.unit, c.start, c.end, c.text, digest, place_of(path, roots)) for c in chunks])
        db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)",
                   (key, st.st_size, st.st_mtime, digest, status, doc.note, withheld, reason))
        if status == "data":
            report.problems.setdefault("data_file", []).append(key)
        elif status != "ok":
            report.problems.setdefault(status, []).append(key)
        if withheld:
            report.withheld[key] = withheld
        report.changed += was_known
        report.added += not was_known
        return len(chunks)

    def _sync_texts(self, db, parsed: dict[str, parse.Doc]) -> None:
        """Make the extracted-text store hold exactly the texts the indexed files need: write the missing ones
        (converting again only a file not converted in this update) and remove everything else in its folder. Runs
        last, inside the update's transaction, while no other update can write."""
        folder = self.home / extracted.FOLDER
        want: dict[str, str] = {}                              # content hash -> a path holding it
        for path, status, note, sha in db.execute("SELECT path, status, note, sha FROM files ORDER BY path"):
            if extracted.needed(path, status, note):
                want.setdefault(sha, path)
        have = {sha for (sha,) in db.execute("SELECT sha FROM texts")}
        for sha in have - set(want):
            db.execute("DELETE FROM texts WHERE sha=?", (sha,))
        for sha, path in want.items():
            if sha in have and (folder / f"{sha}.txt").is_file():
                continue
            doc = parsed.get(sha) or parse.read(Path(path))
            if doc.status != "ok":                             # changed or gone since it was hashed: next update
                db.execute("DELETE FROM texts WHERE sha=?", (sha,))
                continue
            text, units = extracted.render(doc)
            extracted.write(folder, sha, text)
            db.execute("INSERT OR REPLACE INTO texts VALUES (?, ?)", (sha, json.dumps(units)))
        kept = {f"{sha}.txt" for (sha,) in db.execute("SELECT sha FROM texts")}
        if folder.is_dir():
            for f in folder.iterdir():
                if f.name not in kept:
                    f.unlink(missing_ok=True)
        assert kept <= {f"{sha}.txt" for sha in want} and all((folder / k).is_file() for k in kept), \
            "the extracted-text store and its table disagree"

    def extracted(self, path) -> tuple[Path, list] | None:
        """(store file, line map) holding `path`'s extracted text, or None when its bytes are its text, or it is not
        indexed. See extracted.locate for the map."""
        row = self.read("SELECT t.sha, t.units FROM files f JOIN texts t ON t.sha = f.sha WHERE f.path=?",
                        (str(path),))
        return (self.home / extracted.FOLDER / f"{row[0][0]}.txt", json.loads(row[0][1])) if row else None

    def embed(self, stop=None, progress=None, batch: int = EMBED_BATCH) -> int:
        """Give every chunk without a vector one from the embedder, `batch` chunks a request, each batch committed:
        stopped (`stop()` true before a batch) it raises Stopped and keeps what it did; run again, it goes on. Vectors of
        another embedder, or of another length behind the same name (a model swapped on the server; one query probes
        it), are dropped and made again. `progress(done, total)` after each batch. The number of chunks embedded."""
        if self._embedder is None:
            return 0
        from peresearch.zetokrag.embed import EndpointError

        signature = self.signature()
        with self._write_lock:
            if self._writer is None:
                self._writer = _connect(self.home / "index.sqlite")
            db, place = self._writer, self.context() == "place"
            with _transaction(db):
                stored = db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
                if stored is None or stored[0] != self.signature():
                    db.execute("DELETE FROM vectors")
                    db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (self.signature(),))
            one = db.execute("SELECT vec FROM vectors LIMIT 1").fetchone()
            if one is not None:
                probe = self._embedder.queries(["probe"])
                if probe.shape[1] != len(np.frombuffer(one[0], dtype=np.float16)):
                    with _transaction(db):
                        active = db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
                        if active is None or active[0] != signature:
                            raise EndpointError("embedding configuration changed during probe; kept the new vectors")
                        db.execute("DELETE FROM vectors")
            total = db.execute("SELECT COUNT(*) FROM chunks c LEFT JOIN vectors v ON v.id = c.id "
                               "WHERE v.id IS NULL").fetchone()[0]
            done = 0
            while True:
                if stop is not None and stop():
                    raise Stopped("embedding was asked to stop; the next update goes on from here")
                rows = db.execute("SELECT c.id, c.section, c.text, c.place FROM chunks c LEFT JOIN vectors v "
                                  "ON v.id = c.id WHERE v.id IS NULL ORDER BY c.id LIMIT ?", (batch,)).fetchall()
                if not rows:
                    return done
                vecs = self._embedder.documents([embed_text(r[1], r[2], r[3] if place else "") for r in rows]
                                                ).astype(np.float16)
                assert vecs.shape[0] == len(rows), f"{vecs.shape[0]} vectors for {len(rows)} chunks"
                with _transaction(db):              # another updater may have embedded or dropped them meanwhile
                    active = db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
                    if active is None or active[0] != signature:
                        raise EndpointError("embedding configuration changed during update; discarded the old batch")
                    before = db.total_changes
                    db.executemany("INSERT OR IGNORE INTO vectors SELECT ?, ? WHERE EXISTS (SELECT 1 FROM chunks "
                                   "WHERE id = ?)", [(r[0], v.tobytes(), r[0]) for r, v in zip(rows, vecs)])
                    done += db.total_changes - before
                if progress is not None:
                    progress(done, total)
