"""Contracts of the index (zetokrag.index) and the search (zetokrag.search).

The index is checked by a state machine (Hypothesis): it writes, copies, edits, touches and deletes files in a
small tree of declared folders (some declared inside others), updates the index, searches, and after every update
checks the index against what it must hold:

- exactly the chunks a rebuild from parse and chunk alone gives, each with the vector of its own text (an
  incremental update equals a fresh build: no stale chunk or vector, no reused id, nothing walked twice);
- every content indexed once, under the first path in walk order; every other copy and every data file listed in
  the report; every file of the tree accounted for;
- no credential stored anywhere in the database files.

Searches between updates must never fail and must only return excerpts that are verbatim in their file, or say
the file changed. Concurrency (threads and processes) and storage changes have their own tests below.
"""
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from peresearch import guard
from peresearch.zetokrag import chunk, index as index_module, parse, search
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher
from tests.conftest import FakeEmbedder, FakeReranker
from tests.test_guard import CANARIES

PATHS = ["a.md", "b.md", "z.txt", "sub/a.md", "sub/c.txt", "sub/deep/d.md", "other/e.md"]
WORDS = ["gradient", "descent", "học", "máy", "tokenizer", "BPE", "phở", "nước", "dùng", "机器学习", "attention", "rank"]
SECRET = CANARIES["github-token"]
words = st.lists(st.sampled_from(WORDS), min_size=1, max_size=8).map(" ".join)
note = st.lists(st.tuples(st.sampled_from(["#", "##", ""]), words, st.lists(words, min_size=1, max_size=3)),
                min_size=1, max_size=4).map(
    lambda parts: "\n\n".join((f"{h} {t}\n\n" if h else "") + "\n".join(body) for h, t, body in parts) + "\n")
data = st.sampled_from(["".join(f"http://www.imdb.com/title/tt{i:07d}/\n" for i in range(60)),
                        "".join(f"{i / 7:.5f}\n" for i in range(80))])
long_note = st.lists(words, min_size=40, max_size=90).map(lambda ws: "# Long\n\n" + "\n\n".join(ws) + "\n")
content = st.one_of(note, note, note, long_note, data, note.map(lambda t: t + f"\n\ntoken: {SECRET}\n"), st.just(""),
                    note.map(lambda t: t + "\x00binary"))


def walk_order(roots: list[Path]) -> list[Path]:
    """Files under the outermost roots in the order the index walks them (folders and names sorted)."""
    out = []
    for root in guard.outermost(roots):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in guard.SKIP_DIRS)
            out += [Path(dirpath) / f for f in sorted(filenames) if parse.kind(Path(dirpath) / f) != "unsupported"]
    return out


def expected(roots: list[Path]):
    """(chunks, status per file, copy -> original, withheld per file) the index must hold, computed with parse and
    chunk alone."""
    chunks, statuses, first, copies, withheld = set(), {}, {}, {}, {}
    for p in walk_order(roots):
        digest = index_module.sha256(p)
        if digest in first:
            statuses[str(p)], copies[str(p)] = "duplicate", str(first[digest])
            continue
        first[digest] = p
        doc = parse.read(p)
        if doc.status == "ok" and index_module.is_data(doc):
            statuses[str(p)] = "data"
            continue
        statuses[str(p)] = doc.status
        safe = lambda section, text: not (guard.find_secrets(text) or guard.find_secrets(section))
        cs, held = chunk.chunk(doc, keep=safe) if doc.status == "ok" else ([], 0)
        chunks |= {(str(p), c.section, c.unit, c.start, c.end, c.text) for c in cs}
        if held:
            withheld[str(p)] = held
    return chunks, statuses, copies, withheld


class IndexMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "notes"
        self.root.mkdir()
        self.home = self.tmp / "home"
        self.embedder = FakeEmbedder()
        self.idx = Index(self.home, embedder=self.embedder)
        self.searcher = Searcher(self.idx, reranker=FakeReranker())
        self.roots = [self.root]
        self.env = os.environ.get("PERESEARCH_HOME")
        os.environ["PERESEARCH_HOME"] = str(self.home)                     # the audit log goes here

    def teardown(self):
        if self.env is None:
            os.environ.pop("PERESEARCH_HOME", None)
        else:
            os.environ["PERESEARCH_HOME"] = self.env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stored(self):
        return ({r[1:] for r in self.idx.rows()}, dict(self.idx.read("SELECT path, status FROM files")))

    def files(self):
        return sorted(p for p in self.root.rglob("*") if p.is_file())

    @rule(path=st.sampled_from(PATHS), text=content)
    def write(self, path, text):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    @rule(path=st.sampled_from(PATHS), text=note)
    def write_latin1(self, path, text):
        """Text that is not UTF-8 is indexed with replacement characters and searched without failing."""
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8") + "\ncafé\n".encode("latin-1"))

    @precondition(lambda self: self.files())
    @rule(data=st.data(), path=st.sampled_from(PATHS))
    def copy(self, data, path):
        source, target = data.draw(st.sampled_from(self.files())), self.root / path
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    @precondition(lambda self: self.files())
    @rule(data=st.data())
    def delete(self, data):
        data.draw(st.sampled_from(self.files())).unlink()

    @precondition(lambda self: self.files())
    @rule(data=st.data())
    def touch(self, data):
        p = data.draw(st.sampled_from(self.files()))
        s = p.stat()
        os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns + 1_000_000_000))       # same content, a new time

    @rule()
    def update_that_fails(self):
        """An update that fails midway (here the embedder) leaves the index as it was; the next one completes."""
        before = self.stored()

        def broken(texts):
            raise RuntimeError("embedder failed")
        self.embedder.documents, real = broken, self.embedder.documents
        try:
            self.idx.update(self.roots)
            failed = False                              # nothing new to embed: an ordinary update
        except RuntimeError:
            failed = True
        finally:
            self.embedder.documents = real
        if failed:
            assert self.stored() == before

    @rule(nested=st.booleans())
    def update(self, nested):
        """With `nested`, a folder inside the root is declared too (the owner's config does this)."""
        self.roots = [self.root / "sub", self.root] if nested and (self.root / "sub").is_dir() else [self.root]
        known = {r[0]: r[1:] for r in self.idx.read("SELECT path, size, mtime, sha, status FROM files")}
        hashed, real_sha = [], index_module.sha256

        def counting(path):
            hashed.append(str(path))
            return real_sha(path)
        index_module.sha256 = counting
        try:
            report = self.idx.update(self.roots)
        finally:
            index_module.sha256 = real_sha
        stat = {str(p): p.stat() for p in walk_order(self.roots)}
        assert sorted(hashed) == sorted(f for f, st_ in stat.items()                # an untouched file is not
                                        if f not in known or (known[f][0], known[f][1]) != (st_.st_size, st_.st_mtime))   # read
        want_chunks, want_status, copies, withheld = expected(self.roots)
        rows = self.idx.rows()
        assert len(rows) == len(want_chunks)                                # no chunk stored twice
        assert {(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows} == want_chunks
        assert dict(self.idx.read("SELECT path, status FROM files")) == want_status   # every file accounted for
        assert {f: n for f, n in self.idx.read("SELECT path, withheld FROM files") if n} == withheld
        ids, vecs = self.idx.dense()
        assert sorted(ids.tolist()) == sorted(r[0] for r in rows)          # one vector per chunk
        by_id = {r[0]: r for r in rows}
        texts = [index_module.embed_text(by_id[i][2], by_id[i][6]) for i in ids.tolist()]
        assert not texts or np.array_equal(vecs, self.embedder.documents(texts))   # of its own text
        assert sorted(report.problems.get("duplicate", [])) == sorted(f"{p} (same as {o})" for p, o in copies.items())
        assert sorted(report.problems.get("data_file", [])) == sorted(p for p, s in want_status.items() if s == "data")
        read = {f for f, s in want_status.items() if s != "duplicate" and (
            f not in known or known[f][3] == "duplicate" or known[f][2] != index_module.sha256(Path(f)))}
        assert report.added == sum(f not in known for f in read)                    # files read again: new or
        assert report.changed == sum(f in known for f in read)                      # changed; the rest unchanged
        assert report.unchanged == sum(s != "duplicate" for s in want_status.values()) - len(read)
        assert report.removed == len(set(known) - set(want_status)) and report.chunks == len(rows)
        assert report.withheld == {f: n for f, n in withheld.items() if f in read}
        assert set(report.problems) <= {"duplicate", "data_file", "binary", "too_large", "needs_ocr", "error"}
        for status in ("binary", "too_large", "needs_ocr", "error"):
            assert sorted(report.problems.get(status, [])) == sorted(f for f in read if want_status[f] == status)
        last = (self.home / "audit.jsonl").read_text().splitlines()[-1]
        assert {k: v for k, v in __import__("json").loads(last).items() if k != "time"} == {
            "event": "index", "added": report.added, "changed": report.changed, "removed": report.removed,
            "chunks": report.chunks, "withheld": sum(report.withheld.values())}

    @rule(query=words, k=st.integers(1, 8))
    def find(self, query, k):
        ranked, scores = self.searcher.rank(query)
        hits, verdict, stale = self.searcher.find(query, k=k)
        kept, dropped = [], 0                                    # what find must return, from rank and the files
        for row, score in zip(ranked, scores):
            if len(kept) == k or score < search.PARTIAL:
                break
            p = Path(row[1])
            now = index_module.sha256(p) if p.exists() else None
            if now != row[7] and not (now and all(x in p.read_text(errors="replace") for x in row[6].split("\n\n"))):
                dropped += 1
                continue
            kept.append(row[0])
        assert [h.id for h in hits] == kept and stale == dropped
        assert verdict == search.verdict(float(scores[0]) if len(scores) else 0.0)
        assert len(hits) <= k and verdict in ("enough", "partial", "none") and stale >= 0
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
        assert all(h.score >= search.PARTIAL for h in hits)
        rows = {r[0]: r for r in self.idx.rows()}
        for h in hits:
            lines = Path(h.path).read_text(errors="replace").splitlines()
            span = "\n".join(lines[h.start - 1:h.end])
            assert h.file_changed or all(part.strip() in span for part in h.text.split("\n\n")), h.where
            row = rows[h.id]                                                   # the stored chunk, field by field
            assert (h.path, h.section, h.unit, h.start, h.end, h.text) == row[1:7]
            assert h.file_changed == (index_module.sha256(Path(h.path)) != row[7])
            assert h.neighbours == [rows[i][6] for i in (h.id - 1, h.id + 1)
                                    if i in rows and rows[i][1] == h.path and rows[i][2] == h.section]

    @invariant()
    def no_credential_is_stored(self):
        for f in self.home.glob("index.sqlite*"):                          # the database, its WAL and shared memory
            assert SECRET.encode() not in f.read_bytes(), f.name


IndexMachine.TestCase.settings = settings(max_examples=150, stateful_step_count=20, deadline=None,
                                          suppress_health_check=[HealthCheck.too_slow])
test_index_contract = IndexMachine.TestCase


# Past index bugs, replayed through the same machine and checks (each is a sequence of its rules).
REPLAYS = {
    "a copy takes the place of its deleted original": [
        ("write", "a.md", "# A\n\ngradient descent\n"), ("copy", "a.md", "b.md"), ("update", False),
        ("delete", "a.md"), ("update", False)],
    "a folder declared inside another is walked once (668 files counted for 530)": [
        ("write", "sub/a.md", "# S\n\nBPE tokenizer\n"), ("write", "a.md", "# T\n\nphở\n"), ("update", True)],
    "an edited file's old chunks and vectors are dropped": [
        ("write", "a.md", "# A\n\nrank\n\nattention\n"), ("update", False),
        ("write", "a.md", "# A\n\nmáy\n"), ("update", False), ("touch", "a.md"), ("update", False)],
    "two files removed in one update are both counted": [
        ("write", "a.md", "# A\n\nrank\n"), ("write", "b.md", "# B\n\nmáy\n"), ("write", "z.txt", "BPE\n"),
        ("update", False), ("delete", "a.md"), ("delete", "b.md"), ("update", False)],
    "a data file is reported, not indexed, on every update": [
        ("write", "z.txt", "".join(f"http://x.org/{i}\n" for i in range(60))), ("update", False), ("update", False)],
}


@pytest.mark.parametrize("name", REPLAYS)
def test_past_index_bugs_stay_fixed(name):
    m = IndexMachine()
    try:
        for step, *args in REPLAYS[name]:
            if step == "copy":
                (m.root / args[1]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(m.root / args[0], m.root / args[1])
            elif step == "delete":
                (m.root / args[0]).unlink()
            elif step == "touch":
                s = (m.root / args[0]).stat()
                os.utime(m.root / args[0], ns=(s.st_atime_ns, s.st_mtime_ns + 1_000_000_000))
            else:
                getattr(m, step)(*args)
        m.no_credential_is_stored()
    finally:
        m.teardown()


# --- search: stale excerpts, neighbours, the verdict -----------------------------------------------------------------

def make(home):
    idx = Index(home, embedder=FakeEmbedder())
    return idx, Searcher(idx, reranker=FakeReranker())


@pytest.fixture
def notes(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "ml.md").write_text("# Học máy\n\nGradient descent cập nhật trọng số theo đạo hàm.\n")
    (root / "food.md").write_text("# Phở\n\nNước dùng hầm xương bò trong nhiều giờ.\n")
    return root


def test_an_edited_file_never_yields_a_misquote(home, notes, monkeypatch):
    """Before an excerpt is returned its file is read again: edited away → dropped and counted; still there in an
    edited file → returned and flagged; deleted between the existence check and the read → gone."""
    idx, s = make(home)
    idx.update([notes])
    (notes / "ml.md").write_text("# Học máy\n\nSomething else entirely.\n")
    hits, _, stale = s.find("gradient descent trọng số", k=2)
    assert stale == 1 and not any(h.path.endswith("ml.md") for h in hits)
    (notes / "food.md").write_text("# Phở\n\nNước dùng hầm xương bò trong nhiều giờ.\n\nThêm hành.\n")
    hits, _, _ = s.find("nước dùng xương bò", k=1)
    assert hits[0].path.endswith("food.md") and hits[0].file_changed

    def vanished(path):
        raise FileNotFoundError(2, "No such file or directory")
    monkeypatch.setattr(search, "sha256", vanished)
    hits, _, stale = s.find("nước dùng xương bò", k=3)
    assert hits == [] and stale >= 1


def test_neighbours_are_the_adjacent_chunks_of_the_same_file_and_section(home, tmp_path):
    root = tmp_path / "n"
    root.mkdir()
    body = "\n\n".join(("para %d " % i) + "w" * 900 for i in range(4))   # one paragraph per chunk
    (root / "a.md").write_text("# S\n\n" + body + "\n\n# T\n\nother section\n")
    (root / "b.md").write_text("# S\n\nanother file\n")
    idx, s = make(home)
    idx.update([root])
    rows = [r for r in idx.rows() if r[1].endswith("a.md")]
    hits, _, _ = Searcher(idx, reranker=SpyReranker(lambda t: 0.9 if "para 1" in t else 0.0)).find("para", k=1)
    assert hits[0].neighbours == s.neighbours(rows[1][0], rows[1][1], "S") and len(hits[0].neighbours) == 2

    def paras(cid):
        return [[f"para {i}" in n for i in range(4)].index(True) for n in s.neighbours(cid, rows[0][1], "S")]
    assert [paras(r[0]) for r in rows[:4]] == [[1], [0, 2], [1, 3], [2]]     # not section T, not b.md


def test_the_verdict_follows_the_best_reranker_score():
    for best, want in ((0.0, "none"), (search.PARTIAL - 1e-6, "none"), (search.PARTIAL, "partial"),
                       (search.ENOUGH - 1e-6, "partial"), (search.ENOUGH, "enough"), (1.0, "enough")):
        assert search.verdict(best) == want


def test_an_empty_index_answers_none(home, tmp_path):
    (tmp_path / "e").mkdir()
    idx, s = make(home)
    idx.update([tmp_path / "e"])
    assert s.find("anything", k=3) == ([], "none", 0)
    rows, scores = s.rank("anything")
    assert rows == [] and len(scores) == 0


class SpyReranker:
    """Scores from a table keyed by text start; records what it was shown."""

    def __init__(self, score):
        self.score, self.seen = score, []

    def scores(self, query, texts):
        self.seen += texts
        return np.array([self.score(t) for t in texts], dtype=np.float32)


def test_ranking_reads_each_chunk_with_its_heading_and_words_as_typed(home, tmp_path):
    """The reranker and BM25 see "heading path + text"; BM25 matches words as typed (no folding); an excerpt
    scoring exactly PARTIAL is kept, one just below is not."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# Tokenizer notes\n\nmerges pairs of symbols\n\n" + "w " * 745 + "\n\nsecond chunk\n")
    (root / "b.md").write_text("# Cooking\n\nhọc nấu phở\n")
    (root / "c.md").write_text("# Plain\n\nhoc bai\n")
    spy = SpyReranker(lambda t: search.PARTIAL if "Tokenizer" in t else search.PARTIAL - 1e-3)
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    s = Searcher(idx, reranker=spy)
    hits, verdict, _ = s.find("tokenizer", k=5)
    assert {h.path for h in hits} == {str(root / "a.md")} and len(hits) == 3 and verdict == "partial"
    assert "Tokenizer notes\n# Tokenizer notes\n\nmerges pairs of symbols" in spy.seen
    ids, bm = idx.bm25()
    paths = {r[0]: r[1] for r in idx.rows()}
    rows = {r[0]: r for r in idx.rows()}
    matched = [rows[int(i)] for i, x in zip(ids, bm.scores("tokenizer")) if x > 0]
    assert len(matched) == 3 and sum("Tokenizer" not in r[6] for r in matched) == 2   # matched by its heading alone
    assert [paths[int(i)] for i, x in zip(ids, bm.scores("học")) if x > 0] == [str(root / "b.md")]
    assert [paths[int(i)] for i, x in zip(ids, bm.scores("hoc")) if x > 0] == [str(root / "c.md")]


def test_the_walk_leaves_out_what_is_not_notes_and_counts_it(home, tmp_path, monkeypatch):
    """Tooling folders and symlinked folders are not entered; protected folders and files are counted as denied;
    a folder with more files than MAX_FILES_PER_FOLDER is a dataset, reported by name and size."""
    monkeypatch.setattr(index_module, "MAX_FILES_PER_FOLDER", 3)
    root, outside = tmp_path / "n", tmp_path / "outside"
    for d in ("node_modules", ".git", "keys/.ssh", "keys/.aws", "keys/zz", "reviews", "ok", "three", "zz"):
        (root / d).mkdir(parents=True)
    outside.mkdir()
    (outside / "o.md").write_text("# O\n\noutside\n")
    os.symlink(outside, root / "linked")
    (root / "node_modules" / "x.md").write_text("# X\n\nvendored\n")
    (root / ".git" / "y.md").write_text("# Y\n\ngit\n")
    (root / "keys" / ".ssh" / "id.md").write_text("# K\n\nkey\n")
    (root / "ok" / ".env").write_text("A=1\n")
    for i in range(4):
        (root / "reviews" / f"{i}.txt").write_text(f"review {i}\n")
    for i in range(3):                                             # exactly the limit: notes, kept
        (root / "three" / f"{i}.md").write_text(f"# T{i}\n\nthree {i}\n")
    (root / "ok" / "note.md").write_text("# N\n\nnote\n")
    (root / "ok" / "tool.exe").write_bytes(b"MZ")
    (root / "keys" / "zz" / "k.md").write_text("# K\n\nafter the protected folders\n")
    (root / "zz" / "last.md").write_text("# L\n\nafter the dataset\n")
    idx = Index(home, embedder=FakeEmbedder())
    r = idx.update([root])
    want = {root / "ok" / "note.md", root / "keys" / "zz" / "k.md", root / "zz" / "last.md"} | \
        {root / "three" / f"{i}.md" for i in range(3)}
    assert {row[1] for row in idx.rows()} == {str(w) for w in want}
    assert {r_[0] for r_ in idx.read("SELECT path FROM files")} == {str(w) for w in want}   # no .exe either
    assert r.denied == 3                                           # .ssh, .aws and the .env file
    assert r.problems["large_folder"] == [f"{root / 'reviews'} (4 files)"]


# --- storage: one database, rebuilt when its layout or its embedder changes ------------------------------------------

def test_an_index_in_an_older_layout_is_rebuilt(home, notes):
    home.mkdir(parents=True)
    db = sqlite3.connect(home / "index.sqlite")
    db.executescript("CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT, section TEXT, unit TEXT, start INTEGER, "
                     "end INTEGER, text TEXT, sha TEXT); INSERT INTO chunks VALUES (1, '/gone.md', '', 'line', 1, 1, "
                     "'old text', 'x');")
    db.commit()
    db.close()
    for name in index_module.OLD_FILES:
        (home / name).write_bytes(b"old")
    idx, _ = make(home)
    assert idx.rows() == [] and not any((home / n).exists() for n in index_module.OLD_FILES)
    assert idx.update([notes]).added == 2 and len(idx.dense()[0]) == 2


class CountingEmbedder(FakeEmbedder):
    def __init__(self, signature):
        self.signature, self.embedded = signature, 0

    def documents(self, texts):
        self.embedded += len(texts)
        return super().documents(texts)


def test_vectors_of_another_embedder_are_never_mixed_in(home, notes, tmp_path):
    runs = [CountingEmbedder(sig) for sig in ("model-a/512", "model-a/512", "model-a/32768")]
    for e in runs:
        Index(home, embedder=e).update([notes])
    assert [e.embedded for e in runs] == [2, 0, 2]

    class Plain(FakeEmbedder):                                   # no signature: its class name stands for it
        def __init__(self):
            self.embedded = 0

        def documents(self, texts):
            self.embedded += len(texts)
            return super().documents(texts)

    class Other(Plain):
        pass
    other_home = tmp_path / "elsewhere" / "index"                # not $PERESEARCH_HOME, not made yet
    for e, want in ((Plain(), 2), (Plain(), 0), (Other(), 2)):
        Index(other_home, embedder=e).update([notes])
        assert e.embedded == want
    assert (other_home / "index.sqlite").exists()
    from peresearch.zetokrag import models
    assert Index(other_home).signature() == models.signature(index_module.EMBEDDER)   # without loading the model


# --- concurrency: searches go on while the index changes -------------------------------------------------------------

def test_another_process_can_search_during_an_update_and_sees_it_after(home, notes):
    """While another connection's update is uncommitted, a search sees the old index whole; after, the new one."""
    idx, s = make(home)
    idx.update([notes])
    s.find("phở nước dùng", k=1)                                       # this process's snapshot is built
    (notes / "new.md").write_text("# Đà Lạt\n\nĐồi thông và hồ Xuân Hương.\n")
    (notes / "food.md").unlink()
    during = []

    class SearchingEmbedder(FakeEmbedder):                             # runs inside the other update
        def documents(self, texts):
            during.append((s.find("gradient descent", k=3)[0], s.find("đồi thông hồ xuân hương", k=3)[0]))
            return super().documents(texts)
    Index(home, embedder=SearchingEmbedder()).update([notes])
    assert len(during) == 1
    unchanged, uncommitted = during[0]
    assert unchanged and unchanged[0].path.endswith("ml.md") and not uncommitted
    assert s.find("đồi thông hồ xuân hương", k=1)[0][0].path.endswith("new.md")
    assert not any(h.path.endswith("food.md") for h in s.find("phở nước dùng xương bò", k=3)[0])


def test_searching_while_the_same_index_is_updated_in_another_thread(home, notes):
    """The interface answers questions while it brings the index up to date: one Index, two threads."""
    idx, s = make(home)
    idx.update([notes])
    errors, rounds = [], {"search": 0}
    done = threading.Event()

    def search_loop():
        while not done.is_set():
            try:
                s.find("gradient descent", k=3)
                rounds["search"] += 1
            except Exception as e:                                    # any failure is the bug
                errors.append(repr(e))
                return

    def churn():
        try:
            for i in range(15):
                (notes / f"extra{i}.md").write_text(f"# Extra {i}\n\nNote number {i} about optimizers and gradient descent.\n")
                if i % 3 == 2:
                    (notes / f"extra{i - 2}.md").unlink()
                idx.update([notes])
        except Exception as e:
            errors.append(repr(e))
        finally:
            done.set()
    threads = [threading.Thread(target=search_loop) for _ in range(2)] + [threading.Thread(target=churn)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == [] and rounds["search"] > 5, (errors, rounds)
    hits, _, _ = s.find("Note number 14 about optimizers", k=3)
    assert hits and hits[0].path.endswith("extra14.md")


# --- data files (the rule is being replaced by a prose test, approved 2026-10-02) ------------------------------------

def test_data_files_stay_readable_by_the_agent(tmp_path):
    from peresearch.tools import Toolbox
    root = tmp_path / "n"
    root.mkdir()
    (root / "urls.txt").write_text("".join(f"http://www.imdb.com/title/tt{i:07d}/\n" for i in range(60)))
    out = Toolbox(roots=[root]).read(str(root / "urls.txt"), 1, 2)
    assert "tt0000000" in out and "tt0000001" in out


def test_is_data_flags_lines_of_urls_or_numbers_only():
    def doc(name, text):
        return parse.Doc(Path(name), "ok", [parse.Block(text, "line", 1)])
    urls = "".join(f"http://x.org/{i}\n" for i in range(60))
    nums = "".join(f"{i / 3:.4f}\n" for i in range(60))
    prose = "".join(f"Line {i} of a description, in words.\n" for i in range(60))
    assert index_module.is_data(doc("u.txt", urls)) and index_module.is_data(doc("n.txt", nums))
    assert not index_module.is_data(doc("p.txt", prose))
    assert not index_module.is_data(doc("short.txt", urls[:200]))                     # fewer than 50 lines
    assert not index_module.is_data(doc("u.md", urls)) and not index_module.is_data(doc("u.py", urls))


def test_an_update_without_folders_uses_the_declared_ones(home, notes):
    guard.set_roots([notes])
    idx = Index(home, embedder=FakeEmbedder())
    assert idx.update().added == 2 and {Path(r[1]).name for r in idx.rows()} == {"ml.md", "food.md"}


def test_only_the_best_fused_candidates_are_reranked(home, tmp_path):
    """BM25 and dense scores are fused (rho) and only the best N_RERANK candidates go to the reranker."""
    root = tmp_path / "n"
    root.mkdir()
    for i in range(search.N_RERANK + 10):
        (root / f"{i:02d}.md").write_text(f"# N{i}\n\n" + " ".join(["rank"] * (i + 1) + ["filler"] * 30) + "\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    spy = SpyReranker(lambda t: 0.9)
    Searcher(idx, reranker=spy).rank("rank")
    ids, bm, vecs = idx.snapshot()
    q = FakeEmbedder().queries(["rank"])[0].astype(np.float32)
    fused, _ = __import__("peresearch.zetokrag.core", fromlist=["x"]).fuse_minmax(bm.scores("rank"), vecs @ q,
                                                                               search.RHO, search.N_FIRST)
    texts = {r[0]: index_module.embed_text(r[2], r[6]) for r in idx.rows()}
    assert spy.seen == [texts[int(ids[i])] for i in fused[:search.N_RERANK]]
