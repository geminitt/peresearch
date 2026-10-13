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

The search: the fused candidates, the first PERESEARCH_RERANK_DEPTH of them reordered by the reranker and the rest
in fused order without a score; near-copies folded into the better-ranked one; pages of k showing each candidate
once, in order, with no score cutting one; a hit's place in the units of its file; the reranker's inputs and its
internal checks.
"""
import json
import os
import re
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

from peresearch import files, guard
from peresearch.zetokrag import chunk, extracted, index as index_module, parse, search
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher
from tests.conftest import FakeEmbedder, FakeReranker
from tests import test_parse_chunk as chunking
from tests.test_guard import CANARIES

PATHS = ["a.md", "b.md", "z.txt", "sub/a.md", "sub/c.txt", "sub/deep/d.md", "other/e.md", "code.py", "sub/tool.py",
         "dump.sql", "results.csv", "run.log"]
WORDS = ["gradient", "descent", "học", "máy", "tokenizer", "BPE", "phở", "nước", "dùng", "机器学习", "attention", "rank"]
SECRET = CANARIES["github-token"]
words = st.lists(st.sampled_from(WORDS), min_size=1, max_size=8).map(" ".join)
note = st.lists(st.tuples(st.sampled_from(["#", "##", ""]), words, st.lists(words, min_size=1, max_size=3)),
                min_size=1, max_size=4).map(
    lambda parts: "\n\n".join((f"{h} {t}\n\n" if h else "") + "\n".join(body) for h, t, body in parts) + "\n")
data = st.sampled_from(["".join(f"http://www.imdb.com/title/tt{i:07d}/\n" for i in range(60)),
                        "".join(f"{i / 7:.5f}\n" for i in range(80))])
long_note = st.lists(words, min_size=40, max_size=90).map(lambda ws: "# Long\n\n" + "\n\n".join(ws) + "\n")
sql = st.sampled_from(["".join(f"INSERT INTO t VALUES ({i}, 'name {i}');\n" for i in range(50)),     # data
                       "CREATE TABLE t (\n  id INT,\n  name TEXT\n);\n\n-- the people\n" * 3])          # read
secret_in_long_line = note.map(lambda t: t + "\n\n" + "x" * 1480 + f" {SECRET} " + "y" * 600 + "\n")   # cut there
# what the default chunking keeps whole or cuts at sentences (tests/test_parse_chunk.py's strategies)
blocks = st.one_of(chunking.table, chunking.big_table, chunking.bullets, chunking.blank_fence, chunking.sentence_line)
content = st.one_of(note, note, note, long_note, data, note.map(lambda t: t + f"\n\ntoken: {SECRET}\n"), st.just(""),
                    note.map(lambda t: t + "\x00binary"), chunking.python, sql, secret_in_long_line,
                    note.map(lambda t: f"# key {SECRET}\n\n" + t),                  # a heading holds the secret
                    st.lists(blocks, min_size=1, max_size=3).map(lambda bs: "# B\n\n" + "\n\n".join(bs) + "\n"))


def walk_order(roots: list[Path]) -> list[Path]:
    """Files under the outermost roots in the order the index walks them (files.walk, then by type)."""
    return [p for p in files.walk(guard.outermost(roots), {}, lambda s: None, lambda: None)
            if parse.kind(p) not in ("unsupported", "data")]


def current_span(excerpt, current):
    """Independent oracle: walk literal paragraph occurrences, accepting only blank lines between them."""
    parts = excerpt.split("\n\n")
    spans = []
    for start in range(len(current)):
        if not current.startswith(parts[0], start):
            continue
        end = start + len(parts[0])
        for part in parts[1:]:
            next_start = current.find(part, end)
            gap = current[end:next_start] if next_start >= 0 else "invalid"
            if gap.strip(" \t\n") or gap.count("\n") < 2:
                break
            end = next_start + len(part)
        else:
            spans.append((current.count("\n", 0, start) + 1, current.count("\n", 0, end) + 1, current[start:end]))
    return spans[0] if len(spans) == 1 else None


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
        """Text that is not UTF-8 is indexed as the converter decodes it and searched without failing."""
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8") + "\ncafé\n".encode("latin-1"))

    @rule(cells=st.lists(note, min_size=1, max_size=3), copy=st.booleans())
    def write_notebook(self, cells, copy):
        """A notebook (its text lives in the extracted-text store), sometimes copied next to itself."""
        p = self.root / "sub" / "n.ipynb"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": c} for c in cells]}))
        if copy:
            shutil.copyfile(p, self.root / "other" / "n.ipynb") if (self.root / "other").is_dir() else None

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

    @rule(after=st.integers(0, 3))
    def update_that_fails(self, after):
        """An update that fails midway (the embedder breaks after `after` calls; it commits after every file here)
        keeps what it committed: every file it holds is whole, as it was or as it is now; its chunks are committed
        before their vectors are asked for, so a chunk may wait for its vector (searched by its words meanwhile), but
        no vector is left without its chunk; the next update completes (the `update` rule checks it all)."""
        by_file = lambda rows: {p: {r for r in rows if r[0] == p} for p in {r[0] for r in rows}}
        before = by_file({r[1:7] for r in self.idx.rows()})
        now = by_file(expected(self.roots)[0])
        calls, real = [0], self.embedder.documents

        def breaking(texts):
            calls[0] += 1
            if calls[0] > after:
                raise RuntimeError("embedder failed")
            return real(texts)
        self.embedder.documents = breaking
        saved, index_module.COMMIT_SECONDS = index_module.COMMIT_SECONDS, 0
        try:
            self.idx.update(self.roots)
        except RuntimeError:
            pass
        finally:
            self.embedder.documents, index_module.COMMIT_SECONDS = real, saved
        held = by_file({r[1:7] for r in self.idx.rows()})
        for path, rows in held.items():
            assert rows == before.get(path) or rows == now.get(path), f"{path} half updated"
        assert self.idx.read("SELECT COUNT(*) FROM vectors v LEFT JOIN chunks c ON c.id = v.id WHERE c.id IS NULL") \
            == [(0,)]

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
        assert sorted(hashed) == sorted(stat)  # every refresh verifies bytes, even with unchanged metadata
        want_chunks, want_status, copies, withheld = expected(self.roots)
        rows = self.idx.rows()
        assert len(rows) == len(want_chunks)                                # no chunk stored twice
        assert {(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows} == want_chunks
        assert dict(self.idx.read("SELECT path, status FROM files")) == want_status   # every file accounted for
        assert {f: n for f, n in self.idx.read("SELECT path, withheld FROM files") if n} == withheld
        ids, vecs = self.idx.dense()
        assert sorted(ids.tolist()) == sorted(r[0] for r in rows)          # one vector per chunk
        by_id = {r[0]: r for r in rows}
        tops = sorted(self.roots, key=lambda r: len(Path(r).parts))      # a file's place is under its outermost root
        place = {r[0]: next(f"{Path(t).name}/{Path(r[1]).relative_to(t).as_posix()}" for t in tops
                            if Path(r[1]).is_relative_to(t)) for r in rows}
        assert all(r[8] == place[r[0]] for r in rows)
        texts = [index_module.embed_text(by_id[i][2], by_id[i][6], place[i]) for i in ids.tolist()]
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
        assert set(report.problems) <= {"duplicate", "data_file", "data_type", "binary", "too_large", "needs_ocr", "error"}
        tables = [str(p) for p in files.walk(guard.outermost(self.roots), {}, lambda s: None, lambda: None)
                  if parse.kind(p) == "data"]                       # .csv, .log: read by the tools, not ranked
        assert sorted(report.problems.get("data_type", [])) == sorted(tables)
        for status in ("binary", "too_large", "needs_ocr", "error"):
            assert sorted(report.problems.get(status, [])) == sorted(f for f in read if want_status[f] == status)
        assert self.idx.held() == (sum(s == "ok" for s in want_status.values()), len(rows))
        held = self.idx.coverage()                          # each reason's list is the files it counts (for /status)
        for reason, n in held.items():
            listed, total = self.idx.left_out(reason, limit=n + 1)
            assert total == n == len(listed) == len(set(listed)) and listed == sorted(listed)
            assert self.idx.left_out(reason, limit=1) == (listed[:1], n)
        assert self.idx.left_out("no such reason", limit=5) == ([], 0)
        assert sorted(self.idx.left_out("data", limit=10**6)[0]) == sorted(
            p for p, s in want_status.items() if s == "data")
        self.store_matches_the_index()
        last = (self.home / "audit.jsonl").read_text().splitlines()[-1]
        assert {k: v for k, v in __import__("json").loads(last).items() if k != "time"} == {
            "event": "index", "added": report.added, "changed": report.changed, "removed": report.removed,
            "chunks": report.chunks, "withheld": sum(report.withheld.values())}

    @rule(query=words, k=st.integers(1, 8), page=st.integers(1, 4))
    def find(self, query, k, page):
        ranked, scores, _ = self.searcher.candidates(query)
        hits, stale, total = self.searcher.find(query, k=k, page=page)
        kept, dropped, relocated = [], 0, {}      # positions must describe the current file, not the old index
        for row, score in list(zip(ranked, scores))[(page - 1) * k:page * k]:
            p = Path(row[1])
            now = index_module.sha256(p) if p.exists() else None
            if now != row[7]:
                location = current_span(row[6], parse.decode(p.read_bytes())[0].replace("\r\n", "\n")) \
                    if now and row[3] == "line" else None
                if location is None:
                    dropped += 1
                    continue
                relocated[row[0]] = location
            kept.append(row[0])
        assert [h.id for h in hits] == kept and stale == dropped and total == len(ranked)  # no score cuts a hit
        assert len(hits) <= k and stale >= 0
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
        rows = {r[0]: r for r in self.idx.rows()}
        for h in hits:
            doc = parse.read(Path(h.path))                                      # as the index converted it
            span = ("\n".join(parse.lines(doc.blocks[0].text)[h.start - 1:h.end]) if h.unit == "line" else
                    "\n\n".join(b.text for b in doc.blocks if h.start <= b.start <= h.end))
            assert all(part.strip() in span for part in h.text.split("\n\n")), h.where
            row = rows[h.id]                                                   # the stored chunk, field by field
            if h.id in relocated:
                assert (h.start, h.end, h.text) == relocated[h.id]
                assert h.path == row[1] and h.unit == row[3] and h.section == ""
                assert h.before is None and h.after is None
            else:
                assert (h.path, h.section, h.unit, h.start, h.end, h.text) == row[1:7]
            assert h.file_changed == (index_module.sha256(Path(h.path)) != row[7])
            for side, i in (("before", h.id - 1), ("after", h.id + 1)):              # same file and section only
                ctx = getattr(h, side)
                near = rows.get(i)
                assert ctx == (search.Context(near[4], near[5], near[6])
                               if not h.file_changed and near and near[1] == h.path and near[2] == h.section else None)

    def store_matches_the_index(self):
        """The extracted-text store holds one text per indexed content that is not its own bytes, each the file's
        converted text, and nothing else."""
        want = {}
        for path, status, note, sha in self.idx.read("SELECT path, status, note, sha FROM files ORDER BY path"):
            if extracted.needed(path, status, note):
                want.setdefault(sha, path)
        assert {r[0] for r in self.idx.read("SELECT sha FROM texts")} == set(want)
        assert _store(self.home) == sorted(f"{sha}.txt" for sha in want)
        for sha, path in want.items():
            text, units = extracted.render(parse.read(Path(path)))
            assert (self.home / extracted.FOLDER / f"{sha}.txt").read_bytes() == text.encode("utf-8")
            assert self.idx.extracted(path)[1] == units

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
    edited file, in whatever encoding the file is written → returned and flagged; deleted between the existence check
    and the read → gone."""
    bun = "# Bún\n\nBún chả nướng than hoa.\n"
    (notes / "bun.md").write_bytes(bun.encode("utf-16"))                 # a byte order mark names the encoding
    idx, s = make(home)
    idx.update([notes])
    (notes / "bun.md").write_bytes((bun + "\nThêm rau sống.\n").encode("utf-16"))
    hits, stale, _ = s.find("bún chả nướng than", k=1)
    assert stale == 0 and hits[0].path.endswith("bun.md") and hits[0].file_changed
    (notes / "ml.md").write_text("# Học máy\n\nSomething else entirely.\n")
    hits, stale, _ = s.find("gradient descent trọng số", k=2)
    assert stale == 1 and not any(h.path.endswith("ml.md") for h in hits)
    (notes / "food.md").write_text("# Phở\n\nNước dùng hầm xương bò trong nhiều giờ.\n\nThêm hành.\n")
    hits, _, _ = s.find("nước dùng xương bò", k=1)
    assert hits[0].path.endswith("food.md") and hits[0].file_changed

    (notes / "two.md").write_text("# Two\n\nalpha beta gamma\n\ndelta epsilon zeta\n")
    idx.update([notes])
    (notes / "two.md").write_text("# Two\n\n\nalpha beta gamma\n\n\n\ndelta epsilon zeta\n")   # paragraphs re-spaced
    hits, stale, _ = s.find("alpha beta gamma delta epsilon zeta", k=1)
    assert stale == 0 and hits[0].path.endswith("two.md") and hits[0].file_changed
    (notes / "two.md").write_text("# Two\n\ngamma beta alpha\n\nzeta epsilon delta\n")       # same words, not the text
    hits, stale, _ = s.find("alpha beta gamma delta epsilon zeta", k=1)
    assert stale == 1 and not any(h.path.endswith("two.md") for h in hits)
    (notes / "ml.md").write_text("# Học máy\n\nGradient descent cập nhật trọng số theo đạo hàm.\n")
    (notes / "three.md").write_text("# Three\n\nalpha beta gamma\n")
    (notes / "four.md").write_text("# Four\n\nalpha beta delta\n")
    idx.update([notes])
    (notes / "three.md").write_text("# Three\n\nsomething else\n")                    # two excerpts go stale
    (notes / "four.md").write_text("# Four\n\nsomething else too\n")
    ranked, _, _ = s.candidates("alpha beta gamma delta")
    hits, stale, total = s.find("alpha beta gamma delta", k=len(ranked))
    assert stale == 2 and len(hits) == total - 2                                          # counted, the rest kept

    def vanished(path):
        raise FileNotFoundError(2, "No such file or directory")
    monkeypatch.setattr(search, "sha256", vanished)
    hits, stale, _ = s.find("nước dùng xương bò", k=3)
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
    h = hits[0]
    assert (h.before, h.after) == s.context(rows[1][0], rows[1][1], "S")
    assert "para 0" in h.before.text and "para 2" in h.after.text and h.before.end < h.start <= h.end < h.after.start
    assert h.span == f"{rows[1][1]}:L{h.before.start}-{h.after.end}"

    def paras(cid):
        return [None if c is None else [f"para {i}" in c.text for i in range(4)].index(True)
                for c in s.context(cid, rows[0][1], "S")]
    assert [paras(r[0]) for r in rows[:4]] == [[None, 1], [0, 2], [1, 3], [2, None]]   # not section T, not b.md


def test_pages_show_every_ranked_candidate_once_in_order_and_no_score_cuts_one(home, mixed):
    """No verdict and no threshold (on the gold dev split the best reranker score told answerable questions from
    unanswerable ones no better than chance, AUROC 0.56 [0.37, 0.75], and the 0.25 cut with k = 5 kept 0.60 of the
    answering passages from the model against 0.71 in the top 10, 2026-10-04): the pages, in order, are the ranked
    candidates, each once; a page past the end is empty; an empty index has none."""
    idx, s = make(home)
    idx.update([mixed])
    ranked, _, _ = s.candidates("alpha first cell phở")
    pages = [s.find("alpha first cell phở", k=2, page=p) for p in range(1, 5)]
    assert all(total == len(ranked) and stale == 0 for _, stale, total in pages)
    assert [h.id for hits, _, _ in pages for h in hits] == [r[0] for r in ranked]
    assert pages[-1][0] == [] and len(ranked) <= 2 * 3
    assert [h.id for h in s.find("alpha first cell phở", k=None)[0]] == [r[0] for r in ranked]   # k None: all
    many = home.parent / "many"
    many.mkdir()
    for i in range(25):
        (many / f"n{i:02d}.md").write_text(f"# Note {i}\n\nalpha topic number {i} with words of its own w{i}x\n")
    midx, ms = make(home.parent / "many-home")
    midx.update([many])
    ranked, _, _ = ms.candidates("alpha topic")
    assert len(ranked) > search.PAGE and [h.id for h in ms.find("alpha topic", k=None)[0]] == [r[0] for r in ranked]
    with pytest.raises(ValueError, match=r"k and page start at 1 \(got k=0, page=1\)"):
        s.find("q", k=0)
    with pytest.raises(ValueError, match=r"k and page start at 1 \(got k=10, page=0\)"):
        s.find("q", page=0)
    (home.parent / "empty").mkdir()
    idx, s = make(home.parent / "empty-home")
    idx.update([home.parent / "empty"])                  # an index of nothing: no candidates, no page, no error
    assert s.find("anything", k=3) == ([], 0, 0) and s.find("anything", page=2) == ([], 0, 0)
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
    """The reranker and BM25 see "file place + heading path + text"; BM25 matches words as typed (no folding); a low
    score is shown, after the higher ones, never cut."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# Tokenizer notes\n\nmerges pairs of symbols\n\n" + "w " * 745 + "\n\nsecond chunk\n")
    (root / "b.md").write_text("# Cooking\n\nhọc nấu phở\n")
    (root / "c.md").write_text("# Plain\n\nhoc bai\n")
    spy = SpyReranker(lambda t: 0.9 if "Tokenizer" in t else 0.01)
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    s = Searcher(idx, reranker=spy)
    hits, _, total = s.find("tokenizer", k=5)
    assert [Path(h.path).name for h in hits][:3] == ["a.md"] * 3 and len(hits) == total == 5
    assert [h.score for h in hits] == pytest.approx([0.9] * 3 + [0.01] * 2)
    assert "n/a.md\nTokenizer notes\n# Tokenizer notes\n\nmerges pairs of symbols" in spy.seen
    ids, bm = idx.bm25()
    paths = {r[0]: r[1] for r in idx.rows()}
    rows = {r[0]: r for r in idx.rows()}
    matched = [rows[int(i)] for i, x in zip(ids, bm.scores("tokenizer")) if x > 0]
    assert len(matched) == 3 and sum("Tokenizer" not in r[6] for r in matched) == 2   # matched by its heading alone
    assert [paths[int(i)] for i, x in zip(ids, bm.scores("học")) if x > 0] == [str(root / "b.md")]
    assert [paths[int(i)] for i, x in zip(ids, bm.scores("hoc")) if x > 0] == [str(root / "c.md")]


def test_the_walk_leaves_out_what_is_not_notes_and_counts_it(home, tmp_path, monkeypatch):
    """The index walks by the rules of files.py: folders a tool made, hidden names and symlinked folders are not
    entered; a folder a person named (data, runs…) is; protected folders and files are only counted; a folder with
    more than MAX_FILES_PER_FOLDER files is a likely dataset. Everything left out except a protected path is reported
    and kept with its reason, so `why_not` can say why a path is missing; a protected path is never named."""
    monkeypatch.setattr(files, "MAX_FILES_PER_FOLDER", 3)
    root, outside = tmp_path / "n", tmp_path / "outside"
    for d in ("node_modules", ".git", "keys/.ssh", "keys/.aws", "keys/zz", "reviews", "ok", "three", "zz", "data"):
        (root / d).mkdir(parents=True)
    outside.mkdir()
    (outside / "o.md").write_text("# O\n\noutside\n")
    os.symlink(outside, root / "linked")
    (root / "node_modules" / "x.md").write_text("# X\n\nvendored\n")
    (root / ".git" / "y.md").write_text("# Y\n\ngit\n")
    (root / "keys" / ".ssh" / "id.md").write_text("# K\n\nkey\n")
    (root / "ok" / ".env").write_text("A=1\n")
    (root / "zz" / ".notes.md").write_text("# H\n\nhidden\n")      # zz: below the 3-file limit
    for i in range(4):
        (root / "reviews" / f"{i}.txt").write_text(f"review {i}\n")
    for i in range(3):                                             # exactly the limit: notes, kept
        (root / "three" / f"{i}.md").write_text(f"# T{i}\n\nthree {i}\n")
    (root / "ok" / "note.md").write_text("# N\n\nnote\n")
    (root / "ok" / "tool.exe").write_bytes(b"MZ")
    (root / "zz" / "metrics.csv").write_text("step,loss\n1,0.9\n")    # read by the tools, not ranked (yet)
    (root / "keys" / "zz" / "k.md").write_text("# K\n\nafter the protected folders\n")
    (root / "zz" / "last.md").write_text("# L\n\nafter the dataset\n")
    (root / "data" / "results.md").write_text("# R\n\naccuracy 0.91\n")    # a name a person chose: walked
    idx = Index(home, embedder=FakeEmbedder())
    r = idx.update([root])
    want = {root / "ok" / "note.md", root / "keys" / "zz" / "k.md", root / "zz" / "last.md",
            root / "data" / "results.md"} | {root / "three" / f"{i}.md" for i in range(3)}
    assert {row[1] for row in idx.rows()} == {str(w) for w in want}
    assert {r_[0] for r_ in idx.read("SELECT path FROM files")} == {str(w) for w in want}   # no .exe either
    assert r.denied == 3                                           # .ssh, .aws and the .env file
    assert r.problems["large_folder"] == [f"{root / 'reviews'} (4 files)"]
    assert sorted(r.problems["tool_made"]) == [str(root / ".git"), str(root / "node_modules")]
    assert r.problems["symlinked_folder"] == [str(root / "linked")]
    assert r.problems["hidden"] == [str(root / "zz" / ".notes.md")]
    assert r.problems["unsupported"] == [str(root / "ok" / "tool.exe")]
    assert r.problems["data_type"] == [str(root / "zz" / "metrics.csv")]
    assert idx.why_not(root / "zz" / "metrics.csv") == "data_type: .csv"
    assert idx.why_not(root / "node_modules" / "x.md") == "tool_made: node_modules: made by a package manager"
    assert idx.why_not(root / ".git" / "y.md") == "tool_made: version control"
    assert idx.why_not(root / "linked" / "o.md") == "symlinked_folder"
    assert idx.why_not(root / "reviews" / "2.txt") == "large_folder: 4 files"
    assert idx.why_not(root / "ok" / "tool.exe") == "unsupported: .exe"
    assert idx.why_not(root / "zz" / ".notes.md") == "hidden"
    assert idx.why_not(root / "ok" / "note.md") is None                # indexed
    assert idx.why_not(root / "keys" / ".ssh" / "id.md") is None       # protected: never named
    (root / "ok" / "tool.exe").unlink()
    idx.update([root])
    assert idx.why_not(root / "ok" / "tool.exe") is None               # the record follows the folders as they are now


def test_the_users_patterns_reach_the_index_and_a_bad_one_stops_it(home, tmp_path):
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nkeep\n")
    (root / "draft.md").write_text("# D\n\ndraft\n")
    guard.set_roots([root])
    cfg = guard.config()
    cfg["exclude"] = ["draft.md"]
    guard.config_path().write_text(json.dumps(cfg))
    idx = Index(home, embedder=FakeEmbedder())
    r = idx.update([root])
    assert {row[1] for row in idx.rows()} == {str(root / "a.md")}
    assert idx.why_not(root / "draft.md") == "user_excluded: config.json exclude: draft.md"
    guard.set_roots([root])                                        # declaring folders keeps the user's patterns
    assert guard.config()["exclude"] == ["draft.md"] and r.problems["user_excluded"] == [str(root / "draft.md")]
    cfg["exclude"] = "draft.md"
    guard.config_path().write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="list of patterns"):
        idx.update([root])


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
    """Counts the documents it embeds (a query, such as Index.embed's probe of the vectors' length, is not one)."""

    def __init__(self, signature):
        self.signature, self.embedded = signature, 0

    def documents(self, texts):
        self.embedded += len(texts)
        return super().documents(texts)

    def queries(self, texts):
        return FakeEmbedder.documents(self, texts)


def test_vectors_of_another_embedder_are_never_mixed_in(home, notes, tmp_path, monkeypatch):
    """Another embedder (the user's endpoint changed: its model or where it runs) drops the vectors alone: the chunks
    stay, no file is read again, the new embedder fills them in."""
    runs = [CountingEmbedder(sig) for sig in ("model-a at http://a/v1", "model-a at http://a/v1", "model-b at http://a/v1")]
    reads = []
    real = index_module.parse.read
    monkeypatch.setattr(index_module.parse, "read", lambda *a, **k: reads.append(a[0]) or real(*a, **k))
    for e in runs:
        Index(home, embedder=e).update([notes])
    assert [e.embedded for e in runs] == [2, 0, 2] and len(reads) == 2           # files read once, by the first
    monkeypatch.undo()

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
    assert Index(other_home, embedder=None).signature() == ""                  # none set: nothing to match


def test_with_no_embedder_the_index_is_lexical_and_the_vectors_wait(home, notes):
    """No embedding endpoint (the default): the chunks are indexed and searched by their words (BM25); vectors kept
    from an embedder set before are neither used nor dropped — set again, it finds them all."""
    first = CountingEmbedder("model-a at http://a/v1")
    Index(home, embedder=first).update([notes])
    lexical = Index(home, embedder=None)
    report = lexical.update([notes])
    assert report.chunks == 2 and report.embedded == 0 and report.unembedded == 0 and not report.embed_error
    hits, _, total = Searcher(lexical, reranker=None).find("nước dùng xương bò", k=None)
    assert total == 1 and hits[0].path.endswith("food.md") and hits[0].score is None     # the words alone
    again = CountingEmbedder("model-a at http://a/v1")
    Index(home, embedder=again).update([notes])
    assert again.embedded == 0                                                    # the old vectors still fit


def test_an_embedding_endpoint_that_fails_leaves_the_index_searchable_and_the_next_update_fills_it(home, notes):
    """The chunks are committed before their vectors are asked for: an endpoint that is down leaves every file
    indexed and searched by its words, says why, and the next update (or Index.embed) embeds what is missing."""
    from peresearch.zetokrag.embed import EndpointError

    class Down(FakeEmbedder):
        signature = "m at http://down/v1"

        def documents(self, texts):
            raise EndpointError("http://down/v1/embeddings: ConnectError after 3 tries")
    idx = Index(home, embedder=Down())
    report = idx.update([notes])
    assert report.chunks == 2 and report.embedded == 0 and report.unembedded == 2
    assert report.embed_error == "http://down/v1/embeddings: ConnectError after 3 tries"
    assert Searcher(idx, reranker=None).find("gradient descent", k=None)[0][0].path.endswith("ml.md")
    up = CountingEmbedder("m at http://down/v1")
    report = Index(home, embedder=up).update([notes])
    assert up.embedded == 2 and report.embedded == 2 and report.unembedded == 0 and not report.embed_error


def test_an_update_counts_the_chunks_embedded_by_their_head(home, notes):
    e = CountingEmbedder("m at http://a/v1")
    e.halved = 0

    def documents(texts):
        e.halved += 1                                                    # as embed.Embedder does when it halves
        return FakeEmbedder.documents(e, texts)
    e.documents = documents
    assert Index(home, embedder=e).update([notes]).halved == 1


def test_embedding_stops_when_asked_and_goes_on_where_it_stopped(home, tmp_path):
    """Index.embed commits each batch: stopped, it keeps what it embedded; run again, it embeds the rest, each chunk
    once."""
    root = tmp_path / "many"
    root.mkdir()
    for i in range(12):
        (root / f"n{i:02d}.md").write_text(f"# Note {i}\n\nword{i} about topic {i}\n")
    e = CountingEmbedder("m at http://a/v1")
    idx = Index(home, embedder=None)
    idx.update([root])
    idx = Index(home, embedder=e)
    calls = []
    with pytest.raises(index_module.Stopped):
        idx.embed(stop=lambda: len(calls) >= 2, progress=lambda done, total: calls.append((done, total)), batch=5)
    assert calls == [(5, 12), (10, 12)] and e.embedded == 10
    assert idx.embed(batch=5) == 2 and e.embedded == 12 and idx.embed() == 0           # the rest, then nothing


def test_a_partly_embedded_index_ranks_every_chunk(home, tmp_path):
    """While the vectors are filled, a chunk without one is ranked by its words (last in the dense view): a strong
    word match is still found."""
    root = tmp_path / "part"
    root.mkdir()
    (root / "a.md").write_text("# A\n\ncommon words here\n")
    (root / "b.md").write_text("# B\n\nzymurgy fermentation brewing\n")
    idx = Index(home, embedder=None)
    idx.update([root])
    e = CountingEmbedder("m at http://a/v1")
    idx = Index(home, embedder=e)
    with pytest.raises(index_module.Stopped):
        idx.embed(stop=lambda: e.embedded >= 1, batch=1)                          # one of the two embedded
    ids, _, vecs = idx.snapshot()
    assert len(ids) == 2 and np.isnan(vecs).all(axis=1).tolist().count(True) == 1    # the waiting one marked, not 0
    hits, _, _ = Searcher(idx, reranker=None).find("zymurgy fermentation", k=None)
    assert hits and hits[0].path.endswith("b.md")


def test_a_chunk_without_a_vector_is_last_in_the_meaning_view_even_when_similarities_are_negative(home, notes):
    """Cosines run from -1 to 1: a chunk waiting for its vector takes the lowest similarity there is, not 0 (which
    would put it above every chunk unlike the query)."""
    idx = Index(home, embedder=CountingEmbedder("m at http://a/v1"))
    idx.update([notes])
    s = Searcher(idx, reranker=None)
    vecs = np.array([[1.0, 0.0], [np.nan, np.nan], [0.0, 1.0]], dtype=np.float32)
    idx._embedder.queries = lambda texts: np.array([[-1.0, -0.5]], dtype=np.float16)
    assert s._dense("q", vecs).tolist() == [-1.0, -1.0, -0.5] and s.words_only == ""


def test_an_endpoint_that_fails_during_a_search_leaves_the_search_working_and_says_why(home, notes):
    """The embedding endpoint down at the question: ranked by words, said; the rerank endpoint down: the fused
    order, no score, said — for excerpts and for documents alike."""
    from peresearch.zetokrag.embed import EndpointError
    idx = Index(home, embedder=CountingEmbedder("m at http://a/v1"))
    idx.update([notes])

    def down(*a):
        raise EndpointError("http://a/v1/embeddings: ConnectError after 3 tries")
    idx._embedder.queries = down
    s = Searcher(idx, reranker=None)
    assert s.find("gradient descent", k=None)[0][0].path.endswith("ml.md")
    assert s.words_only == "the embedding endpoint did not answer (http://a/v1/embeddings: ConnectError after 3 tries)"

    class Down:
        def scores(self, query, texts):
            raise EndpointError("http://r/v1/rerank: 503 after 3 tries")
    r = Searcher(idx, reranker=Down())
    hits = r.find("gradient descent", k=None)[0]
    assert hits and all(h.score is None for h in hits) and r.unranked.endswith("503 after 3 tries)")
    docs = r.documents("gradient descent")
    assert docs and all(d.score is None for d in docs)


def test_the_index_and_the_searcher_take_the_users_endpoints_by_default(home, monkeypatch):
    from peresearch.zetokrag import embed
    monkeypatch.setenv("PERESEARCH_EMBED_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("PERESEARCH_EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("PERESEARCH_RERANK_URL", "http://localhost:8081/v1")
    idx = Index(home)
    assert isinstance(idx.embedder, embed.Embedder) and idx.signature() == "bge-m3 at http://localhost:11434/v1"
    assert isinstance(Searcher(idx).reranker, embed.Reranker)
    for name in ("PERESEARCH_EMBED_URL", "PERESEARCH_RERANK_URL"):
        monkeypatch.delenv(name)
    assert Index(home).embedder is None and Searcher(Index(home)).reranker is None


def test_a_query_vector_unlike_the_stored_ones_falls_back_to_words_and_the_next_embedding_redoes_them(home, notes):
    """The same model name, another model behind it (vectors of another length): the search goes on by words, and
    the next embedding pass drops the stored vectors and makes them again."""
    idx = Index(home, embedder=CountingEmbedder("m at http://a/v1"))
    idx.update([notes])

    class Wider(CountingEmbedder):
        dim = 512
    wide = Wider("m at http://a/v1")
    swapped = Index(home, embedder=wide)
    hits, _, _ = Searcher(swapped, reranker=None).find("gradient descent", k=None)
    assert hits[0].path.endswith("ml.md")
    assert swapped.embed() == 2 and wide.embedded == 2                          # a query probes it, then every chunk
    assert {len(np.frombuffer(v, np.float16)) for (v,) in swapped.read("SELECT vec FROM vectors")} == {512}


# --- concurrency: searches go on while the index changes -------------------------------------------------------------

def test_another_process_can_search_during_an_update_and_sees_it_after(home, notes):
    """While another connection's update is uncommitted, a search sees the old index whole; after, the new one."""
    idx, s = make(home)
    idx.update([notes])
    s.find("phở nước dùng", k=1)                                       # this process's snapshot is built
    (notes / "new.md").write_text("# Đà Lạt\n\nĐồi thông và hồ Xuân Hương.\n")
    (notes / "food.md").unlink()
    during, checks = [], []

    def searching():                                                   # before each file, inside the open update
        checks.append(1)
        if len(checks) == 2:                                           # one file written, nothing committed yet
            during.append((s.find("gradient descent", k=3)[0], s.find("đồi thông hồ xuân hương", k=3)[0]))
        return False
    Index(home, embedder=FakeEmbedder()).update([notes], stop=searching)
    assert len(during) == 1
    unchanged, uncommitted = during[0]
    assert unchanged and unchanged[0].path.endswith("ml.md") and not any(h.path.endswith("new.md") for h in uncommitted)
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


def test_office_documents_and_pages_are_indexed_and_a_new_reader_reads_its_files_again(home, tmp_path, monkeypatch):
    """Word, PowerPoint and HTML pages are indexed by their text (Word cut by its headings, a slide a page), their
    text kept in the store; a workbook is data (like a .csv: read by the tools, not ranked). When one kind's reader
    changes (parse.READERS), the files of that kind are read again, and only those: the HTML pages indexed raw before
    2026-10-05 would otherwise keep their markup."""
    docx = pytest.importorskip("docx")
    pptx = pytest.importorskip("pptx")
    openpyxl = pytest.importorskip("openpyxl")
    root = tmp_path / "n"
    root.mkdir()
    d = docx.Document()
    d.add_heading("Tokenizers", level=1)
    d.add_paragraph("Byte pair encoding merges the most frequent pair of symbols.")
    d.save(root / "notes.docx")
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Results"
    slide.placeholders[1].text_frame.text = "Unigram wins on short texts"
    deck.save(root / "talk.pptx")
    wb = openpyxl.Workbook()
    wb.active.append(["model", "score"])
    wb.save(root / "scores.xlsx")
    (root / "page.html").write_text("<html><body><nav>Home</nav><article><h1>WordPiece</h1>" + "".join(
        f"<p>Point {i}: WordPiece picks the merge that raises the likelihood of the training data most.</p>"
        for i in range(1, 6)) + "</article></body></html>")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    rows = {Path(r[1]).name: r for r in idx.rows()}
    assert set(rows) == {"notes.docx", "talk.pptx", "page.html"}                      # the workbook is data
    assert rows["notes.docx"][2] == "Tokenizers" and rows["notes.docx"][3] == "line"   # section, unit
    assert rows["talk.pptx"][3:5] == ("page", 1) and "Unigram wins" in rows["talk.pptx"][6]
    assert rows["page.html"][6].startswith("# WordPiece") and "Home" not in rows["page.html"][6]
    assert idx.left_out("data_type", 5) == ([str(root / "scores.xlsx")], 1)
    stored = {Path(p).name for (p,) in idx.read("SELECT path FROM files WHERE sha IN (SELECT sha FROM texts)")}
    assert stored == {"notes.docx", "talk.pptx", "page.html"}                         # their text is not their bytes
    read_again = []
    real = parse.read
    monkeypatch.setattr(parse, "read", lambda path: read_again.append(Path(path).name) or real(path))
    idx.update([root])
    assert read_again == []                                                           # nothing changed
    monkeypatch.setitem(parse.READERS, "html", parse.READERS["html"] + 1)
    idx.update([root])
    assert read_again == ["page.html"]                                                # that kind only


def test_is_data_names_what_a_data_file_looks_like():
    """The index asks the data filter (test_datafilter.py holds its cases) about plain text only."""
    def doc(name, text):
        return parse.Doc(Path(name), "ok", [parse.Block(text, "line", 1)])
    urls = "".join(f"http://x.org/{i}\n" for i in range(60))
    log = "".join(f"081109 2036{i:02d} 148 INFO dfs.DataNode: served block {i}\n" for i in range(60))
    prose = "".join(f"Line {i} of a description, in words that a person would write.\n" for i in range(60))
    assert index_module.is_data(doc("u.txt", urls)) == "URLs or numbers alone"
    assert index_module.is_data(doc("run.txt", log)) == "log"
    assert index_module.is_data(doc("p.txt", prose)) is None
    assert index_module.is_data(doc("u.md", urls)) is None and index_module.is_data(doc("u.py", urls)) is None
    assert index_module.is_data(parse.Doc(Path("p.pdf"), "ok", [parse.Block(log, "page", 1)])) is None


def test_a_data_file_is_kept_out_with_its_reason_and_an_older_index_gains_the_column(home, tmp_path, monkeypatch):
    root = tmp_path / "n"
    root.mkdir()
    (root / "0019.txt").write_text("".join(f"{i} 2 Car 0 0 -1.79 296.74 161.75 455.22 292.37 2.00 1.82 4.43\n"
                                           for i in range(40)))
    (root / "note.md").write_text("# N\n\nnote\n")
    (root / "empty.txt").write_text("")                                 # read, nothing to chunk, not data
    (root / "big.md").write_text("y" * 5000)
    (root / "README").write_text("no extension")
    (root / "blank.ipynb").write_text(json.dumps({"cells": []}))
    home.mkdir(parents=True)
    old = sqlite3.connect(home / "index.sqlite")                       # the layout before the columns existed
    old.executescript(index_module.SCHEMA.replace(", withheld INTEGER, reason TEXT", ", withheld INTEGER")
                      .replace(", sha TEXT, place TEXT DEFAULT ''", ", sha TEXT"))
    old.execute(f"PRAGMA user_version={index_module.VERSION}")
    old.close()
    idx = Index(home, embedder=FakeEmbedder())
    monkeypatch.setattr(index_module.parse, "MAX_BYTES", 4000)                     # 0019.txt holds 2,560
    report = idx.update([root])
    assert report.problems["data_file"] == [str(root / "0019.txt")]
    assert idx.why_not(root / "0019.txt") == "data: numbers and codes"
    assert {(r[1], r[8]) for r in idx.rows()} == {(str(root / "note.md"), "n/note.md")}    # with its place
    assert idx.why_not(root / "big.md") == "too_large" and idx.why_not(root / "empty.txt") is None
    assert idx.read("SELECT status FROM files WHERE path=?", (str(root / "empty.txt"),)) == [("ok",)]
    assert idx.why_not(root / "README") == "unsupported: no extension"
    assert idx.read("SELECT status FROM files WHERE path=?", (str(root / "blank.ipynb"),)) == [("ok",)]


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
    texts = {r[0]: idx.ranking_text(r) for r in idx.rows()}
    assert spy.seen == [texts[int(ids[i])] for i in fused[:search.N_RERANK]]


class NoReranker:
    def scores(self, query, texts):
        pytest.fail("the reranker was used at depth 0")


@pytest.mark.parametrize("depth", [0, 1, 10, search.N_RERANK - 1, search.N_RERANK])
def test_the_reranker_reorders_only_the_first_depth_fused_candidates(home, tmp_path, depth):
    """PERESEARCH_RERANK_DEPTH (for a machine without a GPU): the first `depth` of the N_RERANK fused candidates are
    reordered by the reranker, best first; the rest follow in fused order, without a score. At 0 the reranker is
    never used (nor loaded)."""
    root = tmp_path / "n"
    root.mkdir()
    for i in range(search.N_RERANK + 10):                 # distinct words: no two notes are near-copies
        (root / f"{i:02d}.md").write_text(f"# N{i}\n\n" + " ".join(["rank"] * (i + 1) + [f"w{i}q{j}" for j in range(30)])
                                          + "\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    score = lambda t: int(t.split("# N")[1].split()[0]) % 7 / 10       # noqa: E731  (ties: kept in fused order)
    spy = SpyReranker(score) if depth else NoReranker()
    s = Searcher(idx, reranker=spy, depth=depth)
    rows, scores = s.rank("rank")
    ids, bm, vecs = idx.snapshot()
    q = FakeEmbedder().queries(["rank"])[0].astype(np.float32)
    fused, _ = __import__("peresearch.zetokrag.core", fromlist=["x"]).fuse_minmax(bm.scores("rank"), vecs @ q,
                                                                               search.RHO, search.N_FIRST)
    fused = [int(ids[i]) for i in fused[:search.N_RERANK]]
    texts = {r[0]: idx.ranking_text(r) for r in idx.rows()}
    head = sorted(fused[:depth], key=lambda c: -score(texts[c]))
    assert [r[0] for r in rows] == head + fused[depth:]
    if depth:
        assert spy.seen == [texts[c] for c in fused[:depth]]
    assert not np.isnan(scores[:depth]).any() and np.isnan(scores[depth:]).all()
    hits, _, total = s.find("rank", k=search.N_RERANK)
    assert total == search.N_RERANK and [h.id for h in hits] == head + fused[depth:]
    assert all(h.score is not None for h in hits[:depth]) and all(h.score is None for h in hits[depth:])


def test_the_rerank_depth_is_a_setting_from_0_to_30(home, monkeypatch):
    """Read like every setting (the environment, then settings.env); 30 when unset; anything else is refused with
    the setting's name and range, before a search starts."""
    assert search.rerank_depth() == search.N_RERANK == 30
    home.mkdir(parents=True)
    (home / "settings.env").write_text("PERESEARCH_RERANK_DEPTH=10\n")
    assert search.rerank_depth() == 10
    idx = Index(home, embedder=FakeEmbedder())
    r = FakeReranker()
    assert Searcher(idx, r).depth == 10 and Searcher(idx, r, depth=0).depth == 0
    assert Searcher(idx, reranker=None).depth == 0                       # no rerank endpoint: nothing reranked
    for good, value in ((" 0 ", 0), ("30", 30), ("7", 7)):
        monkeypatch.setenv("PERESEARCH_RERANK_DEPTH", good)
        assert search.rerank_depth() == value and Searcher(idx, r).depth == value
    for bad in ("31", "-1", "ten", "10.5", "1e1"):
        monkeypatch.setenv("PERESEARCH_RERANK_DEPTH", bad)
        with pytest.raises(ValueError, match=rf"PERESEARCH_RERANK_DEPTH must be a whole number from 0 to 30 \(got '{bad}'\)"):
            Searcher(idx, r)
    for bad in (31, -1):
        with pytest.raises(ValueError, match="from 0 to 30"):
            Searcher(idx, depth=bad)


def test_a_broken_internal_check_stops_the_search_or_the_update_where_it_breaks(home, tmp_path):
    """Scores the verdict cannot read (a reranker giving logits, not probabilities) stop the search; a vector
    without its chunk stops the update and rolls all of it back, so nothing half-done is kept."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\ntokenizer notes\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    for bad in (3.0, 1.5, -0.5):
        with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
            Searcher(idx, reranker=SpyReranker(lambda t: bad)).find("tokenizer")
        with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
            Searcher(idx, reranker=SpyReranker(lambda t: bad)).documents("tokenizer")
    assert [h.score for h in Searcher(idx, reranker=SpyReranker(lambda t: 1.0)).find("tokenizer")[0]] == [1.0]

    class Narrow(FakeEmbedder):                    # an embedder unlike the index's: no vector product is attempted
        dim = 128

    class Swapped(Narrow):                         # the same name, another model behind it (vectors of another length)
        signature = "FakeEmbedder"

    for cls, why in ((Narrow, "another embedder's"), (Swapped, "another model's")):
        for method in ("find", "documents"):
            searcher = Searcher(Index(home, embedder=cls()), reranker=FakeReranker())
            got = getattr(searcher, method)("tokenizer")
            assert got and why in searcher.words_only                       # by its words, and said why
    db = sqlite3.connect(home / "index.sqlite")
    db.execute("INSERT INTO vectors VALUES (?, ?)", (10**6, np.zeros(FakeEmbedder.dim, np.float16).tobytes()))
    db.commit()
    (root / "b.md").write_text("# B\n\nanother note\n")
    with pytest.raises(AssertionError, match="1 vectors without a chunk"):
        idx.update([root])
    assert {r[1] for r in idx.rows()} == {str(root / "a.md")}       # b.md's chunks rolled back with the update


# --- the extracted-text store: what plain-text search reads instead of the file -------------------------------------

def _pdf(path, pages):
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    for text in pages:
        d.new_page().insert_text((72, 300), text)
    d.save(path)


def _store(home):
    folder = home / extracted.FOLDER
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


@pytest.fixture
def mixed(tmp_path):
    """A PDF and its copy, a notebook, a UTF-16 text file (each needs a converted text) and a UTF-8 note (needs none)."""
    root = tmp_path / "mixed"
    root.mkdir()
    _pdf(root / "paper.pdf", ["alpha on page one", "beta on page two\nsecond line", "gamma on page three"])
    shutil.copyfile(root / "paper.pdf", root / "copy.pdf")
    (root / "run.ipynb").write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": "# Run\nfirst cell"},
        {"cell_type": "code", "source": ["x = 1\n", "x"], "outputs": [{"output_type": "stream", "text": "1\n"}]}]}))
    (root / "old.txt").write_bytes("phở bò\r\nbún chả\nnem rán\n".encode("utf-16"))
    (root / "plain.md").write_text("# Plain\n\nalready text\n")
    return root


def test_text_that_is_not_the_files_bytes_is_kept_for_plain_text_search(home, mixed):
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([mixed])
    sha = {name: index_module.sha256(mixed / name) for name in ("paper.pdf", "run.ipynb", "old.txt", "plain.md")}
    assert _store(home) == sorted(f"{sha[n]}.txt" for n in ("paper.pdf", "run.ipynb", "old.txt"))
    assert idx.extracted(mixed / "plain.md") is None                        # its own bytes are its text
    assert idx.extracted(mixed / "copy.pdf") == idx.extracted(mixed / "paper.pdf")   # copies share one text
    places = {}
    for name in ("paper.pdf", "run.ipynb", "old.txt"):
        path, units = idx.extracted(mixed / name)
        assert path == home / extracted.FOLDER / f"{sha[name]}.txt"
        text = path.read_bytes().decode("utf-8")                         # UTF-8 on every system, "\n" untouched
        places[name] = [(line, extracted.locate(units, n)) for n, line in enumerate(text.split("\n")[:-1], 1)]
    assert places["paper.pdf"] == [("alpha on page one", ("page", 1)), ("beta on page two", ("page", 2)),
                                   ("second line", ("page", 2)), ("gamma on page three", ("page", 3))]
    assert places["run.ipynb"] == [("# Run", ("cell", 1)), ("first cell", ("cell", 1)), ("x = 1", ("cell", 2)),
                                   ("x", ("cell", 2)), ("", ("cell", 2)), ("Output:", ("cell", 2)), ("1", ("cell", 2))]
    assert places["old.txt"] == [("phở bò\r", ("line", 1)), ("bún chả", ("line", 2)), ("nem rán", ("line", 3))]
    folder = home / extracted.FOLDER                                    # a copy of the user's text: theirs only
    assert folder.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in folder.iterdir())
    log = mixed / "server.log.txt"                    # a data file (kept out of ranking) not in UTF-8: grep needs it
    log.write_bytes("".join(f"081109 2036{i:02d} INFO dfs.DataNode: café block {i}\n" for i in range(30))
                    .encode("latin-1"))
    idx.update([mixed])
    assert idx.read("SELECT status FROM files WHERE path=?", (str(log),)) == [("data",)]
    path, units = idx.extracted(log)
    assert path.read_text(encoding="utf-8").startswith("081109 203600 INFO dfs.DataNode: café block 0\n")


def test_the_store_follows_the_index_and_mends_itself(home, mixed):
    embedded = []
    embedder = FakeEmbedder()
    real = embedder.documents
    embedder.documents = lambda texts: (embedded.extend(texts), real(texts))[1]
    idx = Index(home, embedder=embedder)
    idx.update([mixed])
    first = set(_store(home))
    old_sha = index_module.sha256(mixed / "old.txt")
    (mixed / "old.txt").write_bytes("phở gà\n".encode("utf-16"))           # a new content: a new text, the old gone
    idx.update([mixed])
    assert set(_store(home)) == first - {f"{old_sha}.txt"} | {f"{index_module.sha256(mixed / 'old.txt')}.txt"}
    paper = f"{index_module.sha256(mixed / 'paper.pdf')}.txt"
    (mixed / "paper.pdf").unlink()                                       # its copy still needs the text
    idx.update([mixed])
    assert paper in _store(home)
    (mixed / "copy.pdf").unlink()
    idx.update([mixed])
    assert paper not in _store(home)
    embedded.clear()
    want = _store(home)
    folder = home / extracted.FOLDER
    (folder / want[0]).unlink()                                          # removed by someone else
    (folder / f"{'0' * 64}.txt").write_text("stray")                     # left by an update that never committed
    (folder / f".{'1' * 64}.123.tmp").write_text("torn")
    idx.update([mixed])
    assert _store(home) == want and embedded == []                       # mended without embedding anything again
    idx.db.execute("DELETE FROM texts")                                  # an index from before the store
    shutil.rmtree(folder)
    idx.update([mixed])
    assert _store(home) == want and embedded == []


def test_an_update_that_fails_leaves_a_store_the_next_one_mends(home, mixed):
    embedder = FakeEmbedder()
    idx = Index(home, embedder=embedder)
    idx.update([mixed])
    (mixed / "old.txt").write_bytes("cơm tấm\n".encode("utf-16"))
    _pdf(mixed / "new.pdf", ["delta on page one"])
    real = embedder.documents

    def broken(texts):
        raise RuntimeError("embedder failed")
    embedder.documents = broken
    with pytest.raises(RuntimeError):
        idx.update([mixed])
    embedder.documents = real
    idx.update([mixed])
    need = {index_module.sha256(mixed / n) for n in ("paper.pdf", "run.ipynb", "old.txt", "new.pdf")}
    assert _store(home) == sorted(f"{s}.txt" for s in need)
    assert {r[0] for r in idx.read("SELECT sha FROM texts")} == need


def test_a_store_line_is_traced_to_its_place():
    units = [[1, "page", 3], [4, "page", 5], [6, "line", 1]]
    assert [extracted.locate(units, n) for n in range(1, 9)] == [
        ("page", 3), ("page", 3), ("page", 3), ("page", 5), ("page", 5), ("line", 1), ("line", 2), ("line", 3)]
    with pytest.raises(AssertionError):
        extracted.locate(units, 0)                                       # ripgrep counts from 1: a 0 is our bug


def test_a_new_filter_judges_the_indexed_plain_text_again_and_embeds_only_what_moved(home, tmp_path, monkeypatch):
    root = tmp_path / "n"
    root.mkdir()
    (root / "train_log.txt").write_text("".join(f"081109 2036{i:02d} INFO epoch {i} loss 0.{i:02d}\n" for i in range(40)))
    (root / "words.txt").write_text("".join(f"Line {i} of a description, in words that a person would write.\n"
                                            for i in range(40)))
    (root / "note.txt").write_text("".join(f"Gradient descent step {i} moves the weights against the gradient.\n"
                                           for i in range(40)))
    (root / "api_pb2.py").write_text("# Generated by the protocol buffer compiler.  DO NOT EDIT!\nx = 1\n")
    (root / "nums.txt").write_text("".join(f"{i / 7:.5f}\n" for i in range(40)))
    embedded = []
    embedder = FakeEmbedder()
    real = embedder.documents
    embedder.documents = lambda texts: (embedded.extend(texts), real(texts))[1]
    idx = Index(home, embedder=embedder)
    first = {"train_log.txt": None, "words.txt": "word list", "note.txt": None,     # an older filter's verdicts
             "nums.txt": "numbers and codes"}
    monkeypatch.setattr(index_module.datafilter, "reason", lambda name, text: first.get(name))
    idx.update([root])
    assert {Path(r[1]).name for r in idx.rows()} == {"train_log.txt", "note.txt", "api_pb2.py"}
    embedded.clear()
    monkeypatch.undo()                                                  # the real filter, one version on
    monkeypatch.setattr(index_module.datafilter, "VERSION", index_module.datafilter.VERSION + 1)
    report = idx.update([root])
    assert {Path(r[1]).name for r in idx.rows()} == {"words.txt", "note.txt"}
    assert idx.why_not(root / "train_log.txt") == "data: numbers and codes" and report.changed == 3
    assert idx.why_not(root / "api_pb2.py") == "data: generated code"
    assert idx.why_not(root / "nums.txt") == "data: URLs or numbers alone"        # still data: its new reason,
    assert report.changed == 3                                                   # not read or counted again
    assert embedded and all("Line" in t for t in embedded)               # only the file that became prose
    embedded.clear()
    reads, read = [], index_module.parse.read
    monkeypatch.setattr(index_module.parse, "read", lambda path: (reads.append(path), read(path))[1])
    idx.update([root])                                                  # same version: nothing judged again,
    assert embedded == [] and reads == [] and idx.update([root]).changed == 0   # nothing parsed again
    hashed, real_sha = [], index_module.sha256
    monkeypatch.setattr(index_module, "sha256", lambda p: (hashed.append(p), real_sha(p))[1])
    st_ = (root / "note.txt").stat()
    os.utime(root / "note.txt", ns=(st_.st_atime_ns, st_.st_mtime_ns + 10**9))   # touched: same bytes, a new time
    idx.update([root])
    idx.update([root])
    assert sorted(hashed) == sorted(list(root.iterdir()) * 2) and reads == []
    assert embedded == []
    assert idx.read("SELECT mtime FROM files WHERE path=?", (str(root / "note.txt"),)) == [
        (st_.st_mtime + 1,)]  # touched bytes stay reusable; the new timestamp is retained


def test_the_report_names_the_files_with_the_most_chunks(home, tmp_path, monkeypatch):
    root = tmp_path / "n"
    root.mkdir()
    (root / "big.md").write_text("# B\n\n" + "\n\n".join(f"paragraph {i} " + "w" * 1400 for i in range(6)) + "\n")
    (root / "mid.md").write_text("# M\n\n" + "\n\n".join(f"paragraph {i} " + "w" * 1400 for i in range(3)) + "\n")
    (root / "small.md").write_text("# S\n\nshort\n")
    monkeypatch.setattr(index_module, "LARGE_FILE_CHUNKS", 3)
    report = Index(home, embedder=FakeEmbedder()).update([root])
    assert report.largest == [(str(root / "big.md"), 6), (str(root / "mid.md"), 3)]     # the heading joins the first


def test_a_new_chunking_rebuilds_the_index_and_never_mixes_two(home, tmp_path, monkeypatch):
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\n```py\nx = 1\n\ny = 2\n```\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    assert [(r[4], r[5]) for r in idx.rows()] == [(1, 7)]                         # small: packed whole
    monkeypatch.setattr(index_module.chunk, "DEFAULT", index_module.chunk.Options("blocks-test", max_chars=12,
                                                                                    blocks=True))
    report = idx.update([root])
    recorded = f"blocks-test v{index_module.chunk.VERSION}"                     # the options and the cutter's version
    assert report.changed == 1 and idx.read("SELECT value FROM meta WHERE key='chunking'") == [(recorded,)]
    assert [(r[4], r[5]) for r in idx.rows()] == [(1, 1), (3, 7)]                 # the fence whole, alone
    monkeypatch.setattr(index_module.chunk, "VERSION", index_module.chunk.VERSION + 1)   # a fix to the cutter alone:
    assert idx.update([root]).changed == 1                                           # cut again too
    # what finds secrets changed (new rules, a fix): every file judged again, or a secret the old rules missed stays;
    # an index from before this was recorded was judged by older rules
    assert idx.read("SELECT value FROM meta WHERE key='secrets'") == [(guard.fingerprint(),)]
    monkeypatch.setattr(guard, "fingerprint", lambda: "another set of rules")
    assert idx.update([root]).changed == 1 and idx.update([root]).changed == 0
    with sqlite3.connect(home / "index.sqlite") as db:
        db.execute("DELETE FROM meta WHERE key='secrets'")
    assert idx.update([root]).changed == 1
    # an index from before the context was recorded ranks with the section only: kept under "section", rebuilt for
    # "place"
    monkeypatch.setattr(index_module, "CONTEXT", "section")
    idx.update([root])
    with sqlite3.connect(home / "index.sqlite") as db:
        db.execute("DELETE FROM meta WHERE key='context'")
    assert idx.update([root]).changed == 0
    with sqlite3.connect(home / "index.sqlite") as db:
        db.execute("DELETE FROM meta WHERE key='context'")
    monkeypatch.setattr(index_module, "CONTEXT", "place")
    assert idx.update([root]).changed == 1
    # an index from before the chunking was recorded was cut the baseline way: a new default cuts it again
    monkeypatch.setattr(index_module.chunk, "DEFAULT", index_module.chunk.PRESETS["baseline"])
    idx.update([root])
    db = sqlite3.connect(home / "index.sqlite")                       # as the layout before the record existed
    with db:
        db.execute("DELETE FROM meta WHERE key='chunking'")
    db.close()
    monkeypatch.setattr(index_module.chunk, "DEFAULT", index_module.chunk.Options("blocks-test", max_chars=12,
                                                                                    blocks=True))
    report = idx.update([root])
    assert report.changed == 1 and [(r[4], r[5]) for r in idx.rows()] == [(1, 1), (3, 7)]


def test_the_ranking_context_is_what_every_ranker_sees_and_a_change_of_it_or_of_a_place_reembeds(home, tmp_path,
                                                                                                monkeypatch):
    """With the "place" context, BM25, the embedder and the reranker all see the file's place (declared folder's
    name, then the path inside it) and the heading path before a chunk; the context is kept in the index, so a change
    embeds everything again; a file whose declared folder changes gets its new place and a new vector."""
    class Recorder(FakeEmbedder):
        seen: list = []

        def documents(self, texts):
            Recorder.seen += texts
            return super().documents(texts)

    monkeypatch.setattr(index_module, "CONTEXT", "section")
    proj = tmp_path / "work" / "proj"
    (proj / "sub").mkdir(parents=True)
    (proj / "sub" / "a.md").write_text("# Title\n\nalpha beta\n")
    idx = Index(home, embedder=Recorder())
    idx.update([proj])
    assert Recorder.seen == ["Title\n# Title", "Title\nalpha beta"] or Recorder.seen == ["Title\n# Title\n\nalpha beta"]
    assert idx.context() == "section"
    assert idx.bm25()[1].scores("proj")[0] == 0                                         # BM25 without the place
    embedded = list(Recorder.seen)
    spy = SpyReranker(lambda t: 0.9)
    Searcher(idx, reranker=spy).rank("alpha")
    assert spy.seen == embedded and not any("proj" in t for t in spy.seen)
    Recorder.seen[:] = embedded                                                         # the query is not a document
    first = list(Recorder.seen)
    Recorder.seen.clear()
    monkeypatch.setattr(index_module, "CONTEXT", "place")
    report = idx.update([proj])
    assert report.changed == 1 and idx.context() == "place"
    assert Recorder.seen == [f"proj/sub/a.md\n{t}" for t in first]                    # place, heading path, text
    ids, bm, _ = idx.snapshot()
    assert bm.scores("proj")[0] > 0 and bm.scores("sub")[0] > 0                         # BM25 reads the place too
    embedded = list(Recorder.seen)
    spy = SpyReranker(lambda t: 0.9)
    Searcher(idx, reranker=spy).rank("alpha")
    assert spy.seen == embedded
    Recorder.seen.clear()
    report = idx.update([tmp_path / "work"])                                              # a parent declared instead
    assert Recorder.seen == [f"work/proj/sub/a.md\n{t}" for t in first]
    assert {r[8] for r in idx.rows()} == {"work/proj/sub/a.md"}
    Recorder.seen.clear()
    idx.update([tmp_path / "work"])
    assert Recorder.seen == []                                                            # nothing moved: no new vector
    assert index_module.embed_text("", "t", "") == "t" and index_module.embed_text("S", "t") == "S\nt"


@pytest.mark.parametrize("how", ["interrupted (a power cut, Ctrl+C)", "asked to stop (the interface quits)"])
def test_a_stopped_update_keeps_what_it_committed_and_the_next_one_resumes(home, tmp_path, monkeypatch, how):
    """An update commits as it goes (every COMMIT_SECONDS of work): stopped midway, it keeps the files before the
    stop, each whole (its chunks, its extracted text), and asks for no vector before every file is reached; the next
    update reads only the rest and embeds every chunk once, ending as an unbroken one would. A failure keeps what was
    committed (here after every file); an update asked to stop (here at the fifth file) commits its open batch itself
    (no commit was due: COMMIT_SECONDS as shipped) and raises Stopped."""
    asked = how.startswith("asked")
    if not asked:
        monkeypatch.setattr(index_module, "COMMIT_SECONDS", 0)
    root = tmp_path / "n"
    root.mkdir()
    for i in range(5):
        (root / f"{i}.md").write_text(f"# Note {i}\n\nword{i} text\n")

    class Counting(FakeEmbedder):
        seen = []

        def documents(self, texts):
            Counting.seen += texts
            return super().documents(texts)

    checks, progress = [], []

    def reached(done, total):
        progress.append((done, total))
        if not asked and done == 2:                                  # Ctrl+C right after the second file
            raise KeyboardInterrupt
    idx = Index(home, embedder=Counting())
    with pytest.raises(index_module.Stopped if asked else KeyboardInterrupt):
        idx.update([root], stop=(lambda: checks.append(1) or len(checks) > 4) if asked else None, progress=reached)
    assert progress == [(i, 5) for i in range(1, 5 if asked else 3)]        # each file reached, of all to look at
    kept = {r[1] for r in idx.rows()}
    assert kept == {str(root / f"{i}.md") for i in range(4 if asked else 2)}  # the files before the stop, whole
    assert idx.held() == (len(kept), len(kept)) and Counting.seen == []       # no vector asked for yet
    idx.update([root])
    assert {r[1] for r in idx.rows()} == {str(root / f"{i}.md") for i in range(5)}
    assert sorted(Counting.seen) == sorted(set(Counting.seen)) and len(Counting.seen) == 5   # each chunk once
    whole = Index(tmp_path / "other", embedder=FakeEmbedder())
    whole.update([root])
    assert sorted(r[1:] for r in idx.rows()) == sorted(r[1:] for r in whole.rows())
    assert idx.read("SELECT COUNT(*) FROM chunks c LEFT JOIN vectors v ON v.id = c.id WHERE v.id IS NULL") == [(0,)]


def test_a_stopped_recut_keeps_every_file_searchable(home, tmp_path, monkeypatch):
    """A new chunking re-cuts every file as it is reached; stopped midway, a file not reached keeps its old chunks
    and vectors (still found) and its extracted text (a stopped re-cut once blanked the content hashes: the store
    lost those texts and held an entry without a hash), a file reached has its new ones, and the next update
    finishes the re-cut."""
    monkeypatch.setattr(index_module, "COMMIT_SECONDS", 0)
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\n```py\nx = 1\n\ny = 2\n```\n")
    (root / "b.md").write_text("# B\n\n```py\nalpha = 1\n\nbeta = 2\n```\n")
    nb = root / "c.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": "zeta notes"}]}))
    monkeypatch.setattr(index_module.chunk, "DEFAULT", index_module.chunk.PRESETS["baseline"])
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    old_b = {r[1:7] for r in idx.rows() if r[1].endswith("b.md")}
    monkeypatch.setattr(index_module.chunk, "DEFAULT", index_module.chunk.Options("blocks-test", max_chars=12,
                                                                                    blocks=True))

    def breaks(done, total):                                  # Ctrl+C once the first file is re-cut
        if done == 1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        Index(home, embedder=FakeEmbedder()).update([root], progress=breaks)
    assert {r[1:7] for r in idx.rows() if r[1].endswith("b.md")} == old_b                    # not reached: as it was
    assert [(r[4], r[5]) for r in idx.rows() if r[1].endswith("a.md")] == [(1, 1), (3, 7)]   # reached: re-cut
    assert Searcher(idx, reranker=FakeReranker()).find("alpha beta", k=1)[0][0].path.endswith("b.md")
    assert idx.extracted(nb) is not None and idx.extracted(nb)[0].is_file()
    assert idx.read("SELECT COUNT(*) FROM texts WHERE sha = ''") == [(0,)]
    report = idx.update([root])
    assert report.changed == 2 and [(r[4], r[5]) for r in idx.rows() if r[1].endswith("b.md")] == [(1, 1), (3, 7)]


def test_two_updates_at_once_end_as_one_would(home, tmp_path, monkeypatch):
    """Two updaters on one index (the interface and `peresearch index`, each with its own connection), their commits
    interleaved file by file: the index ends whole and as a single update leaves it."""
    monkeypatch.setattr(index_module, "COMMIT_SECONDS", 0)
    root = tmp_path / "n"
    root.mkdir()
    for i in range(index_module.EMBED_BATCH + 6):                    # more chunks than one embedding batch
        (root / f"{i:02d}.md").write_text(f"# Note {i}\n\nword{i} " + "text " * (i * 20) + "\n")
    (root / "copy.md").write_text((root / "03.md").read_text())
    errors = []

    def run():
        try:
            Index(home, embedder=FakeEmbedder()).update([root])
        except Exception as e:                                         # any failure is the bug ("database is
            errors.append(repr(e))                                     # locked" when the WAL switch did not wait)
    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    idx = Index(home, embedder=FakeEmbedder())
    whole = Index(tmp_path / "other", embedder=FakeEmbedder())
    whole.update([root])
    assert sorted(r[1:] for r in idx.rows()) == sorted(r[1:] for r in whole.rows())
    assert sorted(idx.read("SELECT path, status FROM files")) == sorted(whole.read("SELECT path, status FROM files"))
    assert idx.read("SELECT COUNT(*) FROM chunks c LEFT JOIN vectors v ON v.id = c.id WHERE v.id IS NULL") == [(0,)]


def test_two_indexes_opened_at_once_on_a_new_file_keep_each_others_tables(home, tmp_path, monkeypatch):
    """Opening an index read its layout version and then emptied an older layout, two steps a second connection
    could come between: B read version 0 of a new file, A set it up and indexed, then B dropped A's tables
    ("no such table: vectors" in the two-updaters test, seen once in mutmut's stats run, 2026-10-05). B here stops
    right after reading the version, for as long as A needs or two seconds."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nalpha note\n")
    real, read, a_done = index_module._connect, threading.Event(), threading.Event()

    class Paused:                                     # B's connection: pauses after its first version read
        def __init__(self, db):
            self.db, self.first = db, True

        def execute(self, sql, *args):
            out = self.db.execute(sql, *args)
            if self.first and sql.strip().upper().startswith("PRAGMA USER_VERSION"):
                self.first = False
                read.set()
                a_done.wait(2)
            return out

        def __getattr__(self, name):
            return getattr(self.db, name)

    monkeypatch.setattr(index_module, "_connect", lambda path: Paused(real(path)))
    opened = []
    b = threading.Thread(target=lambda: opened.append(Index(home, embedder=FakeEmbedder())))
    b.start()
    assert read.wait(10)
    monkeypatch.setattr(index_module, "_connect", real)
    a = Index(home, embedder=FakeEmbedder())          # waits for B's lock when the version is read under it
    a.update([root])
    a_done.set()
    b.join(10)
    assert opened and [r[1] for r in a.rows()] == [str(root / "a.md")]               # A's index survived B
    assert a.read("SELECT COUNT(*) FROM vectors") == [(1,)]


def test_an_older_layout_is_emptied_and_a_failed_set_up_leaves_no_transaction_open(home, monkeypatch):
    """An index of an earlier layout version is emptied (its tables dropped, rebuilt by the next update); a set-up that
    fails is rolled back, and the file opens again afterwards."""
    home.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(home / "index.sqlite")
    db.executescript("CREATE TABLE files (path TEXT PRIMARY KEY, size INTEGER); INSERT INTO files VALUES ('old', 1);"
                     "CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT); INSERT INTO chunks VALUES (1, 'old');"
                     "CREATE TABLE vectors (id INTEGER PRIMARY KEY, vec BLOB); INSERT INTO vectors VALUES (1, x'00');"
                     "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT); INSERT INTO meta VALUES ('embedder', 'old');"
                     "PRAGMA user_version = 1;")
    db.close()
    (home / "dense.npz").write_bytes(b"old vectors")
    idx = Index(home, embedder=FakeEmbedder())
    assert [idx.read(f"SELECT COUNT(*) FROM {t}")[0][0] for t in ("files", "chunks", "vectors", "meta")] == [0, 0, 0, 0]
    assert not (home / "dense.npz").exists() and idx.read("PRAGMA user_version") == [(index_module.VERSION,)]
    other = home / "other"
    other.mkdir()
    monkeypatch.setattr(index_module, "SCHEMA", index_module.SCHEMA + "\nCREATE TABLE broken (;")
    with pytest.raises(sqlite3.OperationalError):
        Index(other, embedder=FakeEmbedder())
    monkeypatch.undo()
    assert Index(other, embedder=FakeEmbedder()).read("SELECT COUNT(*) FROM files") == [(0,)]   # not left locked


def test_a_connection_is_in_wal_mode_and_waits_for_locks(home, monkeypatch):
    """_connect's contract: the connection waits for another one's lock (30 s) and is in WAL mode. Switching to WAL
    is retried while SQLite answers "database is locked" (two connections opening a new file at once: reproduced
    in test_two_updates_at_once_end_as_one_would); a drive that never allows WAL ends the wait with a reason (the
    loop once had no exit when SQLite answered with another journal mode)."""
    import sqlite3 as sq
    home.mkdir(parents=True)
    db = index_module._connect(home / "index.sqlite")
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 30000 and db.isolation_level is None

    def fake(answers):
        class Fake:
            def __init__(self, *a, **k):
                pass

            def execute(self, sql, *a):
                answer = answers.pop(0) if len(answers) > 1 else answers[0]
                if isinstance(answer, Exception):
                    raise answer
                return type("C", (), {"fetchone": lambda self: (answer,)})()
        return Fake
    clock = iter(range(0, 10**6))                                                  # one second a reading
    monkeypatch.setattr(index_module.time, "monotonic", lambda: next(clock))
    sleeps = []
    monkeypatch.setattr(index_module.time, "sleep", sleeps.append)
    locked = sq.OperationalError("database is locked")
    monkeypatch.setattr(index_module.sqlite3, "connect", fake([locked, locked, "WAL"]))
    index_module._connect(home / "index.sqlite")                                  # waited through two refusals
    assert sleeps == [0.05, 0.05]                                                   # 50 ms between tries
    monkeypatch.setattr(index_module.sqlite3, "connect", fake(["delete"]))
    with pytest.raises(sq.OperationalError, match="WAL"):
        index_module._connect(home / "index.sqlite")
    monkeypatch.setattr(index_module.sqlite3, "connect", fake([sq.OperationalError("disk I/O error")]))
    sleeps.clear()
    with pytest.raises(sq.OperationalError, match="disk I/O"):                    # other errors are not retried:
        index_module._connect(home / "index.sqlite")
    assert sleeps == []                                                             # raised at once, no wait


def test_a_hit_says_where_it_is_in_the_units_of_its_file(home, mixed):
    """The place a citation shows: lines L3 or L1-3, a PDF's pages p.2 or p.1-3, a notebook's cells cell 4 or cell 1-2
    (short pages and cells pack together; the first of two copies in sorted order is the one indexed)."""
    idx, s = make(home)
    idx.update([mixed])
    where = {Path(h.path).name: h.where for q in ("beta on page two second line", "first cell", "phở bò bún chả")
             for h in s.find(q, k=1)[0]}
    assert where == {"copy.pdf": f"{mixed / 'copy.pdf'}:p.1-3", "run.ipynb": f"{mixed / 'run.ipynb'}:cell 1-2",
                     "old.txt": f"{mixed / 'old.txt'}:L1-3"}
    for unit, label in (("line", "L4"), ("page", "p.4"), ("cell", "cell 4")):
        assert search.Hit(1, "f", "", unit, 4, 9, "t", 0.9)._place(4, 4) == f"f:{label}"


def test_a_near_copy_is_folded_into_the_better_ranked_candidate(home, tmp_path):
    """A chunk that is a near-copy of a better-ranked one (README.template beside README, a notebook beside its .txt
    source: 4.9% of the top-10 places on the gold set) is not a candidate of its own: the better one lists where the
    copy is, and the page holds another passage instead. Copies are told by 8-character shingles (any script)."""
    root = tmp_path / "n"
    root.mkdir()
    body = "Gradient descent moves the weights against the gradient, a small step at a time, until the loss stops. "
    (root / "README.md").write_text("# Notes\n\n" + body * 3 + "\n")
    (root / "README.template.md").write_text("# Notes\n\n" + body * 3 + "Rendered by report.py.\n")   # a near-copy
    (root / "zh.md").write_text("# 笔记\n\n" + "梯度下降法沿着梯度的反方向更新权重，每次走一小步。" * 6 + "\n")
    (root / "zh-copy.md").write_text("# 笔记\n\n" + "梯度下降法沿着梯度的反方向更新权重，每次走一小步。" * 6 + "完。\n")
    (root / "other.md").write_text("# Other\n\nMomentum keeps a running average of past gradient steps.\n")
    idx, s = make(home)
    idx.update([root])
    rows, scores, copies = s.candidates("gradient descent weights 梯度")
    group = {"README.md": "en", "README.template.md": "en", "zh.md": "zh", "zh-copy.md": "zh", "other.md": "other"}
    names = [Path(r[1]).name for r in rows]
    assert len(scores) == len(rows) and sorted(group[n] for n in names) == ["en", "other", "zh"]  # each text once
    folded = [Path(c[1]).name for cs in copies.values() for c in cs]
    assert sorted(group[n] for n in folded) == ["en", "zh"] and not set(folded) & set(names)       # the copies named
    assert all(group[Path(rows[i][1]).name] == group[Path(c[1]).name] for i, cs in copies.items() for c in cs)
    hits, _, total = s.find("gradient descent weights 梯度", k=10)
    assert total == 3 and all(Path(h.path).name != "other.md" or not h.copies for h in hits)
    assert any(h.copies for h in hits) and all(":L" in w for h in hits for w in h.copies)
    assert search.near_copy("abc" * 10, "abc" * 10) and not search.near_copy("abcdefgh" * 4, "zyxwvuts" * 4)
    assert search.near_copy("a  b\n c" * 5, "a b c" * 5)                                # spacing does not count
    assert search._alike({1, 2, 3, 4}, {1, 2, 3, 4, 5}) and not search._alike({1, 2, 3, 4}, {1, 2, 3, 4, 5, 6})  # 0.8
    assert not search.near_copy("x" * 8 + "abcdefghijklmnopq", "x" * 8 + "ABCDEFGHI0123456")      # half the same
    # every 8-character window of the text joined by single spaces, the last one included, no shorter one: close to
    # 0.8, a window more or less (or another joint) moves a pair across it
    assert search.near_copy("it notes score by by score", "ix notes score by by score")         # 17/21 shared, 0.81
    assert search.near_copy("the model answers", "the model answery")                           # 9/11, 0.82
    assert not search.near_copy("them the model reads", "txem the model reads")                 # 11/15, 0.73
    rows, _, copies = s.candidates("gradient descent weights 梯度")
    by_name = {Path(r[1]).name: i for i, r in enumerate(rows)}
    for h in hits:                                                  # each copy named by its exact place
        i = next(j for j, r in enumerate(rows) if r[0] == h.id)
        assert h.copies == [search.place(c[1], c[3], c[4], c[5]) for c in copies.get(i, [])]
    assert any(w.endswith(":L1-3") for h in hits for w in h.copies)


@pytest.mark.parametrize("depth", [0, 1, search.N_RERANK])
def test_documents_rank_each_file_once_by_its_best_passage_with_the_searchs_fusion(home, tmp_path, depth):
    """overview's ranking (runs/gold/overview-documents-2026-10-06.py): the search's own fusion (BM25 and dense,
    each view's top N_FIRST, rho), each file once, standing for it its best fused chunk, with how many of its chunks
    are candidates; the first `depth` documents reordered by the reranker on that chunk (ties in fused order), the
    rest in fused order without a score. A folder keeps the candidates inside it before the fusion; an empty folder,
    an unknown one or an empty index gives none."""
    root = tmp_path / "n"
    (root / "subX").mkdir(parents=True)                   # a name ending in X: only "/" is taken off a folder
    for i in range(search.N_RERANK + 10):                 # distinct words: no two notes are near-copies
        where = root / "subX" if i % 3 == 0 else root
        words = lambda j: ["rank"] * (i % 5 + j) + [f"w{i}q{j}x{k}" for k in range(90)]          # noqa: E731
        body = "\n\n".join("\n".join(" ".join(words(j)[k:k + 30]) for k in range(0, 95, 30)) for j in range(3))
        (where / f"{i:02d}.md").write_text(f"# N{i}\n\n{body}\n")
    (root / "other.md").write_text("# Other\n\nnothing alike here\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    score = lambda t: int(re.search(r"(\d\d)\.md", t).group(1)) % 7 / 10 if re.search(r"\d\d\.md", t) else 0.0  # noqa
    spy = SpyReranker(score) if depth else NoReranker()
    s = Searcher(idx, reranker=spy, depth=depth)
    ids, bm, vecs = idx.snapshot()
    rows = {r[0]: r for r in idx.rows()}
    q = FakeEmbedder().queries(["rank"])[0].astype(np.float32)
    core = __import__("peresearch.zetokrag.core", fromlist=["x"])

    def expected(keep):
        sel = np.array([i for i, c in enumerate(ids) if keep(rows[int(c)][1])], dtype=np.int64)
        if len(sel) == 0:
            return [], [], {}, {}
        fused, _ = core.fuse_minmax(bm.scores("rank")[sel], (vecs @ q)[sel], search.RHO, search.N_FIRST)
        order, best, count = [], {}, {}
        for i in fused:
            r = rows[int(ids[sel[i]])]
            count[r[1]] = count.get(r[1], 0) + 1
            if r[1] not in best:
                best[r[1]] = r
                order.append(r[1])
        head = sorted(order[:depth], key=lambda p: -score(idx.ranking_text(best[p])))
        return head + order[depth:], order, best, count

    for folder, keep in ((None, lambda p: True), (str(root / "subX"), lambda p: p.startswith(str(root / "subX") + "/"))):
        spy = SpyReranker(score) if depth else NoReranker()
        s = Searcher(idx, reranker=spy, depth=depth)
        order, fused, best, count = expected(keep)
        docs = s.documents("rank", folder)
        assert [d.path for d in docs] == order and len(order) == len(set(order)) > search.N_RERANK - 20
        assert all((d.unit, d.start, d.end, d.text, d.section, d.sha) == (best[d.path][3], best[d.path][4],
                   best[d.path][5], best[d.path][6], best[d.path][2], best[d.path][7]) for d in docs)
        assert [d.matches for d in docs] == [count[p] for p in order]
        assert all(d.score is not None for d in docs[:depth]) and all(d.score is None for d in docs[depth:])
        assert [d.score for d in docs[:depth]] == pytest.approx([score(idx.ranking_text(best[p])) for p in order[:depth]])
        if depth:
            assert spy.seen == [idx.ranking_text(best[p]) for p in fused[:depth]]
        if folder:
            assert all(p.startswith(folder + "/") for p in order)
            assert [d.path for d in Searcher(idx, reranker=SpyReranker(score) if depth else NoReranker(),
                                             depth=depth).documents("rank", folder + "/")] == order
        assert max(count.values()) > 1                                  # files with several candidate chunks
        assert any(d.start != d.end for d in docs) and len({d.score for d in docs[:depth]}) == min(depth, 7)
    assert s.documents("rank", str(root / "nowhere")) == [] and s.documents("rank", str(root / "sub")) == []
    (tmp_path / "empty").mkdir()
    empty = Index(home.parent / "empty-home", embedder=FakeEmbedder())
    empty.update([tmp_path / "empty"])
    assert Searcher(empty, reranker=NoReranker()).documents("rank") == []


def test_an_outline_names_a_files_headings_in_order_once_and_its_first_line(home, tmp_path):
    """overview's one-line picture of a file: the last part of each heading path, in order, each once (at most
    OUTLINE_HEADINGS, then how many more), and the first line of text that is not a heading."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# Intro\n\nFirst words here.\n\n## Method\n\nm\n\n## Results\n\nr\n\n# Intro\n\nagain\n")
    (root / "b.py").write_text("\n\n'''A module.'''\nx = 1\n")
    (root / "c.md").write_text("".join(f"# H{i}\n\ntext {i}\n\n" for i in range(search.OUTLINE_HEADINGS + 3)))
    (root / "d.md").write_text("# A\n\n## B\n\n### C\n\n    # indented, not a heading line of text\n\nfirst text\n")
    (root / "e.md").write_text("# Only\n\n## Headings\n")
    for extra in (0, 1):                                    # just at the limit: no "more"; one past it: "1 more"
        (root / f"f{extra}.md").write_text("".join(f"# K{i}\n\nk\n\n" for i in range(search.OUTLINE_HEADINGS + extra)))
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    s = Searcher(idx, reranker=FakeReranker())
    assert s.outline(str(root / "a.md")) == (["Intro", "Method", "Results"], "First words here.")
    assert s.outline(str(root / "b.py")) == ([], "'''A module.'''")
    heads, lead = s.outline(str(root / "c.md"))
    assert heads == [f"H{i}" for i in range(search.OUTLINE_HEADINGS)] + ["… and 3 more"] and lead == "text 0"
    assert s.outline(str(root / "missing.md")) == ([], "")
    assert s.outline(str(root / "d.md")) == (["A", "B", "C"], "first text")
    assert s.outline(str(root / "e.md")) == (["Only", "Headings"], "")
    assert s.outline(str(root / "f0.md"))[0] == [f"K{i}" for i in range(search.OUTLINE_HEADINGS)]
    assert s.outline(str(root / "f1.md"))[0] == [f"K{i}" for i in range(search.OUTLINE_HEADINGS)] + ["… and 1 more"]


def test_a_file_gone_between_the_walk_and_its_reading_is_left_for_the_next_update(home, tmp_path, monkeypatch):
    """A file can vanish after the walk lists it and before it is read: Word's "~$name.docx" owner file goes when the
    document is closed. The update goes on without it (the next one finds it gone, or back), whether it vanished
    before its size was taken or before its content was hashed."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nalpha\n")
    (root / "b.md").write_text("# B\n\nbeta\n")
    idx = Index(home, embedder=FakeEmbedder())
    real_walk = index_module.files.walk

    def walk(*a, **kw):
        for p in real_walk(*a, **kw):
            if p.name == "~$gone.md":
                p.unlink()                                  # listed, then gone before it is looked at
            yield p
    monkeypatch.setattr(index_module.files, "walk", walk)
    (root / "~$gone.md").write_text("owner file\n")
    r = idx.update([root])
    assert {Path(p).name for (p,) in idx.read("SELECT path FROM files")} == {"a.md", "b.md"} and r.added == 2
    real_sha = index_module.sha256

    def sha(path):
        if path.name == "~$late.md":
            path.unlink()                                   # its size taken, then gone before its hash
        return real_sha(path)
    monkeypatch.setattr(index_module, "sha256", sha)
    (root / "~$late.md").write_text("owner file\n")
    r = idx.update([root])
    assert {Path(p).name for (p,) in idx.read("SELECT path FROM files")} == {"a.md", "b.md"} and r.unchanged == 2
    monkeypatch.setattr(index_module, "sha256", real_sha)
    locked = root / "locked.md"                             # no permission to read it: said, not a crash
    locked.write_text("# L\n\nlocked\n")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):                      # run as root: every file is readable
            pytest.skip("files cannot be made unreadable here")
        r = idx.update([root])
        assert r.problems.get("unreadable") == [str(locked)] and idx.why_not(locked) == "unreadable: PermissionError"
        assert {Path(p).name for (p,) in idx.read("SELECT path FROM files")} == {"a.md", "b.md"}
    finally:
        locked.chmod(0o644)


def test_a_near_copy_document_is_folded_into_the_better_ranked_one(home, tmp_path):
    """As search_notes folds an excerpt into its near-copy, overview's documents fold a file whose best chunk is a
    near-copy of a better-ranked file's (8-character shingles, Jaccard >= COPY_JACCARD): it is named under the one
    kept, and frees its line (runs/gold/overview-documents-2026-10-06-folded.log: +0.04 [+0.01, +0.10] relevant files
    in 20 lines, none worse). Files merely alike are both kept."""
    root = tmp_path / "n"
    root.mkdir()
    body = " ".join(f"tokenizer w{i}" for i in range(60))
    (root / "README.md").write_text(f"# Guide\n\n{body} end\n")
    (root / "README.template.md").write_text(f"# Guide\n\n{body} fin\n")          # one word apart: a near-copy
    (root / "other.md").write_text("# Other\n\n" + " ".join(f"tokenizer z{i}" for i in range(60)) + "\n")
    for k in range(3):                                   # ranked after the pair: still listed after a fold
        (root / f"later{k}.md").write_text(f"# Later {k}\n\ntokenizer " + " ".join(f"q{k}x{i}" for i in range(80)) + "\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    docs = Searcher(idx, reranker=FakeReranker(), depth=0).documents("tokenizer")
    names = [Path(d.path).name for d in docs]
    assert sorted(names) in (["README.md", "later0.md", "later1.md", "later2.md", "other.md"],
                             ["README.template.md", "later0.md", "later1.md", "later2.md", "other.md"])
    assert names.index("later0.md") > 0 and any(n.startswith("later") for n in names[1:])
    kept = next(d for d in docs if d.path.startswith(str(root / "README")))
    folded = "README.template.md" if Path(kept.path).name == "README.md" else "README.md"
    assert [Path(c).name for c in kept.copies] == [folded]
    assert next(d for d in docs if d.path.endswith("other.md")).copies == []
