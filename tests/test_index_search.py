import os
from pathlib import Path

import pytest

from peresearch import guard
from peresearch.zetokrag import search
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher
from tests.conftest import FakeEmbedder, FakeReranker


def make(home, root):
    idx = Index(home, embedder=FakeEmbedder())
    return idx, Searcher(idx, reranker=FakeReranker())


@pytest.fixture
def notes(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "ml.md").write_text("# Học máy\n\nGradient descent cập nhật trọng số theo đạo hàm.\n")
    (root / "food.md").write_text("# Phở\n\nNước dùng hầm xương bò trong nhiều giờ.\n")
    return root


def test_find_with_exact_source(home, notes):
    idx, s = make(home, notes)
    r = idx.update([notes])
    assert (r.added, r.chunks) == (2, 2)
    hits, verdict, stale = s.find("gradient descent cập nhật trọng số", k=1)
    assert hits[0].path.endswith("ml.md") and hits[0].where.endswith("ml.md:L1-3") and stale == 0
    assert hits[0].section == "Học máy" and verdict in ("enough", "partial")


def test_incremental_update_and_removal(home, notes):
    idx, _ = make(home, notes)
    idx.update([notes])
    r = idx.update([notes])
    assert (r.added, r.changed, r.unchanged) == (0, 0, 2)
    (notes / "ml.md").write_text("# Học máy\n\nAdam dùng moment bậc một và bậc hai.\n")
    os.utime(notes / "ml.md", (1, 1))
    r = idx.update([notes])
    assert (r.changed, r.unchanged) == (1, 1)
    assert "Adam" in " ".join(row[6] for row in idx.rows())
    (notes / "food.md").unlink()
    r = idx.update([notes])
    assert r.removed == 1 and r.chunks == 1
    ids, vecs = idx.dense()
    assert len(ids) == len(vecs) == 1


def test_protected_files_symlinks_and_secrets_never_indexed(home, notes, tmp_path):
    (notes / ".env").write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n")
    (notes / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "diary.md").write_text("# secret diary\n\nnot declared\n")
    os.symlink(outside / "diary.md", notes / "link.md")          # a link out of the declared folder
    os.symlink(outside, notes / "linkdir")
    (notes / "setup.md").write_text("# Setup\n\nHF token: hf_" + "a1B2" * 9 + "\n\n# Other\n\nsafe text\n")
    idx, _ = make(home, notes)
    r = idx.update([notes])
    text = " ".join(row[6] for row in idx.rows())
    assert "sk-proj" not in text and "PRIVATE KEY" not in text and "diary" not in text and "hf_a1B2" not in text
    assert "safe text" in text and r.withheld == {str(notes / "setup.md"): 1}
    paths = {row[1] for row in idx.rows()}
    assert not any(p.endswith((".env", "id_ed25519", "link.md")) for p in paths)


def test_stale_excerpts_are_dropped_not_misquoted(home, notes):
    idx, s = make(home, notes)
    idx.update([notes])
    (notes / "ml.md").write_text("# Học máy\n\nSomething else entirely.\n")   # edited after indexing
    hits, _, stale = s.find("gradient descent trọng số", k=2)
    assert stale == 1 and not any(h.path.endswith("ml.md") for h in hits)
    (notes / "food.md").write_text("# Phở\n\nNước dùng hầm xương bò trong nhiều giờ.\n\nThêm hành.\n")
    hits, _, _ = s.find("nước dùng xương bò", k=1)
    assert hits[0].path.endswith("food.md") and hits[0].file_changed   # still present, so kept and flagged


def test_verdict_thresholds():
    assert search.verdict(search.ENOUGH) == "enough"
    assert search.verdict(search.PARTIAL) == "partial"
    assert search.verdict(0.0) == "none"


def test_neighbours_stay_in_section(home, tmp_path):
    root = tmp_path / "n"
    root.mkdir()
    body = "\n\n".join(("para %d " % i) + "w" * 900 for i in range(4))   # one paragraph per chunk
    (root / "a.md").write_text("# S\n\n" + body + "\n\n# T\n\nother section\n")
    idx, s = make(home, root)
    idx.update([root])
    rows = idx.rows()
    assert len(s.neighbours(rows[1][0], rows[1][1], rows[1][2])) == 2      # para 0 and para 2
    last = s.neighbours(rows[3][0], rows[3][1], rows[3][2])                # para 2 only, not section T
    assert len(last) == 1 and "other section" not in last[0]


def test_weak_matches_are_not_shown(home, notes):
    idx, s = make(home, notes)
    idx.update([notes])
    hits, verdict, _ = s.find("quantum chromodynamics lattice", k=3)
    assert hits == [] and verdict == "none"


def test_dataset_folders_are_skipped_and_reported(home, notes, monkeypatch):
    from peresearch.zetokrag import index as index_module
    monkeypatch.setattr(index_module, "MAX_FILES_PER_FOLDER", 3)
    big = notes / "reviews"
    big.mkdir()
    for i in range(5):
        (big / f"{i}.txt").write_text(f"review {i}")
    idx, _ = make(home, notes)
    r = idx.update([notes])
    assert r.added == 2 and any("reviews" in p for p in r.problems["large_folder"])


def test_one_plain_bm25_index_without_diacritic_folding(home, tmp_path):
    """Owner, 2026-10-02: Vietnamese typed without diacritics is rare, so there is one BM25 index over words as
    typed and no rule choosing a folded one by the query's diacritics ("ma" no longer matches "má")."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nmá tôi nấu cơm\n")
    (root / "b.md").write_text("# B\n\nma quỷ trong truyện\n")
    idx, s = make(home, root)
    idx.update([root])
    ids, bm = idx.bm25()
    paths = dict((row[0], row[1]) for row in idx.rows())
    matched = [paths[int(i)] for i, score in zip(ids, bm.scores("ma")) if score > 0]
    assert len(matched) == 1 and matched[0].endswith("b.md")
    assert s.rank("ma quỷ")[0][0][1].endswith("b.md")


def test_identical_files_are_indexed_once(home, tmp_path):
    """Five homework folders held the same vendored stb_image.h: 1,030 chunks of one file. A copy is listed in the
    report with the file it copies; when that file goes, the copy is indexed in its place."""
    root = tmp_path / "n"
    for d in ("hw0", "hw1"):
        (root / d).mkdir(parents=True)
        (root / d / "lib.h").write_text("/* image loader */\nint load_image(const char *path);\n")
    (root / "hw1" / "notes.md").write_text("# Notes\n\nown text\n")
    idx, _ = make(home, root)
    r = idx.update([root])
    paths = [row[1] for row in idx.rows()]
    assert sum(p.endswith("lib.h") for p in paths) == 1 and any(p.endswith("hw0/lib.h") for p in paths)
    assert r.problems["duplicate"] == [f"{root / 'hw1' / 'lib.h'} (same as {root / 'hw0' / 'lib.h'})"]
    assert idx.update([root]).problems.get("duplicate")                     # still reported on the next run
    (root / "hw0" / "lib.h").unlink()
    r = idx.update([root])
    paths = [row[1] for row in idx.rows()]
    assert any(p.endswith("hw1/lib.h") for p in paths) and "duplicate" not in r.problems
    ids, vecs = idx.dense()
    assert sorted(ids.tolist()) == sorted(row[0] for row in idx.rows())


def test_data_files_stay_out_of_the_index_but_remain_readable(home, tmp_path):
    """The IMDB folder's urls_*.txt (one URL per line) and imdbEr.txt (one number per line) were 3,940 of 8,512
    chunks. A text file whose lines are nearly all URLs or numbers is data: not indexed, named in the report, and
    still open to the agent's read and grep (only retrieval leaves it out)."""
    root = tmp_path / "n"
    root.mkdir()
    (root / "urls_pos.txt").write_text("".join(f"http://www.imdb.com/title/tt{i:07d}/usercomments\n" for i in range(200)))
    (root / "imdbEr.txt").write_text("".join(f"{(i - 100) / 37:.6f}\n" for i in range(200)))
    (root / "readme.txt").write_text("".join(f"Line {i} of the dataset description, in words.\n" for i in range(200)))
    (root / "links.txt").write_text("See https://arxiv.org/abs/1706.03762\nhttps://example.org\n")   # short: kept
    idx, _ = make(home, root)
    r = idx.update([root])
    paths = {row[1] for row in idx.rows()}
    assert {Path(p).name for p in paths} == {"readme.txt", "links.txt"}
    assert sorted(Path(p).name for p in r.problems["data_file"]) == ["imdbEr.txt", "urls_pos.txt"]
    from peresearch.tools import Toolbox
    out = Toolbox(roots=[root]).read(str(root / "urls_pos.txt"), 1, 3)
    assert "tt0000000" in out and "tt0000002" in out


def test_vectors_are_stored_in_sqlite_with_their_chunks(home, notes):
    """One store: the vectors live in index.sqlite and are written in the same transaction as their chunks, so a
    reader never sees chunks without vectors or the reverse (dense.npz is gone)."""
    idx, _ = make(home, notes)
    idx.update([notes])
    assert not (home / "dense.npz").exists()
    ids, vecs = idx.dense()
    assert ids.tolist() == sorted(row[0] for row in idx.rows()) and vecs.shape == (2, FakeEmbedder.dim)
    n = idx.read("SELECT COUNT(*) FROM vectors")[0][0]
    assert n == 2


def test_an_index_in_the_old_layout_is_rebuilt(home, notes):
    """An index from before the single store (dense.npz beside the database) is dropped and built again."""
    import sqlite3

    import numpy as np
    home.mkdir(parents=True)
    db = sqlite3.connect(home / "index.sqlite")
    db.executescript("CREATE TABLE files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, sha TEXT, status TEXT, "
                     "note TEXT, withheld INTEGER); CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT, "
                     "section TEXT, unit TEXT, start INTEGER, end INTEGER, text TEXT, sha TEXT);"
                     "INSERT INTO chunks VALUES (1, '/gone.md', '', 'line', 1, 1, 'old text', 'x');")
    db.commit()
    db.close()
    np.savez(home / "dense.npz", ids=np.array([1]), vecs=np.zeros((1, 4), np.float16))
    idx, s = make(home, notes)
    assert not (home / "dense.npz").exists() and idx.rows() == []
    r = idx.update([notes])
    assert r.added == 2 and len(idx.dense()[0]) == 2


class CountingEmbedder(FakeEmbedder):
    def __init__(self, signature):
        self.signature, self.embedded = signature, 0

    def documents(self, texts):
        self.embedded += len(texts)
        return super().documents(texts)


def test_a_different_embedder_embeds_every_chunk_again(home, notes):
    """Vectors from another embedder (or the same one reading more tokens) cannot be mixed with new ones."""
    first = CountingEmbedder("model-a/512")
    Index(home, embedder=first).update([notes])
    same = CountingEmbedder("model-a/512")
    Index(home, embedder=same).update([notes])
    other = CountingEmbedder("model-a/32768")
    Index(home, embedder=other).update([notes])
    assert (first.embedded, same.embedded, other.embedded) == (2, 0, 2)


def test_float32_vectors_are_kept_until_the_index_changes(home, notes):
    idx, s = make(home, notes)
    idx.update([notes])
    ids, v = idx.dense32()
    assert idx.dense32()[1] is v                                      # no reload or conversion per query
    ids16, v16 = idx.dense()
    assert (ids == ids16).all() and (v == v16.astype("float32")).all()   # the same numbers as converting each time
    (notes / "new.md").write_text("# Mới\n\nMột ghi chú mới.\n")
    other = Index(home, embedder=FakeEmbedder())                      # another process updates the index
    other.update([notes])
    assert len(idx.dense32()[0]) == 3                                 # the change on disk is picked up


def test_search_sees_an_update_made_by_another_process(home, notes):
    idx, s = make(home, notes)
    idx.update([notes])
    s.find("phở nước dùng", k=1)                                      # builds this process's BM25
    (notes / "new.md").write_text("# Đà Lạt\n\nĐồi thông và hồ Xuân Hương.\n")
    Index(home, embedder=FakeEmbedder()).update([notes])              # another process updates the index
    hits, _, _ = s.find("đồi thông hồ xuân hương", k=1)
    assert hits and hits[0].path.endswith("new.md")


class SearchingEmbedder(FakeEmbedder):
    """Embeds like FakeEmbedder, but first lets a search run: another process querying mid-update."""

    def __init__(self, search):
        self.search = search

    def documents(self, texts):
        self.search()
        return super().documents(texts)


def test_search_works_while_another_process_is_indexing(home, notes):
    idx, s = make(home, notes)
    idx.update([notes])
    s.find("phở nước dùng", k=1)
    (notes / "new.md").write_text("# Đà Lạt\n\nĐồi thông và hồ Xuân Hương.\n")
    during = []
    Index(home, embedder=SearchingEmbedder(lambda: during.append(s.find("đồi thông", k=3)))).update([notes])
    assert len(during) == 1                                           # the search mid-update did not fail
    hits, _, _ = s.find("đồi thông hồ xuân hương", k=1)
    assert hits and hits[0].path.endswith("new.md")


def test_the_models_use_the_cpu_when_the_gpu_is_busy(monkeypatch):
    torch = pytest.importorskip("torch")
    from peresearch.zetokrag import models

    monkeypatch.delenv("PERESEARCH_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 * 2**30, 6 * 2**30))   # a vLLM server holds 5 GB
    assert models._device() == "cpu"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (5 * 2**30, 6 * 2**30))
    assert models._device() == "cuda"
    monkeypatch.setenv("PERESEARCH_DEVICE", "cpu")
    assert models._device() == "cpu"


def test_searching_while_the_same_index_is_updated_in_another_thread(home, notes):
    """The interface now answers questions while it brings the index up to date: one Index, one SQLite
    connection, two threads (one updating, one searching). No error, and the result is right afterwards."""
    import threading

    idx, s = make(home, notes)
    idx.update([notes])
    errors, rounds = [], {"search": 0}
    done = threading.Event()

    def search():
        while not done.is_set():
            try:
                hits, _, _ = s.find("gradient descent", k=3)
                rounds["search"] += 1
            except Exception as e:                                    # any failure is the bug
                errors.append(repr(e))
                return

    def churn():
        try:
            for i in range(15):
                (notes / f"extra{i}.md").write_text(f"# Extra {i}\n\nNote number {i} about optimisers and gradient descent.\n")
                if i % 3 == 2:
                    (notes / f"extra{i - 2}.md").unlink()
                idx.update([notes])
        except Exception as e:
            errors.append(repr(e))
        finally:
            done.set()
    readers = [threading.Thread(target=search) for _ in range(2)]
    writer = threading.Thread(target=churn)
    for t in readers + [writer]:
        t.start()
    for t in readers + [writer]:
        t.join(120)
    assert errors == [] and rounds["search"] > 5, (errors, rounds)
    hits, _, _ = s.find("Note number 14 about optimisers", k=3)
    assert hits and hits[0].path.endswith("extra14.md")


def test_a_file_that_vanishes_while_it_is_checked_is_treated_as_gone(home, notes, monkeypatch):
    """The search checks a hit's file is unchanged: it asked whether the file exists, then read it, and a file
    deleted in between raised FileNotFoundError out of the whole question (caught by the threaded stress test)."""
    idx, s = make(home, notes)
    idx.update([notes])

    def vanished(path):
        raise FileNotFoundError(2, "No such file or directory")
    monkeypatch.setattr(search, "sha256", vanished)                  # exists() said yes, then the file went
    hits, _, stale = s.find("gradient descent", k=3)
    assert hits == [] and stale >= 1


def test_a_folder_declared_inside_another_is_walked_once(home, tmp_path):
    """The owner's config lists ~/projects and four projects inside it: their files were read twice, so the report
    said 668 files added for 530 files and listed some copies twice."""
    root = tmp_path / "projects"
    (root / "a").mkdir(parents=True)
    (root / "top.md").write_text("# Top\n\ntop note\n")
    (root / "a" / "inner.md").write_text("# Inner\n\ninner note\n")
    (root / "a" / "copy.md").write_text("# Top\n\ntop note\n")
    idx, _ = make(home, root)
    r = idx.update([root, root / "a"])
    assert (r.added, r.unchanged) == (2, 0)
    assert r.problems["duplicate"] == [f"{root / 'a' / 'copy.md'} (same as {root / 'top.md'})"]
    r = idx.update([root / "a", root])
    assert (r.added, r.unchanged) == (0, 2) and len(r.problems["duplicate"]) == 1
