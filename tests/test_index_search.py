import os

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


def test_find_across_diacritics_with_exact_source(home, notes):
    idx, s = make(home, notes)
    r = idx.update([notes])
    assert (r.added, r.chunks) == (2, 2)
    hits, verdict, stale = s.find("gradient descent cap nhat trong so", k=1)
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


def test_accented_queries_use_plain_bm25_and_bare_queries_folded(home, tmp_path):
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nmá tôi nấu cơm\n")
    (root / "b.md").write_text("# B\n\nma quỷ trong truyện\n")
    idx, s = make(home, root)
    idx.update([root])
    assert s.rank("ma quỷ")[0][0][1].endswith("b.md")
    ids, bm = idx.bm25(folded=False)
    assert bm.scores("má").tolist().count(0.0) == 1          # plain: "má" only in a.md
    ids, bm = idx.bm25(folded=True)
    assert (bm.scores("ma") > 0).sum() == 2                    # bare: matches both


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
