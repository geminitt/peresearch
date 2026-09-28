import json

import pytest

from peresearch.zetokrag import chunk, parse


def test_markdown_sections_and_exact_lines(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# Tokenizers\n\nBPE merges pairs.\n\n## SuperBPE\n\nSpans across words.\nSecond line.\n")
    chunks = chunk.chunk(parse.read(p))[0]
    assert [(c.section, c.start, c.end) for c in chunks] == [("Tokenizers", 1, 3), ("Tokenizers > SuperBPE", 5, 8)]
    assert chunks[1].text.endswith("Second line.")
    lines = p.read_text().splitlines()
    for c in chunks:   # the recorded lines hold exactly the chunk's paragraphs
        assert all(part in "\n".join(lines[c.start - 1:c.end]) for part in c.text.split("\n\n"))


def test_headings_inside_code_fences_are_not_sections(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# A\n\n```bash\n# not a heading\n```\n")
    assert {c.section for c in chunk.chunk(parse.read(p))[0]} == {"A"}


def test_long_paragraph_is_cut_at_lines_with_correct_ranges(tmp_path):
    p = tmp_path / "long.txt"
    lines = [f"line {i} " + "x" * 90 for i in range(1, 51)]
    p.write_text("\n".join(lines))
    chunks = chunk.chunk(parse.read(p))[0]
    assert len(chunks) > 1 and all(len(c.text) <= chunk.MAX_CHARS for c in chunks)
    for c in chunks:
        assert c.text == "\n".join(lines[c.start - 1:c.end])
    assert chunks[0].start == 1 and chunks[-1].end == 50


def test_notebook_cells_and_pdf_pages(tmp_path):
    nb = tmp_path / "a.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": ["# Title\n", "text"]},
                                        {"cell_type": "code", "source": "x = 1", "outputs": [{"text": "noise"}]}]}))
    doc = parse.read(nb)
    assert [(b.unit, b.start) for b in doc.blocks] == [("cell", 1), ("cell", 2)] and "noise" not in str(doc.blocks)
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "a.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "Page one text")
    d.new_page()
    d.new_page().insert_text((72, 72), "Page three text")
    d.save(pdf)
    doc = parse.read(pdf)
    assert doc.status == "ok" and [b.start for b in doc.blocks] == [1, 3] and "1 pages without text" in doc.note


def test_problem_files_are_reported_not_fatal(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    scan = tmp_path / "scan.pdf"
    d = pymupdf.open()
    png = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), 0).tobytes("png")
    d.new_page().insert_image(pymupdf.Rect(0, 0, 100, 100), stream=png)
    d.save(scan)
    assert parse.read(scan).status == "needs_ocr"
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    assert parse.read(tmp_path / "broken.pdf").status == "error"
    (tmp_path / "b.txt").write_bytes(b"abc\x00def")
    assert parse.read(tmp_path / "b.txt").status == "binary"
    (tmp_path / "x.exe").write_bytes(b"MZ")
    assert parse.read(tmp_path / "x.exe").status == "unsupported"
    monkeypatch.setattr(parse, "MAX_BYTES", 10)
    (tmp_path / "big.md").write_text("y" * 100)
    assert parse.read(tmp_path / "big.md").status == "too_large"


def test_withheld_paragraphs_are_never_bridged(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# A\n\nkeep one\n\nDROP me\n\nkeep two\n")
    chunks, withheld = chunk.chunk(parse.read(p), keep=lambda s, t: "DROP" not in t)
    assert withheld == 1 and [(c.start, c.end) for c in chunks] == [(1, 3), (7, 7)]
    assert all("DROP" not in c.text for c in chunks)
