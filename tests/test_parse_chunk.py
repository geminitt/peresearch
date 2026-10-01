"""Contracts of zetokrag.parse (a file becomes located blocks, a broken one a status) and zetokrag.chunk (blocks
become chunks that lose nothing, invent nothing, stay within their section and say exactly where they are)."""
import json
import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from peresearch.zetokrag import chunk, parse

# --- parse ----------------------------------------------------------------------------------------------------------


@given(st.text(max_size=300).filter(lambda t: "\x00" not in t), st.sampled_from([".md", ".txt", ".py", ".rst"]))
@settings(deadline=None, max_examples=60)
def test_a_text_file_is_one_block_of_lines_holding_the_file_as_read(tmp_path_factory, text, ext):
    p = tmp_path_factory.mktemp("t") / f"n{ext}"
    p.write_bytes(text.encode("utf-8"))
    doc = parse.read(p)
    assert doc.status == "ok" and [(b.unit, b.start) for b in doc.blocks] == [("line", 1)]
    assert doc.blocks[0].text == text


def test_kind_follows_the_extension():
    assert [parse.kind(parse.Path(f"x{e}")) for e in (".md", ".MD", ".txt", ".py", ".pdf", ".ipynb", ".csv", ".exe")] == \
        ["markdown", "markdown", "text", "text", "pdf", "notebook", "unsupported", "unsupported"]


def test_notebook_cells_and_pdf_pages_keep_their_numbers(tmp_path):
    nb = tmp_path / "a.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": ["# Title\n", "text"]},
                                        {"cell_type": "raw", "source": "raw cell"},
                                        {"cell_type": "code", "source": "", "outputs": []},
                                        {"cell_type": "code", "source": "x = 1", "outputs": [{"text": "noise"}]}]}))
    doc = parse.read(nb)
    assert [(b.unit, b.start, b.text) for b in doc.blocks] == [("cell", 1, "# Title\ntext"), ("cell", 4, "x = 1")]
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "a.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "Page one text")
    d.new_page()
    d.new_page().insert_text((72, 72), "Page three text")
    d.save(pdf)
    doc = parse.read(pdf)
    assert doc.status == "ok" and doc.path == pdf and doc.note == "1 pages without text"
    assert [(b.unit, b.start, b.text.strip()) for b in doc.blocks] == [("page", 1, "Page one text"), ("page", 3, "Page three text")]
    full, blank = tmp_path / "full.pdf", tmp_path / "blank.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "only page")
    d.save(full)
    d = pymupdf.open()
    d.new_page()
    d.save(blank)
    assert parse.read(full).note == "" and parse.read(blank).note == "1 pages, no text layer"


def test_a_file_that_cannot_be_read_gets_a_status_never_an_exception(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    scan = tmp_path / "scan.pdf"
    d = pymupdf.open()
    png = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), 0).tobytes("png")
    d.new_page().insert_image(pymupdf.Rect(0, 0, 100, 100), stream=png)
    d.save(scan)
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    (tmp_path / "broken.ipynb").write_text("{not json")
    (tmp_path / "b.txt").write_bytes(b"abc\x00def")
    (tmp_path / "x.exe").write_bytes(b"MZ")
    (tmp_path / "big.md").write_text("y" * 100)
    monkeypatch.setattr(parse, "MAX_BYTES", 10)
    statuses = {name: parse.read(tmp_path / name).status
                for name in ("scan.pdf", "broken.pdf", "broken.ipynb", "b.txt", "x.exe", "big.md", "missing.md")}
    assert statuses == {"scan.pdf": "too_large", "broken.pdf": "too_large", "broken.ipynb": "error", "b.txt": "binary",
                        "x.exe": "unsupported", "big.md": "too_large", "missing.md": "error"}
    big = parse.read(tmp_path / "big.md")
    assert big.path == tmp_path / "big.md" and big.note == "100 bytes"
    monkeypatch.setattr(parse, "MAX_BYTES", 20 * 2**20)
    ocr, broken = parse.read(scan), parse.read(tmp_path / "broken.pdf")
    assert (ocr.status, ocr.note, ocr.path) == ("needs_ocr", "1 pages, no text layer, 1 images", scan)
    assert broken.status == "error" and broken.note.startswith(("FileDataError", "RuntimeError")) and broken.path
    assert parse.read(tmp_path / "missing.md").note.startswith("FileNotFoundError: ")
    assert len(parse.read(tmp_path / "broken.ipynb").note) <= 200
    for name in ("b.txt", "x.exe"):
        assert parse.read(tmp_path / name).path == tmp_path / name


def test_the_size_limit_and_the_binary_check_hold_at_their_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "MAX_BYTES", 100)
    (tmp_path / "at.md").write_text("y" * 100)
    (tmp_path / "over.md").write_text("y" * 101)
    assert parse.read(tmp_path / "at.md").status == "ok" and parse.read(tmp_path / "over.md").status == "too_large"
    monkeypatch.setattr(parse, "MAX_BYTES", 20 * 2**20)
    (tmp_path / "late.txt").write_bytes(b"a" * 8192 + b"\x00")          # a NUL after the first 8 KiB: still text
    (tmp_path / "early.txt").write_bytes(b"a" * 8191 + b"\x00")
    assert parse.read(tmp_path / "late.txt").status == "ok" and parse.read(tmp_path / "early.txt").status == "binary"


def test_text_that_is_not_utf8_is_read_with_replacement_characters(tmp_path):
    p = tmp_path / "latin1.txt"
    p.write_bytes("café naïve\n".encode("latin-1"))
    doc = parse.read(p)
    assert doc.status == "ok" and doc.blocks[0].text == "caf\ufffd na\ufffdve\n"


def test_notebook_cells_without_source_or_with_a_list_source(tmp_path):
    nb = tmp_path / "a.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "markdown"}, {"cell_type": "code", "source": ["a = 1\n", "b = 2"]}]}))
    assert [(b.start, b.text) for b in parse.read(nb).blocks] == [(2, "a = 1\nb = 2")]
    (tmp_path / "empty.ipynb").write_text("{}")
    assert parse.read(tmp_path / "empty.ipynb").blocks == [] and parse.read(tmp_path / "empty.ipynb").status == "ok"


# --- chunk ----------------------------------------------------------------------------------------------------------

words = st.text(alphabet="abcdefghij KLMNOP học máy 机器学习 0123", min_size=1, max_size=40).map(str.strip).filter(bool)
paragraph = st.lists(words, min_size=1, max_size=6).map("\n".join)
long_line = st.integers(1501, 3200).map(lambda n: ("long " * n)[:n])
heading = st.tuples(st.integers(1, 4), words).map(lambda h: "#" * h[0] + " " + h[1])
fence = st.lists(st.one_of(words, words.map(lambda w: "# " + w)), max_size=3).map(lambda b: "```\n" + "\n".join(b) + "\n```")
wide = st.lists(st.text(alphabet="abcdefgh xyz", min_size=60, max_size=110).map(str.strip).filter(bool),
               min_size=15, max_size=40).map("\n".join)                      # a paragraph longer than MAX_CHARS
indented_fence = words.map(lambda w: "  ```\n# " + w + "\n  ```")
lines_then_long = st.tuples(paragraph, long_line, paragraph).map("\n".join)   # an overlong line inside a paragraph
exact = st.integers(1, 3).map(lambda k: "\n".join(["y" * 499] * k) if k < 3 else "z" * 1500)   # 1,499 or 1,500 chars
element = st.one_of(paragraph, paragraph, heading, fence, indented_fence, long_line, wide, lines_then_long, exact)
documents = st.lists(element, min_size=1, max_size=12).map("\n\n".join)
MARK = "WITHHOLD"


def sections(lines: list[str]) -> list[str]:
    """The Markdown heading path in effect at each line (headings inside code fences do not count)."""
    out, heads, fenced = [], [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
        m = None if fenced else re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            heads = heads[:len(m.group(1)) - 1] + [m.group(2)]
        out.append(" > ".join(heads))
    return out


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


@given(documents, st.sampled_from([".md", ".markdown", ".txt"]))
@settings(deadline=None, max_examples=150)
@example("# Tokenizers\n\nBPE merges pairs.\n\n## SuperBPE\n\nSpans across words.\nSecond line.\n", ".md")
@example("# A\n\n```bash\n# not a heading\n```\n", ".md")                  # a comment in a fence is not a heading
@example("a" * 749 + "\n\n" + "b" * 749, ".txt")                       # 749 + 2 + 749 = 1,500: one chunk
@example("a" * 749 + "\n\n" + "b" * 750, ".txt")                       # 1,501: two chunks
@example("x" * 1500, ".txt")                                             # exactly the limit: not cut
@example("short\n" + "x" * 1501 + "\nafter", ".txt")                   # an overlong line inside a paragraph
def test_chunks_lose_nothing_invent_nothing_and_say_where_they_are(tmp_path_factory, text, ext):
    p = tmp_path_factory.mktemp("c") / f"n{ext}"
    p.write_text(text)
    chunks, withheld = chunk.chunk(parse.read(p))
    lines = text.splitlines()
    assert withheld == 0
    assert squeeze("".join(c.text for c in chunks)) == squeeze(text)          # every character, once, in order
    where = sections(lines) if ext != ".txt" else [""] * len(lines)
    for c in chunks:
        assert c.unit == "line" and 1 <= c.start <= c.end <= len(lines) and 0 < len(c.text.strip()) <= chunk.MAX_CHARS
        span = "\n".join(lines[c.start - 1:c.end])
        assert all(part.strip() in span for part in c.text.split("\n\n"))       # verbatim, inside its lines
        body = c.text.strip().split("\n")
        assert body[0].strip() in lines[c.start - 1] and body[-1].strip() in lines[c.end - 1]   # exactly its lines
        assert {where[i - 1] for i in range(c.start, c.end + 1) if lines[i - 1].strip()} == {c.section}
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)
    for a, b in zip(chunks, chunks[1:]):                       # packed greedily: the next paragraph did not fit
        if a.section == b.section and a.end < b.start:
            assert len(a.text) + 2 + len(b.text.split("\n\n")[0]) > chunk.MAX_CHARS


@given(st.lists(st.tuples(paragraph, st.booleans()), min_size=1, max_size=10))
@settings(deadline=None, max_examples=100)
@example([("keep one", False), ("drop me", True), ("keep two", False)])
def test_withheld_paragraphs_are_left_out_and_never_bridged(tmp_path_factory, parts):
    text = "# A\n\n" + "\n\n".join(f"{MARK} {para}" if drop else para for para, drop in parts) + "\n"
    p = tmp_path_factory.mktemp("w") / "n.md"
    p.write_text(text)
    chunks, withheld = chunk.chunk(parse.read(p), keep=lambda section, t: MARK not in t)
    lines = text.splitlines()
    dropped = {i for i, line in enumerate(lines, 1) if MARK in line}
    assert withheld == sum(drop for _, drop in parts)
    assert all(MARK not in c.text and not dropped & set(range(c.start, c.end + 1)) for c in chunks)
    kept = "".join(line for i, line in enumerate(lines, 1) if i not in dropped and not _in_dropped_paragraph(lines, i, dropped))
    assert squeeze("".join(c.text for c in chunks)) == squeeze(kept)


def _in_dropped_paragraph(lines, i, dropped):
    """Whether line i belongs to a paragraph (lines between blank lines) that holds a dropped line."""
    a = i
    while a > 1 and lines[a - 2].strip():
        a -= 1
    b = i
    while b < len(lines) and lines[b].strip():
        b += 1
    return bool(dropped & set(range(a, b + 1)))


def test_cells_and_pages_are_chunked_within_their_unit(tmp_path):
    nb = tmp_path / "a.ipynb"
    cells = [{"cell_type": "markdown", "source": "first\n\nsecond"}, {"cell_type": "code", "source": "x" * 2000},
             {"cell_type": "code", "source": "\n".join(["y" * 80] * 30)}]                 # 2,429 chars on 30 lines
    nb.write_text(json.dumps({"cells": cells}))
    chunks, _ = chunk.chunk(parse.read(nb))
    assert all(c.unit == "cell" and c.start == c.end and c.section == "" for c in chunks)
    assert [c.start for c in chunks] == [1, 2, 2, 3, 3] and all(len(c.text) <= chunk.MAX_CHARS for c in chunks)
    assert squeeze("".join(c.text for c in chunks)) == squeeze("first second" + "x" * 2000 + "y" * 2400)
