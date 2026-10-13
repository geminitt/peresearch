"""Contracts of zetokrag.parse (a file becomes located blocks, a broken one a status) and zetokrag.chunk (blocks
become chunks that lose nothing, invent nothing, stay within their section and say exactly where they are)."""
import ast
import codecs
import json
from pathlib import Path
import re
import unicodedata

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from peresearch.zetokrag import chunk, parse

# --- parse ----------------------------------------------------------------------------------------------------------


@given(st.text(max_size=300).filter(lambda t: "\x00" not in t), st.sampled_from([".md", ".txt", ".py", ".rst"]),
       st.sampled_from(["utf-8", "utf-8-sig", "utf-16", "utf-32"]))
@settings(deadline=None, max_examples=120)
def test_a_text_file_is_one_block_of_lines_holding_the_file_as_read(tmp_path_factory, text, ext, encoding):
    """UTF-8, or any encoding its byte order mark names (UTF-16 and UTF-32 hold NUL bytes: still text), read back
    exactly, the mark left out."""
    p = tmp_path_factory.mktemp("t") / f"n{ext}"
    p.write_bytes(text.encode(encoding))
    doc = parse.read(p)
    assert doc.status == "ok" and [(b.unit, b.start) for b in doc.blocks] == [("line", 1)]
    assert doc.blocks[0].text == text


def test_kind_follows_the_extension():
    assert [parse.kind(parse.Path(f"x{e}")) for e in (".md", ".MD", ".txt", ".py", ".pdf", ".ipynb", ".csv", ".exe")] == \
        ["markdown", "markdown", "text", "text", "pdf", "notebook", "data", "unsupported"]
    assert {parse.kind(parse.Path(f"x{e}")) for e in (".csv", ".TSV", ".json", ".jsonl", ".log")} == {"data"}
    assert [parse.kind(parse.Path(f"x{e}")) for e in (".docx", ".DOCX", ".pptx", ".xlsx", ".html", ".htm", ".doc")] == \
        ["docx", "docx", "pptx", "spreadsheet", "html", "html", "unsupported"]
    # what is converted (its text is not its bytes): kept in the extracted-text store, grepped there, opened outside
    assert parse.CONVERTED == {"pdf", "notebook", "docx", "pptx", "spreadsheet", "html"}


def test_a_results_table_or_a_log_is_read_as_lines_in_its_encoding(tmp_path):
    """Tables, JSON and logs hold a research project's results: the tools read them like any text."""
    (tmp_path / "results.csv").write_bytes(cp1258("model,score\nphở bò tái,0.91\nbún chả Hà Nội,0.87\n") * 2)
    doc = parse.read(tmp_path / "results.csv")
    assert doc.status == "ok" and [b.unit for b in doc.blocks] == ["line"] and doc.blocks[0].text.startswith("model,score\nphở")
    assert doc.note.startswith("decoded as")


def test_notebook_cells_and_pdf_pages_keep_their_numbers(tmp_path):
    nb = tmp_path / "a.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": ["# Title\n", "text"]},
                                        {"cell_type": "raw", "source": "raw cell"},
                                        {"cell_type": "code", "source": "", "outputs": []},
                                        {"cell_type": "code", "source": "x = 1", "outputs": [{"text": "noise"}]}]}))
    doc = parse.read(nb)
    assert [(b.unit, b.start, b.text) for b in doc.blocks] == [("cell", 1, "# Title\ntext"),
                                                                ("cell", 4, "x = 1\n\nOutput:\nnoise")]
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


def test_office_documents_and_web_pages_are_read_as_their_text(tmp_path):
    """Word as Markdown (headings, paragraphs, lists, tables), so it is cut by its sections; PowerPoint a page per
    slide (title, text, tables, the speaker's notes); Excel a table per sheet, tab-separated (data, as a .csv: read by
    the tools, not ranked); an HTML page as its main text in Markdown, a template (little text in much markup) as
    the code it is. What fails to open is an error, never an exception."""
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Tokenizers", level=1)
    d.add_paragraph("Byte pair encoding merges the most frequent pair.")
    d.add_heading("BPE", level=2)
    d.add_paragraph("first step", style="List Bullet")
    d.add_paragraph("second step", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    for (r, c), v in {(0, 0): "model", (0, 1): "score", (1, 0): "bpe", (1, 1): "0.5"}.items():
        t.cell(r, c).text = v
    d.add_paragraph("")
    d.save(tmp_path / "a.docx")
    doc = parse.read(tmp_path / "a.docx")
    assert doc.status == "ok" and doc.markdown and [(b.unit, b.start) for b in doc.blocks] == [("line", 1)]
    assert parse.lines(doc.blocks[0].text) == [
        "# Tokenizers", "", "Byte pair encoding merges the most frequent pair.", "", "## BPE", "", "- first step",
        "- second step", "", "| model | score |", "| --- | --- |", "| bpe | 0.5 |"]
    assert [c.section for c in chunk.chunk(doc)[0]][-1] == "Tokenizers > BPE"           # cut by its sections

    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    one = deck.slides.add_slide(deck.slide_layouts[1])
    one.shapes.title.text = "Results"
    one.placeholders[1].text_frame.text = "BPE wins\non short texts"
    deck.slides.add_slide(deck.slide_layouts[6])                                          # a blank slide
    three = deck.slides.add_slide(deck.slide_layouts[5])
    three.shapes.title.text = "Table"
    table = three.shapes.add_table(2, 2, 0, 0, 100, 100).table
    for (r, c), v in {(0, 0): "a", (0, 1): "b", (1, 0): "1", (1, 1): "2"}.items():
        table.cell(r, c).text = v
    three.notes_slide.notes_text_frame.text = "say why"
    deck.save(tmp_path / "a.pptx")
    doc = parse.read(tmp_path / "a.pptx")
    assert doc.status == "ok" and not doc.markdown
    assert [(b.unit, b.start, b.text) for b in doc.blocks] == [
        ("page", 1, "Results\nBPE wins\non short texts"),
        ("page", 3, "Table\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nNotes: say why")]

    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Results"
    for row in (["model", "score"], ["bpe", 0.5], [None, None], ["unigram", 0.25]):
        wb.active.append(row)
    wb.create_sheet("Empty")
    wb.create_sheet("Notes").append(["x"])
    wb.save(tmp_path / "a.xlsx")
    doc = parse.read(tmp_path / "a.xlsx")
    assert doc.status == "ok" and [(b.unit, b.start) for b in doc.blocks] == [("line", 1)]
    assert parse.lines(doc.blocks[0].text) == ["## Results", "model\tscore", "bpe\t0.5", "unigram\t0.25", "",
                                              "## Notes", "x"]

    article = ("<html><head><title>BPE</title></head><body><nav><a href='/'>Home</a></nav><article><h1>BPE</h1>"
               + "".join(f"<p>Step {i}: byte pair encoding merges the most frequent pair of symbols in the "
                         f"corpus, then counts the pairs again ({i} merges so far).</p>" for i in range(1, 9))
               + "</article><footer>© 2026</footer></body></html>")
    (tmp_path / "page.html").write_text(article)
    doc = parse.read(tmp_path / "page.html")
    assert doc.status == "ok" and doc.markdown and doc.note == "main text of the page"
    text = doc.blocks[0].text
    assert text.startswith("# BPE") and "Step 8: byte pair encoding" in text and "Home" not in text and "©" not in text
    from peresearch.zetokrag.index import is_data
    long = parse.Doc(tmp_path / "long.html", "ok", [parse.Block("# Title\n\n" + "A long paragraph of prose. " * 40, "line", 1)],
                     note="main text of the page", markdown=True)
    assert is_data(long) is None                     # a page's text is prose, never judged as code (minified)
    template = "{% extends 'base.html' %}\n{% block content %}\n<div class=\"x\">{{ user.name }}</div>\n{% endblock %}\n"
    (tmp_path / "base.html").write_text(template)
    doc = parse.read(tmp_path / "base.html")
    assert doc.status == "ok" and not doc.markdown and doc.blocks[0].text == template and doc.note == ""

    (tmp_path / "broken.docx").write_bytes(b"not a zip at all")
    for name in ("broken.docx",):
        assert parse.read(tmp_path / name).status == "error"
    (tmp_path / "broken.pptx").write_bytes(b"PK\x03\x04 truncated")
    (tmp_path / "broken.xlsx").write_bytes(b"")
    assert parse.read(tmp_path / "broken.pptx").status == "error" and parse.read(tmp_path / "broken.xlsx").status == "error"


def test_converted_documents_hold_at_their_edges(tmp_path, monkeypatch):
    """Word: a Title, numbered lists, a paragraph numbered without a list style, a line break inside a paragraph, an
    empty paragraph between two others, a table cell holding a bar and a line break; PowerPoint: a group's text, a
    slide with only notes; Excel: spaces in a cell, a formula (its value, never the formula); an empty document of
    each kind holds nothing; each is limited by its text (MAX_TEXT_CHARS) as a PDF is; a short HTML page is code; a
    page's links and comments are left out, its tables kept; raw HTML is judged as code (minified)."""
    docx = pytest.importorskip("docx")
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from peresearch.zetokrag.index import is_data
    d = docx.Document()
    d.add_paragraph("Notes on tokenizers", style="Title")
    d.add_paragraph("count the pairs", style="List Number")
    d.add_paragraph("merge the best", style="List Number")
    numbered = d.add_paragraph("numbered by hand")
    num = OxmlElement("w:numPr")
    for tag, val in (("w:ilvl", "0"), ("w:numId", "1")):
        e = OxmlElement(tag)
        e.set(qn("w:val"), val)
        num.append(e)
    numbered._p.get_or_add_pPr().append(num)
    broken = d.add_paragraph("first half")
    broken.add_run().add_break()
    broken.add_run("second half")
    d.add_paragraph("")
    d.add_paragraph("after an empty one")
    d.add_paragraph("key\tvalue  as typed")                          # tabs and spaces inside a line kept
    from lxml import etree
    t = d.add_table(rows=2, cols=1)
    t._tbl.addprevious(etree.Comment(" a note in the XML "))        # not an element: its tag is a function, not a
    t.cell(0, 0).text = "a | b\nc"                                   # name (python-docx's own test files hold some)
    t.cell(1, 0).text = "x"
    d.save(tmp_path / "b.docx")
    doc = parse.read(tmp_path / "b.docx")
    text = doc.blocks[0].text
    assert doc.note == "converted from Word" and parse.lines(text) == [
        "# Notes on tokenizers", "", "1. count the pairs", "1. merge the best", "- numbered by hand", "",
        "first half second half", "", "after an empty one", "", "key\tvalue  as typed", "", "| a \\| b c |",
        "| --- |", "| x |"]
    plain = docx.Document()
    plain.styles["Normal"].element.set(qn("w:default"), "0")          # no default style: a paragraph has none
    plain.add_paragraph("unstyled")
    plain.save(tmp_path / "plain.docx")
    assert parse.lines(parse.read(tmp_path / "plain.docx").blocks[0].text) == ["unstyled"]
    docx.Document().save(tmp_path / "empty.docx")
    assert parse.read(tmp_path / "empty.docx").blocks == []
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", len(text))
    assert parse.read(tmp_path / "b.docx").status == "ok"                               # its text fits exactly
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", len(text) - 1)
    over = parse.read(tmp_path / "b.docx")
    assert over.status == "error" and f"over {len(text) - 1} characters of text" in over.note
    monkeypatch.undo()

    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches
    deck = pptx.Presentation()
    one = deck.slides.add_slide(deck.slide_layouts[5])             # "Title Only"
    one.shapes.title.text = "Intro"
    one.shapes.add_textbox(Inches(1), Inches(3), Inches(2), Inches(1)).text_frame.text = "before"
    title = one.shapes.title._element
    title.getparent().append(title)                                  # the title last in the shapes' order: still first
    group = one.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text_frame.text = "inside a group"
    one.shapes.add_table(1, 1, 0, 0, 100, 100).table.cell(0, 0).text = "cell"
    two = deck.slides.add_slide(deck.slide_layouts[6])
    two.notes_slide.notes_text_frame.text = "only notes here"
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    data = CategoryChartData()
    data.categories = ["en", "vi"]
    data.add_series("bpe", (0.5, 0.6))
    data.add_series("unigram", (0.4, None))                         # a missing value: an empty cell
    three = deck.slides.add_slide(deck.slide_layouts[6])            # a slide holding only a chart (as MarkItDown)
    chart = three.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, 0, 0, Inches(4), Inches(3), data).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "Scores"
    from pptx.chart.data import XyChartData
    xy = XyChartData()
    xy.add_series("loss").add_data_point(1, 2.5)
    four = deck.slides.add_slide(deck.slide_layouts[6])             # no title and no categories: the word alone
    four.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, 0, 0, Inches(4), Inches(3), xy).chart.has_title = False
    from pptx.oxml.ns import qn as pqn
    odd = four.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, 0, 0, Inches(4), Inches(3), data).chart
    odd.has_title = True
    odd.chart_title.text_frame.text = "Odd"
    odd._chartSpace.xpath(".//c:barChart")[0].tag = pqn("c:area3DChart")   # a kind python-pptx cannot read: the
                                                                          # title kept, the deck still read
    deck.save(tmp_path / "b.pptx")
    doc = parse.read(tmp_path / "b.pptx")
    assert doc.note == "converted from PowerPoint"
    first = "Intro\nbefore\ninside a group\n| cell |\n| --- |"                               # every shape kept, in order
    chart = "Chart: Scores\n| Category | bpe | unigram |\n| --- | --- | --- |\n| en | 0.5 | 0.4 |\n| vi | 0.6 |  |"
    assert [(b.start, b.text) for b in doc.blocks] == [(1, first), (2, "Notes: only notes here"), (3, chart),
                                                        (4, "Chart\nChart: Odd")]

    class Unreadable:                                   # a chart whose every part raises, its title too: the word alone
        def __getattr__(self, name):
            raise RuntimeError(name)
    assert parse._chart(Unreadable()) == ["Chart"]
    pptx.Presentation().save(tmp_path / "empty.pptx")
    assert parse.read(tmp_path / "empty.pptx").blocks == []
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", len(first) + len("Notes: only notes here") + len(chart) + 16)
    assert parse.read(tmp_path / "b.pptx").status == "ok"                             # it fits exactly
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", len(first) + len("Notes: only notes here") + len(chart) + 15)
    assert parse.read(tmp_path / "b.pptx").status == "error"                          # the deck's text, all slides
    monkeypatch.undo()

    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.append(["  spaced   out\ncell ", 2])
    wb.active.append(["=1+1", "=SUM(B1:B1)"])                                        # formulas, no cached value
    wb.save(tmp_path / "b.xlsx")
    doc = parse.read(tmp_path / "b.xlsx")
    assert doc.note == "converted from Excel" and doc.path == tmp_path / "b.xlsx"
    assert parse.lines(doc.blocks[0].text) == ["## Sheet", "spaced out cell\t2"]       # never a formula's text
    openpyxl.Workbook().save(tmp_path / "empty.xlsx")
    assert parse.read(tmp_path / "empty.xlsx").blocks == []
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", len("spaced out cell\t2"))
    assert parse.read(tmp_path / "b.xlsx").status == "ok"
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", len("spaced out cell\t2") - 1)
    assert parse.read(tmp_path / "b.xlsx").status == "error"
    monkeypatch.undo()

    short = "<html><body><article><h1>Hi</h1><p>One short line of text only.</p></article></body></html>"
    (tmp_path / "short.html").write_text(short)
    assert parse.read(tmp_path / "short.html").markdown is False                        # under HTML_MIN_TEXT: code
    rich = ("<html><body><article><h1>Results</h1>" + "".join(
        f"<p>Run {i} compared the tokenizers on the same corpus and kept the <a href='https://e.org/{i}'>log "
        f"{i}</a> for later.</p>" for i in range(1, 6)) + "<table><tr><th>model</th><th>score</th></tr><tr><td>bpe</td>"
        "<td>0.5</td></tr></table></article><div id='comments' class='comments'><p>Nice post, visit my shop</p></div>"
        "</body></html>")
    (tmp_path / "rich.html").write_text(rich)
    text = parse.read(tmp_path / "rich.html").blocks[0].text
    assert "| model | score |" in text and "bpe" in text                                  # its table kept
    assert "https://" not in text and "visit my shop" not in text                      # links and comments left out
    (tmp_path / "min.html").write_text("<div class=a>" + "<i class=b></i>" * 900 + "</div>\n")   # markup, no text
    assert is_data(parse.read(tmp_path / "min.html")) == "minified code"               # read raw: judged as code
    (tmp_path / "packed.html").write_text("<article>" + "".join(f"<p>Packed page, point {i}: the text is what "
                                          f"counts.</p>" for i in range(12)) + "</article>")   # minified, with text
    assert is_data(parse.read(tmp_path / "packed.html")) is None                       # its text is read, not judged


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
    assert statuses == {"scan.pdf": "needs_ocr", "broken.pdf": "error", "broken.ipynb": "error", "b.txt": "binary",
                        "x.exe": "unsupported", "big.md": "too_large", "missing.md": "error"}   # a PDF: its text decides
    big = parse.read(tmp_path / "big.md")
    assert big.path == tmp_path / "big.md" and big.note == "100 bytes"
    monkeypatch.setattr(parse, "MAX_BYTES", 20 * 2**20)
    ocr, broken = parse.read(scan), parse.read(tmp_path / "broken.pdf")
    assert (ocr.status, ocr.note, ocr.path) == ("needs_ocr", "1 pages, no text layer, 1 images", scan)
    assert broken.status == "error" and broken.note.startswith(("FileDataError", "RuntimeError")) and broken.path
    assert parse.read(tmp_path / "missing.md").note.startswith("FileNotFoundError: ")
    assert len(parse.read(tmp_path / "broken.ipynb").note) <= 200
    web = parse.read_pdf(b"%PDF-1.4 garbage", "https://example.org/a.pdf")         # a PDF fetched from the web
    assert (web.status, web.path) == ("error", Path("https://example.org/a.pdf"))
    assert web.note.startswith(("FileDataError: ", "RuntimeError: ")) and len(web.note) <= 200
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


def test_text_whose_encoding_cannot_be_recognized_is_read_with_replacement_characters_and_said_so(tmp_path):
    p = tmp_path / "latin1.txt"
    p.write_bytes("café naïve\n".encode("latin-1"))                    # too short for a language to show
    short = tmp_path / "vi.txt"
    short.write_bytes(cp1258("Học máy là gì?"))                          # 18 bytes: guessed shift_jis_2004 before
    assert parse.read(short).note.startswith("not UTF-8 and the encoding was not recognized")
    doc = parse.read(p)
    assert doc.status == "ok" and doc.blocks[0].text == "caf\ufffd na\ufffdve\n"
    assert doc.note == "not UTF-8 and the encoding was not recognized: unreadable bytes replaced"


@pytest.mark.parametrize("coherence, size, text, trusted", [
    (0.5, 10, "short but recognized", True), (0.0, 64, "long enough, plausible", True),
    (0.0, 63, "one byte short", False), (0.0, 64, "rare \U00020000 character", False), (None, 64, "", False),
], ids=["recognized", "64-bytes", "63-bytes", "implausible", "no-guess"])
def test_a_guessed_encoding_is_trusted_only_by_its_rule(monkeypatch, coherence, size, text, trusted):
    """Not UTF-8: the guess is kept when its language is recognized (coherence > 0), or the text is at least
    MIN_GUESS_BYTES long and holds no character a wrong guess lands in; else UTF-8 with replacement."""
    import charset_normalizer
    best = None if coherence is None else type("G", (), {"coherence": coherence, "encoding": "cp1252",
                                                          "__str__": lambda self: text})()
    monkeypatch.setattr(charset_normalizer, "from_bytes", lambda raw: type("R", (), {"best": lambda self: best})())
    raw = b"\xff" * size                                                          # never UTF-8
    got, note = parse.decode(raw)
    assert (got, note == "decoded as cp1258 (guessed)".replace("cp1258", "cp1252")) == (
        (text, True) if trusted else ("\ufffd" * size, False))


@pytest.mark.parametrize("code, plausible", [
    (0x1FFFF, True), (0x20000, False), (0xDFFF, True), (0xE000, False), (0xF8FF, False), (0xF900, True),
    (0xFB4F, True), (0xFB50, False), (0xFDFF, False), (0xFE00, True), (0xFE6F, True), (0xFE70, False),
    (0xFEFC, False), (0xFEFD, True)])
def test_a_wrong_guess_is_told_by_the_ranges_it_lands_in(code, plausible):
    assert parse._plausible("ab" + chr(code)) is plausible


def test_running_heads_are_lines_on_at_least_three_pages_and_half_of_them():
    head = ["Journal of Tests"]
    assert parse._boilerplate([head, head]) == set()                                # two pages: too few to tell
    assert parse._boilerplate([head, head, head]) == {"Journal of Tests"}
    assert parse._boilerplate([head, head, []]) == set()                            # 2 of 3 pages
    assert parse._boilerplate([head * 2, head, []]) == set()                        # twice on a page counts once
    assert parse._boilerplate([head] * 4 + [[]] * 4) == {"Journal of Tests"}        # 4 of 8: half
    assert parse._boilerplate([head] * 3 + [[]] * 5) == set()                       # 3 of 8
    assert parse._boilerplate([[f"Page {i}"] for i in range(1, 4)]) == {parse._key("Page 1")}   # digits ignored


LETTER_MARKS = {"\u0302", "\u0306", "\u031b"}                          # circumflex, breve, horn: letters in cp1258


def cp1258(text: str) -> bytes:
    """Vietnamese as cp1258 files hold it: base letters (ă â ê ô ơ ư đ) followed by combining tone marks."""
    out, i, d = [], 0, unicodedata.normalize("NFD", text)
    while i < len(d):
        base, marks = d[i], []
        i += 1
        while i < len(d) and unicodedata.combining(d[i]):
            marks.append(d[i])
            i += 1
        out.append(unicodedata.normalize("NFC", base + "".join(m for m in marks if m in LETTER_MARKS))
                   + "".join(m for m in marks if m not in LETTER_MARKS))
    return "".join(out).encode("cp1258")


VI = "Học máy là một lĩnh vực của trí tuệ nhân tạo, nghiên cứu cách máy tính học từ dữ liệu mà không cần lập trình. "
ZH = "机器学习是人工智能的一个分支，研究计算机如何从数据中学习，而不需要明确的编程。深度学习使用多层神经网络。"
JA = "機械学習は人工知能の一分野であり、コンピュータがデータから学習する方法を研究する。深層学習は多層ニューラルネットを用いる。"
KO = "기계 학습은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하는 방법을 연구한다. 딥러닝은 다층 신경망을 사용한다."
HI = "यंत्र अधिगम कृत्रिम बुद्धि की एक शाखा है जो डेटा से सीखने वाले एल्गोरिदम का अध्ययन करती है। "


ZH_T = "機器學習是人工智慧的一個分支，研究電腦如何從資料中學習。深度學習使用多層神經網路。"


@pytest.mark.parametrize("text, raw, said", [
    (VI * 3, cp1258(VI * 3), r"decoded as cp1258 \(guessed\)"),
    (ZH * 2, (ZH * 2).encode("gbk"), r"decoded as gb18030 \(guessed\)"),
    (ZH_T * 2, (ZH_T * 2).encode("big5"), r"decoded as (big5|cp950|big5hkscs) \(guessed\)"),
    (JA * 2, (JA * 2).encode("shift_jis"), r"decoded as (cp932|shift_jis\S*) \(guessed\)"),
    (KO * 2, (KO * 2).encode("euc_kr"), r"decoded as (cp949|euc_kr) \(guessed\)"),
    (HI * 2, codecs.BOM_UTF16_LE + (HI * 2).encode("utf-16-le"), r"UTF-16 \(byte order mark\)"),
    (VI * 2, codecs.BOM_UTF32_BE + (VI * 2).encode("utf-32-be"), r"UTF-32 \(byte order mark\)"),
    (VI, codecs.BOM_UTF8 + VI.encode("utf-8"), r"UTF-8 \(byte order mark\)"),
], ids=["vi-cp1258", "zh-gbk", "zh-big5", "ja-shift_jis", "ko-euc_kr", "hi-utf16-bom", "vi-utf32-bom", "vi-utf8-bom"])
def test_text_is_read_in_the_encoding_it_was_written_in(tmp_path, text, raw, said):
    p = tmp_path / "note.txt"
    p.write_bytes(raw)
    doc = parse.read(p)
    assert doc.status == "ok" and re.fullmatch(said, doc.note), doc.note
    assert doc.blocks[0].text == text                                    # composed (NFC), as typed queries are


def test_a_byte_order_mark_file_is_text_not_binary(tmp_path):
    p = tmp_path / "w.txt"
    p.write_bytes(codecs.BOM_UTF16_LE + "a\nb\n".encode("utf-16-le"))         # NUL bytes, yet text
    assert parse.read(p).status == "ok" and parse.read(p).blocks[0].text == "a\nb\n"


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
# Characters Python's str.splitlines() breaks at but editors, git and ripgrep do not, and Windows line ends
odd = st.tuples(words, st.sampled_from(["\x0c", "\u2028", "\u2029", "\x85", "\x0b", "\x1c", "\r"]), words).map("".join)
crlf = st.lists(words, min_size=1, max_size=3).map("\r\n".join)
# The blocks the default chunking keeps whole (a table, a list, a fence with blank lines), up to and past BLOCK_CHARS
table = st.lists(words, min_size=2, max_size=8).map(lambda ws: "| k | v |\n|---|---|\n" + "\n".join(f"| {w} | {w} |"
                                                                                               for w in ws))
big_table = st.integers(40, 80).map(lambda n: "\n".join(["| k | " + "v" * 91 + " |"] * n))   # 3,960 to 7,920 chars
bullets = st.tuples(st.sampled_from(["- ", "* ", "1. ", "  - "]), st.lists(words, min_size=1, max_size=8)).map(
    lambda t: "\n".join(t[0] + w for w in t[1]))
blank_fence = st.lists(words, min_size=2, max_size=4).map(lambda b: "```\n" + "\n\n".join(b) + "\n```")
# A paragraph kept on one line, in sentences of the six target languages (cut after a sentence end)
PROSE = ("Gradient descent moves the weights, one small step at a time. ", "Thuật toán cập nhật trọng số từng bước nhỏ. ",
         "梯度下降法沿着梯度的反方向更新权重。", "勾配降下法は重みを少しずつ更新する。", "경사 하강법은 가중치를 조금씩 갱신한다! ",
         "ग्रेडिएंट डिसेंट भार को थोड़ा-थोड़ा बदलता है। ")
sentence_line = st.lists(st.sampled_from(PROSE), min_size=10, max_size=60).map(lambda ss: "".join(ss).strip())
element = st.one_of(paragraph, paragraph, heading, fence, indented_fence, long_line, wide, lines_then_long, exact,
                    odd, crlf, table, big_table, bullets, blank_fence, sentence_line)
documents = st.lists(element, min_size=1, max_size=12).map("\n\n".join)
# Python that parses: top-level definitions (decorators and comments above them, blank lines inside), statements
name = st.sampled_from(["f", "load", "train_step", "Model", "run_2"])
body = st.lists(words.map(lambda w: f"    x = {w!r}"), min_size=1, max_size=4)
definition = st.tuples(st.lists(st.sampled_from(["# note", "@cache", "@app.route('/x')"]), max_size=2), name, body,
                       st.booleans(), st.booleans()).map(
    lambda t: "\n".join(t[0] + [f"{'class' if t[4] else 'def'} {t[1]}{'' if t[4] else '()'}:"] + t[2][:1]
                        + ([""] if t[3] else []) + t[2][1:]))
statement = st.one_of(words.map(lambda w: f"value = {w!r}"), words.map(lambda w: "# " + w), st.just("import os"))
python = st.lists(st.one_of(definition, definition, statement), min_size=1, max_size=10).map("\n\n".join)
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


def editor_lines(text: str) -> list[str]:
    """Lines as an editor, git or ripgrep number them: each ends at "\n", a "\r" before it belongs to the break."""
    out = text.split("\n")
    if out[-1] == "":
        out.pop()
    return [x[:-1] if x.endswith("\r") else x for x in out]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


@pytest.mark.parametrize("opt", list(chunk.PRESETS.values()), ids=list(chunk.PRESETS))
@given(st.one_of(documents, python), st.sampled_from([".md", ".markdown", ".txt", ".py"]))
@settings(deadline=None, max_examples=150)
@example("# Tokenizers\n\nBPE merges pairs.\n\n## SuperBPE\n\nSpans across words.\nSecond line.\n", ".md")
@example("# A\n\n```bash\n# not a heading\n```\n", ".md")                  # a comment in a fence is not a heading
@example("a" * 749 + "\n\n" + "b" * 749, ".txt")                       # 749 + 2 + 749 = 1,500: one chunk
@example("a" * 749 + "\n\n" + "b" * 750, ".txt")                       # 1,501: two chunks
@example("x" * 1500, ".txt")                                             # exactly the limit: not cut
@example("short\n" + "x" * 1501 + "\nafter", ".txt")                   # an overlong line inside a paragraph
@example("# T\r\n\r\nfirst\x0c part\r\nsecond\u2028x\nTARGET line\n", ".md")    # numbered as an editor shows it
@example("y" * 499 + "\n\n" + "z" * 1500, ".py")       # a Python stretch with a blank line, cut: never on the blank
@example("a" * 1000 + "\n" + "b" * 499 + "\n\n" + "c" * 10, ".py")   # a piece that would start on the blank line
@example("a" * 700 + "\n" + "b" * 700 + "\n\n" + "z" * 1500, ".py")   # one of three lines that would end on it
@example("a" * 1499 + "\n\n\n" + "z" * 1500, ".py")      # a piece of blank lines only: left out, the rest kept
def test_chunks_lose_nothing_invent_nothing_and_say_where_they_are(tmp_path_factory, opt, text, ext):
    p = tmp_path_factory.mktemp("c") / f"n{ext}"
    p.write_bytes(text.encode())
    chunks, withheld = chunk.chunk(parse.read(p), options=opt)
    lines = editor_lines(text)
    assert withheld == 0
    assert squeeze("".join(c.text for c in chunks)) == squeeze(text)          # every character, once, in order
    where = sections(lines) if ext in (".md", ".markdown") else [""] * len(lines)        # headings: Markdown only
    for c in chunks:
        fits = opt.size(c.text) <= opt.cap or (opt.blocks and len(c.text) <= chunk.BLOCK_CHARS)
        assert c.unit == "line" and 1 <= c.start <= c.end <= len(lines) and 0 < len(c.text.strip()) and fits
        span = "\n".join(lines[c.start - 1:c.end])
        assert all(part.strip() in span for part in c.text.split("\n\n"))       # verbatim, inside its lines
        body = c.text.strip().split("\n")
        assert body[0].strip() in lines[c.start - 1] and body[-1].strip() in lines[c.end - 1]   # exactly its lines
        assert {where[i - 1] for i in range(c.start, c.end + 1) if lines[i - 1].strip()} == {c.section}
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)
    for a, b in zip(chunks, chunks[1:]):                       # packed greedily: the next paragraph did not fit
        # (a kept block's parts hold blank lines; so does a Python definition, the unit code_defs packs)
        if a.section == b.section and a.end < b.start and not opt.blocks and not (opt.code_defs and ext == ".py"):
            assert opt.size(a.text + "\n\n" + b.text.split("\n\n")[0]) > opt.cap


@given(st.lists(st.tuples(paragraph, st.booleans()), min_size=1, max_size=10))
@settings(deadline=None, max_examples=100)
@example([("keep one", False), ("drop me", True), ("keep two", False)])
# a secret in one long line, where the line is cut at the size: the halves were stored (a key, reproduced 2026-10-04)
@example([("keep one", False), ("x" * 1480 + " middle " + "y" * 1500, True), ("keep two", False)])
def test_withheld_paragraphs_are_left_out_and_never_bridged(tmp_path_factory, parts):
    text = "# A\n\n" + "\n\n".join(f"{MARK} {para}" if drop else para for para, drop in parts) + "\n"
    p = tmp_path_factory.mktemp("w") / "n.md"
    p.write_bytes(text.encode())
    chunks, withheld = chunk.chunk(parse.read(p), keep=lambda section, t: MARK not in t)
    lines = editor_lines(text)
    dropped = {i for i, line in enumerate(lines, 1) if MARK in line}
    assert withheld == sum(drop for _, drop in parts)
    assert all(MARK not in c.text and not dropped & set(range(c.start, c.end + 1)) for c in chunks)
    kept = "".join(line for i, line in enumerate(lines, 1) if i not in dropped and not _in_dropped_paragraph(lines, i, dropped))
    assert squeeze("".join(c.text for c in chunks)) == squeeze(kept)
    # judged with its heading path: a secret in a heading withholds what is under it (the path is stored with a chunk)
    q = tmp_path_factory.mktemp("h") / "n.md"
    q.write_text(f"# key {MARK}\n\nbody one\n\n## Sub\n\nbody two\n")
    under, held = chunk.chunk(parse.read(q), keep=lambda section, t: MARK not in t and MARK not in section)
    assert (under, held) == ([], 4)


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


def _pdf_pages(path, pages):
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    for lines in pages:
        page = d.new_page()
        for y, text in lines:
            page.insert_text((72, y), text)
    d.save(path)


def test_running_heads_and_page_numbers_leave_a_pdf_and_the_text_stays(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _pdf_pages(pdf, [[(40, "Journal of Tests, Vol. 3"), (300, f"Body of page {i}"), (320, "Figure 1: a caption")
                      if i < 2 else (320, f"More on page {i}"), (800, f"Page {i + 1}")] for i in range(4)])
    doc = parse.read(pdf)
    assert [b.start for b in doc.blocks] == [1, 2, 3, 4]
    for i, b in enumerate(doc.blocks):
        assert f"Body of page {i}" in b.text and "Journal of Tests" not in b.text and f"Page {i + 1}" not in b.text
    assert sum("Figure 1: a caption" in b.text for b in doc.blocks) == 2       # on 2 of 4 pages: not a running head
    body = tmp_path / "quoted.pdf"                     # the head quoted in the body stays; a note under it stays
    _pdf_pages(body, [[(40, "Journal of Tests"), (52, f"Special note {w}"), (300, "Journal of Tests is quoted"),
                       (320, "Journal of Tests")] for w in ("alpha", "beta", "gamma", "delta")])
    for w, b in zip(("alpha", "beta", "gamma", "delta"), parse.read(body).blocks):
        assert b.text.count("Journal of Tests") == 2 and f"Special note {w}\n" in b.text and "XX" not in b.text
    short = tmp_path / "two.pdf"
    _pdf_pages(short, [[(40, "Same head"), (300, f"Text {i}")] for i in range(2)])
    assert all("Same head" in b.text for b in parse.read(short).blocks)        # too few pages to tell


def test_a_pdf_is_limited_by_its_text_not_its_bytes(tmp_path, monkeypatch):
    pdf = tmp_path / "book.pdf"
    _pdf_pages(pdf, [[(300, "x" * 50)] for _ in range(3)])
    monkeypatch.setattr(parse, "MAX_BYTES", 10)                                # a text file's limit: not a PDF's
    assert parse.read(pdf).status == "ok"
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", 120)
    doc = parse.read(pdf)
    assert (doc.status, doc.note, doc.path) == ("too_large", "over 120 characters of text by page 3", pdf)
    total = sum(len(t) for page in __import__("pymupdf").open(pdf) for t, _ in parse._blocks(page))
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", total)                         # exactly at the limit: read
    assert parse.read(pdf).status == "ok"
    monkeypatch.setattr(parse, "MAX_TEXT_CHARS", total - 1)
    assert parse.read(pdf).status == "too_large"


def test_a_code_cell_keeps_its_text_outputs_but_not_its_images(tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "MAX_OUTPUT_CHARS", 30)
    nb = tmp_path / "run.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source": "train()", "outputs": [
        "not an output (a damaged file)",
        {"output_type": "stream", "name": "stdout", "text": ["epoch 1 loss 0.9\n", "epoch 2 loss 0.5\n"]},
        {"output_type": "execute_result", "data": {"text/plain": "accuracy 0.91", "image/png": "iVBOR..."}},
        {"output_type": "display_data", "data": {"image/png": "iVBOR..."}},
        {"output_type": "error", "ename": "ValueError", "evalue": "bad shape", "traceback": ["\x1b[31m…"]}]},
        {"cell_type": "markdown", "source": "notes", "outputs": [{"text": "not a code cell's"}]}]}))
    blocks = parse.read(nb).blocks
    assert blocks[0].text == "train()\n\nOutput:\nepoch 1 loss 0.9\nepoch 2 loss \n… [39 more characters of output]"
    assert blocks[1].text == "notes"
    whole = parse.read(nb, cut_outputs=False).blocks           # read (the tool) shows a cell's outputs whole; the
    assert whole[0].text.endswith("epoch 2 loss 0.5\naccuracy 0.91\nValueError: bad shape") and whole[1].text == "notes"
    #                                                          index keeps the cut (a training log stays out of it)
    monkeypatch.setattr(parse, "MAX_OUTPUT_CHARS", 2000)                        # each output on its own line
    assert parse.read(nb).blocks[0].text.endswith("Output:\nepoch 1 loss 0.9\nepoch 2 loss 0.5\naccuracy 0.91\n"
                                                  "ValueError: bad shape")
    cell = {"outputs": [{"output_type": "error"}, {"text": "kept  \n"}, {"text": "end"}]}
    assert parse._outputs(cell) == "Error: \nkept  \nend"                       # names missing; spaces inside kept
    monkeypatch.setattr(parse, "MAX_OUTPUT_CHARS", len("Error: \nkept  \nend"))
    assert parse._outputs(cell) == "Error: \nkept  \nend"                       # exactly at the limit: not cut


def test_a_pdf_glyph_without_a_character_is_marked_unreadable_not_passed_as_a_control_code():
    class Page:                                           # what PyMuPDF gives for a math font without a ToUnicode map
        rect = type("Rect", (), {"height": 842})()

        def get_text(self, kind):
            return [(72, 300, 200, 320, "\x00d(z)⊤q(x)\x01\tend\n", 0, 0), (72, 400, 90, 420, "<image>", 1, 1)]
    assert parse._blocks(Page()) == [("�d(z)⊤q(x)�\tend\n", False)]


def test_two_columns_are_read_one_after_the_other_not_across(tmp_path):
    """PyMuPDF's own order (the content stream's) kept the prose of 69 of 69 two-column pages of 10 arXiv papers in
    reading order (measured 2026-10-03); sorting blocks by position read every one across the columns."""
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    page = d.new_page()
    for i in range(3):
        page.insert_textbox(pymupdf.Rect(50, 100 + 150 * i, 280, 240 + 150 * i), f"left paragraph {i}")
    for i in range(3):
        page.insert_textbox(pymupdf.Rect(315, 100 + 150 * i, 545, 240 + 150 * i), f"right paragraph {i}")
    d.save(tmp_path / "two.pdf")
    text = parse.read(tmp_path / "two.pdf").blocks[0].text
    order = [text.index(f"{side} paragraph {i}") for side in ("left", "right") for i in range(3)]
    assert order == sorted(order)



@given(st.lists(paragraph, max_size=3), st.one_of(table, big_table, bullets, blank_fence), st.lists(paragraph, max_size=3),
       python)
@settings(deadline=None, max_examples=100)
def test_each_chunking_option_does_what_it_says(tmp_path_factory, before, block, after, source):
    """With `blocks` (in the default) a table, list or code block up to BLOCK_CHARS is one chunk; with `code_defs` (in
    the default) a top-level Python definition (its decorators and the comments right above it) that fits is one
    chunk. The invariants of every chunking hold too (test_chunks_lose_nothing_invent_nothing_and_say_where_they_are)."""
    tmp_path = tmp_path_factory.mktemp("o")
    whole = tmp_path / "w.md"
    whole.write_text("\n\n".join(before + [block] + after) + "\n")
    for opt in (chunk.PRESETS["blocks"], chunk.DEFAULT):
        kept = any(block.strip() in c.text for c in chunk.chunk(parse.read(whole), options=opt)[0])
        assert kept is (len(block.strip()) <= chunk.BLOCK_CHARS), (opt.name, len(block))
    code = tmp_path / "s.py"
    code.write_text(source + "\n")
    rows = source.split("\n")
    spans = [c for c in chunk.chunk(parse.read(code))[0]]
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            while first > 1 and rows[first - 2].lstrip().startswith("#"):
                first -= 1
            if len("\n".join(rows[first - 1:node.end_lineno])) <= chunk.MAX_CHARS:
                assert any(c.start <= first and node.end_lineno <= c.end for c in spans), (first, node.end_lineno)
    md = tmp_path / "n.md"
    fence = "```python\ndef f():\n    a = 1  # " + "x" * 900 + "\n\n    return a  # " + "y" * 900 + "\n```"   # 1,850
    table = "| k | v |\n|---|---|\n" + "".join(f"| row {i} | {'v' * 40} |\n" for i in range(40))
    md.write_text(f"# T\n\n{fence}\n\n{table}\nafter\n")
    base = chunk.chunk(parse.read(md), options=chunk.PRESETS["baseline"])[0]
    blocks = chunk.chunk(parse.read(md), options=chunk.PRESETS["blocks"])[0]
    assert [c.text for c in chunk.chunk(parse.read(md))[0]] == [c.text for c in
                                                               chunk.chunk(parse.read(md), options=chunk.DEFAULT)[0]]
    assert not any("def f():" in c.text and "return a" in c.text for c in base)          # cut at the blank line
    assert any("def f():" in c.text and "return a" in c.text for c in blocks)            # kept whole
    assert sum("| row" in c.text for c in base) > 1 and sum("| row" in c.text for c in blocks) == 1   # 2,200 chars
    py = tmp_path / "m.py"
    py.write_text("import os\n\n\n# helper\n@cache\ndef f(x):\n    y = x\n\n    return y\n\n\nclass A:\n    pass\n")
    defs = chunk.chunk(parse.read(py), options=chunk.Options("t", max_chars=50, code_defs=True))[0]
    assert [(c.start, c.end) for c in defs] == [(1, 1), (4, 9), (12, 13)]               # decorator and comment kept
    cjk = tmp_path / "z.txt"
    cjk.write_text("\n\n".join("机器学习" * 30 for _ in range(10)))
    small = chunk.chunk(parse.read(cjk), options=chunk.Options("t", max_tokens=150))[0]
    assert all(chunk.tokens(c.text) <= 150 for c in small) and len(small) == 10             # 120 tokens a paragraph
    assert chunk.tokens("机器学习") == 4 and chunk.tokens("abcdefgh") == 2


def test_each_size_holds_at_its_edge_and_a_cut_is_as_late_as_the_size_allows(tmp_path):
    """A paragraph of exactly the size is kept whole, one character more is cut at line ends; a line over the size
    is cut by characters, each piece as long as fits (in characters or estimated tokens); every piece keeps its
    lines; cells and pages are cut within their unit; a table up to BLOCK_CHARS stays whole with `blocks`."""
    o50, t10 = chunk.Options("c50", max_chars=50), chunk.Options("t10", max_tokens=10)
    assert chunk.tokens("a" * 16) == 4 and t10.size("a" * 40) == 10 and o50.size("a" * 40) == 40

    def cut(text, opt, name="n.txt"):
        p = tmp_path / name
        p.write_text(text)
        return [(c.start, c.end, c.text) for c in chunk.chunk(parse.read(p), options=opt)[0]]

    assert cut("a" * 50 + "\n", o50) == [(1, 1, "a" * 50)]
    assert cut("a" * 24 + "\n" + "b" * 25 + "\n", o50) == [(1, 2, "a" * 24 + "\n" + "b" * 25)]       # 50 with the break
    assert cut("a" * 24 + "\n" + "b" * 26 + "\n", o50) == [(1, 1, "a" * 24), (2, 2, "b" * 26)]
    assert cut("short\n" + "x" * 120 + "\nend\n", o50) == [(1, 1, "short"), (2, 2, "x" * 50), (2, 2, "x" * 50),
                                                            (2, 3, "x" * 20 + "\nend")]
    assert [len(t) for _, _, t in cut("a" * 1000 + "\n", t10)] == [40] * 25                  # 40 characters = 10 tokens
    nb = tmp_path / "c.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": "p" * 30 + "\n\n" + "q" * 30},
                                        {"cell_type": "markdown", "source": "ab\n" + "x" * 120}]}))
    got = [(c.unit, c.start, c.end, c.text) for c in chunk.chunk(parse.read(nb), options=o50)[0]]
    assert got == [("cell", 1, 1, "p" * 30), ("cell", 1, 2, "q" * 30 + "\n\nab"),          # cells pack like paragraphs
                   ("cell", 2, 2, "x" * 50), ("cell", 2, 2, "x" * 50), ("cell", 2, 2, "x" * 20)]
    row = "| k | " + "v" * 91 + " |"                                                       # 99 characters a row
    for size, whole in ((chunk.BLOCK_CHARS, True), (chunk.BLOCK_CHARS + 1, False)):
        table = "\n".join([row] * 59)
        table += "\n| k | " + "v" * (size - len(table) - 9) + " |"                            # exactly `size`
        assert len(table) == size
        got = cut("# T\n\n" + table + "\n", chunk.PRESETS["blocks"], "t.md")
        assert (len([t for _, _, t in got if t.startswith("|")]) == 1) is whole


def test_lists_and_python_definitions_are_cut_where_they_say(tmp_path):
    md = tmp_path / "l.md"
    items = "\n".join(f"- item {i}: " + "w" * 60 for i in range(30))                      # 2,200 characters
    md.write_text(f"# L\n\n{items}\n\n  {items.replace(chr(10), chr(10) + '  ')}\n\n{items}\nand a closing line\n")
    blocks = chunk.chunk(parse.read(md), options=chunk.PRESETS["blocks"])[0]
    lists = [c for c in blocks if "item 0" in c.text]
    assert len(lists) == 3 and all("item 29" in c.text for c in lists)                     # each list kept whole,
    assert chunk.chunk(parse.read(md), options=chunk.PRESETS["baseline"])[0][1].text.count("item") < 30   # cut before
    py = tmp_path / "m.py"
    py.write_text("\n\n# top note\ndef f():\n    pass\n\n\nimport os\n\n# a\n    # indented\n# c\ndef g():\n"
                  "    return 1\n")
    defs = chunk.chunk(parse.read(py), options=chunk.Options("t", max_chars=80, code_defs=True))[0]
    assert [(c.start, c.end, c.section) for c in defs] == [(3, 8, ""), (10, 14, "")]   # leading blanks, comments
    # code Python 3.12 parses with a SyntaxWarning (1if, a "\d" escape): cut all the same, and nothing printed (the
    # CLI's index report got "<unknown>:2: SyntaxWarning: invalid decimal literal", naming no file; seen 2026-10-07;
    # the TUI's own stderr is captured by Textual)
    py.write_text('def f():\n    return 1if True else 2  # ' + 'w' * 60 + '\n\n\ndef g():\n'
                  '    return re.match("\\d", "1")  # ' + 'w' * 60 + '\n')
    import warnings
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        defs = chunk.chunk(parse.read(py), options=chunk.Options("t", max_chars=150, code_defs=True))[0]
    assert [(c.start, c.end) for c in defs] == [(1, 2), (5, 6)] and [str(w.message) for w in said] == []


SENTENCES = {
    "en": "Gradient descent moves the weights against the gradient, one small step at a time. ",
    "vi": "Thuật toán hạ gradient cập nhật trọng số ngược hướng đạo hàm, từng bước nhỏ một. ",
    "zh": "梯度下降法沿着梯度的反方向更新权重，每次走一小步，直到损失不再下降为止。",
    "ja": "勾配降下法は勾配と逆の方向に重みを少しずつ更新し、損失が下がらなくなるまで続ける。",
    "ko": "경사 하강법은 손실이 더 이상 줄지 않을 때까지 기울기의 반대 방향으로 가중치를 조금씩 갱신한다. ",
    "hi": "ग्रेडिएंट डिसेंट भार को ग्रेडिएंट की उलटी दिशा में थोड़ा-थोड़ा बदलता है, जब तक हानि घटना बंद न हो जाए। ",
}


@pytest.mark.parametrize("opt", [chunk.Options("c200", max_chars=200), chunk.Options("t60", max_tokens=60)],
                         ids=["chars", "tokens"])
@pytest.mark.parametrize("lang", list(SENTENCES))
def test_a_paragraph_written_on_one_line_is_cut_where_a_sentence_ends(tmp_path, opt, lang):
    """Some editors keep a paragraph on one line: a line over the size is cut after the last sentence that fits,
    else at the last space, and only a run with neither (base64) by characters; each piece fits, none is shorter
    than half the size unless it is the last, and the pieces give back the line exactly."""
    def pieces(line):
        p = tmp_path / f"{lang}.md"
        p.write_text(line + "\n")
        return [c.text for c in chunk.chunk(parse.read(p), options=opt)[0]]

    line = (SENTENCES[lang] * 40).strip()
    got = pieces(line)
    ends = tuple(".!?。！？।")
    assert "".join(got) == line.replace("\n", "") or " ".join(x.strip() for x in got) == line
    assert all(opt.size(x) <= opt.cap for x in got) and len(got) > 1
    assert all(x.rstrip().endswith(ends) for x in got[:-1]), [x[-12:] for x in got]       # whole sentences
    assert all(opt.size(x) >= opt.cap // 2 for x in got[:-1])
    words = " ".join(f"word{i}" for i in range(400))                                    # no sentence end: spaces
    got = pieces(words)
    assert " ".join(x.strip() for x in got) == words and all(not x.rstrip()[-1:].isspace() for x in got)
    assert all(x.strip().split()[-1].startswith("word") and x.strip().split()[-1][4:].isdigit() for x in got)
    blob = "QUJD" * 300                                                                 # nothing to cut at
    assert "".join(pieces(blob)) == blob


END = re.compile(r"[.!?…।](?=\s)|[。！？]")


@given(st.text(alphabet="ab .!?…।。！？\t", max_size=600), st.sampled_from([chunk.Options("c200", max_chars=200),
                                                                         chunk.Options("t30", max_tokens=30)]))
@settings(deadline=None, max_examples=300)
@example("a" * 99 + ". " + "b" * 300, chunk.Options("c200", max_chars=200))     # a sentence end at exactly half
@example("a" * 29 + ". " + "b" * 300, chunk.Options("c200", max_chars=200))     # before half: by characters
@example("a" * 79 + ". " + "b" * 300, chunk.Options("c200", max_chars=200))
@example("a" * 40 + " " + "b" * 300, chunk.Options("c200", max_chars=200))      # a space before half: not used
@example("a" * 80 + " " + "b" * 300, chunk.Options("c200", max_chars=200))
@example("a" * 120 + " " + "b" * 30 + " " + "c" * 300, chunk.Options("c200", max_chars=200))   # the last space
@example("a" * 120 + ". " + "b" * 30 + ". " + "c" * 300, chunk.Options("c200", max_chars=200))  # the last end
@example("a" * 198 + ". b", chunk.Options("c200", max_chars=200))              # an end whose space is past the size
def test_a_long_line_is_cut_at_the_last_boundary_in_the_second_half_of_the_size(line, opt):
    """Where a line over the size is cut: n = what fits; after the last sentence end (one of . ! ? … । followed by a
    space within what fits, or 。！？) at or after n/2, else after the last space in [n/2, n), else at n."""
    if opt.size(line) <= opt.cap:
        return                                                  # _cut is asked only for a line over the size
    k = chunk._cut(line, opt)
    n = max(i for i in range(1, len(line) + 1) if i == 1 or opt.size(line[:i]) <= opt.cap)
    ends = [m.end() for m in END.finditer(line[:n]) if m.end() >= n // 2]
    spaces = [i + 1 for i in range(n // 2, n) if line[i] == " " and i > 0]
    assert k == (ends[-1] if ends else spaces[-1] if spaces else n)
    assert 0 < k <= n and opt.size(line[:k]) <= opt.cap


def test_a_broken_pdf_reads_without_mupdf_writing_to_the_terminal(tmp_path):
    """MuPDF prints its own complaints ("MuPDF error: syntax error: …") on the process's stdout, through the TUI's
    screen and `peresearch index`'s report (reproduced 2026-10-08): they stay out of the terminal and are counted in
    the file's note instead."""
    import subprocess
    import sys
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "hello world")
    raw = d.tobytes()
    at = raw.find(b"stream")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(raw[:at + 8] + b"BT /F9 12 Tf (x) Tj ET q 1 0 0 1 0 0 cm BI garbage EI Q" + raw[at + 8:])
    code = ("import sys; sys.path.insert(0, 'src'); from pathlib import Path; from peresearch.zetokrag import parse; "
            f"d = parse.read(Path({str(bad)!r})); print('STATUS', d.status, '|', d.note)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert r.stdout.startswith("STATUS ok") and r.stdout.count("\n") == 1 and r.stderr == "", (r.stdout, r.stderr)
    assert "MuPDF reported" in r.stdout
