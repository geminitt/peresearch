"""Contracts of the agent's tools (peresearch.tools).

Access, checked as one property over any sequence of tool calls, path spellings and user answers: a tool never
shows what lies in a hidden folder outside the declared ones, a protected file, peresearch's own data or outside the
home folder; it shows a file outside the declared folders only after the user allowed it; it never shows a
credential; it never raises. (A hidden file inside a declared folder is read when named: the folder was declared
whole; grep, glob and tree skip hidden names unless the pattern names them, as the index does.)
Then each tool's output: read's windows, grep's matches, glob's files, tree's counts, the sources and their ids,
the web gate, the error messages and the schemas. Web providers and pages have their own tests (test_web.py).
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from peresearch.llm import chat_tools
from peresearch import files as files_module, guard, tools
from peresearch.zetokrag import parse
from peresearch.tools import Sources, Toolbox
from tests.test_guard import CANARIES

SECRET = CANARIES["github-token"]
NEVER = ("MARK_DOTCONFIG", "MARK_ENV", "MARK_OWN", "MARK_OUTSIDE", "MARK_KEY", "MARK_DENIED", SECRET)


def make_world(base: Path) -> dict:
    """A home folder with a declared folder and every kind of place a tool must not, or may only with leave, show."""
    user, outside = base / "user", base / "outside"
    notes, private, own = user / "notes", user / "private", user / "pehome"
    for d in (notes / "sub", notes / ".hidden", notes / "data", private / "deep", user / ".config", own, outside):
        d.mkdir(parents=True)
    (notes / "a.md").write_text(f"# A\n\nMARK_NOTE alpha\n\ntoken for the API: {SECRET}\n")
    (notes / "sub" / "b.md").write_text("# B\n\nMARK_NOTE beta\n")
    (notes / "data" / "d.md").write_text("# D\n\nMARK_NOTE in a folder the walk skips\n")
    (notes / ".hidden" / "h.md").write_text("MARK_HIDDEN\n")
    (notes / ".env").write_text("MARK_ENV\n")
    (notes / "id_rsa").write_text("MARK_KEY\n")
    (private / "p.md").write_text("# P\n\nMARK_PRIVATE\n")
    (private / "deep" / "q.md").write_text("MARK_PRIVATE deeper\n")
    (private / ".cache").mkdir()
    (private / ".cache" / "x.md").write_text("MARK_PRIVHIDDEN\n")
    (user / ".config" / "c.md").write_text("MARK_DOTCONFIG\n")
    (own / "config.json").write_text('{"roots": [], "note": "MARK_OWN"}')
    (outside / "o.md").write_text("MARK_OUTSIDE\n")
    os.symlink(private / "p.md", notes / "link.md")
    os.symlink(outside, notes / "linkdir")
    os.symlink(user / ".config" / "c.md", private / "innocent.md")      # a link in an allowed folder, to a hidden file
    (private / "secrets.toml").write_text("MARK_DENIED\n")
    os.symlink(private / "secrets.toml", private / "settings.md")       # ... and to a protected one
    return {"user": user, "notes": notes, "private": private, "own": own, "outside": outside}


def spellings(w):
    """Ways a model may name places: relative, absolute, ~, .., through symlinks, folders and patterns."""
    u, n = w["user"], w["notes"]
    return [
        "a.md", "sub/b.md", "data/d.md", str(n / "a.md"), "~/notes/a.md", "sub/../a.md", "link.md", "~/private/innocent.md",
        "linkdir/o.md", ".env", ".hidden/h.md", "id_rsa", "sub/../.env", "../private/p.md", "~/private/p.md",
        str(w["private"] / "deep" / "q.md"), "~/private", "~/.config/c.md", str(u / ".config"), "~/pehome/config.json",
        str(w["own"]), str(w["outside"] / "o.md"), str(w["outside"]), "/etc/hostname", "~/notes/sub", "~/notes", "notes",
        "**/*", "**/*.md", "*", "~/private/*", "~/private/**/*.md", "~/.config/*", "~/pehome/*", str(w["outside"]) + "/*",
        "../**", "/etc/*", "~/private/deep", "~", "~/*", "~/**/*", "~/private/.cache/x.md", "~/private/innocent.md",
        "~/private/settings.md", "~/private/secrets.toml",
    ]


tool_calls = st.lists(st.tuples(st.sampled_from(["read", "grep", "glob", "tree"]), st.integers(0, 44)),  # 45 spellings
                      min_size=1, max_size=6)


@given(tool_calls, st.sampled_from([None, "no", "once", "session"]))
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@example([("read", 14)], "session")                                     # ~/private/p.md, allowed by the user
@example([("read", 43)], "once")             # a link outside the declared folders to a protected file: refused

def test_tools_show_only_what_they_may_whatever_the_path_and_the_answer(calls, answer):
    base = Path(tempfile.mkdtemp())
    env = {k: os.environ.get(k) for k in ("HOME", "PERESEARCH_HOME")}
    try:
        w = make_world(base)
        os.environ["HOME"], os.environ["PERESEARCH_HOME"] = str(w["user"]), str(w["own"])
        asked = []
        ask = None if answer is None else (lambda tool, path: (asked.append(Path(path)), answer)[1])
        tb = Toolbox(None, None, roots=[w["notes"]], ask=ask)
        names = spellings(w)
        for tool, i in calls:
            arg = names[i]
            args = {"read": {"path": arg}, "grep": {"pattern": "MARK|token", "glob": arg},
                    "glob": {"pattern": arg}, "tree": {"path": arg, "depth": 4}}[tool]
            out = tb.call(tool, args)
            assert isinstance(out, str)
            assert not any(m in out for m in NEVER + ("MARK_PRIVHIDDEN",)), (tool, arg, out)
            if "MARK_PRIVATE" in out:
                assert answer in ("once", "session") and asked, (tool, arg)
        for p in asked:                                                  # only ever asked about the user's own,
            real = Path(os.path.realpath(p))                             # visible, unprotected files and folders
            assert (real == w["user"] or w["user"] in real.parents) and w["own"] != real and w["own"] not in real.parents
            assert not any(part.startswith(".") for part in real.relative_to(w["user"]).parts)
    finally:
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(base, ignore_errors=True)


def test_a_link_in_an_allowed_folder_does_not_open_a_hidden_file(tmp_path, monkeypatch):
    w = make_world(tmp_path)
    monkeypatch.setenv("HOME", str(w["user"]))
    monkeypatch.setenv("PERESEARCH_HOME", str(w["own"]))
    tb = Toolbox(None, None, roots=[w["notes"]], ask=lambda tool, path: "once")
    assert "MARK_DOTCONFIG" not in tb.call("grep", {"pattern": "MARK", "glob": "~"})
    assert "MARK_DOTCONFIG" not in tb.call("read", {"path": "~/private/innocent.md"})


def test_allowing_the_whole_home_does_not_open_its_hidden_folders(tmp_path, monkeypatch):
    w = make_world(tmp_path)
    monkeypatch.setenv("HOME", str(w["user"]))
    monkeypatch.setenv("PERESEARCH_HOME", str(w["own"]))
    tb = Toolbox(None, None, roots=[w["notes"]], ask=lambda tool, path: "session")
    tb.call("tree", {"path": "~/private"})                           # the user allows ~/private for the session
    assert "MARK_PRIVHIDDEN" not in tb.call("grep", {"pattern": "MARK", "glob": "~/private/**/*"})
    tb.call("tree", {"path": "~"})                                   # and then the whole home
    assert "MARK_DOTCONFIG" not in tb.call("grep", {"pattern": "MARK", "glob": "~"})
    assert ".config" not in tb.call("glob", {"pattern": "~/**/*"})


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = make_world(tmp_path)
    monkeypatch.setenv("HOME", str(w["user"]))
    monkeypatch.setenv("PERESEARCH_HOME", str(w["own"]))
    return w


def test_leave_given_once_is_asked_again_and_for_the_session_covers_the_folder(world):
    asked, answers = [], iter(["once", "session", "no", "no", "no"])
    tb = Toolbox(None, None, roots=[world["notes"]], ask=lambda tool, p: (asked.append((tool, p)), next(answers))[1])
    assert "MARK_PRIVATE" in tb.call("read", {"path": "~/private/p.md"})
    assert "MARK_PRIVATE" in tb.call("read", {"path": "~/private/p.md"})          # asked again: session this time
    assert "deeper" in tb.call("read", {"path": "~/private/deep/q.md"})           # inside the allowed folder
    assert "q.md" in tb.call("glob", {"pattern": "~/private/**/*.md"})
    assert asked == [("read", world["private"] / "p.md")] * 2
    assert tb.granted == [world["private"]]                               # the file's folder, not its parent
    assert tb.call("tree", {"path": "~"}) == (f"refused: the user did not allow reading {world['user']}; "
                                              "do not ask for it again")
    tb.call("glob", {"pattern": "~/pehome2/*"})
    tb.call("grep", {"pattern": "x", "glob": str(world["user"]) + "/*.md"})
    assert [t for t, _ in asked] == ["read", "read", "tree", "glob", "grep"]
    assert Toolbox(None, None, roots=[world["notes"]], ask=lambda t, p: "session").call(
        "tree", {"path": "~/private/deep"}) and True
    folder = Toolbox(None, None, roots=[world["notes"]], ask=lambda t, p: "session")
    folder.call("tree", {"path": "~/private/deep"})
    assert folder.granted == [world["private"] / "deep"]                 # a folder is granted itself
    from peresearch import guard as g
    g.set_roots([world["notes"]])
    assert Toolbox(None, None).roots == [world["notes"]]                  # the config's folders by default
    assert "refused" in Toolbox(None, None, roots=[world["notes"]]).call("read", {"path": "~/private/p.md"})
    for path in (str(world["outside"] / "o.md"), "/etc/hostname"):
        assert tb.call("read", {"path": path}) == f"refused: {path}: outside your home folder"
    assert tb.call("read", {"path": "~/.config/c.md"}).endswith("hidden or protected")
    assert tb.call("read", {"path": "~/nothing.md"}) == f"refused: {world['user'] / 'nothing.md'}: not found"
    assert Toolbox(None, None, roots=[world["notes"]]).call("read", {"path": "~/private/p.md"}) == (
        f"refused: {world['private'] / 'p.md'}: outside the declared folders (tree without a path lists them); the "
        "user can allow it when asked in peresearch chat, or declare it with /add")      # a model guessing learns where
    assert tb.call("read", {"path": "nothing.md"}) == ("refused: nothing.md: not found in the declared folders (tree "
                                                       "without a path lists them)")


# --- read: windows of lines, pages or cells ---------------------------------------------------------------------

@given(st.integers(0, 260), st.integers(-3, 270), st.one_of(st.none(), st.integers(-3, 500)))
@settings(deadline=None, max_examples=80)
@example(250, 1, None)                                   # no end: to the end of the file, however long
@example(5, 9, None)                                     # starting past the end
@example(0, 1, None)                                     # an empty file
@example(3_000, 1, None)                                 # 30,000 characters: whole
def test_read_returns_the_asked_lines_and_the_whole_file_by_default(tmp_path_factory, n, start, end):
    root = tmp_path_factory.mktemp("r")
    lines = [f"line {i}" for i in range(1, n + 1)]
    (root / "f.txt").write_text("\n".join(lines))
    out = Toolbox(None, None, roots=[root]).read("f.txt", start, end)
    if n == 0:                                            # an empty file: no window, it says so
        assert out == f"{root / 'f.txt'}: holds no text"
        return
    s = max(1, start)
    e = min(n, end or n)
    assert out == f"[N1] {root / 'f.txt'}:L{s}-{e}\n" + "\n".join(lines[s - 1:e])


def test_lines_are_numbered_as_an_editor_shows_them(tmp_path):
    """A line ends at "\n" only: a form feed or U+2028 inside a line does not shift the numbers `code -g` opens."""
    (tmp_path / "f.md").write_bytes("# T\r\n\r\nfirst\x0c part\r\nsecond\u2028x\nTARGET line\n".encode())
    tb = Toolbox(None, None, roots=[tmp_path])
    assert tb.grep("TARGET") == f"[N1] {tmp_path / 'f.md'}:L5: TARGET line"
    assert tb.read("f.md", 5, 5) == f"[N2] {tmp_path / 'f.md'}:L5-5\nTARGET line"
    assert tb.read("f.md", 3, 3) == f"[N3] {tmp_path / 'f.md'}:L3-3\nfirst\x0c part"


def test_read_shows_a_notebook_cells_outputs_whole(tmp_path):
    """The index keeps 2,000 characters of a cell's outputs (parse.MAX_OUTPUT_CHARS); read shows them whole (owner,
    2026-10-08): the last line of a long training log is often the number asked for."""
    log = "".join(f"epoch {i} loss {1 / (i + 1):.4f}\n" for i in range(400))
    (tmp_path / "train.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": "fit()",
                                                                 "outputs": [{"output_type": "stream", "text": log}]}]}))
    out = Toolbox(None, None, roots=[tmp_path]).read("train.ipynb")
    assert len(log) > 2 * parse.MAX_OUTPUT_CHARS and out.endswith("Output:\n" + log.rstrip("\n"))
    assert "more characters of output" not in out


def test_read_opens_a_results_table_and_a_log(tmp_path):
    (tmp_path / "metrics.csv").write_text("step,loss\n1,0.9\n2,0.5\n")
    (tmp_path / "train.log").write_text("epoch 1 done\n")
    tb = Toolbox(None, None, roots=[tmp_path])
    assert tb.read("metrics.csv", 2, 3) == f"[N1] {tmp_path / 'metrics.csv'}:L2-3\n1,0.9\n2,0.5"
    assert tb.grep("epoch") == f"[N2] {tmp_path / 'train.log'}:L1: epoch 1 done"


def test_read_numbers_pages_and_cells_and_reports_an_unreadable_file(tmp_path):
    import json
    nb = tmp_path / "n.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source": f"c{i}"} for i in range(1, 21)]}))
    (tmp_path / "bin.txt").write_bytes(b"a\x00b")
    tb = Toolbox(None, None, roots=[tmp_path])
    assert tb.read("n.ipynb", 2, 3) == f"[N1] {nb}:cell 2-3\nc2\n\nc3"
    assert tb.read("n.ipynb", 6) == f"[N2] {nb}:cell 6-20\n" + "\n\n".join(f"c{i}" for i in range(6, 21))   # to the end
    assert tb.read("n.ipynb", 30) == f"[N3] {nb}:cell 30\n"
    assert tb.sources.items["N1"].title == "n.ipynb"
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    for i in range(3):
        d.new_page().insert_text((72, 72), f"page {i + 1}")
    d.save(tmp_path / "p.pdf")
    assert tb.read("p.pdf", 2, 3).split("\n")[0] == f"[N4] {tmp_path / 'p.pdf'}:p.2-3"
    assert tb.grep("page 3", "*.pdf") == f"[N5] {tmp_path / 'p.pdf'}:p.3: page 3"
    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    for title in ("one", "two", "three"):
        deck.slides.add_slide(deck.slide_layouts[5]).shapes.title.text = f"slide {title}"
    deck.save(tmp_path / "d.pptx")
    assert tb.read("d.pptx", 2, 2) == f"[N6] {tmp_path / 'd.pptx'}:p.2-2\nslide two"           # a slide is a page
    big = tmp_path / "big.md"
    big.write_text("x")
    from peresearch.zetokrag import parse as parse_module
    import unittest.mock as mock
    with mock.patch.object(parse_module, "MAX_BYTES", 0):
        assert tb.read("big.md") == f"{big}: cannot be read (too_large: 1 bytes)"
    assert tb.read("bin.txt") == f"{tmp_path / 'bin.txt'}: cannot be read (binary)"
    assert tb.call("read", {"path": "missing.md"}) == ("refused: missing.md: not found in the declared folders (tree "
                                                       "without a path lists them)")
    pptx.Presentation().save(tmp_path / "empty.pptx")            # documents with no text at all: said so
    (tmp_path / "empty.md").write_text("")
    (tmp_path / "blank.txt").write_text("\n  \n")
    (tmp_path / "empty.ipynb").write_text(json.dumps({"cells": []}))
    for name in ("empty.pptx", "empty.md", "blank.txt", "empty.ipynb"):
        assert tb.call("read", {"path": name}) == f"{tmp_path / name}: holds no text"
    docx = pytest.importorskip("docx")
    docx.Document().save(tmp_path / "empty.docx")
    assert tb.call("read", {"path": "empty.docx"}) == f"{tmp_path / 'empty.docx'}: holds no text"


# --- grep and glob ----------------------------------------------------------------------------------------------

LINES = ["alpha", "Beta gamma", "ALPHA beta", "", "x" * 400, "phở alpha", "  gamma  "]
files_with_lines = st.dictionaries(st.sampled_from(["a.md", "b.txt", "s/c.md", "s/t/d.py", ".h.md", "r.csv"]),
                                   st.tuples(st.lists(st.sampled_from(LINES), max_size=8),
                                             st.sampled_from(["utf-8", "crlf", "utf-16"])), min_size=1)


def expected_grep(root: Path, files: dict, pattern: str) -> str:
    """What grep must say, from what was written: every matching line, whole, in turns — each file's first match in
    walk order, then each file's second, and so on (the head the model sees first never holds one file only) — and
    the notes."""
    order = [p for p in files_module.walk([root], {}, lambda s: None, lambda: None)]
    hits = {p: [(f"{p}:L{i}", line) for i, line in enumerate(files[p.relative_to(root).as_posix()][0], 1)
                if re.search(pattern, line, re.I)] for p in order}
    turns = max((len(v) for v in hits.values()), default=0)
    out = [f"{w}: {line.strip()}" for t in range(turns) for p in order if t < len(hits[p])
           for w, line in [hits[p][t]]] or ["no match"]
    if ".h.md" in files:
        out.append("(not searched: 1 hidden; a folder named in the pattern is searched unless the user excluded it)")
    return "\n".join(out)


@given(files_with_lines, st.sampled_from(["alpha", "beta", "^x+$", "gam+a$", "phở", "(?<=al)pha"]))
@settings(deadline=None, max_examples=60)
def test_grep_shows_every_match_in_turns_across_files_by_either_engine(tmp_path_factory, files, pattern):
    """ripgrep and the converter give the same lines at the same places, whatever the line ends or the encoding."""
    root = tmp_path_factory.mktemp("g")
    for name, (lines, how) in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        text = ("\r\n" if how == "crlf" else "\n").join(lines)
        (root / name).write_bytes(text.encode("utf-16") if how == "utf-16" else text.encode())
    want = expected_grep(root, files, pattern)
    for engine in (tools.RG, None):
        old, tools.RG = tools.RG, engine
        try:
            out = Toolbox(None, None, roots=[root]).grep(pattern)
        finally:
            tools.RG = old
        assert re.sub(r"(?m)^\[N\d+\] ", "", out) == want, engine


def _converted_world(root: Path):
    import json
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    for text in ("intro page", "the alpha result\nsecond line", "end"):
        d.new_page().insert_text((72, 300), text)
    d.save(root / "paper.pdf")
    (root / "run.ipynb").write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": "setup"},
                                                         {"cell_type": "code", "source": "alpha = 1"}]}))
    (root / "old.txt").write_bytes("first\nsecond\nthird alpha\n".encode("utf-16"))
    (root / "plain.md").write_text("# P\n\nan alpha note\n")
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Notes", level=1)
    d.add_paragraph("an alpha paragraph")
    d.save(root / "notes.docx")
    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    deck.slides.add_slide(deck.slide_layouts[5]).shapes.title.text = "Intro"
    deck.slides.add_slide(deck.slide_layouts[5]).shapes.title.text = "alpha slide"
    deck.save(root / "talk.pptx")
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    for row in (["model", "score"], ["alpha", 1]):
        wb.active.append(row)
    wb.save(root / "scores.xlsx")                                       # data: not indexed, converted on demand


def test_grep_reads_the_indexs_converted_texts_and_converts_only_what_the_index_has_not_seen(home, tmp_path, monkeypatch):
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    root = tmp_path / "n"
    root.mkdir()
    _converted_world(root)
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    (root / "late.txt").write_bytes("late alpha\n".encode("utf-16"))               # not indexed yet
    converted = []
    real = tools.parse.read
    monkeypatch.setattr(tools.parse, "read", lambda path: (converted.append(Path(path).name), real(path))[1])
    tb = Toolbox(Searcher(idx, reranker=FakeReranker()), None, roots=[root])
    out = re.sub(r"(?m)^\[N\d+\] ", "", tb.grep("alpha"))
    assert out.splitlines() == [f"{root / 'late.txt'}:L1: late alpha", f"{root / 'notes.docx'}:L3: an alpha paragraph",
                                f"{root / 'old.txt'}:L3: third alpha", f"{root / 'paper.pdf'}:p.2: the alpha result",
                                f"{root / 'plain.md'}:L3: an alpha note", f"{root / 'run.ipynb'}:cell 2: alpha = 1",
                                f"{root / 'scores.xlsx'}:L3: alpha\t1", f"{root / 'talk.pptx'}:p.2: alpha slide"]
    assert sorted(converted) == ["late.txt", "scores.xlsx"]             # the rest came from the store or as it is
    monkeypatch.setattr(tools, "RG", None)
    assert re.sub(r"(?m)^\[N\d+\] ", "", Toolbox(None, None, roots=[root]).grep("alpha")) == out


def test_grep_says_what_it_did_not_show_or_search(tmp_path, monkeypatch):
    for f in ("a", "b"):
        (tmp_path / f"{f}.md").write_text("\n".join(f"hit {i}" for i in range(20)))
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "x.md").write_text("hit in a package\n")
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "d.md").write_text("hit in a draft\n")
    guard.set_roots([tmp_path])
    cfg = guard.config()
    cfg["exclude"] = ["drafts/"]
    guard.config_path().write_text(__import__("json").dumps(cfg))
    tb = Toolbox(None, None, roots=[tmp_path])
    lines = tb.grep("hit").splitlines()
    assert [x.split(": ")[1] for x in lines[:4]] == ["hit 0", "hit 0", "hit 1", "hit 1"]             # in turn
    assert len(lines) == 41 and sum("a.md" in x for x in lines) == sum("b.md" in x for x in lines) == 20   # all
    left = lines[40]
    assert left == ("(not searched: 1 made by tools or peresearch's own, 1 excluded by the user; a folder named "
                        "in the pattern is searched unless the user excluded it)")
    assert "hit in a package" in tb.grep("hit", "node_modules/**")          # named: searched
    assert tb.grep("hit", "drafts/**").startswith("no match")                # the user's exclude still wins
    monkeypatch.setattr(tools, "GREP_SECONDS", 0)
    assert tb.grep("hit").endswith("(stopped after 0 s: some files were not searched)\n" + left)
    monkeypatch.undo()
    locked = tmp_path / "locked.md"                     # a file the user may not read: said, the rest still searched
    locked.write_text("hit locked\n")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("files cannot be made unreadable here")
        out = tb.grep("hit")
        assert out.splitlines()[:5] == lines[:5] and "(1 file could not be read: no permission)" in out
        assert "hit locked" not in out and not out.startswith("grep failed")
        assert "(1 file could not be read" in Toolbox(None, None, roots=[tmp_path]).grep("hit", "locked.md")
        (tmp_path / "locked2.md").write_text("hit locked 2\n")
        (tmp_path / "locked2.md").chmod(0)
        (tmp_path / "zz.md").write_text("hit after them\n")       # files after the unreadable ones: still searched
        (tmp_path / "0.xyz").write_text("hit, a type grep skips\n")   # and after one of a type it does not read
        out = tb.grep("hit")
        assert "(2 files could not be read: no permission)" in out and "hit after them" in out
        real_read_bytes = Path.read_bytes

        def flaky(self):                                    # closed to us between the check and the read
            if self.name == "a.md":
                raise PermissionError(13, "Permission denied")
            return real_read_bytes(self)
        monkeypatch.setattr(Path, "read_bytes", flaky)
        out = tb.grep("hit")
        assert "a.md" not in out and "hit after them" in out and "b.md:L1" in out
    finally:
        locked.chmod(0o644)
        if (tmp_path / "locked2.md").exists():
            (tmp_path / "locked2.md").chmod(0o644)


@given(st.integers(tools.LINE_WHOLE + 1, 150_000), st.floats(0, 1), st.integers(-3_000, 3_000))
@settings(deadline=None, max_examples=80)
@example(118_000, 0.7, 40)                       # the owner's folders: matching lines of ~118,000 characters
@example(5_000, 0.5, -tools.LINE_WINDOW // 2 - 20)   # a secret across the window's edge
def test_a_long_matching_line_is_shown_around_its_match(tmp_path_factory, length, where, offset):
    """grep locates; read reads (Claude Code, opencode and pi-coding-agent all cut long matching lines, 2026-10-07): a
    line up to LINE_WHOLE characters is shown whole; a longer one as LINE_WINDOW characters around its first match —
    not its head, which may not hold the match —, saying its length and that read shows it whole. A secret near the
    window is redacted, never shown in part; the source holds what is shown."""
    root = tmp_path_factory.mktemp("long")
    at = int(where * (length - 6))
    key, quoted = at + offset, f'"{SECRET}"'         # delimited, as in data: a token runs on through letters after it
    assume(key + len(quoted) < at or key > at + 6)                    # beside the match, not across it
    assume(0 <= key <= length - len(quoted))
    line = list("a" * length)
    line[key:key + len(quoted)] = quoted
    line[at:at + 6] = "needle"
    (root / "data.json").write_text("".join(line) + "\n")
    tb = Toolbox(None, None, roots=[root])
    out = tb.grep("needle").split("\n")[0]
    m = re.fullmatch(r"\[(N\d+)\] (.+?:L1): (…?)(.*?)(…?) \[a line of ([\d,]+) characters, shown around the match: "
                     r"read (.+?) for all of it\]", out)
    assert m, out[:200]
    sid, place, before, shown, after, total, again = m.groups()
    assert "needle" in shown and len(shown) == tools.LINE_WINDOW and int(total.replace(",", "")) == length   # full
    if before and after:
        assert shown.find("needle") == tools.LINE_WINDOW // 2                   # centred on it, secrets redacted
    redacted = guard.redact("".join(line))[0]
    assert again == place and tb.sources.items[sid].text == shown
    assert bool(before) == (not redacted.startswith(shown)) and bool(after) == (not redacted.endswith(shown))
    assert not any(SECRET[i:i + 8] in out for i in range(len(SECRET) - 7))      # whole or nothing of a secret


def test_a_long_line_cut_short_by_redaction_still_says_it_goes_on(tmp_path):
    """A token with no end (letters run on after it) is redacted to the end of the part looked at: the line still
    goes on past it, and the window says so."""
    (tmp_path / "d.json").write_text("x" * 3_000 + " needle " + SECRET + "a" * 10_000 + "\n")
    out = Toolbox(None, None, roots=[tmp_path]).grep("needle")
    assert "[REDACTED:github-token] [a line of" not in out and "[REDACTED:github-token]… [a line of 13," in out
    assert SECRET[:8] not in out


def test_grep_keeps_to_its_time_while_it_builds_the_result(tmp_path, monkeypatch):
    """GREP_SECONDS bounds the whole call, the result built from the matches included (redacting and numbering
    1,522,222 lines ran past ten minutes on the owner's folders, 2026-10-07): what was shown and what was found are
    said; the matches shown are the first of every file in turn."""
    for f in ("a", "b"):
        (tmp_path / f"{f}.md").write_text("\n".join(f"hit {i}" for i in range(20)))
    clock = NS(t=0.0)
    real = guard.redact

    def slow(text):                                         # each line redacted takes a second
        clock.t += 1
        return real(text)
    monkeypatch.setattr(tools, "time", NS(monotonic=lambda: clock.t))
    monkeypatch.setattr(guard, "redact", slow)
    monkeypatch.setattr(tools, "GREP_SECONDS", 10)
    out = Toolbox(None, None, roots=[tmp_path]).grep("hit").split("\n")
    shown = [x for x in out if x.startswith("[N")]
    assert len(shown) == 10 and [x.split(": ")[1] for x in shown[:4]] == ["hit 0", "hit 0", "hit 1", "hit 1"]
    assert out[10:] == ["(stopped after 10 s: 10 of 40 matching lines shown; narrow the pattern or the glob for the "
                        "rest)"]


def test_a_pattern_only_python_reads_is_searched_by_the_converter(tmp_path):
    (tmp_path / "a.md").write_text("alpha\n")
    assert Toolbox(None, None, roots=[tmp_path]).grep(r"\N{LATIN SMALL LETTER A}lpha") == f"[N1] {tmp_path / 'a.md'}:L1: alpha"


def test_grep_names_pages_and_cells_redacts_and_explains_a_bad_pattern(tmp_path):
    import json
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": "x = 1"},
                                                             {"cell_type": "code", "source": f"y = 2\nkey = '{SECRET}'"}]}))
    tb = Toolbox(None, None, roots=[tmp_path])
    out = tb.grep("key")
    assert out.startswith(f"[N1] {tmp_path / 'n.ipynb'}:cell 2: key = ") and SECRET not in out and "[REDACTED" in out
    assert tb.sources.items["N1"].title == "n.ipynb" and tb.grep("nothing at all") == "no match"
    assert tb.grep("(").startswith("invalid regular expression")


@pytest.mark.parametrize("pattern, want, note", [
    ("*.md", ["a.md", "s/c.md"], True),               # no folder in the pattern: any depth, by name; .h.md hidden
    ("**/*.md", ["a.md", "s/c.md"], True),            # **/ also matches no folder at all
    ("s/*", ["s/c.md", "s/t/d.py"], True),            # fnmatch's * crosses folders
    ("s/t/*.py", ["s/t/d.py"], True),
    ("*.txt", ["b.txt"], True),
    ("c.md", ["s/c.md"], True),                       # a bare name: found at any depth
    ("t/d.py", [], True),                             # with a folder: from the root only
    ("nothing*", [], True),
    (".h.md", [".h.md"], True),                       # a hidden file asked for by name
    ("s/.git/*", ["s/.git/x.md"], True),              # a tool's folder asked for by name (.h.md still hidden)
])
def test_glob_lists_the_files_its_pattern_names_and_what_it_left_out(tmp_path, pattern, want, note):
    for name in ("a.md", "b.txt", "s/c.md", "s/t/d.py", ".h.md", "s/.git/x.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    out = Toolbox(None, None, roots=[tmp_path]).glob(pattern).splitlines()
    left = [x for x in out if x.startswith("(not listed")]
    assert [x for x in out if not x.startswith("(")] == ([str(tmp_path / w) for w in want] or ["no file matches"])
    assert bool(left) == note
    many = Toolbox(None, None, roots=[tmp_path])
    for i in range(300):
        (tmp_path / "many" / f"{i:03d}.md").parent.mkdir(exist_ok=True)
        (tmp_path / "many" / f"{i:03d}.md").write_text("x")
    listed = [x for x in many.glob("many/*.md").splitlines() if not x.startswith("(")]
    assert listed == [str(tmp_path / "many" / f"{i:03d}.md") for i in range(300)]                          # all


# --- tree ---------------------------------------------------------------------------------------------------------

tree_names = st.sampled_from(["a", "b", ".h", "runs", "node_modules", "z.md", "y.txt", ".env", "k.pem"])
trees = st.recursive(st.just("file"), lambda kids: st.dictionaries(tree_names, kids, max_size=5), max_leaves=25)


def build(root: Path, spec):
    for name, sub in spec.items():
        if sub == "file":
            (root / (name if "." in name[1:] else name + ".md")).write_text("x")
        else:
            (root / name).mkdir(exist_ok=True)
            build(root / name, sub)


@given(st.dictionaries(tree_names, trees, max_size=6), st.integers(1, 4))
@settings(deadline=None, max_examples=80)
def test_tree_counts_what_it_shows_down_to_its_depth(tmp_path_factory, spec, depth):
    root = tmp_path_factory.mktemp("t")
    build(root, spec)

    def visible(folder):
        out = []
        for e in sorted(os.scandir(folder), key=lambda e: (not e.is_dir(), e.name.lower())):
            if e.name.startswith(".") or guard.denied(Path(e.path)):
                continue
            out.append(e)
        return out

    def expect(folder, level, lines):
        for e in visible(folder):
            indent = "  " * level
            if not e.is_dir():
                lines.append(indent + e.name)
            elif e.name == "node_modules":                        # made by a tool; runs/, a person's name, is opened
                lines.append(f"{indent}node_modules/ (not searched: node_modules: made by a package manager)")
            elif level >= depth:
                lines.append(f"{indent}{e.name}/ …")
            else:
                lines.append(f"{indent}{e.name}/ {count(visible(e.path))}")
                expect(e.path, level + 1, lines)
        return lines

    def count(entries):
        d = sum(e.is_dir() for e in entries)
        f = len(entries) - d
        return f"({d} folder{'s' * (d != 1)}, {f} file{'s' * (f != 1)})"
    tb = Toolbox(None, None, roots=[root])
    out = tb.tree(depth=depth)
    text = "\n".join([f"{root}/ {count(visible(root))}"] + expect(root, 1, []))
    assert out == f"[N1] listing of {root}\n" + text
    assert (tb.sources.items["N1"].where, tb.sources.items["N1"].title, tb.sources.items["N1"].text) == \
        (f"{root} (listing)", "folder listing", text)


def test_tree_shows_every_entry_of_a_folder(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "only.md").write_text("x")
    assert f"{tmp_path / 'one'}/ (0 folders, 1 file)" in Toolbox(None, None, roots=[tmp_path / "one"]).tree()
    shutil.rmtree(tmp_path / "one")
    for i in range(120):
        (tmp_path / f"f{i:03d}.md").write_text("x")
    out = Toolbox(None, None, roots=[tmp_path]).tree().split("\n")
    assert out[1:] == [f"{tmp_path}/ (0 folders, 120 files)"] + [f"  f{i:03d}.md" for i in range(120)]
    assert Toolbox(None, None, roots=[]).tree() == "no folders are declared"
    assert Toolbox(None, None, roots=[tmp_path]).tree(str(tmp_path / "f00.md")) == f"{tmp_path / 'f00.md'}: not a folder"
    assert Toolbox._count([NS(is_dir=lambda follow_symlinks: True)]) == "(1 folder, 0 files)"


# --- sources and their ids ----------------------------------------------------------------------------------------

@given(st.lists(st.tuples(st.sampled_from(["file", "web"]), st.sampled_from(["p:L1", "p:L2", "https://u", "https://v"]),
                          st.text(max_size=5)), max_size=12))
def test_a_source_keeps_its_id_for_the_same_immutable_excerpt_version(adds):
    s = Sources()
    first, retained, urls = {}, {}, {}
    for kind, where, text in adds:
        got = s.add(kind, where, "t", text)
        key = (kind, where, text)
        first.setdefault(key, len([k for k, _, _ in first if k == kind]) + 1)
        assert got.id == ("N" if kind == "file" else "W") + str(first[key]) and got.text == text
        retained.setdefault(got.id, (got, text))
        assert all(source.text == original for source, original in retained.values())
        if kind == "web":
            urls[where] = got.id
    assert s.urls == urls


def test_adding_a_source_does_not_grow_with_the_sources_held():
    """A grep over a large corpus adds a source per matching line (1,522,222 lines for "e" on the owner's folders,
    2026-10-07): adding one must not scan those held, or ten times the lines take a hundred times as long."""
    import time

    def seconds(n):
        s, t0 = tools.Sources(), time.perf_counter()
        for i in range(n):
            s.add("file", f"p:L{i}", "t", "x")
            s.add("file", f"p:L{i // 2}", "t", "xy")                    # a place seen again: the longer text kept
        return time.perf_counter() - t0
    small, large = min(seconds(4_000) for _ in range(3)), min(seconds(40_000) for _ in range(3))
    assert large / small < 30, (small, large)                           # linear: about 10; scanning: about 100


# --- search_notes, the web gate, errors and schemas ----------------------------------------------------------------

class FakeSearcher:
    def __init__(self, hits, stale=0):
        self.hits, self.stale, self.asked = hits, stale, []

    def find(self, query, k=None, page=1):
        self.asked.append((query, k, page))
        return (self.hits if k is None else self.hits[(page - 1) * k:page * k]), self.stale, len(self.hits)


def hit(where, text, section="", before=None, after=None, score=0.5, span=None, copies=()):
    return NS(where=where, text=text, section=section, before=before, after=after, score=score, span=span or where,
              copies=list(copies))


def test_search_notes_shows_every_ranked_excerpt_with_ids_context_and_scores():
    tb = Toolbox(FakeSearcher([hit("a.md:L1-3", "first", "S", after=NS(start=5, end=6, text="next door"),
                                   span="a.md:L1-6", copies=["a.template.md:L1-6"]), hit("b.md:L5", f"key {SECRET}", score=0.03)],
                              stale=2), None, roots=[])
    out = tb.search_notes("q")
    assert out.split("\n\n") == [
        "2 ranked candidates, best first (relevance: the reranker's score from 0 to 1, for ordering; a low score "
        "does not show the files lack the answer)",
        "[N1] a.md:L1-6 (relevance 0.50) § S; the match: L1-3\nfirst\n(after) next door\n(also in: a.template.md:L1-6)",
        "[N2] b.md:L5 (relevance 0.03)\nkey [REDACTED:github-token]",
        "(2 excerpts dropped: their files changed since indexing)"]
    assert tb.call("search_notes", {"query": "q", "k": 1, "page": 2}) == tb.search_notes("q")   # old arguments:
    assert {"k", "page"}.isdisjoint(tb.SPECS["search_notes"][1])                                 # dropped
    assert tb.searcher.asked[-1] == ("q", None, 1)                                               # every candidate
    assert Toolbox(FakeSearcher([]), None, roots=[]).search_notes("q") == "no candidates: nothing in the index matched"
    big = Toolbox(FakeSearcher([hit(f"x{i}.md:L1", "x", copies=["c1.md:L1"]) for i in range(40)]), None, roots=[])
    out = big.search_notes("q")
    assert out.startswith("40 ranked candidates") and out.count("(also in: c1.md:L1)") == 40 and "[N40] x39.md" in out
    assert tb.sources.items["N1"].title == "S" and tb.sources.items["N2"].text == "key [REDACTED:github-token]"
    assert tb.sources.items["N1"].where == "a.md:L1-6" and tb.sources.items["N1"].text == "first\nnext door"
    long = Toolbox(FakeSearcher([hit("c.md:L1", "z" * 2000)]), None, roots=[]).search_notes("q")
    assert long.endswith("\n" + "z" * 2000)                                                     # whole
    # a machine set to rerank fewer than all: the rest carry no relevance, and the model is told why
    searcher = FakeSearcher([hit("a.md:L1", "a", score=0.7), hit("b.md:L2", "b", score=None)])
    searcher.depth = 1
    out = Toolbox(searcher, None, roots=[]).search_notes("q").split("\n\n")
    assert out[1] == "[N1] a.md:L1 (relevance 0.70)\na" and out[2] == "[N2] b.md:L2 (not reranked)\nb"
    assert out[3] == ("(an excerpt marked 'not reranked' keeps the search's order: this machine reranks only the "
                      "first 1 candidates, PERESEARCH_RERANK_DEPTH)")
    assert "not reranked" not in Toolbox(FakeSearcher([hit("a.md:L1", "a")]), None, roots=[]).search_notes("q")
    keyed = Toolbox(FakeSearcher([hit("k.md:L1", "k", section=f"S {SECRET}")]), None, roots=[])  # a heading the index
    assert SECRET not in keyed.search_notes("q") and SECRET not in keyed.sources.items["N1"].title  # kept: redacted


def test_search_notes_says_how_it_ranked():
    """No rerank endpoint: no relevance score, said once, the excerpts in the search's order; ranked by words alone
    (no embedding endpoint, vectors not made yet, the endpoint down): said why, so neither the model nor the user
    takes a miss for an absence."""
    plain = FakeSearcher([hit("a.md:L1", "a", score=None), hit("b.md:L2", "b", score=None)])
    plain.reranker, plain.depth, plain.words_only, plain.unranked = None, 0, "no embedding endpoint is set", ""
    out = Toolbox(plain, None, roots=[]).search_notes("q").split("\n\n")
    assert out[0] == "2 ranked candidates, best first (in the search's order: no rerank endpoint is set, so no relevance score)"
    assert out[1] == "[N1] a.md:L1\na" and out[2] == "[N2] b.md:L2\nb"
    assert out[3] == ("(ranked by the passages' words alone: no embedding endpoint is set; a search by meaning finds "
                      "what is said in other words or another language)")
    from peresearch.zetokrag.search import Document
    docs = NS(index=None, reranker=None, depth=0, words_only="", unranked="",
              documents=lambda q, f=None: [Document("/n/a.md", "", "line", 1, 1, "a", "x", None, 2)])
    lister = Toolbox(docs, None, roots=[])
    lister._picture = lambda path: ("", "")
    lister.searcher.index = NS(coverage=dict, read=lambda *a: [])
    listed = lister.overview("q").split("\n\n")
    assert listed[0].endswith("(in the search's order: no rerank endpoint is set, so no relevance score)")
    assert listed[1].splitlines()[0].endswith("/n/a.md (2 matching passages)")             # no score, no "not reranked"
    down = FakeSearcher([hit("a.md:L1", "a", score=None)])
    down.reranker, down.depth, down.words_only, down.unranked = object(), 30, "", "the rerank endpoint did not answer (x)"
    said = Toolbox(down, None, roots=[]).search_notes("q")
    assert "(not reranked: the rerank endpoint did not answer (x))" in said and "this machine reranks only" not in said


def test_a_long_hit_keeps_the_context_next_to_it_in_reading_order():
    """The hit and each neighbour are shown whole, in reading order."""
    before = NS(start=1, end=9, text="old start " + "b" * 990 + " END OF BEFORE")
    after = NS(start=20, end=29, text="START OF AFTER " + "a" * 990 + " old end")
    tb = Toolbox(FakeSearcher([hit("d.md:L10-19", "h" * 1500, "S", before, after, span="d.md:L1-29")]), None, roots=[])
    out = tb.search_notes("q").split("\n\n")[1]
    head, b, h, a = out.split("\n")
    assert head == "[N1] d.md:L1-29 (relevance 0.50) § S; the match: L10-19"
    assert b == "(before) " + before.text and h == "h" * 1500 and a == "(after) " + after.text


class FakeWeb:
    """The web as Toolbox sees it: a search server set (`searcher`), its text and links, pages to fetch."""
    searcher = "fake"

    def __init__(self):
        self.searched, self.pages = [], {"https://u/1": "page text", "https://u/2": ""}

    def search(self, query, n):
        from peresearch.web import Found, links
        self.searched.append(query)
        text = "\n\n".join(["Title: T1\nURL: https://u/1\ns1", "Title: T2\nURL: https://u/2\ns2"][:n])
        return Found("fake", text, links(text))

    def fetch(self, url):
        return self.pages[url]


class FakePapers:
    def __init__(self, found):
        self.found, self.sources, self.asked, self.missed = found, ["one"], [], []

    def search(self, query, n):
        self.asked.append((query, n))
        return self.found[:n]


def paper(**kw):
    from peresearch.papers import Paper
    base = dict(title="Attention Is All You Need", url="https://doi.org/10.5555/1", pdf="https://arxiv.org/pdf/1706.03762",
                year=2017, venue="NeurIPS", kind="conference paper", citations=120000, retracted=False, doi="10.5555/1",
                arxiv="1706.03762", authors=["A. Vaswani", "N. Shazeer", "N. Parmar", "J. Uszkoreit"],
                abstract="The dominant sequence transduction models.", found_by=["semantic scholar", "arxiv"])
    return Paper(**{**base, **kw})


def test_papers_are_searched_after_gaps_with_what_kind_of_record_each_is_and_their_pdf_can_be_read():
    web = FakeWeb()
    web.pages["https://arxiv.org/pdf/1706.03762"] = "[p. 1]\nfull text"
    found = [paper(), paper(title="Withdrawn", url="https://doi.org/10.9/w", pdf=None, kind="journal article",
                            venue="J. Results", retracted=True, citations=40, authors=["C. Author"], abstract="x" * 700,
                            year=2019, found_by=["openalex"]),
             paper(title="Fresh", url="https://arxiv.org/abs/2601.00002v1", pdf=None, kind="preprint; published as ICML 2026",
                   venue="ICML 2026", citations=None, authors=[], abstract="", year=None, found_by=["arxiv"])]
    tb = Toolbox(None, web, roots=[], papers=FakePapers(found))
    assert tb.call("search_papers", {"query": "attention"}).startswith("refused: first record with gaps")
    tb.call("gaps", {"have": [], "missing": "x"})
    out = tb.call("search_papers", {"query": "attention", "n": 3})
    assert out.split("\n\n") == [
        "[W1] Attention Is All You Need (2017)\nconference paper · NeurIPS · 120,000 citations · found by semantic "
        "scholar, arxiv\nA. Vaswani, N. Shazeer, N. Parmar et al.\nhttps://doi.org/10.5555/1\n"
        "PDF: https://arxiv.org/pdf/1706.03762\nThe dominant sequence transduction models.",
        "[W2] Withdrawn (2019)\nRETRACTED · journal article · J. Results · 40 citations · found by openalex\nC. Author\n"
        "https://doi.org/10.9/w\n" + "x" * 700,                                       # the abstract whole
        "[W3] Fresh\npreprint; published as ICML 2026 · found by arxiv\nhttps://arxiv.org/abs/2601.00002v1"]
    assert tb.web_calls == 1                                            # one search of the question's budget
    assert tb.call("fetch", {"url": "https://arxiv.org/pdf/1706.03762"}) == "[W4] https://arxiv.org/pdf/1706.03762\n[p. 1]\nfull text"
    assert tb.sources.items["W4"].where == "https://arxiv.org/pdf/1706.03762" and tb.sources.items["W4"].text == "[p. 1]\nfull text"
    assert tb.sources.items["W1"].text == "The dominant sequence transduction models."
    tb.papers.found = []
    assert tb.call("search_papers", {"query": "nothing"}) == "no papers found"
    tb.papers.found, tb.papers.missed = found[:1], ["semantic scholar (429, throttled)"]
    assert tb.call("search_papers", {"query": "again"}).endswith(
        "\n\n(did not answer: semantic scholar (429, throttled); results may be missing)")


class FakeWiki:
    def __init__(self, found, missed=()):
        self.found, self.missed, self.asked = found, list(missed), []

    def search(self, query, langs, n):
        self.asked.append((query, langs, n))
        return self.found[:n], self.missed


def test_wikipedia_is_searched_after_gaps_with_each_articles_titles_in_other_languages():
    from peresearch.wiki import Article
    web = FakeWeb()
    web.pages["https://vi.wikipedia.org/wiki/Học_máy"] = "full article"
    found = [Article("Học máy", "https://vi.wikipedia.org/wiki/Học_máy", "vi", "Học máy là " + "x" * 700,
                     {"en": "Machine learning", "ja": "機械学習"}),
             Article("Học sâu", "https://vi.wikipedia.org/wiki/Học_sâu", "vi", "", {})]
    tb = Toolbox(None, web, roots=[], wiki=FakeWiki(found, ["en.wikipedia (503)"]))
    assert tb.call("search_wikipedia", {"query": "học máy"}).startswith("refused: first record with gaps")
    tb.call("gaps", {"have": [], "missing": "x"})
    out = tb.call("search_wikipedia", {"query": "học máy", "languages": ["vi", "en"], "n": 2})
    assert out.split("\n\n") == [
        "[W1] Học máy (vi.wikipedia: an encyclopedia, a starting point, not primary evidence)\n"
        "https://vi.wikipedia.org/wiki/Học_máy\nHọc máy là " + "x" * 700 + "\n"                   # whole
        "in other languages: en: Machine learning · ja: 機械学習",
        "[W2] Học sâu (vi.wikipedia: an encyclopedia, a starting point, not primary evidence)\n"
        "https://vi.wikipedia.org/wiki/Học_sâu",
        "(did not answer: en.wikipedia (503); results may be missing)"]
    assert tb.wiki.asked == [("học máy", ["vi", "en"], 2)] and tb.web_calls == 1
    assert tb.call("fetch", {"url": "https://vi.wikipedia.org/wiki/Học_máy"}).endswith("\nfull article")
    tb.call("search_wikipedia", {"query": "y"})
    assert tb.wiki.asked[-1] == ("y", ["en"], 3)                         # English by default
    tb.wiki.found, tb.wiki.missed = [], ["ja.wikipedia (503)", "ko.wikipedia (ConnectError)"]
    assert tb.call("search_wikipedia", {"query": "z", "languages": "vi"}) == (
        "no articles found\n\n(did not answer: ja.wikipedia (503), ko.wikipedia (ConnectError); results may be missing)")
    assert tb.wiki.asked[-1][1] == ["vi"] and tb.web_calls == 3          # a single code as a string; three searches
    tb.call("search_wikipedia", {"query": "w", "languages": []})
    assert tb.wiki.asked[-1][1] == ["en"]
    assert (tb.sources.items["W1"].title, tb.sources.items["W1"].text) == ("Học máy", "Học máy là " + "x" * 700)  # longest
    assert (tb.sources.items["W2"].title, tb.sources.items["W2"].text) == ("Học sâu", "Học sâu")   # no summary


def test_a_long_page_is_returned_whole():
    """A paper fetched runs to tens of thousands of characters: fetch returns it whole (the agent shows the model
    what fits its context and keeps the rest to read on); the source keeps the whole text for the quote check."""
    web = FakeWeb()
    page = "\n".join(f"line {i:05d} " + "w" * 40 for i in range(2_000))       # 100,000 characters
    web.pages["https://u/1"] = page
    tb = Toolbox(None, web, roots=[])
    tb.call("gaps", {"have": [], "missing": "x"})
    tb.call("web_search", {"query": "x"})
    assert tb.call("fetch", {"url": "https://u/1"}) == "[W3] https://u/1\n" + page
    assert tb.call("fetch", {"url": "https://u/1", "part": 2}) == "[W3] https://u/1\n" + page   # an old argument: dropped
    assert tb.sources.items["W3"].text == page
    assert tb.sources.items["W3"].title == tb.sources.items["W1"].title
    assert tb.sources.items["W1"].text == "Title: T1\nURL: https://u/1\ns1"
    assert "part" not in tb.SPECS["fetch"][1]


def test_the_web_is_searched_after_gaps_and_only_returned_pages_are_read():
    web = FakeWeb()
    tb = Toolbox(None, web, roots=[])
    assert tb.call("web_search", {"query": "x"}) == ("refused: first record with gaps(have, missing) what the user's "
                                                     "files already cover and what is missing; then search the web for "
                                                     "what is missing") and web.searched == []
    assert tb.call("gaps", {"have": "one point", "missing": ["a", "b"]}) == \
        ("recorded: 1 point(s) the user's files cover, 2 missing or not enough; web_search, search_papers and "
         "search_wikipedia may now look for what is missing")
    assert tb.recorded_gaps == {"have": ["one point"], "missing": ["a", "b"]}
    assert tb.call("web_search", {"query": "x", "n": 1}) == (
        "(web search: fake)\n\nTitle: T1\nURL: https://u/1\ns1\n\nsources (cite their ids; fetch reads a page):\n"
        "[W1] T1 — https://u/1 (web page)")                     # the server's text as it came, then its links
    assert tb.web_calls == 1
    assert tb.sources.items["W1"].title == "T1" and web.searched == ["x"]
    assert tb.sources.items["W1"].text == "Title: T1\nURL: https://u/1\ns1"                 # its paragraph
    assert tb.call("fetch", {"url": "https://u/2"}).startswith("refused: fetch only opens URLs")
    tb.call("web_search", {"query": "y"})
    assert tb.web_calls == 2 and web.searched == ["x", "y"]
    assert tb.call("fetch", {"url": "https://u/1"}) == "[W3] https://u/1\npage text"
    assert tb.sources.items["W3"].text == "page text"
    assert tb.sources.items["W1"].text == "Title: T1\nURL: https://u/1\ns1"
    assert tb.call("fetch", {"url": "https://u/2"}) == "[W2] https://u/2\nthe page could not be read"
    assert tb.sources.items["W2"].text == "Title: T2\nURL: https://u/2\ns2"   # an empty page keeps its paragraph
    from peresearch.web import Found
    web.search = lambda query, n: Found("fake", "nothing found for that", [])
    assert tb.call("web_search", {"query": "z"}) == "(web search: fake)\n\nnothing found for that\n\n(no link in it)"
    web.search = lambda query, n: Found("fake", "  ", [])
    assert tb.call("web_search", {"query": "z"}) == "(web search: fake)\n\nno results"
    tb.call("gaps", {"have": ["a", "b"], "missing": "c"})
    assert tb.recorded_gaps == {"have": ["a", "b"], "missing": ["c"]}
    tb.call("gaps", {"have": None, "missing": None})
    assert tb.recorded_gaps == {"have": [], "missing": []}


def test_a_tool_error_is_text_for_the_model_never_a_crash(tmp_path):
    tb = Toolbox(FakeSearcher([]), None, roots=[tmp_path])
    tb.searcher.find = lambda query, **kw: (_ for _ in ()).throw(RuntimeError("x" * 300))
    assert tb.call("nope", {}) == "unknown tool 'nope'"
    assert tb.call("read", {}).startswith("bad arguments for read: ")
    assert tb.call("tree", {"depth": 1, "colour": "red"}).startswith("[N1] listing of")       # unknown args dropped
    got = []
    tb.tree = lambda **kw: got.append(kw) or "ran"
    assert tb.call("tree", {"colour": "red", "depth": "2"}) == "ran" and got == [{"depth": 2}]   # wherever they stand
    got.clear()                      # null for an optional argument is "not given" (OpenAI's strict schemas write it so)
    assert tb.call("tree", {"path": None, "depth": None}) == "ran" and got == [{}]
    assert tb.call("read", {"path": None}).startswith("bad arguments for read: ")             # a required one: missing
    assert tb.call("search_notes", {"query": "q"}) == "search_notes failed: RuntimeError: " + "x" * 200


INTEGER_ARGS = [(name, k) for name, (_, props, _) in Toolbox.SPECS.items() for k, (t, _) in props.items()
                if t == "integer"]


@pytest.mark.parametrize("name, arg", INTEGER_ARGS)
@pytest.mark.parametrize("given, reaches", [(3, 3), ("3", 3), (" 12 ", 12), ("-2", -2), (3.0, 3), ("3.0", 3),
                                            ("two", None), ("3.5", None), (True, None), ([3], None)])
def test_arguments_are_read_by_their_schema_type(tmp_path, name, arg, given, reaches):
    """A model may write a number as a string (Llama-3.2-1B through Ollama sent {"k": "10", "page": "2"}, recorded
    2026-10-06; read then failed on "'>' not supported between 'str' and 'int'"): every integer argument reaches its
    tool as a number when it is one; otherwise the model is told which argument and why, and the tool does not run."""
    tb = Toolbox(None, None, roots=[tmp_path])
    got = []
    setattr(tb, name, lambda **kw: got.append(kw) or "ran")
    out = tb.call(name, {arg: given})
    if reaches is None:
        assert out == f"bad arguments for {name}: {arg} must be a whole number (got {given!r})" and got == []
    else:
        assert out == "ran" and got == [{arg: reaches}] and type(got[0][arg]) is int


def test_schemas_describe_every_tool_and_hide_the_web_without_a_search_server():
    full = Toolbox(None, FakeWeb(), roots=[], papers=FakePapers([]), wiki=FakeWiki([])).schemas()
    assert [s.name for s in full] == list(Toolbox.SPECS)
    only_papers = [s.name for s in Toolbox(None, None, roots=[], papers=FakePapers([])).schemas()]
    assert only_papers == [n for n in Toolbox.SPECS if n not in ("web_search", "search_wikipedia")]   # gaps, fetch: both
    only_web = [s.name for s in Toolbox(None, FakeWeb(), roots=[]).schemas()]
    assert only_web == [n for n in Toolbox.SPECS if n not in ("search_papers", "search_wikipedia")]
    only_wiki = [s.name for s in Toolbox(None, None, roots=[], wiki=FakeWiki([])).schemas()]
    assert only_wiki == [n for n in Toolbox.SPECS if n not in ("web_search", "search_papers")]
    wire = chat_tools(full)                                               # as the gateway sends them
    read = next(s for s in wire if s["function"]["name"] == "read")      # the shape the model's API reads
    desc, props, required = Toolbox.SPECS["read"]
    assert read == {"type": "function", "function": {"name": "read", "description": desc, "parameters": {
        "type": "object", "required": required, "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()}}}}
    gaps = next(s for s in wire if s["function"]["name"] == "gaps")["function"]["parameters"]["properties"]
    assert gaps["have"] == {"type": "array", "description": Toolbox.SPECS["gaps"][1]["have"][1], "items": {"type": "string"}}
    for s in full:
        params = s.parameters
        assert set(params["required"]) <= set(params["properties"])
        assert all(v.get("items") == {"type": "string"} for v in params["properties"].values() if v["type"] == "array")
    offline = [s.name for s in Toolbox(None, FakeWeb(), roots=[], papers=FakePapers([]),
                                                       wiki=FakeWiki([])).schemas(web=False)]
    assert offline == [n for n in Toolbox.SPECS if n not in ("gaps", "web_search", "search_papers", "search_wikipedia",
                                                             "fetch")]
    assert [s.name for s in Toolbox(None, None, roots=[]).schemas()] == offline
    no_providers = FakeWeb()
    no_providers.searcher = None                                 # no search server set
    assert [s.name for s in Toolbox(None, no_providers, roots=[]).schemas()] == offline
    web, papers, wiki = FakeWeb(), FakePapers([paper()]), FakeWiki([])
    local = Toolbox(None, web, roots=[], papers=papers, wiki=wiki)
    local.local_only = True                                      # /local: the user turned the web off
    assert [s.name for s in local.schemas()] == offline
    for name, args in (("gaps", {"have": [], "missing": ["x"]}), ("web_search", {"query": "x"}),
                       ("search_papers", {"query": "x"}), ("search_wikipedia", {"query": "x"}),
                       ("fetch", {"url": "https://u/1"})):
        assert local.call(name, args) == "refused: the user turned the web off for this session (local only)"
    assert web.searched == [] and papers.asked == [] and wiki.asked == [] and local.web_calls == 0
    local.local_only = False                                     # on again: the web as before
    assert [s.name for s in local.schemas()] == list(Toolbox.SPECS)
    local.gaps([], ["x"])
    assert "[W1] T1" in local.web_search("x") and web.searched == ["x"]


def test_a_failed_internal_check_reaches_the_user_not_the_model(tmp_path):
    """A tool's error is text for the model; a failed assert is a bug in peresearch and leaves the call."""
    def fails(error):
        return NS(find=lambda query, k=10, page=1: (_ for _ in ()).throw(error))
    assert Toolbox(fails(RuntimeError("disk")), None, roots=[tmp_path]).call(
        "search_notes", {"query": "x"}) == "search_notes failed: RuntimeError: disk"
    with pytest.raises(AssertionError, match="broken"):
        Toolbox(fails(AssertionError("broken")), None, roots=[tmp_path]).call("search_notes", {"query": "x"})


def test_a_search_says_what_the_index_does_not_hold(home, tmp_path):
    """The model must tell "not in the files" from "not searched by meaning": every search says what the index left
    out, counted by reason, and that grep and read still reach it."""
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    pymupdf = pytest.importorskip("pymupdf")
    root = tmp_path / "n"
    (root / "node_modules").mkdir(parents=True)
    (root / "node_modules" / "x.md").write_text("vendored\n")
    (root / "results.csv").write_text("model,score\nbig,0.9\n")
    scan = pymupdf.open()
    scan.new_page()
    scan.save(root / "scan.pdf")                                         # no text layer
    (root / "note.md").write_text("# Note\n\ngradient descent\n")
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    tb = Toolbox(Searcher(idx, reranker=FakeReranker()), None, roots=[root])
    weak = tb.search_notes("quantum chromodynamics")
    assert weak.endswith("(not searched by meaning: 1 PDF without a text layer, 1 folder or file made by tools or "
                         "peresearch's own, 1 table, JSON or log file; grep and read still reach the files)")
    assert "not searched by meaning" in tb.search_notes("gradient descent")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "note.md").write_text("# Note\n\ngradient descent\n")
    cidx = Index(home.parent / "clean-home", embedder=FakeEmbedder())
    cidx.update([clean])
    assert "not searched" not in Toolbox(Searcher(cidx, reranker=FakeReranker()), None, roots=[clean]).search_notes(
        "gradient descent")                                                       # nothing left out: nothing said
    tb.searcher.index.coverage = lambda: {"some_new_reason": 2}                        # a reason without a label
    assert tb.search_notes("gradient descent").endswith("(not searched by meaning: 2 some_new_reason; grep and read "
                                                       "still reach the files)")


def test_a_file_over_the_size_limit_is_not_searched_by_either_engine_and_said_so(tmp_path, monkeypatch):
    """ripgrep's --max-filesize does not apply to a path it is given: the limit is peresearch's to keep."""
    (tmp_path / "small.md").write_text("hit\n")
    (tmp_path / "dataset.jsonl").write_text('{"text": "hit"}\n' * 10)
    monkeypatch.setattr(tools.parse, "MAX_BYTES", 50)
    for engine in (tools.RG, None):
        monkeypatch.setattr(tools, "RG", engine)
        out = Toolbox(None, None, roots=[tmp_path]).grep("hit")
        assert out == f"[N1] {tmp_path / 'small.md'}:L1: hit\n(1 file over 50 bytes not searched)", engine


def test_a_store_text_removed_by_an_update_during_a_search_is_read_by_the_converter(home, tmp_path, monkeypatch):
    """An index update may remove a store text between grep's plan and ripgrep's read: the file is still searched."""
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    root = tmp_path / "n"
    root.mkdir()
    _converted_world(root)
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    real = tools._rg

    def update_meanwhile(paths, pattern, deadline):
        for p in paths:
            if Path(p).parent == home / "text":
                Path(p).unlink()
        return real(paths, pattern, deadline)
    monkeypatch.setattr(tools, "_rg", update_meanwhile)
    out = Toolbox(Searcher(idx, reranker=FakeReranker()), None, roots=[root]).grep("alpha")
    assert re.sub(r"(?m)^\[N\d+\] ", "", out).splitlines() == [
        f"{root / 'notes.docx'}:L3: an alpha paragraph", f"{root / 'old.txt'}:L3: third alpha",
        f"{root / 'paper.pdf'}:p.2: the alpha result", f"{root / 'plain.md'}:L3: an alpha note",
        f"{root / 'run.ipynb'}:cell 2: alpha = 1", f"{root / 'scores.xlsx'}:L3: alpha\t1",
        f"{root / 'talk.pptx'}:p.2: alpha slide"]


@pytest.mark.parametrize("name", ["Papers [2024]", "what?", "a*b"])
def test_a_folder_named_in_the_pattern_is_searched_whatever_the_declared_folders_name(tmp_path, name):
    root = tmp_path / name
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "x.md").write_text("hit in a package\n")
    out = Toolbox(None, None, roots=[root]).grep("hit", "node_modules/**")
    assert out == f"[N1] {root / 'node_modules' / 'pkg' / 'x.md'}:L1: hit in a package"


# --- what the mutation run of 2026-10-03 showed no test held ----------------------------------------------------

@pytest.mark.parametrize("pattern, hidden, want", [
    ("node_modules/pkg/**", True, ["node_modules/", "node_modules/pkg/"]),
    ("a/*/b.md", True, ["a/"]),                                         # up to the first wildcard
    ("*.md", True, []), ("../x/y", True, []), ("./docs//a.md", True, ["docs/", "docs/a.md"]),
    (".github/workflows/ci.yml", True, [".github/", ".github/workflows/", ".github/workflows/ci.yml"]),
    (".github/workflows/ci.yml", False, []),                            # not where hidden entries are refused
])
def test_named_takes_the_literal_leading_parts_of_a_pattern(tmp_path, pattern, hidden, want):
    assert tools._named(tmp_path, pattern, hidden) == tuple(str(tmp_path) + "/" + w for w in want)


def test_grep_engines_agree_on_files_ripgreps_own_settings_would_change(tmp_path, monkeypatch):
    """A user's ripgrep config (RIPGREP_CONFIG_PATH), a file name that is not UTF-8, a NUL byte after 8 KiB, an
    unreadable type before the match: both engines give the same lines."""
    cfg = tmp_path / "rgrc"
    cfg.write_text("--fixed-strings\n--max-count=1\n--smart-case\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(cfg))
    root = tmp_path / "n"
    root.mkdir()
    (root / "a.exe").write_bytes(b"MZ hit")                             # a type nobody reads, first in order
    (root / "b.md").write_text("HIT one\nhit two\n")
    (root / os.fsdecode(b"caf\xe9.md")).write_text("hit in a latin-1 name\n")
    (root / "late.txt").write_bytes(b"hit early\n" + b"x" * 9000 + b"\x00tail\n")   # binary for both
    (root / "0.txt").write_bytes(b"\x00 hit")                           # binary, first: the rest still searched
    outs = []
    for engine in (tools.RG, None):
        monkeypatch.setattr(tools, "RG", engine)
        outs.append(Toolbox(None, None, roots=[root]).grep("h.t"))
    assert outs[0] == outs[1]
    got = re.sub(r"(?m)^\[N\d+\] ", "", outs[0]).splitlines()
    assert got == [f"{root / 'b.md'}:L1: HIT one", f"{root / os.fsdecode(b'caf\xe9.md')}:L1: hit in a latin-1 name",
                   f"{root / 'b.md'}:L2: hit two"]                                   # in turns across files


def test_the_converter_path_keeps_the_limits_and_says_so(tmp_path, monkeypatch):
    import json
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    d.new_page().insert_text((72, 300), "hit in a big pdf")
    d.save(tmp_path / "big.pdf")
    (tmp_path / "nb.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": "\n".join(["hit"] * 9)}]}))
    for name in ("x.log", "y.log"):
        (tmp_path / name).write_text("hit\n" * 300)
    monkeypatch.setattr(tools.parse, "MAX_BYTES", 2**8)                 # the logs and the PDF are over it
    for engine in (tools.RG, None):
        monkeypatch.setattr(tools, "RG", engine)
        out = Toolbox(None, None, roots=[tmp_path]).grep("hit")
        assert f"{tmp_path / 'big.pdf'}:p.1: hit in a big pdf" in out                # a PDF has its own limit
        assert out.count(f"{tmp_path / 'nb.ipynb'}:cell 1: hit") == 9 and out.count(": hit") == 10   # every match
        assert "(2 files over 256 bytes not searched)" in out
    monkeypatch.setattr(tools.parse, "MAX_BYTES", 2**20)
    (tmp_path / "x.log").write_bytes(b"hit\n" * 300000)
    assert "(1 file over 1 MiB not searched)" in Toolbox(None, None, roots=[tmp_path]).grep("hit")
    monkeypatch.setattr(tools, "GREP_SECONDS", 0)
    assert "(stopped after 0 s: some files were not searched)" in Toolbox(None, None, roots=[tmp_path]).grep("hit")


def test_a_pattern_only_python_reads_still_searches_the_converted_files(tmp_path):
    import json
    (tmp_path / "a.md").write_text("alpha\n")
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": "alpha = 1"}]}))
    out = Toolbox(None, None, roots=[tmp_path]).grep(r"\N{LATIN SMALL LETTER A}lpha")
    assert re.sub(r"(?m)^\[N\d+\] ", "", out).splitlines() == [f"{tmp_path / 'a.md'}:L1: alpha",
                                                               f"{tmp_path / 'n.ipynb'}:cell 1: alpha = 1"]


def test_context_is_redacted_and_shown_whole():
    before = NS(start=1, end=2, text=f"key {SECRET}")
    after = NS(start=9, end=9, text="a" * 3_000)
    tb = Toolbox(FakeSearcher([hit("/n:2024/x.md:L3-8", "hit", "S", before, after, span="/n:2024/x.md:L1-9")]), None,
                 roots=[])
    out = tb.search_notes("q")
    assert SECRET not in out and "(before) key [REDACTED:github-token]" in out
    assert f"(after) {'a' * 3_000}\n" in out + "\n" and "…" not in out
    assert "§ S; the match: L3-8" in out                                # a path holding ":" keeps its place


def test_glob_tree_gaps_and_read_at_their_edges(tmp_path, monkeypatch):
    import json
    for name in ("s/a.md", "s/b.md", "t/c.md", "big/1.md", "big/2.md", "big/3.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x\n" * 500)
    tb = Toolbox(None, None, roots=[tmp_path])
    assert tb.glob(str(tmp_path / "s")) == f"{tmp_path / 's' / 'a.md'}\n{tmp_path / 's' / 'b.md'}"   # a folder: its files
    assert tb.glob("**/*.md").count("\n") == 5 and not tb.glob("*.md").startswith("(")
    monkeypatch.setattr(tools.files, "MAX_FILES_PER_FOLDER", 2)
    out = tb.tree()
    assert "  big/ (not searched: 3 files)" in out
    both = Toolbox(None, None, roots=[tmp_path / "s", tmp_path / "t"]).tree()
    assert both.startswith(f"[N1] listing of {tmp_path / 's'}, {tmp_path / 't'}\n")
    assert tb.gaps(have="one point", missing="a gap") == ("recorded: 1 point(s) the user's files cover, 1 missing or "
                                                         "not enough; web_search, search_papers and search_wikipedia "
                                                         "may now look for what is missing")
    assert tb.read("s/a.md", 1, 450).split("\n")[0].endswith(":L1-450")                 # every line asked for
    assert tb.read("s/a.md").split("\n")[0].endswith(":L1-500")                         # no end: the whole file
    (tmp_path / "one.ipynb").write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": "only"}]}))
    assert tb.read("one.ipynb").endswith(":cell 1-1\nonly")


def _overview_world(home, root):
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    (root / "course" / "deep").mkdir(parents=True)
    (root / "a.md").write_text("# Tokenizers\n\nBPE merges pairs.\n\n## SuperBPE\n\nMerges cross spaces.\n")
    (root / "course" / "b.md").write_text(f"# BPE notes\n\nBPE again, key {SECRET}\n")
    (root / "course" / "deep" / "c.py").write_text("'''BPE trainer.'''\ndef train(): pass\n")
    (root / "food.md").write_text("# Phở\n\nNước dùng.\n")
    (root / "long.md").write_text("# Long\n\n" + " ".join(["BPE"] + [f"word{i}" for i in range(120)]) + "\n")
    (root / "bare.py").write_text("# only a comment\n")
    (root / "emptydir").mkdir()
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.md").write_text("vendored BPE\n")                  # left out: made by tools
    idx = Index(home, embedder=FakeEmbedder())
    idx.update([root])
    return Searcher(idx, reranker=FakeReranker())


def test_overview_lists_every_document_each_a_source_with_its_outline(home, tmp_path):
    """overview: every document the search ranks, best first, each once with its relevance, its candidate passages,
    its headings and first line, and its best passage as a citable source, shown whole on one line (redacted, like
    every heading and line shown); an empty index; a file changed since indexing is named, not quoted; it says
    what the index does not hold."""
    root = tmp_path / "n"
    s = _overview_world(home, root)
    search_depth = s.depth
    tb = Toolbox(s, None, roots=[root])
    docs = s.documents("BPE")
    out = tb.overview("BPE")
    parts = out.split("\n\n")
    assert parts[0] == (f"{len(docs)} documents matching, best first (relevance: the reranker's score of each "
                        "document's best passage, for ordering; a low score does not show a document lacks the topic)")
    for i, d in enumerate(docs):
        sid, src = f"N{i + 1}", tb.sources.items[f"N{i + 1}"]
        heads, lead = s.outline(d.path)
        entry = parts[1 + i].split("\n")
        assert entry[0] == (f"[{sid}] {d.path} (relevance {d.score:.2f}; {d.matches} matching "
                            f"passage{'s' if d.matches != 1 else ''})")
        assert src.where == d.where and src.title == guard.redact(d.section)[0] and src.text == guard.redact(d.text)[0]
        assert ("  headings: " + guard.redact(", ".join(heads))[0] in entry) == bool(heads)
        assert ("  starts with: " + guard.redact(lead)[0] in entry) == bool(lead)
        assert entry[-2] == f"  best passage: {d.where}" + (f" § {guard.redact(d.section)[0]}" if d.section else "")
        assert entry[-1] == "  " + " ".join(guard.redact(d.text)[0].split())
    assert SECRET not in out                                  # the index withheld its paragraph
    from dataclasses import replace
    leaky = Toolbox(NS(index=s.index, depth=s.depth, documents=lambda q, f=None: [replace(docs[0], text=f"k {SECRET}",
                                                                                           section=f"s {SECRET}")],
                       outline=lambda path: ([f"h {SECRET}"], f"l {SECRET}")), None, roots=[root])
    shown = leaky.overview("BPE") + leaky.overview()          # whatever a searcher hands over, shown redacted
    many = Toolbox(NS(index=s.index, depth=s.depth, documents=lambda q, f=None: [replace(docs[0], text="p " * 2_000)] * 45,
                      outline=lambda path: ([], "")), None, roots=[root]).overview("BPE").split("\n\n")
    assert sum(e.startswith("[N1] ") for e in many) == 45 and many[1].endswith("\n  " + "p " * 1_999 + "p")   # all, whole
    catalog = Toolbox(NS(index=NS(read=lambda sql, *a: [(f"/n/f{i:02d}.md",) for i in range(70)], coverage=dict),
                         depth=s.depth, outline=lambda path: ([], "")), None, roots=[root]).overview().split("\n")
    assert catalog[0].startswith("70 files") and catalog[1:] == [f"/n/f{i:02d}.md" for i in range(70)]
    copied = Toolbox(NS(index=s.index, depth=s.depth, documents=lambda q, f=None: [replace(docs[0], copies=[
        "/n/a.template.md", "/n/b.md"])], outline=lambda path: ([], "")), None, roots=[root]).overview("BPE")
    assert copied.split("\n\n")[1].endswith("\n  also in: /n/a.template.md, /n/b.md")    # near-copies folded into it
    assert all("also in" not in e for e in out.split("\n\n"))
    assert SECRET not in shown and shown.count("[REDACTED:github-token]") >= 4
    assert parts[-1].startswith("(read a document with read") and len(parts) == len(docs) + 3   # nothing else
    assert parts[-2] == ("(not searched by meaning: 1 folder or file made by tools or peresearch's own; grep and "
                         "read still reach the files)")
    longest = next(e for e in parts if f" {root / 'long.md'} (" in e).split("\n")[-1]
    assert longest.endswith("word118 word119")                                        # whole, on one line
    assert Toolbox(s, None, roots=[root]).call("overview", {"query": "BPE", "page": 2}) == out   # an old argument:
    assert "page" not in tb.SPECS["overview"][1]                                                 # dropped
    s.depth = 1                                                        # a machine reranking fewer than all
    shallow = Toolbox(s, None, roots=[root]).overview("BPE").split("\n\n")
    assert "(relevance " in shallow[1].split("\n")[0] and all("(not reranked; " in e.split("\n")[0]
                                                               for e in shallow[2:len(docs) + 1])
    assert shallow[len(docs) + 1] == ("(a document marked 'not reranked' keeps the search's order: this machine "
                                      "reranks only the first 1 documents, PERESEARCH_RERANK_DEPTH)")
    s.depth = search_depth
    (root / "a.md").write_text("# Rewritten\n\nnothing of the old text\n")          # changed since indexing
    after = Toolbox(s, None, roots=[root])
    changed = after.overview("BPE")
    entry = next(e for e in changed.split("\n\n") if f"{root / 'a.md'} (" in e)
    assert entry.endswith("  (changed since indexing: read the file for its current text)")
    assert "Merges cross spaces" not in changed
    assert not entry.startswith("[") and all(str(root / "a.md") not in x.where for x in after.sources.items.values())
    assert len(entry.split("\n")) == 2                      # nothing the index holds of it: it is not what is there
    assert sum("matching passage" in e.split("\n")[0] for e in changed.split("\n\n")) == len(docs)   # all still listed
    assert changed.split("\n\n").index(entry) < len(docs)                 # and the changed one is not the last
    assert [x.id for x in after.sources.items.values()] == [f"N{i}" for i in range(1, len(after.sources.items) + 1)]
    (root / "a.md").unlink()                                                          # deleted since indexing
    gone = Toolbox(s, None, roots=[root]).overview("BPE")
    assert next(e for e in gone.split("\n\n") if f"{root / 'a.md'} (" in e).endswith("(changed since indexing: "
                                                                                     "read the file for its current text)")
    (tmp_path / "empty").mkdir()
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    idx = Index(home.parent / "empty-home", embedder=FakeEmbedder())
    idx.update([tmp_path / "empty"])
    assert Toolbox(Searcher(idx, reranker=FakeReranker()), None, roots=[tmp_path / "empty"]).overview("BPE") == \
        "no documents: nothing in the index matched"
    for searcher in (None, FakeSearcher([])):                          # none, or one without an index
        assert Toolbox(searcher, None, roots=[root]).overview("BPE") == ("overview needs the index, which is not "
                                                                         "open in this session; use tree, glob and grep")


def test_overview_without_a_topic_is_the_catalog_of_a_folder_and_keeps_to_the_declared_ones(home, tmp_path, monkeypatch):
    """No query: every file the index searched, inside the folder (or every declared one), by path, each with its
    headings or first line; a folder is named as for the other tools (relative to a declared folder, ~, absolute)
    and must lie inside a declared one: overview reads only the index, so it never asks to read elsewhere."""
    root = tmp_path / "n"
    s = _overview_world(home, root)
    asked = []
    tb = Toolbox(s, None, roots=[root], ask=lambda tool, path: asked.append(path) or "session")
    files = sorted(p for (p,) in s.index.read("SELECT path FROM files WHERE status='ok'"))
    out = tb.overview()
    assert out.split("\n")[0] == f"{len(files)} files searched by meaning, by path (give a topic to rank them)"
    lines = out.split("\n")[1:1 + len(files)]
    assert "(more" not in out and out.endswith("\n\n(not searched by meaning: 1 folder or file made by tools or "
                                                "peresearch's own; grep and read still reach the files)")
    assert f"{root / 'bare.py'}\n" in out                                # no headings, no first line: the path alone
    for line, path in zip(lines, files):
        heads, lead = s.outline(path)
        assert line == path + (" — " + guard.redact(", ".join(heads))[0] if heads else
                               " — " + guard.redact(lead)[0] if lead else "")
    inside = tb.overview(folder="course")
    assert inside.split("\n")[0] == f"2 files searched by meaning in {root / 'course'}, by path (give a topic to " \
                                    "rank them)"
    assert tb.overview(folder=str(root / "course")) == inside
    entries = [e for e in tb.overview("BPE", folder="course").split("\n\n") if e.startswith("[")]
    assert len(entries) == 2 and all(str(root / "course") in e.split(" (")[0] for e in entries)
    assert tb.overview("BPE", folder="course").startswith(f"2 documents matching in {root / 'course'}, best first (")
    assert tb.call("overview", {"page": 2}) == out                                    # an old argument: dropped
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.md").write_text("BPE\n")
    assert tb.overview("BPE", folder=str(outside)) == (f"{outside}: overview covers only the declared folders (the "
                                                       "index); tree, glob, grep and read reach other folders")
    other = tmp_path / "other"
    other.mkdir()
    two = Toolbox(s, None, roots=[other, root])                  # the folder named lies in the second declared one
    assert two.overview(folder=str(root / "course")) == tb.overview(folder="course")
    os.symlink(root / "course", root / "link")                 # a link to an indexed folder: the folder it names
    assert tb.overview(folder="link") == tb.overview(folder="course") != tb.overview(folder="link/..")
    monkeypatch.setenv("HOME", str(tmp_path))                     # ~ spelled: the same folder
    assert tb.overview(folder="~/n/course") == tb.overview(folder="course")
    assert tb.overview(folder="nowhere") == "nowhere: not found in the declared folders (tree without a path lists them)"
    assert tb.overview(folder="emptydir") == f"no files: the index holds none in {root / 'emptydir'}"
    assert tb.overview("BPE", folder="emptydir") == f"no documents: nothing in the index matched in {root / 'emptydir'}"
    assert asked == []
