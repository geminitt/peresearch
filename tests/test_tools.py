"""Contracts of the agent's tools (peresearch.tools).

Access, checked as one property over any sequence of tool calls, path spellings and user answers: a tool never
shows what lies in a hidden folder outside the declared ones, a protected file, peresearch's own data or outside the
home folder; it shows a file outside the declared folders only after the user allowed it; it never shows a
credential; it never raises. (Hidden files inside a declared folder are readable: the folder was declared whole.)
Then each tool's output: read's windows, grep's matches, glob's files, tree's counts, the sources and their ids,
the web gate, the error messages and the schemas. Web providers and pages have their own tests (test_web.py).
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from peresearch import guard, tools
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
        f"refused: {world['private'] / 'p.md'}: outside the declared folders; the user can allow it when asked in "
        "peresearch chat, or declare it with /add")
    assert tb.call("read", {"path": "nothing.md"}) == "refused: nothing.md: not found in the declared folders"


# --- read: windows of lines, pages or cells ---------------------------------------------------------------------

@given(st.integers(0, 260), st.integers(-3, 270), st.one_of(st.none(), st.integers(-3, 500)))
@settings(deadline=None, max_examples=80)
@example(250, 1, None)                                   # the default window: 200 lines
@example(5, 9, None)                                     # starting past the end
def test_read_returns_the_asked_lines_within_200(tmp_path_factory, n, start, end):
    root = tmp_path_factory.mktemp("r")
    lines = [f"line {i}" for i in range(1, n + 1)]
    (root / "f.txt").write_text("\n".join(lines))
    out = Toolbox(None, None, roots=[root]).read("f.txt", start, end)
    s = max(1, start)
    e = min(n, end or s + tools.MAX_READ_LINES - 1, s + tools.MAX_READ_LINES - 1)
    assert out == f"[N1] {root / 'f.txt'}:L{s}-{e}\n" + "\n".join(lines[s - 1:e])


def test_read_numbers_pages_and_cells_and_reports_an_unreadable_file(tmp_path):
    import json
    nb = tmp_path / "n.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source": f"c{i}"} for i in range(1, 21)]}))
    (tmp_path / "bin.txt").write_bytes(b"a\x00b")
    tb = Toolbox(None, None, roots=[tmp_path])
    assert tb.read("n.ipynb", 2, 3) == f"[N1] {nb}:cell 2-3\nc2\n\nc3"
    assert tb.read("n.ipynb", 6) == f"[N2] {nb}:cell 6-10\n" + "\n\n".join(f"c{i}" for i in range(6, 11))   # five
    assert tb.read("n.ipynb", 30) == f"[N3] {nb}:cell 30\n"
    assert tb.sources.items["N1"].title == "n.ipynb"
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    for i in range(3):
        d.new_page().insert_text((72, 72), f"page {i + 1}")
    d.save(tmp_path / "p.pdf")
    assert tb.read("p.pdf", 2, 3).split("\n")[0] == f"[N4] {tmp_path / 'p.pdf'}:p.2-3"
    assert tb.grep("page 3", "*.pdf") == f"[N5] {tmp_path / 'p.pdf'}:p.3: page 3"
    big = tmp_path / "big.md"
    big.write_text("x")
    from peresearch.zetokrag import parse as parse_module
    import unittest.mock as mock
    with mock.patch.object(parse_module, "MAX_BYTES", 0):
        assert tb.read("big.md") == f"{big}: cannot be read (too_large: 1 bytes)"
    assert tb.read("bin.txt") == f"{tmp_path / 'bin.txt'}: cannot be read (binary)"
    assert tb.call("read", {"path": "missing.md"}) == "refused: missing.md: not found in the declared folders"


# --- grep and glob ----------------------------------------------------------------------------------------------

files_with_lines = st.dictionaries(st.sampled_from(["a.md", "b.txt", "s/c.md", "s/t/d.py", ".h.md"]),
                                   st.lists(st.sampled_from(["alpha", "Beta gamma", "ALPHA beta", "", "x" * 400]),
                                            max_size=8), min_size=1)


@given(files_with_lines, st.sampled_from(["alpha", "beta", "^x+$", "gam+a"]), st.integers(1, 6))
@settings(deadline=None, max_examples=80)
def test_grep_reports_every_matching_line_where_it_is_in_walk_order(tmp_path_factory, files, pattern, cap):
    root = tmp_path_factory.mktemp("g")
    for name, lines in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("\n".join(lines))
    want = []
    for p in tools._files([(root, True)]):
        for i, line in enumerate((root / p).read_text().splitlines(), 1):
            if re.search(pattern, line, re.I):
                want.append(f"{p}:L{i}: {line.strip()[:300]}")
    out = Toolbox(None, None, roots=[root]).grep(pattern, max_results=cap)
    got = [re.sub(r"^\[N\d+\] ", "", x) for x in out.splitlines() if x.startswith("[N")]
    assert got == want[:cap]
    assert out.endswith(f"(stopped at {cap} matches)") == (len(want) >= cap) or (out == "no match" and not want)


def test_grep_names_pages_and_cells_redacts_and_explains_a_bad_pattern(tmp_path):
    import json
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": "x = 1"},
                                                             {"cell_type": "code", "source": f"y = 2\nkey = '{SECRET}'"}]}))
    tb = Toolbox(None, None, roots=[tmp_path])
    out = tb.grep("key")
    assert out.startswith(f"[N1] {tmp_path / 'n.ipynb'}:cell 2: key = ") and SECRET not in out and "[REDACTED" in out
    assert tb.sources.items["N1"].title == "n.ipynb" and tb.grep("nothing at all") == "no match"
    assert tb.grep("(").startswith("invalid regular expression")


@pytest.mark.parametrize("pattern, want", [
    ("*.md", [".h.md", "a.md", "s/c.md"]),         # no folder in the pattern: any depth, by name
    ("**/*.md", [".h.md", "a.md", "s/c.md"]),      # **/ also matches no folder at all (hidden files: declared)
    ("s/*", ["s/c.md", "s/t/d.py"]),               # fnmatch's * crosses folders
    ("s/t/*.py", ["s/t/d.py"]),
    ("*.txt", ["b.txt"]),
    ("c.md", ["s/c.md"]),                          # a bare name: found at any depth
    ("t/d.py", []),                                # with a folder: from the root only
    ("nothing*", []),
])
def test_glob_lists_the_files_its_pattern_names(tmp_path, pattern, want):
    for name in ("a.md", "b.txt", "s/c.md", "s/t/d.py", ".h.md", "s/.git/x.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    out = Toolbox(None, None, roots=[tmp_path]).glob(pattern)
    assert out == ("\n".join(str(tmp_path / w) for w in want) or "no file matches")
    assert Toolbox(None, None, roots=[tmp_path]).glob("**/*", max_results=2).count("\n") == 1


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
            if e.name.startswith(".") or e.name in tools.TREE_HIDDEN or guard.denied(Path(e.path)):
                continue
            out.append(e)
        return out

    def expect(folder, level, lines):
        for e in visible(folder):
            indent = "  " * level
            if not e.is_dir():
                lines.append(indent + e.name)
            elif e.name in guard.SKIP_DIRS:
                lines.append(f"{indent}{e.name}/ (not read)")
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


def test_tree_shows_at_most_40_entries_a_folder_and_says_how_many_more(tmp_path, monkeypatch):
    (tmp_path / "forty").mkdir()
    for i in range(40):
        (tmp_path / "forty" / f"g{i:02d}.md").write_text("x")
    assert not Toolbox(None, None, roots=[tmp_path / "forty"]).tree().endswith("more")
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "only.md").write_text("x")
    assert f"{tmp_path / 'one'}/ (0 folders, 1 file)" in Toolbox(None, None, roots=[tmp_path / "one"]).tree()
    shutil.rmtree(tmp_path / "forty")
    shutil.rmtree(tmp_path / "one")
    for i in range(43):
        (tmp_path / f"f{i:02d}.md").write_text("x")
    out = Toolbox(None, None, roots=[tmp_path]).tree()
    assert "f39.md" in out and "f40.md" not in out and out.endswith("  … 3 more")
    assert f"{tmp_path}/ (0 folders, 43 files)" in out
    assert Toolbox(None, None, roots=[]).tree() == "no folders are declared"
    assert Toolbox(None, None, roots=[tmp_path]).tree(str(tmp_path / "f00.md")) == f"{tmp_path / 'f00.md'}: not a folder"
    assert Toolbox._count([NS(is_dir=lambda follow_symlinks: True)]) == "(1 folder, 0 files)"


# --- sources and their ids ----------------------------------------------------------------------------------------

@given(st.lists(st.tuples(st.sampled_from(["file", "web"]), st.sampled_from(["p:L1", "p:L2", "https://u", "https://v"]),
                          st.text(max_size=5)), max_size=12))
def test_a_source_keeps_its_id_and_its_longest_text(adds):
    s = Sources()
    first, longest = {}, {}
    for kind, where, text in adds:
        got = s.add(kind, where, "t", text)
        first.setdefault(where, (kind, len([k for k, _ in first.values() if k == kind]) + 1))
        longest[where] = max(longest.get(where, ""), text, key=len)
        k, n = first[where]
        assert got.id == ("N" if k == "file" else "W") + str(n) and got.text == longest[where]
    assert s.urls == {where: src.id for src in s.items.values() if src.kind == "web" for where in [src.where]}


@given(st.text(max_size=50), st.integers(1, 40))
def test_clip_keeps_short_text_and_says_how_much_was_cut(text, n):
    out = tools._clip(text, n)
    assert out == text if len(text) <= n else out == text[:n] + f"\n… [{len(text) - n} more characters]"


# --- search_notes, the web gate, errors and schemas ----------------------------------------------------------------

class FakeSearcher:
    def __init__(self, hits, verdict="partial", stale=0):
        self.result, self.asked = (hits, verdict, stale), []

    def find(self, query, k):
        self.asked.append((query, k))
        return self.result[0][:k], self.result[1], self.result[2]


def hit(where, text, section="", neighbours=(), score=0.5):
    return NS(where=where, text=text, section=section, neighbours=list(neighbours), score=score)


def test_search_notes_shows_each_excerpt_with_its_id_neighbours_and_verdict():
    tb = Toolbox(FakeSearcher([hit("a.md:L1-3", "first", "S", ["next door"]), hit("b.md:L5", f"key {SECRET}")],
                              stale=2), None, roots=[])
    out = tb.search_notes("q", k=5)
    assert out.split("\n\n") == [
        "verdict: partial (does the user's own material answer this?)",
        "[N1] a.md:L1-3 (relevance 0.50) § S\nfirst\nnext door",
        "[N2] b.md:L5 (relevance 0.50)\nkey [REDACTED:github-token]",
        "(2 excerpts dropped: their files changed since indexing)"]
    assert tb.search_notes("q", k=1).count("[N") == 1 and tb.searcher.asked == [("q", 5), ("q", 1)]
    assert tb.sources.items["N1"].title == "S" and tb.sources.items["N2"].text == "key [REDACTED:github-token]"
    long = Toolbox(FakeSearcher([hit("c.md:L1", "z" * 2000)]), None, roots=[]).search_notes("q")
    assert long.endswith("z" * 1500 + "\n… [500 more characters]")


class FakeWeb:
    providers = ["one"]

    def __init__(self):
        self.searched, self.pages = [], {"https://u/1": "page text", "https://u/2": ""}

    def search(self, query, n):
        self.searched.append(query)
        return "fake", [NS(title="T1", url="https://u/1", snippet="s1"), NS(title="T2", url="https://u/2", snippet="s2")][:n]

    def fetch(self, url):
        return self.pages[url]


def test_the_web_is_searched_after_gaps_and_only_returned_pages_are_read():
    web = FakeWeb()
    tb = Toolbox(None, web, roots=[])
    assert tb.call("web_search", {"query": "x"}) == ("refused: first record with gaps(have, missing) what the user's "
                                                     "files already cover and what is missing; then search the web for "
                                                     "what is missing") and web.searched == []
    assert tb.call("gaps", {"have": "one point", "missing": ["a", "b"]}) == \
        "recorded: 1 point(s) the user's files cover, 2 missing or not enough; web_search may now look for what is missing"
    assert tb.recorded_gaps == {"have": ["one point"], "missing": ["a", "b"]}
    assert tb.call("web_search", {"query": "x", "n": 1}) == "(fake)\n\n[W1] T1\nhttps://u/1\ns1" and tb.web_calls == 1
    assert tb.sources.items["W1"].title == "T1" and web.searched == ["x"]
    assert tb.call("fetch", {"url": "https://u/2"}).startswith("refused: fetch only opens URLs")
    tb.call("web_search", {"query": "y"})
    assert tb.web_calls == 2 and web.searched == ["x", "y"]
    assert tb.call("fetch", {"url": "https://u/1"}) == "[W1] https://u/1\npage text"
    assert tb.sources.items["W1"].text == "page text"
    assert tb.call("fetch", {"url": "https://u/2"}) == "[W2] https://u/2\nthe page could not be read"
    assert tb.sources.items["W2"].text == "s2"                           # an empty page keeps the snippet
    web.search = lambda query, n: ("fake", [NS(title="L", url="https://u/3", snippet="s" * 700)])
    assert tb.call("web_search", {"query": "z"}).endswith("s" * 600 + "\n… [100 more characters]")
    web.search = lambda query, n: ("fake", [])
    assert tb.call("web_search", {"query": "z"}) == "no results"
    tb.call("gaps", {"have": ["a", "b"], "missing": "c"})
    assert tb.recorded_gaps == {"have": ["a", "b"], "missing": ["c"]}
    tb.call("gaps", {"have": None, "missing": None})
    assert tb.recorded_gaps == {"have": [], "missing": []}


def test_a_tool_error_is_text_for_the_model_never_a_crash(tmp_path):
    tb = Toolbox(FakeSearcher([]), None, roots=[tmp_path])
    tb.searcher.find = lambda query, k: (_ for _ in ()).throw(RuntimeError("x" * 300))
    assert tb.call("nope", {}) == "unknown tool 'nope'"
    assert tb.call("read", {}).startswith("bad arguments for read: ")
    assert tb.call("tree", {"depth": 1, "colour": "red"}).startswith("[N1] listing of")       # unknown args dropped
    assert tb.call("search_notes", {"query": "q"}) == "search_notes failed: RuntimeError: " + "x" * 200


def test_schemas_describe_every_tool_and_hide_the_web_without_providers():
    full = Toolbox(None, FakeWeb(), roots=[]).schemas()
    assert [s["function"]["name"] for s in full] == list(Toolbox.SPECS)
    for s in full:
        params = s["function"]["parameters"]
        assert set(params["required"]) <= set(params["properties"])
        assert all(v.get("items") == {"type": "string"} for v in params["properties"].values() if v["type"] == "array")
    offline = [s["function"]["name"] for s in Toolbox(None, FakeWeb(), roots=[]).schemas(web=False)]
    assert offline == [n for n in Toolbox.SPECS if n not in ("gaps", "web_search", "fetch")]
    assert [s["function"]["name"] for s in Toolbox(None, None, roots=[]).schemas()] == offline
    no_providers = FakeWeb()
    no_providers.providers = []
    assert [s["function"]["name"] for s in Toolbox(None, no_providers, roots=[]).schemas()] == offline
