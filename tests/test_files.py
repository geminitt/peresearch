"""Contracts of files.py: which files peresearch reads.

- A folder a tool made is left out whatever its name; a folder a person named (data, runs, third_party…) is walked.
- Hidden names below a declared folder are left out unless the user includes them.
- Patterns follow the documented glob rules; a .peresearchignore applies to its folder and below; config.json wins.
- A large folder is left out unless declared or included.
- Every file under a declared folder is accounted for: walked, left out with a reason (itself or a folder above
  it), or counted as protected. Nothing disappears silently, whatever the tree.
"""
import os
import string
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from peresearch import files
from peresearch.files import IGNORE_FILE, Skipped, compile_pattern, made_by_a_tool, matches


def run(roots, config=None):
    skipped, denied = [], []
    walked = list(files.walk([Path(r) for r in roots], config or {}, skipped.append, lambda: denied.append(1)))
    return walked, skipped, len(denied)


def touch(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_folder_a_tool_made_is_left_out_whatever_its_name(tmp_path):
    touch(tmp_path / "my-env" / "pyvenv.cfg")                         # a venv named by its user
    touch(tmp_path / "my-env" / "lib" / "x.py")
    (tmp_path / "py311" / "conda-meta").mkdir(parents=True)
    touch(tmp_path / "py311" / "x.py")
    touch(tmp_path / "target" / "CACHEDIR.TAG", files.CACHEDIR_SIGNATURE.decode() + "\n# tagged\n")
    touch(tmp_path / "target" / "x.py")
    touch(tmp_path / "fake-cache" / "CACHEDIR.TAG", "not the signature\n")   # a look-alike is not a cache
    touch(tmp_path / "fake-cache" / "notes.md")
    for name in ("node_modules", "site-packages", "__pycache__", "pkg.egg-info", "pkg-1.0.dist-info", ".git"):
        touch(tmp_path / name / "x.py")
    walked, skipped, _ = run([tmp_path])
    assert walked == [tmp_path / "fake-cache" / "CACHEDIR.TAG", tmp_path / "fake-cache" / "notes.md"]
    assert {Path(s.path).name: s.note for s in skipped if s.reason == "tool_made"} == {
        "my-env": "Python environment (pyvenv.cfg)", "py311": "conda environment (conda-meta)",
        "target": "cache (CACHEDIR.TAG)", "node_modules": "node_modules: made by a package manager",
        "site-packages": "site-packages: made by a package manager", "__pycache__": "__pycache__: made by a package manager",
        "pkg.egg-info": "pkg.egg-info: made by a package manager",
        "pkg-1.0.dist-info": "pkg-1.0.dist-info: made by a package manager", ".git": "version control"}
    assert made_by_a_tool(tmp_path / "fake-cache") is None


def test_peresearchs_own_folder_is_never_walked(tmp_path, monkeypatch):
    """Its index, extracted texts and private eval corpus would come back as the user's notes, whatever its name."""
    own = tmp_path / "pr-data"                                          # neither hidden nor marked by a tool
    monkeypatch.setenv("PERESEARCH_HOME", str(own))
    touch(own / "text" / "abc.txt")
    touch(own / "eval" / "gold" / "corpus" / "n.md")
    touch(tmp_path / "note.md")
    walked, skipped, _ = run([tmp_path])
    assert walked == [tmp_path / "note.md"]
    assert [(Path(s.path).name, s.reason, s.note) for s in skipped] == [("pr-data", "tool_made", "peresearch's own data")]
    assert run([tmp_path], {"include": ["pr-data/"]})[0] == [tmp_path / "note.md"]          # not even when included
    walked, skipped, _ = run([own / "eval"])                                               # nor when declared
    assert walked == [] and [(s.path, s.reason) for s in skipped] == [(str(own / "eval"), "tool_made")]
    (tmp_path / "link").symlink_to(own)                                 # PERESEARCH_HOME given through a link
    monkeypatch.setenv("PERESEARCH_HOME", str(tmp_path / "link"))
    assert run([tmp_path])[0] == [tmp_path / "note.md"]


def test_folders_a_person_named_are_walked(tmp_path):
    names = ("data", "datasets", "runs", "outputs", "results", "build", "dist", "checkpoints", "wandb", "third_party",
             "vendor")
    for n in names:
        touch(tmp_path / n / "notes.md")
    walked, skipped, _ = run([tmp_path])
    assert sorted(p.parent.name for p in walked) == sorted(names) and skipped == []


def test_hidden_names_are_left_out_unless_included(tmp_path):
    """A hidden name, and an Office owner file ("~$report.docx": made while the document is open, a few bytes naming
    who opened it, hidden on Windows but not once copied or seen through WSL), are left out unless included."""
    touch(tmp_path / ".obsidian" / "app.json")
    touch(tmp_path / ".hidden-note.md")
    touch(tmp_path / "note.md")
    touch(tmp_path / "~$report.docx")
    root = tmp_path / ".declared"                                       # a declared folder may itself be hidden
    touch(root / "inside.md")
    walked, skipped, _ = run([tmp_path])
    assert walked == [tmp_path / "note.md"]
    assert sorted((Path(s.path).name, s.reason, s.note) for s in skipped) == [
        (".declared", "hidden", ""), (".hidden-note.md", "hidden", ""), (".obsidian", "hidden", ""),
        ("~$report.docx", "tool_made", "Office owner file (~$)")]
    walked, _, _ = run([tmp_path], {"include": [".obsidian/", ".hidden-note.md", "~$report.docx"]})
    assert set(walked) == {tmp_path / "note.md", tmp_path / ".hidden-note.md", tmp_path / ".obsidian" / "app.json",
                           tmp_path / "~$report.docx"}
    assert run([root])[0] == [root / "inside.md"]


@pytest.mark.parametrize("pattern, path, is_folder, want", [
    ("*.log", "a/b/run.log", False, True),
    ("*.log", "a/b/run.log.md", False, False),
    ("build/", "x/build", True, True),
    ("build/", "x/build", False, False),                                 # a file named build is not the folder
    ("docs/*.md", "docs/a.md", False, True),
    ("docs/*.md", "docs/sub/a.md", False, False),                        # * stays inside one part
    ("docs/**/*.md", "docs/sub/deep/a.md", False, True),
    ("docs/**/*.md", "docs/a.md", False, True),                          # **/ also matches no folder
    ("**/tmp", "a/b/tmp", True, True),
    ("data?.csv", "data1.csv", False, True),
    ("data?.csv", "data12.csv", False, False),
    ("[!a]*.txt", "b.txt", False, True),
    ("[!a]*.txt", "a.txt", False, False),
    ("ghi chú.md", "x/ghi chú.md", False, True),                         # spaces and Vietnamese, as typed
    ("笔记/*.md", "笔记/a.md", False, True),
    ("Notes.md", "notes.md", False, False),                              # case-sensitive
    ("Papers [[]2024]/*.md", "Papers [2024]/a.md", False, True),         # [[] is a literal "["
    ("docs/**", "docs/a/b.md", False, True),                             # ** at the end crosses parts
    ("x/a**z.md", "x/a/b/z.md", False, True),                           # ** inside a part crosses parts too
    ("x/a**z.md", "x/a/b/y.md", False, False),
    ("[^x]1.md", "^1.md", False, True),                                  # ^ in a set is a character, not "not"
    ("[^x]1.md", "a1.md", False, False),
])
def test_patterns_follow_the_documented_globs(tmp_path, pattern, path, is_folder, want):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # a regex Python reads two ways is a bug
        p = compile_pattern(pattern, tmp_path, False, "test")
    assert matches(p, tmp_path / path, is_folder) is want


def test_absolute_and_home_patterns_match_the_full_path(tmp_path):
    p = compile_pattern(str(tmp_path / "a" / "*.md"), Path("/elsewhere"), False, "test")
    assert matches(p, tmp_path / "a" / "x.md", False) and not matches(p, tmp_path / "b" / "x.md", False)
    q = compile_pattern("~/notes/*.md", Path("/elsewhere"), False, "test")
    assert matches(q, Path.home() / "notes" / "x.md", False)
    x = compile_pattern("*X/", tmp_path, False, "t")                      # only "/" is taken off a pattern's ends
    assert matches(x, tmp_path / "a" / "dirX", True) and not matches(x, tmp_path / "a" / "dir", True)
    assert matches(compile_pattern("/Xa/*.md", tmp_path, False, "t"), Path("/Xa/b.md"), False)
    owner = compile_pattern("~$*", tmp_path, False, "test")              # a name starting with "~", not a home
    assert matches(owner, tmp_path / "a" / "~$x.docx", False) and not matches(owner, tmp_path / "x.docx", False)
    assert matches(compile_pattern("~", Path("/elsewhere"), False, "t"), Path.home(), True)          # "~": the home
    assert compile_pattern("# a comment", tmp_path, False, "t") is None and compile_pattern("  ", tmp_path, False, "t") is None


def test_an_ignore_file_rules_its_folder_and_below_and_config_has_the_last_word(tmp_path):
    touch(tmp_path / "a" / IGNORE_FILE, "*.log\n!keep.log\nscratch/\n# a comment\n")
    for p in ("a/x.log", "a/keep.log", "a/sub/y.log", "a/scratch/s.md", "a/note.md", "b/x.log"):
        touch(tmp_path / p)
    walked, skipped, _ = run([tmp_path])
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in walked)
    assert rel == ["a/keep.log", "a/note.md", "b/x.log"]                 # b is outside a's ignore file
    assert {Path(s.path).relative_to(tmp_path).as_posix(): s.reason for s in skipped} == {
        "a/scratch": "user_excluded", "a/x.log": "user_excluded", "a/sub/y.log": "user_excluded"}
    assert all(s.note.startswith(str(tmp_path / "a" / IGNORE_FILE)) for s in skipped)
    walked, _, _ = run([tmp_path], {"exclude": ["keep.log"], "include": ["x.log"]})
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in walked)
    assert rel == ["a/note.md", "a/x.log", "b/x.log"]


def test_nothing_below_an_excluded_folder_comes_back(tmp_path):
    touch(tmp_path / "out" / "important.md")
    walked, skipped, _ = run([tmp_path], {"exclude": ["out/"], "include": ["important.md"]})
    assert walked == [] and [(Path(s.path).name, s.reason) for s in skipped] == [("out", "user_excluded")]


def test_a_large_folder_is_left_out_unless_declared_or_included(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "MAX_FILES_PER_FOLDER", 3)
    for i in range(4):
        touch(tmp_path / "reviews" / f"{i}.txt")
        touch(tmp_path / "vault" / f"{i}.md")
    touch(tmp_path / "reviews" / "sub" / "kept.md")                      # the files of a large folder only
    walked, skipped, _ = run([tmp_path])
    assert [p.relative_to(tmp_path).as_posix() for p in walked] == ["reviews/sub/kept.md"]
    assert sorted((Path(s.path).name, s.reason, s.note) for s in skipped) == [
        ("reviews", "large_folder", "4 files"), ("vault", "large_folder", "4 files")]
    walked, _, _ = run([tmp_path], {"include": ["vault/"]})
    assert sum(p.parent.name == "vault" for p in walked) == 4
    assert len(run([tmp_path / "vault"])[0]) == 4                        # declared: the user chose it


def test_malformed_patterns_in_config_are_the_users_error(tmp_path):
    with pytest.raises(ValueError, match="exclude"):
        run([tmp_path], {"exclude": "*.log"})
    with pytest.raises(ValueError, match="include"):
        run([tmp_path], {"include": [3]})
    touch(tmp_path / "Xold" / "a.md")                                  # a config pattern starting with X: whole
    touch(tmp_path / "keep.md")
    walked, skipped, _ = run([tmp_path], {"exclude": ["Xold/"]})
    assert walked == [tmp_path / "keep.md"] and [(Path(s.path).name, s.reason) for s in skipped] == [("Xold", "user_excluded")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every folder")
def test_an_unreadable_folder_is_reported_not_fatal(tmp_path):
    touch(tmp_path / "locked" / "a.md")
    touch(tmp_path / "open.md")
    os.chmod(tmp_path / "locked", 0)
    try:
        walked, skipped, _ = run([tmp_path])
    finally:
        os.chmod(tmp_path / "locked", 0o755)
    assert walked == [tmp_path / "open.md"]
    assert [(Path(s.path).name, s.reason) for s in skipped] == [("locked", "unreadable")]


def test_protected_paths_are_counted_never_named(tmp_path):
    touch(tmp_path / ".ssh" / "id_rsa")
    touch(tmp_path / "notes" / ".env")
    touch(tmp_path / "notes" / "a.md")
    walked, skipped, denied = run([tmp_path])
    assert walked == [tmp_path / "notes" / "a.md"] and skipped == [] and denied == 2


# --- every file accounted for, whatever the tree -------------------------------------------------------------------

NAMES = ["a", "b", "data", "runs", ".h", "node_modules", "env", "x.md", "y.log", ".z.md", "ghi chú", "笔记"]
tree = st.lists(st.tuples(st.lists(st.sampled_from(NAMES), min_size=1, max_size=4),
                          st.sampled_from(["note.md", "run.log", ".dot.md", "data.csv", "~$o.docx"])), min_size=1,
                 max_size=25)
markers = st.lists(st.tuples(st.lists(st.sampled_from(NAMES), min_size=1, max_size=3),
                             st.sampled_from(["pyvenv.cfg", IGNORE_FILE])), max_size=4)
patterns = st.lists(st.sampled_from(["*.log", "data/", "!run.log", "**/a/*.md", "runs", ".h/", "笔记/"]), max_size=3)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tree, markers, patterns, patterns)
def test_every_file_is_walked_skipped_or_counted(tmp_path_factory, layout, marks, ignore_lines, excludes):
    root = tmp_path_factory.mktemp("tree")
    made = set()
    for parts, name in layout:
        made.add(touch(root.joinpath(*parts, name)))
    for parts, name in marks:
        touch(root.joinpath(*parts, name), "\n".join(ignore_lines) + "\n" if name == IGNORE_FILE else "home = x\n")
    walked, skipped, denied = run([root], {"exclude": [e.lstrip("!") for e in excludes]})
    assert len(walked) == len(set(walked))
    left_out = {Path(s.path) for s in skipped}
    for f in made:
        covered = f in walked or any(f == s or s in f.parents for s in left_out)
        assert covered or denied, f"{f.relative_to(root)} neither walked, skipped nor counted"
    for f in walked:                                                    # nothing walked inside what was left out
        assert not any(s in f.parents for s in left_out)
        rel = f.relative_to(root).parts
        assert not any(p.startswith((".", "~$")) for p in rel)          # no pattern here includes a hidden name


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tree, markers, patterns, patterns, st.integers(1, 4))
@example([(["a"], "run.log"), (["a", "runs"], "note.md"), (["b"], "run.log")], [(["a"], IGNORE_FILE)],
         ["*.log", "runs"], [], 4)                                    # an ignore file rules its folder and below only
def test_check_gives_the_verdict_walk_gives_for_every_path(tmp_path_factory, layout, marks, ignore_lines, excludes,
                                                           limit):
    root = tmp_path_factory.mktemp("tree")
    made = set()
    for parts, name in layout:
        made.add(touch(root.joinpath(*parts, name)))
    for parts, name in marks:
        touch(root.joinpath(*parts, name), "\n".join(ignore_lines) + "\n" if name == IGNORE_FILE else "home = x\n")
    touch(root / ".ssh" / "id_rsa")                                     # protected: counted by walk, never named
    config = {"exclude": [e.lstrip("!") for e in excludes]}
    old = files.MAX_FILES_PER_FOLDER
    files.MAX_FILES_PER_FOLDER = limit
    try:
        walked, skipped, _ = run([root], config)
        by_path = {Path(s.path): s for s in skipped}
        for f in made | {root / ".ssh" / "id_rsa"}:
            verdict = files.check(f, [root], config)
            if f in walked:
                assert verdict is None, f
                continue
            assert verdict is not None, f
            where = Path(verdict.path)
            assert where == f or where in f.parents
            if verdict.reason == files.PROTECTED:
                assert where not in by_path                             # walk only counted it
            else:
                assert by_path[where] == verdict                        # the same place, reason and note
        for s in skipped:                                               # and each folder walk left out, asked itself
            if Path(s.path).is_dir() and s.reason != "large_folder":
                assert files.check(Path(s.path), [root], config) == s
    finally:
        files.MAX_FILES_PER_FOLDER = old


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are a POSIX thing")
def test_a_named_pipe_or_socket_is_left_out_and_never_opened(tmp_path):
    """Opening a named pipe waits for a writer for ever: walking, indexing and grep hung on one (2026-10-03)."""
    import signal
    import socket as socketlib
    os.mkfifo(tmp_path / "pipe.md")
    sock = socketlib.socket(socketlib.AF_UNIX)
    sock.bind(str(tmp_path / "s.md"))
    touch(tmp_path / "note.md")
    os.symlink(tmp_path / "pipe.md", tmp_path / "link-to-pipe.md")
    os.symlink("/dev/null", tmp_path / "null.md")                       # a device, reached through a link

    def stuck(*_):
        raise TimeoutError("a special file was opened")
    old = signal.signal(signal.SIGALRM, stuck)
    signal.alarm(5)
    try:
        walked, skipped, denied = run([tmp_path])
        from peresearch.zetokrag import parse
        doc = parse.read(tmp_path / "pipe.md")
        device = parse.read(tmp_path / "null.md")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        sock.close()
    assert walked == [tmp_path / "note.md"]
    assert sorted((Path(s.path).name, s.reason, s.note) for s in skipped) == [
        ("link-to-pipe.md", "special_file", "named pipe"), ("pipe.md", "special_file", "named pipe"),
        ("s.md", "special_file", "socket")]
    assert (doc.status, doc.note) == ("unsupported", "named pipe")
    assert (device.status, device.note) == ("unsupported", "device") and denied == 1    # the link leads outside


def test_rules_at_their_edges(tmp_path, monkeypatch):
    """What the mutation run of 2026-10-03 showed no test held."""
    # an anchored pattern says nothing about a path outside its folder
    assert not matches(compile_pattern("docs/*.md", tmp_path / "a", False, "t"), tmp_path / "b" / "docs" / "x.md", False)
    # a "!" before a config.json pattern means nothing there: the pattern is the rest
    touch(tmp_path / "keep.md")
    assert run([tmp_path], {"exclude": ["!keep.md"]})[0] == []
    # an ignore file that is not UTF-8 is still read, and each reason names its line
    (tmp_path / "keep.md").unlink()
    (tmp_path / IGNORE_FILE).write_bytes(b"# r\xe9sum\xe9\n*.log\n")
    touch(tmp_path / "x.log")
    touch(tmp_path / "build")                                           # a file: not the folder build/
    _, skipped, _ = run([tmp_path], {"exclude": ["build/"]})
    assert [(Path(s.path).name, s.note) for s in skipped] == [("x.log", f"{tmp_path / IGNORE_FILE}:2")]


def test_peresearchs_own_folder_reached_through_a_linked_declared_folder(tmp_path, monkeypatch):
    real = tmp_path / "real"
    touch(real / "pr-data" / "n.md")
    touch(real / "note.md")
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setenv("PERESEARCH_HOME", str(real / "pr-data"))
    walked, skipped, _ = run([tmp_path / "link"])
    assert walked == [tmp_path / "link" / "note.md"]
    assert [(Path(s.path).name, s.reason, s.note) for s in skipped] == [("pr-data", "tool_made", files.OWN)]
    other = touch(tmp_path / "other" / "o.md")
    walked, skipped, _ = run([real / "pr-data", tmp_path / "other"])        # the next declared folder still walked
    assert walked == [other] and skipped == [Skipped(str(real / "pr-data"), "tool_made", files.OWN)]
    assert files.check(real / "pr-data" / "n.md", [real / "pr-data"], {}) == Skipped(str(real / "pr-data"),
                                                                                     "tool_made", files.OWN)


def test_check_outside_any_folder_in_a_large_declared_one_and_below_an_unreadable_one(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "MAX_FILES_PER_FOLDER", 2)
    for i in range(3):
        touch(tmp_path / f"{i}.md")                                     # a declared folder is never "large"
    assert files.check(tmp_path / "0.md", [tmp_path], {}) is None
    assert files.check(tmp_path.parent / "elsewhere.md", [tmp_path], {}) is None
    sub = touch(tmp_path / "sub" / "a.md")
    for i in range(2):
        touch(tmp_path / "sub" / f"b{i}.md")
    assert files.check(sub, [tmp_path], {}) == Skipped(str(tmp_path / "sub"), "large_folder", "3 files")
    if os.geteuid() != 0:
        os.chmod(tmp_path / "sub", 0o300)                               # can enter, cannot list
        try:
            assert files.check(sub, [tmp_path], {}) == Skipped(str(tmp_path / "sub"), "unreadable", "PermissionError")
            assert run([tmp_path])[1] == [Skipped(str(tmp_path / "sub"), "unreadable", "PermissionError")]
        finally:
            os.chmod(tmp_path / "sub", 0o755)
