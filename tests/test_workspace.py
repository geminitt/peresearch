"""Contracts of the declared folders (peresearch.workspace) and the settings (peresearch.settings)."""

import os
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from hypothesis import given, settings as hsettings
from hypothesis import strategies as st

from peresearch import guard, settings
from peresearch.workspace import PROBLEMS, Workspace, brief, complete, quiet, summary


@given(st.lists(st.tuples(st.sampled_from(["add", "remove"]), st.sampled_from(["a", "a/b", "a/b/c", "d", "a/../a/b"])),
                max_size=8))
@hsettings(deadline=None, max_examples=80)
def test_the_declared_folders_never_overlap_and_a_parent_absorbs_its_children(tmp_path_factory, ops):
    base = tmp_path_factory.mktemp("w")
    home = base / "home"
    for d in ("a/b/c", "d"):
        (base / d).mkdir(parents=True, exist_ok=True)
    saved, os.environ["PERESEARCH_HOME"] = os.environ.get("PERESEARCH_HOME"), str(home)
    try:
        ws, model = Workspace(index=None), []
        for op, rel in ops:
            p = (base / rel).resolve()
            if op == "remove":
                assert ws.remove(str(base / rel)) == (p in model)
                model = [r for r in model if r != p]
                continue
            inside = next((r for r in model if r == p or r in p.parents), None)
            if inside:
                with pytest.raises(ValueError, match=f"already inside the declared folder {inside}"):
                    ws.add(str(base / rel))
            else:
                assert ws.add(str(base / rel)) == p
                model = [r for r in model if p not in r.parents] + [p]
            assert ws.folders() == sorted(model, key=str)
            assert not any(r != q and r in q.parents for r in model for q in model)
    finally:
        os.environ.pop("PERESEARCH_HOME") if saved is None else os.environ.__setitem__("PERESEARCH_HOME", saved)


def test_a_folder_that_is_not_one_protected_or_peresearchs_own_is_refused(home, tmp_path):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / "f.md").write_text("x")
    home.mkdir(parents=True, exist_ok=True)
    (home / "sub").mkdir()
    ws = Workspace(index=None)
    for path, why in ((tmp_path / "nope", "not a folder"), (tmp_path / "f.md", "not a folder"),
                      (tmp_path / ".ssh", "matches a protected pattern"), (home, "peresearch's own data"),
                      (home / "sub", "peresearch's own data")):
        with pytest.raises(ValueError, match=why):
            ws.add(str(path))
    assert ws.folders() == []


def test_peresearchs_own_folder_is_refused_when_reached_through_a_link(tmp_path, monkeypatch):
    real = tmp_path / "real-data"
    (real / "sub").mkdir(parents=True)
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setenv("PERESEARCH_HOME", str(tmp_path / "link"))      # the setting names the link
    with pytest.raises(ValueError, match="peresearch's own data"):
        Workspace(index=None).add(str(real / "sub"))


def test_the_index_is_made_when_first_needed_and_updated_over_the_declared_folders(home, tmp_path):
    calls = []
    ws = Workspace(index=NS(update=lambda stop=None, progress=None: calls.append((stop, progress)) or "report"))
    asked, told = (lambda: False), (lambda done, total: None)
    assert ws.update() == "report" and ws.update(stop=asked, progress=told) == "report"
    assert calls == [(None, None), (asked, told)]                                         # passed through
    from peresearch.zetokrag.index import Index
    lazy = Workspace()
    assert isinstance(lazy.index, Index) and lazy.index is lazy.index and lazy.index.home == home


def test_the_summary_lists_counts_problems_and_withheld_files():
    r = NS(added=2, changed=1, unchanged=7, removed=0, chunks=40, denied=3,
           problems={"too_large": ["big.pdf"], "duplicate": ["b (same as a)"], "odd": ["x"]},
           withheld={"keys.md": 2, "env.md": 1})
    assert summary(r).splitlines() == [
        "2 added · 1 changed · 7 unchanged · 0 removed · 40 chunks · 3 protected paths skipped",
        f"1 {PROBLEMS['duplicate']}:", "  b (same as a)", "1 odd:", "  x", f"1 {PROBLEMS['too_large']}:", "  big.pdf",
        "3 chunks withheld because they look like credentials:", "  keys.md (2)", "  env.md (1)"]
    many = NS(**{**vars(r), "problems": {"error": [f"f{i}" for i in range(30)]}, "withheld": {}})
    assert len(summary(many, limit=5).splitlines()) == 1 + 1 + 5
    big = NS(**{**vars(r), "problems": {}, "withheld": {}, "largest": [("lib/stb_image.h", 206), ("thesis.pdf", 120)]})
    assert summary(big).splitlines()[1:] == [
        "files with the most chunks (one that is not yours, such as a copied library, can be excluded in "
        ".peresearchignore):", "  lib/stb_image.h (206 chunks)", "  thesis.pdf (120 chunks)"]


@pytest.mark.parametrize("change, problems, withheld, line", [
    ((0, 0, 0), {}, {}, "Index up to date · 40 chunks from 10 files"),
    ((2, 1, 0), {}, {}, "Index updated: 2 added, 1 changed, 0 removed · 40 chunks from 10 files"),
    ((0, 0, 3), {"hidden": [".x"]}, {}, "Index updated: 0 added, 0 changed, 3 removed · 40 chunks from 10 files · "
                                        "what was left out: /status"),
    ((1, 0, 0), {}, {"keys.md": 2, "env.md": 1}, "Index updated: 1 added, 0 changed, 0 removed · 40 chunks from "
                                                 "10 files · 3 chunks withheld: they look like credentials"),
    ((1200, 0, 0), {}, {}, "Index updated: 1,200 added, 0 changed, 0 removed · 40 chunks from 1,200 files")])
def test_the_brief_says_what_an_update_did_in_one_line(change, problems, withheld, line):
    """The interface's one line after an update (the full report is the CLI's and /status's): no path, ever."""
    added, changed, removed = change
    r = NS(added=added, changed=changed, unchanged=max(0, 10 - added - changed), removed=removed, chunks=40, denied=0,
           problems=problems, withheld=withheld)
    assert brief(r) == line and "\n" not in brief(r)
    assert quiet(r) == (change == (0, 0, 0) and not withheld)
    one = NS(**{**vars(r), "added": 1, "changed": 0, "unchanged": 0, "removed": 0, "chunks": 1, "problems": {},
               "withheld": {"k.md": 1}})
    assert brief(one) == ("Index updated: 1 added, 0 changed, 0 removed · 1 chunk from 1 file · 1 chunk withheld: it "
                          "looks like a credential")


def test_the_reports_say_what_the_embedding_did():
    """The vectors are made after the files are indexed (the user's endpoint): both reports say how many were made,
    how many still wait — searched by their words meanwhile — and why, when the endpoint did not answer; with no
    endpoint set, nothing about vectors is said (the search is by words, and /setup and check say so)."""
    from peresearch.zetokrag.index import Report
    r = Report(added=2, chunks=7, embedded=5, unembedded=2, embed_error="http://e/v1/embeddings: 503 after 3 tries")
    assert summary(r).splitlines()[1:] == [
        "5 chunks embedded · 2 still without a vector (searched by their words until the next update): "
        "http://e/v1/embeddings: 503 after 3 tries"]
    assert brief(r).endswith(" · 2 chunks wait for their vectors: the embeddings endpoint did not answer")
    cut = Report(added=1, chunks=3, embedded=3, halved=2)
    assert summary(cut).splitlines()[1:] == ["3 chunks embedded · 2 by their head only: the embedding model takes "
                                             "less than a chunk (all of each is still searched by its words)"]
    done = Report(added=2, chunks=7, embedded=7)
    assert summary(done).splitlines()[1:] == ["7 chunks embedded"] and "vector" not in brief(done)
    assert len(summary(Report(chunks=7)).splitlines()) == 1 and brief(Report(chunks=7)) == \
        "Index up to date · 7 chunks from 0 files"                       # no embedder, or nothing new to embed


def test_completion_offers_folders_as_a_shell_would(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for d in ("projects", "papers", ".hidden", "pics"):
        (tmp_path / d).mkdir()
    (tmp_path / "plan.md").write_text("x")
    assert complete(str(tmp_path) + "/p") == [str(tmp_path / n) + "/" for n in ("papers", "pics", "projects")]
    assert complete("~/pa") == ["~/papers/"]
    assert complete("~") == ["~/papers/", "~/pics/", "~/projects/"]                 # hidden only when asked
    assert complete("~/.h") == ["~/.hidden/"]
    assert complete("~/pl") == [] and complete("~/pl", folders_only=False) == ["~/plan.md"]          # a file: no "/"
    monkeypatch.chdir(tmp_path)
    assert complete("") == ["papers/", "pics/", "projects/"]                     # nothing typed: the current folder
    assert complete(str(tmp_path / "nothing" / "x")) == [] and complete(None) == complete("")
    assert complete(str(tmp_path) + "/") == [str(tmp_path / n) + "/" for n in ("papers", "pics", "projects")]


def test_a_setting_comes_from_the_environment_then_the_saved_file_then_the_default(home, monkeypatch):
    monkeypatch.delenv("PERESEARCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert settings.saved() == {} and settings.get("PERESEARCH_LLM_MODEL") == "" and settings.get("TAVILY_API_KEY") == ""
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.env").write_text("# a comment\n#EXA_API_KEY=old\n\n TAVILY_API_KEY = 'tvly-a=b' \n"
                                       "PERESEARCH_LLM_MODEL=\"qwen\"\nbroken line\n")
    assert settings.saved() == {"TAVILY_API_KEY": "tvly-a=b", "PERESEARCH_LLM_MODEL": "qwen"}
    assert settings.get("TAVILY_API_KEY") == "tvly-a=b" and settings.get("PERESEARCH_LLM_MODEL") == "qwen"
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    assert settings.get("TAVILY_API_KEY") == "from-env"
    assert guard.denied(home / "settings.env")                                      # never indexed or read


@pytest.mark.parametrize("key, value, want", [
    ("steps", " 20 ", ("PERESEARCH_STEPS", "20")), ("web-searches", "0", ("PERESEARCH_WEB_SEARCHES", "0")),
    ("pages", "30", ("PERESEARCH_PAGES", "30")), ("seconds", "30", ("PERESEARCH_SECONDS", "30")),
    ("thinking", "OFF", ("PERESEARCH_LLM_THINKING", "off")), ("thinking", "on", ("PERESEARCH_LLM_THINKING", "on")),
    ("context", "32768", ("PERESEARCH_LLM_CONTEXT", "32768")), ("rerank-depth", "0", ("PERESEARCH_RERANK_DEPTH", "0")),
    ("context", "1024", ("PERESEARCH_LLM_CONTEXT", "1024")),
    ("steps", "0", "steps must be a whole number from 1 to 50 (got '0')"),
    ("steps", "51", "steps must be a whole number from 1 to 50 (got '51')"),
    ("steps", "1.5", "steps must be a whole number from 1 to 50 (got '1.5')"),
    ("seconds", "29", "seconds must be a whole number from 30 to 7200 (got '29')"),
    ("thinking", "maybe", "thinking must be on or off (got 'maybe')"),
    ("context", "1023", "context must be a whole number of tokens, at least 1024 (got '1023')"),
    ("rerank-depth", "31", "rerank-depth must be a whole number from 0 to 30 (got '31')"),
    ("colour", "red", "unknown setting colour: steps, web-searches, pages, seconds, thinking, context, rerank-depth")])
def test_a_config_value_is_checked_before_it_is_kept(key, value, want):
    """/config's settings (the harness's limits, the reasoning switch, the context, the rerank depth): each value is
    checked against its range and kept as written; a bad one is refused with the setting and its range."""
    if isinstance(want, tuple):
        assert settings.check(key, value) == want
    else:
        with pytest.raises(ValueError, match=f"^{re.escape(want)}$"):
            settings.check(key, value)
    assert [k for k, *_ in settings.CONFIG] == ["steps", "web-searches", "pages", "seconds", "thinking", "context",
                                                "rerank-depth"]
