"""Layer 1 of bounding a tool's output (PLAN.md step 3.3): a tool stops collecting past registry.COLLECT_BYTES of
UTF-8 (1 MiB, owner 2026-10-21, after measuring the declared folders) and says what it left and how to ask for less;
the registry cuts anything a tool still returns past it. And grep shows what it found even when finding took its time."""

import json

import pytest

from peresearch import registry, tools
from peresearch.registry import COLLECT_BYTES, Registry, ToolSpec
from peresearch.tools import Toolbox
from tests.test_tools import FakeWeb


def test_the_collection_limit_is_the_owners_one_mib():
    assert COLLECT_BYTES == 1 << 20


def test_grep_shows_what_it_found_when_finding_it_took_the_whole_time(tmp_path, monkeypatch):
    """On the owner's folders grep `e` found 5,786,799 lines in its 60 s and showed none, its first line reading "no
    match" (2026-10-21): the one deadline covered building the result too."""
    (tmp_path / "a.md").write_text("alpha\nbeta alpha\n")
    tb = Toolbox(None, None, roots=[tmp_path])
    clock = [0.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    found = tb._matches

    def slow(*args):
        hits, _, too_large, unreadable = found(*args)
        clock[0] += 10_000                                  # the search used all its time
        return hits, True, too_large, unreadable
    monkeypatch.setattr(tb, "_matches", slow)
    out = tb.grep("alpha")
    assert out.splitlines()[0].endswith(": alpha") and "beta alpha" in out and "no match" not in out
    assert "(stopped after 60 s: some files were not searched)" in out


def test_grep_returns_at_most_the_limit_in_turns_across_files_and_says_how_many_it_left(tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text("".join(f"match {i:06d} " + "x" * 90 + "\n" for i in range(12_000)))
    out = Toolbox(None, None, roots=[tmp_path]).grep("match")
    shown = [line for line in out.splitlines() if "match " in line and line.startswith("[")]
    assert len(out.encode()) <= COLLECT_BYTES + 300
    assert {"a.md", "b.md"} <= {line.split(":L")[0].rsplit("/", 1)[-1] for line in shown}   # both files, in turns
    assert out.splitlines()[-1].startswith(f"({len(shown):,} of 24,000 matching lines shown, 1 MiB: each file's first")
    assert "narrow the pattern or the glob" in out


def test_glob_tree_and_the_catalog_stop_at_the_limit_and_say_so(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 2_000)
    for i in range(200):
        (tmp_path / f"note-{i:03d}.md").write_text("x")
    tb = Toolbox(None, None, roots=[tmp_path])
    listed = tb.glob("*.md")
    assert len(listed.encode()) <= 2_000 + 200
    files = [line for line in listed.splitlines() if line.endswith(".md")]
    assert listed.splitlines()[-1] == f"({len(files)} of 200 files listed, 2,000 bytes; narrow the pattern)"
    tree = tb.tree(str(tmp_path))
    assert len(tree.encode()) <= 2_000 + 200
    assert tree.splitlines()[-1] == "(the listing stops at 2,000 bytes; give tree a folder inside it, or glob)"


def test_read_returns_the_limit_and_says_where_to_read_on(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 1_000)
    (tmp_path / "big.txt").write_text("".join(f"line {i:04d} " + "é" * 20 + "\n" for i in range(400)))
    tb = Toolbox(None, None, roots=[tmp_path])
    out = tb.read("big.txt")
    head, *body = out.splitlines()
    shown = [line for line in body if line.startswith("line ")]
    assert head.endswith(f":L1-{len(shown)}") and len("\n".join(shown).encode()) <= 1_000
    assert body[-1] == f"(lines 1-{len(shown)} of 400 shown, 1,000 bytes; read on with start={len(shown) + 1})"
    again = tb.read("big.txt", len(shown) + 1)
    assert again.splitlines()[1] == f"line {len(shown):04d} " + "é" * 20
    whole = tb.read("big.txt", 399, 400)                                   # what fits is returned as before
    assert whole.splitlines()[0].endswith(":L399-400") and "read on" not in whole


def test_read_of_one_line_longer_than_the_limit_returns_its_head(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 1_000)
    (tmp_path / "min.js").write_text("a" * 5_000 + "\nsecond\n")
    out = Toolbox(None, None, roots=[tmp_path]).read("min.js")
    assert out.splitlines()[1] == "a" * 1_000
    assert out.splitlines()[-1] == "(line 1 cut at 1,000 bytes; 2 lines in all; read on with start=2)"


def test_pages_and_cells_stop_at_a_whole_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 1_000)
    cells = [{"cell_type": "markdown", "source": f"cell {i} " + "y" * 300} for i in range(10)]
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": cells}))
    out = Toolbox(None, None, roots=[tmp_path]).read("n.ipynb")
    assert out.splitlines()[0].endswith(":cell 1-3") and "cell 2 " in out and "cell 3 " not in out   # from cell 1
    assert out.splitlines()[-1] == "(cells 1-3 of 10 shown, 1,000 bytes; read on with start=4)"


def test_a_fetched_page_past_the_limit_is_cut_and_said(monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 1_000)
    web = FakeWeb()
    web.pages["https://u/1"] = "".join(f"para {i}\n" for i in range(500))
    tb = Toolbox(None, web, roots=[])
    tb.call("web_search", {"query": "x"})
    out = tb.call("fetch", {"url": "https://u/1"})
    assert len(out.encode()) <= 1_000 + 200
    assert out.splitlines()[-1] == "(the first 1,000 bytes of the page; the rest is not kept)"


def test_the_registry_cuts_any_tool_that_returns_more_and_bounds_what_a_refusal_repeats(monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 1_000)
    big = "".join(f"row {i}\n" for i in range(1_000))
    r = Registry([ToolSpec("lookup", "", {}, "web", lambda **kw: big),
                  ToolSpec("count", "", {"type": "object", "properties": {"n": {"type": "integer"}}}, "files",
                           lambda **kw: "ok")])
    out = r.run("lookup", {}).text
    assert out.startswith("row 0\n") and len(out.encode()) <= 1_000 + 200
    assert out.splitlines()[-1] == f"[cut at 1,000 bytes: lookup returned {len(big):,}; ask it for less]"
    said = r.run("count", {"n": "x" * 50_000}).text
    assert said.startswith("bad arguments for count: n must be a whole number (got 'xxx") and len(said) < 200


def test_overview_and_its_catalog_stop_at_the_limit_and_say_so(home, tmp_path, monkeypatch):
    from tests.test_tools import _overview_world
    root = tmp_path / "n"
    s = _overview_world(home, root)
    tb = Toolbox(s, None, roots=[root])
    whole_docs, whole_catalog = tb.overview("BPE"), tb.overview()
    monkeypatch.setattr(registry, "COLLECT_BYTES", 400)
    docs = tb.overview("BPE")
    listed = [p for p in docs.split("\n\n") if p.startswith("[N")]
    assert docs != whole_docs and len(listed) >= 1
    assert f"({len(listed)} of {len(s.documents('BPE'))} documents listed, 400 bytes; narrow the topic or give a " \
           "folder)" in docs
    catalog = tb.overview()
    assert catalog != whole_catalog and ", 400 bytes; give a folder or a topic)" in catalog
    shown = catalog.split("\n\n")[0].splitlines()
    assert shown[-1].startswith(f"({len(shown) - 2} of ")


# --- layer 2: the model gets one page of a result at a time (agent.PAGE_BYTES, PAGE_LINES: OpenCode's) ---

from peresearch.agent import MORE, Agent, _wrap  # noqa: E402
from peresearch.llm import Reply, ToolCall  # noqa: E402
from tests.test_agent import FakeModel, FakeToolbox  # noqa: E402


class RoomyModel(FakeModel):
    """A model with a context far larger than any page (a million tokens), or none known (`context` None)."""

    def __init__(self, replies, context=10**6):
        super().__init__(replies)
        self.context = context

    def context_length(self):
        return self.context


def asked(model, call_id):
    """What the model was shown for one call, in the last request."""
    return next(m["content"] for m in model.seen[-1]["messages"] if m.get("tool_call_id") == call_id)


@pytest.mark.parametrize("context", [10**6, None])
def test_a_result_longer_than_a_page_is_shown_one_page_at_a_time(home, context):
    from peresearch import agent as agent_module
    assert (agent_module.PAGE_BYTES, agent_module.PAGE_LINES) == (50 * 1024, 2000)
    result = "".join(f"line {i:05d} " + "é" * 30 + "\n" for i in range(3_000))     # 3,000 lines, 216 KB
    tb = FakeToolbox()
    tb.call = lambda name, args: result
    model = RoomyModel([Reply("", [ToolCall("c1", "tree", {})]),
                        Reply("", [ToolCall("c2", "recall", {"ref": "R1", "from": 0})]), Reply("done")], context)
    Agent(model, tb, record=True).ask("q")
    first = asked(model, "c1")
    body = first.split("\n", 1)[1].rsplit("\n</tool_output>", 1)[0]
    head, more = body.rsplit("\n… [", 1)
    assert len(head.encode()) <= 50 * 1024 and head.count("\n") < 2000 and head.endswith("\n")
    assert result.startswith(head) and len(head) < len(result)
    assert ("… [" + more) == MORE.format(ref="R1", total=len(result), upto=len(head)).lstrip("\n")
    again = asked(model, "c2")                                            # recall from 0: one page again, not all
    assert len(again.encode()) <= 50 * 1024 + 300 and "recall(\"R1\", from=" in again


def test_a_page_stops_at_2000_lines_however_short_they_are(home):
    result = "".join(f"{i}\n" for i in range(5_000))
    tb = FakeToolbox()
    tb.call = lambda name, args: result
    model = RoomyModel([Reply("", [ToolCall("c1", "tree", {})]), Reply("done")])
    Agent(model, tb, record=False).ask("q")
    body = asked(model, "c1").split("\n", 1)[1].rsplit("\n… [", 1)[0]
    assert body.count("\n") == 2000 and body.endswith("1999\n")


def test_a_result_within_a_page_is_shown_whole(home):
    result = "short result\n" * 100
    tb = FakeToolbox()
    tb.call = lambda name, args: result
    model = RoomyModel([Reply("", [ToolCall("c1", "tree", {})]), Reply("done")])
    Agent(model, tb, record=False).ask("q")
    assert asked(model, "c1") == _wrap("tree", result)


def test_sizes_count_a_file_name_that_is_not_utf8(tmp_path):
    import os
    from peresearch.registry import fitting, head
    name = os.fsdecode(b"caf\xe9.md")                                   # an undecodable byte, as os.fsdecode keeps it
    assert fitting([name], room=9) == 1 and fitting([name], room=8) == 0    # the stray byte kept as 3 (surrogate)
    assert head(name * 3, 18) == name * 2
    (tmp_path / name).write_text("hit\n")
    assert Toolbox(None, None, roots=[tmp_path]).grep("hit").endswith(":L1: hit")
