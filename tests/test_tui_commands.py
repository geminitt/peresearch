"""Contracts of the interface's commands and dialogs (peresearch.tui.Chat), driven with Textual's test pilot:
what every /command does in every state (no workspace, a workspace, busy, nothing yet), /resume and its menu,
Tab completion, the permission dialog's four answers, and what a clicked link may do."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from rich.cells import cell_len

from peresearch import tui
from peresearch.agent import Agent, Answer, Check, Limits, history_file
from peresearch.tools import Source
from peresearch.llm import Reply, ToolCall
from peresearch.tui import COMMANDS, KEYS, Permission, Prompt
from peresearch.workspace import Workspace
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import ScriptedLLM, agent, notes, settle, texts, toolbox, tui_run, user_home


async def say(app, pilot, text):
    app.query_one("#ask", Prompt).value = text
    await pilot.press("enter")
    await pilot.pause()


def notes_of(app):
    from textual.widgets import Static
    return [str(w.render()) for w in app.query("#log > .note")]


def test_every_command_answers_when_there_is_nothing_to_act_on(home, notes):
    async def steps(app, pilot):
        await pilot.pause()
        for text in ("/folders", "/add ~/x", "/remove ~/x", "/index", "/retry", "/sources", "/resume", "/bogus x"):
            await say(app, pilot, text)
        await say(app, pilot, "/help")
        await say(app, pilot, "/keys")
        app.before_exit = texts(app)
        app.code_before = app.return_code
        await say(app, pilot, "/exit")
        app.code_after = app.return_code
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=None)
    assert notes_of_shown(app.before_exit) == [
        "no folders declared: /add <folder>", "folders cannot be changed here", "folders cannot be changed here",
        "folders cannot be changed here", "nothing to retry yet", "no sources yet",
        "no earlier conversation in this project", "unknown command /bogus; / for the list"]
    assert "**Commands** (the keys: `/keys`)\n" + "\n".join(f"- `{c}` {d}" for c, d in COMMANDS.items()) in app.before_exit
    assert "**Keys**\n" + "\n".join(f"- **{k}** {d}" for k, d in KEYS.items()) in app.before_exit
    assert (app.code_before, app.code_after) == (None, 0)                  # /exit quits


def notes_of_shown(shown):
    keep = ("no ", "folders cannot", "nothing to", "unknown command")
    return [line for line in shown.splitlines() if line.startswith(keep)]


def test_folders_are_added_listed_indexed_and_removed(home, notes, tmp_path):
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    more = tmp_path / "more"
    more.mkdir()
    update, labels = ws.update, []
    ws.update = lambda: (labels.append(ws.app.indexing), update())[1]     # what the status line says meanwhile

    async def steps(app, pilot):
        ws.app = app
        await settle(app, pilot, lambda: not app.indexing)
        for text in ("/add", f"/add {tmp_path / 'nope'}", f"/add {notes}"):
            await say(app, pilot, text)
        await settle(app, pilot, lambda: not app.indexing)
        app.indexing = "Indexing"                                            # an update still running
        await say(app, pilot, "/index")
        app.indexing = ""
        ws.add(str(more))
        app.declared = ws.folders()
        await say(app, pilot, "/folders")
        await say(app, pilot, f"/remove {tmp_path}")
        await say(app, pilot, "/index")
        await settle(app, pilot, lambda: not app.indexing)
        await say(app, pilot, f"/remove {notes}")
        await settle(app, pilot, lambda: not app.indexing)
        await say(app, pilot, "/remove")
        app.all_notes = notes_of(app)
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws, index_on_start=True)
    n = app.all_notes
    assert n[0] == "No folders yet: /add <folder> lets peresearch read one."
    assert n[1:4] == ["usage: /add <folder>", f"not a folder: {tmp_path / 'nope'}", f"Added {notes.resolve()}"]
    assert "wait for the current work to finish (a question: esc interrupts it)" in n
    assert len(app.declared) == 2 and "\n".join(map(str, app.declared)) in n      # one folder per line
    assert f"{tmp_path} was not a declared folder" in n
    assert labels == ["Indexing the new folder", "Dropping its chunks from the index", "Indexing",
                      "Dropping its chunks from the index"]
    assert any(x.startswith("Index up to date: ") and "chunks from 3 files" in x for x in n)
    assert f"Removed {notes}" in n and n[-1] == "usage: /remove <folder>"
    assert any("3 added" in x for x in n)                                # the first index of the new folder


def test_busy_guards_retry_sources_and_new(home, notes):
    gate = threading.Event()

    class Slow(ScriptedLLM):
        def chat(self, messages, tools=None, thinking=None, stop=None):
            while not gate.is_set():
                if stop and stop():
                    from peresearch.llm import Interrupted
                    raise Interrupted("stopped")
                time.sleep(0.02)
            return super().chat(messages, tools, thinking, stop)
    llm = Slow(Reply("", [ToolCall("c", "read", {"path": "bpe.md"})]), Reply("BPE merges pairs [N1]."),
               Reply("again [N1]."), Reply("fresh."))

    async def steps(app, pilot):
        await say(app, pilot, "How does BPE work?")
        for text in ("/new", "/retry", "/resume", "another question"):
            await say(app, pilot, text)
        app.busy_notes = list(notes_of(app))
        gate.set()
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.first = app.last
        app.last = None
        await say(app, pilot, "/retry")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        await say(app, pilot, "/new")
        app.after_new = texts(app)
        await say(app, pilot, "something else")
        await settle(app, pilot, lambda: not app.busy and len(llm.seen) >= 4)
    app = tui_run(lambda e: Agent(llm, toolbox(home, notes)), steps)
    assert app.busy_notes == ["wait for the current question to finish, or press esc"] * 3 + [
        "still working on the previous question (esc to interrupt it)"]
    retried = [m["content"] for m in llm.seen[2][0] if m["role"] == "user"]
    assert retried[-1].endswith("How does BPE work?")                    # /retry asks the same question again
    assert app.after_new.strip() == "New conversation: earlier questions are no longer sent as context."
    assert [m["content"] for m in llm.seen[3][0] if m["role"] == "user"] == ["something else"]   # no history


def write_history(conversations):
    """A project's history as the agent writes it: one line per turn, with its conversation."""
    path = history_file("default")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"time": when, "conversation": cid, "question": q, "answer": a, "sources": src}) + "\n"
                            for cid, turns in conversations for when, q, a, src in turns))


def minute(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def shown_log(app):
    from textual.widgets import Markdown
    return [("page", w.source) if isinstance(w, Markdown) else (" ".join(sorted(w.classes)), str(w.render()))
            for w in app.query("#log > *")]


def test_resume_shows_a_conversation_as_it_was_and_continues_it(home, notes, monkeypatch):
    """/resume lists the 50 most recent conversations, each on a line that fits the menu; the chosen one is shown as
    it was — when it started, how many questions, each question as typed, each answer with its sources as links
    (their excerpts are not kept) — before the agent exists or after, and the next question continues it."""
    opened = []
    monkeypatch.setattr(tui, "open_outside", lambda target, line=None: opened.append((target, line)) or
                        ("not here" if target == "/n/bpe.md" else None))
    old = ("c-old", [(1_000_000_000, "What is [b]BPE[/b]?", "See [N1] and [W1].\n<sub>kept</sub>",
                      {"N1": "/n/bpe.md:L3-5", "W1": "https://example.org/a:L3"})])
    two = ("c-two", [(1_100_000_000, "How does BPE work?", "BPE merges pairs.", {}),
                     (1_100_000_060, "And SuperBPE?", "It has a second stage.", {})])
    long = ("c-long", [(1_200_000_000, "a long question about tokenization " * 10, "ok", {})])
    fillers = [(f"c-fill-{i}", [(900_000_000 + i, f"filler {i}", "ok", {})]) for i in range(48)]
    write_history([old, two, long] + fillers)

    def make(on_event):
        a, llm = agent(home, notes, Reply("It comes from compression."), Reply("More on it."))
        make.llm = llm
        return a

    async def pick(app, pilot, conversation, key):
        from textual.widgets import OptionList
        menu = app.query_one("#commands", OptionList)
        menu.highlighted = [menu.get_option_at_index(i).id for i in range(menu.option_count)].index("resume:" + conversation)
        await pilot.press(key)
        await pilot.pause()

    async def steps(app, pilot):
        from textual.widgets import Markdown, OptionList
        await say(app, pilot, "/resume")
        menu = app.query_one("#commands", OptionList)
        options = [menu.get_option_at_index(i) for i in range(menu.option_count)]
        app.listed = [o.id for o in options]
        app.long_row = cell_len(str(options[0].prompt)), app.size.width
        await pick(app, pilot, "c-old", "enter")                     # the box is empty: Enter takes the row
        app.old = shown_log(app)
        page = app.query(Page).last()
        for href in ("source:N1", "source:W1", "open:W1"):
            page.post_message(Markdown.LinkClicked(page, href))
            await pilot.pause()
        shown = app.query(Page).nodes
        shown[1].post_message(Markdown.LinkClicked(shown[1], "open:N1"))   # "open the file" on the shown source
        await pilot.pause()
        app.clicked = shown_log(app)[len(app.old):]
        await say(app, pilot, "Where does it come from?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.resume("c-gone")                                        # gone from the history meanwhile
        await pilot.pause()
        app.gone = shown_log(app)
        await say(app, pilot, "/resume")
        await pick(app, pilot, "c-two", "tab")                       # the agent exists now
        app.two = shown_log(app)
        app.last = None
        await say(app, pilot, "More?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    from peresearch.tui import Page
    app = tui_run(make, steps)
    assert app.listed == ["resume:" + c for c in ["c-long", "c-two", "c-old"] + [f"c-fill-{i}" for i in range(47, 0, -1)]]
    assert app.long_row[0] == app.long_row[1] - 6                       # cut to the menu's inner width
    assert app.old == [
        ("note", f"Resumed the conversation of {minute(1_000_000_000)} (1 question); the next question continues it."),
        ("question", "❯ What is [b]BPE[/b]?"),
        ("page", "See [N1](source:N1) and [W1](source:W1).\n<sub>kept</sub>\n\n**Sources**\n"
                 "- [N1](source:N1) /n/bpe.md:L3-5\n- [W1](source:W1) https://example.org/a:L3\n")]
    assert app.clicked == [
        ("page", "**N1** /n/bpe.md:L3-5\n\n> _(the excerpt is not kept in the history)_\n\n[open the file](open:N1)"),
        ("page", "**W1** https://example.org/a:L3\n\n> _(the excerpt is not kept in the history)_\n\n"
                 "[open the page](open:W1)"),
        ("note", "opened https://example.org/a:L3"), ("note", "not opened: not here")]
    assert opened == [("https://example.org/a:L3", None), ("/n/bpe.md", 3)]   # a web place is not a file's line
    no_source = "\n\n\n_Answered without any source: from the model's own knowledge._\n"
    assert app.two == [
        ("note", f"Resumed the conversation of {minute(1_100_000_000)} (2 questions); the next question continues it."),
        ("question", "❯ How does BPE work?"), ("page", "BPE merges pairs." + no_source),
        ("question", "❯ And SuperBPE?"), ("page", "It has a second stage." + no_source)]
    assert app.gone == [("note", "Resumed the conversation of ? (0 questions); the next question continues it.")]
    first, then = ([m["content"] for m in seen[0] if m["role"] == "user"] for seen in make.llm.seen)
    assert first[0] == "What is [b]BPE[/b]?" and first[-1].endswith("Where does it come from?")
    assert then[:2] == ["How does BPE work?", "And SuperBPE?"] and then[-1].endswith("More?")


@pytest.mark.parametrize("take", ["enter", "tab", "click"])
def test_enter_tab_and_a_click_take_the_chosen_row_alike(take, home, notes, user_home):
    """In every menu, Enter, Tab and a click take the chosen row the same way — except that Enter and a click run a
    command where Tab only completes it; /add and /remove then wait for a folder."""
    from textual.widgets import OptionList
    (user_home / "Docs").mkdir()
    write_history([("c-one", [(1_000_000_000, "What is BPE?", "Merges.", {})])])

    async def pick(app, pilot, row):
        menu = app.query_one("#commands", OptionList)
        menu.highlighted = [menu.get_option_at_index(i).id for i in range(menu.option_count)].index(row)
        if take == "click":
            menu.action_select()
        else:
            await pilot.press(take)
        await pilot.pause()
        return app.query_one("#ask", Prompt).value, menu.display

    async def steps(app, pilot):
        prompt = app.query_one("#ask", Prompt)
        app.got = {}
        for typed, row in (("/", "/help"), ("/ad", "/add"), ("/add ~/Do", "path:~/Downloads/")):
            prompt.value = typed
            await pilot.pause()
            if row.startswith("path:"):
                await pilot.press("tab")                              # Tab offers the paths
            app.got[row] = await pick(app, pilot, row)
            prompt.value = ""
        app.helped = "**Commands**" in texts(app)
        await say(app, pilot, "/resume")
        app.got["resume"] = await pick(app, pilot, "resume:c-one")
        app.resumed = texts(app).startswith("Resumed the conversation of")
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=Workspace(Index(home, embedder=FakeEmbedder())))
    runs = take != "tab"
    assert app.got == {"/help": ("" if runs else "/help ", False), "/add": ("/add ", False),
                       "path:~/Downloads/": ("/add ~/Downloads/", False), "resume": ("", False)}
    assert app.helped == runs and app.resumed


def test_tab_completes_commands_and_folders(home, notes, user_home):
    ws = Workspace(Index(home, embedder=FakeEmbedder()))

    async def steps(app, pilot):
        from textual.widgets import OptionList
        prompt, menu = app.query_one("#ask", Prompt), app.query_one("#commands", OptionList)
        prompt.value = "/ad"
        await pilot.pause()
        await pilot.press("tab")
        app.after_command_tab = prompt.value
        prompt.value = "/add ~/Dow"
        await pilot.press("tab")                                         # one match: completed in place
        app.after_one = prompt.value
        (user_home / "Docs").mkdir()
        (user_home / "My Papers").mkdir()
        prompt.value = "/add ~/My Pa"
        await pilot.press("tab")                                         # a space in the folder's name
        app.spaced = prompt.value
        for i in range(51):
            (user_home / f"d{i:02}").mkdir()
        prompt.value = "/add ~/d"
        await pilot.press("tab")
        app.many = menu.option_count
        await pilot.press("escape")
        prompt.value = "/add ~/Do"
        await pilot.press("tab")                                         # two: a menu of paths
        app.paths = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        await pilot.press("down", "enter")
        app.picked = prompt.value
        await say(app, pilot, f"/add {notes}")
        await settle(app, pilot, lambda: not app.indexing)
        ws.add(str(user_home / "Docs"))
        prompt.value = "/remove "
        await pilot.press("tab")
        app.remove_both = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        await pilot.press("escape")
        prompt.value = f"/remove {notes.resolve()}"[:-2]
        await pilot.press("tab")                                         # the typed start picks one
        app.remove_tab = prompt.value
        prompt.value = "/re"
        await pilot.pause()
        await pilot.press("enter")                                       # /remove needs a folder: wait for it
        app.after_enter = prompt.value
        prompt.value = "/"
        await pilot.pause()
        app.all_commands = menu.option_count
        await pilot.press("escape")
        app.menu_after_escape = menu.display
        app.ws = None
        prompt.value = "/remove "
        await pilot.press("tab")                                         # no workspace: nothing to offer
        app.without = (prompt.value, menu.display)
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws)
    assert app.after_command_tab == "/add " and app.after_one == "/add ~/Downloads/"
    assert app.spaced == "/add ~/My Papers/" and app.many == 50
    assert sorted(app.remove_both) == sorted(f"path:{f}/" for f in (notes.resolve(), (user_home / "Docs").resolve()))
    assert app.paths == ["path:~/Docs/", "path:~/Downloads/"] and app.picked == "/add ~/Downloads/"
    assert app.remove_tab == f"/remove {notes.resolve()}/" and app.after_enter == "/remove "
    assert app.all_commands == len(COMMANDS) and app.menu_after_escape is False
    assert app.without == ("/remove ", False)


@pytest.mark.parametrize("tool, verb, folder, keys, choice", [
    ("read", "read", False, ("enter",), "once"), ("tree", "list", True, ("down", "enter"), "session"),
    ("glob", "list the files in", False, ("down", "down", "enter"), "no"), ("grep", "search in", False, ("escape",), "stop"),
    ("other", "other", False, ("down", "enter"), "session")])
def test_the_permission_dialog_names_the_action_and_the_folder(tool, verb, folder, keys, choice, tmp_path):
    """It says what the agent wants to do and to what; "for this session" names the folder (the path itself when it
    is one); each row gives its answer and Esc stops the question."""
    from textual.app import App
    from textual.widgets import Static

    file = tmp_path / "Downloads" / "paper.md"
    file.parent.mkdir()
    file.write_text("x")
    path = file.parent if folder else file
    got = {}

    class Host(App):
        def on_mount(self):
            self.push_screen(Permission(tool, path), lambda answer: got.setdefault("choice", answer))

    async def run():
        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            got["text"] = str(app.screen.query_one("#permission > Static").render())
            got["hint"] = [str(w.render()) for w in app.screen.query("#permission > .hint")]
            got["options"] = [str(app.screen.query_one("#choices").get_option_at_index(i).prompt) for i in range(3)]
            await pilot.press(*keys)
            await pilot.pause()
    import asyncio
    asyncio.run(run())
    assert got["text"] == f"peresearch wants to {verb} something outside your declared folders:\n  {path}"
    assert got["hint"] == ["esc stops the question"]
    assert got["options"] == ["Yes, this once", f"Yes, and everything in {file.parent} for this session", "No"]
    assert got["choice"] == choice


def test_the_agent_waits_for_the_permission_answer_and_gets_no_when_the_interface_closes(home, notes, tmp_path):
    """Asked from the agent's thread: the status says it waits; the answer comes back; Esc refuses and stops the
    question; closing the interface with the dialog open refuses instead of leaving the agent waiting forever."""
    class Stoppable:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    async def ask(app, pilot):
        thread = threading.Thread(target=lambda: app.answers.append(app.permission("read", tmp_path / "a.md")))
        thread.start()
        await settle(app, pilot, lambda: isinstance(app.screen, Permission))
        app.waiting.append(app.doing)
        return thread

    async def steps(app, pilot):
        app.agent, app.answers, app.waiting = Stoppable(), [], []
        for keys in (("down", "enter"), ("escape",)):
            thread = await ask(app, pilot)
            await pilot.press(*keys)
            await settle(app, pilot, lambda: not thread.is_alive())
        app.stopped = app.agent.cancelled
        app.open_thread = await ask(app, pilot)
        app.exit()
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    app.open_thread.join(5)
    assert app.answers == ["session", "no", "no"] and app.stopped
    assert app.waiting == ["Waiting for your permission"] * 3


def test_a_source_is_shown_with_its_place_and_excerpt_and_sources_lists_them_all(home, notes, monkeypatch):
    """A clicked citation shows where the source is, its title, the first 600 characters as a quote and a link that
    opens it (a page or a file); /sources shows every source of the last answer the same way, without the links."""
    from textual.widgets import Markdown
    from peresearch.tui import Page
    opened = []
    monkeypatch.setattr(tui, "open_outside", lambda target, line=None: opened.append((target, line)))
    passage = "\n".join(f"line {i} of a long passage" for i in range(40))
    answer = Answer("See [N1] and [W1].", {"N1": Source("N1", "file", "/n/bpe.md:L3-42", "bpe.md", passage),
                                           "W1": Source("W1", "web", "https://e.org/w", "", "short\nweb")}, Check(), 1)

    async def steps(app, pilot):
        await pilot.pause()
        app.last = answer
        app.show_answer(answer)
        await pilot.pause()
        page = app.query(Page).last()
        for href in ("source:N1", "source:W1"):
            page.post_message(Markdown.LinkClicked(page, href))
            await pilot.pause()
        n1 = app.query(Page).nodes[1]
        n1.post_message(Markdown.LinkClicked(n1, "open:N1"))
        await pilot.pause()
        await say(app, pilot, "/sources")
        listing = app.query(Page).last()
        listing.post_message(Markdown.LinkClicked(listing, "source:N1"))     # its ids are not links to anything
        await pilot.pause()
        app.log_after = shown_log(app)[1:]
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    quoted = passage[:600].replace("\n", "\n> ")
    assert app.log_after == [
        ("page", f"**N1** /n/bpe.md:L3-42 — bpe.md\n\n> {quoted}\n\n[open the file](open:N1)"),
        ("page", "**W1** https://e.org/w\n\n> short\n> web\n\n[open the page](open:W1)"),
        ("note", "opened /n/bpe.md"),
        ("page", f"**`N1`** /n/bpe.md:L3-42\n\n> {quoted}\n\n**`W1`** https://e.org/w\n\n> short\n> web"),
        ("note", "N1: no such source")]
    assert opened == [("/n/bpe.md", 3)]


def test_a_clicked_link_shows_a_source_or_opens_only_what_is_safe(home, notes, monkeypatch):
    from textual.widgets import Markdown
    opened = []
    monkeypatch.setattr(tui, "open_outside", lambda target, line=None: opened.append((target, line)) or
                        ("refused here" if target.startswith("https://bad") else None))

    def make(on_event):
        a, llm = agent(home, notes, Reply("", [ToolCall("c1", "read", {"path": "bpe.md", "start": 2, "end": 4})]),
                       Reply("", [ToolCall("c2", "gaps", {"have": [], "missing": ["x"]})]),
                       Reply("", [ToolCall("c3", "web_search", {"query": "bpe"})]),
                       Reply("BPE merges pairs [N1] and has a history [W1]."))
        return a

    async def steps(app, pilot):
        app.system_opened = []
        app.open_url = lambda url, **kw: app.system_opened.append(url)       # Textual's own opener: never reached
        await say(app, pilot, "How does BPE work?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        page = app.pages[-1][0]
        for href in ("source:N1", "source:N9", "open:N1", "open:W1", "https://ok.example/a", "https://bad.example",
                     "mailto:a@b.c"):
            page.post_message(Markdown.LinkClicked(page, href))
            await pilot.pause()
        app.after = notes_of(app)
        app.shown_source = texts(app)
    app = tui_run(make, steps)
    path = str(notes / "bpe.md")
    assert opened == [(path, 2), ("https://example.org/bpe", None), ("https://ok.example/a", None),
                      ("https://bad.example", None)]
    assert app.after == ["N9: no such source", f"opened {path}", "opened https://example.org/bpe",
                         "opened https://ok.example/a", "not opened: refused here",
                         "not opened: only web links (http, https) and your own files open"]
    assert app.system_opened == []
    assert f"**N1** {path}:L2-4 — bpe.md" in app.shown_source and "[open the file](open:N1)" in app.shown_source
