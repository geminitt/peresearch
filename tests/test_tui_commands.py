"""Application action and dialog contracts, driven with Textual's pilot.

State, access, session and settings actions remain covered through the shared controller. The public grouped
window routing and transcript isolation are exercised by test_tui_windows.py with actual input events.
"""

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from rich.cells import cell_len

from peresearch import settings, tui
from peresearch.agent import Agent, Answer, Check, Limits, history_file
from peresearch.tools import Source
from peresearch.llm import Reply, ToolCall
from peresearch.tui import COMMANDS, KEYS, Permission, Prompt
from peresearch.workspace import Workspace
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import ScriptedLLM, agent, notes, settle, texts, toolbox, tui_run, user_home


async def say(app, pilot, text):
    if text.startswith("/"):
        app.action_command(text)
        await pilot.pause()
        return
    app.query_one("#ask", Prompt).value = text
    await pilot.press("enter")
    await pilot.pause()


def notes_of(app):
    from textual.widgets import Static
    return [str(w.render()) for w in app.query("#log > .note")]


def test_every_command_answers_when_there_is_nothing_to_act_on(home, notes):
    async def steps(app, pilot):
        await pilot.pause()
        for text in ("/folders", "/add ~/x", "/remove ~/x", "/index", "/retry", "/sources", "/resume", "/bogus [b]x[/b]",
                     "/status", "/status data_type"):
            await say(app, pilot, text)
        await say(app, pilot, "/help")
        await say(app, pilot, "/keys")
        app.before_exit = texts(app)
        app.code_before = app.return_code
        await say(app, pilot, "/exit")
        app.code_after = app.return_code
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=None,
                  status=lambda reason=None: f"**Status** asked for {reason}")
    assert "**Status** asked for None" in app.before_exit and "**Status** asked for data_type" in app.before_exit
    typed = ["/folders", "/add ~/x", "/remove ~/x", "/index", "/retry", "/sources", "/resume", "/bogus [b]x[/b]", "/status",
             "/status data_type", "/help", "/keys"]
    assert not any(x.startswith("❯ /") for x in app.before_exit.splitlines())
    lines = app.before_exit.splitlines()
    assert "no folders declared: /folders, then a" in lines
    assert notes_of_shown(app.before_exit) == [
        "no folders declared: /folders, then a", "folders cannot be changed here", "folders cannot be changed here",
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
    update, labels, failures = ws.update, [], []
    def watched(stop=None, progress=None):                                   # what the status line says meanwhile
        assert stop is not None and stop() is False                          # quitting can stop it; nobody has
        assert progress is not None                                          # and it says how far it is
        labels.append(ws.app.indexing)
        if failures:
            raise failures.pop(0)
        return update(stop=stop, progress=progress)
    ws.update = watched

    async def steps(app, pilot):
        ws.app = app
        await settle(app, pilot, lambda: not app.indexing)
        app.startup_notice = app.flash
        for text in ("/add", f"/add {tmp_path / 'nope'}", f"/add {notes}"):
            await say(app, pilot, text)
        await settle(app, pilot, lambda: not app.indexing)
        await say(app, pilot, "/index")                                      # asked for: said even when nothing changed
        await settle(app, pilot, lambda: not app.indexing)
        app.indexing = "Indexing"                                            # an update still running
        await say(app, pilot, "/index")
        app.indexing = ""
        ws.add(str(more))
        app.declared = ws.folders()
        await say(app, pilot, "/folders")
        await say(app, pilot, f"/remove {tmp_path}")
        await settle(app, pilot, lambda: not app.indexing)                  # else /index can find it still busy
        await say(app, pilot, "/index")
        await settle(app, pilot, lambda: not app.indexing)
        failures.extend([OSError("disk full"), AssertionError("chunks and vectors differ")])   # said, then usable
        for _ in failures[:]:
            await say(app, pilot, "/index")
            await settle(app, pilot, lambda: not app.indexing)
        await say(app, pilot, f"/remove {notes}")
        await settle(app, pilot, lambda: not app.indexing)
        await say(app, pilot, "/remove")
        app.all_notes = notes_of(app)
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws, index_on_start=True)
    n = app.all_notes
    assert app.startup_notice == "No folders yet: /folders lets peresearch read one."
    assert n[0:3] == ["usage: /add <folder>", f"not a folder: {tmp_path / 'nope'}", f"Added {notes.resolve()}"]
    assert "wait for the current work to finish (a question: esc interrupts it)" in n
    assert len(app.declared) == 2 and "\n".join(map(str, app.declared)) in n      # one folder per line
    assert f"{tmp_path} was not a declared folder" in n
    assert labels == ["Indexing the new folder", "Indexing", "Dropping its chunks from the index", "Indexing",
                      "Indexing", "Indexing", "Dropping its chunks from the index"]
    assert "indexing failed: OSError: disk full" in n
    assert any(re.fullmatch(r"indexing failed: an internal check failed at peresearch/tui\.py:\d+: "   # its own
                            r"chunks and vectors differ", x) for x in n)                    # deepest frame
    one = [x for x in n if x.startswith("Index")]                        # one line each, never a list of paths
    assert re.fullmatch(r"Index updated: 3 added, 0 changed, 0 removed · \d+ chunks from 3 files", one[0])
    assert re.fullmatch(r"Index up to date · \d+ chunks from 3 files", one[1])
    assert f"Removed {notes}" in n and n[-1] == "usage: /remove <folder>"


def test_busy_guards_retry_sources_new_clear_and_setup(home, notes):
    gate = threading.Event()

    class Slow(ScriptedLLM):
        def answer(self, messages, tools=None, thinking=None, stop=None):
            while not gate.is_set():
                if stop and stop():
                    from peresearch.llm import Interrupted
                    raise Interrupted("stopped")
                time.sleep(0.02)
            return super().answer(messages, tools, thinking, stop)
    llm = Slow(Reply("", [ToolCall("c", "read", {"path": "bpe.md"})]), Reply("BPE merges pairs [N1]."),
               Reply("again [N1]."), Reply("fresh."))

    async def steps(app, pilot):
        await say(app, pilot, "How does BPE work?")
        for text in ("/new", "/clear", "/retry", "/resume", "another question"):   # /setup: a window now
            await say(app, pilot, text)
        app.busy_notes = list(notes_of(app))
        gate.set()
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.first = app.last
        app.last = None
        await say(app, pilot, "/retry")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        await say(app, pilot, "/clear")                                # /new under the name Claude Code uses
        app.after_new = texts(app)
        await say(app, pilot, "something else")
        await settle(app, pilot, lambda: not app.busy and len(llm.seen) >= 4)
    app = tui_run(lambda e: Agent(llm, toolbox(home, notes)), steps)
    assert app.busy_notes == ["wait for the current question to finish, or press esc"] * 4 + [
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
            for w in app.query("#log > *") if w.id != "welcome"]


def test_resume_shows_a_conversation_as_it_was_and_continues_it(home, notes, monkeypatch):
    """/resume lists the 50 most recent conversations, each on a line that fits the menu; the chosen one is shown as
    it was — when it started, how many questions, each question as typed, each answer with its sources as links
    (legacy excerpts require a new read) — before the agent exists or after, and the next question continues it."""
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
        app.link_notices = []
        notice = app.link_notice
        def recorded_notice(text):
            app.link_notices.append(text)
            notice(text)
        app.link_notice = recorded_notice
        from textual.widgets import Markdown, OptionList
        await say(app, pilot, "/resume")
        menu = app.query_one("#commands", OptionList)
        options = [menu.get_option_at_index(i) for i in range(menu.option_count)]
        app.listed = [o.id for o in options]
        app.long_row = cell_len(str(options[0].prompt)), app.size.width
        app.first_chosen = menu.highlighted                          # the newest is chosen as the menu opens
        await pick(app, pilot, "c-old", "enter")                     # the box is empty: Enter takes the row
        app.old = shown_log(app)
        page = app.query(Page).last()
        app.clicked = []
        for href in ("source:N1", "source:W1"):
            page.post_message(Markdown.LinkClicked(page, href))
            await pilot.pause()
            preview = app.screen.query_one(Markdown)
            app.clicked.append(("page", preview.source))
            if href == "source:N1":
                first_preview = preview
            await pilot.press("escape")
            await pilot.pause()
        page.post_message(Markdown.LinkClicked(page, "open:W1"))
        page.post_message(Markdown.LinkClicked(page, "open:N1"))
        await pilot.pause()
        app.clicked += [("notice", text) for text in app.link_notices]
        assert shown_log(app) == app.old
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
    assert app.long_row[0] == app.long_row[1] - 2                       # cut to the menu's inner width (its padding)
    assert app.first_chosen == 0                                         # Enter at once takes the newest
    assert app.old == [
        ("note", f"Resumed the conversation of {minute(1_000_000_000)} (1 question); the next question continues it."),
        ("question", "❯ What is [b]BPE[/b]?"),
        ("page", "See [N1](source:N1) and [W1](source:W1).\n<sub>kept</sub>\n\n**Sources read**\n"
                 "- [N1](source:N1) /n/bpe.md:L3-5 — Evidence identity unknown; read again\n"
                 "- [W1](source:W1) https://example.org/a:L3 — Evidence identity unknown; read again\n"
                 "\n> **Check:** citation evidence unavailable; read again: N1, W1\n")]
    assert app.clicked == [
        ("page", "**N1** /n/bpe.md:L3-5 — Evidence identity unknown; read again\n\n"
                 "> Evidence identity is unknown. Read this source again before citing it.\n\n[open the file](open:N1)"),
        ("page", "**W1** https://example.org/a:L3 — Evidence identity unknown; read again\n\n"
                 "> Evidence identity is unknown. Read this source again before citing it.\n\n"
                 "[open the page](open:W1)"),
        ("notice", "opened https://example.org/a:L3"), ("notice", "not opened: not here")]
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
    command where Tab only completes it; /add and /remove then wait for a folder; an @path completes the last @."""
    from textual.widgets import OptionList
    (user_home / "Docs").mkdir()
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))
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
        app.got_help = False
        for typed, row in (("/", "/help"), ("/fo", "/folders"), ("/add ~/Do", "path:~/Downloads/"),
                           ("compare @bpe.md and @", "mention:food.md")):
            prompt.value = typed
            await pilot.pause()
            if row.startswith(("path:", "mention:")):
                await pilot.press("tab")                              # Tab offers the paths
            app.got[row] = await pick(app, pilot, row)
            from textual.screen import ModalScreen
            if isinstance(app.screen, ModalScreen):
                from textual.widgets import Markdown
                if row == "/help":
                    app.got_help = any("/folders" in w.source for w in app.screen.query("Markdown, .columns"))
                else:
                    assert app.screen.id == "folders-window"
                await pilot.press("escape")
                await pilot.pause()
            prompt.value = ""
        app.helped = app.got_help
        await say(app, pilot, "/resume")
        app.got["resume"] = await pick(app, pilot, "resume:c-one")
        app.resumed = texts(app).startswith("Resumed the conversation of")
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws)
    runs = take != "tab"
    assert app.got == {"/help": ("" if runs else "/help ", False), "/folders": ("" if runs else "/folders ", False),
                       "path:~/Downloads/": ("/add ~/Downloads/", False),
                       "mention:food.md": ("compare @bpe.md and @food.md", False), "resume": ("", False)}
    assert app.helped == runs and app.resumed


def test_tab_completes_commands_and_folders(home, notes, user_home):
    ws = Workspace(Index(home, embedder=FakeEmbedder()))

    async def steps(app, pilot):
        from textual.widgets import OptionList
        prompt, menu = app.query_one("#ask", Prompt), app.query_one("#commands", OptionList)
        prompt.value = "/fo"
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
        prompt.value = "/fo"
        await pilot.pause()
        await pilot.press("enter")                                       # opens the folder group
        app.after_enter = prompt.value
        assert app.screen.id == "folders-window"
        await pilot.press("escape")
        await pilot.pause()
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
    assert app.after_command_tab == "/folders " and app.after_one == "/add ~/Downloads/"
    assert app.spaced == "/add ~/My Papers/" and app.many == 50
    assert sorted(app.remove_both) == sorted(f"path:{f}/" for f in (notes.resolve(), (user_home / "Docs").resolve()))
    assert app.paths == ["path:~/Docs/", "path:~/Downloads/"] and app.picked == "/add ~/Downloads/"
    assert app.remove_tab == f"/remove {notes.resolve()}/" and app.after_enter == ""
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
            got["options"] = [str(app.screen.query_one("#choices").get_option_at_index(i).prompt).strip()
                              for i in range(3)]                       # each row after the pointer's gutter
            await pilot.press(*keys)
            await pilot.pause()
    import asyncio
    asyncio.run(run())
    assert got["text"] == f"peresearch wants to {verb} something outside your declared folders:\n  {tui.home_path(str(path))}"
    assert len(got["hint"]) == 1 and got["hint"][0].endswith("esc stops the question")
    assert got["options"] == ["Yes, this once", f"Yes, and everything in {tui.home_path(str(file.parent))} for this session",
                              "No"]
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
        app.link_notices = []
        notice = app.link_notice
        def recorded_notice(text):
            app.link_notices.append(text)
            notice(text)
        app.link_notice = recorded_notice
        await pilot.pause()
        app.last = answer
        app.show_answer(answer)
        await pilot.pause()
        page = app.query(Page).last()
        app.previews = []
        for href in ("source:N1", "source:W1"):
            page.post_message(Markdown.LinkClicked(page, href))
            await pilot.pause()
            preview = app.screen.query_one(Markdown)
            app.previews.append(("page", preview.source))
            if href == "source:N1":
                preview.post_message(Markdown.LinkClicked(preview, "open:N1"))
                await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
        await say(app, pilot, "/sources")
        listing = app.query(Page).last()
        listing.post_message(Markdown.LinkClicked(listing, "source:N1"))     # its ids are not links to anything
        await pilot.pause()
        app.log_after = app.previews + [("notice", app.link_notices[0])] + shown_log(app)[1:] + [("notice", app.link_notices[1])]
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    quoted = passage[:600].replace("\n", "\n> ")
    assert app.log_after == [
        ("page", f"**N1** /n/bpe.md:L3-42 — bpe.md\n\n> {quoted}\n\n[open the file](open:N1)"),
        ("page", "**W1** https://e.org/w\n\n> short\n> web\n\n[open the page](open:W1)"),
        ("notice", "opened /n/bpe.md"),
        ("page", f"**`N1`** /n/bpe.md:L3-42\n\n> {quoted}\n\n**`W1`** https://e.org/w\n\n> short\n> web"),
        ("notice", "N1: no such source")]
    assert opened == [("/n/bpe.md", 3)]


def test_a_clicked_link_shows_a_source_or_opens_only_what_is_safe(home, notes, monkeypatch):
    from textual.widgets import Markdown
    opened = []
    monkeypatch.setattr(tui, "open_outside", lambda target, line=None: opened.append((target, line)) or
                        ("refused here" if target.startswith("https://bad") else None))

    def make(on_event):
        a, llm = agent(home, notes, Reply("", [ToolCall("c1", "read", {"path": "bpe.md", "start": 2, "end": 4})]),
                       Reply("", [ToolCall("c3", "web_search", {"query": "bpe"})]),
                       Reply("BPE merges pairs [N1] and has a history [W1]."))
        return a

    async def steps(app, pilot):
        app.link_notices = []
        notice = app.link_notice
        def recorded_notice(text):
            app.link_notices.append(text)
            notice(text)
        app.link_notice = recorded_notice
        app.preview_body = ""
        app.system_opened = []
        app.open_url = lambda url, **kw: app.system_opened.append(url)       # Textual's own opener: never reached
        await say(app, pilot, "How does BPE work?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        page = app.pages[-1][0]
        for href in ("source:N1", "source:N9", "open:N1", "open:W1", "https://ok.example/a", "https://bad.example",
                     "mailto:a@b.c"):
            page.post_message(Markdown.LinkClicked(page, href))
            await pilot.pause()
            if href == "source:N1":
                app.preview_body = app.screen.query_one(Markdown).source
                await pilot.press("escape")
                await pilot.pause()
        app.after = app.link_notices
        app.shown_source = app.preview_body
    app = tui_run(make, steps)
    path = str(notes / "bpe.md")
    assert opened == [(path, 2), ("https://example.org/bpe", None), ("https://ok.example/a", None),
                      ("https://bad.example", None)]
    assert app.after == ["N9: no such source", f"opened {path}", "opened https://example.org/bpe",
                         "opened https://ok.example/a", "not opened: refused here",
                         "not opened: only web links (http, https) and your own files open"]
    assert app.system_opened == []
    assert f"**N1** {path}:L2-4 — bpe.md" in app.shown_source and "[open the file](open:N1)" in app.shown_source


def test_local_turns_the_web_off_for_the_session_and_back_on(home, notes):
    """/local: the web off for this session (the toolbox offers and runs no web tool, the model is told why), shown on
    the idle status line; it holds for an agent made later, and /local again turns the web back on."""
    made = []

    def make(event):
        a, llm = agent(home, notes, Reply("files only [N1]"), Reply("with the web"))
        made.append((a, llm))
        return a

    async def steps(app, pilot):
        await pilot.pause()
        await say(app, pilot, "/local")                       # before any agent exists
        await pilot.pause(0.3)
        app.status_off = str(app.status.render())
        await say(app, pilot, "What do my notes say about BPE?")
        await settle(app, pilot, lambda: not app.busy)
        app.during = made[0][0].toolbox.local_only
        await say(app, pilot, "/local")
        await pilot.pause(0.3)
        app.status_on = str(app.status.render())
        app.after = made[0][0].toolbox.local_only
        await say(app, pilot, "and the web?")
        await settle(app, pilot, lambda: not app.busy)
        app.shown = texts(app)
    app = tui_run(make, steps, workspace=None)
    assert app.status_off == "local only · web off" and app.status_on == ""
    assert app.during is True and app.after is False
    said = [x for x in app.shown.splitlines() if x.startswith(("Local only", "The web is on"))]
    assert said == ["Local only: the web is off for this session (/local turns it back on).", "The web is on again."]
    llm = made[0][1]
    first_tools = [t["function"]["name"] for t in llm.seen[0][1]]
    later_tools = [t["function"]["name"] for t in llm.seen[1][1]]
    assert "web_search" not in first_tools and "fetch" not in first_tools and "overview" in first_tools
    assert "web_search" in later_tools
    assert "turned the web OFF" in llm.seen[0][0][0]["content"] and "turned the web OFF" not in llm.seen[1][0][0]["content"]
    assert "/local" in tui.COMMANDS


def test_export_saves_this_sessions_questions_answers_and_sources_as_markdown(home, notes, tmp_path, monkeypatch):
    """/export [file or folder]: this session's questions and answers, each with its sources (id, place, title; not
    their text), as Markdown; a folder or nothing gets a dated name (the current folder); an existing file is never
    overwritten and a protected place is refused; nothing asked yet: nothing written."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.chdir(out_dir)
    (out_dir / "taken.md").write_text("mine\n")
    ans = Answer("BPE merges pairs [N1].", {"N1": Source("N1", "file", "/n/bpe.md:L3-5", "Tokenization", "secret text"),
                                           "W1": Source("W1", "web", "https://example.org/bpe", "BPE explained", "x"),
                                           "N2": Source("N2", "file", "/n/x.md:L1", "", "x")},
                 Check([], []), 1, 3, 2)
    (out_dir / ".ssh").mkdir()
    (out_dir / "link.md").symlink_to(out_dir / ".ssh" / "y.md")      # a new name, leading into a protected place
    second = Answer("No web needed.", {}, Check([], []), 1, 3, 2)

    class Fake:
        def __init__(self):
            self.answers = [ans, second]

        def ask(self, question):
            return self.answers.pop(0)

    async def steps(app, pilot):
        await pilot.pause()
        await say(app, pilot, "/export")                          # nothing yet
        await say(app, pilot, "What is BPE?")
        await settle(app, pilot, lambda: not app.busy)
        await say(app, pilot, "And the web?")
        await settle(app, pilot, lambda: not app.busy)
        for arg in ("", str(out_dir / "mine.md"), "taken.md", str(out_dir / "sub"), str(out_dir / ".ssh" / "x.md"),
                    "link.md", str(out_dir / "nosuch" / "x.md")):
            await say(app, pilot, f"/export {arg}".strip())
        await say(app, pilot, "/local")                           # an agent without a toolbox: only the flag
        app.query_one("#ask", Prompt).value = "@x"
        await pilot.press("tab")                                  # no workspace: nothing to complete, no failure
        app.shown = texts(app)
    app = tui_run(lambda e: Fake(), steps, workspace=None, project="thesis")
    lines = app.shown.splitlines()
    said = [x for x in lines if x.startswith(("nothing to export", "Saved to", "not saved"))]
    dated = sorted(p for p in out_dir.iterdir() if p.name.startswith("peresearch-thesis-"))
    assert len(dated) == 1 and re.fullmatch(r"peresearch-thesis-\d{4}-\d{2}-\d{2}-\d{6}\.md", dated[0].name)
    assert said == ["nothing to export yet: ask a question first", f"Saved to {dated[0]}", f"Saved to {out_dir / 'mine.md'}",
                    f"not saved: {out_dir / 'taken.md'} exists (give another name)",
                    f"not saved: {out_dir / 'sub'} is not a folder",
                    f"not saved: {out_dir / '.ssh' / 'x.md'} is a protected place",   # a temporary one
                    f"not saved: {out_dir / 'link.md'} is a protected place",
                    f"not saved: {out_dir / 'nosuch' / 'x.md'}: No such file or directory"]
    assert "Local only: the web is off for this session (/local turns it back on)." in app.shown
    want = ("# peresearch — thesis\n\n"
            "## What is BPE?\n\nBPE merges pairs [N1].\n\n**Sources**\n\n"
            "- **N1** /n/bpe.md:L3-5 — Tokenization\n- **W1** https://example.org/bpe — BPE explained\n"
            "- **N2** /n/x.md:L1\n\n"
            "## And the web?\n\nNo web needed.\n")
    assert dated[0].read_text(encoding="utf-8") == want and (out_dir / "mine.md").read_text(encoding="utf-8") == want
    assert (out_dir / "taken.md").read_text() == "mine\n" and "secret text" not in want
    assert list((out_dir / ".ssh").iterdir()) == []
    assert tui.Chat(lambda e: None, workspace=Workspace()).index_on_start is True     # opening checks the folders
    assert tui.Chat(lambda e: None).index_on_start is False                           # unless there are none
    assert "/sessions" in tui.COMMANDS and "/export" not in tui.COMMANDS


def test_a_path_named_with_at_is_completed_by_tab_and_handed_to_the_model(home, notes, user_home, monkeypatch):
    """@path: Tab completes it from the declared folders (files and folders; hidden ones only when typed), and a
    question naming existing places tells the model to read them (files with read, folders with tree), so every
    rule of those tools still holds; a name that is no place, a protected one, or an @ inside a word, is left as
    typed."""
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))
    (notes / "sub").mkdir()
    (notes / "sub" / "deep.md").write_text("deep\n")
    (user_home / "home-note.md").write_text("x\n")
    (user_home / "@inbox").mkdir()
    (notes / "many").mkdir()
    for i in range(51):
        (notes / "many" / f"f{i:02}.md").write_text("x\n")
    assert tui.with_mentions("check @.env and @id_rsa", [notes]) == "check @.env and @id_rsa"   # protected: no note
    assert tui.with_mentions("see @nothere.md", [notes]) == "see @nothere.md"                    # no such place
    (notes / "innocent.md").symlink_to(notes / ".env")                   # a plain name leading to a protected file
    assert tui.with_mentions("see @innocent.md", [notes]) == "see @innocent.md"
    asked = []

    class Fake:
        def ask(self, question):
            asked.append(question)
            return Answer("ok", {}, Check([], []), 1, 3, 2)

    async def steps(app, pilot):
        from textual.widgets import OptionList
        prompt, menu = app.query_one("#ask", Prompt), app.query_one("#commands", OptionList)
        prompt.value = "what is in @bp"
        await pilot.press("tab")
        app.one = prompt.value
        prompt.value = "compare @su"
        await pilot.press("tab")
        app.folder = prompt.value
        prompt.value = "look at @"
        await pilot.press("tab")
        app.listed = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        await pilot.press("down", "enter")
        app.picked = prompt.value
        prompt.value = "and @sub/d"
        await pilot.press("tab")
        app.deeper = prompt.value
        prompt.value = "@bp"                                     # an @ at the very start
        await pilot.press("tab")
        app.first_word = prompt.value
        prompt.value = "plain words"                             # no @, no command: Tab does nothing
        await pilot.press("tab")
        app.plain = (prompt.value, menu.display)
        prompt.value = f"see @{notes}/fo"                         # an absolute path
        await pilot.press("tab")
        app.absolute = prompt.value
        prompt.value = "see @~/home-n"                            # from the home folder
        await pilot.press("tab")
        app.from_home = prompt.value
        prompt.value = "/add @in"                                 # a command's own path, though it starts with @
        await pilot.press("tab")
        app.add_at = prompt.value
        prompt.value = "@many/"
        await pilot.press("tab")
        app.many = menu.option_count
        await pilot.press("escape")
        await say(app, pilot, "summarize @bpe.md and @sub and @~/nothing.md and mail@bpe.md")
        await settle(app, pilot, lambda: not app.busy)
        app.shown = texts(app)
    monkeypatch.chdir(user_home)                                  # "/add @in": a path from the current folder
    app = tui_run(lambda e: Fake(), steps, workspace=ws)
    assert app.one == "what is in @bpe.md" and app.folder == "compare @sub/" and app.deeper == "and @sub/deep.md"
    assert app.first_word == "@bpe.md" and app.plain == ("plain words", False)
    assert app.absolute == f"see @{notes}/food.md" and app.from_home == "see @~/home-note.md"
    assert app.add_at == "/add @inbox/" and app.many == 50
    assert app.listed == ["mention:bpe.md", "mention:food.md", "mention:innocent.md", "mention:many/", "mention:sub/",
                          "mention:trap.md"]                     # a name is listed; the note is what is withheld
    assert app.picked == "look at @food.md"
    assert asked == [f"summarize @bpe.md and @sub and @~/nothing.md and mail@bpe.md\n\n(The user named with @: "
                     f"{notes / 'bpe.md'} (a file: read it with read), {notes / 'sub'} (a folder: look at it with "
                     "tree))"]
    assert "❯ summarize @bpe.md and @sub and @~/nothing.md and mail@bpe.md" in app.shown       # shown as typed


def test_a_config_broken_while_the_interface_runs_is_said_by_the_folder_commands(home, notes):
    """The user may edit config.json while peresearch chat runs: /folders, /add and /remove say what is wrong in it
    and the interface goes on."""
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))

    async def steps(app, pilot):
        await pilot.pause()
        (home / "config.json").write_text('{"roots": [}')
        for text in ("/folders", f"/add {notes}", f"/remove {notes}"):
            await say(app, pilot, text)
        app.shown = texts(app)
        app.alive = app.is_running
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws)
    said = [x for x in app.shown.splitlines() if x.startswith("config.json (")]
    assert len(said) == 3 and all("is not valid JSON (line 1, column 12)" in x for x in said) and app.alive


def test_conversations_are_named_cleared_resumed_by_name_and_their_usage_shown(home, notes):
    """The session store in the interface (storage frame): /rename names the conversation — before its first
    question too —, /clear [name] (also /new, /reset) starts a fresh one, named when a name is given; /resume <name or
    id> goes straight to it and the /resume list shows names; /usage sums model requests and tokens for this
    conversation, today and the project. A bad name or an unknown one is said, nothing changes."""
    from peresearch.agent import conversations, read_turns
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    write_history([("c-2001", [(1_000_000_000, "An old one", "ok", {})])])      # an earlier day: not today's
    with history_file("default").open("a") as f:
        for when, question, n in ((1_000_000_001, "Old?", 5), (midnight - 1800, "Late last night?", 7),
                                  (midnight, "Just after midnight?", 11)):            # the last one is today's
            f.write(json.dumps({"time": when, "conversation": "c-2001", "question": question, "answer": "ok",
                                "sources": {}, "usage": {"requests": n, "prompt_tokens": 100 * n,
                                                         "completion_tokens": 10 * n}}) + "\n")

    def make(on_event):
        a, llm = agent(home, notes, Reply("Zero."), Reply("BPE merges pairs.", prompt_tokens=1200, completion_tokens=40),
                       Reply("It has two stages.", prompt_tokens=900, completion_tokens=30),
                       Reply("Again.", prompt_tokens=10, completion_tokens=2))
        return a

    async def steps(app, pilot):
        from textual.widgets import OptionList
        await say(app, pilot, "/rename dropped")                     # a name given, then the conversation cleared:
        await say(app, pilot, "/clear")                              # the fresh one is not named so
        await say(app, pilot, "Zero?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.zero = app.agent.conversation
        await say(app, pilot, "/clear")
        app.last = None
        await say(app, pilot, "/rename   BPE   notes ")               # given, then kept until the next question
        await say(app, pilot, "How does BPE work?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.first = app.agent.conversation
        await say(app, pilot, "/usage")
        app.usage_one = shown_log(app)[-1]
        await say(app, pilot, "/clear " + "y" * 81)                  # a bad name: said, the conversation goes on
        app.kept = (app.agent.conversation, notes_of(app)[-1])
        await say(app, pilot, "/clear tokenizers")
        app.last = None
        await say(app, pilot, "And SuperBPE?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.second = app.agent.conversation
        await say(app, pilot, "/usage")
        app.usage_two = shown_log(app)[-1]
        for bad in ("/rename", "/rename " + "x" * 81, "/resume nothing like it"):
            await say(app, pilot, bad)
        app.said = notes_of(app)[-3:]
        await say(app, pilot, "/resume bpe NOTES")
        app.resumed = shown_log(app)
        app.last = None
        await say(app, pilot, "Again?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.third = app.agent.conversation
        await say(app, pilot, "/reset")
        app.reset = notes_of(app)[-1]
        await say(app, pilot, "/resume")
        menu = app.query_one("#commands", OptionList)
        app.rows = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
    app = tui_run(make, steps)
    path = history_file("default")
    assert [(c.id, c.name, c.questions) for c in conversations(path)] == [
        (app.first, "BPE notes", 2), (app.second, "tokenizers", 1), (app.zero, "", 1), ("c-2001", "", 4)]
    assert app.kept == (app.first, "a name is at most 80 characters")
    assert app.third == app.first and read_turns(path)[-1]["question"] == "Again?"
    assert app.usage_one == ("page", "**Usage** (model requests · tokens sent · tokens generated)\n"
                                     "- this conversation: 1 · 1,200 · 40\n- today, this project: 13 · 2,300 · 150\n"
                                     "- this project: 25 · 3,500 · 270")
    assert app.usage_two == ("page", "**Usage** (model requests · tokens sent · tokens generated)\n"
                                     "- this conversation: 1 · 900 · 30\n- today, this project: 14 · 3,200 · 180\n"
                                     "- this project: 26 · 4,400 · 300")
    assert app.said == ["/rename <name>: a name for this conversation (/resume finds it by name)",
                        "a name is at most 80 characters", "no conversation named or numbered “nothing like it”"]
    assert ("question", "❯ How does BPE work?") in app.resumed and ("question", "❯ And SuperBPE?") not in app.resumed
    assert app.reset == "New conversation: earlier questions are no longer sent as context."
    assert "BPE notes · How does BPE work?" in app.rows[0] and "tokenizers · And SuperBPE?" in app.rows[1]
    assert all(c in COMMANDS for c in ("/sessions", "/new", "/status"))
    assert not any(c in COMMANDS for c in ("/clear", "/reset", "/rename", "/resume", "/usage"))

    def fresh(on_event):
        return agent(home, notes, Reply("Hi."))[0]

    async def before_any_question(app, pilot):                     # no agent yet: nothing to count, a name waits
        await say(app, pilot, "/usage")
        app.usage_none = shown_log(app)[-1]
        await say(app, pilot, "/rename [/b] odd [b]name")            # what the user typed, shown as typed
        await say(app, pilot, "/resume [/x]")
        app.odd = notes_of(app)[-2:]
        await say(app, pilot, "/rename for another")                 # a waiting name, then another conversation:
        await say(app, pilot, "/resume tokenizers")                  # it was not for that one
        await say(app, pilot, "Hello?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    early = tui_run(fresh, before_any_question)

    async def named_first(app, pilot):                               # a name given before the first question
        await say(app, pilot, "/rename early name")
        await say(app, pilot, "Hello?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.made = app.agent.conversation
    early = (early, tui_run(fresh, named_first))
    early, later = early
    early.made = later.made
    assert early.usage_none[1].startswith("**Usage** (model requests · tokens sent · tokens generated)\n"
                                          "- this conversation: 0 · 0 · 0\n")
    assert next(c for c in conversations(path) if c.id == early.made).name == "early name"
    assert early.odd == ["This conversation is named “[/b] odd [b]name”.", "no conversation named or numbered “[/x]”"]
    assert next(c for c in conversations(path) if c.id == app.second).name == "tokenizers"   # not "for another"


def test_context_shows_what_the_next_request_holds(home, notes):
    """/context: the next question's request before its own messages, part by part, in characters and estimated
    tokens, against the model's context (or said unknown); before the first question there is no agent to measure."""
    def make(on_event):
        a, llm = agent(home, notes, Reply("BPE merges pairs.", prompt_tokens=2_000))
        llm.context_length = lambda: 16_384
        return a

    async def steps(app, pilot):
        await say(app, pilot, "/context")
        app.before = notes_of(app)[-1]
        await say(app, pilot, "How does BPE work?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        await say(app, pilot, "/context")
        app.page = shown_log(app)[-1]
        app.report = app.agent.context_report()
    app = tui_run(make, steps)
    assert app.before == "/context measures the model's requests: ask a question first"
    r = app.report
    lines = [f"- {name}: {n:,} characters ≈ {round(n / r['ratio']):,} tokens" for name, n in r["parts"]]
    assert app.page == ("page", f"**Context** (the next question's request before its own messages; "
                                f"{r['ratio']:.1f} characters a token, as last measured)\n" + "\n".join(lines) +
                        f"\n- in all: ≈ {r['tokens']:,} tokens of the model's 16,384 ({r['tokens'] / 16_384:.0%})")
    assert "/status" in COMMANDS and "/context" not in COMMANDS
    unknown = tui.Chat.context_text({"parts": [("system prompt", 2_600)], "ratio": 2.6, "tokens": 1_000, "context": None})
    assert unknown.endswith("- in all: ≈ 1,000 tokens (the model's context is not known: set PERESEARCH_LLM_CONTEXT)")


def test_config_shows_sets_and_resets_the_harness_settings(home, notes, monkeypatch):
    """/config lists in full only what is set (its value and where it comes from: the environment or the file); the
    defaults in use on one line — what fits without the user (step 4, 2026-10-08) —, rerank-depth only when a rerank
    endpoint is set; /config <name> <value> checks and saves one (the next question gets an agent built on it, in the
    same conversation); /config <name> resets it; a bad value or name is said and nothing changes; a setting the
    environment holds is named, since it wins."""
    for name in ("PERESEARCH_STEPS", "PERESEARCH_WEB_SEARCHES", "PERESEARCH_PAGES", "PERESEARCH_SECONDS",
                 "PERESEARCH_LLM_THINKING", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_RERANK_DEPTH", "PERESEARCH_RERANK_URL"):
        monkeypatch.delenv(name, raising=False)
    made = []

    def make(on_event):
        a, llm = agent(home, notes, Reply("One."), Reply("Two."), limits=Limits.from_settings())   # as make_agent
        made.append(a)
        return a

    async def steps(app, pilot):
        await say(app, pilot, "Hello?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        first = app.agent.conversation
        await say(app, pilot, "/config")
        app.listing = shown_log(app)[-1]
        await say(app, pilot, "/config steps 20")
        await say(app, pilot, "/config pages 99")
        await say(app, pilot, "/config colour red")
        app.said = notes_of(app)[-3:]
        app.last = None
        await say(app, pilot, "Again?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.same = app.agent.conversation == first
        await say(app, pilot, "/config steps")
        monkeypatch.setenv("PERESEARCH_PAGES", "3")
        await say(app, pilot, "/config pages 4")
        app.reset = notes_of(app)[-2:]
        app.busy = True                                   # a question running: its agent is not swapped under it
        await say(app, pilot, "/config steps 5")
        app.busy = False
        app.refused = notes_of(app)[-1]
        await say(app, pilot, "/config colour")
        await say(app, pilot, "/config thinking OFF")
        await say(app, pilot, "/config")
        app.later = notes_of(app)[-2:], shown_log(app)[-1][1].split("\n")
    app = tui_run(make, steps)
    assert app.listing == ("page", "**Settings** (`/config <name> <value>` sets one, `/config <name>` resets it; "
                                   "they apply from the next question)\nNone set: each has its default — steps no limit · "
                                   "web-searches no limit · pages no limit · seconds 900 · thinking the server's · "
                                   "context the server's")
    assert app.said == ["steps is now 20; it applies from the next question.",
                        "pages must be a whole number from 0 to 30 (got '99')",
                        "unknown setting colour: steps, web-searches, pages, seconds, thinking, context, rerank-depth"]
    assert len(made) == 2 and made[1].limits.steps == 20 and app.same          # rebuilt, same conversation
    assert app.reset == ["steps is back to its default (no limit).",
                         "pages is now 4; it applies from the next question. PERESEARCH_PAGES is set in the "
                         "environment, which wins over the file."]
    assert app.refused == "wait for the current question to finish, or press esc"
    assert app.later[0] == [
        "unknown setting colour: steps, web-searches, pages, seconds, thinking, context, rerank-depth",
        "thinking is now off; it applies from the next question."]
    assert app.later[1][1:] == ["- pages: 3 (environment) — web pages read per question",
                                "- thinking: off (saved) — the model's reasoning: on or off",
                                "Defaults: steps no limit · web-searches no limit · seconds 900 · context the "
                                "server's"]
    assert settings.saved() == {"PERESEARCH_PAGES": "4", "PERESEARCH_LLM_THINKING": "off"} and "/settings" in COMMANDS and "/config" not in COMMANDS


def test_an_agent_the_interface_replaces_has_its_search_server_stopped():
    from types import SimpleNamespace
    closed = []
    app = tui.Chat(lambda event: None)
    app.agent = SimpleNamespace(conversation="c1", toolbox=SimpleNamespace(web=SimpleNamespace(
        close=lambda: closed.append(1))))
    app._retire_agent()
    assert closed == [1] and app.agent is None and app.pending_resume == "c1"
    app._retire_agent()                                                  # none to retire: nothing happens
    assert closed == [1]


def test_the_rerank_depth_is_listed_once_a_rerank_endpoint_is_set(home, monkeypatch):
    for name in ("PERESEARCH_STEPS", "PERESEARCH_WEB_SEARCHES", "PERESEARCH_PAGES", "PERESEARCH_SECONDS",
                 "PERESEARCH_LLM_THINKING", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_RERANK_DEPTH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PERESEARCH_RERANK_URL", "http://localhost:8081/v1")
    assert tui.config_listing().endswith("context the server's · rerank-depth 30")
