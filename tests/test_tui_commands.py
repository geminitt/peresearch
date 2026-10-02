"""Contracts of the interface's commands and dialogs (peresearch.tui.Chat), driven with Textual's test pilot:
what every /command does in every state (no workspace, a workspace, busy, nothing yet), /resume and its menu,
Tab completion, the permission dialog's four answers, and what a clicked link may do."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from peresearch import tui
from peresearch.agent import Agent, Limits
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
        await say(app, pilot, "/exit")
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=None)
    assert notes_of_shown(app.before_exit) == [
        "no folders declared: /add <folder>", "folders cannot be changed here", "folders cannot be changed here",
        "folders cannot be changed here", "nothing to retry yet", "no sources yet",
        "no earlier conversation in this project", "unknown command /bogus; / for the list"]
    assert "**Commands** (the keys: `/keys`)\n" + "\n".join(f"- `{c}` {d}" for c, d in COMMANDS.items()) in app.before_exit
    assert "**Keys**\n" + "\n".join(f"- **{k}** {d}" for k, d in KEYS.items()) in app.before_exit
    assert app.return_code is not None or not app.is_running                 # /exit quits


def notes_of_shown(shown):
    keep = ("no ", "folders cannot", "nothing to", "unknown command")
    return [line for line in shown.splitlines() if line.startswith(keep)]


def test_folders_are_added_listed_indexed_and_removed(home, notes, tmp_path):
    ws = Workspace(Index(home, embedder=FakeEmbedder()))

    async def steps(app, pilot):
        await settle(app, pilot, lambda: not app.indexing)
        for text in ("/add", f"/add {tmp_path / 'nope'}", f"/add {notes}"):
            await say(app, pilot, text)
        await settle(app, pilot, lambda: not app.indexing)
        app.indexing = "Indexing"                                            # an update still running
        await say(app, pilot, "/index")
        app.indexing = ""
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
    assert str(notes.resolve()) in n and f"{tmp_path} was not a declared folder" in n
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
        await say(app, pilot, "/sources")
        app.sources_shown = texts(app)
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
    assert "**`N1`**" in app.sources_shown and "BPE merges frequent symbol pairs" in app.sources_shown
    retried = [m["content"] for m in llm.seen[2][0] if m["role"] == "user"]
    assert retried[-1].endswith("How does BPE work?")                    # /retry asks the same question again
    assert app.after_new.strip() == "New conversation: earlier questions are no longer sent as context."
    assert [m["content"] for m in llm.seen[3][0] if m["role"] == "user"] == ["something else"]   # no history


def test_resume_lists_conversations_and_continues_the_chosen_one(home, notes):
    first, _ = agent(home, notes, Reply("BPE merges pairs [N1]."), Reply("It has a second stage [W1]."))
    first.ask("How does BPE work?")
    first.ask("And SuperBPE?")
    second, _ = agent(home, notes, Reply("Pho is a soup."))
    second.ask("What is pho?")

    def make(on_event):
        a, llm = agent(home, notes, Reply("More on BPE."), Reply("x"))
        make.llm = llm
        return a

    async def steps(app, pilot):
        from textual.widgets import OptionList
        await say(app, pilot, "/resume")
        menu = app.query_one("#commands", OptionList)
        app.rows = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
        await pilot.press("down")
        await pilot.press("enter")                                       # the prompt is empty: resume the chosen
        await pilot.pause()
        app.resumed = texts(app)
        await say(app, pilot, "Where does it come from?")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert len(app.rows) == 2 and "What is pho?" in app.rows[0] and "2 questions" in app.rows[1]
    assert "Resumed the conversation of " in app.resumed and "(2 questions); the next question continues it." in app.resumed
    assert "❯ How does BPE work?" in app.resumed and "[N1](source:N1)" in app.resumed and "What is pho" not in app.resumed
    assert "<sub>" not in app.resumed                                     # no step count for an old answer
    sent = [m["content"] for m in make.llm.seen[0][0] if m["role"] == "user"]
    assert sent[:2] == ["How does BPE work?", "And SuperBPE?"] and "What is pho?" not in sent


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
        prompt.value = "/add ~/Do"
        await pilot.press("tab")                                         # two: a menu of paths
        app.paths = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        await pilot.press("down", "enter")
        app.picked = prompt.value
        await say(app, pilot, f"/add {notes}")
        await settle(app, pilot, lambda: not app.indexing)
        prompt.value = "/remove "
        await pilot.press("tab")
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
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws)
    assert app.after_command_tab == "/add " and app.after_one == "/add ~/Downloads/"
    assert app.paths == ["path:~/Docs/", "path:~/Downloads/"] and app.picked == "/add ~/Downloads/"
    assert app.remove_tab == f"/remove {notes.resolve()}/" and app.after_enter == "/remove "
    assert app.all_commands == len(COMMANDS) and app.menu_after_escape is False


@pytest.mark.parametrize("tool, verb", sorted(Permission.VERBS.items()) + [("other", "other")])
def test_the_permission_dialog_names_the_action_and_the_folder(tool, verb, tmp_path):
    from textual.app import App
    from textual.widgets import Static

    file = tmp_path / "Downloads" / "paper.md"
    file.parent.mkdir()
    file.write_text("x")
    got = {}

    class Host(App):
        def on_mount(self):
            self.push_screen(Permission(tool, file), lambda choice: got.setdefault("choice", choice))

    async def run():
        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            got["text"] = " ".join(str(w.render()) for w in app.screen.query(Static))
            got["options"] = [str(app.screen.query_one("#choices").get_option_at_index(i).prompt) for i in range(3)]
            await pilot.press("down", "enter")
    import asyncio
    asyncio.run(run())
    assert f"peresearch wants to {verb} something outside your declared folders:" in got["text"]
    assert str(file) in got["text"] and "esc stops the question" in got["text"]
    assert got["options"] == ["Yes, this once", f"Yes, and everything in {file.parent} for this session", "No"]
    assert got["choice"] == "session"


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
    assert f"**N1** {path}:L2-4 — bpe.md" in app.shown_source and "[open the file](open:N1)" in app.shown_source
