"""The interface's look, screen by screen, against approved layouts (tests/data/tui-look/*.txt).

Behavior tests kept passing while the interface grew ugly (2026-10-13: three-line buttons, windows of 85% height
around one line, a hidden settings field, the mascot across the command menu). Each screen here is drawn by
Textual's own compositor at a fixed size and compared character by character with its approved layout, so a
change of look fails until a person has looked at it and approved it:

    APPROVE_TUI_LOOK=1 pixi run pytest tests/test_tui_look.py     # rewrites them (conftest drops PERESEARCH_*)

then review `git diff tests/data/tui-look/` (or the files themselves) before keeping them. Colors are checked by
the palette tests (test_tui.py, test_tui_input.py); this file checks what goes where."""

import os
from pathlib import Path

import pytest

from peresearch import connections, settings, tui
from peresearch.llm import Reply, ToolCall
from peresearch.agent import Agent
from tests.helpers import ScriptedLLM, settle, toolbox, tui_run

APPROVED = Path(__file__).parent / "data" / "tui-look"
SIZE = (100, 30)
WELCOME = "✻ peresearch · research over your files and the web\n\n  model    qwen3-8b\n  folders  ~/notes"


def drawn(app) -> str:
    return "\n".join(strip.text.rstrip() for strip in app.screen._compositor.render_strips()) + "\n"


def compare(name: str, text: str) -> None:
    path = APPROVED / f"{name}.txt"
    if os.environ.get("APPROVE_TUI_LOOK"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.exists(), f"no approved layout for {name}: run with APPROVE_TUI_LOOK=1 and review it"
    assert text == path.read_text(encoding="utf-8"), f"{name}: the screen differs from {path}"


@pytest.fixture
def notes_at_home():
    """Notes inside the (sandboxed) home folder, so their paths read `~/notes/…` on any machine."""
    root = Path.home() / "notes"
    root.mkdir(parents=True, exist_ok=True)
    (root / "bpe.md").write_text("# Tokenization\n\nBPE merges frequent symbol pairs into new tokens.\n\n"
                                 "SuperBPE adds a second stage whose merges may cross spaces.\n")
    return root


def run(steps, make=lambda event: None, **kw):
    kw.setdefault("welcome", lambda: WELCOME)
    return tui_run(make, steps, size=SIZE, **kw)


async def open_window(app, pilot, command):
    app.query_one("#ask", tui.Prompt).value = command
    await pilot.press("enter")
    await pilot.pause(0.3)


def test_look_at_start(home):
    settings.save({"PERESEARCH_LLM_URL": "http://model.test/v1", "PERESEARCH_LLM_MODEL": "qwen3-8b"})

    async def steps(app, pilot):
        app.update_setup_line(now=True)
        await pilot.pause(0.3)
        app.look = drawn(app)
    compare("start", run(steps).look)


def test_look_of_an_answer_with_a_tool_call_code_and_sources(home, notes_at_home):
    settings.save({"PERESEARCH_LLM_URL": "http://model.test/v1", "PERESEARCH_LLM_MODEL": "qwen3-8b"})
    answer = ("Byte pair encoding merges the most frequent pair of symbols [N1].\n\n"
              "- Each merge creates one new token.\n- SuperBPE lets merges cross spaces [N1].\n\n"
              "```python\nfor _ in range(merges):\n    vocab.add(most_frequent(pairs))\n```\n")

    def make(event):
        return Agent(ScriptedLLM(Reply("", [ToolCall("c1", "search_notes", {"query": "byte pair encoding"})]),
                                 Reply(answer)), toolbox(home, notes_at_home), on_event=event)

    async def steps(app, pilot):
        app.update_setup_line(now=True)
        app.query_one("#ask", tui.Prompt).value = "What do my notes say about BPE?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.pages)
        await pilot.pause(0.3)
        app.look = drawn(app)
    compare("answer", run(steps, make).look)


def test_look_of_the_command_menu(home):
    async def steps(app, pilot):
        await pilot.press("/")
        await pilot.pause(0.3)
        app.look = drawn(app)
    compare("menu", run(steps).look)


@pytest.mark.parametrize("command", ["/model", "/settings", "/folders", "/help", "/check"])
def test_look_of_each_window(home, notes_at_home, command):
    connections.add("chat", "Local server", "http://127.0.0.1:8000/v1", "", "qwen3-8b")
    settings.save({"PERESEARCH_LLM_URL": "http://127.0.0.1:8000/v1", "PERESEARCH_LLM_MODEL": "qwen3-8b"})
    from peresearch.workspace import Workspace
    from peresearch.zetokrag.index import Index
    from tests.conftest import FakeEmbedder
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes_at_home))

    async def steps(app, pilot):
        await open_window(app, pilot, command)
        app.look = drawn(app)
    compare("window-" + command.strip("/"), run(steps, workspace=ws).look)


@pytest.mark.parametrize("step", ["connect", "key", "pick"])
def test_look_of_connecting_a_model(home, step):
    async def steps(app, pilot):
        await open_window(app, pilot, "/model")
        app.screen.discover_models = lambda entry: ["gpt-6.1-mini", "gpt-6.1-sol", "text-embedding-4"]
        await pilot.press("a")
        await pilot.pause(0.3)
        if step in ("key", "pick"):
            listing = app.screen.query_one("#connect-list")
            ids = [listing.get_option_at_index(i).id for i in range(listing.option_count)]
            listing.highlighted = ids.index("preset:OpenAI")
            await pilot.press("enter")
            await pilot.pause(0.3)
        if step == "pick":
            app.screen.query_one("#api-key").value = "synthetic-key"
            await pilot.press("enter")
            await settle(app, pilot, lambda: app.screen.id == "pick-window")
            await pilot.pause(0.3)
        app.look = drawn(app)
    compare("connect-" + step, run(steps).look)
