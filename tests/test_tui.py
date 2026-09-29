"""The terminal interface driven with Textual's test pilot: layout, commands, history, colors, permission
dialog, pasting (the real terminal path is in test_tui_input.py)."""

import asyncio

from peresearch.agent import Agent
from peresearch.llm import Reply, ToolCall
from peresearch.tools import Source
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import ScriptedLLM, notes, toolbox, agent, tui_run, settle, texts, user_home


def test_the_terminal_interface_shows_tools_answer_and_sources(home, notes):
    events = []

    def make(on_event):
        a, _ = agent(home, notes, Reply("", [ToolCall("c1", "grep", {"pattern": "SuperBPE"})]),
                     Reply("Your notes cover it [N1]."))
        a.on_event = lambda k, d: (events.append(k), on_event(k, d))
        return a

    async def steps(app, pilot):
        from textual.widgets import Input
        app.query_one("#ask", Input).value = "BPE merges pairs?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    shown = app.shown
    assert "❯ BPE merges pairs?" in shown and "● grep(" in shown and "⎿" in shown
    assert "search_notes" not in shown                                   # only what the model chose to call
    assert "Your notes cover it [N1]." in shown and "**Sources**" in shown and "answer" in events


def test_the_conversation_fills_the_screen(home, notes):
    from textual.containers import VerticalScroll
    from textual.widgets import Input

    from peresearch.tui import Chat

    async def run():
        app = Chat(lambda on_event: agent(home, notes, Reply("x [N1]."))[0])
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            log, ask = app.query_one("#log", VerticalScroll), app.query_one("#ask", Input)
            return log.size.height, ask.region.bottom
    height, bottom = asyncio.run(run())
    assert height >= 30 and bottom == 39                   # the log takes the screen; input, then the key hints


def test_slash_opens_the_command_menu_filters_it_and_runs_the_choice(home, notes):
    async def steps(app, pilot):
        from textual.widgets import OptionList
        menu = app.query_one("#commands", OptionList)
        await pilot.press("/")
        await pilot.pause()
        assert menu.display and menu.option_count == 8
        await pilot.press("f", "o")
        await pilot.pause()
        assert menu.option_count == 1 and menu.get_option_at_index(0).id == "/folders"
        await pilot.press("enter")
        await pilot.pause()
        assert not menu.display
    from peresearch.workspace import Workspace
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws)
    assert str(notes) in app.shown


def test_command_descriptions_line_up_in_one_column(home, notes):
    from peresearch.tui import COMMANDS

    async def steps(app, pilot):
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        menu = app.query_one("#commands")
        rows = ["".join(s.text for s in menu.render_line(y)) for y in range(menu.size.height)]
        app.starts = {c: row.index(COMMANDS[c]) for row in rows for c in COMMANDS if f"{c} " in row and COMMANDS[c] in row}
    app = tui_run(lambda e: agent(home, notes)[0], steps, size=(100, 40))
    assert len(app.starts) >= 4 and len(set(app.starts.values())) == 1, app.starts


def test_up_recalls_earlier_questions_and_new_forgets_them(home, notes):
    replies = [Reply("", [ToolCall("a", "search_notes", {"query": "BPE"})]), Reply("first [N1]."),
               Reply("", [ToolCall("b", "search_notes", {"query": "SuperBPE"})]), Reply("second [N1].")]

    def make(on_event):
        a, llm = agent(home, notes, *replies)
        make.llm = llm
        return a

    async def steps(app, pilot):
        from textual.widgets import Input
        prompt = app.query_one("#ask", Input)
        prompt.value = "BPE merges pairs?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        await pilot.press("up")
        assert prompt.value == "BPE merges pairs?"
        prompt.value = "/new"
        await pilot.press("enter")
        await pilot.pause()
        prompt.value = "SuperBPE merges?"
        app.last = None
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert app.last.sources, "the second question, run in another thread, must still search the files"
    second = make.llm.seen[-1][0]
    assert all(m["content"] != "BPE merges pairs?" for m in second if m["role"] == "user")   # history was reset


def test_escape_interrupts_a_running_question(home, notes):
    import threading
    gate = threading.Event()

    class SlowLLM(ScriptedLLM):
        def chat(self, messages, tools=None, thinking=None):
            gate.wait(2)
            return super().chat(messages, tools, thinking)

    def make(on_event):
        tb = toolbox(home, notes)
        return Agent(SlowLLM(*[Reply("", [ToolCall(f"c{i}", "grep", {"pattern": f"x{i}"})]) for i in range(10)]), tb)

    async def steps(app, pilot):
        from textual.widgets import Input
        app.query_one("#ask", Input).value = "BPE merges pairs?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: app.agent is not None)
        await pilot.press("escape")
        gate.set()
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert app.last.stopped == "cancelled" and app.last.steps <= 2


def test_folders_are_added_with_tab_completion_indexed_and_removed_in_the_interface(home, notes, tmp_path):
    from textual.widgets import Input

    from peresearch.workspace import Workspace
    extra = tmp_path / "course-notes"
    extra.mkdir()
    (extra / "attention.md").write_text("# Attention\n\nScaled dot-product attention divides by the square root of d.\n")
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))

    async def steps(app, pilot):
        await settle(app, pilot, lambda: not app.busy and "Index" in texts(app) or "added" in texts(app))
        assert "added" in texts(app)                                      # indexed on start: notes was new
        prompt = app.query_one("#ask", Input)
        prompt.value = f"/add {tmp_path}/course-no"
        await pilot.press("tab")
        assert prompt.value == f"/add {extra}/"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and str(extra) in texts(app) and "1 added" in texts(app))
        assert "2 folders" in str(app.query_one("#heading").render())
        prompt.value = f"/remove {extra}"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and "1 removed" in texts(app))
    tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws, index_on_start=True,
            heading=lambda: f"peresearch · {len(ws.folders())} folders")
    assert ws.folders() == [notes.resolve()]
    assert any("attention.md" in r[1] for r in ws.index.rows()) is False       # its chunks left the index


def test_vietnamese_input_method_bursts_type_correctly(home, notes):
    from textual import events

    from peresearch.tui import Prompt

    async def steps(app, pilot):
        await pilot.pause()
        typed = [("c", "c"), ("h", "h"), ("a", "a"), ("o", "o"), ("backspace", "\x7f"), ("backspace", "\x08"),
                 ("à", "à"), ("o", "o")]                    # Telex "chaof": raw letters, then ⌫⌫ + "ào", in one burst
        for key, ch in typed:
            e = events.Key(key, ch)
            e.set_sender(app)
            app._driver.send_message(e)                      # back to back, as one terminal read delivers them
        await settle(app, pilot, lambda: app.query_one("#ask", Prompt).value == "chào", tries=50)
        app.typed = app.query_one("#ask", Prompt).value
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    assert app.typed == "chào"                               # Textual alone gave "chao": ⌫ ran after the letters


def test_typing_after_a_click_on_the_conversation_still_goes_to_the_prompt(home, notes):
    from peresearch.tui import Prompt

    async def steps(app, pilot):
        await pilot.pause()
        for i in range(60):
            app.note(f"line {i}")
        await pilot.pause()
        await pilot.click("#log", offset=(10, 5))
        await pilot.press("x", "i", "n")
        await pilot.pause()
        app.typed = app.query_one("#ask", Prompt).value
        app.focus_after = app.focused
        log = app.query_one("#log")
        bottom = log.scroll_y
        await pilot.press("pageup")
        await pilot.pause()
        app.scrolled = (bottom, log.scroll_y)
    app = tui_run(lambda e: agent(home, notes)[0], steps, size=(100, 30))
    assert app.typed == "xin"                                # the click had moved focus: keys were lost
    assert isinstance(app.focus_after, Prompt)
    assert app.scrolled[1] < app.scrolled[0]                 # PageUp from the prompt scrolls the conversation


def test_pasted_lines_reach_the_model_whole(home, notes):
    """Textual's input keeps only the first line of a paste; the prompt shows a multi-line paste as a marker, as
    Claude Code does, and the model gets every line."""
    from textual import events

    from peresearch.tui import Prompt

    def make(on_event):
        a, llm = agent(home, notes, Reply("Seen."))
        make.llm = llm
        return a

    async def steps(app, pilot):
        await pilot.pause()
        prompt = app.query_one("#ask", Prompt)
        prompt.value = "why does this fail: "
        prompt.cursor_position = len(prompt.value)
        prompt.post_message(events.Paste("Traceback (most recent call last):\n  File \"x.py\", line 1\nKeyError: 'a'\n"))
        await pilot.pause()
        app.shown_value = prompt.value
        prompt.post_message(events.Paste("one line"))
        await pilot.pause()
        app.after_single = prompt.value
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert app.shown_value == "why does this fail: [Pasted text #1 +3 lines]"
    assert app.after_single.endswith("[Pasted text #1 +3 lines]one line")
    sent = [m["content"] for m in make.llm.seen[0][0] if m["role"] == "user"][-1]
    assert "KeyError: 'a'" in sent and "File \"x.py\", line 1" in sent and "[Pasted text" not in sent
    assert "❯ why does this fail: [Pasted text #1 +3 lines]one line" in app.shown       # the log keeps the marker


def test_the_interface_uses_only_colors_every_terminal_scheme_can_show(home, notes):
    """Only the default colors and the six plain hues: black and white are some scheme's background, and the bright
    colors are grays in Solarized. Nothing sits on a colored background (text on one is unreadable in some scheme);
    emphasis is reversed instead. Textual's command palette (Ctrl+P, its own look and themes) is off."""
    from textual.highlight import HighlightTheme

    from peresearch.tui import Fence

    plain = {"ansi_default", "transparent", "ansi_red", "ansi_green", "ansi_yellow", "ansi_blue", "ansi_magenta",
             "ansi_cyan"}

    async def steps(app, pilot):
        await pilot.pause()
        app.variables = app.get_css_variables()
        app.selection = app.screen.get_component_rich_style("screen--selection")   # text selected with the mouse
        await pilot.press("ctrl+p")
        await pilot.pause()
        app.screen_after = type(app.screen).__name__
        app.fence_theme = Fence.highlight("x = `ls`", "bash", ansi=True, dark=app.current_theme.dark)
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    colors = {k: v.split()[0] for k, v in app.variables.items() if isinstance(v, str) and v.startswith("ansi_")}
    assert {k: v for k, v in colors.items() if v not in plain} == {}
    assert {k: v for k, v in colors.items() if "background" in k and v not in ("ansi_default", "transparent")} == {}
    assert app.screen_after == "Screen"
    assert app.selection.reverse
    spans = [str(s.style) for s in app.fence_theme.spans]
    assert spans and not [s for s in spans if "bright" in s or "black" in s or "white" in s], spans


def test_no_scrollbar_anywhere_yet_everything_still_scrolls(home, notes):
    """The conversation and the command menu both overflow here; neither shows a scrollbar, and both still scroll:
    the conversation with the mouse wheel and PageUp, the menu by moving the choice past its last visible row."""
    from textual import events

    async def steps(app, pilot):
        await pilot.pause()
        for i in range(80):
            app.note(f"line {i}")
        await pilot.press("slash")                               # 8 commands in a menu that shows fewer
        await pilot.pause()
        app.bars = {f"{type(w).__name__}#{w.id}": (w.scrollbar_size_vertical, w.scrollbar_size_horizontal)
                    for w in app.query("*") if w.scrollbar_size_vertical or w.scrollbar_size_horizontal}
        menu, log = app.query_one("#commands"), app.query_one("#log")
        app.menu_overflows = menu.max_scroll_y > 0
        for _ in range(len(menu.options) - 1):
            await pilot.press("down")
        await pilot.pause()
        app.menu_scrolled = menu.scroll_y
        bottom = log.scroll_y
        for _ in range(3):
            log.post_message(events.MouseScrollUp(log, 5, 5, 0, 0, 0, False, False, False))
        await pilot.pause()
        app.wheel = (bottom, log.scroll_y)
    app = tui_run(lambda e: agent(home, notes)[0], steps, size=(100, 30))
    assert app.bars == {}
    assert app.menu_overflows and app.menu_scrolled > 0
    assert app.wheel[1] < app.wheel[0]


def test_an_answer_that_used_no_source_says_so():
    from peresearch.agent import Answer, Check
    from peresearch.tui import render

    bare = render(Answer("SuperBPE+ appeared in 2026.", {}, Check(), 1))
    assert "Answered without any source" in bare                        # the model's memory only: flagged
    cited = render(Answer("BPE merges pairs [N1].", {"N1": Source("N1", "file", "a.md:L1", "", "BPE merges pairs")},
                          Check(), 2))
    assert "without any source" not in cited


def test_the_interface_asks_before_reading_outside_the_declared_folders(home, notes, user_home):
    from textual.widgets import Input

    from peresearch.tui import Permission

    def make(on_event):
        a, _ = agent(home, notes, Reply("", [ToolCall("c1", "read", {"path": "~/Downloads/paper.md"})]),
                     Reply("Your paper says attention is all you need [N1]."),
                     Reply("", [ToolCall("c2", "read", {"path": "~/Downloads/sub/more.md"})]),
                     Reply("I was not allowed to read it."))
        return a

    async def steps(app, pilot):
        from textual.widgets import Input
        prompt = app.query_one("#ask", Input)
        prompt.value = "what does the paper in Downloads say?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: isinstance(app.screen, Permission))
        app.dialog = isinstance(app.screen, Permission) and "paper.md" in str(app.screen.path)
        await pilot.press("enter")                                          # "Yes, this once"
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.first = app.last
        app.last = None
        prompt.value = "and the other file?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: isinstance(app.screen, Permission))
        await pilot.press("escape")                                         # no
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.focus_after = app.focused
    app = tui_run(make, steps)
    assert app.dialog and "N1" in app.first.sources and "Attention" in app.first.sources["N1"].text
    assert app.last.sources == {} and isinstance(app.focus_after, Input)


def test_the_interface_keeps_working_while_a_dialog_is_open(home, notes):
    """The spinner ticks and notes arrive while another screen (the permission dialog) is on top: they must find
    the conversation's widgets, not look for them on the dialog."""
    from textual.screen import ModalScreen
    from textual.widgets import Static

    class Dialog(ModalScreen):
        def compose(self):
            yield Static("a dialog")

    async def steps(app, pilot):
        await pilot.pause()
        app.start("Working")
        app.push_screen(Dialog())
        await pilot.pause(0.35)                                # a few spinner ticks under the dialog
        app.note("a note that arrives meanwhile")
        await pilot.pause(0.1)
        app.pop_screen()
        await pilot.pause(0.1)
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    assert "a note that arrives meanwhile" in app.shown


def test_a_late_spinner_tick_after_the_interface_closed_is_harmless(home, notes):
    """The spinner's timer can fire once more while the app shuts down; its widgets are gone by then (this made
    test_up_recalls_earlier_questions_and_new_forgets_them fail about once in ten runs)."""
    async def steps(app, pilot):
        await pilot.pause()
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    app.busy = True
    app.tick()                                                   # after unmounting: must not raise
