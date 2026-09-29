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
        from peresearch.tui import Prompt
        app.query_one("#ask", Prompt).value = "BPE merges pairs?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    shown = app.shown
    assert "❯ BPE merges pairs?" in shown and "● grep(" in shown and "⎿" in shown
    assert "search_notes" not in shown                                   # only what the model chose to call
    assert "Your notes cover it [N1](source:N1)." in shown and "**Sources**" in shown and "answer" in events


def test_the_conversation_fills_the_screen(home, notes):
    from textual.containers import VerticalScroll
    from peresearch.tui import Prompt

    from peresearch.tui import Chat

    async def run():
        app = Chat(lambda on_event: agent(home, notes, Reply("x [N1]."))[0])
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            log, ask = app.query_one("#log", VerticalScroll), app.query_one("#ask", Prompt)
            return log.size.height, ask.region.bottom
    height, bottom = asyncio.run(run())
    assert height >= 30 and bottom == 39                   # the log takes the screen; input, then the key hints


def test_slash_opens_the_command_menu_filters_it_and_runs_the_choice(home, notes):
    async def steps(app, pilot):
        from textual.widgets import OptionList
        menu = app.query_one("#commands", OptionList)
        await pilot.press("/")
        await pilot.pause()
        from peresearch.tui import COMMANDS
        assert menu.display and menu.option_count == len(COMMANDS)
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
        from peresearch.tui import Prompt
        prompt = app.query_one("#ask", Prompt)
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


def test_escape_interrupts_a_running_question_at_once(home, notes):
    """Esc while the model is still answering: the question ends within a second, not when the answer arrives."""
    import time

    from peresearch.llm import Interrupted

    class SlowLLM(ScriptedLLM):
        def chat(self, messages, tools=None, thinking=None, stop=None):
            end = time.time() + 30                               # a long answer on a slow model
            while time.time() < end:
                if stop and stop():
                    raise Interrupted("stopped")
                time.sleep(0.05)
            return super().chat(messages, tools, thinking)

    def make(on_event):
        return Agent(SlowLLM(Reply("never")), toolbox(home, notes))

    async def steps(app, pilot):
        from peresearch.tui import Prompt
        app.query_one("#ask", Prompt).value = "BPE merges pairs?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: app.agent is not None)
        await pilot.pause(0.3)
        app.pressed = time.time()
        await pilot.press("escape")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.took = time.time() - app.pressed
    app = tui_run(make, steps)
    assert app.last.stopped == "cancelled" and app.took < 1.5 and "interrupted" in app.shown


def test_folders_are_added_with_tab_completion_indexed_and_removed_in_the_interface(home, notes, tmp_path):
    from peresearch.tui import Prompt

    from peresearch.workspace import Workspace
    extra = tmp_path / "course-notes"
    extra.mkdir()
    (extra / "attention.md").write_text("# Attention\n\nScaled dot-product attention divides by the square root of d.\n")
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))

    async def steps(app, pilot):
        await settle(app, pilot, lambda: not app.busy and "Index" in texts(app) or "added" in texts(app))
        assert "added" in texts(app)                                      # indexed on start: notes was new
        prompt = app.query_one("#ask", Prompt)
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
        prompt.value = "why does this fail: "                              # the cursor goes to the end
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
    """Yes lets the read happen; No refuses it and the model goes on; Esc refuses it and stops the question (it
    used to mean only No, so Esc on the dialog did not stop anything — seen on the local vLLM)."""
    from peresearch.tui import Permission, Prompt

    def make(on_event):
        a, llm = agent(home, notes, Reply("", [ToolCall("c1", "read", {"path": "~/Downloads/paper.md"})]),
                       Reply("Your paper says attention is all you need [N1]."),
                       Reply("", [ToolCall("c2", "read", {"path": "~/Downloads/sub/more.md"})]),
                       Reply("I was not allowed to read it."),
                       Reply("", [ToolCall("c3", "read", {"path": "~/Downloads/sub/more.md"})]),
                       Reply("never reached"))
        make.llm = llm
        return a

    async def ask(app, pilot, question):
        prompt = app.query_one("#ask", Prompt)
        app.last = None
        prompt.value = question
        await pilot.press("enter")
        await settle(app, pilot, lambda: isinstance(app.screen, Permission))
        return isinstance(app.screen, Permission)

    async def steps(app, pilot):
        app.dialog = await ask(app, pilot, "what does the paper in Downloads say?") and "paper.md" in str(app.screen.path)
        await pilot.press("enter")                                          # "Yes, this once"
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.first = app.last
        await ask(app, pilot, "and the other file?")
        await pilot.press("down", "down", "enter")                          # "No"
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.second = app.last
        await ask(app, pilot, "please, that other file")
        from textual.widgets import Static
        app.shown_dialog = " ".join(str(w.render()) for w in app.screen.query(Static))
        await pilot.press("escape")                                         # refuse, and stop the question
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.third = app.last
        app.focus_after = app.focused
    app = tui_run(make, steps)
    assert app.dialog and "N1" in app.first.sources and "Attention" in app.first.sources["N1"].text
    assert app.second.sources == {} and app.second.text == "I was not allowed to read it." and not app.second.stopped
    assert app.third.stopped == "cancelled" and "never reached" not in app.shown
    assert "esc stops the question" in app.shown_dialog
    assert isinstance(app.focus_after, Prompt)


def test_an_interrupted_answer_is_not_called_an_answer_without_sources():
    from peresearch.agent import Answer, Check
    from peresearch.tui import render

    shown = render(Answer("_(interrupted)_", {}, Check(), 1, stopped="cancelled"))
    assert "stopped: interrupted" in shown and "without any source" not in shown


def test_a_late_spinner_tick_after_the_interface_closed_is_harmless(home, notes):
    """The spinner's timer can fire once more while the app shuts down; its widgets are gone by then (this made
    test_up_recalls_earlier_questions_and_new_forgets_them fail about once in ten runs)."""
    async def steps(app, pilot):
        await pilot.pause()
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    app.busy = True
    app.tick()                                                   # after unmounting: must not raise


def test_the_prompt_takes_several_lines(home, notes):
    """Enter sends, Ctrl+Enter (a terminal sends it as Ctrl+J) starts a new line; the box grows with its text up to
    its limit; ↑ moves between lines and recalls an earlier question only from the first line."""
    from peresearch.tui import Prompt

    def make(on_event):
        a, llm = agent(home, notes, Reply("One."), Reply("Two."))
        make.llm = llm
        return a

    async def steps(app, pilot):
        prompt = app.query_one("#ask", Prompt)
        await pilot.press(*"first", "enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        one_line = prompt.region.height
        await pilot.press(*"line one", "ctrl+j", *"line two", "ctrl+j", *"line three")
        await pilot.pause()
        app.grown = (one_line, prompt.region.height)
        await pilot.press("up")                                     # from the third line to the second
        app.after_up = (prompt.value, prompt.cursor_location[0])
        for _ in range(12):
            await pilot.press("ctrl+j", "x")
        await pilot.pause()
        app.capped = prompt.region.height
        prompt.value = "line one\nline two"
        await pilot.press("up", "up")                               # up to the first line, then to the history
        app.recalled = prompt.value
        prompt.value = "a\nb"
        app.last = None
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.emptied = prompt.value
    app = tui_run(make, steps)
    assert app.grown[1] == app.grown[0] + 2                         # three lines of text, same border
    assert app.after_up == ("line one\nline two\nline three", 1)
    assert app.capped == app.grown[0] + 7                           # at most 8 lines, then it scrolls inside
    assert app.recalled == "first"
    sent = [m["content"] for m in make.llm.seen[-1][0] if m["role"] == "user"][-1]
    assert sent.endswith("a\nb") and app.emptied == ""
    assert "❯ a\nb" in app.shown


def test_ctrl_c_copies_the_selection_and_ctrl_q_quits(home, notes):
    """Ctrl+C used to quit even with text selected; now it copies (the conversation's mouse selection first, else
    the prompt's), and Ctrl+Q quits."""
    from peresearch.tui import Prompt

    async def steps(app, pilot):
        await pilot.pause()
        app.note("SuperBPE merges across spaces")
        await pilot.pause()
        note = app.query(".note").last()
        await pilot.mouse_down(note, offset=(0, 0))                 # drag over the note's first words
        await pilot.hover(note, offset=(8, 0))
        await pilot.mouse_up(note, offset=(8, 0))
        await pilot.press("ctrl+c")
        app.from_log = app.clipboard
        app.running_after_copy = app.is_running
        app.screen.clear_selection()
        prompt = app.query_one("#ask", Prompt)
        prompt.value = "copy me"
        await pilot.press("shift+home", "ctrl+c")
        app.from_prompt = app.clipboard
        app._clipboard = ""
        prompt.value = ""
        await pilot.press("ctrl+c")                                 # nothing selected: a reminder, no quitting
        await pilot.pause()
        app.reminded = str(app.query_one("#status").render())
        await pilot.press("ctrl+q")
        await pilot.pause()
        app.running_after_quit = app.is_running
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    assert app.from_log.startswith("SuperBPE") and app.running_after_copy
    assert app.from_prompt == "copy me"
    assert "ctrl+q" in app.reminded
    assert not app.running_after_quit


def test_the_interface_opens_fresh_and_resume_brings_back_a_conversation(home, notes):
    from peresearch.tui import Prompt

    earlier, _ = agent(home, notes, Reply("BPE merges pairs [N1]."))
    earlier.ask("How does BPE work?")

    def make(on_event):
        a, llm = agent(home, notes, Reply("It came from compression [N1]."))
        make.llm = llm
        return a

    async def steps(app, pilot):
        await pilot.pause()
        app.opened = app.shown if hasattr(app, "shown") else texts(app)
        prompt = app.query_one("#ask", Prompt)
        prompt.value = "/resume"
        await pilot.press("enter")
        await pilot.pause()
        menu = app.query_one("#commands")
        app.listed = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
        await pilot.press("enter")                                    # the newest one
        await pilot.pause()
        app.resumed = texts(app)
        prompt.value = "Where does it come from?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert "BPE" not in app.opened                                    # nothing from the earlier conversation on open
    assert len(app.listed) == 1 and "How does BPE work?" in app.listed[0] and "1 question" in app.listed[0]
    assert "How does BPE work?" in app.resumed and "BPE merges pairs" in app.resumed
    sent = [m["content"] for m in make.llm.seen[0][0] if m["role"] == "user"]
    assert sent[0] == "How does BPE work?"


def test_resume_with_no_earlier_conversation_says_so(home, notes):
    from peresearch.tui import Prompt

    async def steps(app, pilot):
        await pilot.pause()
        app.query_one("#ask", Prompt).value = "/resume"
        await pilot.press("enter")
        await pilot.pause()
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    assert "no earlier conversation" in app.shown


def test_a_model_that_does_not_answer_is_explained_and_retry_asks_again(home, notes):
    """The error names what is wrong and what to do; /retry sends the same question again."""
    class APIConnectionError(Exception):
        pass

    class Flaky(ScriptedLLM):
        fail = True

        def chat(self, messages, tools=None, thinking=None, stop=None):
            if Flaky.fail:
                raise APIConnectionError("Connection error.")
            return super().chat(messages, tools, thinking)

    def make(on_event):
        llm = Flaky(Reply("BPE merges pairs [N1]."))
        make.llm = llm
        return Agent(llm, toolbox(home, notes))

    async def steps(app, pilot):
        from peresearch.tui import Prompt
        prompt = app.query_one("#ask", Prompt)
        prompt.value = "/retry"
        await pilot.press("enter")                                    # nothing to retry yet
        await pilot.pause()
        prompt.value = "How does BPE work?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and "does not answer" in texts(app))
        app.error = texts(app)
        Flaky.fail = False
        prompt.value = "/retry"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert "nothing to retry" in app.error
    assert "does not answer" in app.error and "/retry" in app.error and "APIConnectionError" not in app.error.split("does not answer")[0][-40:]
    assert app.last.text == "BPE merges pairs [N1]."
    sent = [m["content"] for m in make.llm.seen[-1][0] if m["role"] == "user"]
    assert sent[-1].endswith("How does BPE work?")


def test_no_model_endpoint_says_how_to_set_one(home, notes):
    def make(on_event):
        raise RuntimeError("no model endpoint: set PERESEARCH_LLM_URL (see peresearch/settings.py)")

    async def steps(app, pilot):
        from peresearch.tui import Prompt
        app.query_one("#ask", Prompt).value = "hello"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and "setup" in texts(app))
    app = tui_run(make, steps)
    assert "peresearch setup" in app.shown


def test_ctrl_r_shows_and_hides_the_models_reasoning(home, notes):
    from peresearch.tui import Prompt

    def make(on_event):
        return agent(home, notes, Reply("BPE merges pairs [N1].", reasoning="first I recall the notes"),
                     Reply("It has two stages [N1].", reasoning="then SuperBPE"))[0]

    async def steps(app, pilot):
        prompt = app.query_one("#ask", Prompt)
        prompt.value = "How does BPE work?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.hidden = texts(app)
        await pilot.press("ctrl+r")
        await settle(app, pilot, lambda: "first I recall" in texts(app))
        app.shown_once = texts(app)
        app.last = None
        prompt.value = "And SuperBPE?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        await settle(app, pilot, lambda: "then SuperBPE" in texts(app))
        app.shown_both = texts(app)
        await pilot.press("ctrl+r")
        await settle(app, pilot, lambda: "first I recall" not in texts(app))
    app = tui_run(make, steps)
    assert "first I recall" not in app.hidden and "BPE merges pairs" in app.hidden
    assert "first I recall the notes" in app.shown_once and "Reasoning" in app.shown_once
    assert "then SuperBPE" in app.shown_both                         # new answers follow the setting
    assert "first I recall" not in app.shown and "then SuperBPE" not in app.shown


def test_a_long_reasoning_is_shown_from_its_end():
    """Twelve model calls can think tens of thousands of characters; drawing all of it stalls the interface."""
    from peresearch.agent import Answer, Check
    from peresearch.tui import REASONING_SHOWN, render

    long = "early thought. " * 2000 + "the last step before answering"
    shown = render(Answer("ok", {}, Check(), 1, reasoning=long), reasoning=True)
    assert "the last step before answering" in shown and len(shown) < REASONING_SHOWN + 500
    assert f"{len(long) - REASONING_SHOWN:,} earlier characters not shown" in shown


def test_citations_and_sources_become_links():
    from peresearch.agent import Answer, Check
    from peresearch.tools import Source
    from peresearch.tui import render

    src = {"N1": Source("N1", "file", "/n/bpe.md:L3-5", "bpe.md", "BPE merges"),
           "W1": Source("W1", "web", "https://example.org/bpe", "BPE", "history")}
    shown = render(Answer("BPE merges pairs [N1, W1]. More [W1].", src, Check(), 1))
    assert "[N1](source:N1), [W1](source:W1)" in shown and "More [W1](source:W1)." in shown
    assert "- [N1](source:N1) /n/bpe.md:L3-5" in shown


def test_clicking_a_citation_shows_the_source_and_links_open_only_what_is_safe(home, notes, monkeypatch):
    from textual.widgets import Markdown

    from peresearch import tui

    opened = []
    monkeypatch.setattr(tui, "open_outside", lambda target, line=None: opened.append((target, line)))

    def make(on_event):
        return agent(home, notes, Reply("", [ToolCall("c1", "read", {"path": "bpe.md", "start": 1, "end": 4})]),
                     Reply("BPE merges pairs [N1]. See [the paper](https://arxiv.org/abs/1508.07909), "
                           "not [this](javascript:alert(1)) or [that](file:///etc/passwd)."))[0]

    async def steps(app, pilot):
        from peresearch.tui import Prompt
        app.query_one("#ask", Prompt).value = "How does BPE work?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        answer_page = app.pages[-1][0]
        answer_page.post_message(Markdown.LinkClicked(answer_page, "source:N1"))
        await settle(app, pilot, lambda: "open the file" in texts(app))
        app.after_citation = texts(app)
        shown = [w for w in app.query("#log > *") if isinstance(w, Markdown)][-1]
        shown.post_message(Markdown.LinkClicked(shown, "open:N1"))
        for href in ("https://arxiv.org/abs/1508.07909", "javascript:alert(1)", "file:///etc/passwd"):
            answer_page.post_message(Markdown.LinkClicked(answer_page, href))
        await pilot.pause(0.2)
    app = tui_run(make, steps)
    assert "bpe.md" in app.after_citation and "BPE merges frequent symbol pairs" in app.after_citation
    path = str(notes / "bpe.md")
    assert opened == [(path, 1), ("https://arxiv.org/abs/1508.07909", None)]
    assert app.shown.count("not opened") == 2


def test_open_outside_never_runs_a_file(monkeypatch, tmp_path):
    """A file is shown in an editor at its line, or selected in Explorer — never handed to Explorer to open, which
    would run a .bat or .exe; only http(s) goes to the browser."""
    from peresearch import tui

    runs = []
    monkeypatch.setattr(tui.subprocess, "Popen", lambda cmd, **kw: runs.append(cmd))
    monkeypatch.setattr(tui, "on_wsl", lambda: True)
    monkeypatch.setattr(tui.shutil, "which", lambda name: "/usr/bin/code" if name == "code" else f"/x/{name}")
    evil = tmp_path / "run me.bat"
    evil.write_text("echo hi")
    tui.open_outside(str(evil), 3)
    tui.open_outside("https://example.org/a?b=1&c=2")
    assert runs[0] == ["code", "-g", f"{evil}:3"] and runs[1] == ["explorer.exe", "https://example.org/a?b=1&c=2"]
    monkeypatch.setattr(tui.shutil, "which", lambda name: None if name == "code" else f"/x/{name}")
    monkeypatch.setattr(tui, "windows_path", lambda p: "C:\\\\x\\\\run me.bat")
    tui.open_outside(str(evil), 3)
    assert runs[2] == ["explorer.exe", "/select,C:\\\\x\\\\run me.bat"]
    for bad in ("javascript:alert(1)", "file:///etc/passwd", str(tmp_path / "missing.md")):
        assert tui.open_outside(bad) and len(runs) == 3                # refused, nothing run
