"""The terminal interface driven with Textual's test pilot: layout, the log and status line, history, colors,
permission dialog, pasting (the real terminal path is in test_tui_input.py)."""

import asyncio
import itertools
import threading
import time
from types import SimpleNamespace as NS

import pytest

from peresearch import mascot, tui
from peresearch.agent import Answer, Check

from peresearch.agent import Agent
from peresearch.llm import Interrupted, Reply, ToolCall
from peresearch.tools import Source
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import ScriptedLLM, notes, toolbox, agent, tui_run, settle, texts, user_home
from tests.helpers import DictModel


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
    assert "Your notes cover it [N1](source:N1)." in shown and "**Sources cited**" in shown and "answer" in events


def test_the_screen_at_start(home, notes):
    """The log takes the screen and the input box the bottom: the welcome opens the log (no fixed heading, as in
    Claude Code), the box with its placeholder, no line numbers, a long line wrapped inside it; Enter on the empty box
    does nothing."""
    from textual.containers import VerticalScroll
    from peresearch.tui import Prompt

    async def steps(app, pilot):
        await pilot.pause()
        log, ask = app.query_one("#log", VerticalScroll), app.query_one("#ask", Prompt)
        app.sizes = log.size.height, app.query_one("#input-box").region.bottom
        app.lines = screen_lines(app)
        app.first = [(w.id, str(w.render())) for w in app.query("#log > *")][:1]
        bird, status, box = app.query_one("#mascot"), app.query_one("#status"), app.query_one("#input-box")
        app.perch = (bird.region, box.region, status.region, str(bird.render()))
        app.busy = True                                              # at work: the bubble above, the fox in place
        for _ in range(3):
            app.tick()
        await pilot.pause()
        app.working = (bird.region, str(bird.render()))
        app.busy = False
        app.tick()
        await pilot.press("/")                                        # the menu opens under the box: it stays
        await pilot.pause()
        menu = app.query_one("#commands")
        app.tick()
        await pilot.pause()
        app.on_menu = (bird.region, menu.region, box.region)
        await pilot.press("escape")
        await pilot.pause()
        ask.value = ""                                                # the "/" typed for the menu, gone
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app.after_enter = texts(app)
        ask.value = "word " * 40
        await pilot.pause()
        app.box = (ask.gutter_width, ask.wrapped_document.height)
        app.text_right = (ask.content_region.right, box.region.right)   # where typed text may reach
    welcome = "✻ hello\n\n  folders  ~/Papers [2024], ~/[b]drafts[/b]"   # a folder's name is not markup
    app = tui_run(lambda e: agent(home, notes, Reply("x [N1]."))[0], steps, welcome=lambda: welcome)
    height, bottom = app.sizes
    assert height == 34 and bottom == 40          # the log takes the screen: no fixed heading; 3 lines for the bar
    assert app.first == [("welcome", welcome)]                         # the welcome opens the log, and scrolls away
    assert app.lines[3].strip("│ ") == "folders  ~/Papers [2024], ~/[b]drafts[/b]"
    assert app.lines[0].strip().startswith("╭") and app.lines[1].strip("│ ") == "✻ hello"
    bird, box, status, drawn = app.perch           # lying on the box's top border, its tail over the right edge
    assert drawn == plain_frame(None) and bird.height == mascot.REST_LINES
    assert bird.right == box.right and bird.bottom == box.y + 2       # its last two lines: the border and the first
    assert status.right <= bird.x                                      # the status line never under it
    working, drawn = app.working
    assert drawn == plain_frame(0) and working.height == mascot.WORK_LINES
    assert (working.right, working.bottom) == (bird.right, bird.bottom)
    on_menu, menu, box_then = app.on_menu
    assert menu.y >= box_then.bottom                                   # the menu opens under the box, Claude Code's way
    assert (on_menu.right, on_menu.bottom) == (box_then.right, box_then.y + 2)   # the fox stays on the box it pushed up
    tail = [line.plain[-mascot.TAIL:] for line in mascot.frame(None)[-2:]]   # over the border and the first line
    assert app.lines[-3].endswith(tail[0]) and app.lines[-2].endswith(tail[1])
    assert app.lines[-2][:-mascot.TAIL].strip("│ ") == "❯ Ask about your files or the web"   # the text stays clear
    text_right, box_right = app.text_right                             # a long line wraps before the tail
    assert text_right <= box_right - mascot.TAIL
    assert app.after_enter == "" and app.box[0] == 0 and app.box[1] > 1


def test_slash_opens_the_command_menu_filters_it_and_runs_the_choice(home, notes):
    async def steps(app, pilot):
        from textual.widgets import OptionList
        menu = app.query_one("#commands", OptionList)
        await pilot.press("/")
        await pilot.pause()
        from peresearch.tui import COMMANDS
        assert menu.display and menu.option_count == len(COMMANDS)
        await pilot.press("backspace")                           # the box empties: the menu closes
        await pilot.pause()
        assert not menu.display
        await pilot.press("/", "f", "o")
        await pilot.pause()
        assert menu.option_count == 1 and menu.get_option_at_index(0).id == "/folders"
        await pilot.press("enter")
        await pilot.pause()
        assert not menu.display
        assert app.screen.id == "folders-window"
        app.folder_rows = [str(option.prompt) for option in app.screen.query_one(OptionList)._options]
    from peresearch.workspace import Workspace
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    ws.add(str(notes))
    app = tui_run(lambda e: agent(home, notes)[0], steps, workspace=ws)
    assert str(notes) in "\n".join(app.folder_rows) and str(notes) not in app.shown


def test_the_chosen_menu_row_has_the_pointer_and_the_one_under_the_mouse_bold(home, notes):
    """As in Claude Code: the chosen row starts with `❯` and takes the accent color, never a reversed bar; the row
    under the mouse is bold (the descriptions' column is checked on the rows themselves in test_tui_text.py)."""
    from rich.color import ColorType

    async def steps(app, pilot):
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        menu = app.query_one("#commands")
        menu.highlighted, menu._mouse_hovering_over = 1, 2
        await pilot.pause()
        app.rows = []
        for y in range(4):
            strip = menu.render_line(y)
            styles = [seg.style for seg in strip if seg.text.strip()]
            words = "".join(seg.text for seg in strip).split()
            accent = all(st.color is not None and st.color.type == ColorType.STANDARD and st.color.number == 4
                         for st in styles[:2])
            app.rows.append((words[0], words[1], any(st.reverse for st in styles), accent,
                             all(st.bold for st in styles)))
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    assert [(a, b) for a, b, *_ in app.rows] == [("/help", "the"), ("❯", "/folders"), ("/model", "choose"),
                                                ("/settings", "research")], app.rows
    assert not any(reverse for _, _, reverse, _, _ in app.rows)
    assert app.rows[1][3] and not app.rows[0][3]                        # the accent only on the chosen row
    assert app.rows[2][4] and not app.rows[3][4]                        # bold under the mouse


def test_new_forgets_the_earlier_questions(home, notes):
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


class Instant:
    """An agent that answers at once, and records what it was asked."""

    def __init__(self):
        self.asked, self.cancelled = [], threading.Event()

    def ask(self, question):
        self.asked.append(question)
        return Answer("ok", {}, Check(), 1)

    def cancel(self):
        self.cancelled.set()


def test_up_and_down_walk_through_the_questions_of_this_run(home, notes):
    """↑ on the box's first line goes one question back (staying on the first), ↓ on its last line one forward, and
    past the newest the box empties; on the other lines of a multi-line question they move the cursor. Checked on
    every sequence of up to five presses against that rule."""
    from peresearch.tui import Prompt
    asked = ["one", "two\nlines", "three"]

    async def steps(app, pilot):
        prompt = app.query_one("#ask", Prompt)
        await pilot.pause()
        await pilot.press("up", "down")                               # nothing asked yet: nothing happens
        app.before = prompt.value
        app.agent = Instant()
        for q in asked:
            prompt.value = q
            await pilot.press("enter")
            await settle(app, pilot, lambda: not app.busy)
        app.wrong = []
        for n in range(1, 6):
            for keys in itertools.product(("up", "down"), repeat=n):
                app.recall, prompt.value = None, ""
                pos, row = len(asked), 0                              # pos = len(asked): the fresh, empty box
                for key in keys:
                    await pilot.press(key)
                    text = asked[pos] if pos < len(asked) else ""
                    if key == "up" and row > 0 or key == "down" and row < text.count("\n"):
                        row += -1 if key == "up" else 1
                    elif key == "up" or pos < len(asked):
                        pos = max(0, pos - 1) if key == "up" else pos + 1
                        row = (asked[pos] if pos < len(asked) else "").count("\n")
                    want = (asked[pos] if pos < len(asked) else "", row)
                    if (prompt.value, prompt.cursor_location[0]) != want:
                        app.wrong.append((keys, (prompt.value, prompt.cursor_location[0]), want))
                        break
    app = tui_run(lambda e: Instant(), steps)
    assert app.before == "" and app.wrong == []


def test_the_log_and_the_status_line_follow_what_the_agent_does(home, notes, monkeypatch):
    """Each tool call is a `● call` line and its result a `⎿ result` line (control sequences stripped); the status
    line names what runs after the tool (searching, looking through folders, noting what is missing, reading), the
    model thinking or waiting, with a spinner frame per tick (about ten a second), the seconds since the question and
    the tokens so far; the index update alone says how far it is and that questions can still be asked; when nothing
    runs, a short message stays until its time is up, and then the line is empty."""
    from peresearch.tui import SPINNER
    clock = NS(now=1_000.0)
    monkeypatch.setattr(tui, "time", NS(time=lambda: clock.now, strftime=time.strftime, localtime=time.localtime))
    events = [("tool", 'search_notes("bpe")', "Searching"), ("result", "3 sources\x1b[31m", None),
              ("tool", "grep([/b]SuperBPE[b])", "Searching"), ("tool", "glob(*.md)", "Searching"),   # the model's
              ("result", "a line [/i] of a page", None),                     # text: shown as written, never markup
              ("tool", 'web_search("bpe")', "Searching"), ("tool", "tree(~/notes)", "Looking through folders"),
              ("tool", "read(bpe.md)", "Reading"),
              ("tool", "fetch(https://e.org)", "Reading"), ("tokens", "12345", None), ("think", "", "Thinking"),
              ("wait", "Waking the model up\x1b[2J", "Waking the model up"), ("result", "page read", None)]

    async def steps(app, pilot):
        await pilot.pause()
        app.status_of = lambda: str(app.status.render())
        app.seen, app.widgets = [], []
        for kind, detail, doing in events:
            app.on_agent_event(kind, detail)
            app.seen.append((app.doing, app.tokens))
            if kind in ("tool", "result"):
                app.widgets.append((kind, ("● " if kind == "tool" else "  ⎿  ") + tui.guard.sanitize(detail)))
        await pilot.pause()
        app.logged = [(" ".join(sorted(w.classes)), str(w.render())) for w in app.query("#log > *")][1:]
        app.busy, app.started, app.indexing = True, clock.now - 42.4, "Indexing"
        app.ticks = []
        for tokens in (12345, 12345, 0):
            app.tokens, frame = tokens, app.frame
            app.tick()
            app.ticks.append((app.status_of(), SPINNER[(frame + 1) % len(SPINNER)]))
        app.busy = False
        frame = app.frame
        app.tick()
        app.index_only = (app.status_of(), SPINNER[(frame + 1) % len(SPINNER)])
        app.index_progress, frame = (120, 6512), app.frame
        app.tick()
        app.index_far = (app.status_of(), SPINNER[(frame + 1) % len(SPINNER)])
        app.indexing = ""
        app.done()
        app.say("copied 5 characters", seconds=2)
        app.flashes = [app.status_of()]
        clock.now += 2                                               # its time is up
        app.tick()
        app.flashes.append(app.status_of())
        app.busy, frame = True, app.frame
        await pilot.pause(1.0)
        app.spun = app.frame - frame
        app.busy = False
        bird = lambda: str(app.query_one("#mascot").render())          # noqa: E731
        app.tick()
        app.rest = bird()
        app.busy, app.walk = True, []
        for _ in range(12):
            app.tick()
            app.walk.append(bird())
        app.busy, app.indexing = False, "Indexing"                     # an index update keeps it going
        app.tick()
        app.walk.append(bird())
        app.indexing = ""
        app.tick()
        app.back = bird()
    app = tui_run(lambda e: Instant(), steps)
    doing, tokens = "", 0
    for (kind, detail, label), got in zip(events, app.seen):
        doing, tokens = label or doing, int(detail) if kind == "tokens" else tokens
        assert got == (doing, tokens), (kind, detail, got)
    assert app.logged == app.widgets
    for (status, spin), tokens in zip(app.ticks, (12345, 12345, 0)):
        extra = " · 12,345 tokens" if tokens else ""
        assert status == f"{spin} Waking the model up… (42s{extra} · esc to interrupt)"
    assert app.ticks[0][1] != app.ticks[1][1]                          # one frame per tick
    assert app.index_only[0] == f"{app.index_only[1]} Indexing… (you can ask meanwhile)"
    assert app.index_far[0] == f"{app.index_far[1]} Indexing… 120/6,512 files (you can ask meanwhile)"
    assert app.flashes == ["copied 5 characters", ""]
    assert app.spun >= 2                                               # the spinner turns by itself
    ticks = round(mascot.STEP / tui.TICK)                              # 500 ms a frame, the owner's sketch
    assert app.rest == app.back == plain_frame(None)                   # at rest: no bubble
    assert app.walk == [plain_frame((i // ticks) % mascot.FRAMES) for i in range(1, 14)]


def test_a_question_is_shown_as_typed_and_esc_stops_it_at_once(home, notes):
    """The question goes in the log as typed (markup is not interpreted, control sequences are stripped) and the
    status says what it does; Esc while the model is still answering says it is interrupting and ends the question
    within a second, not when the answer arrives."""
    from peresearch.llm import Interrupted
    from peresearch.tui import Prompt, SPINNER

    class SlowLLM(ScriptedLLM):
        def answer(self, messages, tools=None, thinking=None, stop=None):
            end = time.time() + 30                               # a long answer on a slow model
            while time.time() < end:
                if stop and stop():
                    raise Interrupted("stopped")
                time.sleep(0.05)
            return super().answer(messages, tools, thinking)

    def make(on_event):
        return Agent(SlowLLM(Reply("never")), toolbox(home, notes), on_event=on_event)

    async def steps(app, pilot):
        app.query_one("#ask", Prompt).value = "is [b]this[/b] bold?\x1b[31m"
        await pilot.press("enter")
        await settle(app, pilot, lambda: app.doing == "Thinking")      # the models loaded, the model asked
        app.question = [(" ".join(sorted(w.classes)), str(w.render())) for w in app.query("#log > *")][1:]
        app.started = time.time()                                     # the shown seconds count from here
        frame = app.frame
        app.tick()
        app.status_seen = (str(app.status.render()), SPINNER[(frame + 1) % len(SPINNER)])
        await pilot.pause(0.3)
        app.pressed = time.time()
        await pilot.press("escape")
        app.said_on_esc = app.doing             # not `stopping`: the app's own quitting event (a string broke it)
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        app.took = time.time() - app.pressed
    app = tui_run(make, steps)
    assert app.question == [("question", "❯ is [b]this[/b] bold?")]
    assert app.status_seen[0] == f"{app.status_seen[1]} Thinking… (0s · esc to interrupt)"
    assert app.said_on_esc == "Interrupting"
    assert app.last.stopped == "cancelled" and app.took < 1.5 and "interrupted" in app.shown


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
        app.followed = bottom == log.max_scroll_y > 0                    # the log shows its newest line
        await pilot.press("pageup")
        up = log.scroll_y
        await pilot.press("pagedown")
        app.scrolled = (bottom, up, log.scroll_y)
    app = tui_run(lambda e: agent(home, notes)[0], steps, size=(100, 30))
    assert app.typed == "xin"                                # the click had moved focus: keys were lost
    assert isinstance(app.focus_after, Prompt)
    assert app.followed
    bottom, up, down = app.scrolled
    assert up < bottom and down == bottom                    # PageUp/PageDown scroll the conversation, at once


def test_pasted_lines_reach_the_model_whole(home, notes):
    """Textual's input keeps only the first line of a paste; the prompt shows a multi-line paste as a marker, as
    Claude Code does, and the model gets every line."""
    from textual import events

    from peresearch.tui import Prompt

    def make(on_event):
        a, llm = agent(home, notes, Reply("Seen."), Reply("Seen again."))
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
        await pilot.press("up")                                            # ↑ brings the question back, still a paste
        app.recalled = prompt.value
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and len(make.llm.seen) > 1)
    app = tui_run(make, steps)
    assert app.recalled == app.after_single
    again = [m["content"] for m in make.llm.seen[1][0] if m["role"] == "user"][-1]
    assert "KeyError: 'a'" in again and "\u2060" not in again
    assert app.shown_value.replace("\u2060", "") == "why does this fail: [Pasted text #1 +3 lines]"
    assert app.after_single.replace("\u2060", "").endswith("[Pasted text #1 +3 lines]one line")
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
        app.query_one("#ask", tui.Prompt).value = "research question"
        await pilot.press("enter")
        await pilot.pause()
        echo = app.query(".question").last()
        app.echo = (str(echo.render()), bool(echo.rich_style.reverse), bool(echo.rich_style.bold))
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    colors = {k: v.split()[0] for k, v in app.variables.items() if isinstance(v, str) and v.startswith("ansi_")}
    assert {k: v for k, v in colors.items() if v not in plain} == {}
    assert {k: v for k, v in colors.items() if "background" in k and v not in ("ansi_default", "transparent")} == {}
    assert app.screen_after == "Screen"
    assert app.selection.reverse
    assert app.echo == ("❯ research question", False, False)       # no reversed bar, not bold (Windows Terminal: bright)
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
    the prompt's, a paste marker as the text it shows, without its invisible mark), and Ctrl+Q quits."""
    from textual import events

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
        prompt.value = ""
        prompt.post_message(events.Paste("a\nb\n"))
        await pilot.pause()
        prompt.select_all()
        await pilot.press("ctrl+c")
        app.marker = app.clipboard
        app._clipboard = ""
        prompt.value = ""
        await pilot.press("ctrl+c")                                 # nothing selected: a reminder, no quitting
        await pilot.pause()
        app.reminded = app.flash
        await pilot.press("ctrl+q")
        await pilot.pause()
        app.running_after_quit = app.is_running
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    assert app.from_log.startswith("SuperBPE") and app.running_after_copy
    assert app.from_prompt == "copy me" and app.marker == "[Pasted text #1 +2 lines]"    # no U+2060 on the clipboard
    assert app.reminded == "select text with the mouse to copy it · ctrl+q quits"
    assert not app.running_after_quit


def test_a_model_that_does_not_answer_is_explained_and_retry_asks_again(home, notes):
    """The error names what is wrong and what to do; /retry sends the same question again."""
    class APIConnectionError(Exception):
        pass

    class Flaky(ScriptedLLM):
        fail = True

        def answer(self, messages, tools=None, thinking=None, stop=None):
            if Flaky.fail:
                raise APIConnectionError("Connection error.")
            return super().answer(messages, tools, thinking)

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
        app.retry_notice = app.flash
        prompt.value = "How does BPE work?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and "does not answer" in texts(app))
        app.error = texts(app)
        Flaky.fail = False
        prompt.value = "/retry"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
    app = tui_run(make, steps)
    assert "nothing to retry" in app.retry_notice and "nothing to retry" not in app.error
    assert "does not answer" in app.error and "/retry" in app.error and "APIConnectionError" not in app.error.split("does not answer")[0][-40:]
    assert app.last.text == "BPE merges pairs [N1]."
    sent = [m["content"] for m in make.llm.seen[-1][0] if m["role"] == "user"]
    assert sent[-1].endswith("How does BPE work?")


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
        app.said = [app.flash]
        app.last = None
        prompt.value = "And SuperBPE?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: not app.busy and app.last is not None)
        await settle(app, pilot, lambda: "then SuperBPE" in texts(app))
        app.shown_both = texts(app)
        await pilot.press("ctrl+r")
        await settle(app, pilot, lambda: "first I recall" not in texts(app))
        app.said.append(app.flash)
    app = tui_run(make, steps)
    assert app.said == ["the model's reasoning is shown (ctrl+r hides it)",
                        "the model's reasoning is hidden (ctrl+r shows it)"]
    assert "first I recall" not in app.hidden and "BPE merges pairs" in app.hidden
    assert "first I recall the notes" in app.shown_once and "Reasoning" in app.shown_once
    assert "then SuperBPE" in app.shown_both                         # new answers follow the setting
    assert "first I recall" not in app.shown and "then SuperBPE" not in app.shown


@pytest.mark.parametrize("busy", ["an update that stops when asked", "an update that does not stop",
                                  "a question waiting on the model"])
def test_quitting_returns_at_once_whatever_runs_in_the_background(home, notes, busy):
    """Ctrl+Q waited for the index worker (2026-10-03, the owner's folders: the update went on and Ctrl+C was
    needed): an update is asked to stop (what it did is kept, the next one goes on), a question is interrupted,
    work that cannot stop at once (an embedding on a CPU) does not hold the program open past a short grace, and
    work ending after the screen is gone touches nothing. The work left running is released when the test ends: a
    thread looping on time.sleep went on into later tests, and one that replaced the clock counted its sleeps as its
    own (test_llm after test_tui, 2026-10-07)."""
    started, seen, released = threading.Event(), {}, threading.Event()

    class Busy:
        def folders(self):
            return [notes]

        def update(self, stop=None, progress=None):
            started.set()
            end = time.time() + 20
            while time.time() < end and not released.is_set() and not (
                    stop and stop() and busy == "an update that stops when asked"):
                time.sleep(0.05)
            seen["stopped"] = stop is not None and stop()
            if seen["stopped"]:
                time.sleep(0.5)                              # commits its open batch: what the grace waits for
            raise tui.Stopped("asked to stop")

    class Stuck(DictModel):                                            # the model never answers, but hears the interrupt
        def answer(self, messages, tools=None, thinking=None, stop=None):
            started.set()
            while not stop():
                time.sleep(0.05)
            seen["stopped"] = True
            raise Interrupted("stopped")

    def make(on_event):
        return Agent(Stuck(), toolbox(home, notes), on_event=on_event)

    async def steps(app, pilot):
        if busy == "a question waiting on the model":
            app.query_one("#ask", tui.Prompt).value = "How does BPE work?"
            await pilot.press("enter")
        await settle(app, pilot, started.is_set)
        seen["t0"] = time.perf_counter()
        await pilot.press("ctrl+q")
    on_start = busy != "a question waiting on the model"
    app = None
    try:
        app = tui_run(make, steps, workspace=Busy() if on_start else None, index_on_start=on_start)
        assert started.is_set()
        check_quit(app, busy, seen)
    finally:
        released.set()
        left = app.background if app else []
        for t in left:
            t.join(5)
        assert not any(t.is_alive() for t in left), "work left running after the test"


def check_quit(app, busy, seen):
    took = time.perf_counter() - seen["t0"]
    assert app.background and all(t.daemon for t in app.background)    # Python waits for the others at exit
    if busy != "an update that does not stop":          # it stops when asked: over before the program, well in time
        assert seen.get("stopped") is True and not any(t.is_alive() for t in app.background)
        assert took < 1.0, f"quitting took {took:.1f} s"
    app.indexed()                       # work left behind ends after the screen is gone: nothing to update, no error
    app.done()                          # (NoMatches from #heading once failed a run, 2026-10-05)
    app.busy = True
    app.tick()                          # the spinner's timer firing once more (once failed a run in ten)
    assert took < 3.5, f"quitting took {took:.1f} s"                          # QUIT_GRACE at most, then left


@pytest.mark.parametrize("added", [0, 2])
def test_a_question_asked_while_the_folders_are_checked_is_answered(home, notes, added):
    """On opening, the index is checked in the background, the status line saying how far it is; a question asked
    meanwhile used to be refused with "still working on the previous question" (there was none), since both shared
    one busy flag. The check says one line when it changed something, and nothing at all otherwise."""
    import time as _t
    from types import SimpleNamespace

    class SlowWorkspace:
        def folders(self):
            return [notes]

        def update(self, stop=None, progress=None):
            progress(1, 3)
            _t.sleep(1.0)
            return SimpleNamespace(added=added, changed=0, removed=0, withheld={}, unchanged=3 - added, chunks=7,
                                   problems={})

    async def steps(app, pilot):
        from peresearch.tui import Prompt
        await pilot.pause(0.1)                                       # the check has started
        app.tick()
        app.checking = str(app.status.render())[2:]
        app.query_one("#ask", Prompt).value = "How does BPE work?"
        await pilot.press("enter")
        await settle(app, pilot, lambda: app.last is not None and not app.indexing)
    app = tui_run(lambda e: agent(home, notes, Reply("BPE merges pairs [N1]."))[0], steps,
                  workspace=SlowWorkspace(), index_on_start=True)
    assert app.checking == "Checking your folders for changes… 1/3 files (you can ask meanwhile)"
    assert "still working" not in app.shown
    assert app.last is not None and app.last.text == "BPE merges pairs [N1]."
    assert app.indexing == "" and app.index_progress is None          # the check finished
    said = [x for x in app.shown.splitlines() if "ndex" in x]
    assert said == []
    if added:
        assert app.flash == "Index updated: 2 added, 0 changed, 0 removed · 7 chunks from 3 files"


def test_resume_rows_line_up_and_long_questions_end_in_an_ellipsis(home, notes):
    """One line per conversation: the date, the count and the first question in columns; a question too long for
    the width is cut near the end with "…" instead of wrapping, and one with line breaks is kept on one line."""
    import json as _json

    from peresearch.agent import history_file
    from peresearch.tui import Prompt

    path = history_file("default")
    path.parent.mkdir(parents=True, exist_ok=True)
    firsts = ["short", "a question with\nseveral lines\ninside it",
              "a very long first question about tokenization " * 8, "twelve"]
    lines = []
    for n, (first, count) in enumerate(zip(firsts, (1, 2, 1, 12))):
        for i in range(count):
            lines.append({"time": 1_800_000_000 + n * 1000 + i, "conversation": f"c{n}",
                          "question": first if i == 0 else f"follow-up {i}", "answer": "ok", "sources": {}})
    lines.append({"time": 1_800_000_000 + 2002, "conversation": "c2", "name": "a named conversation"})
    path.write_text("\n".join(_json.dumps(x) for x in lines) + "\n")     # the long one named: its name, then the question

    async def steps(app, pilot):
        app.query_one("#ask", Prompt).value = "/resume"
        await pilot.press("enter")
        await pilot.pause()
        menu = app.screen.query_one("#session-list")
        app.rows = ["".join(seg.text for seg in menu.render_line(y)).rstrip() for y in range(menu.size.height)]
        app.width = menu.size.width
    app = tui_run(lambda e: agent(home, notes)[0], steps, size=(90, 30))
    rows = [r for r in app.rows if r.strip()]
    assert len(rows) == 4                                             # one line each: nothing wrapped
    starts = {r.index(key) for r, key in zip(rows, ("twelve", "a named conversation", "a question with", "short"))}
    assert len(starts) == 1                                           # the questions start in one column
    long_row = rows[1]
    assert long_row.endswith("…") and len(long_row) <= app.width and "a named conversation · a very long" in long_row
    assert "a question with several lines inside it" in rows[2]
    assert "12 questions" in rows[0] and "1 question " in rows[3] + " "


def plain_frame(pose):
    return "\n".join(line.plain for line in mascot.frame(pose))


def screen_lines(app) -> list[str]:
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


def test_no_key_hints_line_and_keys_lists_them(home, notes):
    """The owner's choice (2026-09-30): no line of key hints under the prompt (at 100 columns it was already cut
    before "ctrl+q quit"); /keys shows the keys."""
    async def steps(app, pilot):
        await pilot.pause()
        await pilot.press(*"/keys", "enter")
        await pilot.pause(0.2)
        from textual.widgets import Markdown
        app.keys_body = "\n".join(w.source for w in app.screen.query("Markdown, .columns"))   # columns: /help
        await pilot.press("escape")
        await pilot.pause()
        app.lines = screen_lines(app)
    app = tui_run(lambda e: agent(home, notes)[0], steps, size=(100, 40))
    assert not app.query("#hints")
    assert app.lines[-1].startswith("╰"), app.lines[-3:]               # the prompt's border is the last line
    for key in ("Enter", "Ctrl+Enter", "Esc", "Ctrl+R", "Ctrl+C", "Ctrl+Q", "PgUp"):
        assert key in app.keys_body and key not in app.shown, key


def test_a_long_code_line_wraps_instead_of_being_cut(home, notes):
    code = "print('a fairly long line of code that will not fit a narrow terminal at all, ending here')"

    async def steps(app, pilot):
        await pilot.pause()
        await pilot.press(*"q", "enter")
        await settle(app, pilot, lambda: not app.busy and len(app.pages) > 0)
        await pilot.pause(0.2)
        app.lines = screen_lines(app)
    app = tui_run(lambda e: agent(home, notes, Reply(f"Code:\n\n```python\n{code}\n```\n"))[0], steps, size=(50, 30))
    joined = "".join(line.strip().removeprefix("│") for line in app.lines)   # the code block's rule on its left
    assert "".join(code.split()) in "".join(joined.split()), app.lines


def test_the_cursor_never_stops_inside_a_paste_marker(home, notes):
    """Every way the cursor moves — word jumps, ↑/↓ between lines, a click, a selection — ends at an edge of a marker:
    the way it was heading, or the nearer edge when it has no heading (a click, another line); a selection that
    touches a marker covers all of it."""
    from textual import events
    from textual.widgets.text_area import Selection

    from peresearch.tui import Prompt

    async def steps(app, pilot):
        await pilot.pause()
        prompt = app.query_one("#ask", Prompt)
        prompt.value = "ab "
        prompt.post_message(events.Paste("one\ntwo\nthree\n"))
        await pilot.pause()
        await pilot.press(*" cd", "ctrl+j", *"0123456789012345678901234567890123456789")
        start = 3
        end = start + len(prompt.document.get_line(0)) - len("ab  cd")
        app.span = (start, end)
        seen = []
        prompt.move_cursor((0, len(prompt.document.get_line(0))))
        for _ in range(8):                                           # word jumps back, then forward
            await pilot.press("ctrl+left")
            seen.append(("ctrl+left", prompt.cursor_location))
        for _ in range(8):
            await pilot.press("ctrl+right")
            seen.append(("ctrl+right", prompt.cursor_location))
        prompt.move_cursor((1, start + 3))                            # ↑ from under the marker's start
        await pilot.press("up")
        seen.append(("up near the start", prompt.cursor_location))
        prompt.move_cursor((1, end - 3))                              # ↑ from under its end
        await pilot.press("up")
        seen.append(("up near the end", prompt.cursor_location))
        await pilot.click("#ask", offset=(start + 4, 0))             # the box's border and `❯` are outside #ask
        seen.append(("click near the start", prompt.cursor_location))
        await pilot.click("#ask", offset=(end - 2, 0))
        seen.append(("click near the end", prompt.cursor_location))
        await pilot.click("#ask", offset=((start + end) // 2, 0))   # the marker is 26 wide: a tie
        seen.append(("click in the middle", prompt.cursor_location))
        prompt.selection = Selection((0, 1), (0, start + 5))
        seen.append(("select into it", prompt.selection))
        prompt.selection = Selection((0, end - 2), (0, 1))
        seen.append(("select back into it", prompt.selection))
        app.seen = seen
    app = tui_run(lambda e: agent(home, notes)[0], steps)
    start, end = app.span
    inside = [(how, at) for how, at in app.seen[:16] if at[0] == 0 and start < at[1] < end]
    assert not inside, (app.span, inside)
    got = dict(app.seen[16:])
    assert got["up near the start"] == (0, start) and got["up near the end"] == (0, end), (app.span, got)
    assert got["click near the start"] == (0, start) and got["click near the end"] == (0, end), (app.span, got)
    assert got["click in the middle"] == (0, start), (app.span, got)       # a tie goes to the start
    assert tuple(got["select into it"]) == ((0, 1), (0, end)), (app.span, got)
    assert tuple(got["select back into it"]) == ((0, end), (0, 1)), (app.span, got)
