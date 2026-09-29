"""The terminal interface, full screen, laid out the way Claude Code's is.

- The conversation scrolls above; each question is shown as `> question`, each tool the agent calls as
  `● tool(args)` with a `⎿ result` line under it, then the cited answer, its sources and the citation check.
- A status line while the agent works: a spinner, what it is doing, elapsed seconds and tokens so far.
- The prompt at the bottom: Enter asks, ↑/↓ recall earlier questions, `/` opens the command menu (↑/↓ to pick,
  Tab or Enter to take it, Esc to close), Esc during a question interrupts it at the next step. PageUp/PageDown
  scroll the conversation; the conversation never takes the focus, so typing always reaches the prompt, also
  after a click on it.
- Folders are managed here too: /add <folder> (Tab completes the path), /remove, /folders, /index. On start, the
  index is brought up to date with the declared folders (only new or changed files are read).
Colors are the terminal's own: the default foreground and background and its 16-color palette, never RGB, so
the terminal's color scheme (light or dark) decides how everything looks; muted text is dimmed, not blended.
The agent runs in a worker thread. Everything that came from a file, a page or the model is stripped of terminal
control sequences before it is shown.
"""

import time

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.theme import Theme
from textual.widgets import Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from peresearch import guard
from peresearch.workspace import complete, summary

COMMANDS = {
    "/help": "keys and commands",
    "/add": "<folder> let peresearch read a folder (Tab completes the path), then index it",
    "/remove": "<folder> stop reading a folder",
    "/folders": "the folders peresearch may read",
    "/index": "bring the index up to date with the folders",
    "/new": "start a fresh conversation (earlier questions no longer sent as context)",
    "/sources": "every source of the last answer, in full",
    "/exit": "quit",
}
SPINNER = "∗✢✻✢"
# The terminal's palette only. Everything that would need a shade of the background (panels, cursors, selections)
# is the default colors, reversed or dimmed, so the same theme reads on a light and on a dark scheme.
TERMINAL = Theme(
    name="terminal", ansi=True, dark=True,
    primary="ansi_blue", secondary="ansi_cyan", accent="ansi_blue", warning="ansi_yellow", error="ansi_red",
    success="ansi_green", foreground="ansi_default", background="ansi_default", surface="ansi_default",
    panel="ansi_default", boost="ansi_default",
    variables={
        "ansi-background": "ansi_default", "ansi-foreground": "ansi_default",
        "border": "ansi_blue", "border-blurred": "ansi_blue",
        "input-cursor-background": "ansi_default", "input-cursor-foreground": "ansi_default",
        "input-cursor-text-style": "reverse",
        "input-selection-background": "ansi_blue", "input-selection-foreground": "ansi_default",
        "screen-selection-background": "ansi_blue", "screen-selection-foreground": "ansi_default",
        "block-cursor-background": "ansi_blue", "block-cursor-foreground": "ansi_default",
        "block-cursor-blurred-background": "ansi_default", "block-cursor-blurred-foreground": "ansi_default",
        "block-hover-background": "ansi_default",
        "scrollbar": "ansi_blue", "scrollbar-hover": "ansi_cyan", "scrollbar-active": "ansi_cyan",
        "scrollbar-background": "ansi_default", "scrollbar-background-hover": "ansi_default",
        "scrollbar-background-active": "ansi_default", "scrollbar-corner-color": "ansi_default",
        "footer-background": "ansi_default", "footer-key-foreground": "ansi_blue",
    },
)


def render(answer) -> str:
    """The answer, then its sources, then what the citation check found (Markdown)."""
    parts = [guard.sanitize(answer.text or "_(no answer)_"), ""]
    if answer.sources:
        parts.append("**Sources**")
        parts += [f"- `{s.id}` {guard.sanitize(s.where)}" + (f" — {guard.sanitize(s.title)}" if s.title else "")
                  for s in answer.sources.values()]
    c = answer.check
    if c.unknown_ids:
        parts.append(f"\n> **Check:** cited but never retrieved: {', '.join(c.unknown_ids)}")
    if c.unsupported_quotes:
        parts.append("\n> **Check:** quotes not found in the cited sources: "
                     + "; ".join(f"“{guard.sanitize(q)[:80]}”" for q in c.unsupported_quotes))
    if c.uncited:
        parts.append("\n> **Check:** the answer cites no source")
    if answer.stopped:
        parts.append(f"\n> stopped: {answer.stopped}")
    parts.append(f"\n<sub>{answer.steps} model calls · {answer.prompt_tokens + answer.completion_tokens:,} tokens</sub>")
    return "\n".join(parts)


class Prompt(Input):
    """The input line; ↑/↓, Tab and Esc go to the app (history, command menu, interrupt).

    Backspace is handled here, in order with the typed characters. A Vietnamese input method types the raw
    letters, then sends backspaces and the accented text in one burst ("chao", ⌫, ⌫, "ào"); Textual inserts
    characters at once but runs the backspace binding later, so the burst came out as "chao" or "chaà"."""

    BINDINGS = [Binding("up", "app.up", show=False), Binding("down", "app.down", show=False),
                Binding("tab", "app.complete", show=False), Binding("escape", "app.escape", show=False),
                Binding("pageup", "app.page(-1)", show=False), Binding("pagedown", "app.page(1)", show=False)]

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "backspace":
            event.stop()
            event.prevent_default()
            self.action_delete_left()
            return
        await super()._on_key(event)


class Chat(App):
    CSS = """
    Screen { layout: vertical; }
    Screen { background: ansi_default; color: ansi_default; }
    #heading { height: auto; padding: 0 1; border: round ansi_blue; }
    #log { height: 1fr; padding: 0 1; background: ansi_default; }
    .question { margin: 1 0 0 0; color: ansi_default; text-style: bold; padding: 0 1; }
    .tool { color: ansi_green; margin: 1 0 0 0; }
    .result { color: ansi_default; text-style: dim; }
    .note { color: ansi_default; text-style: dim; margin: 1 0 0 0; }
    #status { height: 1; padding: 0 1; color: ansi_yellow; }
    #commands { height: auto; max-height: 8; display: none; border: round ansi_blue; background: ansi_default; }
    #ask { border: round ansi_blue; background: ansi_default; color: ansi_default; }
    #hints { height: 1; padding: 0 1; color: ansi_default; text-style: dim; }
    """
    BINDINGS = [Binding("ctrl+c", "quit", "Quit", priority=True)]

    def __init__(self, make_agent, heading=lambda: "peresearch", workspace=None, index_on_start: bool = True):
        super().__init__()
        self.make_agent, self.heading, self.ws = make_agent, heading, workspace
        self.index_on_start = index_on_start and workspace is not None
        self.agent, self.busy, self.last, self.pending_new = None, False, None, False
        self.asked, self.recall = [], None
        self.started, self.doing, self.tokens, self.frame = 0.0, "", 0, 0

    def compose(self) -> ComposeResult:
        yield Static(f"✻ {self.heading()}", id="heading")
        yield VerticalScroll(id="log", can_focus=False)
        yield Static("", id="status")
        yield OptionList(id="commands")
        yield Prompt(placeholder="> Ask about your files or the web", id="ask")
        yield Static("/ commands · ↑↓ earlier questions · pgup/pgdn scroll · esc interrupt · ctrl+c quit", id="hints")

    def on_mount(self) -> None:
        self.register_theme(TERMINAL)
        self.theme = "terminal"
        self.query_one("#ask", Prompt).focus()
        self.set_interval(0.1, self.tick)
        if self.index_on_start:
            if self.ws.folders():
                self.reindex("Checking your folders for changes")
            else:
                self.note("No folders yet: /add <folder> lets peresearch read one.")

    # --- the conversation ---

    def add(self, widget) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)

    def note(self, text: str) -> None:
        self.add(Static(guard.sanitize(text), classes="note"))

    def tick(self) -> None:
        status = self.query_one("#status", Static)
        if not self.busy:
            status.update("")
            return
        self.frame += 1
        spin = SPINNER[self.frame % len(SPINNER)]
        extra = f" · {self.tokens:,} tokens" if self.tokens else ""
        status.update(f"{spin} {guard.sanitize(self.doing)}… ({time.time() - self.started:.0f}s{extra} · esc to interrupt)")

    def on_agent_event(self, kind: str, detail: str) -> None:
        if kind == "tool":
            self.add(Static(f"● {guard.sanitize(detail)}", classes="tool"))
            self.doing = "Searching" if detail.startswith(("search", "grep", "glob", "web")) else "Reading"
        elif kind == "result":
            self.add(Static(f"  ⎿  {guard.sanitize(detail)}", classes="result"))
        elif kind == "tokens":
            self.tokens = int(detail)
        elif kind == "think":
            self.doing = "Thinking"
        elif kind == "wait":
            self.doing = guard.sanitize(detail)

    # --- asking ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        menu = self.query_one("#commands", OptionList)
        if menu.display and menu.highlighted is not None and event.value.strip().startswith("/"):
            chosen = menu.get_option_at_index(menu.highlighted).id
            if chosen.startswith("path:"):                  # a path picked from the Tab suggestions
                self.set_prompt(event.value.split(" ")[0] + " " + chosen[5:])
                self.hide_menu()
                return
            if event.value.strip().split(" ")[0] != chosen:
                event.input.value = chosen
                if chosen in ("/add", "/remove"):          # these need a folder: wait for it
                    self.set_prompt(chosen + " ")
                    self.hide_menu()
                    return
        q = event.input.value.strip()
        event.input.value = ""
        self.hide_menu()
        if not q:
            return
        if q.startswith("/"):
            self.command(q)
            return
        if self.busy:
            self.note("still working on the previous question (esc to interrupt it)")
            return
        self.asked.append(q)
        self.recall = None
        self.start("Starting")
        self.add(Static(f"> {guard.sanitize(q)}", classes="question"))
        self.run_agent(q)

    @work(thread=True, exclusive=True)
    def run_agent(self, question: str) -> None:
        def event(kind, detail):
            self.call_from_thread(self.on_agent_event, kind, detail)
        try:
            if self.agent is None:
                self.call_from_thread(self.on_agent_event, "wait", "Loading the local models")
                self.agent = self.make_agent(event)
                if self.pending_new:
                    self.agent.new_session()
            answer = self.agent.ask(question)
            self.last = answer
            self.call_from_thread(self.add, Markdown(render(answer)))
        except Exception as e:
            self.call_from_thread(self.add, Markdown(f"> **Error:** {guard.sanitize(type(e).__name__ + ': ' + str(e))[:500]}"))
        finally:
            self.call_from_thread(self.done)

    def done(self) -> None:
        self.busy = False
        self.query_one("#heading", Static).update(f"✻ {self.heading()}")

    @work(thread=True, exclusive=True)
    def reindex(self, doing: str) -> None:
        self.call_from_thread(self.start, doing)
        try:
            r = self.ws.update()
            quiet = not (r.added or r.changed or r.removed or r.withheld)
            text = f"Index up to date: {r.chunks:,} chunks from {r.added + r.changed + r.unchanged} files." if quiet \
                else summary(r, limit=5)
            self.call_from_thread(self.note, text)
        except Exception as e:
            self.call_from_thread(self.note, f"indexing failed: {type(e).__name__}: {e}"[:300])
        finally:
            self.call_from_thread(self.done)

    def start(self, doing: str) -> None:
        self.busy, self.started, self.doing, self.tokens = True, time.time(), doing, 0

    # --- commands ---

    def command(self, text: str) -> None:
        name = text.split(" ")[0]
        if name == "/exit":
            self.exit()
        elif name == "/help":
            self.add(Markdown("**Keys** — Enter ask · ↑↓ earlier questions · `/` commands (↑↓ pick, Tab/Enter take, "
                              "Esc close) · PgUp/PgDn scroll · Esc interrupt · Ctrl+C quit\n\n**Commands**\n"
                              + "\n".join(f"- `{c}` {d}" for c, d in COMMANDS.items())))
        elif name == "/new":
            if self.busy:
                self.note("wait for the current question to finish, or press esc")
                return
            if self.agent is not None:
                self.agent.new_session()
            self.pending_new = self.agent is None
            self.query_one("#log", VerticalScroll).remove_children()
            self.note("New conversation: earlier questions are no longer sent as context.")
        elif name == "/sources":
            if not self.last or not self.last.sources:
                self.note("no sources yet")
                return
            self.add(Markdown("\n\n".join(f"**`{s.id}`** {guard.sanitize(s.where)}\n\n> "
                                          + guard.sanitize(s.text[:600]).replace("\n", "\n> ")
                                          for s in self.last.sources.values())))
        elif name == "/folders":
            folders = self.ws.folders() if self.ws else []
            self.note("\n".join(map(str, folders)) or "no folders declared: /add <folder>")
        elif name in ("/add", "/remove", "/index"):
            if self.ws is None:
                self.note("folders cannot be changed here")
                return
            if self.busy:
                self.note("wait for the current work to finish, or press esc")
                return
            arg = text[len(name):].strip()
            if name == "/index":
                self.reindex("Indexing")
            elif not arg:
                self.note(f"usage: {name} <folder>")
            elif name == "/add":
                try:
                    self.note(f"Added {self.ws.add(arg)}")
                    self.reindex("Indexing the new folder")
                except ValueError as e:
                    self.note(str(e))
            else:
                self.note(f"Removed {arg}" if self.ws.remove(arg) else f"{arg} was not a declared folder")
                self.reindex("Dropping its chunks from the index")
        else:
            self.note(f"unknown command {name}; / for the list")

    # --- the command menu and the question history ---

    def on_input_changed(self, event: Input.Changed) -> None:
        v = event.value
        if v.startswith(("/add ", "/remove ")):
            return
        if v.startswith("/") and " " not in v:
            matches = [c for c in COMMANDS if c.startswith(v)]
            menu = self.query_one("#commands", OptionList)
            menu.clear_options()
            menu.add_options([Option(f"{c}  [dim]{d}[/dim]", id=c) for c, d in COMMANDS.items() if c in matches])
            menu.display = bool(matches)
            if matches:
                menu.highlighted = 0
        else:
            self.hide_menu()

    def hide_menu(self) -> None:
        self.query_one("#commands", OptionList).display = False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        prompt = self.query_one("#ask", Prompt)
        self.hide_menu()
        if event.option.id.startswith("path:"):
            self.set_prompt(prompt.value.split(" ")[0] + " " + event.option.id[5:])
        elif event.option.id in ("/add", "/remove"):
            self.set_prompt(event.option.id + " ")
        else:
            prompt.value = ""
            self.command(event.option.id)
        prompt.focus()

    def action_up(self) -> None:
        menu = self.query_one("#commands", OptionList)
        if menu.display:
            menu.action_cursor_up()
        elif self.asked:
            self.recall = len(self.asked) - 1 if self.recall is None else max(0, self.recall - 1)
            self.set_prompt(self.asked[self.recall])

    def action_down(self) -> None:
        menu = self.query_one("#commands", OptionList)
        if menu.display:
            menu.action_cursor_down()
        elif self.recall is not None:
            self.recall += 1
            if self.recall >= len(self.asked):
                self.recall = None
            self.set_prompt(self.asked[self.recall] if self.recall is not None else "")

    def action_complete(self) -> None:
        menu = self.query_one("#commands", OptionList)
        value = self.query_one("#ask", Prompt).value
        if menu.display and menu.highlighted is not None:
            chosen = menu.get_option_at_index(menu.highlighted).id
            self.set_prompt((value.split(" ")[0] + " " + chosen[5:]) if chosen.startswith("path:") else chosen + " ")
            self.hide_menu()
        elif value.startswith(("/add ", "/remove ")):
            cmd, _, partial = value.partition(" ")
            if cmd == "/remove":
                matches = [str(f) + "/" for f in (self.ws.folders() if self.ws else []) if str(f).startswith(partial)]
            else:
                matches = complete(partial)
            if len(matches) == 1:
                self.set_prompt(f"{cmd} {matches[0]}")
            elif matches:
                menu.clear_options()
                menu.add_options([Option(m, id="path:" + m) for m in matches[:50]])
                menu.display, menu.highlighted = True, 0

    def action_page(self, direction: int) -> None:
        log = self.query_one("#log", VerticalScroll)
        (log.scroll_page_down if direction > 0 else log.scroll_page_up)(animate=False)

    def action_escape(self) -> None:
        menu = self.query_one("#commands", OptionList)
        if menu.display:
            self.hide_menu()
        elif self.busy and self.agent is not None:
            self.agent.cancel()
            self.doing = "Interrupting"

    def set_prompt(self, text: str) -> None:
        prompt = self.query_one("#ask", Prompt)
        prompt.value = text
        prompt.cursor_position = len(text)
