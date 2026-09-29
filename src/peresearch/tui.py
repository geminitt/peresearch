"""The terminal interface, full screen, laid out the way Claude Code's is.

- The conversation scrolls above; each question is shown as `> question`, each tool the agent calls as
  `● tool(args)` with a `⎿ result` line under it, then the cited answer, its sources and the citation check.
- A status line while the agent works: a spinner, what it is doing, elapsed seconds and tokens so far.
- The prompt at the bottom: Enter asks, ↑/↓ recall earlier questions, `/` opens the command menu (↑/↓ to pick,
  Tab or Enter to take it, Esc to close), Esc during a question interrupts it at the next step.
The agent runs in a worker thread. Everything that came from a file, a page or the model is stripped of terminal
control sequences before it is shown.
"""

import time

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from peresearch import guard

COMMANDS = {
    "/help": "keys and commands",
    "/new": "start a fresh conversation (earlier questions no longer sent as context)",
    "/sources": "every source of the last answer, in full",
    "/folders": "the folders peresearch may read",
    "/exit": "quit",
}
SPINNER = "✻✢✳∗✳✢"


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
    """The input line; ↑/↓, Tab and Esc go to the app (history, command menu, interrupt)."""

    BINDINGS = [Binding("up", "app.up", show=False), Binding("down", "app.down", show=False),
                Binding("tab", "app.complete", show=False), Binding("escape", "app.escape", show=False)]


class Chat(App):
    CSS = """
    Screen { layout: vertical; }
    #heading { height: auto; padding: 0 1; border: round $accent; color: $text; }
    #log { height: 1fr; padding: 0 1; }
    .question { margin: 1 0 0 0; background: $boost; color: $text; text-style: bold; padding: 0 1; }
    .tool { color: $accent; margin: 1 0 0 0; }
    .result { color: $text-muted; }
    .note { color: $text-muted; margin: 1 0 0 0; }
    #status { height: 1; padding: 0 1; color: $warning; }
    #commands { height: auto; max-height: 8; display: none; border: round $accent; }
    #ask { border: round $accent; }
    #hints { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [Binding("ctrl+c", "quit", "Quit", priority=True)]

    def __init__(self, make_agent, title: str = "peresearch", folders=None):
        super().__init__()
        self.make_agent, self.heading, self.folders = make_agent, title, folders or []
        self.agent, self.busy, self.last, self.pending_new = None, False, None, False
        self.asked, self.recall = [], None
        self.started, self.doing, self.tokens, self.frame = 0.0, "", 0, 0

    def compose(self) -> ComposeResult:
        yield Static(f"✻ {self.heading}", id="heading")
        yield VerticalScroll(id="log")
        yield Static("", id="status")
        yield OptionList(id="commands")
        yield Prompt(placeholder="> Ask about your files or the web", id="ask")
        yield Static("/ commands · ↑↓ earlier questions · esc interrupt · ctrl+c quit", id="hints")

    def on_mount(self) -> None:
        self.query_one("#ask", Prompt).focus()
        self.set_interval(0.1, self.tick)

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
            if event.value.strip().split(" ")[0] != chosen:
                event.input.value = chosen
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
        self.busy, self.started, self.doing, self.tokens = True, time.time(), "Starting", 0
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

    # --- commands ---

    def command(self, text: str) -> None:
        name = text.split(" ")[0]
        if name == "/exit":
            self.exit()
        elif name == "/help":
            self.add(Markdown("**Keys** — Enter ask · ↑↓ earlier questions · `/` commands (↑↓ pick, Tab/Enter take, "
                              "Esc close) · Esc interrupt · Ctrl+C quit\n\n**Commands**\n"
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
            self.note("\n".join(map(str, self.folders)) or "no folders declared (peresearch add <folder>)")
        else:
            self.note(f"unknown command {name}; / for the list")

    # --- the command menu and the question history ---

    def on_input_changed(self, event: Input.Changed) -> None:
        v = event.value
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
        prompt.value = ""
        self.hide_menu()
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
        if menu.display and menu.highlighted is not None:
            self.set_prompt(menu.get_option_at_index(menu.highlighted).id + " ")
            self.hide_menu()

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
