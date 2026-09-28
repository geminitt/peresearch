"""The terminal interface: ask, watch the agent work, read the cited answer. Built on Textual.

The agent runs in a worker thread; its events (tool calls, waiting for a cold model) show in the status line.
Every piece of text that came from a file, a web page or the model is stripped of terminal control sequences
before it is shown.
"""

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Markdown, Static

from peresearch import guard


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
        parts.append(f"\n> stopped by the {answer.stopped} limit")
    parts.append(f"\n<sub>{answer.steps} model calls · {answer.prompt_tokens + answer.completion_tokens:,} tokens</sub>")
    return "\n".join(parts)


class Chat(App):
    CSS = """
    #log { height: 1fr; }
    #status { height: 1; color: $text-muted; }
    .question { color: $accent; text-style: bold; margin: 1 0 0 0; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, make_agent, title: str = "peresearch"):
        super().__init__()
        self.make_agent, self.heading = make_agent, title
        self.agent = None
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Static(self.heading, id="heading")
        yield VerticalScroll(id="log")
        yield Static("", id="status")
        yield Input(placeholder="Ask a question (your files first, then the web) — /exit to quit", id="ask")

    def on_mount(self) -> None:
        self.query_one("#ask", Input).focus()

    def status(self, text: str) -> None:
        self.query_one("#status", Static).update(guard.sanitize(text))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        q = event.value.strip()
        event.input.value = ""
        if not q:
            return
        if q in ("/exit", "/quit"):
            self.exit()
            return
        if self.busy:
            self.status("still working on the previous question")
            return
        self.busy = True
        log = self.query_one("#log", VerticalScroll)
        log.mount(Static(guard.sanitize(q), classes="question"))
        log.scroll_end(animate=False)
        self.run_agent(q)

    @work(thread=True, exclusive=True)
    def run_agent(self, question: str) -> None:
        def event(kind, detail):
            self.call_from_thread(self.status, f"{kind}: {detail}" if detail else kind)
        try:
            if self.agent is None:
                self.call_from_thread(self.status, "loading the local models…")
                self.agent = self.make_agent(event)
            answer = self.agent.ask(question)
            self.call_from_thread(self.show, render(answer))
        except Exception as e:
            self.call_from_thread(self.show, f"> **Error:** {guard.sanitize(type(e).__name__ + ': ' + str(e))[:500]}")
        finally:
            self.call_from_thread(self.done)

    def show(self, markdown: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(Markdown(markdown))
        log.scroll_end(animate=False)

    def done(self) -> None:
        self.busy = False
        self.status("")
