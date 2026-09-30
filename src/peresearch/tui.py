"""The terminal interface, full screen, laid out the way Claude Code's is.

- The conversation scrolls above; each question is shown as `❯ question`, each tool the agent calls as
  `● tool(args)` with a `⎿ result` line under it, then the cited answer, its sources and the citation check.
- A status line while the agent works: a spinner, what it is doing, elapsed seconds and tokens so far.
- The prompt at the bottom: Enter asks, ↑/↓ recall earlier questions, `/` opens the command menu (↑/↓ to pick,
  Tab or Enter to take it, Esc to close), Esc during a question interrupts it at the next step. PageUp/PageDown
  scroll the conversation; the conversation never takes the focus, so typing always reaches the prompt, also
  after a click on it. No line of key hints (the owner's choice): /keys lists the keys.
- Nothing needs a minimum width (as in Claude Code): text, menu descriptions and code lines wrap to the terminal.
- Folders are managed here too: /add <folder> (Tab completes the path), /remove, /folders, /index. On start, the
  index is brought up to date with the declared folders (only new or changed files are read).
- Before the agent reads anything outside the declared folders, a dialog asks: yes this once, yes for that folder
  for the rest of the session, or no (Esc).
Colors are the terminal's own, so its color scheme (light or dark) decides how everything looks: the default
foreground and background and the six plain hues, never RGB, 256 colors, black, white or the bright colors (black
or white is the background in some scheme, the bright ones are grays in Solarized). Muted text is dimmed; the
cursor, selections and the chosen menu row are the default colors reversed; nothing is drawn on a colored background.
The agent runs in a worker thread. Everything that came from a file, a page or the model is stripped of terminal
control sequences before it is shown.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from rich.style import Style as RichStyle
from pygments.token import Token
from rich.cells import cell_len
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.highlight import ANSIDarkHighlightTheme, highlight
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.theme import Theme
from textual.message import Message
from textual.widgets import Markdown, OptionList, Static, TextArea
from textual.widgets._markdown import MarkdownFence
from textual.widgets.option_list import Option

from peresearch import guard
from peresearch.agent import _CITE, conversations, history_file, read_turns
from peresearch.tools import Source
from peresearch.workspace import complete, summary

COMMANDS = {
    "/help": "the commands",
    "/keys": "the keys",
    "/add": "<folder> let peresearch read a folder (Tab completes the path), then index it",
    "/remove": "<folder> stop reading a folder",
    "/folders": "the folders peresearch may read",
    "/index": "bring the index up to date with the folders",
    "/new": "start a fresh conversation (earlier questions no longer sent as context)",
    "/resume": "pick an earlier conversation of this project and continue it",
    "/retry": "ask the last question again",
    "/sources": "every source of the last answer, in full",
    "/exit": "quit",
}
KEYS = {
    "Enter": "ask", "Ctrl+Enter": "new line", "↑ ↓": "earlier questions (in the menu: pick)",
    "/": "the command menu (Tab or Enter takes the chosen command, Esc closes it)",
    "PgUp PgDn": "scroll the conversation", "Esc": "interrupt the question",
    "Ctrl+R": "show or hide the model's reasoning", "Ctrl+C": "copy the text selected with the mouse",
    "Ctrl+Q": "quit",
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
        "input-selection-background": "ansi_default", "input-selection-foreground": "ansi_default",
        "screen-selection-background": "ansi_default", "screen-selection-foreground": "ansi_default",
        "link-color-hover": "ansi_blue", "link-background-hover": "ansi_default", "link-style-hover": "bold underline",
        "markdown-h2-color": "ansi_blue",
        "block-cursor-background": "ansi_default", "block-cursor-foreground": "ansi_default",
        "block-cursor-blurred-background": "ansi_default", "block-cursor-blurred-foreground": "ansi_default",
        "block-hover-background": "ansi_default",
        "scrollbar": "ansi_blue", "scrollbar-hover": "ansi_cyan", "scrollbar-active": "ansi_cyan",
        "scrollbar-background": "ansi_default", "scrollbar-background-hover": "ansi_default",
        "scrollbar-background-active": "ansi_default", "scrollbar-corner-color": "ansi_default",
        "footer-background": "ansi_default", "footer-key-foreground": "ansi_blue",
    },
)


WHY = {
    "APIConnectionError": "The model server does not answer. If it is the local one, start it with `pixi run "
                          "chat-local`; a Modal endpoint may still be waking up.",
    "APITimeoutError": "The model server took too long to answer.",
    "AuthenticationError": "The model server refused the key (`PERESEARCH_LLM_KEY`, set with `peresearch setup`).",
    "PermissionDeniedError": "The model server refused the key (`PERESEARCH_LLM_KEY`, set with `peresearch setup`).",
    "NotFoundError": "The model server has no such path or model: check `PERESEARCH_LLM_URL` (ending in /v1) and "
                     "`PERESEARCH_LLM_MODEL`.",
    "RateLimitError": "The model server is limiting requests; wait a moment.",
    "InternalServerError": "The model server failed while answering.",
    "StreamCut": "The connection to the model server was cut mid-answer, again and again.",
}


def explain(error: Exception) -> str:
    """An error as the reader can act on it (Markdown), the technical name kept at the end."""
    name, text = type(error).__name__, guard.sanitize(str(error))[:300]
    if name == "RuntimeError" and "no model endpoint" in text:
        why = "No model endpoint is set: run `peresearch setup` (or set `PERESEARCH_LLM_URL`)."
    elif name == "BudgetExceeded":
        why = text
    else:
        why = WHY.get(name, text)
    return f"> **Error:** {why} `/retry` asks again. <sub>({name}{': ' + text if name not in WHY else ''})</sub>"


REASONING_SHOWN = 8000     # characters of reasoning drawn per answer: tens of thousands stall the interface


def command_row(command: str, description: str) -> Table:
    """A row of the command menu: the descriptions in one column, a wrapped one going on under itself."""
    row = Table.grid(padding=(0, 2))
    row.add_column(width=max(map(len, COMMANDS)), no_wrap=True)
    row.add_column(ratio=1)
    row.add_row(command, Text(description, style="dim"))
    return row


def conversation_rows(past, width: int) -> list[Text]:
    """One line per conversation, in columns — when, how many questions, its first question — the question cut
    near the end with "…" when the line would not fit, and kept on one line if it had line breaks."""
    counts = [f"{c.questions} question{'' if c.questions == 1 else 's'}" for c in past]
    count_width = max(map(len, counts), default=0)
    rows = []
    for c, count in zip(past, counts):
        head = f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(c.last))}  {count:<{count_width}}  "
        first = Text(" ".join(guard.sanitize(c.first).split()), style="dim")
        first.truncate(max(8, width - cell_len(head)), overflow="ellipsis")
        rows.append(Text.assemble(head, first, no_wrap=True, overflow="ellipsis"))
    return rows


def render(answer, reasoning: bool = False) -> str:
    """The answer, then its sources, then what the citation check found (Markdown); with `reasoning`, what the
    model thought first, as a quote above it."""
    parts = []
    if reasoning and getattr(answer, "reasoning", ""):
        thought = guard.sanitize(answer.reasoning)
        cut = len(thought) - REASONING_SHOWN
        if cut > 0:                                    # its end, just before the answer, matters most
            thought = f"_({cut:,} earlier characters not shown)_\n\n…" + thought[cut:]
        parts += ["> **Reasoning**\n>\n> " + thought.replace("\n", "\n> "), ""]
    text = _CITE.sub(lambda m: ", ".join(f"[{i.strip()}](source:{i.strip()})" for i in m.group(1).split(",")),
                     guard.sanitize(answer.text or "_(no answer)_"))
    parts += [text, ""]
    if answer.sources:
        parts.append("**Sources**")
        parts += [f"- [{s.id}](source:{s.id}) {guard.sanitize(s.where)}" + (f" — {guard.sanitize(s.title)}" if s.title else "")
                  for s in answer.sources.values()]
    c = answer.check
    if c.unknown_ids:
        parts.append(f"\n> **Check:** cited but never retrieved: {', '.join(c.unknown_ids)}")
    if c.unsupported_quotes:
        parts.append("\n> **Check:** quotes not found in the cited sources: "
                     + "; ".join(f"“{guard.sanitize(q)[:80]}”" for q in c.unsupported_quotes))
    if c.uncited:
        parts.append("\n> **Check:** the answer cites no source")
    if not answer.sources and answer.stopped != "cancelled":
        parts.append("\n_Answered without any source: from the model's own knowledge._")
    if answer.stopped:
        why = {"steps": "the step limit was reached", "time": "the time limit was reached",
               "cancelled": "interrupted", "length": "the answer was cut at the model's token limit"}
        parts.append(f"\n> stopped: {why.get(answer.stopped, answer.stopped)}")
    parts.append(f"\n<sub>{answer.steps} model calls · {answer.prompt_tokens + answer.completion_tokens:,} tokens</sub>")
    return "\n".join(parts)


class Prompt(TextArea):
    """The input box, as in Claude Code: Enter sends; Ctrl+Enter starts a new line (terminals send it as Ctrl+J,
    and Shift+Enter as a plain Enter); the box grows with its text up to 8 lines, then scrolls. ↑/↓ move between
    lines; from the first (last) line they go through earlier questions, and while the command menu is open they
    pick in it. Tab and Esc go to the app (completion, interrupt).

    Every key bound to an action runs here, in order with the typed characters. Textual inserts characters at once
    but runs a key's action later, so a burst came out reordered: a Vietnamese input method types the raw letters,
    then backspaces and the accented text in one burst ("chao", ⌫, ⌫, "ào"), which came out as "chaà"; "abc ⌃U
    xyz" came out empty, and "one ⏎ two ⏎" as "onetwo"."""

    class Submitted(Message):
        """Enter: the text as it was, sent before the box empties."""

        def __init__(self, prompt: "Prompt", value: str):
            super().__init__()
            self.prompt, self.value = prompt, value

    OWN = {"enter": "submit", "ctrl+j": "newline", "ctrl+enter": "newline", "up": "up", "down": "down",
           "tab": "app.complete", "escape": "app.escape", "pageup": "app.page(-1)", "pagedown": "app.page(1)",
           "ctrl+c": "app.copy", "ctrl+r": "app.toggle_reasoning"}
    BINDINGS = [Binding(key, action, show=False) for key, action in OWN.items()]

    def __init__(self, placeholder: str = "", id: str | None = None):
        super().__init__(id=id, soft_wrap=True, show_line_numbers=False, tab_behavior="focus", placeholder=placeholder)
        self.pasted: dict[str, str] = {}              # marker shown in the box -> the text it stands for

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text
        self.move_cursor(self.document.end)

    def _on_paste(self, event: events.Paste) -> None:
        """A paste of several lines shows as a marker, as in Claude Code; the marker is replaced by the whole text
        when the question is sent (`expand`)."""
        text = event.text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.rstrip("\n").count("\n") + 1
        event.stop()                                   # TextArea's handler lets it bubble; the app would paste it again
        if lines == 1:
            return                                     # TextArea's own handler runs next (Textual calls each class's)
        event.prevent_default()
        marker = f"[Pasted text #{len(self.pasted) + 1} +{lines} lines]"
        self.pasted[marker] = text
        self._replace_via_keyboard(marker, *self.selection)

    def expand(self, text: str) -> str:
        for marker, full in self.pasted.items():
            text = text.replace(marker, "\n" + full.rstrip("\n") + "\n")
        return text.strip()

    async def _on_key(self, event: events.Key) -> None:
        action = self.OWN.get(event.key)
        if action is None:
            bound = self._bindings.key_to_bindings.get(event.key)
            action = bound[0].action if bound else None
        if action:
            event.stop()
            event.prevent_default()
            await self.run_action(action)
            return
        await super()._on_key(event)

    def action_submit(self) -> None:
        """Sends the text and empties the box at once, before any key typed after Enter is handled."""
        self.post_message(self.Submitted(self, self.text))
        self.clear()

    def action_newline(self) -> None:
        self._replace_via_keyboard("\n", *self.selection)

    def action_up(self) -> None:
        if self.app.menu_open() or self.cursor_location[0] == 0:
            self.app.action_up()
        else:
            self.action_cursor_up()

    def action_down(self) -> None:
        if self.app.menu_open() or self.cursor_location[0] == self.document.line_count - 1:
            self.app.action_down()
        else:
            self.action_cursor_down()


class CodeColors(ANSIDarkHighlightTheme):
    """Textual's ANSI code colors, without its one bright color (shell backticks, bright black: nearly the
    background in Solarized Dark)."""

    STYLES = {**ANSIDarkHighlightTheme.STYLES, Token.Literal.String.Backtick: "ansi_green"}


class Fence(MarkdownFence):
    """A code block colored with CodeColors, its long lines wrapped (Textual's scrolled sideways, with no
    scrollbar here, so the end of a line was out of sight)."""

    DEFAULT_CSS = """
    Fence { overflow: hidden hidden; }
    Fence > Label { width: 1fr; text-wrap: wrap; text-overflow: fold; }
    """

    @property
    def allow_horizontal_scroll(self) -> bool:
        return False

    @classmethod
    def highlight(cls, code: str, language: str, ansi: bool = False, dark: bool = False):
        if not ansi:
            return super().highlight(code, language, ansi, dark)
        return highlight(code, language=language or None, theme=CodeColors)


class Page(Markdown):
    """Markdown whose code blocks use CodeColors. It does not open links itself (Textual's Markdown handed any
    clicked link to webbrowser.open, whatever its scheme): the app decides, see `Chat.on_markdown_link_clicked`."""

    BLOCKS = {**Markdown.BLOCKS, "fence": Fence, "code_block": Fence}

    def __init__(self, markdown: str = "", **kwargs):
        super().__init__(markdown, open_links=False, **kwargs)


def on_wsl() -> bool:
    return "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))


def windows_path(path: str) -> str:
    return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True, check=True).stdout.strip()


def open_outside(target: str, line: int | None = None) -> str | None:
    """A web page (http or https only) in the browser; one of the user's files in an editor at its line, or selected
    in the file manager; a folder in the file manager. A file is never given to the system's default action, which
    would run a .bat or an .exe. Returns why it was refused, or None."""
    if re.match(r"(?i)https?://", target):
        cmd = (["explorer.exe", target] if on_wsl() else ["open", target] if sys.platform == "darwin"
               else ["xdg-open", target])
    elif not target.startswith("/"):
        return "only web links (http, https) and your own files open"
    elif os.path.isfile(target):
        if shutil.which("code"):
            cmd = ["code", "-g", f"{target}:{line or 1}"]
        elif on_wsl():
            cmd = ["explorer.exe", f"/select,{windows_path(target)}"]
        else:
            cmd = ["xdg-open", os.path.dirname(target)]
    elif os.path.isdir(target):
        cmd = ["explorer.exe", windows_path(target)] if on_wsl() else ["xdg-open", target]
    else:
        return f"{target}: not found"
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return None


def location(where: str) -> tuple[str, int | None]:
    """A source's place ("path:L10-20", "path:p.3", "path:cell 4", "path (listing)") as a path and a line."""
    if where.endswith(" (listing)"):
        return where[:-len(" (listing)")].split(", ")[0], None
    m = re.match(r"(.+):(?:L(\d+)(?:-\d+)?|p\.\d+(?:-\d+)?|cell \d+(?:-\d+)?)$", where)
    return (m.group(1), int(m.group(2)) if m.group(2) else None) if m else (where, None)


class Menu(OptionList):
    """The command menu. It never has the focus (typing stays on the prompt), and Textual draws a list row from a
    style that drops `reverse`, so with the terminal's colors its chosen row looked like every other. The chosen row
    is reversed here, the row under the mouse bold."""

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        try:
            index, _ = self._lines[self.scroll_offset.y + y]
        except IndexError:
            return strip
        if index == self.highlighted:
            return strip.apply_style(RichStyle(reverse=True))
        if index == self._mouse_hovering_over:
            return strip.apply_style(RichStyle(bold=True))
        return strip


class Permission(ModalScreen[str]):
    """May the agent read a path outside the declared folders? Dismissed with "once", "session", "no" (the model
    is told and goes on) or "stop" (Esc: refuse and stop the question, as Esc does everywhere else)."""

    CSS = """
    Permission { align: center bottom; }
    #permission { width: 100%; height: auto; margin: 0 0 3 0; padding: 0 1; border: round ansi_yellow;
                  background: ansi_default; }
    #choices { height: auto; border: none; padding: 0; background: ansi_default; }
    #permission > .hint { color: ansi_default; text-style: dim; }
    """
    BINDINGS = [Binding("escape", "stop", show=False)]
    VERBS = {"read": "read", "tree": "list", "glob": "list the files in", "grep": "search in"}

    def __init__(self, tool: str, path: Path):
        super().__init__()
        self.tool, self.path = tool, path

    def compose(self) -> ComposeResult:
        folder = self.path if self.path.is_dir() else self.path.parent
        with Vertical(id="permission"):
            yield Static(f"[b]peresearch wants to {self.VERBS.get(self.tool, self.tool)}[/b] something outside your "
                         f"declared folders:\n  {escape(guard.sanitize(str(self.path)))}")
            yield Menu(Option("Yes, this once", id="once"),
                       Option(f"Yes, and everything in {escape(guard.sanitize(str(folder)))} for this session",
                              id="session"),
                       Option("No", id="no"), id="choices")
            yield Static("esc stops the question", classes="hint")

    def on_mount(self) -> None:
        menu = self.query_one("#choices", Menu)
        menu.highlighted = 0
        menu.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_stop(self) -> None:
        self.dismiss("stop")


class Chat(App):
    CSS = """
    /* No scrollbar anywhere; the mouse wheel, PageUp/PageDown and the arrow keys still scroll. */
    * { scrollbar-size: 0 0; }
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
    #ask { height: auto; min-height: 3; max-height: 10; border: round ansi_blue; padding: 0 1;
           background: ansi_default; color: ansi_default; }
    /* The cursor and the selection are the default colors reversed (TextArea's own ANSI rules), as a terminal's
       cursor is: a black-on-white cursor vanished where black is the background (Solarized Dark). */
    #ask > .text-area--cursor-line { background: ansi_default; }
    #ask > .text-area--placeholder { color: ansi_default; text-style: dim; }
    Screen > .screen--selection { background: ansi_default; color: ansi_default; text-style: reverse; }
    """
    BINDINGS = [Binding("ctrl+q", "quit", "Quit", priority=True)]     # Ctrl+C copies (the owner's choice)
    ENABLE_COMMAND_PALETTE = False       # Textual's own palette (Ctrl+P): another look, and it switches themes

    def __init__(self, make_agent, heading=lambda: "peresearch", workspace=None, index_on_start: bool = True,
                 project: str = "default"):
        super().__init__()
        self.make_agent, self.heading, self.ws, self.project = make_agent, heading, workspace, project
        self.index_on_start = index_on_start and workspace is not None
        self.agent, self.busy, self.last, self.pending_new = None, False, None, False
        self.pending_resume = None                    # a conversation picked before the agent existed
        self.last_question = None                     # (as shown, as sent) for /retry
        self.show_reasoning, self.pages = False, []   # Ctrl+R; this session's answers, to redraw them
        self.sources_of = {}                          # a shown page -> the answer whose sources its links name
        self.menu_kind = ""                           # what the menu lists: "commands", "paths" or "conversations"
        self.asked, self.recall = [], None
        self.started, self.doing, self.tokens, self.frame = 0.0, "", 0, 0
        self.indexing = ""                             # what the index update is doing, apart from a question
        self.flash, self.flash_until = "", 0.0         # a short message on the status line when idle

    def compose(self) -> ComposeResult:
        yield Static(f"✻ {self.heading()}", id="heading")
        yield VerticalScroll(id="log", can_focus=False)
        self.status = Static("", id="status")       # kept: the spinner may tick once more after unmounting
        yield self.status
        yield Menu(id="commands")
        yield Prompt(placeholder="❯ Ask about your files or the web", id="ask")

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
        status = self.status
        if not self.busy and not self.indexing:
            status.update(self.flash if time.time() < self.flash_until else "")
            return
        self.frame += 1
        spin = SPINNER[self.frame % len(SPINNER)]
        if not self.busy:                              # only the index update runs
            status.update(f"{spin} {guard.sanitize(self.indexing)}… (you can ask meanwhile)")
            return
        extra = f" · {self.tokens:,} tokens" if self.tokens else ""
        status.update(f"{spin} {guard.sanitize(self.doing)}… ({time.time() - self.started:.0f}s{extra} · esc to interrupt)")

    def on_agent_event(self, kind: str, detail: str) -> None:
        if kind == "tool":
            self.add(Static(f"● {guard.sanitize(detail)}", classes="tool"))
            self.doing = ("Searching" if detail.startswith(("search", "grep", "glob", "web")) else
                          "Looking through folders" if detail.startswith("tree") else
                          "Noting what is missing" if detail.startswith("gaps") else "Reading")
        elif kind == "result":
            self.add(Static(f"  ⎿  {guard.sanitize(detail)}", classes="result"))
        elif kind == "tokens":
            self.tokens = int(detail)
        elif kind == "think":
            self.doing = "Thinking"
        elif kind == "wait":
            self.doing = guard.sanitize(detail)

    # --- asking ---

    def on_prompt_submitted(self, event: Prompt.Submitted) -> None:
        menu = self.query_one("#commands", OptionList)
        q = event.value.strip()                            # the prompt has already emptied itself
        if menu.display and menu.highlighted is not None and self.menu_kind == "conversations" and not q:
            self.resume(menu.get_option_at_index(menu.highlighted).id.removeprefix("resume:"))
            return
        if menu.display and menu.highlighted is not None and q.startswith("/"):
            chosen = menu.get_option_at_index(menu.highlighted).id
            if chosen.startswith("path:"):                  # a path picked from the Tab suggestions
                self.set_prompt(event.value.split(" ")[0] + " " + chosen[5:])
                self.hide_menu()
                return
            if q.split(" ")[0] != chosen:
                q = chosen
                if chosen in ("/add", "/remove"):          # these need a folder: wait for it
                    self.set_prompt(chosen + " ")
                    self.hide_menu()
                    return
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
        self.ask(q, self.query_one("#ask", Prompt).expand(q))

    def ask(self, shown: str, sent: str) -> None:
        self.last_question = (shown, sent)
        self.start("Starting")
        self.add(Static(f"❯ {guard.sanitize(shown)}", classes="question", markup=False))
        self.run_agent(sent)

    @work(thread=True, exclusive=True, group="agent")
    def run_agent(self, question: str) -> None:
        def event(kind, detail):
            self.call_from_thread(self.on_agent_event, kind, detail)
        try:
            if self.agent is None:
                self.call_from_thread(self.on_agent_event, "wait", "Loading the local models")
                self.agent = self.make_agent(event)
                if hasattr(self.agent, "toolbox"):
                    self.agent.toolbox.ask = self.permission
                if self.pending_resume:
                    self.agent.resume(self.pending_resume)
                elif self.pending_new:
                    self.agent.new_session()
            answer = self.agent.ask(question)
            self.last = answer
            self.call_from_thread(self.show_answer, answer)
        except Exception as e:
            self.call_from_thread(self.add, Page(explain(e)))
        finally:
            self.call_from_thread(self.done)

    def show_answer(self, answer) -> None:
        page = Page(render(answer, self.show_reasoning))
        self.pages.append((page, answer))
        self.sources_of[page] = answer
        self.add(page)

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """[N1] shows that source; "open the file / page" opens it; a web link opens in the browser; nothing else."""
        href = event.href
        if href.startswith(("source:", "open:")):
            kind, sid = href.split(":", 1)
            source = getattr(self.sources_of.get(event.markdown), "sources", {}).get(sid)
            if source is None:
                self.note(f"{sid}: no such source")
            elif kind == "source":
                self.show_source(source, self.sources_of[event.markdown])
            else:
                target, line = (source.where, None) if source.kind == "web" else location(source.where)
                why = open_outside(target, line)
                self.note(f"not opened: {why}" if why else f"opened {target}")
        elif re.match(r"(?i)https?://", href):
            why = open_outside(href)
            self.note(f"not opened: {why}" if why else f"opened {href}")
        else:
            self.note("not opened: only web links (http, https) and your own files open")

    def show_source(self, source, answer) -> None:
        excerpt = guard.sanitize(source.text[:600]).replace("\n", "\n> ") if source.text else \
            "_(the excerpt is not kept in the history)_"
        what = "open the page" if source.kind == "web" else "open the file"
        page = Page(f"**{source.id}** {guard.sanitize(source.where)}"
                    + (f" — {guard.sanitize(source.title)}" if source.title else "")
                    + f"\n\n> {excerpt}\n\n[{what}](open:{source.id})")
        self.sources_of[page] = answer
        self.add(page)

    def action_toggle_reasoning(self) -> None:
        """Ctrl+R: show or hide what the model thought before answering, on every answer of this session."""
        self.show_reasoning = not self.show_reasoning
        for page, answer in self.pages:
            if page.is_attached:
                page.update(render(answer, self.show_reasoning))
        self.say("the model's reasoning is shown (ctrl+r hides it)" if self.show_reasoning
                 else "the model's reasoning is hidden (ctrl+r shows it)")

    def permission(self, tool: str, path: Path) -> str:
        """Called from the agent's thread: shows the dialog and waits for the answer."""
        answer, answered = {}, threading.Event()

        def show():
            self.doing = "Waiting for your permission"
            self.push_screen(Permission(tool, path), lambda choice: (answer.update(choice=choice), answered.set()))
        self.call_from_thread(show)
        while not answered.wait(0.1):
            if not self.is_running:                     # quit while the dialog was open
                return "no"
        if answer.get("choice") == "stop":              # Esc on the dialog: refuse, and stop the whole question
            self.agent.cancel()
            return "no"
        return answer.get("choice") or "no"

    def done(self) -> None:
        self.busy = False
        self.query_one("#heading", Static).update(f"✻ {self.heading()}")

    @work(thread=True, exclusive=True, group="index")      # a group of its own: a question does not cancel it
    def reindex(self, doing: str) -> None:
        """Brings the index up to date in the background; questions can be asked meanwhile (search copes with an
        index being updated), and the status line shows the question first."""
        self.call_from_thread(setattr, self, "indexing", doing)
        try:
            r = self.ws.update()
            quiet = not (r.added or r.changed or r.removed or r.withheld)
            text = f"Index up to date: {r.chunks:,} chunks from {r.added + r.changed + r.unchanged} files." if quiet \
                else summary(r, limit=5)
            self.call_from_thread(self.note, text)
        except Exception as e:
            self.call_from_thread(self.note, f"indexing failed: {type(e).__name__}: {e}"[:300])
        finally:
            self.call_from_thread(self.indexed)

    def indexed(self) -> None:
        self.indexing = ""
        self.query_one("#heading", Static).update(f"✻ {self.heading()}")

    def start(self, doing: str) -> None:
        self.busy, self.started, self.doing, self.tokens = True, time.time(), doing, 0

    # --- commands ---

    def command(self, text: str) -> None:
        name = text.split(" ")[0]
        if name == "/exit":
            self.exit()
        elif name == "/help":
            self.add(Page("**Commands** (the keys: `/keys`)\n" + "\n".join(f"- `{c}` {d}" for c, d in COMMANDS.items())))
        elif name == "/keys":
            self.add(Page("**Keys**\n" + "\n".join(f"- **{k}** {d}" for k, d in KEYS.items())))
        elif name == "/new":
            if self.busy:
                self.note("wait for the current question to finish, or press esc")
                return
            if self.agent is not None:
                self.agent.new_session()
            self.pending_new, self.pending_resume = self.agent is None, None
            self.query_one("#log", VerticalScroll).remove_children()
            self.pages, self.sources_of = [], {}
            self.note("New conversation: earlier questions are no longer sent as context.")
        elif name == "/retry":
            if self.busy:
                self.note("wait for the current question to finish, or press esc")
            elif self.last_question is None:
                self.note("nothing to retry yet")
            else:
                self.ask(*self.last_question)
        elif name == "/resume":
            if self.busy:
                self.note("wait for the current question to finish, or press esc")
                return
            past = conversations(history_file(self.project))
            if not past:
                self.note("no earlier conversation in this project")
                return
            menu = self.query_one("#commands", OptionList)
            menu.clear_options()
            menu.add_options([Option(row, id="resume:" + c.id)
                              for c, row in zip(past, conversation_rows(past[:50], self.size.width - 6))])
            menu.display, menu.highlighted, self.menu_kind = True, 0, "conversations"
        elif name == "/sources":
            if not self.last or not self.last.sources:
                self.note("no sources yet")
                return
            self.add(Page("\n\n".join(f"**`{s.id}`** {guard.sanitize(s.where)}\n\n> "
                                          + guard.sanitize(s.text[:600]).replace("\n", "\n> ")
                                          for s in self.last.sources.values())))
        elif name == "/folders":
            folders = self.ws.folders() if self.ws else []
            self.note("\n".join(map(str, folders)) or "no folders declared: /add <folder>")
        elif name in ("/add", "/remove", "/index"):
            if self.ws is None:
                self.note("folders cannot be changed here")
                return
            if self.busy or self.indexing:
                self.note("wait for the current work to finish (a question: esc interrupts it)")
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

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        v = event.text_area.text
        if v.startswith(("/add ", "/remove ")):
            return
        if not v and self.menu_kind == "conversations":   # the prompt emptying itself after "/resume"
            return
        if v.startswith("/") and " " not in v:
            matches = [c for c in COMMANDS if c.startswith(v)]
            menu = self.query_one("#commands", OptionList)
            menu.clear_options()
            menu.add_options([Option(command_row(c, d), id=c) for c, d in COMMANDS.items() if c in matches])
            menu.display, self.menu_kind = bool(matches), "commands"
            if matches:
                menu.highlighted = 0
        else:
            self.hide_menu()

    def menu_open(self) -> bool:
        return self.query_one("#commands", OptionList).display

    def hide_menu(self) -> None:
        self.query_one("#commands", OptionList).display = False
        self.menu_kind = ""

    def resume(self, conversation: str) -> None:
        """Show an earlier conversation and make it the context of the next question."""
        self.hide_menu()
        if self.agent is not None:
            turns = self.agent.resume(conversation)
        else:
            turns = [t for t in read_turns(history_file(self.project)) if t["conversation"] == conversation]
            self.pending_resume, self.pending_new = conversation, False
        log = self.query_one("#log", VerticalScroll)
        log.remove_children()
        self.pages, self.sources_of = [], {}
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(turns[0].get("time", 0))) if turns else "?"
        self.note(f"Resumed the conversation of {when} ({len(turns)} question{'' if len(turns) == 1 else 's'}); "
                  "the next question continues it.")
        for t in turns:
            self.add(Static(f"❯ {guard.sanitize(t['question'])}", classes="question", markup=False))
            kept = {k: Source(k, "web" if str(v).startswith("http") else "file", str(v), "", "")
                    for k, v in (t.get("sources") or {}).items()}
            page = Page(render(SimpleNamespace(text=t["answer"], sources=kept, check=SimpleNamespace(
                unknown_ids=[], unsupported_quotes=[], uncited=False), stopped="", steps=0, prompt_tokens=0,
                completion_tokens=0)).rsplit("\n<sub>", 1)[0])
            self.sources_of[page] = SimpleNamespace(sources=kept)
            self.add(page)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        prompt = self.query_one("#ask", Prompt)
        self.hide_menu()
        if event.option.id.startswith("resume:"):
            self.resume(event.option.id.removeprefix("resume:"))
        elif event.option.id.startswith("path:"):
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
            if chosen.startswith("resume:"):
                self.resume(chosen.removeprefix("resume:"))
                return
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
                menu.display, menu.highlighted, self.menu_kind = True, 0, "paths"

    def say(self, text: str, seconds: float = 2.5) -> None:
        """A short message on the status line, shown while nothing runs."""
        self.flash, self.flash_until = text, time.time() + seconds
        self.tick()

    def action_copy(self) -> None:
        """Ctrl+C: the text selected with the mouse in the conversation, else the prompt's selection."""
        text = self.screen.get_selected_text() or self.query_one("#ask", Prompt).selected_text
        if text:
            self.copy_to_clipboard(text)
            self.say(f"copied {len(text):,} characters")
        else:
            self.say("select text with the mouse to copy it · ctrl+q quits")

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
