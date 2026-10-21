"""The terminal interface, full screen, laid out the way Claude Code's is.

- The conversation scrolls above; each question is shown as `❯ question`, each tool the agent calls as
  `● tool(args)` with a `⎿ result` line under it, then the cited answer, its sources and the citation check.
- A status line while the agent works: a spinner, what it is doing, elapsed seconds and tokens so far.
- The input box at the bottom, `❯` in it and the setup in use on its bottom border: Enter asks, ↑/↓ recall earlier
  questions, `/` opens the command menu under the box (↑/↓ to pick, Tab or Enter to take it, Esc to close), Esc
  during a question interrupts it at the next step. PageUp/PageDown
  scroll the conversation; the conversation never takes the focus, so typing always reaches the prompt, also
  after a click on it. No line of key hints (the owner's choice): /keys lists the keys.
- Nothing needs a minimum width (as in Claude Code): text, menu descriptions and code lines wrap to the terminal.
- Controls open in windows (windows.py): folders, models and APIs, settings, status, conversations, checks. On
  start, the index is brought up to date with the declared folders (only new or changed files are read).
- Before the agent reads anything outside the declared folders, a dialog asks: yes this once, yes for that folder
  for the rest of the session, or no (Esc).
Colors are the terminal's own, so its color scheme (light or dark) decides how everything looks: the default
foreground and background and the six plain hues, never RGB, 256 colors, black, white or the bright colors (black
or white is the background in some scheme, the bright ones are grays in Solarized). Muted text is dimmed; the
accent is blue (`❯`, the chosen row of a list, the focused field); the cursor and selections are the default colors
reversed; nothing is drawn on a colored background.
The agent runs in a worker thread. Everything that came from a file, a page or the model is stripped of terminal
control sequences before it is shown.
"""

import asyncio
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

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from pygments.token import Token
from rich.cells import cell_len
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.highlight import ANSIDarkHighlightTheme, highlight
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.message import Message
from textual.css.query import NoMatches
from textual.widgets import Input, Markdown, OptionList, Static, TextArea
from textual.widgets._markdown import MarkdownFence
from textual.widgets.option_list import Option
from textual.widgets.text_area import Selection

from peresearch import connections, guard, mascot, settings
from peresearch.agent import (Answer, Check, USAGE, _CITE, check, clean_name, conversations, find_conversation, history_file, read_turns,
                              usage)
from peresearch.tools import Source
from peresearch.widgets import GUTTER, SelectList, hints, home_path, row
from peresearch.workspace import brief, complete, quiet
from peresearch.zetokrag import parse
from peresearch.zetokrag.index import Stopped

QUIT_GRACE = 2.0          # seconds quitting waits for background work to stop before leaving it

COMMANDS = {
    "/help": "the commands",
    "/folders": "manage folders",
    "/model": "choose model for chat, embedding and reranker with reasoning effort to use",
    "/settings": "research limits and search services",
    "/status": "inspect setup, index, context and usage",
    "/sessions": "manage conversations",
    "/local": "local only: turn the web off for this session (again: back on)",
    "/check": "check configured connections after confirmation (remote services may charge)",
    "/new": "start a fresh conversation",
    "/retry": "ask the last question again",
    "/sources": "preview sources of the last answer",
    "/exit": "quit",
}
KEYS = {
    "Enter": "ask", "Ctrl+Enter": "new line", "↑ ↓": "earlier questions (in the menu: pick)",
    "/": "the command menu (Tab or Enter takes the chosen command, Esc closes it)",
    "@": "name a file or folder for the question (Tab completes it from your folders)",
    "PgUp PgDn": "scroll the conversation", "Esc": "interrupt the question",
    "Ctrl+R": "show or hide the model's reasoning", "Ctrl+C": "copy the text selected with the mouse",
    "Ctrl+Q": "quit",
}
SPINNER = "⣾⣽⣻⢿⡿⣟⣯⣷"
TICK = 0.1                  # seconds between two redraws of the status line and the mascot
MARK = "\u2060"          # WORD JOINER, zero width: ends a real paste marker; no key types it
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
    "APIConnectionError": "The model server does not answer: check that it runs and that `PERESEARCH_LLM_URL` "
                          "points to it (`/check` tests it); a server that is starting may need minutes.",
    "APITimeoutError": "The model server took too long to answer.",
    "AuthenticationError": "The model server refused the key (`PERESEARCH_LLM_KEY`, set with `/model`).",
    "PermissionDeniedError": "The model server refused the key (`PERESEARCH_LLM_KEY`, set with `/model`).",
    "NotFoundError": "The model server has no such path or model: check `PERESEARCH_LLM_URL` (ending in /v1) and "
                     "`PERESEARCH_LLM_MODEL`.",
    "RateLimitError": "The model server is limiting requests; wait a moment.",
    "InternalServerError": "The model server failed while answering.",
    "StreamCut": "The connection to the model server was cut mid-answer, again and again.",
}


def _drawn(beat: int | None) -> Text:
    """The mascot's frame as one block of text: at rest (None), or a beat of its thinking."""
    return Text("\n").join(mascot.frame(beat))


def place(error: BaseException) -> str:
    """Where an error was raised in peresearch's own code (the innermost such frame), as file:line. The package
    folder itself is the root: installed, the folder above it holds every library."""
    package, found, tb = Path(__file__).resolve().parent, "", error.__traceback__
    while tb is not None:
        path = Path(tb.tb_frame.f_code.co_filename).resolve()
        if package in path.parents:
            found = f"{path.relative_to(package.parent)}:{tb.tb_lineno}"
        tb = tb.tb_next
    return found


def explain(error: Exception) -> str:
    """An error as the reader can act on it (Markdown), the technical name kept at the end. A failed internal check
    (an assert) is a bug in peresearch: it says so and where."""
    name, text = type(error).__name__, guard.sanitize(str(error))[:300]
    if isinstance(error, AssertionError):
        where = place(error)
        return (f"> **Error:** an internal check of peresearch failed{f' at `{where}`' if where else ''}: "
                f"{text or '(no message)'}. This is a bug in peresearch, not in your setup. `/retry` asks again. "
                f"<sub>({name})</sub>")
    if name == "RuntimeError" and "no model endpoint" in text:
        why = ("No model endpoint is set: /model chooses it (or set "
               "`PERESEARCH_LLM_URL`).")
    else:
        why = WHY.get(name, text)
    return f"> **Error:** {why} `/retry` asks again. <sub>({name}{': ' + text if name not in WHY else ''})</sub>"


REASONING_SHOWN = 8000     # characters of reasoning drawn per answer: tens of thousands stall the interface


MODEL_FIELDS = ("PERESEARCH_LLM_URL", "PERESEARCH_LLM_MODEL", "PERESEARCH_LLM_KEY")
ENDPOINT_FIELDS = tuple(f"PERESEARCH_{kind}_{part}" for kind in ("EMBED", "RERANK") for part in ("URL", "MODEL", "KEY")) \
    + ("PERESEARCH_SEARCH_MCP", "PERESEARCH_SEARCH_MCP_KEY", "PERESEARCH_SEARCH_TOOL")


def config_listing() -> str:
    """/config's page: what is set, in full, with where it comes from; the defaults in use on one line (what fits
    without the user is not laid out: step 4, 2026-10-08); rerank-depth only when a rerank endpoint is set."""
    saved, rows, defaults = settings.saved(), [], []
    for key, setting, default, what, _ in settings.CONFIG:
        if key == "rerank-depth" and not settings.get("PERESEARCH_RERANK_URL"):
            continue
        if os.environ.get(setting):
            rows.append(f"- {key}: {os.environ[setting]} (environment) — {what}")
        elif saved.get(setting):
            rows.append(f"- {key}: {saved[setting]} (saved) — {what}")
        else:
            defaults.append(f"{key} {default}")
    head = ("**Settings** (`/config <name> <value>` sets one, `/config <name>` resets it; they apply from the next "
            "question)")
    tail = ("None set: each has its default — " if not rows else "Defaults: ") + " · ".join(defaults) if defaults else ""
    return "\n".join([head, *rows] + ([tail] if tail else []))


def check_settings() -> tuple[bool, list[str]]:
    """The model check on the model in the settings (llm.check_model)."""
    from peresearch.llm import LLM, check_model
    return check_model(LLM())


def check_endpoints() -> list[str]:
    """The search's endpoints in the settings (zetokrag/embed.check)."""
    from peresearch.cli import web_check
    from peresearch.zetokrag import embed
    return embed.check(*embed.from_settings()) + web_check()


def command_row(command: str, description: str) -> Padding:
    """A row of the command menu: the pointer's gutter, the command, then its description dimmed in one column (a
    wrapped one goes on under itself)."""
    line = Table.grid(padding=(0, 2))
    line.add_column(width=max(map(len, COMMANDS)), no_wrap=True)
    line.add_column(ratio=1)
    line.add_row(command, Text(description, style="dim"))
    return Padding(line, (0, 0, 0, len(GUTTER)))


def columns(rows: dict[str, str]) -> Table:
    """Names and what they do in two aligned columns, the second dimmed (/help's commands and keys)."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    for name, what in rows.items():
        grid.add_row(name, Text(what, style="dim"))
    return grid


def conversation_rows(past, width: int) -> list[Text]:
    """One line per conversation, in columns — when, how many questions, its name if it has one, its first question —
    the question cut near the end with "…" when the line would not fit, and kept on one line if it had line breaks."""
    counts = [f"{c.questions} question{'' if c.questions == 1 else 's'}" for c in past]
    count_width = max(map(len, counts), default=0)
    rows = []
    for c, count in zip(past, counts):
        head = f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(c.last))}  {count:<{count_width}}  "
        named = f"{guard.sanitize(c.name)} · " if getattr(c, "name", "") else ""
        first = Text(" ".join(guard.sanitize(c.first).split()), style="dim")
        first.truncate(max(8, width - cell_len(head) - cell_len(named)), overflow="ellipsis")
        head += named
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
    # the sources the answer cites; how many others were read (owner, 2026-10-22: /sources lists every one)
    cited = {i.strip() for m in _CITE.finditer(answer.text or "") for i in m.group(1).split(",")}
    shown = [s for s in answer.sources.values() if s.id in cited]
    if shown:
        parts.append("**Sources cited**")
        parts += [f"- [{s.id}](source:{s.id}) {guard.sanitize(home_path(s.where))}"
                  + (f" — {guard.sanitize(s.title)}" if s.title else "") for s in shown]
    others = len(answer.sources) - len(shown)
    if others:
        parts.append(f"\n_{others:,} {'other ' if shown else ''}source{'s' * (others != 1)} read"   # not in the list
                     f"{'' if shown else ', none cited'}: /sources lists them._")
    c = answer.check
    if c.unknown_ids:
        parts.append(f"\n> **Check:** citation evidence unavailable; read again: {', '.join(c.unknown_ids)}")
    if c.unsupported_quotes:
        parts.append("\n> **Check:** quotes not found in the cited sources: "
                     + "; ".join(f"“{guard.sanitize(q)[:80]}”" for q in c.unsupported_quotes))
    if c.uncited:
        parts.append("\n> **Check:** the answer cites none of the sources read")
    if not answer.sources and answer.stopped != "cancelled":
        parts.append("\n_Answered without any source: from the model's own knowledge._")
    if answer.stopped:
        why = {"steps": "the step limit was reached", "time": "the time limit was reached",
               "cancelled": "interrupted", "length": "the answer was cut at the model's token limit",
               "refused": "the model kept making calls that were refused"}
        parts.append(f"\n> stopped: {why.get(answer.stopped, answer.stopped)}")
    passed = not (c.unknown_ids or c.unsupported_quotes or c.uncited)   # any check record, not only agent.Check
    checked = "✓ citations checked · " if answer.sources and passed else ""
    parts.append(f"\n<sub>{checked}{answer.steps} model calls · {answer.prompt_tokens + answer.completion_tokens:,} "
                 "tokens</sub>")
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
        # MARK at the end only: the cursor at the marker's start would sit on a leading one and be invisible
        marker = f"[Pasted text #{len(self.pasted) + 1} +{lines} lines]{MARK}"
        self.pasted[marker] = text
        self._put(marker, *self.selection)

    def expand(self, text: str) -> str:
        for marker, full in self.pasted.items():
            text = text.replace(marker, "\n" + full.rstrip("\n") + "\n")
        return text.replace(MARK, "").strip()

    # A marker is one block: only a paste makes one (its MARK cannot be typed and is stripped from pasted text),
    # an edit that touches it removes all of it, and the cursor never stops inside it.

    def _spans(self, row: int) -> list[tuple[int, int]]:
        line, spans = self.document.get_line(row), []
        for marker in getattr(self, "pasted", ()):    # TextArea sets a selection before __init__ sets `pasted`
            at = line.find(marker)
            while at >= 0:
                spans.append((at, at + len(marker)))
                at = line.find(marker, at + 1)
        return spans

    def _outside(self, location, forward: bool):
        row, column = location
        for start, end in self._spans(row):
            if start < column < end:
                return (row, end if forward else start)
        return location

    def _put(self, insert: str, start, end):
        start, end = sorted((start, end))
        if start == end:                               # typed inside a marker: goes after it
            start = end = self._outside(start, True)
        return super()._replace_via_keyboard(insert, self._outside(start, False), self._outside(end, True))

    def _replace_via_keyboard(self, insert: str, start, end):
        return self._put(insert.replace(MARK, ""), start, end)

    def _delete_via_keyboard(self, start, end):
        start, end = sorted((start, end))
        return super()._delete_via_keyboard(self._outside(start, False), self._outside(end, True))

    def _nearer_edge(self, location):
        row, column = location
        for start, end in self._spans(row):
            if start < column < end:
                return (row, start if column - start <= end - column else end)
        return location

    def _validate_selection(self, selection: Selection) -> Selection:
        """Every cursor move and selection passes here (keys, word jumps, lines, clicks): a cursor that would stop
        inside a marker goes to its edge the way it was heading, or to the nearer edge with no heading (a click,
        another line); a selection that touches a marker covers all of it."""
        start, end = super()._validate_selection(selection)
        if start != end:
            forward = end > start
            return Selection(self._outside(start, not forward), self._outside(end, forward))
        before = self.selection.end
        if end[0] == before[0] and end != before:
            end = self._outside(end, end > before)
        else:
            end = self._nearer_edge(end)
        return Selection(end, end)

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
    Fence { overflow: hidden hidden; border: none; border-left: solid ansi_blue; padding: 0 0 0 1; margin: 1 0; }
    /* Textual's ANSI rule pads the code with a blank line above and below and drops the margin: the rule on the
       left then ran over empty lines, and the next heading touched the code. */
    Fence:ansi { margin: 1 0; }
    Fence:ansi > Label { padding: 0; }
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
    in the file manager (always so for a converted file: a PDF, a notebook, Word, PowerPoint, Excel, HTML, which an
    editor cannot show, its own program can); a folder in the file manager. A file is never given to the system's default action, which
    would run a .bat or an .exe. Returns why it was refused, or None."""
    if re.match(r"(?i)https?://", target):
        cmd = (["explorer.exe", target] if on_wsl() else ["open", target] if sys.platform == "darwin"
               else ["xdg-open", target])
    elif not target.startswith("/"):
        return "only web links (http, https) and your own files open"
    elif os.path.isfile(target):
        if shutil.which("code") and parse.kind(Path(target)) not in parse.CONVERTED:   # a PDF or a deck: not text
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


class Menu(SelectList):
    """The command menu, under the input box as in Claude Code. It never has the focus (typing stays on the prompt);
    the chosen row shows the `❯` pointer in the accent color (SelectList)."""


WINDOW_CSS = """
    {name} {{ align: center middle; background: ansi_default; }}
    #{box} {{ width: 100%; max-width: 96; height: auto; max-height: 90%; padding: 0 1; border: round {color};
             background: ansi_default; }}
    #{box} > .title {{ height: 1; text-style: bold; margin-bottom: 1; }}
    #{box} > .hint {{ height: auto; color: ansi_default; text-style: dim; margin-top: 1; }}
"""


class ConnectionCheck(ModalScreen[bool]):
    """A deliberate user action before connection checks can contact paid services: r runs them, esc does not."""

    CSS = WINDOW_CSS.format(name="ConnectionCheck", box="connection-check", color="ansi_blue")
    BINDINGS = [Binding("escape", "cancel", show=False), Binding("r", "run", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="connection-check"):
            yield Static("Check connections", classes="title", markup=False)
            yield Static("This sends test requests to your configured model and search services. Remote usage may "
                         "be charged by those services. It does not send your documents.", markup=False)
            yield Static(hints(("r", "run checks"), ("esc", "cancel")), classes="hint", markup=False)

    def action_run(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class Permission(ModalScreen[str]):
    """May the agent read a path outside the declared folders? Dismissed with "once", "session", "no" (the model
    is told and goes on) or "stop" (Esc: refuse and stop the question, as Esc does everywhere else). Shown above the
    input box, as Claude Code asks for a tool's permission."""

    CSS = """
    Permission { align: center bottom; background: transparent; }
    #permission { width: 100%; height: auto; margin: 0 0 3 0; padding: 0 1; border: round ansi_yellow;
                  background: ansi_default; }
    #permission > .title { height: auto; margin-bottom: 1; }
    #choices { height: auto; border: none; padding: 0; background: ansi_default; }
    #permission > .hint { height: auto; color: ansi_default; text-style: dim; margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "stop", show=False)]
    VERBS = {"read": "read", "tree": "list", "glob": "list the files in", "grep": "search in"}

    def __init__(self, tool: str, path: Path):
        super().__init__()
        self.tool, self.path = tool, path

    def compose(self) -> ComposeResult:
        folder = self.path if self.path.is_dir() else self.path.parent
        with Vertical(id="permission"):
            yield Static(Text.assemble((f"peresearch wants to {self.VERBS.get(self.tool, self.tool)}", "bold"),
                                       " something outside your declared folders:\n",
                                       (f"  {guard.sanitize(home_path(str(self.path)))}", "yellow")),
                         classes="title", markup=False)
            yield Menu(Option(row("Yes, this once"), id="once"),
                       Option(row(f"Yes, and everything in {guard.sanitize(home_path(str(folder)))} for this session"),
                              id="session"),
                       Option(row("No"), id="no"), id="choices")
            yield Static(hints(("↑↓", "choose"), ("enter", "answer"), ("esc", "stops the question")),
                         classes="hint", markup=False)

    def on_mount(self) -> None:
        menu = self.query_one("#choices", Menu)
        menu.highlighted = 0
        menu.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_stop(self) -> None:
        self.dismiss("stop")


def asked_line(question: str) -> Text:
    """A question in the conversation: the accent `❯`, then the question as typed."""
    return Text.assemble(("❯ ", "blue"), guard.sanitize(question))


MENTION = re.compile(r"(?<!\S)@(\S+)")           # @path at the start of a word


def mention_matches(partial: str, roots: list) -> list[str]:
    """What completes `@partial`: a path from ~ or / as typed, else one inside a declared folder, relative to it."""
    if partial.startswith(("/", "~")):
        return complete(partial, folders_only=False)
    out = set()
    for root in roots:
        base = str(root).rstrip("/") + "/"
        out.update(m[len(base):] for m in complete(base + partial, folders_only=False) if m.startswith(base))
    return sorted(out)


def with_mentions(question: str, roots: list) -> str:
    """The question for the model: the places it names with @ that exist (inside a declared folder, from ~ or /)
    and are not protected, added as a note to read them with the tools, so every rule of read and tree still holds."""
    named = []
    for token in MENTION.findall(question):
        p = Path(token).expanduser()
        if not p.is_absolute():
            p = next((Path(r) / token for r in roots if (Path(r) / token).exists()), None)
        if p is not None and p.exists() and p not in [q for q, _ in named] and not (
                guard.denied(p) or guard.denied(Path(os.path.realpath(p)))):     # a protected place: no note
            named.append((p, p.is_dir()))
    if not named:
        return question
    return question + "\n\n(The user named with @: " + ", ".join(
        f"{p} (a folder: look at it with tree)" if is_dir else f"{p} (a file: read it with read)"
        for p, is_dir in named) + ")"


def export_markdown(project: str, exchanges: list) -> str:
    """Questions and answers as Markdown, each answer followed by its sources: id, place and title (not their text,
    which may be long and is the user's own)."""
    out = f"# peresearch — {project}\n"
    for question, answer in exchanges:
        out += f"\n## {' '.join(question.split())}\n\n{answer.text.strip()}\n"
        if answer.sources:
            out += "\n**Sources**\n\n" + "\n".join(f"- **{s.id}** {s.where}" + (f" — {s.title}" if s.title else "")
                                                   for s in answer.sources.values()) + "\n"
    return out


class Chat(App):
    # The layout of Claude Code's: the conversation above; the status line; the input box with its `❯` and, on its
    # bottom border, the setup in use; under it the command menu while it is open. No line of key hints (the owner's
    # choice, 2026-09-30): /keys lists them.
    CSS = """
    /* No scrollbar in the conversation; the mouse wheel, PageUp/PageDown and the arrow keys still scroll. */
    * { scrollbar-size: 0 0; }
    Screen { layout: vertical; layers: base mascot; }
    Screen { background: ansi_default; color: ansi_default; }
    #welcome { width: auto; max-width: 100%; height: auto; padding: 0 1; border: round ansi_blue; }
    #log { height: 1fr; padding: 0 1; background: ansi_default; }
    /* No bold for emphasis: Windows Terminal draws bold as a brighter color, which Solarized makes a pale gray. */
    .question { margin: 1 0 0 0; color: ansi_default; }
    .tool { color: ansi_default; margin: 1 0 0 0; text-wrap: nowrap; text-overflow: ellipsis; }
    .result { color: ansi_default; text-style: dim; text-wrap: nowrap; text-overflow: ellipsis; }
    .note { color: ansi_default; text-style: dim; margin: 1 0 0 0; }
    #log > Page { margin: 1 0 0 0; padding: 0 0 0 2; }
    #log > Page.draft { color: ansi_default; }
    /* The mascot lies on the input box (mascot.py), on a layer of its own: three lines above the box hold its
       body; the status line keeps clear of it; the box's right padding keeps typed text from under its tail. */
    #status { width: 1fr; height: 100%; padding: 0 1; color: ansi_yellow; content-align: left bottom; }
    #input-box { height: auto; border: round ansi_blue; background: ansi_default;
                 border-subtitle-color: ansi_default; border-subtitle-style: dim; }
    #prompt-mark { width: 2; height: 1; color: ansi_blue; }
    #ask { height: auto; min-height: 1; max-height: 8; border: none; padding: 0;
           background: ansi_default; color: ansi_default; }
    #ask:focus { border: none; }
    #commands { height: auto; max-height: 10; display: none; padding: 0 1; }
    /* The cursor and the selection are the default colors reversed (TextArea's own ANSI rules), as a terminal's
       cursor is: a black-on-white cursor vanished where black is the background (Solarized Dark). */
    #ask > .text-area--cursor-line { background: ansi_default; }
    #ask > .text-area--placeholder { color: ansi_default; text-style: dim; }
    Screen > .screen--selection { background: ansi_default; color: ansi_default; text-style: reverse; }
    """ + (f"#bar {{ height: {mascot.REST_LINES - 2}; }} #perch {{ width: {mascot.WIDTH}; }} "
           f"#mascot {{ layer: mascot; position: absolute; width: {mascot.WIDTH}; height: auto; }} "
           f"#input-box {{ padding: 0 {mascot.TAIL - 1} 0 1; }}")
    BINDINGS = [Binding("ctrl+q", "quit", "Quit", priority=True)]     # Ctrl+C copies (the owner's choice)
    ENABLE_COMMAND_PALETTE = False       # Textual's own palette (Ctrl+P): another look, and it switches themes

    def __init__(self, make_agent, welcome=lambda: "✻ peresearch", status=lambda reason=None: "",
                 workspace=None, index_on_start: bool = True, project: str = "default", model_check=None,
                 endpoint_check=None, setup_on_start: bool = False):
        super().__init__(ansi_color=True)
        self.model_check = model_check or check_settings
        self.endpoint_check = endpoint_check or check_endpoints
        self.setup_on_start, self._checking_connections = setup_on_start, False
        self.make_agent, self.welcome, self.ws, self.project = make_agent, welcome, workspace, project
        self.status_text = status                      # /status
        self.index_progress = None                     # (files reached, files to look at), set by the update
        self.walking, self.pose = 0, None              # ticks since work began; the mascot's beat (None: at rest)
        self.index_on_start = index_on_start and workspace is not None
        self.agent, self.busy, self.last = None, False, None
        self.pending_resume = None                    # a conversation picked before the agent existed
        self.pending_name = None                      # a name given before the agent existed
        self.last_question = None                     # (as shown, as sent) for /retry
        self.show_reasoning, self.pages = False, []   # Ctrl+R; this session's answers, to redraw them
        self.sources_of = {}                          # a shown page -> the answer whose sources its links name
        self.menu_kind = ""                           # what the menu lists: "commands", "paths" or "conversations"
        self.asked, self.recall = [], None
        self.started, self.doing, self.tokens, self.frame = 0.0, "", 0, 0
        self.indexing = ""                             # what the index update is doing, apart from a question
        self.flash, self.flash_until = "", 0.0         # a short message on the status line when idle
        self.stopping, self.background = threading.Event(), []   # set on quitting; the threads doing work
        self.local_only = False                        # /local: the web off for this session
        self.exchanges = []                            # (question as sent, answer) since opening or /new: /export
        self.asking = ""                               # the question being answered, as sent
        self._control_sink = None                      # a control window owns its administrative feedback

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log", can_focus=False)
        self.status = Static("", id="status")       # kept: the spinner may tick once more after unmounting
        self.bird = Static(_drawn(None), id="mascot")      # lying on the box below it (perch)
        with Horizontal(id="bar"):
            yield self.status
            yield Static("", id="perch")                   # the room the mascot's body takes above the box
        with Horizontal(id="input-box"):
            yield Static("❯", id="prompt-mark")
            yield Prompt(placeholder="Ask about your files or the web", id="ask")
        yield Menu(id="commands")
        yield self.bird

    def on_mount(self) -> None:
        self.register_theme(TERMINAL)
        self.theme = "terminal"
        self.query_one("#ask", Prompt).focus()
        self.set_interval(TICK, self.tick)
        self.call_after_refresh(self.place_mascot)
        self.add(Static(self.welcome(), id="welcome", markup=False))     # it opens the log and scrolls away
        if self.setup_on_start and not settings.get("PERESEARCH_LLM_URL"):
            self.call_after_refresh(self.window_command, "/model")
            return
        if self.index_on_start:
            if self.ws.folders():
                self.reindex("Checking your folders for changes", opening=True)
            else:
                self.control_notice("No folders yet: /folders lets peresearch read one.")

    def place_mascot(self) -> None:
        """Lay the mascot on the input box: its belly on the box's top border, its last two lines over the border and
        the box's first line, its tail over the box's right edge. Asked on every tick: the box grows with the text.
        While a window is open it is hidden, so it never lies across one."""
        try:
            box = self.query_one("#input-box")
        except NoMatches:                                   # the screen is gone (quitting)
            return
        shown = len(self.screen_stack) <= 1
        if self.bird.display != shown:
            self.bird.display = shown
        if not box.region.width:                            # not laid out yet
            return
        lines = mascot.REST_LINES if self.pose is None else mascot.WORK_LINES
        offset = (box.region.right - mascot.WIDTH, box.region.y + 2 - lines)
        if tuple(self.bird.styles.offset) != offset:
            self.bird.styles.offset = offset

    # --- the conversation ---

    def add(self, widget) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)

    def note(self, text: str) -> None:
        if self._control_sink is not None:
            self._control_sink(text)
            return
        self.add(Static(guard.sanitize(text), classes="note", markup=False))   # names, paths, typed text: as typed

    def run_control(self, action, sink) -> None:
        previous, self._control_sink = self._control_sink, sink
        try:
            action()
        finally:
            self._control_sink = previous

    def control_notice(self, text: str) -> None:
        self.say(guard.redact(text)[0])

    def window_command(self, text: str) -> bool:
        from peresearch.windows import ChecksWindow, FolderWindow, InfoWindow, ModelWindow, SessionWindow, SettingsWindow

        name = text.split(" ")[0]
        if name in ("/folders", "/add", "/remove", "/index"):
            self.push_screen(FolderWindow(self.ws))
        elif name in ("/model", "/setup"):
            self.push_screen(ModelWindow(self.setup_done))
        elif name in ("/settings", "/config"):
            self.push_screen(SettingsWindow(self.setup_done))
        elif name in ("/sessions", "/resume", "/rename", "/export"):
            self.push_screen(SessionWindow())
        elif name == "/status":
            try:
                body = self.status_text(text[len(name):].strip() or None)
            except Exception as error:
                body = explain(error)
            tabs = {"Setup and index": body, "Usage": self.usage_text(),
                    "Context": self.context_text(self.agent.context_report()) if self.agent is not None and
                    hasattr(self.agent, "context_report") else "Ask a question first to measure model context."}
            self.push_screen(InfoWindow("Status", body, tabs))
        elif name in ("/help", "/keys"):
            commands, keys = columns(COMMANDS), columns(KEYS)
            self.push_screen(InfoWindow("Help", "", {"Commands": commands, "Keys": keys}))
        elif name in ("/usage", "/context"):
            body = self.usage_text() if name == "/usage" else self.context_text(self.agent.context_report()) \
                if self.agent is not None and hasattr(self.agent, "context_report") else "Ask a question first."
            self.push_screen(InfoWindow("Usage" if name == "/usage" else "Context", body))
        elif name == "/sources":
            if self.last and self.last.sources:
                body = "\n\n".join(f"**{source.id}** {guard.sanitize(source.where)}\n\n> "
                                     + guard.sanitize(source.text[:600]).replace("\n", "\n> ")
                                     for source in self.last.sources.values())
            else:
                body = "No sources yet."
            self.push_screen(InfoWindow("Sources", body))
        elif name == "/check":
            self.push_screen(ChecksWindow())
        else:
            return False
        return True

    SETUP_EVERY = 2.0             # seconds between two readings of the settings for the input box's border

    def setup_line(self) -> str:
        """What the input box's bottom border says: the chat model in use, the declared folders, the web off."""
        model = settings.get("PERESEARCH_LLM_MODEL") or (connections.host(settings.get("PERESEARCH_LLM_URL"))
                                                        if settings.get("PERESEARCH_LLM_URL") else "")
        try:
            folders = len(self.ws.folders()) if self.ws else 0
        except guard.ConfigError:
            folders = 0
        parts = [guard.sanitize(model) if model else "no model: /model",
                 f"{folders} folder{'' if folders == 1 else 's'}"]
        return " · ".join(parts + (["web off"] if self.local_only else []))

    def update_setup_line(self, now: bool = False) -> None:
        if not now and time.time() < getattr(self, "_setup_read", 0.0):
            return
        self._setup_read = time.time() + self.SETUP_EVERY
        try:
            self.query_one("#input-box").border_subtitle = self.setup_line()
        except NoMatches:
            pass

    def tick(self) -> None:
        status = self.status
        self.walking = self.walking + 1 if self.busy or self.indexing else 0    # the mascot thinks while work runs
        pose = self.walking // round(mascot.STEP / TICK) % mascot.FRAMES if self.walking else None
        if pose != self.pose and self.is_running:           # a late tick after closing draws nothing
            self.pose = pose
            self.bird.update(_drawn(pose))
        if self.is_running:
            self.place_mascot()
            self.update_setup_line()
        if not self.busy and not self.indexing:
            idle = "Checking connections…" if self._checking_connections else "local only · web off" if self.local_only else ""
            status.update(self.flash if time.time() < self.flash_until else idle)
            return
        self.frame += 1
        spin = SPINNER[self.frame % len(SPINNER)]
        if not self.busy:                              # only the index update runs
            done, total = self.index_progress or (0, 0)
            far = f" {done:,}/{total:,} files" if total else ""
            status.update(f"{spin} {guard.sanitize(self.indexing)}…{far} (you can ask meanwhile)")
            return
        extra = f" · {self.tokens:,} tokens" if self.tokens else ""
        status.update(f"{spin} {guard.sanitize(self.doing)}… ({time.time() - self.started:.0f}s{extra} · esc to interrupt)")

    def show_draft(self, text: str) -> None:
        """The answer as the model writes it, under the conversation: one block, redrawn with the text so far."""
        page = getattr(self, "_draft_page", None)       # held, not queried: a draft mounted a moment ago counts
        if page is None:
            self._draft_page = Page(guard.sanitize(text), classes="draft")
            self.add(self._draft_page)
        elif not page.is_mounted:              # Markdown renders its initial text when it mounts: that text moves on
            page._initial_markdown = guard.sanitize(text)
        else:
            page.update(guard.sanitize(text))
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    def drop_draft(self) -> None:
        page, self._draft_page = getattr(self, "_draft_page", None), None
        if page is not None:
            page.remove()

    def on_agent_event(self, kind: str, detail: str) -> None:
        if kind == "draft":
            self.show_draft(detail)
            return
        if kind in ("tool", "think"):       # a call, or the next model call: the text drafted so far was not the answer
            self.drop_draft()
        if kind == "tool":
            self.add(Static(Text.assemble(("● ", "green"), guard.sanitize(detail)), classes="tool"))   # its arguments
            self.doing = ("Searching" if detail.startswith(("search", "grep", "glob", "web")) else
                          "Looking through folders" if detail.startswith("tree") else "Reading")
        elif kind == "result":
            self.add(Static(f"  ⎿  {guard.sanitize(detail)}", classes="result", markup=False))
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
        if menu.display and menu.highlighted is not None and \
                menu.get_option_at_index(menu.highlighted).id.startswith("mention:"):     # an @path picked
            self.take_mention(menu.get_option_at_index(menu.highlighted).id, event.value)
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
        if self._checking_connections:
            self.note("Wait for the connection checks to finish before asking a question.")
            return
        if self.busy:
            self.note("still working on the previous question (esc to interrupt it)")
            return
        self.asked.append(q)
        self.recall = None
        self.ask(q, with_mentions(self.query_one("#ask", Prompt).expand(q), self.ws.folders() if self.ws else []))

    def ask(self, shown: str, sent: str) -> None:
        self.last_question = (shown, sent)
        self.asking = sent
        self.start("Starting")
        self.add(Static(asked_line(shown.replace(MARK, "")), classes="question"))
        self.run_agent(sent)

    def _in_background(self, job, *args) -> None:
        """A question or an index update, in a daemon thread. Textual's thread workers run in asyncio's executor,
        which Python waits for on exit: quitting took 20 s behind an index update and 300 s behind a model call
        (measured 2026-10-05). A daemon thread is asked to stop and left if it does not (the index rolls back an
        unfinished batch)."""
        thread = threading.Thread(target=job, args=args, daemon=True)
        self.background = [t for t in self.background if t.is_alive()] + [thread]
        thread.start()

    def _ui(self, fn, *args) -> None:
        """Call `fn` on the interface from a background thread, unless the interface is gone (quitting)."""
        if not self.is_running or self.stopping.is_set():
            return
        try:
            self.call_from_thread(fn, *args)
        except RuntimeError:                             # closed between the check and the call
            pass

    def run_agent(self, question: str) -> None:
        self._in_background(self._run_agent, question)

    def _run_agent(self, question: str) -> None:
        def event(kind, detail):
            self._ui(self.on_agent_event, kind, detail)
        try:
            if self.agent is None:
                self._ui(self.on_agent_event, "wait", "Loading your configured services")
                self.agent = self.make_agent(event)
                if hasattr(self.agent, "toolbox"):
                    self.agent.toolbox.ask = self.permission
                    self.agent.toolbox.local_only = self.local_only
                if self.pending_resume:                 # a new agent starts a new conversation, else this one
                    self.agent.resume(self.pending_resume)
                if self.pending_name:
                    self.agent.rename(self.pending_name)
                    self.pending_name = None
            answer = self.agent.ask(question)
            self.last = answer
            self._ui(self.show_answer, answer)
        except Exception as e:
            self._ui(self.add, Page(explain(e)))
        finally:
            self._ui(self.done)

    def show_answer(self, answer) -> None:
        self.drop_draft()                      # the checked answer, with its links, replaces what was drafted
        self.exchanges.append((self.asking, answer))
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
                self.link_notice(f"{sid}: no such source")
            elif kind == "source":
                self.show_source(source, self.sources_of[event.markdown])
            else:
                target, line = (source.where, None) if source.kind == "web" else location(source.where)
                why = open_outside(target, line)
                self.link_notice(f"not opened: {why}" if why else f"opened {target}")
        elif re.match(r"(?i)https?://", href):
            why = open_outside(href)
            self.link_notice(f"not opened: {why}" if why else f"opened {href}")
        else:
            self.link_notice("not opened: only web links (http, https) and your own files open")

    def link_notice(self, text: str) -> None:
        from peresearch.windows import Window
        if isinstance(self.screen, Window):
            self.screen.message(text)
        else:
            self.control_notice(text)

    def show_source(self, source, answer) -> None:
        from peresearch.windows import SourceWindow
        excerpt = guard.sanitize(source.text[:600]).replace("\n", "\n> ") if source.text else \
            "_(the excerpt is not kept in the history)_"
        what = "open the page" if source.kind == "web" else "open the file"
        body = (f"**{source.id}** {guard.sanitize(source.where)}"
                    + (f" — {guard.sanitize(source.title)}" if source.title else "")
                    + f"\n\n> {excerpt}\n\n[{what}](open:{source.id})")
        self.push_screen(SourceWindow(body, answer))

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
        self._ui(show)
        while not answered.wait(0.1):
            if not self.is_running or self.stopping.is_set():      # quit while the dialog was open
                return "no"
        if answer.get("choice") == "stop":              # Esc on the dialog: refuse, and stop the whole question
            self.agent.cancel()
            return "no"
        choice = answer.get("choice") or "no"
        assert choice in ("once", "session", "no"), f"permission dialog answered {choice!r}"
        return choice

    def done(self) -> None:
        self.busy = False
        self.drop_draft()                      # interrupted or failed: no draft is left as if it were an answer

    def reindex(self, doing: str, opening: bool = False) -> None:
        notify = self._control_sink or (self.control_notice if opening else None)
        self._in_background(self._reindex, doing, opening, notify)

    def _reindex(self, doing: str, opening: bool = False, notify=None) -> None:
        """Brings the index up to date in the background; questions can be asked meanwhile (search copes with an
        index being updated), and the status line shows the question first and how far the update is. Quitting stops
        it at the next file. One line says what it did (none when opening found nothing to do); /status the rest."""
        self._ui(setattr, self, "indexing", doing)
        try:
            r = self.ws.update(stop=self.stopping.is_set,
                               progress=lambda done, total: setattr(self, "index_progress", (done, total)))
            if not (opening and quiet(r)):
                self._ui(notify or self.note, brief(r))
        except Stopped:                                  # quitting: what it did is kept, the next start goes on
            pass
        except Exception as e:
            why = (f"an internal check failed at {place(e)}: {e}" if isinstance(e, AssertionError)
                   else f"{type(e).__name__}: {e}")
            self._ui(notify or self.note, f"indexing failed: {why}"[:300])
        finally:
            self._ui(self.indexed)

    def indexed(self) -> None:
        self.indexing, self.index_progress = "", None

    async def action_quit(self) -> None:
        """Quit at once: an index update is asked to stop (what it did is kept), a question is interrupted, and work
        that has not stopped after QUIT_GRACE seconds (an embedding on a CPU) is left behind."""
        self.stopping.set()
        if self.agent is not None and hasattr(self.agent, "cancel"):
            self.agent.cancel()
        end = time.monotonic() + QUIT_GRACE
        while any(t.is_alive() for t in self.background) and time.monotonic() < end:
            await asyncio.sleep(0.05)
        self.exit()

    def start(self, doing: str) -> None:
        self.busy, self.started, self.doing, self.tokens = True, time.time(), doing, 0

    # --- commands ---

    def command(self, text: str) -> None:
        """User-facing command router: groups open windows; immediate actions use the status line."""
        if not self.window_command(text):
            self.run_control(lambda: self.action_command(text), self._control_sink or self.control_notice)

    def action_command(self, text: str) -> None:
        """Application actions retained for window controllers and existing integrations."""
        name = text.split(" ")[0]
        if self._checking_connections and name not in ("/exit", "/help", "/keys", "/status", "/usage"):
            self.note("Wait for the connection checks to finish before changing the session or settings.")
            return
        if name == "/exit":
            self.run_worker(self.action_quit())
        elif name == "/help":
            self.add(Page("**Commands** (the keys: `/keys`)\n" + "\n".join(f"- `{c}` {d}" for c, d in COMMANDS.items())))
        elif name == "/keys":
            self.add(Page("**Keys**\n" + "\n".join(f"- **{k}** {d}" for k, d in KEYS.items())))
        elif name in ("/clear", "/new", "/reset"):
            if self.busy:
                self.note("wait for the current question to finish, or press esc")
                return
            given = text[len(name):].strip()
            try:
                given = clean_name(given) if given else ""
            except ValueError as e:
                self.note(str(e))
                return
            if self.agent is not None:
                self.agent.new_session()
            self.pending_resume, self.pending_name = None, None
            self.query_one("#log", VerticalScroll).remove_children()
            self.pages, self.sources_of, self.exchanges = [], {}, []
            self.last, self.last_question = None, None
            self.note("New conversation: earlier questions are no longer sent as context.")
            if given:
                self.name_conversation(given)
        elif name == "/rename":
            given = text[len(name):].strip()
            if not given:
                self.note("/rename <name>: a name for this conversation (/resume finds it by name)")
            else:
                self.name_conversation(given)
        elif name == "/usage":
            self.add(Page(self.usage_text()))
        elif name == "/context":
            if self.agent is None or not hasattr(self.agent, "context_report"):
                self.note("/context measures the model's requests: ask a question first")
            else:
                self.add(Page(self.context_text(self.agent.context_report())))
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
            key = text[len(name):].strip()
            if key:
                found = find_conversation(history_file(self.project), key)
                if found is None:
                    self.note(f"no conversation named or numbered “{key}”")
                else:
                    self.resume(found.id)
                return
            past = conversations(history_file(self.project))
            if not past:
                self.note("no earlier conversation in this project")
                return
            menu = self.query_one("#commands", OptionList)
            menu.clear_options()
            menu.add_options([Option(row(text), id="resume:" + c.id)
                              for c, text in zip(past, conversation_rows(past[:50], self.size.width - 4))])
            menu.display, menu.highlighted, self.menu_kind = True, 0, "conversations"
        elif name == "/sources":
            if not self.last or not self.last.sources:
                self.note("no sources yet")
                return
            self.add(Page("\n\n".join(f"**`{s.id}`** {guard.sanitize(s.where)}\n\n> "
                                          + guard.sanitize(s.text[:600]).replace("\n", "\n> ")
                                          for s in self.last.sources.values())))
        elif name == "/local":
            self.local_only = not self.local_only
            if self.agent is not None and hasattr(self.agent, "toolbox"):
                self.agent.toolbox.local_only = self.local_only
            self.note("Local only: the web is off for this session (/local turns it back on)." if self.local_only
                      else "The web is on again.")
            self.update_setup_line(now=True)
        elif name == "/setup":                         # the old name of /model's window
            self.window_command("/model")
        elif name == "/check":
            if self.busy or self.indexing or self._checking_connections:
                self.note("Wait for the current operation to finish before checking connections.")
                return
            self.push_screen(ConnectionCheck(), self.check_connections)
        elif name == "/config":
            self.config(text[len(name):].split())
        elif name == "/export":
            self.export(text[len(name):].strip())
        elif name == "/status":
            self.add(Page(guard.sanitize(self.status_text(text[len(name):].strip() or None))))
        elif name in ("/folders", "/add", "/remove", "/index"):
            try:
                self.folder_command(name, text)
            except guard.ConfigError as e:              # the user broke config.json while this runs: said
                self.note(str(e))
        else:
            self.note(f"unknown command {name}; / for the list")

    def name_conversation(self, name: str) -> None:
        """Name the current conversation; before the agent exists, the name waits for it."""
        try:
            if self.agent is not None:
                self.agent.rename(name)
            elif self.pending_resume:
                # Renaming a resumed conversation must persist even before the next model is constructed.
                import json
                from peresearch.agent import _append_private
                _append_private(history_file(self.project), json.dumps({"time": time.time(),
                                "conversation": self.pending_resume, "name": clean_name(name)}))
                self.pending_name = None
            else:
                self.pending_name = clean_name(name)
        except ValueError as e:
            self.note(str(e))
            return
        self.note(f"This conversation is named “{clean_name(name)}”.")

    @staticmethod
    def context_text(r: dict) -> str:
        """/context's page from Agent.context_report."""
        lines = [f"- {name}: {n:,} characters ≈ {round(n / r['ratio']):,} tokens" for name, n in r["parts"]]
        whole = (f"of the model's {r['context']:,} ({r['tokens'] / r['context']:.0%})" if r["context"] else
                 "(the model's context is not known: set PERESEARCH_LLM_CONTEXT)")
        return (f"**Context** (the next question's request before its own messages; {r['ratio']:.1f} characters a "
                f"token, as last measured)\n" + "\n".join(lines) + f"\n- in all: ≈ {r['tokens']:,} tokens {whole}")

    def usage_text(self) -> str:
        """Model requests and tokens (sent, generated) of this conversation, of today and of the project."""
        path = history_file(self.project)
        current = self.agent.conversation if self.agent is not None else self.pending_resume
        today = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        rows = [("this conversation", usage(path, current) if current else dict.fromkeys(USAGE, 0)),
                ("today, this project", usage(path, since=today)), ("this project", usage(path))]
        return "**Usage** (model requests · tokens sent · tokens generated)\n" + "\n".join(
            f"- {label}: {u['requests']:,} · {u['prompt_tokens']:,} · {u['completion_tokens']:,}" for label, u in rows)

    def folder_command(self, name: str, text: str) -> None:
        if name == "/folders":
            folders = self.ws.folders() if self.ws else []
            self.note("\n".join(map(str, folders)) or "no folders declared: /folders, then a")
        else:
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
            self.update_setup_line(now=True)

    def take_mention(self, chosen: str, value: str) -> None:
        """An @path row taken (Tab, Enter or a click) into `value`, the typed text: it replaces what follows its
        last @."""
        self.set_prompt(value[:value.rindex("@") + 1] + chosen.removeprefix("mention:"))
        self.hide_menu()

    def config(self, words: list[str]) -> None:
        """/config shows each setting with its value and where it comes from; /config <name> <value> checks and
        saves one, /config <name> resets it; either applies from the next question, in the same conversation."""
        if not words:
            self.add(Page(config_listing()))
            return
        if self.busy:
            self.note("wait for the current question to finish, or press esc")
            return
        key, value = words[0], " ".join(words[1:])
        try:
            _, setting, default, _, _ = settings.row(key)
            if value:
                value = settings.check(key, value)[1]
        except ValueError as e:
            self.note(str(e))
            return
        kept = {k: v for k, v in settings.saved().items() if k != setting}
        settings.save({**kept, setting: value} if value else kept)
        self._retire_agent()
        self.note((f"{key} is now {value}; it applies from the next question." if value
                   else f"{key} is back to its default ({default}).")
                  + (f" {setting} is set in the environment, which wins over the file."
                     if os.environ.get(setting) else ""))

    def _retire_agent(self) -> None:
        """The next question gets a new agent, in the same conversation; the old one's search server is stopped."""
        if self.agent is None:
            return
        web = getattr(getattr(self.agent, "toolbox", None), "web", None)
        if hasattr(web, "close"):
            web.close()
        self.pending_resume, self.agent = self.agent.conversation, None

    def setup_done(self, changes) -> None:
        """Save what /setup changed; the next question gets an agent built on it, in the same conversation."""
        if not changes:
            self.note("Setup closed; nothing changed.")
            return
        path = settings.save({**settings.saved(), **changes})
        self._retire_agent()
        overridden = [n for n in changes if os.environ.get(n)]
        self.note(f"Saved to {path}. The new settings take effect from the next question."
                  + (f" {', '.join(overridden)} {'is' if len(overridden) == 1 else 'are'} set in the environment, which "
                     "wins over the file." if overridden else ""))
        self.note("Connections have not been checked. /check runs tests; remote services may charge. "
                  "Use /folders to add documents, or ask a question when your model is configured.")
        self.update_setup_line(now=True)

    def check_connections(self, confirmed: bool, notify=None) -> None:
        if not confirmed or self.busy or self.indexing or self._checking_connections:
            return
        self._checking_connections = True
        self.note("Checking configured connections…")
        self._in_background(self._check_connections, notify)

    def _check_connections(self, notify=None) -> None:
        try:
            self._check_model(notify)
            if not self.stopping.is_set():
                self._check_endpoints(notify)
        finally:
            self._ui(setattr, self, "_checking_connections", False)

    def _check_endpoints(self, notify=None) -> None:
        try:
            lines = self.endpoint_check()
        except Exception as e:
            self._ui(notify or self.note, f"the endpoints could not be checked: {type(e).__name__}: {e}"[:300])
            return
        body = guard.sanitize("**Search endpoints**\n" + "\n".join(f"- {line}" for line in lines))
        self._ui(notify, body) if notify else self._ui(self.add, Page(body))

    def _check_model(self, notify=None) -> None:
        try:
            works, lines = self.model_check()
        except Exception as e:
            self._ui(notify or self.note, f"the model could not be checked: {type(e).__name__}: {e}"[:300])
            return
        body = guard.sanitize(
            "**Model check** " + ("· it works with the agent" if works else
                                  "· **it does not call tools**: the agent cannot search your files or the web with it")
            + "\n" + "\n".join(f"- {line}" for line in lines))
        self._ui(notify, body) if notify else self._ui(self.add, Page(body))

    def export(self, arg: str) -> None:
        """This conversation as Markdown (export_markdown) into a new file: `arg` a .md file, or a folder (default the
        current one) that gets a dated name. Never overwrites, never writes into a protected place."""
        if not self.exchanges:
            self.note("nothing to export yet: ask a question first")
            return
        target = Path(arg).expanduser() if arg else Path.cwd()
        target = target if target.is_absolute() else Path.cwd() / target
        if guard.denied(target) or guard.denied(Path(os.path.realpath(target))):
            self.note(f"not saved: {target} is a protected place")
            return
        if target.is_dir():
            target = target / f"peresearch-{self.project}-{time.strftime('%Y-%m-%d-%H%M%S')}.md"
        elif target.suffix.lower() != ".md":
            self.note(f"not saved: {target} is not a folder")
            return
        try:
            with open(target, "x", encoding="utf-8") as f:          # "x": an existing file is never overwritten
                f.write(export_markdown(self.project, self.exchanges))
        except FileExistsError:
            self.note(f"not saved: {target} exists (give another name)")
            return
        except OSError as e:
            self.note(f"not saved: {target}: {e.strerror or e}")
            return
        self.note(f"Saved to {target}")

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
            self.pending_resume, self.pending_name = conversation, None   # a name waiting
            # for the conversation left behind is not this one's
        log = self.query_one("#log", VerticalScroll)
        log.remove_children()
        self.pages, self.sources_of, self.exchanges = [], {}, []
        self.last, self.last_question = None, None
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(turns[0].get("time", 0))) if turns else "?"
        self.note(f"Resumed the conversation of {when} ({len(turns)} question{'' if len(turns) == 1 else 's'}); "
                  "the next question continues it.")
        from peresearch.agent import ResultStore
        from peresearch.evidence import Sources
        registry = (self.agent.source_registry() if self.agent is not None else
                    Sources.load(turns, ResultStore(self.project, conversation),
                                 lambda path: guard.allowed(Path(path))))
        for t in turns:
            self.add(Static(asked_line(t["question"]), classes="question"))
            kept = registry.for_turn(t)
            measured = t.get("usage") or {}
            answer = Answer(t["answer"], kept, check(t["answer"], kept), measured.get("requests", 0),
                            measured.get("prompt_tokens", 0), measured.get("completion_tokens", 0))
            self.exchanges.append((t["question"], answer))
            self.last, self.last_question = answer, (t["question"], t["question"])
            page = Page(render(answer).rsplit("\n<sub>", 1)[0])
            self.sources_of[page] = answer
            self.add(page)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        prompt = self.query_one("#ask", Prompt)
        self.hide_menu()
        if event.option.id.startswith("resume:"):
            self.resume(event.option.id.removeprefix("resume:"))
        elif event.option.id.startswith("mention:"):
            self.take_mention(event.option.id, prompt.value)
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
            if chosen.startswith("mention:"):
                self.take_mention(chosen, value)
                return
            self.set_prompt((value.split(" ")[0] + " " + chosen[5:]) if chosen.startswith("path:") else chosen + " ")
            self.hide_menu()
        elif value.rsplit(" ", 1)[-1].startswith("@") and not value.startswith("/"):      # @path being typed
            partial = value[value.rindex("@") + 1:]
            matches = mention_matches(partial, self.ws.folders() if self.ws else [])
            if len(matches) == 1:
                self.set_prompt(value[:value.rindex("@") + 1] + matches[0])
            elif matches:
                menu.clear_options()
                menu.add_options([Option(row(m), id="mention:" + m) for m in matches[:50]])
                menu.display, menu.highlighted, self.menu_kind = True, 0, "paths"
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
                menu.add_options([Option(row(m), id="path:" + m) for m in matches[:50]])
                menu.display, menu.highlighted, self.menu_kind = True, 0, "paths"

    def say(self, text: str, seconds: float = 2.5) -> None:
        """A short message on the status line, shown while nothing runs."""
        self.flash, self.flash_until = text, time.time() + seconds
        self.tick()

    def action_copy(self) -> None:
        """Ctrl+C: the text selected with the mouse in the conversation, else the prompt's selection."""
        text = (self.screen.get_selected_text() or self.query_one("#ask", Prompt).selected_text or "").replace(MARK, "")
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
