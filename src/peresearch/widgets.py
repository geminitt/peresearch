"""Shared interface pieces, in the style of Claude Code and OpenCode: a list whose chosen row carries a `❯` pointer
in the accent color (no reversed bar, no buttons), one-line form fields (a label, then the value, no box around it)
and paths shortened the way a reader scans them (`~` for the home folder, a long one cut in the middle).

Colors are the terminal's own (tui.TERMINAL): the accent is ansi_blue, muted text is dimmed."""

from pathlib import Path

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.strip import Strip
from textual.widgets import Input, OptionList, Static

POINTER = "❯ "
GUTTER = "  "                        # every row starts with it; the chosen row's is replaced by the pointer
ACCENT = Style(color="blue", bold=True)


def home_path(path: str) -> str:
    """A path with the home folder written as `~`."""
    home = str(Path.home()).rstrip("/")
    text = str(path)
    if home and (text == home or text.startswith(home + "/")):
        return "~" + text[len(home):]
    return text


def middle(text: str, width: int) -> str:
    """`text` in at most `width` cells, cut in the middle with "…" (the start and the file name stay readable)."""
    if width <= 1 or cell_len(text) <= width:
        return text if cell_len(text) <= max(width, 1) else "…"
    keep = width - 1
    head, tail = keep - keep // 2, keep // 2
    end = text[-tail:] if tail else ""
    return text[:head] + "…" + end


def row(content) -> Text:
    """A list row: the gutter the pointer takes on the chosen row, then the content (a str or a Text)."""
    return Text.assemble(GUTTER, content if isinstance(content, Text) else Text(str(content)))


class SelectList(OptionList):
    """A list of choices. The chosen row shows `❯` and the accent color; the row under the mouse is bold. Rows are
    built with `row()`, whose two-cell gutter the pointer replaces."""

    DEFAULT_CSS = """
    SelectList { border: none; padding: 0; height: auto; max-height: 16; background: ansi_default; }
    SelectList:focus { border: none; }
    """

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        try:
            index, line = self._lines[self.scroll_offset.y + y]
        except (IndexError, AttributeError):
            return strip
        if index == self.highlighted:                  # the pointer on the row's first line; a wrapped rest follows
            rest = strip.crop(len(GUTTER), strip.cell_length)
            return Strip.join([Strip([Segment(POINTER if line == 0 else GUTTER, ACCENT)]), over(rest, ACCENT)])
        if index == self._mouse_hovering_over:
            return over(strip, Style(bold=True))
        return strip


def over(strip: Strip, style: Style) -> Strip:
    """`style` laid over every segment's own (Strip.apply_style puts it under them: a row's own color won)."""
    return Strip([Segment(text, (own or Style()) + style, control) for text, own, control in strip],
                 strip.cell_length)


class Field(Horizontal):
    """One form row: the label (dim; the accent while its value has the focus) and the value. A narrow value is
    followed by its note on the same line; a window shows any field's note on its detail line while it has the
    focus (Window.on_descendant_focus)."""

    DEFAULT_CSS = """
    Field { height: 1; width: 1fr; }
    Field > .field-label { width: 22; height: 1; color: ansi_default; text-style: dim; }
    Field:focus-within > .field-label { color: ansi_blue; text-style: bold; }
    Field > Input { width: 1fr; height: 1; border: none; padding: 0; background: ansi_default; color: ansi_default; }
    Field > Input:focus { border: none; }
    Field > Input.-narrow { width: 12; }
    Field > .field-note { width: 1fr; height: 1; padding: 0 0 0 2; color: ansi_default; text-style: dim;
                          text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, label: str, value: str = "", *, id: str, password: bool = False, placeholder: str = "",
                 note: str = "", narrow: bool = False):
        super().__init__(classes="field")
        self.label, self.value, self.input_id = label, value, id
        self.password, self.placeholder, self.note_text, self.narrow = password, placeholder, note, narrow

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="field-label", markup=False)
        yield Input(self.value, id=self.input_id, password=self.password, placeholder=self.placeholder,
                    classes="-narrow" if self.narrow else "")
        if self.note_text and self.narrow:            # a short value leaves room to say what it does
            yield Static(self.note_text, classes="field-note", markup=False)


def hints(*pairs: tuple[str, str]) -> str:
    """A key hint line: `key action · key action`, as Claude Code's footers read."""
    return " · ".join(f"{key} {action}" for key, action in pairs)
