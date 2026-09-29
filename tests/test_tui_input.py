"""The TUI in a real pseudo-terminal, through Textual's own driver and key parser (not the test pilot).

- Vietnamese typing: the TUI is fed the bytes UniKey actually made a terminal send (tests/data/unikey_telex.json),
  with their recorded timing. An input method erases the raw letters and types the accented ones in one burst;
  Textual's own Input spelled that sentence wrong in every run, on Windows Terminal and here.
- Colors: everything the TUI draws uses the terminal's default colors and 16-color palette, so the terminal's
  color scheme applies; an RGB or 256-color code would ignore it. Black and white (palette 0, 7, 8, 15) are not
  used either: many schemes make one of them the background (Solarized Dark's black is its background, which
  made a black-on-white cursor invisible)."""
import json
import os
import re
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

pty = pytest.importorskip("pty")
fcntl = pytest.importorskip("fcntl")
termios = pytest.importorskip("termios")

DATA = json.loads((Path(__file__).parent / "data" / "unikey_telex.json").read_text(encoding="utf-8"))

APP = """
import sys, time
from pathlib import Path
from types import SimpleNamespace as NS
from peresearch.tui import Chat

MD = '''## Heading
Some **bold**, *italic*, `code` and a [link](https://example.com) [N1].

```python
def f(x):
    return x + 1  # comment
```

> a quote

| a | b |
|---|---|
| 1 | 2 |

- item
'''

class Stub:
    def __init__(self, on_event):
        self.on_event = on_event
        self.toolbox = NS(ask=None)                     # the interface puts its permission dialog here
    def ask(self, q):
        if "outside" in q:
            choice = self.toolbox.ask("read", Path.home() / "Downloads" / "paper.md")
            return NS(text="allowed: " + choice, sources={}, check=NS(unknown_ids=[], unsupported_quotes=[], uncited=False),
                      stopped="", steps=1, prompt_tokens=1, completion_tokens=1)
        self.on_event("tool", 'search_notes("bpe")')
        self.on_event("result", "1 source")
        time.sleep(0.5)
        source = NS(id="N1", where="/notes/bpe.md", title="BPE", text="text")
        return NS(text=MD, sources={"N1": source}, check=NS(unknown_ids=["N9"], unsupported_quotes=["zz"], uncited=False),
                  stopped="", steps=2, prompt_tokens=10, completion_tokens=5)
    def cancel(self): pass
    def new_session(self): pass

class Probe(Chat):
    def on_input_submitted(self, event):
        with open(sys.argv[1], "a", encoding="utf-8") as f:
            f.write(event.value + "\\n")
        super().on_input_submitted(event)

Probe(lambda e: Stub(e), index_on_start=False).run()
"""


class Terminal:
    """The TUI running in a pty; everything it draws is kept in `screen`."""

    def __init__(self, tmp_path):
        app, self.out = tmp_path / "app.py", tmp_path / "submitted.txt"
        app.write_text(APP)
        self.fd, tty = pty.openpty()
        fcntl.ioctl(tty, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 110, 0, 0))
        self.proc = subprocess.Popen([sys.executable, str(app), str(self.out)], stdin=tty, stdout=tty, stderr=tty,
                                     env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
                                     start_new_session=True)
        os.close(tty)
        self.screen = b""
        assert self.read(60, until=b"commands"), "the interface never drew its hints line"
        self.read(0.5)

    def read(self, seconds: float, until: bytes | None = None) -> bool:
        end = time.time() + seconds
        while time.time() < end and not (until and until in self.screen):
            if select.select([self.fd], [], [], 0.02)[0]:
                try:
                    self.screen += os.read(self.fd, 65536)
                except OSError:
                    break
        return bool(until and until in self.screen)

    def write(self, data: str) -> None:
        os.write(self.fd, data.encode())
        if select.select([self.fd], [], [], 0)[0]:          # keep the pty's output from filling up
            self.screen += os.read(self.fd, 65536)

    def submitted(self) -> str | None:
        self.write("\r")
        end = time.time() + 20
        while time.time() < end and not self.out.exists():
            self.read(0.1)
        self.read(0.2)
        return self.out.read_text(encoding="utf-8").splitlines()[0] if self.out.exists() else None

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()
        os.close(self.fd)


@pytest.fixture
def terminal(tmp_path):
    t = Terminal(tmp_path)
    yield t
    t.close()


def test_a_unikey_telex_sentence_types_as_in_a_shell(terminal):
    for gap, text in DATA["reads"]:
        end = time.perf_counter() + gap
        while time.perf_counter() < end:                  # sub-millisecond pauses, as recorded
            pass
        terminal.write(text)
    terminal.read(0.3)
    assert terminal.submitted() == DATA["expected"]


def test_every_color_comes_from_the_terminal_scheme(terminal):
    terminal.write("/")                                   # the command menu
    terminal.read(0.5)
    terminal.write("\x7fwhat is bpe")
    terminal.submitted()                                  # a tool line, its result, the status line, then the answer
    terminal.read(3, until=b"quotes not found")
    terminal.write("/h")
    terminal.read(0.5)
    terminal.write("\x7f\x7fabc")                         # the cursor after typed text, through a blink or two
    terminal.read(1.5)
    terminal.write("\x1b[1;2H")                           # Shift+Home selects the typed text
    terminal.read(0.6)
    terminal.write("\x15read outside please")              # Ctrl+U clears; then the permission dialog
    terminal.submitted()
    assert terminal.read(5, until=b"Yes, this once"), "the permission dialog never appeared"
    terminal.read(0.5)
    terminal.write("\r")
    assert terminal.read(5, until=b"allowed: once")
    shown = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;?<>=:$]*[a-zA-Z~]", "", terminal.screen.decode("utf-8", "replace")))
    for part in ("Heading", "def f", "a quote", "item", "search_notes", "Sources", "quotes not found", "/help",
                 "wants to read", "No (esc)"):
        assert part in shown, part                        # every kind of element was drawn
    codes = re.findall(rb"\x1b\[([0-9;:]*)m", terminal.screen)
    fixed = {c.decode() for c in codes if re.search(rb"(^|;)(38|48)[;:](2|5)[;:]", c)}
    assert codes and not fixed, sorted(fixed)[:5]
    params = [c.decode().split(";") for c in codes]
    unsafe = {"30", "37", "40", "47"} | {str(n) for n in range(90, 98)} | {str(n) for n in range(100, 108)}
    near_background = {";".join(p) for p in params if unsafe & set(p)}
    assert not near_background, sorted(near_background)[:5]
    # A colored background only under a scrollbar or other blocks, never under text.
    on_color = re.findall(rb"\x1b\[([0-9;]*)m([^\x1b]*)", terminal.screen)
    text_on_color = {(c.decode(), t.decode("utf-8", "replace")) for c, t in on_color
                     if {b"41", b"42", b"43", b"44", b"45", b"46"} & set(c.split(b";")) and re.search(rb"\w", t)}
    assert not text_on_color, sorted(text_on_color)[:5]
    cursor = re.findall(rb"abc\x1b\[0m\x1b\[([0-9;]*)m ", terminal.screen)
    assert any(b"7" in c.split(b";") for c in cursor), cursor   # the cursor is drawn: the default colors reversed
    selected = re.findall(rb"\x1b\[([0-9;]*)mbc", terminal.screen)      # "a" carries the cursor, "bc" the selection
    assert selected and b"7" in selected[-1].split(b";"), selected  # selected text: the default colors reversed


def test_the_chosen_command_stands_out_in_the_menu(terminal):
    """The menu never has the focus (typing stays on the prompt), so its highlight is the blurred one; it must
    still show which command Enter or Tab would take."""
    def style_of(name: bytes) -> set[bytes]:
        found = re.findall(rb"\x1b\[([0-9;]*)m" + re.escape(name) + rb"\b", terminal.screen)
        return set(found[-1].split(b";")) if found else set()
    terminal.write("/")
    terminal.read(0.8)
    assert b"7" in style_of(b"/help") and b"7" not in style_of(b"/add")
    terminal.write("\x1b[B")                              # ↓ moves the choice
    terminal.read(0.8)
    assert b"7" in style_of(b"/add") and b"7" not in style_of(b"/help")


def test_editing_keys_keep_their_place_among_typed_text(terminal):
    """Keys bound to an action (Ctrl+U, arrows, Enter…) arriving in one read with typed text: Textual inserts the
    text at once and ran the action later, so "abc ⌃U xyz" came out empty and "ab ← c" as "abc"."""
    for burst in ("abc\x15xyz\r", "ab\x1b[Dc\r", "one\rtwo\r"):
        terminal.write(burst)
        terminal.read(1.5)
    assert terminal.out.read_text(encoding="utf-8").splitlines() == ["xyz", "acb", "one", "two"]


def test_a_bracketed_paste_keeps_its_place_among_typed_text(terminal):
    terminal.write("see \x1b[200~first line\nsecond line\x1b[201~ ok\r")
    terminal.read(1.5)
    assert terminal.out.read_text(encoding="utf-8").splitlines() == ["see [Pasted text #1 +2 lines] ok"]
