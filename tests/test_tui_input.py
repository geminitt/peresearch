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
    def ask(self, q):
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
    shown = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;?<>=:$]*[a-zA-Z~]", "", terminal.screen.decode("utf-8", "replace")))
    for part in ("Heading", "def f", "a quote", "item", "search_notes", "Sources", "quotes not found", "/help"):
        assert part in shown, part                        # every kind of element was drawn
    codes = re.findall(rb"\x1b\[([0-9;:]*)m", terminal.screen)
    fixed = {c.decode() for c in codes if re.search(rb"(^|;)(38|48)[;:](2|5)[;:]", c)}
    assert codes and not fixed, sorted(fixed)[:5]
    params = [c.decode().split(";") for c in codes]
    near_background = {";".join(p) for p in params if {"30", "37", "40", "47", "90", "97", "100", "107"} & set(p)}
    assert not near_background, sorted(near_background)[:5]
    cursor = re.findall(rb"abc\x1b\[0m\x1b\[([0-9;]*)m ", terminal.screen)
    assert any(b"7" in c.split(b";") for c in cursor), cursor   # the cursor is drawn: the default colors reversed
