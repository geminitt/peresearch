"""The extracted-text store: the text of files whose bytes are not that text (a PDF, a notebook, a text file in an
encoding other than UTF-8), kept as UTF-8 so that a plain-text search (ripgrep, for the grep tool) reads it without
converting the file again, and finds what the index found.

One file per content, `<PERESEARCH_HOME>/text/<sha256 of the source>.txt`: copies share it, and a store file is never
rewritten in place (another content has another name), so a reader sees a whole file or none. Each block of the
source (a page, a cell, or the whole text) starts on a new line of the store file, and `units` maps a line of the
store file back to its place in the source (`locate`). A text file's lines keep their numbers.

The index keeps the store in step with itself (`Index.update`): a text is written for every indexed content that
`needed`, and every other file in the folder is removed, so a crash, an update that failed, or a file deleted by
hand is mended by the next update without converting or embedding anything else. The folder is private to the user
(0700, files 0600): it is a copy of the user's text.
"""

import bisect
import os
from pathlib import Path

from peresearch.zetokrag import parse

FOLDER = "text"


def needed(path: str, status: str, note: str) -> bool:
    """Whether a file indexed with this status and parse note has a text other than its bytes read as UTF-8."""
    if status not in ("ok", "data"):
        return False
    return parse.kind(Path(path)) in parse.CONVERTED or bool(note)   # a text file's note names its encoding


def render(doc: parse.Doc) -> tuple[str, list[list]]:
    """(text, units): the blocks one after another, each from a new line, and [first line, unit, start] per block."""
    parts, units, line = [], [], 1
    for b in doc.blocks:
        text = b.text if b.text.endswith("\n") else b.text + "\n"
        units.append([line, b.unit, b.start])
        parts.append(text)
        line += text.count("\n")
    return "".join(parts), units


def locate(units: list[list], line: int) -> tuple[str, int]:
    """The source's (unit, number) for line `line` (from 1, as ripgrep counts) of a store file."""
    assert units and units[0][0] == 1 and line >= 1, f"line {line} of a store file mapped by {units[:2]}"
    first, unit, start = units[bisect.bisect_right([u[0] for u in units], line) - 1]
    return unit, start + line - first if unit == "line" else start


def write(folder: Path, sha: str, text: str) -> Path:
    """Write the store file for `sha` whole or not at all (a temporary file, flushed, then renamed)."""
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    target, tmp = folder / f"{sha}.txt", folder / f".{sha}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8", errors="replace"))                 # bytes: no newline translation on Windows
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    return target
