"""Split a document into chunks that keep their source: folder → file → section → paragraph.

Paragraphs (separated by blank lines, or notebook cells, or PDF pages) are packed into chunks of at most
MAX_CHARS without crossing a Markdown section; a paragraph longer than that is cut at line boundaries.
Every chunk records its section heading path and the first and last line (or page, or cell) it covers, so
an excerpt can always be quoted with its exact place in the file.
"""

import re
from dataclasses import dataclass

from peresearch.zetokrag.parse import Doc

MAX_CHARS = 1500
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    section: str
    unit: str
    start: int
    end: int
    text: str


def _paragraphs(doc: Doc):
    """(section, unit, start, end, text) per paragraph."""
    markdown = doc.path.suffix.lower() in (".md", ".markdown")
    for block in doc.blocks:
        if block.unit != "line":
            for para in re.split(r"\n\s*\n", block.text):
                if para.strip():
                    yield "", block.unit, block.start, block.start, para.strip()
            continue
        heads: list[str] = []
        buf, first = [], None
        section = ""
        in_fence = False
        for n, line in enumerate(block.text.splitlines(), start=block.start):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            m = _HEADING.match(line) if markdown and not in_fence else None
            if m or not line.strip():
                if buf:
                    yield section, "line", first, n - 1, "\n".join(buf)
                    buf, first = [], None
                if m:
                    level = len(m.group(1))
                    heads = heads[:level - 1] + [m.group(2)]
                    section = " > ".join(heads)
                    yield section, "line", n, n, line.strip()
                continue
            if first is None:
                first = n
            buf.append(line)
        if buf:
            yield section, "line", first, first + len(buf) - 1, "\n".join(buf)


def _split_long(section, unit, start, end, text):
    """Cut a paragraph longer than MAX_CHARS at line boundaries (a single overlong line by characters)."""
    if len(text) <= MAX_CHARS:
        yield section, unit, start, end, text
        return
    lines = text.split("\n")
    buf, first = [], start
    for i, line in enumerate(lines):
        n = start + i if unit == "line" else start
        while len(line) > MAX_CHARS:
            if buf:
                yield section, unit, first, n - 1 if unit == "line" else n, "\n".join(buf)
                buf = []
            yield section, unit, n, n, line[:MAX_CHARS]
            line = line[MAX_CHARS:]
            first = n
        if buf and len("\n".join(buf + [line])) > MAX_CHARS:
            yield section, unit, first, n - 1 if unit == "line" else n, "\n".join(buf)
            buf = []
        if not buf:
            first = n
        buf.append(line)
    if buf:
        yield section, unit, first, end, "\n".join(buf)


def chunk(doc: Doc, keep=None) -> tuple[list[Chunk], int]:
    """(chunks, paragraphs withheld). A paragraph for which `keep(section, text)` is false is left out, and
    no chunk spans across it, so every chunk is still a contiguous, verbatim stretch of the file."""
    out: list[Chunk] = []
    withheld, gap = 0, False
    for para in _paragraphs(doc):
        for section, unit, start, end, text in _split_long(*para):
            if keep is not None and not keep(section, text):
                withheld += 1
                gap = True
                continue
            last = out[-1] if out else None
            if (last and not gap and last.section == section and last.unit == unit
                    and len(last.text) + 2 + len(text) <= MAX_CHARS):
                last.text += "\n\n" + text
                last.end = end
            else:
                out.append(Chunk(section, unit, start, end, text))
            gap = False
    return out, withheld
