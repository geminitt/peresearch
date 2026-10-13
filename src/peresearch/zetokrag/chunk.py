"""Split a document into chunks that keep their source: folder → file → section → paragraph.

Paragraphs (separated by blank lines, or notebook cells, or PDF pages) are packed into chunks of at most a size
(Options: MAX_CHARS characters by default) without crossing a Markdown section; a paragraph longer than that is cut
at line boundaries. Every chunk records its section heading path and the first and last line (or page, or cell) it
covers, so an excerpt can always be quoted with its exact place in the file.

`Options` holds the ways of cutting measured against each other on the gold set (eval/gold.py); `DEFAULT` is the one
the index uses, and its name is kept in the index (a change of chunking rebuilds it). DEFAULT is blocks+code-defs
(owner's choice, 2026-10-03): a table, code block or list is never cut, nor a Python definition. Against the
baseline on the gold dev split (74 questions, pooling bias removed): nDCG@10 +0.007 [-0.007, +0.022], recall@30
+0.032 [-0.001, +0.072], paired bootstrap 95%; no chunking beat the baseline with certainty, this one is chosen on
principle with no measured harm and the best recall@30.
"""

import ast
import warnings
import re
from dataclasses import dataclass

from peresearch.zetokrag.parse import Doc, lines

MAX_CHARS = 1500
VERSION = 2              # 2: a cut piece never starts or ends on a blank line (2026-10-05); recorded in the index
BLOCK_CHARS = 6000          # a code block, table or list kept whole (Options.blocks) may run to this
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")
_LIST = re.compile(r"^\s*([-*+]|\d+[.)])\s")


def tokens(text: str) -> int:
    """A token count close to a BPE tokenizer's: a CJK character about one, other text about four characters each."""
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


@dataclass(frozen=True)
class Options:
    """How documents are cut. Options() reproduces the chunking of baseline-2026-10-03."""
    name: str = "baseline"
    max_chars: int = MAX_CHARS
    max_tokens: int | None = None     # measure size in estimated tokens (`tokens`) instead of characters
    blocks: bool = False              # a fenced code block (blank lines included), a table or a list is not cut
    code_defs: bool = False           # a Python file is cut where its top-level definitions start, not at blank lines

    @property
    def cap(self) -> int:
        return self.max_tokens or self.max_chars

    def size(self, text: str) -> int:
        return tokens(text) if self.max_tokens else len(text)


PRESETS = {o.name: o for o in (Options(), Options("blocks", blocks=True), Options("code-defs", code_defs=True),
                               Options("tokens-450", max_tokens=450),
                               Options("blocks+code-defs", blocks=True, code_defs=True))}
DEFAULT = PRESETS["blocks+code-defs"]


@dataclass
class Chunk:
    section: str
    unit: str
    start: int
    end: int
    text: str


def _is_block(text: str) -> bool:
    first = text.lstrip("\n").split("\n", 1)[0].lstrip()
    return first.startswith(("```", "|")) or bool(_LIST.match(first))


def _python(block) -> list[tuple[int, int, str]] | None:
    """(start, end, text) for each stretch of a Python file from one top-level definition (its decorators and the
    comment lines right above it included) to the next, or None when the file does not parse."""
    try:
        with warnings.catch_warnings():   # 1if, "\d": "<unknown>:2: SyntaxWarning…" among `peresearch index`'s report
            warnings.simplefilter("ignore")
            tree = ast.parse(block.text)
    except (SyntaxError, ValueError):
        return None
    rows = lines(block.text)
    cuts = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            while first > 1 and rows[first - 2].lstrip().startswith("#"):
                first -= 1
            cuts.append(first)
    bounds = sorted({1, *cuts}) + [len(rows) + 1]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        span = list(range(a, b))
        while span and not rows[span[0] - 1].strip():
            span.pop(0)
        while span and not rows[span[-1] - 1].strip():
            span.pop()
        if span:
            out.append((span[0], span[-1], "\n".join(rows[span[0] - 1:span[-1]])))
    return out


def _paragraphs(doc: Doc, opt: Options):
    """(section, unit, start, end, text) per paragraph."""
    markdown = doc.markdown or doc.path.suffix.lower() in (".md", ".markdown")
    for block in doc.blocks:
        if block.unit != "line":
            for para in re.split(r"\n\s*\n", block.text):
                if para.strip():
                    yield "", block.unit, block.start, block.start, para.strip()
            continue
        if opt.code_defs and doc.path.suffix.lower() == ".py":
            stretches = _python(block)
            if stretches is not None:
                for a, b, text in stretches:
                    yield "", "line", block.start + a - 1, block.start + b - 1, text
                continue
        heads: list[str] = []
        buf, first = [], None
        section = ""
        in_fence = False
        for n, line in enumerate(lines(block.text), start=block.start):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            m = _HEADING.match(line) if markdown and not in_fence else None
            if opt.blocks and in_fence and buf and not line.strip():
                buf.append(line)                  # a blank line inside a fence does not end the code block
                continue
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


_SENTENCE_END = re.compile(r"[.!?…।](?=\s)|[。！？]")


def _cut(line: str, opt: Options) -> int:
    """Where to cut a line over the size: after the last sentence end that fits, else after the last space, so that
    a paragraph kept on one line (as some editors write it) is not cut inside a word; by characters only for a run
    with neither (base64). A cut is not made in the first half of what fits: a piece stays at least half the size."""
    if not opt.max_tokens:
        n = opt.max_chars
    else:
        n = 1
        while n < len(line) and opt.size(line[:n + 1]) <= opt.cap:
            n += 1
    ends = [m.end() for m in _SENTENCE_END.finditer(line, 0, n) if m.end() >= n // 2]
    if ends:
        return ends[-1]
    space = line.rfind(" ", n // 2, n)
    return space + 1 if space > 0 else n


def _split_long(section, unit, start, end, text, opt: Options):
    """Cut a paragraph over the size at line boundaries (a single overlong line by characters); a block under
    BLOCK_CHARS is kept whole when Options.blocks."""
    if opt.size(text) <= opt.cap or (opt.blocks and _is_block(text) and len(text) <= BLOCK_CHARS):
        yield section, unit, start, end, text
        return
    rows = text.split("\n")
    buf, first = [], start
    for i, line in enumerate(rows):
        n = start + i if unit == "line" else start
        while opt.size(line) > opt.cap:
            if buf:
                yield section, unit, first, n - 1 if unit == "line" else n, "\n".join(buf)
                buf = []
            k = _cut(line, opt)
            yield section, unit, n, n, line[:k]
            line = line[k:]
            first = n
        if buf and opt.size("\n".join(buf + [line])) > opt.cap:
            yield section, unit, first, n - 1 if unit == "line" else n, "\n".join(buf)
            buf = []
        if not buf:
            first = n
        buf.append(line)
    if buf:
        yield section, unit, first, end, "\n".join(buf)


def chunk(doc: Doc, keep=None, options: Options | None = None) -> tuple[list[Chunk], int]:
    """(chunks, paragraphs withheld). A paragraph for which `keep(section, text)` is false is left out, and
    no chunk spans across it, so every chunk is still a contiguous, verbatim stretch of the file."""
    opt = options or DEFAULT
    out: list[Chunk] = []
    withheld, gap = 0, False
    for para in _paragraphs(doc, opt):
        # judged whole, before it is cut: a secret in a line longer than the size was cut in two, and neither half
        # matched a secret's pattern (a key's halves were stored, reproduced 2026-10-04)
        if keep is not None and not keep(para[0], para[4]):
            withheld += 1
            gap = True
            continue
        for section, unit, start, end, text in _split_long(*para, opt):
            if unit == "line":                  # a Python stretch holds blank lines; a cut may fall on one: left out
                rows = text.split("\n")
                while rows and not rows[0].strip():
                    rows, start = rows[1:], start + 1
                while rows and not rows[-1].strip():
                    rows, end = rows[:-1], end - 1
                text = "\n".join(rows)
            if not text.strip():
                continue
            last = out[-1] if out else None
            if (last and not gap and last.section == section and last.unit == unit
                    and opt.size(last.text + "\n\n" + text) <= opt.cap):
                last.text += "\n\n" + text
                last.end = end
            else:
                out.append(Chunk(section, unit, start, end, text))
            gap = False
    # Every chunk is a verbatim stretch of its file: its first and last lines are the file's own lines at start, end.
    numbered = lines(doc.blocks[0].text) if doc.blocks and doc.blocks[0].unit == "line" else None
    for i, c in enumerate(out):
        fits = opt.size(c.text) <= opt.cap or (opt.blocks and len(c.text) <= BLOCK_CHARS)
        assert c.text and c.unit in ("line", "page", "cell") and c.start <= c.end and fits, \
            f"chunk {i}: {c.unit} {c.start}-{c.end}, {len(c.text)} characters ({opt.name})"
        # a blank first or last line matches any line below: it once hid a place off by a line (2026-10-05)
        assert c.unit != "line" or (c.text.split("\n")[0].strip() and c.text.split("\n")[-1].strip()), \
            f"chunk {i} starts or ends on a blank line ({c.start}-{c.end})"
        assert i == 0 or out[i - 1].start <= c.start, f"chunk {i} starts before the chunk ahead of it"
        if c.unit == "line" and numbered is not None:
            first, last = c.text.split("\n")[0].strip(), c.text.split("\n")[-1].strip()
            assert first in numbered[c.start - 1] and last in numbered[c.end - 1], \
                f"chunk {i}: its text is not at lines {c.start}-{c.end} of {doc.path}"
    return out, withheld
