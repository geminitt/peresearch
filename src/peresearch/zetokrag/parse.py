"""Read one file into text blocks that remember where they came from (line, page or notebook cell).

Nothing here decides what may be read; `guard.allowed` does that before a path gets here.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 20 * 2**20
TEXT = {".md", ".markdown", ".txt", ".rst", ".org", ".tex", ".bib"}   # data files (csv, json, logs) are not notes
CODE = {".py", ".pyi", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".kt", ".scala", ".c", ".h",
        ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m", ".r", ".jl", ".lua", ".sh", ".bash", ".zsh",
        ".fish", ".sql", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".html", ".css", ".scss", ".cu"}
MARKDOWN = {".md", ".markdown"}


@dataclass
class Block:
    text: str
    unit: str        # "line", "page" or "cell"
    start: int       # 1-based line, page or cell number where `text` begins


@dataclass
class Doc:
    path: Path
    status: str      # ok | needs_ocr | too_large | binary | unsupported | error
    blocks: list[Block] = field(default_factory=list)
    note: str = ""


def kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in MARKDOWN:
        return "markdown"
    if ext in TEXT or ext in CODE:
        return "text"
    return {".pdf": "pdf", ".ipynb": "notebook"}.get(ext, "unsupported")


def read(path: Path) -> Doc:
    path = Path(path)
    k = kind(path)
    if k == "unsupported":
        return Doc(path, "unsupported")
    try:
        if path.stat().st_size > MAX_BYTES:
            return Doc(path, "too_large", note=f"{path.stat().st_size} bytes")
        if k == "pdf":
            return _pdf(path)
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            return Doc(path, "binary")
        text = raw.decode("utf-8", errors="replace")
        if k == "notebook":
            return _notebook(path, text)
        return Doc(path, "ok", [Block(text, "line", 1)])
    except Exception as e:  # a broken file is reported, never fatal
        return Doc(path, "error", note=f"{type(e).__name__}: {e}"[:200])


def _pdf(path: Path) -> Doc:
    import pymupdf

    with pymupdf.open(path) as pdf:
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
        n_images = sum(len(page.get_images()) for page in pdf)
    blocks = [Block(t, "page", n) for n, t in pages if t.strip()]
    if not blocks:
        note = f"{len(pages)} pages, no text layer" + (f", {n_images} images" if n_images else "")
        return Doc(path, "needs_ocr", note=note)
    empty = len(pages) - len(blocks)
    return Doc(path, "ok", blocks, note=f"{empty} pages without text" if empty else "")


def _notebook(path: Path, text: str) -> Doc:
    cells = json.loads(text).get("cells", [])
    blocks = []
    for i, cell in enumerate(cells, start=1):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if src.strip() and cell.get("cell_type") in ("markdown", "code"):
            blocks.append(Block(src, "cell", i))
    return Doc(path, "ok", blocks)
