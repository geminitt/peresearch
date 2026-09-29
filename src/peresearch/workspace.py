"""The declared folders and their index, as both interfaces use them (the command line and the TUI)."""

from pathlib import Path

from peresearch import guard


class Workspace:
    def __init__(self, index=None):
        self._index = index

    @property
    def index(self):
        if self._index is None:
            from peresearch.zetokrag.index import Index

            self._index = Index()
        return self._index

    def folders(self) -> list[Path]:
        return guard.roots()

    def add(self, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise ValueError(f"not a folder: {p}")
        if guard.denied(p):
            raise ValueError(f"refused: {p} matches a protected pattern (keys, tokens, credentials)")
        if p == guard.home() or guard.home() in p.parents:
            raise ValueError(f"refused: {p} is peresearch's own data")
        guard.set_roots(guard.roots() + [p])
        return p

    def remove(self, path: str) -> bool:
        p = Path(path).expanduser().resolve()
        before = guard.roots()
        guard.set_roots([r for r in before if r != p])
        return p in before

    def update(self):
        return self.index.update()


PROBLEMS = {"needs_ocr": "need OCR (no text layer)", "too_large": "too large", "binary": "binary",
            "error": "could not be read", "large_folder": "folders skipped as datasets"}


def summary(r, limit: int = 20) -> str:
    """What an index update did, for either interface."""
    lines = [f"{r.added} added · {r.changed} changed · {r.unchanged} unchanged · {r.removed} removed · "
             f"{r.chunks} chunks · {r.denied} protected paths skipped"]
    for status, paths in sorted(r.problems.items()):
        lines.append(f"{len(paths)} {PROBLEMS.get(status, status)}:")
        lines += [f"  {p}" for p in paths[:limit]]
    if r.withheld:
        lines.append(f"{sum(r.withheld.values())} chunks withheld because they look like credentials:")
        lines += [f"  {p} ({n})" for p, n in list(r.withheld.items())[:limit]]
    return "\n".join(lines)


def complete(partial: str, folders_only: bool = True) -> list[str]:
    """Directory names that complete a typed path, as a shell's Tab would (hidden ones only when asked for)."""
    text = partial or ""
    expanded = Path(text).expanduser()
    base, stem = (expanded, "") if text.endswith("/") or text == "~" else (expanded.parent, expanded.name)
    if not base.is_dir():
        return []
    out = []
    for child in sorted(base.iterdir()):
        if child.name.startswith(stem) and (not child.name.startswith(".") or stem.startswith(".")):
            if child.is_dir() or not folders_only:
                shown = str(child) if not text.startswith("~") else "~/" + str(child.relative_to(Path.home()))
                out.append(shown + "/")
    return out
