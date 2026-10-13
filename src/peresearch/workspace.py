"""The declared folders and their index, as both interfaces use them (the command line and the TUI)."""

from pathlib import Path

from peresearch import guard, settings


class Workspace:
    def __init__(self, index=None):
        self._index = index
        self._index_settings = getattr(index, "embedding_settings", None) or self._embedding_settings()

    @staticmethod
    def _embedding_settings():
        return tuple(settings.get("PERESEARCH_EMBED_" + name) for name in ("URL", "MODEL", "KEY"))

    @property
    def index(self):
        current = self._embedding_settings()
        if self._index is None or (getattr(self._index, "from_settings", False) and current != self._index_settings):
            from peresearch.zetokrag.index import Index

            # Keep persisted chunks; a new signature makes old vectors unusable until the next update.
            # An old worker can finish against its own connection/configuration without being closed mid-call.
            previous = self._index
            self._index = Index() if previous is None else Index(previous.home)
            if previous is not None:
                self._index._write_lock = previous._write_lock  # serialize old/new workers across configurations
            self._index_settings = current
        return self._index

    def folders(self) -> list[Path]:
        return guard.roots()

    def add(self, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise ValueError(f"not a folder: {p}")
        if guard.denied(p):
            raise ValueError(f"refused: {p} matches a protected pattern (keys, tokens, credentials)")
        own = guard.home().expanduser().resolve()                         # the setting may name a link
        if p == own or own in p.parents:
            raise ValueError(f"refused: {p} is peresearch's own data")
        roots = guard.roots()
        inside = next((r for r in roots if r == p or r in p.parents), None)
        if inside is not None:
            raise ValueError(f"{p} is already inside the declared folder {inside}")
        guard.set_roots([r for r in roots if p not in r.parents] + [p])      # the parent absorbs its subfolders
        assert p in guard.roots(), f"{p} was added but is not declared"
        return p

    def remove(self, path: str) -> bool:
        p = Path(path).expanduser().resolve()
        before = guard.roots()
        guard.set_roots([r for r in before if r != p])
        assert p not in guard.roots(), f"{p} was removed but is still declared"
        return p in before

    def update(self, stop=None, progress=None):
        return self.index.update(stop=stop, progress=progress)


PROBLEMS = {"needs_ocr": "need OCR (no text layer)", "too_large": "too large", "binary": "binary",
            "error": "could not be read", "large_folder": "folders skipped as datasets",
            "duplicate": "copies of another file (indexed once)",
            "data_file": "data files (logs, tables, word lists, numbers…): not searched by meaning, still readable",
            "tool_made": "folders and files a tool made (environments, caches, packages, version control, Office "
                         "owner files)",
            "hidden": "hidden names (starting with a dot)", "user_excluded": "left out by your patterns",
            "unreadable": "folders or files that could not be read",
            "symlinked_folder": "linked folders not followed", "unsupported": "files of a type not read",
            "data_type": "tables, JSON and logs: read by the tools, not searched by meaning",
            "special_file": "pipes, sockets or devices: never opened"}


def summary(r, limit: int = 20) -> str:
    """What an index update did, for either interface."""
    lines = [f"{r.added} added · {r.changed} changed · {r.unchanged} unchanged · {r.removed} removed · "
             f"{r.chunks} chunks · {r.denied} protected paths skipped"]
    made, waiting, why = getattr(r, "embedded", 0), getattr(r, "unembedded", 0), getattr(r, "embed_error", "")
    if made or waiting:
        lines.append(f"{made:,} chunks embedded" + (f" · {waiting:,} still without a vector (searched by their words "
                                                     f"until the next update)" + (f": {why}" if why else "")
                                                     if waiting else "")
                     + (f" · {getattr(r, 'halved', 0):,} by their head only: the embedding model takes less than a "
                        "chunk (all of each is still searched by its words)" if getattr(r, "halved", 0) else ""))
    for status, paths in sorted(r.problems.items()):
        lines.append(f"{len(paths)} {PROBLEMS.get(status, status)}:")
        lines += [f"  {p}" for p in paths[:limit]]
    if getattr(r, "largest", None):
        lines.append("files with the most chunks (one that is not yours, such as a copied library, can be excluded "
                     "in .peresearchignore):")
        lines += [f"  {p} ({n} chunks)" for p, n in r.largest]
    if r.withheld:
        lines.append(f"{sum(r.withheld.values())} chunks withheld because they look like credentials:")
        lines += [f"  {p} ({n})" for p, n in list(r.withheld.items())[:limit]]
    return "\n".join(lines)


def quiet(r) -> bool:
    """Nothing to tell after an update: no file added, changed or removed, and nothing withheld."""
    return not (r.added or r.changed or r.removed or r.withheld)


def brief(r) -> str:
    """What an index update did, in one line for the interface (no path: the full report is the CLI's, and what was
    left out is /status's)."""
    head = "Index up to date" if not (r.added or r.changed or r.removed) else \
        f"Index updated: {r.added:,} added, {r.changed:,} changed, {r.removed:,} removed"
    line = f"{head} · {count(r.chunks, 'chunk')} from {count(r.added + r.changed + r.unchanged, 'file')}"
    if r.withheld:
        n = sum(r.withheld.values())
        line += f" · {count(n, 'chunk')} withheld: " + ("it looks like a credential" if n == 1 else
                                                         "they look like credentials")
    waiting = getattr(r, "unembedded", 0)
    if waiting:
        line += f" · {count(waiting, 'chunk')} wait for their vectors" + (
            ": the embeddings endpoint did not answer" if getattr(r, "embed_error", "") else "")
    if r.problems:
        line += " · what was left out: /status"
    return line


def count(n: int, word: str) -> str:
    """"1 file", "2 files", "1,200 files"."""
    return f"{n:,} {word}{'s' * (n != 1)}"


def complete(partial: str, folders_only: bool = True) -> list[str]:
    """Names that complete a typed path, as a shell's Tab would: folders (ending in "/"), and files too unless
    `folders_only`; hidden ones only when asked for."""
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
                out.append(shown + ("/" if child.is_dir() else ""))
    return out
