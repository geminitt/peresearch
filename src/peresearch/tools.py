"""What the agent can do: look through and read the declared folders, search the web and read pages it found.

Every tool is read-only. File tools see only the declared folders (`guard.allowed`: symlinks resolved, protected
names refused) and redact anything credential-like before the model sees it. `fetch` only opens URLs that a
web search returned in this session, so the model cannot send data out by encoding it into a URL of its own.
`web_search` is refused until the model has recorded with `gaps` what the user's files already cover and what is
missing, so the web is searched for what the user does not have.
Each result carries source ids (N1, N2… for files, W1, W2… for the web) that answers must cite.
"""

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from peresearch import guard
from peresearch.zetokrag import parse

MAX_OUT = 6000          # characters of one tool result shown to the model
MAX_READ_LINES = 200
TREE_MAX_DEPTH = 4
TREE_PER_FOLDER = 40
TREE_HIDDEN = {"__pycache__", "node_modules", "site-packages"}     # tooling, never the user's own material


@dataclass
class Source:
    id: str
    kind: str            # "file" or "web"
    where: str           # path:L10-20, path:p.3, path:cell 4, or a URL
    title: str
    text: str


@dataclass
class Sources:
    """Every source the agent has seen in one question; ids stay stable for citations."""
    items: dict[str, Source] = field(default_factory=dict)
    urls: dict[str, str] = field(default_factory=dict)       # url -> web source id

    def add(self, kind: str, where: str, title: str, text: str) -> Source:
        for s in self.items.values():
            if s.where == where:
                if len(text) > len(s.text):
                    s.text = text
                return s
        prefix = "N" if kind == "file" else "W"
        sid = f"{prefix}{sum(1 for s in self.items.values() if s.kind == kind) + 1}"
        self.items[sid] = s = Source(sid, kind, where, title, text)
        if kind == "web":
            self.urls[where] = sid
        return s


def _clip(text: str, n: int = MAX_OUT) -> str:
    return text if len(text) <= n else text[:n] + f"\n… [{len(text) - n} more characters]"


def _match(rel: str, pattern: str) -> bool:
    """Glob match on the path relative to its root; a leading **/ also matches no folder at all."""
    return (fnmatch.fnmatch(rel, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:]))
            or ("/" not in pattern and fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)))


def _files(roots: list[Path], pattern: str = "**/*"):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in guard.SKIP_DIRS and not guard.denied(Path(dirpath) / d)
                                 and not os.path.islink(Path(dirpath) / d))
            for f in sorted(filenames):
                p = Path(dirpath) / f
                if _match(p.relative_to(root).as_posix(), pattern) and guard.allowed(p, roots):
                    yield p


def outermost(roots: list[Path]) -> list[Path]:
    """The declared folders without those inside another one, which would list their files twice."""
    real = [Path(os.path.realpath(r)) for r in roots]
    return [r for r, rr in zip(roots, real) if not any(o != rr and o in rr.parents for o in real)
            and rr not in real[:real.index(rr)]]


def _resolve(path: str, roots: list[Path]) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        hits = [r / p for r in roots if (r / p).exists()]
        if not hits:
            raise guard.Refused(f"{path}: not found in the declared folders")
        p = hits[0]
    if not guard.allowed(p, roots):
        raise guard.Refused(f"{path}: outside the declared folders or protected")
    return p


class Toolbox:
    """The tools, their JSON schemas for the model, and the sources they produce."""

    def __init__(self, searcher=None, web=None, roots: list[Path] | None = None):
        self.searcher, self.web = searcher, web
        self.roots = outermost([Path(r) for r in (guard.roots() if roots is None else roots)])
        self.sources = Sources()
        self.web_calls = 0
        self.recorded_gaps = None        # set by gaps(); the agent clears it for every question

    # --- the tools ---

    def tree(self, path: str | None = None, depth: int = 2) -> str:
        """The folders and files under the declared folders (or one folder inside them), `depth` levels down.
        Hidden entries, tooling folders and protected files are left out; folders never read (data, runs…) are
        named but not opened. Each folder that is opened says how many folders and files it holds."""
        starts = [_resolve(path, self.roots)] if path else self.roots
        if not starts:
            return "no folders are declared"
        depth = max(1, min(int(depth), TREE_MAX_DEPTH))
        lines = []
        for start in starts:
            if not start.is_dir():
                return f"{start}: not a folder"
            entries = self._visible(start)
            lines.append(f"{start}/ {self._count(entries)}")
            self._tree(entries, 1, depth, lines)
        text = "\n".join(lines)
        s = self.sources.add("file", f"{', '.join(map(str, starts))} (listing)", "folder listing", text)
        return f"[{s.id}] listing of {', '.join(map(str, starts))}\n{_clip(text)}"

    def _visible(self, folder: Path) -> list:
        """The entries of a folder that tree shows: folders first, then the files the guard allows."""
        try:
            entries = sorted(os.scandir(folder), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError:
            return []
        return [e for e in entries
                if not (e.name.startswith(".") or e.name in TREE_HIDDEN or e.is_symlink() or guard.denied(Path(e.path)))
                and (e.is_dir(follow_symlinks=False) or guard.allowed(Path(e.path), self.roots))]

    @staticmethod
    def _count(entries: list) -> str:
        folders = sum(e.is_dir(follow_symlinks=False) for e in entries)
        files = len(entries) - folders
        return f"({folders} folder{'' if folders == 1 else 's'}, {files} file{'' if files == 1 else 's'})"

    def _tree(self, entries: list, level: int, depth: int, lines: list[str]) -> None:
        indent = "  " * level
        for e in entries[:TREE_PER_FOLDER]:
            if not e.is_dir(follow_symlinks=False):
                lines.append(f"{indent}{e.name}")
            elif e.name in guard.SKIP_DIRS:
                lines.append(f"{indent}{e.name}/ (not read)")
            elif level >= depth:
                lines.append(f"{indent}{e.name}/ …")
            else:
                inner = self._visible(Path(e.path))
                lines.append(f"{indent}{e.name}/ {self._count(inner)}")
                self._tree(inner, level + 1, depth, lines)
        if len(entries) > TREE_PER_FOLDER:
            lines.append(f"{indent}… {len(entries) - TREE_PER_FOLDER} more")

    def gaps(self, have: list[str] | str = (), missing: list[str] | str = ()) -> str:
        """What the user's files already cover and what is missing or not enough: the web is searched for the
        second part."""
        have = [have] if isinstance(have, str) else list(have or [])
        missing = [missing] if isinstance(missing, str) else list(missing or [])
        self.recorded_gaps = {"have": have, "missing": missing}
        return (f"recorded: {len(have)} point(s) the user's files cover, {len(missing)} missing or not enough; "
                "web_search may now look for what is missing")

    def search_notes(self, query: str, k: int = 5) -> str:
        hits, verdict, stale = self.searcher.find(query, k=k)
        lines = [f"verdict: {verdict} (does the user's own material answer this?)"]
        for h in hits:
            text = h.text + ("".join("\n" + n for n in h.neighbours) if h.neighbours else "")
            s = self.sources.add("file", h.where, h.section, guard.redact(text)[0])
            lines.append(f"[{s.id}] {h.where} (relevance {h.score:.2f})" + (f" § {h.section}" if h.section else "")
                         + "\n" + _clip(s.text, 1500))
        if stale:
            lines.append(f"({stale} excerpts dropped: their files changed since indexing)")
        return "\n\n".join(lines)

    def grep(self, pattern: str, glob: str = "**/*", max_results: int = 30) -> str:
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            return f"invalid regular expression: {e}"
        out = []
        for p in _files(self.roots, glob):
            doc = parse.read(p)
            if doc.status != "ok":
                continue
            for b in doc.blocks:
                for i, line in enumerate(b.text.splitlines()):
                    if rx.search(line):
                        n = b.start + i if b.unit == "line" else b.start
                        where = f"{p}:L{n}" if b.unit == "line" else f"{p}:{'p.' if b.unit == 'page' else 'cell '}{n}"
                        s = self.sources.add("file", where, p.name, guard.redact(line.strip())[0])
                        out.append(f"[{s.id}] {where}: {s.text[:300]}")
                        if len(out) >= max_results:
                            return "\n".join(out) + f"\n(stopped at {max_results} matches)"
        return "\n".join(out) or "no match"

    def glob(self, pattern: str, max_results: int = 100) -> str:
        found = []
        roots, pattern = self.roots, pattern.strip()
        if pattern.startswith(("/", "~")):                  # a path as the user wrote it: match inside that root
            full = Path(pattern).expanduser()
            if full.is_dir():
                full = full / "*"                           # a folder: the files directly in it
            base = next((b for b in full.parents if b.is_dir() and any(b == r or r in b.parents for r in self.roots)),
                        None)
            if base is None:
                return "no file matches"
            roots, pattern = [base], full.relative_to(base).as_posix()
        for p in _files(roots, pattern):
            found.append(str(p))
            if len(found) >= max_results:
                break
        return "\n".join(found) or "no file matches"

    def read(self, path: str, start: int = 1, end: int | None = None) -> str:
        p = _resolve(path, self.roots)
        doc = parse.read(p)
        if doc.status != "ok":
            return f"{p}: cannot be read ({doc.status}{': ' + doc.note if doc.note else ''})"
        if doc.blocks[0].unit == "line":
            lines = doc.blocks[0].text.splitlines()
            start = max(1, start)
            end = min(len(lines), end or start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
            where, text = f"{p}:L{start}-{end}", "\n".join(lines[start - 1:end])
        else:                                 # pages or cells: start/end number those units
            unit = doc.blocks[0].unit
            chosen = [b for b in doc.blocks if start <= b.start <= (end or start + 4)]
            label = "p." if unit == "page" else "cell "
            where = f"{p}:{label}{chosen[0].start}-{chosen[-1].start}" if chosen else f"{p}:{label}{start}"
            text = "\n\n".join(b.text for b in chosen)
        s = self.sources.add("file", where, p.name, guard.redact(text)[0])
        return f"[{s.id}] {where}\n{_clip(s.text)}"

    def web_search(self, query: str, n: int = 5) -> str:
        if self.recorded_gaps is None:
            raise guard.Refused("first record with gaps(have, missing) what the user's files already cover and what "
                                "is missing; then search the web for what is missing")
        self.web_calls += 1
        provider, results = self.web.search(query, n)
        lines = [f"({provider})"]
        for r in results:
            s = self.sources.add("web", r.url, r.title, r.snippet)
            lines.append(f"[{s.id}] {r.title}\n{r.url}\n{_clip(r.snippet, 600)}")
        return "\n\n".join(lines) if results else "no results"

    def fetch(self, url: str) -> str:
        if url not in self.sources.urls:
            raise guard.Refused("fetch only opens URLs returned by web_search in this question")
        text = self.web.fetch(url)
        s = self.sources.items[self.sources.urls[url]]
        if text:
            s.text = text
        return f"[{s.id}] {url}\n{_clip(text or 'the page could not be read')}"

    # --- for the model ---

    SPECS = {
        "tree": ("Show the folders and files the user has: every declared folder, or one folder inside them, a few "
                 "levels deep. Use it to see what a project or folder contains.",
                 {"path": ("string", "a folder, e.g. ~/projects/x; default: every declared folder"),
                  "depth": ("integer", "levels to show, default 2, at most 4")}, []),
        "search_notes": ("Search the user's own files (notes, projects, course material) by meaning and keywords. "
                         "Returns excerpts with source ids and a verdict: enough / partial / none.",
                         {"query": ("string", "what to look for"), "k": ("integer", "number of excerpts, default 5")},
                         ["query"]),
        "grep": ("Find lines matching a regular expression (case-insensitive) in the user's files: exact names, "
                 "numbers, error messages.",
                 {"pattern": ("string", "regular expression"),
                  "glob": ("string", "file pattern such as **/*.md, default all files")}, ["pattern"]),
        "glob": ("List the user's files whose path matches a pattern, e.g. **/*.ipynb, notes/** or "
                 "~/projects/x/*.py. To see folders, use tree.",
                 {"pattern": ("string", "glob pattern")}, ["pattern"]),
        "read": ("Read part of one of the user's files: lines for text files, pages for PDFs, cells for notebooks.",
                 {"path": ("string", "path from a previous result"), "start": ("integer", "first line/page/cell"),
                  "end": ("integer", "last line/page/cell")}, ["path"]),
        "gaps": ("Record, before searching the web, what the user's own files already cover and what is missing "
                 "or not enough. web_search is refused until this is called.",
                 {"have": ("array", "points the user's files cover (with source ids), may be empty"),
                  "missing": ("array", "what is missing or not enough, to look for on the web")}, ["have", "missing"]),
        "web_search": ("Search the web for what gaps recorded as missing, or to check that something is current.",
                       {"query": ("string", "search query, no personal or secret data"),
                        "n": ("integer", "number of results, default 5")}, ["query"]),
        "fetch": ("Read the main text of a web page returned by web_search.",
                  {"url": ("string", "a URL from web_search results")}, ["url"]),
    }

    def schemas(self, web: bool = True) -> list[dict]:
        out = []
        for name, (desc, props, required) in self.SPECS.items():
            if name in ("gaps", "web_search", "fetch") and not (web and self.web and self.web.providers):
                continue
            out.append({"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "required": required, "properties": {
                    k: {"type": t, "description": d, **({"items": {"type": "string"}} if t == "array" else {})}
                    for k, (t, d) in props.items()}}}})
        return out

    def call(self, name: str, args: dict) -> str:
        """Run one tool; errors come back as text for the model, never as a crash."""
        if name not in self.SPECS:
            return f"unknown tool {name!r}"
        try:
            return getattr(self, name)(**{k: v for k, v in args.items() if k in self.SPECS[name][1]})
        except guard.Refused as e:
            return f"refused: {e}"
        except TypeError as e:
            return f"bad arguments for {name}: {e}"
        except Exception as e:
            return f"{name} failed: {type(e).__name__}: {str(e)[:200]}"
