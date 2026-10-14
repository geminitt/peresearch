"""What the agent can do: look through and read the declared folders, search the web and read pages it found.

Every tool is read-only. File tools read the declared folders freely (`guard.allowed`: symlinks resolved, protected
names refused); grep, glob and tree walk by the rules of files.py, as the index does (a folder they skip is searched
when named in the pattern, unless the user excluded it). Elsewhere in the home folder they read only what the user
allows when asked (`ask`: once, or a folder for the session); hidden folders, protected names, peresearch's own data
and anything outside the home folder are refused without asking. Anything credential-like is redacted before the
model sees it. `fetch` only opens URLs that a web search returned in this session, so the model cannot send data out
by encoding it into a URL of its own. `web_search` is refused until the model has recorded with `gaps` what the
user's files already cover and what is missing, so the web is searched for what the user does not have.
Each result carries source ids (N1, N2… for files, W1, W2… for the web) that answers must cite.
"""

import base64
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from peresearch import files, guard
from peresearch.evidence import Source, Sources
from peresearch.messages import Tool
from peresearch.web import site_kind
from peresearch.zetokrag import extracted, parse
from peresearch.zetokrag.index import sha256

# Every tool returns its whole output: the agent shows the model what fits its context and keeps the rest to read
# on (agent._spill, recall(ref, from=n)); no size here is fixed (owner, 2026-10-07).
WEB_TOOLS = ("gaps", "web_search", "search_papers", "search_wikipedia", "fetch")   # what /local turns off
TREE_MAX_DEPTH = 4      # bounds the walk's time, as GREP_SECONDS does grep's; a deeper look is a tree of a subfolder
# ripgrep from the environment peresearch runs in (pixi ships it), else from PATH; None: grep reads in Python
RG = shutil.which("rg", path=os.pathsep.join([str(Path(sys.prefix) / "bin"), os.environ.get("PATH", "")]))
RG_BATCH = 400           # paths per ripgrep call, so the command line stays short on every system
GREP_SECONDS = 60        # a search stops here and says what it did not reach
# grep locates, read reads: a matching line up to LINE_WHOLE characters is shown whole, a longer one (data on one line:
# up to ~118,000 characters on the owner's folders, 2026-10-07) as LINE_WINDOW characters around its first match. The
# one fixed size in the tools, on purpose: Claude Code omits such a line, opencode cuts it to 2,000 and pi-coding-agent
# to 500 characters, both from its head, which may not hold the match; here the whole line stays one read away.
LINE_WHOLE, LINE_WINDOW = 2_000, 1_000
REDACT_MARGIN = 2_000    # redacted beyond the window first: a secret across its edge is never shown in part
NOT_INDEXED = {          # index.coverage() reasons, singular and plural, for the model
    "needs_ocr": ("PDF without a text layer", "PDFs without a text layer"),
    "tool_made": ("folder or file made by tools or peresearch's own",
                  "folders or files made by tools or peresearch's own"),
    "data_type": ("table, JSON or log file", "table, JSON or log files"),
    "hidden": ("hidden file or folder", "hidden files or folders"),
    "large_folder": ("large folder", "large folders"),
    "user_excluded": ("path the user excluded", "paths the user excluded"),
    "symlinked_folder": ("linked folder", "linked folders"), "unreadable": ("unreadable folder or file", "unreadable folders or files"),
    "unsupported": ("file of a type not read", "files of a type not read"),
    "special_file": ("pipe, socket or device", "pipes, sockets or devices"),
    "too_large": ("file too large", "files too large"), "binary": ("binary file", "binary files"),
    "error": ("file that could not be read", "files that could not be read"),
    "data": ("list of URLs or numbers", "lists of URLs or numbers"),
}
LEFT_OUT = {"tool_made": "made by tools or peresearch's own", "hidden": "hidden", "large_folder": "large folders",
            "user_excluded": "excluded by the user", "symlinked_folder": "linked folders", "unreadable": "unreadable",
            "special_file": "pipes, sockets or devices"}



def _one_line(text: str) -> str:
    """`text` whole, on one line."""
    return " ".join(text.split())


def _match(rel: str, pattern: str) -> bool:
    """Glob match on the path relative to its root; a leading **/ also matches no folder at all."""
    return (fnmatch.fnmatch(rel, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:]))
            or ("/" not in pattern and fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)))


def _hidden_in_home(path: Path) -> bool:
    """The real place of `path` lies in a hidden folder (or is a hidden file) of the home folder."""
    real, user = Path(os.path.realpath(path)), Path(os.path.realpath(Path.home()))
    return (real == user or user in real.parents) and any(part.startswith(".") for part in real.relative_to(user).parts)


def _named(folder: Path, pattern: str, hidden: bool) -> tuple[str, ...]:
    """Include patterns for what a glob names literally below `folder` (its leading parts, up to the first wildcard):
    a hidden or tool-made folder the model asks for by name is searched. Hidden names only where hidden entries are
    allowed (the declared folders)."""
    parts = [x for x in pattern.split("/") if x not in ("", ".")]
    out = []
    for i, part in enumerate(parts):
        if part == ".." or any(c in part for c in "*?[") or (part.startswith(".") and not hidden):
            break
        literal = re.sub(r"([\[\]*?])", r"[\1]", str(folder.joinpath(*parts[:i + 1])))   # the path, not a glob
        out.append(literal + ("" if i == len(parts) - 1 else "/"))
    return tuple(out)


def _walk(scopes, pattern: str = "**/*") -> tuple[list[Path], Counter]:
    """The files under each (folder, hidden allowed) scope that files.walk lets through and `pattern` names, and what
    was left out, counted by reason. Where hidden entries are not allowed (a folder the user allowed outside the
    declared ones), a link whose target is hidden is skipped too. A file under two scopes is listed once."""
    config, found, left = guard.config(), {}, Counter()
    for root, hidden in scopes:
        for p in files.walk([root], config, lambda s: left.update([s.reason]), lambda: None,
                            _named(root, pattern, hidden)):
            if (hidden or not _hidden_in_home(p)) and _match(p.relative_to(root).as_posix(), pattern):
                found.setdefault(p, None)                     # a declared folder inside an allowed one: once
    return list(found), left


def _left_note(left: Counter, verb: str) -> str:
    parts = [f"{n} {LEFT_OUT.get(r, r)}" for r, n in sorted(left.items()) if n]
    return (f"({verb}: {', '.join(parts)}; a folder named in the pattern is {verb.split()[1]} unless the user "
            "excluded it)") if parts else ""


def _rg(paths: list[str], pattern: str, deadline: float) -> tuple[dict[str, list], bool, bool]:
    """ripgrep over `paths` (no walking of its own): ({path: [(line, text)]}, pattern accepted, finished in time).
    A file ripgrep finds a NUL byte in is dropped, as parse.read calls it binary."""
    out: dict[str, list] = {}
    for i in range(0, len(paths), RG_BATCH):
        cmd = [RG, "--json", "--no-config", "--no-ignore", "--hidden", "--ignore-case", "--crlf", "--engine", "auto",
               "--regexp", pattern, "--", *paths[i:i + RG_BATCH]]
        left = deadline - time.monotonic()
        if left <= 0:
            return out, True, False
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=left)
        except subprocess.TimeoutExpired:
            return out, True, False
        if r.returncode == 2 and b"regex" in r.stderr:
            return out, False, True
        binary = set()
        for line in r.stdout.splitlines():
            event = json.loads(line)
            data = event.get("data", {})
            path = data.get("path", {})
            path = path["text"] if "text" in path else os.fsdecode(base64.b64decode(path.get("bytes", "")))
            if event["type"] == "match":
                lines = data["lines"]
                text = lines["text"] if "text" in lines else base64.b64decode(lines["bytes"]).decode("utf-8", "replace")
                out.setdefault(path, []).append((data["line_number"], text.rstrip("\r\n")))
            elif event["type"] == "end" and data.get("binary_offset") is not None:
                binary.add(path)
        for path in binary:
            out.pop(path, None)
    return out, True, True


def _around(line: str, rx: re.Pattern, where: str) -> tuple[str, str, str, str]:
    """(what grep shows of a matching line, redacted; "…" before it; "…" after it; a note): the line whole up to
    LINE_WHOLE characters; else LINE_WINDOW of them around its first match, "…" where it goes on, and its length with
    where read shows it whole. Only the window and its margins are redacted (a whole line of ~118,000 characters took most of a
    60 s search, 2026-10-07)."""
    line = line.strip()
    if len(line) <= LINE_WHOLE:
        return guard.redact(line)[0], "", "", ""
    hit = rx.search(line)
    at = hit.start() if hit else 0
    a, b = max(0, at - LINE_WINDOW // 2 - REDACT_MARGIN), min(len(line), at + LINE_WINDOW // 2 + REDACT_MARGIN)
    part = guard.redact(line[a:b])[0]
    near = [m.start() for m in rx.finditer(part)] or [min(at - a, len(part))]   # redaction may move it, or take it
    pos = min(near, key=lambda n: abs(n - (at - a)))
    s = max(0, pos - LINE_WINDOW // 2)
    e = min(len(part), s + LINE_WINDOW)
    s = max(0, e - LINE_WINDOW)
    note = f" [a line of {len(line):,} characters, shown around the match: read {where} for all of it]"
    return part[s:e], "…" if a + s > 0 else "", "…" if e < len(part) or b < len(line) else "", note


def _place(path: Path, unit: str, n: int) -> str:
    return f"{path}:" + {"line": "L", "page": "p.", "cell": "cell "}[unit] + str(n)


def _whole(value) -> int | None:
    """A whole number as a model may write it — 3, 3.0, "3", " 3 ", "3.0" — else None (a bool is not one)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value) if "." in value else int(value)     # both take surrounding spaces
        except ValueError:
            return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if isinstance(value, int) else None


class Toolbox:
    """The tools, their JSON schemas for the model, and the sources they produce."""

    def __init__(self, searcher=None, web=None, roots: list[Path] | None = None, ask=None, papers=None, wiki=None):
        """`ask(tool, path) -> "once" | "session" | "no"` asks the user before reading outside the declared
        folders; without it, nothing outside them is read. `papers`: the scholarly search (papers.Papers); `wiki`:
        Wikipedia (wiki.Wikipedia)."""
        self.searcher, self.web, self.ask, self.papers, self.wiki = searcher, web, ask, papers, wiki
        self.granted: list[Path] = []    # folders the user allowed for this session
        self._once: list[Path] = []     # current-question continuation of an explicitly allowed read
        self.last_scope: set[str] = set()  # exact files/listing roots accessed by the last tool call
        self._roots = None if roots is None else guard.outermost([Path(r) for r in roots])
        if roots is None and hasattr(searcher, "allowed"):
            # The persisted policy may change while this toolbox and its index remain alive.
            searcher.allowed = lambda path: guard.allowed(Path(path), self.roots)
        self.sources = Sources()
        self.web_calls = 0
        self.recorded_gaps = None        # set by gaps(); the agent clears it for every question
        self.local_only = False          # /local: the user turned the web off for this session

    # --- where the tools may read ---

    @property
    def roots(self) -> list[Path]:
        """Current declared folders, or the explicit scope supplied by an embedding caller."""
        return guard.outermost(guard.roots()) if self._roots is None else self._roots

    @roots.setter
    def roots(self, roots: list[Path]) -> None:
        self._roots = guard.outermost([Path(r) for r in roots])

    def _readable(self) -> list[Path]:
        return self.roots + self.granted

    def retained_allowed(self, path: str) -> bool:
        """Replaying evidence requires current access, including this question's once grants."""
        return guard.allowed(Path(path), self._readable() + self._once)

    def _outside(self, tool: str, p: Path) -> Path:
        """`p` lies outside the declared folders: refused outright, or read if the user allows it."""
        real, user = Path(os.path.realpath(p)), Path(os.path.realpath(Path.home()))
        own = Path(os.path.realpath(guard.home()))
        if not (real == user or user in real.parents):
            raise guard.Refused(f"{p}: outside your home folder")
        if (any(part.startswith(".") for part in real.relative_to(user).parts) or guard.denied(p) or guard.denied(real)
                or real == own or own in real.parents):
            raise guard.Refused(f"{p}: hidden or protected")
        if not real.exists():
            raise guard.Refused(f"{p}: not found")
        if any(g == real or g in real.parents for g in self.granted):     # in a folder allowed for this session
            return real
        if self.ask is None:
            raise guard.Refused(f"{p}: outside the declared folders (tree without a path lists them); the user can "
                                "allow it when asked in peresearch chat, or declare it with /add")
        answer = self.ask(tool, real)
        if answer == "session":
            self.granted.append(real if real.is_dir() else real.parent)
        elif answer == "once":
            self._once.append(real)
        elif answer != "once":
            raise guard.Refused(f"the user did not allow reading {p}; do not ask for it again")
        return real

    def _path(self, path: str, tool: str) -> tuple[Path, list[Path]]:
        """A path the model named, and the folders the call may read below it."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            hits = [r / p for r in self._readable() if (r / p).exists()]
            if not hits:
                raise guard.Refused(f"{path}: not found in the declared folders (tree without a path lists them)")
            p = hits[0]
        if guard.allowed(p, self.roots):
            self.last_scope.add(str(p.resolve()))
            return p, self._readable()
        real = self._outside(tool, p)                   # hidden, protected or own data refused; else allowed or asked
        self.last_scope.add(str(real))
        assert not (guard.denied(p) or guard.denied(real)), f"{p}: protected, yet the reading rule let it through"
        return real, self._readable() + [real if real.is_dir() else real.parent]

    def _scope(self, pattern: str, tool: str) -> tuple[list[tuple[Path, bool]], str]:
        """Where a glob pattern looks, as (folder, hidden entries allowed) pairs: the declared folders (hidden entries
        allowed: they were declared whole) and the folders allowed this session (not) for a relative pattern, or the
        folder a path names (~/x/*.py, /abs/dir)."""
        pattern = pattern.strip()
        if not pattern.startswith(("/", "~")):
            return [(r, True) for r in self.roots] + [(g, False) for g in self.granted], pattern
        full = Path(pattern).expanduser()
        if full.is_dir():
            full = full / "*"                               # a folder: the files directly in it
        base = next((b for b in full.parents if b.is_dir()), None)
        if base is None:
            return [], ""
        rel = full.relative_to(base).as_posix()
        if any(base == r or r in base.parents for r in self.roots):
            return [(base, True)], rel
        return [(self._outside(tool, base), False)], rel

    # --- the tools ---

    def tree(self, path: str | None = None, depth: int = 2) -> str:
        """The folders and files under the declared folders (or one folder inside them), `depth` levels down.
        Hidden entries and protected files are left out; a folder grep and glob do not search (made by a tool, too
        many files side by side, excluded by the user) is named with the reason, not opened. Each folder that is
        opened says how many folders and files it holds."""
        if path:
            start, scope = self._path(path, "tree")
            starts = [start]
        else:
            starts, scope = self._readable(), self._readable()
        self.last_scope.update(str(p.resolve()) for p in starts)
        if not starts:
            return "no folders are declared"
        depth = max(1, min(int(depth), TREE_MAX_DEPTH))
        lines = []
        for start in starts:
            if not start.is_dir():
                return f"{start}: not a folder"
            entries = self._visible(start, scope)
            lines.append(f"{start}/ {self._count(entries)}")
            self._tree(entries, 1, depth, lines, scope)
        text = "\n".join(lines)
        s = self.sources.add("file", f"{', '.join(map(str, starts))} (listing)", "folder listing", text)
        return f"[{s.id}] listing of {', '.join(map(str, starts))}\n{text}"

    def _visible(self, folder: Path, scope: list[Path]) -> list:
        """The entries of a folder that tree shows: folders first, then the files the guard allows."""
        try:
            entries = sorted(os.scandir(folder), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError:
            return []
        return [e for e in entries
                if not (e.name.startswith(".") or e.is_symlink() or guard.denied(Path(e.path)))
                and (e.is_dir(follow_symlinks=False) or guard.allowed(Path(e.path), scope))]

    @staticmethod
    def _left_out(folder: Path, scope: list[Path]):
        """Why grep and glob do not search `folder` (files.check), or None. A folder of too many files side by side is
        told by one of its files."""
        why = files.check(folder, scope, guard.config())
        if why is not None or not folder.is_dir():
            return why
        try:
            first = next((Path(e.path) for e in os.scandir(folder) if not e.is_dir()), None)
        except OSError:
            return None
        why = files.check(first, scope, guard.config()) if first is not None else None
        return why if why is not None and why.reason == "large_folder" else None

    @staticmethod
    def _count(entries: list) -> str:
        folders = sum(e.is_dir(follow_symlinks=False) for e in entries)
        files = len(entries) - folders
        return f"({folders} folder{'' if folders == 1 else 's'}, {files} file{'' if files == 1 else 's'})"

    def _tree(self, entries: list, level: int, depth: int, lines: list[str], scope: list[Path]) -> None:
        indent = "  " * level
        for e in entries:
            if not e.is_dir(follow_symlinks=False):
                lines.append(f"{indent}{e.name}")
            elif (why := self._left_out(Path(e.path), scope)) is not None:
                lines.append(f"{indent}{e.name}/ (not searched: {why.note or why.reason})")
            elif level >= depth:
                lines.append(f"{indent}{e.name}/ …")
            else:
                inner = self._visible(Path(e.path), scope)
                lines.append(f"{indent}{e.name}/ {self._count(inner)}")
                self._tree(inner, level + 1, depth, lines, scope)

    def gaps(self, have: list[str] | str = (), missing: list[str] | str = ()) -> str:
        """What the user's files already cover and what is missing or not enough: the web is searched for the
        second part."""
        have = [have] if isinstance(have, str) else list(have or [])
        missing = [missing] if isinstance(missing, str) else list(missing or [])
        self.recorded_gaps = {"have": have, "missing": missing}
        return (f"recorded: {len(have)} point(s) the user's files cover, {len(missing)} missing or not enough; "
                "web_search, search_papers and search_wikipedia may now look for what is missing")

    def _ranked(self, marked: str, counted: str, scored: list) -> tuple[bool, list[str]]:
        """(whether a reranker scores what is shown, the notes saying how the search ranked): with no rerank endpoint
        nothing carries a score; a search by words alone, or a rerank that failed, is said with why."""
        s = self.searcher
        rerank = getattr(s, "reranker", True) is not None
        notes = []
        if rerank and any(x is None for x in scored):
            failed = getattr(s, "unranked", "")
            notes.append(f"(not reranked: {failed})" if failed else
                         f"({marked} marked 'not reranked' keeps the search's order: this machine reranks only the "
                         f"first {s.depth} {counted}, PERESEARCH_RERANK_DEPTH)")
        if getattr(s, "words_only", ""):
            notes.append(f"(ranked by the passages' words alone: {s.words_only}; a search by meaning finds what is "
                         "said in other words or another language)")
        return rerank, notes

    def search_notes(self, query: str) -> str:
        """Every ranked excerpt, best first, each with its score and its neighbours whole; no score cuts one or decides
        for the model (search.py's docstring says why)."""
        hits, stale, total = self.searcher.find(query, k=None)
        if total == 0:
            return "no candidates: nothing in the index matched"
        rerank, notes = self._ranked("an excerpt", "candidates", [h.score for h in hits])
        lines = [f"{total} ranked candidates, best first " + (
            "(relevance: the reranker's score from 0 to 1, for ordering; a low score does not show the files lack the "
            "answer)" if rerank else "(in the search's order: no rerank endpoint is set, so no relevance score)")]
        for h in hits:
            self.last_scope.add(str(Path(getattr(h, "path", h.where.rsplit(":", 1)[0])).resolve()))
            before, after = h.before and h.before.text, h.after and h.after.text
            body = guard.redact(h.text)[0]
            before, after = (guard.redact(x)[0] if x else x for x in (before, after))
            section = guard.redact(h.section)[0] if h.section else ""
            s = self.sources.add("file", h.span, section, "\n".join(x for x in (before, body, after) if x))
            match = f"; the match: {h.where.rsplit(':', 1)[1]}" if h.span != h.where else ""
            copies = getattr(h, "copies", None) or []
            score = "" if not rerank else " (not reranked)" if h.score is None else f" (relevance {h.score:.2f})"
            lines.append(f"[{s.id}] {h.span}{score}" + (f" § {section}" if section else "")
                         + match + (" (changed since indexing; location verified in the current file)"
                                    if getattr(h, "file_changed", False) else "")
                         + "\n" + "\n".join(([f"(before) {before}"] if before else []) + [body]
                                                    + ([f"(after) {after}"] if after else [])
                                                    + ([f"(also in: {', '.join(copies)})"] if copies else [])))
        lines += notes
        if stale:
            lines.append(f"({stale} excerpts dropped: their files changed since indexing)")
        lines += self._not_indexed()
        return "\n\n".join(lines)

    def _not_indexed(self) -> list[str]:
        """What the index never ranked, counted by reason (grep and read reach it): said by every search."""
        index = getattr(self.searcher, "index", None)
        if index is None:
            return []
        held = sorted((NOT_INDEXED.get(r, (r, r))[n != 1], n) for r, n in index.coverage().items() if n)
        return [f"(not searched by meaning: {', '.join(f'{n} {label}' for label, n in held)}; "
                "grep and read still reach the files)"] if held else []

    def _indexed_folder(self, folder: str) -> Path | str:
        """The declared folder (or one inside it) `folder` names, or why overview cannot look there: it reads only
        the index, so it never asks to read elsewhere."""
        p = Path(folder).expanduser()
        if not p.is_absolute():
            hits = [r / p for r in self.roots if (r / p).is_dir()]
            if not hits:
                return f"{folder}: not found in the declared folders (tree without a path lists them)"
            p = hits[0]
        if not p.is_dir():
            return f"{folder}: not found in the declared folders (tree without a path lists them)"
        if not guard.allowed(p, self.roots):
            return f"{p}: overview covers only the declared folders (the index); tree, glob, grep and read reach " \
                   "other folders"
        real = Path(os.path.realpath(p))                     # through a link or ~: the folder as the index names it
        for root in self.roots:
            base = Path(os.path.realpath(root))
            if real == base or base in real.parents:
                return root / real.relative_to(base)
        return p

    def _picture(self, path: str) -> tuple[str, str]:
        """(headings, first line) of an indexed file, redacted, for one line of overview."""
        heads, lead = self.searcher.outline(path)
        return guard.redact(", ".join(heads))[0], guard.redact(lead)[0]

    def overview(self, query: str = "", folder: str = "") -> str:
        """Every document on a topic, best first (Searcher.documents), each with its headings, first line and best
        passage (whole, on one line; a source); without a topic, the catalog of the
        folder's indexed files by path. Only what the index holds: a folder outside the declared ones is refused."""
        if self.searcher is None or getattr(self.searcher, "index", None) is None:
            return "overview needs the index, which is not open in this session; use tree, glob and grep"
        where = ""
        scope = None
        if folder.strip():
            scope = self._indexed_folder(folder.strip())
            if isinstance(scope, str):
                return scope
            where = f" in {scope}"
        if not query.strip():
            return self._catalog(scope, where)
        docs = self.searcher.documents(query, None if scope is None else str(scope))
        if not docs:
            return "no documents: nothing in the index matched" + (f"{where}" if where else "")
        rerank, notes = self._ranked("a document", "documents", [d.score for d in docs])
        lines = [f"{len(docs)} documents matching{where}, best first " + (
            "(relevance: the reranker's score of each document's best passage, for ordering; a low score does not show "
            "a document lacks the topic)" if rerank else
            "(in the search's order: no rerank endpoint is set, so no relevance score)")]
        for d in docs:
            self.last_scope.add(str(Path(d.path).resolve()))
            score = "" if not rerank else "not reranked; " if d.score is None else f"relevance {d.score:.2f}; "
            head = f"{d.path} ({score}{d.matches} matching passage{'s' if d.matches != 1 else ''})"
            try:
                from peresearch.zetokrag.index import sha256
                changed = sha256(Path(d.path)) != d.sha
            except OSError:
                changed = True
            if changed:      # what the index holds of it is not what is there: nothing quoted, no source to cite
                lines.append(f"{head}\n  (changed since indexing: read the file for its current text)")
                continue
            text = guard.redact(d.text)[0]
            section = guard.redact(d.section)[0] if d.section else ""
            src = self.sources.add("file", d.where, section, text)
            heads, lead = self._picture(d.path)
            entry = [f"[{src.id}] {head}"]
            entry += [f"  headings: {heads}"] if heads else []
            entry += [f"  starts with: {lead}"] if lead else []
            entry += [f"  best passage: {d.where}" + (f" § {section}" if section else ""), "  " + _one_line(text)]
            entry += [f"  also in: {', '.join(d.copies)}"] if d.copies else []          # near-copies folded into it
            lines.append("\n".join(entry))
        lines += notes + self._not_indexed()
        lines.append("(read a document with read; search_notes finds passages, grep exact words)")
        return "\n\n".join(lines)

    def _catalog(self, scope: Path | None, where: str) -> str:
        inside = None if scope is None else str(scope).rstrip("/") + "/"
        paths = [p for (p,) in self.searcher.index.read("SELECT path FROM files WHERE status='ok' ORDER BY path")
                 if (inside is None or p.startswith(inside))
                 and (self._roots is not None or guard.allowed(Path(p), self.roots))]
        if not paths:
            return "no files: the index holds none" + where
        self.last_scope.update(paths)
        lines = [f"{len(paths)} files searched by meaning{where}, by path (give a topic to rank them)"]
        for path in paths:
            heads, lead = self._picture(path)
            lines.append(path + (f" — {heads}" if heads else f" — {lead}" if lead else ""))
        tail = self._not_indexed()
        return "\n".join(lines) + ("\n\n" + "\n\n".join(tail) if tail else "")

    def grep(self, pattern: str, glob: str = "**/*") -> str:
        """Every line matching `pattern` (a regular expression, case ignored) in the files `glob` names, whole. ripgrep
        reads plain text files and, for PDFs, notebooks and text not in UTF-8, the index's extracted texts; a file the
        index has not seen as it is now is read by the converter. Matches come in turns across files — each file's
        first, then each file's second… — so a file with hundreds does not hide the others in what the model sees
        first; the result says what it did not search."""
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            return f"invalid regular expression: {e}"
        scopes, glob = self._scope(glob, "grep")
        found, left = _walk(scopes, glob)
        deadline = time.monotonic() + GREP_SECONDS            # the whole call: the result built from the matches too
        hits, stopped, too_large, unreadable = self._matches(found, pattern, rx, deadline)
        order = [p for p in found if hits.get(p)]
        out, total, late = [], sum(len(v) for v in hits.values()), False
        for turn in range(max((len(hits[p]) for p in order), default=0)):       # each file's match `turn`, in turn
            for p in order:
                if turn < len(hits[p]):
                    if time.monotonic() >= deadline:
                        late = True
                        break
                    where, line = hits[p][turn]
                    self.last_scope.add(str(p.resolve()))
                    shown, before, after, note = _around(line, rx, where)
                    s = self.sources.add("file", where, p.name, shown)
                    out.append(f"[{s.id}] {where}: {before}{s.text}{after}{note}")
            if late:
                break
        assert len(out) == total or late, "grep showed a wrong number of lines"
        notes = [f"(stopped after {GREP_SECONDS} s: {len(out):,} of {total:,} matching lines shown; narrow the pattern "
                 "or the glob for the rest)"] if late else []
        if too_large:
            limit = f"{parse.MAX_BYTES // 2**20} MiB" if parse.MAX_BYTES % 2**20 == 0 else f"{parse.MAX_BYTES} bytes"
            notes.append(f"({too_large} file{'s' * (too_large != 1)} over {limit} not searched)")
        if unreadable:
            notes.append(f"({unreadable} file{'s' * (unreadable != 1)} could not be read: no permission)")
        if stopped:
            notes.append(f"(stopped after {GREP_SECONDS} s: some files were not searched)")
        if _left_note(left, "not searched"):
            notes.append(_left_note(left, "not searched"))
        return "\n".join((out or ["no match"]) + notes)

    def _matches(self, found: list[Path], pattern: str, rx, deadline: float) -> tuple[dict[Path, list], bool, int, int]:
        """{file: [(place, line)]} for every match, in line order; whether the time ran out; how many text files were
        over parse.MAX_BYTES (not searched, as the converter does not read them; ripgrep's --max-filesize does not
        apply to a path it is given); how many the user may not read."""
        index = getattr(self.searcher, "index", None)
        rows = {r[0]: r[1:] for r in index.read("SELECT path, size, mtime, sha FROM files")} if index else {}
        texts = dict(index.read("SELECT sha, units FROM texts")) if index else {}
        plain, stored, slow, too_large, unreadable = [], {}, [], 0, 0
        for p in found:
            kind = parse.kind(p)
            if kind == "unsupported":
                continue
            try:
                st = p.stat()
            except OSError:                                   # gone since the walk
                continue
            if not os.access(p, os.R_OK):                     # no permission: said, never a failed search
                unreadable += 1
                continue
            if kind not in parse.PACKED and st.st_size > parse.MAX_BYTES:
                too_large += 1
                continue
            row = rows.get(str(p))
            fresh = row is not None and (row[0], row[1]) == (st.st_size, st.st_mtime) and row[2] == sha256(p)
            store = index.home / extracted.FOLDER / f"{row[2]}.txt" if fresh and row[2] in texts else None
            if store is not None and store.is_file():         # the index's converted text, as it is now
                stored.setdefault(str(store), (json.loads(texts[row[2]]), []))[1].append(p)
            elif kind in parse.CONVERTED:                     # its text is not its bytes
                slow.append(p)
            elif fresh:                                       # UTF-8 text as the index saw it
                plain.append(p)
            else:                                             # not seen by the index: UTF-8 as it is, or convert
                try:
                    raw = p.read_bytes()
                except OSError:                               # gone, or closed to us, since the check above
                    continue
                try:
                    raw.decode("utf-8")
                    utf8 = not raw.startswith(tuple(b for b, _, _ in parse.BOMS))
                except UnicodeDecodeError:
                    utf8 = False
                (plain if utf8 else slow).append(p)
        hits: dict[Path, list] = {}
        stopped = False
        if RG is None:
            slow += plain + [q for _, qs in stored.values() for q in qs]
        else:
            got, ok, done = _rg([str(q) for q in plain], pattern, deadline)
            more, ok2, done2 = _rg(list(stored), pattern, deadline) if ok else ({}, False, True)
            if not (ok and ok2):                              # a pattern Python takes and ripgrep does not
                slow += plain + [q for _, qs in stored.values() for q in qs]
            else:
                stopped = not (done and done2)
                for q in plain:
                    if str(q) in got:
                        hits[q] = [(f"{q}:L{n}", line) for n, line in got[str(q)]]
                for store, (units, originals) in stored.items():
                    if not os.path.exists(store):              # removed by an index update meanwhile: convert
                        slow += originals
                        continue
                    for q in originals:
                        if store in more:
                            hits[q] = [(_place(q, *extracted.locate(units, n)), line) for n, line in more[store]]
        for q in slow:
            if time.monotonic() > deadline:
                stopped = True
                break
            doc = parse.read(q)
            if doc.status != "ok" or any("\x00" in b.text for b in doc.blocks):   # a NUL anywhere: binary, as for rg
                continue
            found_here = []
            for b in doc.blocks:
                for i, line in enumerate(parse.lines(b.text)):
                    if rx.search(line):
                        found_here.append((_place(q, b.unit, b.start + i if b.unit == "line" else b.start), line))
            if found_here:
                hits[q] = found_here
        return hits, stopped, too_large, unreadable

    def glob(self, pattern: str) -> str:
        roots, pattern = self._scope(pattern, "glob")
        found, left = _walk(roots, pattern)
        self.last_scope.update(str(p.resolve()) for p in found)
        out = [str(p) for p in found]
        note = _left_note(left, "not listed")
        return "\n".join((out or ["no file matches"]) + ([note] if note else []))

    def read(self, path: str, start: int = 1, end: int | None = None) -> str:
        p, _ = self._path(path, "read")
        doc = parse.read(p, cut_outputs=False)            # a notebook's outputs whole (the index keeps a cut)
        if doc.status != "ok":
            return f"{p}: cannot be read ({doc.status}{': ' + doc.note if doc.note else ''})"
        if not any(b.text.strip() for b in doc.blocks):    # an empty file, deck, document or notebook
            return f"{p}: holds no text"
        if doc.blocks[0].unit == "line":
            lines = parse.lines(doc.blocks[0].text)
            start = max(1, start)
            end = min(len(lines), end or len(lines))
            where, text = f"{p}:L{start}-{end}", "\n".join(lines[start - 1:end])
        else:                                 # pages or cells: start/end number those units
            unit = doc.blocks[0].unit
            chosen = [b for b in doc.blocks if start <= b.start and (end is None or b.start <= end)]
            label = "p." if unit == "page" else "cell "
            where = f"{p}:{label}{chosen[0].start}-{chosen[-1].start}" if chosen else f"{p}:{label}{start}"
            text = "\n\n".join(b.text for b in chosen)
        s = self.sources.add("file", where, p.name, guard.redact(text)[0])
        return f"[{s.id}] {where}\n{s.text}"

    def web_search(self, query: str, n: int = 5) -> str:
        if self.recorded_gaps is None:
            raise guard.Refused("first record with gaps(have, missing) what the user's files already cover and what "
                                "is missing; then search the web for what is missing")
        self.web_calls += 1
        found = self.web.search(query, n)
        text = found.text.strip()
        if not text:
            return f"(web search: {found.provider})\n\nno results"
        blocks = re.split(r"\n\s*\n", text)                # a link's paragraph is its source: quotes check against it
        cited = []
        for title, url in found.links:
            s = self.sources.add("web", url, title or url, next((b for b in blocks if url in b), text))
            cited.append(f"[{s.id}] " + (f"{title} — " if title else "") + f"{url} ({site_kind(url)})")
        return "\n\n".join([f"(web search: {found.provider})", text] + (
            ["sources (cite their ids; fetch reads a page):\n" + "\n".join(cited)] if cited else ["(no link in it)"]))

    def search_papers(self, query: str, n: int = 5) -> str:
        """Scholarly papers, each with what kind of record it is (journal article, conference paper, preprint,
        RETRACTED), venue, year, citations and authors; a paper's open PDF can then be read with fetch. Counts as
        one of the question's web searches."""
        if self.recorded_gaps is None:
            raise guard.Refused("first record with gaps(have, missing) what the user's files already cover and what "
                                "is missing; then search the web for what is missing")
        self.web_calls += 1
        out = []
        for p in self.papers.search(query, n):
            facts = (["RETRACTED"] if p.retracted else []) + ([p.kind] if p.kind else []) + \
                ([p.venue] if p.venue and p.venue not in p.kind else []) + \
                ([f"{p.citations:,} citations"] if p.citations is not None else []) + \
                [f"found by {', '.join(p.found_by)}"]
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            s = self.sources.add("web", p.url, p.title, p.abstract or p.title)
            if p.pdf:
                self.sources.urls.setdefault(p.pdf, s.id)          # fetch may open the paper's PDF
            out.append("\n".join([f"[{s.id}] {p.title}" + (f" ({p.year})" if p.year else ""), " · ".join(facts)]
                                 + ([authors] if authors else []) + [p.url] + ([f"PDF: {p.pdf}"] if p.pdf else [])
                                 + ([p.abstract] if p.abstract else [])))
        missed = getattr(self.papers, "missed", [])
        note = [f"(did not answer: {', '.join(missed)}; results may be missing)"] if missed else []
        return "\n\n".join((out or ["no papers found"]) + note)

    def search_wikipedia(self, query: str, languages: list[str] | str = ("en",), n: int = 3) -> str:
        """Wikipedia articles in the editions named: each one's lead summary, address and title in the other target
        languages. Counts as one of the question's web searches."""
        if self.recorded_gaps is None:
            raise guard.Refused("first record with gaps(have, missing) what the user's files already cover and what "
                                "is missing; then search the web for what is missing")
        langs = [languages] if isinstance(languages, str) else list(languages or ["en"])
        self.web_calls += 1
        articles, missed = self.wiki.search(query, langs, n)
        out = []
        for a in articles:
            s = self.sources.add("web", a.url, a.title, a.summary or a.title)
            other = " · ".join(f"{lang}: {title}" for lang, title in a.other.items())
            out.append("\n".join([f"[{s.id}] {a.title} ({a.lang}.wikipedia: an encyclopedia, a starting point, not "
                                  "primary evidence)", a.url] + ([a.summary] if a.summary else [])
                                 + ([f"in other languages: {other}"] if other else [])))
        note = [f"(did not answer: {', '.join(missed)}; results may be missing)"] if missed else []
        return "\n\n".join((out or ["no articles found"]) + note)

    def fetch(self, url: str) -> str:
        """The whole main text of a page a search returned (a paper runs to tens of thousands of characters: the agent
        shows what fits the context, the rest read on with recall); the source keeps it for the quote check."""
        if url not in self.sources.urls:
            raise guard.Refused("fetch only opens URLs returned by web_search in this question")
        text = self.web.fetch(url)
        s = self.sources.items[self.sources.urls[url]]
        if not text:
            return f"[{s.id}] {url}\nthe page could not be read"
        s = self.sources.add("web", url, s.title, text)
        return f"[{s.id}] {url}\n{text}"

    # --- for the model ---

    SPECS = {
        "tree": ("Show the folders and files the user has: every declared folder, or one folder inside them, a few "
                 "levels deep. Use it to see what a project or folder contains. A folder outside the declared "
                 "ones is shown only if the user allows it when asked.",
                 {"path": ("string", "a folder, e.g. ~/projects/x; default: every declared folder"),
                  "depth": ("integer", "levels to show, default 2, at most 4")}, []),
        "search_notes": ("Search the user's own files (notes, projects, course material) by meaning and keywords. "
                         "Returns the ranked excerpts, best first, with source ids and a relevance score. Scores "
                         "order the excerpts; they do not tell whether the files hold the answer: read the excerpts. "
                         "For a complete list of every place a term occurs, use grep.",
                         {"query": ("string", "what to look for")},
                         ["query"]),
        "overview": ("List the user's DOCUMENTS on a topic, best first, one entry per file with its headings and best "
                     "passage: for questions about everything the user has on a subject, which "
                     "files cover something, or to survey a project. Without a query: every file in a folder with its "
                     "headings. Read the documents that matter, and say how many of the listed ones you read.",
                     {"query": ("string", "the topic; empty for the folder's catalog"),
                      "folder": ("string", "a declared folder or one inside it, e.g. ~/projects/x; default: all")}, []),
        "grep": ("Find lines matching a regular expression (case-insensitive) in the user's files: exact names, "
                 "numbers, error messages; PDFs, notebooks, tables (.csv, .json) and logs included. Folders made by "
                 "tools (environments, node_modules, .git) and hidden ones are searched only when the glob names "
                 "them, e.g. node_modules/pkg/**.",
                 {"pattern": ("string", "regular expression"),
                  "glob": ("string", "file pattern such as **/*.md, default all files")}, ["pattern"]),
        "glob": ("List the user's files whose path matches a pattern, e.g. **/*.ipynb, notes/** or "
                 "~/projects/x/*.py. Folders made by tools and hidden ones are listed only when the pattern names "
                 "them. To see folders, use tree.",
                 {"pattern": ("string", "glob pattern")}, ["pattern"]),
        "read": ("Read one of the user's files, whole or from start to end: lines for text files, pages for PDFs, "
                 "cells for notebooks. "
                 "A file outside the declared folders is read only if the user allows it when asked.",
                 {"path": ("string", "path from a previous result"), "start": ("integer", "first line/page/cell"),
                  "end": ("integer", "last line/page/cell")}, ["path"]),
        "gaps": ("Record, before searching the web, what the user's own files already cover and what is missing "
                 "or not enough. web_search is refused until this is called.",
                 {"have": ("array", "points the user's files cover (with source ids), may be empty"),
                  "missing": ("array", "what is missing or not enough, to look for on the web")}, ["have", "missing"]),
        "web_search": ("Search the web for what gaps recorded as missing, or to check that something is current. "
                       "Each result says what kind of site it comes from (university, preprint server, blog…).",
                       {"query": ("string", "search query, no personal or secret data"),
                        "n": ("integer", "number of results, default 5")}, ["query"]),
        "search_papers": ("Search scholarly papers (Semantic Scholar, arXiv, and OpenAlex when configured): each "
                          "result says what kind of record it is (journal article, conference paper, preprint, "
                          "RETRACTED), "
                          "its venue, year, citation count and authors, and its open PDF when there is one (read it "
                          "with fetch). Use it for research questions; only after gaps. Counts as a web search.",
                          {"query": ("string", "what to look for, in the language of the papers you want"),
                           "n": ("integer", "number of papers, default 5")}, ["query"]),
        "search_wikipedia": ("Search Wikipedia in one or more languages (e.g. [\"vi\", \"en\"]): each article's lead "
                             "summary, its address and its title in English, Vietnamese, Chinese, Japanese, Korean "
                             "and Hindi, to find the term to search for in another language. A starting point and a "
                             "map to sources, not primary evidence. Only after gaps. Counts as a web search.",
                             {"query": ("string", "what to look for, in the edition's language"),
                              "languages": ("array", "edition codes such as en, vi, zh, ja, ko, hi; default en; "
                                                     "at most 3"),
                              "n": ("integer", "articles per edition, default 3")}, ["query"]),
        "fetch": ("Read the main text of a page a search returned (a web page, a paper's PDF, an article).",
                  {"url": ("string", "a URL from search results")}, ["url"]),
    }

    def schemas(self, web: bool = True) -> list[Tool]:
        web = web and not self.local_only
        out = []
        for name, (desc, props, required) in self.SPECS.items():
            assert callable(getattr(self, name, None)), f"tool {name} has a schema but no method"
            pages = bool(web and self.web and self.web.searcher)
            scholarly = bool(web and self.papers and self.papers.sources)
            encyclopedia = bool(web and self.wiki)
            if (name == "web_search" and not pages) or (name == "search_papers" and not scholarly) or \
                    (name == "search_wikipedia" and not encyclopedia) or \
                    (name in ("gaps", "fetch") and not (pages or scholarly or encyclopedia)):
                continue
            out.append(Tool(name, desc, {"type": "object", "required": required, "properties": {
                k: {"type": t, "description": d, **({"items": {"type": "string"}} if t == "array" else {})}
                for k, (t, d) in props.items()}}))
        return out

    def call(self, name: str, args: dict) -> str:
        """Run one tool; errors come back as text for the model, never as a crash. Arguments are read by their
        schema type: a model may write a number as a string (Llama-3.2-1B through Ollama: {"k": "10"}, 2026-10-06)."""
        self.last_scope = set()
        if name not in self.SPECS:
            return f"unknown tool {name!r}"
        if self.local_only and name in WEB_TOOLS:            # offered no more, yet a model may still name them
            return "refused: the user turned the web off for this session (local only)"
        props, kwargs = self.SPECS[name][1], {}
        for k, v in args.items():
            if k not in props or v is None:         # null for an optional argument: not given (strict schemas)
                continue
            if props[k][0] == "integer":
                v = _whole(v)
                if v is None:
                    return f"bad arguments for {name}: {k} must be a whole number (got {args[k]!r})"
            kwargs[k] = v
        try:
            return getattr(self, name)(**kwargs)
        except guard.Refused as e:
            return f"refused: {e}"
        except TypeError as e:
            return f"bad arguments for {name}: {e}"
        except AssertionError:          # an internal check failed: a bug, shown to the user, never handed to the model
            raise
        except Exception as e:
            return f"{name} failed: {type(e).__name__}: {str(e)[:200]}"
