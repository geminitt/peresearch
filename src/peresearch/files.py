"""Which files peresearch reads: the rules shared by the index (zetokrag/index.py) and the file tools.

No rule rests on how a person names a folder. Each rests on something a tool wrote, a convention every system
shares, or the user's own words:
- a folder a tool made is not walked: a Python environment (it holds pyvenv.cfg), a conda environment (conda-meta/),
  a cache or build folder tagged by the Cache Directory Tagging spec (a CACHEDIR.TAG with its signature), a version
  control folder (.git, .hg, .svn) and the folders package managers name themselves (node_modules, site-packages,
  dist-packages, __pycache__, *.egg-info, *.dist-info);
- a hidden name (starting with ".") below a declared folder is left out: tool settings far more often than notes;
  so is an Office owner file ("~$report.docx": a few bytes naming who has the document open, hidden only on Windows);
- a folder with more than MAX_FILES_PER_FOLDER files side by side is left out as a likely dataset, unless it is a
  declared folder or the user includes it (a flat folder of thousands of notes is one `include` away);
- the user decides last: `exclude` and `include` in config.json, and `.peresearchignore` files (below);
- a named pipe, socket or device is never opened (opening a pipe waits for a writer for ever);
- protected paths (guard.DENY) are refused and only counted, never named;
- peresearch's own folder ($PERESEARCH_HOME: the index, extracted texts, the private eval corpus) is never walked,
  whatever its name and whatever the user includes or declares.
Everything else left out is reported with its reason, so `Index.why_not` can say why a path is missing.

Patterns (config.json "exclude" and "include"; one per line in a `.peresearchignore`, which applies to its own folder
and everything below it): a glob where `*` and `?` stay inside one path part and `**` crosses parts; a pattern
without "/" matches a name at any depth, one with "/" matches the path from the pattern's folder (a declared folder
for config.json; an absolute or ~ pattern matches the full path); a trailing "/" matches folders only; in a
`.peresearchignore`, "!" re-includes and "#" starts a comment; the last matching pattern wins, and config.json's come
last. An excluded folder is not entered, so nothing below it can be included again (as in .gitignore). Matching is
case-sensitive.
"""

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from peresearch import guard

CACHEDIR_SIGNATURE = b"Signature: 8a477f597d28d172789f06886806bc55"   # the Cache Directory Tagging spec
PACKAGE_FOLDERS = {"node_modules", "site-packages", "dist-packages", "__pycache__"}
PACKAGE_SUFFIXES = (".egg-info", ".dist-info")
VERSION_CONTROL = {".git", ".hg", ".svn"}
MAX_FILES_PER_FOLDER = 500
IGNORE_FILE = ".peresearchignore"
OWN = "peresearch's own data"
OFFICE_OWNER = "Office owner file (~$)"   # Word, Excel and PowerPoint write one beside an open document
PROTECTED = "protected"                    # a reason walk never reports by path: it only counts these


@dataclass(frozen=True)
class Skipped:
    path: str
    reason: str          # symlinked_folder | tool_made | hidden | user_excluded | large_folder | unreadable |
                         # special_file
    note: str = ""


@dataclass(frozen=True)
class Pattern:
    base: Path           # the folder the pattern is read from (a declared folder for config.json)
    regex: re.Pattern
    anchored: bool       # matched against the path from `base`, not against the name alone
    folders_only: bool
    include: bool
    source: str          # where it was written, for the report


def made_by_a_tool(folder: Path) -> str | None:
    """Why `folder` is a tool's (an environment, a cache, version control, packages), or None."""
    name = folder.name
    if name in VERSION_CONTROL:
        return "version control"
    if name in PACKAGE_FOLDERS or name.endswith(PACKAGE_SUFFIXES):
        return f"{name}: made by a package manager"
    try:                                     # a folder that cannot be looked into is the walk's to report
        if (folder / "pyvenv.cfg").is_file():
            return "Python environment (pyvenv.cfg)"
        if (folder / "conda-meta").is_dir():
            return "conda environment (conda-meta)"
        tag = folder / "CACHEDIR.TAG"
        if tag.is_file():
            with open(tag, "rb") as f:
                if f.read(len(CACHEDIR_SIGNATURE)) == CACHEDIR_SIGNATURE:
                    return "cache (CACHEDIR.TAG)"
    except OSError:
        pass
    return None


def _translate(glob: str) -> str:
    """A glob as a regular expression: `*` and `?` inside one path part, `**` across parts, [...] a set."""
    out, i = [], 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        elif glob[i] == "[" and "]" in glob[i + 2:]:
            j = glob.index("]", i + 2)
            body = glob[i + 1:j]
            negate = body.startswith("!")
            body = (body[1:] if negate else body).replace("\\", "\\\\").replace("^", "\\^").replace("[", "\\[")
            out.append("[" + ("^" if negate else "") + body + "]")
            i = j + 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return "".join(out)


def compile_pattern(text: str, base: Path, include: bool, source: str) -> Pattern | None:
    """One pattern, or None for an empty line or a comment."""
    text = text.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("!"):
        include, text = True, text[1:]
    folders_only = text.endswith("/")
    text = text.rstrip("/")
    if text == "~" or text.startswith("~/"):        # the home folder; "~$x.docx" or "~bob" are names, not homes
        text = str(Path(text).expanduser())
    if text.startswith("/"):
        base, text, anchored = Path("/"), text.lstrip("/"), True
    else:
        anchored = "/" in text
    if not text:
        return None
    return Pattern(Path(base), re.compile(_translate(text)), anchored, folders_only, include, source)


def matches(p: Pattern, path: Path, is_folder: bool) -> bool:
    if p.folders_only and not is_folder:
        return False
    if p.anchored:
        try:
            rel = path.relative_to(p.base).as_posix()
        except ValueError:
            return False
        return bool(p.regex.fullmatch(rel))
    return bool(p.regex.fullmatch(path.name))


def decide(patterns: list[Pattern], path: Path, is_folder: bool) -> Pattern | None:
    """The last pattern that matches `path`, which decides it; None when no pattern speaks."""
    found = None
    for p in patterns:
        if matches(p, path, is_folder):
            found = p
    return found


def config_patterns(config: dict, root: Path) -> list[Pattern]:
    """config.json's `exclude` then `include` patterns, read from `root`. A malformed entry is the user's input,
    reported as an error, not a bug."""
    out = []
    for key, include in (("exclude", False), ("include", True)):
        values = config.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise guard.ConfigError(f"`{key}` must be a list of patterns (strings)")
        for v in values:
            p = compile_pattern(v.lstrip("!"), root, include, f"config.json {key}: {v}")
            if p is not None:
                out.append(p)
    return out


def ignore_file_patterns(folder: Path) -> list[Pattern]:
    """The patterns of `folder`'s own .peresearchignore, if it has one."""
    path = folder / IGNORE_FILE
    if not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        p = compile_pattern(line, folder, False, f"{path}:{n}")
        if p is not None:
            out.append(p)
    return out


def _own() -> set[Path]:
    """peresearch's own folder (its index, extracted texts, eval data), as named and as resolved."""
    return {guard.home(), Path(os.path.realpath(guard.home()))}


def _owned(root: Path, own: set[Path]) -> bool:
    real = Path(os.path.realpath(root))
    return bool(own & {root, *root.parents, real, *real.parents})


def _folder(p: Path, patterns: list[Pattern], own: set[Path]) -> tuple[Skipped | None, bool]:
    """(why the folder `p` is left out, or None; whether a user pattern included it). The order of the rules is
    the module's: protected, symlinked, peresearch's own, the user's patterns, made by a tool, hidden."""
    if guard.denied(p):
        return Skipped(str(p), PROTECTED), False
    if os.path.islink(p):
        return Skipped(str(p), "symlinked_folder"), False
    if p in own or Path(os.path.realpath(p)) in own:                 # whatever the user's patterns say
        return Skipped(str(p), "tool_made", OWN), False
    rule = decide(patterns, p, True)
    if rule is not None and not rule.include:
        return Skipped(str(p), "user_excluded", rule.source), False
    if rule is not None:
        return None, True
    made = made_by_a_tool(p)
    if made:
        return Skipped(str(p), "tool_made", made), False
    if p.name.startswith("."):
        return Skipped(str(p), "hidden"), False
    return None, False


def special(p: Path) -> str | None:
    """What `p` is when it is not a regular file (a named pipe, a socket, a device), following links; opening one
    can wait for ever. None for a regular file, or for one that cannot be looked at (the reader reports that)."""
    try:
        mode = os.stat(p).st_mode
    except OSError:
        return None
    if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
        return None
    return ("named pipe" if stat.S_ISFIFO(mode) else "socket" if stat.S_ISSOCK(mode)
            else "device" if stat.S_ISCHR(mode) or stat.S_ISBLK(mode) else "special file")


def _file(p: Path, patterns: list[Pattern], roots: list[Path]) -> Skipped | None:
    if not guard.allowed(p, roots):                                   # protected, or a link that leads outside
        return Skipped(str(p), PROTECTED)
    kind = special(p)
    if kind is not None:
        return Skipped(str(p), "special_file", kind)
    rule = decide(patterns, p, False)
    if rule is not None and not rule.include:
        return Skipped(str(p), "user_excluded", rule.source)
    if rule is None and p.name.startswith("."):
        return Skipped(str(p), "hidden")
    if rule is None and p.name.startswith("~$"):
        return Skipped(str(p), "tool_made", OFFICE_OWNER)
    return None


def _large(folder: Path, n_files: int, included: bool) -> Skipped | None:
    return Skipped(str(folder), "large_folder", f"{n_files} files") if n_files > MAX_FILES_PER_FOLDER and not included \
        else None


def walk(roots: list[Path], config: dict, skip: Callable[[Skipped], None],
         denied: Callable[[], None], named: tuple[str, ...] = ()) -> Iterator[Path]:
    """Every file the rules let through, under each declared folder; what they leave out goes to `skip`, a protected
    path only to `denied`. The caller still decides by file type. `named`: include patterns for what a tool was asked
    for by name (a hidden or tool-made folder), read before the user's own patterns, so the user's word still wins;
    nothing makes protected paths or peresearch's own data readable."""
    own = _own()
    for root in roots:
        if _owned(root, own):
            skip(Skipped(str(root), "tool_made", OWN))
            continue
        user = [p for p in (compile_pattern(n, root, True, f"asked for by name: {n}") for n in named) if p] \
            + config_patterns(config, root)
        inherited: dict[str, list[Pattern]] = {str(root): []}
        included: set[str] = {str(root)}                     # a declared folder is never "a large folder"

        def unreadable(error: OSError) -> None:             # no permission, or gone: the world, not a bug
            inherited.pop(str(error.filename), None)
            skip(Skipped(str(error.filename), "unreadable", type(error).__name__))

        for dirpath, dirnames, filenames in os.walk(root, onerror=unreadable):
            here = Path(dirpath)
            mine = inherited.pop(dirpath) + ignore_file_patterns(here)
            patterns = mine + user                           # config.json last: the user's word wins
            keep = []
            for d in sorted(dirnames):
                p = here / d
                why, chosen = _folder(p, patterns, own)
                if why is not None:
                    denied() if why.reason == PROTECTED else skip(why)
                    continue
                if chosen:
                    included.add(str(p))
                keep.append(d)
                inherited[str(p)] = mine
            dirnames[:] = keep
            files = [f for f in sorted(filenames) if f != IGNORE_FILE]
            large = _large(here, len(files), dirpath in included)
            if large is not None:
                skip(large)
                continue
            for f in files:
                p = here / f
                why = _file(p, patterns, roots)
                if why is not None:
                    denied() if why.reason == PROTECTED else skip(why)
                    continue
                yield p
        assert not inherited, f"folders queued but never walked: {sorted(inherited)[:3]}"


def check(path: Path, roots: list[Path], config: dict) -> Skipped | None:
    """Why `walk` leaves `path` (a file or folder) out: the verdict on it or on the first folder above it that is left
    out, with walk's reason (PROTECTED for a protected one); None when walk reaches it, or it lies in no declared
    folder. One path at a time, for the tools that look at a single folder (tree)."""
    path = Path(path)
    root = next((r for r in roots if path == r or r in path.parents), None)
    if root is None:
        return None
    own = _own()
    if _owned(root, own):
        return Skipped(str(root), "tool_made", OWN)
    user = config_patterns(config, root)
    folder, mine, chosen = root, [], True
    parts = path.relative_to(root).parts
    for i, name in enumerate(parts):
        mine = mine + ignore_file_patterns(folder)
        p = folder / name
        if i < len(parts) - 1 or (p.is_dir() and not p.is_symlink()):
            why, chosen = _folder(p, mine + user, own)
            if why is not None:
                return why
            folder = p
            continue
        try:
            n = sum(1 for e in os.scandir(folder) if not e.is_dir() and e.name != IGNORE_FILE)
        except OSError as e:
            return Skipped(str(folder), "unreadable", type(e).__name__)
        return _large(folder, n, chosen) or _file(p, mine + user, roots)
    return None
