"""One pipeline for every tool the model may call, built-in or from an MCP server: each tool declares only its own —
name, description, argument schema, permission group and function — and every call goes through the same stages in
order: its arguments checked against the schema, its permission, the question's budget, the run (errors as text,
HTTP 429 waited out), each refusal said in words the model can act on.

Bounding a result, layer 1: a tool stops collecting past COLLECT_BYTES and says what it left; the registry cuts what a
tool still returns past it. Layers 2 and 3 (a page at a time, cut to the context room) are the agent's."""

import time
from dataclasses import dataclass, field
from typing import Callable

from peresearch import guard
from peresearch.messages import Tool

GROUPS = ("files", "web")           # files: the declared folders (outside them the user is asked, by guard); web
RATE_WAITS = (1.0, 2.0)             # seconds before the 2nd and 3rd attempt when a service says no Retry-After
LONGEST_WAIT = 60.0                 # a service asking for longer is not waited for: said to the model instead
# What a tool may return, in UTF-8 (owner, 2026-10-21, after measuring the declared folders: research calls reached
# 113 KB, while grep `def ` returned 87 MB and took 465 s to redact and store; runs/design-revision-2026-10-21/tool-output)
COLLECT_BYTES = 1 << 20
NOTES_BYTES = 4096                  # room past it for a tool's own lines saying what it left
REPEATED = "already called with the same arguments; its result is above. Use it, or answer."
WEB_OFF = "refused: the user turned the web off for this session (local only)"


def whole(value) -> int | None:
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


def size_label() -> str:
    """COLLECT_BYTES as the model reads it: "1 MiB", else "2,000 bytes"."""
    n = COLLECT_BYTES
    return f"{n // 2**20} MiB" if n % 2**20 == 0 else f"{n:,} bytes"


def utf8(text: str) -> bytes:
    """`text` as UTF-8, a file name's undecodable bytes (surrogates, os.fsdecode) included."""
    return text.encode("utf-8", "surrogatepass")


def fitting(parts: list[str], sep: str = "\n", room: int | None = None) -> int:
    """How many of `parts`, joined by `sep`, fit in `room` bytes of UTF-8 (COLLECT_BYTES by default)."""
    room = COLLECT_BYTES if room is None else room
    used = -len(utf8(sep))
    for i, part in enumerate(parts):
        used += len(utf8(sep)) + len(utf8(part))
        if used > room:
            return i
    return len(parts)


def head(text: str, room: int | None = None) -> str:
    """`text` within `room` bytes of UTF-8 (COLLECT_BYTES by default): after its last line end there when one lies in
    the second half, else cut mid-line (never mid-character)."""
    room = COLLECT_BYTES if room is None else room
    raw = utf8(text)
    if len(raw) <= room:
        return text
    for back in range(4):                     # a character is at most 4 bytes: never cut inside one
        try:
            cut = raw[:room - back].decode("utf-8", "surrogatepass")
            break
        except UnicodeDecodeError:
            continue
    end = cut.rfind("\n")
    return cut[:end + 1] if end >= len(cut) // 2 else cut


class RateLimited(Exception):
    """A service refused for the rate of requests (HTTP 429): `retry_after` seconds when it said how long to wait."""

    def __init__(self, service: str, retry_after: float | None = None):
        super().__init__(f"{service}: rate limited")
        self.service, self.retry_after = service, retry_after


@dataclass(frozen=True)
class Kept:
    """A result already in the agent's store, shown again from character `start` (recall)."""
    ref: str
    text: str
    start: int


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict                    # JSON Schema; with no "properties", the arguments pass as given
    group: str                          # GROUPS
    run: Callable                       # run(**arguments) -> str, or Kept
    budget: str = ""                    # "search" or "page": counted against the question's limit, when one is set
    offered: Callable[[], bool] = lambda: True      # whether it works in this session (a search server set…)
    store: bool = True                  # its result is kept in the agent's store (recall's is already there)

    def __post_init__(self):
        assert self.group in GROUPS, f"tool {self.name}: unknown permission group {self.group!r}"
        assert self.budget in ("", "search", "page"), f"tool {self.name}: unknown budget {self.budget!r}"

    def tool(self) -> Tool:
        return Tool(self.name, self.description, self.parameters)


@dataclass
class Budget:
    """One question's limits — None: no limit, as by default — and what it has used."""
    searches: int | None = None
    pages: int | None = None
    deadline: float | None = None       # the clock's value when the question's time ends
    calls: list = field(default_factory=list)
    searched: int = 0
    fetched: int = 0
    read_pages: set = field(default_factory=set)


@dataclass(frozen=True)
class Outcome:
    text: str                           # what the model is told
    store: bool = False                 # a tool ran: its result is kept (refusals before the run are not)
    kept: Kept | None = None
    refused: str = ""                   # the stage that refused it: "tool", "arguments", "permission" or "budget"


class Registry:
    def __init__(self, specs: list[ToolSpec], web_on: Callable[[], bool] = lambda: True, sleep=None, clock=None):
        names = [s.name for s in specs]
        assert len(set(names)) == len(names), f"tools declared twice: {names}"
        self.specs, self.web_on = {s.name: s for s in specs}, web_on
        self.sleep, self.clock = sleep or time.sleep, clock or time.monotonic

    def tools(self) -> list[Tool]:
        """The tools offered to the model now: those that work in this session, web ones while the web is on."""
        return [s.tool() for s in self.specs.values() if s.offered() and (s.group != "web" or self.web_on())]

    def run(self, name: str, args: dict, budget: Budget | None = None, on_run=None) -> Outcome:
        """One call through the stages. Without a budget (a call outside a question) nothing is counted."""
        s = self.specs.get(name)
        if s is None:
            return Outcome(f"unknown tool {name!r}", refused="tool")
        kwargs = self._arguments(s, args)
        if isinstance(kwargs, str):
            return Outcome(kwargs, refused="arguments")
        refused = self._permission(s)
        if refused:
            return Outcome(refused, refused="permission")
        refused = self._budget(s, args, budget) if budget is not None else ""
        if refused:
            return Outcome(refused, refused="budget")
        if on_run and s.store:
            on_run(name, args)
        got = self._run(s, kwargs, budget)
        if isinstance(got, Kept):
            return Outcome(got.text, kept=got)
        size = len(utf8(got))
        if size > COLLECT_BYTES + NOTES_BYTES:    # a tool that did not stop itself (one from an MCP server): cut
            got = head(got) + f"\n[cut at {size_label()}: {s.name} returned {size:,}; ask it for less]"
        return Outcome(got, store=s.store)

    @staticmethod
    def _arguments(s: ToolSpec, args: dict) -> dict | str:
        """The arguments the function receives, read by their declared type (a model may write a number as a string:
        Llama-3.2-1B through Ollama, {"k": "10"}, 2026-10-06), or what is wrong with them. Null is not given (strict
        schemas send it for an optional argument); an argument the schema does not declare is dropped."""
        props = s.parameters.get("properties")
        if props is None:
            return dict(args)
        kwargs = {}
        for k, v in args.items():
            if k not in props or v is None:
                continue
            if props[k].get("type") == "integer":
                v = whole(v)
                if v is None:
                    given = repr(args[k])
                    given = given if len(given) <= 60 else given[:59] + "…"      # never a model's whole argument back
                    return f"bad arguments for {s.name}: {k} must be a whole number (got {given})"
            kwargs[k] = v
        missing = [k for k in s.parameters.get("required", []) if k not in kwargs]
        if not missing:
            return kwargs
        what = props.get(missing[0], {}).get("description")
        return f"bad arguments for {s.name}: {missing[0]} is required" + (f" ({what})" if what else "")

    def _permission(self, s: ToolSpec) -> str:
        """The group's rule: web tools only while the web is on; a tool that does not work in this session is
        refused by name. A path outside the declared folders is the files tools' own (guard asks the user)."""
        if s.group == "web" and not self.web_on():
            return WEB_OFF
        return "" if s.offered() else f"refused: {s.name} is not available in this session"

    @staticmethod
    def _budget(s: ToolSpec, args: dict, b: Budget) -> str:
        """A call made before in this question is not run again (a small model can loop on one); searches and new
        pages count against their limits when the user set one (another part of a page read is not a new page)."""
        if (s.name, args) in b.calls:
            return REPEATED
        b.calls.append((s.name, args))
        if s.budget == "search":
            if b.searches is not None and b.searched >= b.searches:
                return f"limit: at most {b.searches} web searches per question"
            b.searched += 1
        elif s.budget == "page":
            url = args.get("url")
            if url is None or url not in b.read_pages:
                if b.pages is not None and b.fetched >= b.pages:
                    return f"limit: at most {b.pages} pages per question"
                b.fetched += 1
                b.read_pages.add(url)
        return ""

    def _run(self, s: ToolSpec, kwargs: dict, budget: Budget | None):
        """The function's result; an error as text for the model, never a crash (a failed internal check is a bug:
        raised). A rate limit is waited out — as long as the service asks, else 1 then 2 s — over three attempts at
        most, never past the question's time or longer than LONGEST_WAIT; then said."""
        for attempt in range(len(RATE_WAITS) + 1):
            try:
                return s.run(**kwargs)
            except guard.Refused as e:
                return f"refused: {e}"
            except TypeError as e:
                return f"bad arguments for {s.name}: {e}"
            except AssertionError:
                raise
            except Exception as e:
                limited = _rate_limit(e)
                if limited is None:
                    return f"{s.name} failed: {type(e).__name__}: {str(e)[:200]}"
                service, asked = limited
                wait = asked if asked is not None else RATE_WAITS[min(attempt, len(RATE_WAITS) - 1)]
                late = budget is not None and budget.deadline is not None and self.clock() + wait > budget.deadline
                if attempt == len(RATE_WAITS) or wait > LONGEST_WAIT or late:
                    return f"rate limited by {service or s.name}: try again later or use another source"
                self.sleep(wait)


def _rate_limit(e: Exception) -> tuple[str, float | None] | None:
    """(the service, the seconds it asked to wait or None) when `e` is a rate limit: RateLimited, or any client's
    HTTP error carrying a 429 response (httpx, the SDKs); else None."""
    if isinstance(e, RateLimited):
        return e.service, e.retry_after
    response = getattr(e, "response", None)
    if getattr(response, "status_code", None) != 429:
        return None
    raw = (getattr(response, "headers", None) or {}).get("retry-after")
    try:
        asked = float(raw)
    except (TypeError, ValueError):
        asked = None                    # none, or an HTTP date: waited as if unsaid
    request = getattr(e, "request", None)
    host = getattr(getattr(request, "url", None), "host", "") or ""
    return host, asked
