"""Immutable excerpt identities and links to conversation tool results."""
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

UNKNOWN = "Evidence identity is unknown. Read this source again before citing it."
_DECIMAL_DIGITS = sys.int_info.str_digits_check_threshold
_DECIMAL_BASE = 10 ** _DECIMAL_DIGITS


def _number(digits):
    """Parse source counters in chunks without changing Python's global integer-string guard."""
    number = 0
    for start in range(0, len(digits), _DECIMAL_DIGITS):
        part = digits[start:start + _DECIMAL_DIGITS]
        number = number * 10 ** len(part) + int(part)
    return number


def _decimal(number):
    parts = []
    while number >= _DECIMAL_BASE:
        number, part = divmod(number, _DECIMAL_BASE)
        parts.append(str(part).zfill(_DECIMAL_DIGITS))
    return str(number) + "".join(reversed(parts))


def cited_ids(text):
    return {identifier.strip() for match in re.finditer(r"\[([NW]\d+(?:\s*,\s*[NW]\d+)*)\]", text)
            for identifier in match[1].split(",")}


@dataclass(frozen=True)
class Source:
    id: str
    kind: str
    where: str
    title: str
    text: str
    known: bool = True

    @property
    def version(self) -> str:
        data = json.dumps([self.kind, self.where, self.title, self.text], ensure_ascii=False).encode()
        return hashlib.sha256(data).hexdigest()


def unknown(identifier, where="") -> Source:
    return Source(identifier, "file" if identifier.startswith("N") else "web", where,
                  "Evidence identity unknown; read again", UNKNOWN, False)


def link(source: Source, ref: str, result: str) -> dict | None:
    """Store character ranges into the result, including whitespace transformations in catalog/search output."""
    position = result.find(source.text)
    if position >= 0:
        parts = [[position, position + len(source.text)]]
    else:
        parts, position = [], 0
        for token in re.findall(r"\s+|\S+", source.text):
            if token.isspace():
                parts.append(token)
            else:
                start = result.find(token, position)
                if start < 0:
                    return None
                parts.append([start, start + len(token)])
                position = start + len(token)
    return {"id": source.id, "kind": source.kind, "where": source.where, "title": source.title,
            "version": source.version, "ref": ref, "parts": parts}


class Lookup:
    """A store's results by ref, read one at a time (ResultStore.get reads one line) instead of all of them."""

    def __init__(self, store):
        self.store = store

    def __getitem__(self, ref):
        item = self.store.get(ref) if isinstance(ref, str) else None
        if item is None:
            raise KeyError(ref)
        return item


def restore(record: dict, results: dict, allowed=None) -> Source | None:
    """Verify the backing result and excerpt digest; malformed or unavailable evidence stays unknown."""
    try:
        identifier = record["id"]
        if not re.fullmatch(r"[NW][1-9]\d*", identifier):
            return None
        kind = record["kind"]
        if kind != ("file" if identifier.startswith("N") else "web"):
            return None
        result = results[record["ref"]]
        if kind == "file" and allowed is not None:
            scope = result.get("scope")
            if not isinstance(scope, list) or not all(isinstance(p, str) and allowed(p) for p in scope):
                return None
        text, parts = result["text"], []
        for part in record["parts"]:
            if isinstance(part, str) and part.isspace():
                parts.append(part)
            elif isinstance(part, list) and len(part) == 2 and all(type(n) is int for n in part) \
                    and 0 <= part[0] <= part[1] <= len(text):
                parts.append(text[part[0]:part[1]])
            else:
                return None
        if not isinstance(record["where"], str) or not isinstance(record["title"], str):
            return None
        source = Source(identifier, kind, record["where"], record["title"], "".join(parts))
        return source if source.version == record["version"] else None
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class Sources:
    """Conversation IDs identify immutable excerpts; current tracks this question's tool evidence."""
    items: dict[str, Source] = field(default_factory=dict)
    urls: dict[str, str] = field(default_factory=dict)
    current: set[str] = field(default_factory=set, repr=False, compare=False)
    links: dict[str, dict] = field(default_factory=dict, repr=False, compare=False)
    _at: dict = field(default_factory=dict, repr=False, compare=False)
    _kinds: Counter = field(default_factory=Counter, repr=False, compare=False)

    def begin_question(self):
        self.current.clear()
        self.urls.clear()

    def _register(self, source):
        self.items[source.id] = source
        match = re.fullmatch(r"([NW])([1-9]\d*)", source.id)
        if match:
            kind = "file" if match[1] == "N" else "web"
            self._kinds[kind] = max(self._kinds[kind], _number(match[2]))
        if source.known:
            self._at[(source.kind, source.where, source.title, source.text)] = source

    def add(self, kind: str, where: str, title: str, text: str) -> Source:
        key = (kind, where, title, text)
        source = self._at.get(key)
        if source is None:
            self._kinds[kind] += 1
            identifier = f"{'N' if kind == 'file' else 'W'}{_decimal(self._kinds[kind])}"
            assert kind in ("file", "web") and identifier not in self.items
            source = Source(identifier, kind, where, title, text)
            self._register(source)
        self.current.add(source.id)
        if kind == "web":
            self.urls[where] = source.id
        return source

    def retain(self, ref, text, store):
        records = []
        for identifier in sorted(self.current - self.links.keys()):
            record = link(self.items[identifier], ref, text)
            if record is not None:
                self.links[identifier] = record
                records.append(record)
        if records:
            store.set_sources(records)

    @classmethod
    def load(cls, turns, store, allowed=None):
        registry = cls()
        results = {item["ref"]: item for item in store._lines()}
        records, conflicts = {}, set()
        for record in store.source_versions():
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                continue
            identifier = record["id"]
            if identifier in records and record != records[identifier]:
                conflicts.add(identifier)
            records[identifier] = record
        for turn in turns:
            locations = turn.get("sources")
            locations = locations if isinstance(locations, dict) else {}
            versions = turn.get("source_versions")
            versions = versions if isinstance(versions, dict) else {}
            for identifier in cited_ids(turn.get("answer", "")) | versions.keys() | locations.keys():
                if isinstance(identifier, str) and re.fullmatch(r"[NW][1-9]\d*", identifier):
                    where = locations.get(identifier)
                    registry._register(unknown(identifier, where if isinstance(where, str) else ""))
                    record = records.get(identifier)
                    if record is not None and (versions.get(identifier) != record.get("version")
                                               or where != record.get("where")):
                        conflicts.add(identifier)
        for identifier, record in records.items():
            source = restore(record, results, allowed) if identifier not in conflicts else None
            if source is not None:
                registry._register(source)
                registry.links[identifier] = record
            elif re.fullmatch(r"[NW][1-9]\d*", identifier):
                registry._register(unknown(identifier))
        return registry

    def for_turn(self, turn):
        locations = turn.get("sources")
        locations = dict(locations) if isinstance(locations, dict) else {}
        for identifier in cited_ids(turn.get("answer", "")):
            locations.setdefault(identifier, "")
        versions = turn.get("source_versions")
        versions = versions if isinstance(versions, dict) else {}
        return {identifier: source if source.known and versions.get(identifier) == source.version
                and source.where == where else unknown(identifier, str(where))
                for identifier, where in locations.items()
                if isinstance(identifier, str)
                for source in [self.items.get(identifier, unknown(identifier, str(where))) ]}

    def selected(self, text, store, allowed=None):
        cited = cited_ids(text)
        for identifier in sorted(cited - self.items.keys()):
            if re.fullmatch(r"[NW][1-9]\d*", identifier):
                self._register(unknown(identifier))
        selected = self.current | cited
        results = Lookup(store) if allowed is not None else {}   # the refs it needs, one line each
        out = {}
        for identifier, source in self.items.items():
            if identifier not in selected:
                continue
            record = self.links.get(identifier)
            if source.kind == "file" and allowed is not None and record is not None \
                    and restore(record, results, allowed) is None:
                source = unknown(identifier, source.where)
            out[identifier] = source
        return out
