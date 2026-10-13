"""Secret formats from gitleaks' default rules (v8.30.1, MIT; data/gitleaks.toml, its license beside it): the
prefixes and shapes of the keys of some two hundred providers, kept up to date by its community, where a list of
our own covered sixteen.

The rules are Go regular expressions; `python_pattern` gives each the same meaning in Python (22 set a flag mid-
pattern, which Python refuses; one uses a POSIX class, which Python would read as a set of characters without a
word; four end with \\z). A rule applies only when one of its keywords is in the text (as in gitleaks), keeps a
finding only above its entropy floor and off its allowlists and the global one; conditions on a file's path never
hold for text. generic-api-key (a key name and any value) is left out: guard's assigned-secret check covers it with
an entropy floor of its own.
"""

import hashlib
import math
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

RULES_FILE = Path(__file__).parent / "data" / "gitleaks.toml"
SHA256 = "e163e53b9e7e8a8511e77271e2b323ed057759542a6d988258afe3a1fa329caf"       # v8.30.1 as fetched
SKIPPED = {"generic-api-key"}
POSIX = {"alnum": "a-zA-Z0-9", "alpha": "a-zA-Z", "digit": "0-9", "lower": "a-z", "upper": "A-Z",
         "space": r" \t\n\r\f\v", "xdigit": "0-9A-Fa-f", "word": r"\w", "blank": r" \t", "punct": r"!-/:-@\[-`{-~",
         "cntrl": r"\x00-\x1f\x7f", "graph": "!-~", "print": " -~"}
_POSIX = re.compile(r"\[:(\w+):\]")
_FLAGS = re.compile(r"\(\?([a-zA-Z]+)\)")


def python_pattern(go: str) -> str:
    """A Go (RE2) regular expression as Python reads it to the same effect."""
    out, groups, i, n = [], [0], 0, len(go)        # groups: the scoped flags each open group must close
    while i < n:
        c = go[i]
        if c == "\\":
            out.append(r"\Z" if go[i:i + 2] == r"\z" else go[i:i + 2])
            i += 2
        elif c == "[":                              # a class, copied with its POSIX names spelled out
            j, head = i + 1, "["
            if j < n and go[j] == "^":
                head, j = head + "^", j + 1
            if j < n and go[j] == "]":
                head, j = head + r"\]", j + 1
            body = []
            while j < n and go[j] != "]":
                m = _POSIX.match(go, j)
                if go[j] == "\\":
                    body.append(go[j:j + 2])
                    j += 2
                elif m:
                    body.append(POSIX[m.group(1)])
                    j = m.end()
                else:
                    body.append(r"\[" if go[j] == "[" else go[j])     # a literal "[" (Python warns of nested sets)
                    j += 1
            out.append(head + "".join(body) + "]")
            i = j + 1
        elif _FLAGS.match(go, i):                   # (?i) applies to the rest of its group in Go
            m = _FLAGS.match(go, i)
            if "U" in m.group(1):
                raise ValueError(f"ungreedy flag (?U) has no Python equivalent: {go}")
            if i == 0:
                out.append(m.group(0))
            else:
                out.append(f"(?{m.group(1)}:")
                groups[-1] += 1
            i = m.end()
        elif c == "(":
            groups.append(0)
            out.append(c)
            i += 1
        elif c == ")":
            out.append(")" * groups.pop() + ")")
            i += 1
        else:
            out.append(c)
            i += 1
    assert len(groups) == 1, f"unbalanced groups in {go}"
    return "".join(out) + ")" * groups[0]


def entropy(s: str) -> float:
    """Shannon entropy in bits a character, as gitleaks measures a secret."""
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


@dataclass
class Allow:
    regexes: list = field(default_factory=list)
    target: str = "secret"                          # what the regexes look at: the secret, the match, its line
    stopwords: list = field(default_factory=list)   # a secret holding one is not a secret
    condition: str = "OR"
    paths: bool = False                             # a condition on the file's path: never true for text

    def drops(self, secret: str, match: str, line: str) -> bool:
        seen = {"secret": secret, "match": match, "line": line}[self.target]
        checks = ([any(r.search(seen) for r in self.regexes)] if self.regexes else []) + \
                 ([any(w in secret.lower() for w in self.stopwords)] if self.stopwords else []) + \
                 ([False] if self.paths else [])
        return bool(checks) and (all(checks) if self.condition == "AND" else any(checks))


@dataclass
class Rule:
    id: str
    pattern: str                                    # in Python's terms
    keywords: list
    entropy: float | None
    group: int | None                               # the group holding the secret (else the first that matched)
    allows: list

    def __post_init__(self):
        self.rx = re.compile(self.pattern)
        self.keywords = [k.lower() for k in self.keywords]

    def find(self, text: str, lowered: str | None = None) -> list[tuple[int, int]]:
        """(start, end) of each secret this rule finds in `text`."""
        lowered = text.lower() if lowered is None else lowered
        if self.keywords and not any(k in lowered for k in self.keywords):
            return []
        out = []
        for m in self.rx.finditer(text):
            g = self.group or next((k for k in range(1, (m.lastindex or 0) + 1) if m.group(k)), 0)
            a, b = m.span(g)
            secret = text[a:b]
            if not secret or (self.entropy and entropy(secret) <= self.entropy):     # skipped at the floor too, as gitleaks does
                continue
            line = text[text.rfind("\n", 0, m.start()) + 1:(text.find("\n", m.end()) + 1 or len(text) + 1) - 1]
            if any(al.drops(secret, m.group(0), line) for al in self.allows + [_global()]):
                continue
            out.append((a, b))
        return out


def _allows(entries: list[dict]) -> list[Allow]:
    return [Allow([re.compile(python_pattern(r)) for r in e.get("regexes", [])], e.get("regexTarget", "secret"),
                  [w.lower() for w in e.get("stopwords", [])], e.get("condition", "OR"), bool(e.get("paths")))
            for e in entries]


@cache
def _config() -> dict:
    raw = RULES_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:  # a changed rule file changes what is found: never silently
        raise RuntimeError(f"{RULES_FILE} is not gitleaks v8.30.1 as fetched (its sha256 differs)")
    return tomllib.loads(raw.decode("utf-8"))


@cache
def _global() -> Allow:
    return _allows([_config().get("allowlist", {})])[0]


@cache
def rules() -> tuple[Rule, ...]:
    """Every rule with a pattern, but SKIPPED."""
    return tuple(Rule(r["id"], python_pattern(r["regex"]), r.get("keywords", []), r.get("entropy"),
                      r.get("secretGroup"), _allows(r.get("allowlists", [])))
                 for r in _config()["rules"] if "regex" in r and r["id"] not in SKIPPED)
