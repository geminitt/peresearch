"""What peresearch may read, and what may leave the machine.

Reading: only folders the user declared (`peresearch add`), never through a symlink that points outside
them, and never a path that matches DENY (keys, tokens, credentials, .env files). Indexing drops any chunk
that looks like it holds a secret.

Leaving: everything sent off the machine (a prompt to the model, a search query, a URL) goes through
`outbound`, which redacts anything that looks like a secret and records the event. Deny rules at the tool
level alone are not enough (Claude Code's have been bypassed through other tools), so the last check sits
at the exit. `sanitize` strips terminal control sequences before anything untrusted is printed.
"""

import fnmatch
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

from peresearch import secret_rules



def private_folder(path: Path) -> Path:
    """`path` made (with its parents) or closed again so only the user may enter it (0700)."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def open_private(path: Path, flags: int) -> int:
    """A file descriptor on `path`, created if missing, readable by the user alone (0600) — a file found looser is
    closed again (step 3f, 2026-10-08)."""
    fd = os.open(path, flags | os.O_CREAT, 0o600)
    os.fchmod(fd, 0o600)
    return fd


def ensure_home() -> Path:
    """peresearch's own folder, made or closed again so only the user may enter it: it holds the index (the text of
    every note indexed), the sessions, the keys. Its files took the default mode (0644 in a 0755 folder on the owner's
    machine, 2026-10-07); closing the folder closes them all, whatever made them."""
    path = home()
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def home() -> Path:
    """Where peresearch keeps its config, index and logs (outside every declared folder)."""
    return Path(os.environ.get("PERESEARCH_HOME", Path.home() / ".local/share/peresearch"))

# Matched against every component of a path (directories and the file name).
DENY = [
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".kaggle", ".modal.toml", ".netrc",
    ".git-credentials", ".pypirc", ".npmrc", ".env", ".env.*", "*.env", "*.pem", "*.key", "*.p12", "*.pfx",
    "*.kdbx", "id_rsa*", "id_ecdsa*", "id_ed25519*", ".password-store", "hosts.yml",
    # credential files by name only: notes *about* tokens or secrets (tokenizer.md) stay readable
    "token", ".token", "*.token", "access_token", "token.json", "tokens.json", "credentials",
    "credentials.json", "credentials.toml", "credentials.yml", "credentials.yaml", ".secrets", "secrets.json",
    "secrets.toml", "secrets.yml", "secrets.yaml",
]
# A distinctive prefix (ghp_, sk-proj-, hf_…) is matched even right after a letter or digit: a secret glued to the
# text before it, or to another secret, is still found. Short or generic prefixes (sk-, ak-, wk-, rk_live_) keep a
# word boundary before them, so that ordinary words ending in them are not taken for secrets. No pattern needs a
# boundary after it (a secret followed by "_", "é" or another secret was missed when one was required).
SECRET_PATTERNS = {
    "private-key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "aws-access-key": r"AKIA[0-9A-Z]{16}",                        # always 20 characters
    "github-token": r"gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}",
    "huggingface-token": r"hf_[A-Za-z0-9]{30,}",
    "anthropic-key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    # the older form holds a digit (48 random characters: none in 2 of 10,000): "sk-toggleable__label-arrow", a CSS
    # class of scikit-learn, was taken for a key (2026-10-05)
    "openai-key": r"sk-proj-[A-Za-z0-9_\-]{20,}|\bsk-(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{20,}",
    "google-api-key": r"AIza[0-9A-Za-z_\-]{35}",                # always 39 characters
    "slack-token": r"xox[abprs]-[A-Za-z0-9\-]{10,}",
    "stripe-key": r"\b[rs]k_live_[0-9A-Za-z]{20,}",
    "modal-token": r"\ba[ks]-[A-Za-z0-9]{20,}",
    "modal-proxy-token": r"\bw[ks]-[A-Za-z0-9]{20,}",
    "tavily-key": r"tvly-[A-Za-z0-9\-]{20,}",
    "groq-key": r"gsk_[A-Za-z0-9]{40,}",
    "cerebras-key": r"csk-[A-Za-z0-9]{30,}",
    "jina-key": r"jina_[A-Za-z0-9_\-]{40,}",
    "jwt": r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
}
_SECRET_RE = [(kind, re.compile(p)) for kind, p in SECRET_PATTERNS.items()]
# key = "value" assignments whose value looks random enough to be a credential. The key may be quoted itself, as in
# JSON or a dict ("api_key": "…", {'token': …}), and the operator Go's := or Ruby's and PHP's => (all missed before
# 2026-10-05), and a string prefix (r'…', b"…", f"…") before the quote
_ASSIGN_RE = re.compile(r"(?i)(api[_\-]?key|secret|token|passw(?:or)?d|access[_\-]?key|auth)[\w\-]*[\"']?\s*"
                        r"(?::=|=>|[:=])\s*(?:[rbfu]{1,2}(?=[\"'`]))?[\"'`]?([A-Za-z0-9_\-/+=.]{16,})")
# the same where the value must be a string literal: quotes (and a string prefix such as r, b or f, or Go's and JS's
# backquotes) before it
_ASSIGN_QUOTED_RE = re.compile(r"(?i)(api[_\-]?key|secret|token|passw(?:or)?d|access[_\-]?key|auth)[\w\-]*[\"']?\s*"
                               r"(?::=|=>|[:=])\s*(?:[rbfu]{1,2})?[\"'`]([A-Za-z0-9_\-/+=.]{16,})")
# Programming languages, where a secret written into the code is a string literal: there the assigned-value rule
# wants quotes, as detect-secrets' KeywordDetector does for its quotes-required file types (Python, JavaScript, Java,
# Swift, …); unquoted, the value is a name or an expression (134 of 2,336,197 public paragraphs, 2026-10-05).
# Config files (.env, YAML, TOML, INI), shell scripts, notebooks and notes write values bare.
QUOTED = {".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".kt", ".scala", ".c",
          ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m", ".r", ".jl", ".lua", ".cu", ".dart"}
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


class Refused(Exception):
    """Raised when something must not be read or must not leave the machine."""


class ConfigError(ValueError):
    """config.json as the user left it cannot be used (they may edit it: files.py's exclude and include): said with
    where and what, never a traceback."""

    def __init__(self, what: str):
        super().__init__(f"config.json ({config_path()}) {what}. Fix it, or remove it to start again (it lists your "
                         "folders).")


# --- what may be read ------------------------------------------------------------------------------------

def config_path() -> Path:
    return home() / "config.json"


def config() -> dict:
    """config.json as written by the user and by peresearch (`roots`, and the patterns of files.py)."""
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"is not valid JSON (line {e.lineno}, column {e.colno}): {e.msg}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"must hold an object ({{…}}), not a {type(data).__name__}")
    return data


def roots() -> list[Path]:
    return [Path(r) for r in config().get("roots", [])]


def outermost(roots: list[Path]) -> list[Path]:
    """The declared folders without those inside another one, which would list their files twice."""
    real = [Path(os.path.realpath(r)) for r in roots]
    kept = [r for i, (r, rr) in enumerate(zip(roots, real)) if not any(o != rr and o in rr.parents for o in real)
            and rr not in real[:i]]
    inside = [Path(os.path.realpath(k)) for k in kept]
    assert len(set(inside)) == len(inside) and not any(a in b.parents for a in inside for b in inside), \
        "outermost: a kept folder is listed twice or lies inside another kept one"
    return kept


def set_roots(paths: list[Path]) -> None:
    home().mkdir(parents=True, exist_ok=True)
    c = config()                                     # the user's other settings (exclude, include) are kept
    c["roots"] = sorted({str(Path(r).resolve()) for r in paths})
    config_path().write_text(json.dumps(c, indent=1))


# DENY as one expression: matching each part against each pattern took 0.15 ms per path, 5 s of a 13k-file walk
_DENY_RE = re.compile("|".join(f"(?:{fnmatch.translate(pat)})" for pat in DENY))


def denied(path: Path) -> bool:
    return any(_DENY_RE.match(part.lower()) for part in Path(path).parts)


def allowed(path: Path, root_list: list[Path] | None = None) -> bool:
    """Inside a declared root after resolving symlinks, and not denied (neither as given nor as resolved)."""
    root_list = roots() if root_list is None else root_list
    real = Path(os.path.realpath(path))
    own = Path(os.path.realpath(home()))
    if real == own or own in real.parents:
        return False
    inside = any(real == r or r in real.parents for r in (Path(os.path.realpath(r)) for r in root_list))
    return inside and not denied(Path(path)) and not denied(real)


# --- secrets ---------------------------------------------------------------------------------------------

def _entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def quoted(path) -> bool:
    """Whether a file is a programming language's code, where an assigned secret is always a string literal."""
    return Path(path).suffix.lower() in QUOTED


def find_secrets(text: str, quoted: bool = False) -> list[tuple[str, int, int]]:
    """(kind, start, end) of everything that looks like a credential: our patterns (they also find a secret glued to
    other text, where gitleaks' rules want a boundary), gitleaks' rules (secret_rules), and assigned values — in a
    programming language's code (`quoted`) only string literals."""
    hits = []
    for kind, rx in _SECRET_RE:          # overlapping matches too: one secret's match may run into the next one
        m = rx.search(text)
        while m:
            hits.append((kind, m.start(), m.end()))
            m = rx.search(text, m.start() + 1)
    for m in (_ASSIGN_QUOTED_RE if quoted else _ASSIGN_RE).finditer(text):
        value = m.group(2)
        # random-looking: mixes letters and digits (identifiers such as tokenizer_config do not); not a call
        # ("auth = SyncX509WorkloadIdentityAuth(": 87 of 960,398 paragraphs of public code, 2026-10-05)
        called = text[m.end(2):m.end(2) + 1] == "("
        if _entropy(value) >= 3.5 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value) and not called:
            hits.append(("assigned-secret", m.start(2), m.end(2)))
    lowered = text.lower()                  # after ours: where both find a secret, its name is ours (a stable sort)
    for rule in secret_rules.rules():
        hits += [(f"gitleaks:{rule.id}", a, b) for a, b in rule.find(text, lowered)]
    return sorted(hits, key=lambda h: h[1])


SECRETS_VERSION = 3      # find_secrets' own logic (its floors, its order); the patterns are hashed below
#                          2: a value followed by "(" is a call, not a secret; 3: quoted keys, code wants quotes


def fingerprint() -> str:
    """What finds secrets: this code's version, our patterns, the assigned-value rule and gitleaks' pinned rules. The
    index records it; when it changes every file is judged again, or a secret the old rules missed would stay."""
    what = json.dumps([SECRETS_VERSION, SECRET_PATTERNS, _ASSIGN_RE.pattern, _ASSIGN_QUOTED_RE.pattern, secret_rules.SHA256,
                       sorted(secret_rules.SKIPPED)], sort_keys=True)
    return f"v{SECRETS_VERSION}-{hashlib.sha256(what.encode()).hexdigest()[:16]}"


def redact(text: str) -> tuple[str, int]:
    """Every found secret replaced, by the rules for any text (what leaves the machine is redacted so, whatever file
    it came from: the tools and `outbound` must agree, or a quote checked against a source would not match). Matches
    that overlap (two secrets written back to back: one pattern's match runs into the next secret) are redacted as
    one stretch, so no part of either is left behind."""
    hits = find_secrets(text)
    spans: list[list] = []
    for kind, a, b in hits:                       # sorted by start
        if spans and a < spans[-1][2]:
            spans[-1][2] = max(spans[-1][2], b)
        else:
            spans.append([kind, a, b])
    assert all(x[2] <= y[1] for x, y in zip(spans, spans[1:])), "redact: overlapping stretches left unmerged"
    out, last = [], 0
    for kind, a, b in spans:
        out += [text[last:a], f"[REDACTED:{kind}]"]
        last = b
    return "".join(out) + text[last:], len(hits)


# --- what may leave --------------------------------------------------------------------------------------

def audit(event: str, **fields) -> None:
    """Local audit log; never holds the redacted content itself."""
    private_folder(home())
    with os.fdopen(open_private(home() / "audit.jsonl", os.O_WRONLY | os.O_APPEND), "a") as f:
        f.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")


def outbound(text: str, channel: str) -> str:
    """The only way text leaves the machine: secrets are redacted and the crossing is logged."""
    clean, n = redact(text)
    assert not find_secrets(clean), f"outbound ({channel}): a secret is still there after redaction"
    audit("outbound", channel=channel, chars=len(clean), redacted=n)
    return clean


def outbound_query(query: str, channel: str) -> str:
    """Search queries and URLs are refused outright rather than sent with holes in them."""
    if find_secrets(query):
        audit("refused", channel=channel)
        raise Refused(f"{channel}: the text looks like it contains a secret")
    audit("outbound", channel=channel, chars=len(query), redacted=0)
    return query


def sanitize(text: str) -> str:
    """Printable text only: no ANSI escape sequences or control characters (newlines and tabs stay)."""
    return _CONTROL_RE.sub("", _ANSI_RE.sub("", text))
