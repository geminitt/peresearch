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
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path



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
# Not secret, just not worth reading: tooling, environments, and generated data rather than notes.
SKIP_DIRS = {".git", ".hg", ".svn", ".pixi", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ipynb_checkpoints", ".cache", ".local", ".Trash", "site-packages", "third_party",
             "runs", "data", "datasets", "checkpoints", "wandb", "outputs", "build", "dist", ".hadoop-lib"}

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
    "openai-key": r"sk-proj-[A-Za-z0-9_\-]{20,}|\bsk-[A-Za-z0-9_\-]{20,}",
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
# key = "value" assignments whose value looks random enough to be a credential
_ASSIGN_RE = re.compile(r"(?i)(api[_\-]?key|secret|token|passw(?:or)?d|access[_\-]?key|auth)[\w\-]*\s*[:=]\s*"
                        r"[\"']?([A-Za-z0-9_\-/+=.]{16,})")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


class Refused(Exception):
    """Raised when something must not be read or must not leave the machine."""


# --- what may be read ------------------------------------------------------------------------------------

def config_path() -> Path:
    return home() / "config.json"


def roots() -> list[Path]:
    p = config_path()
    return [Path(r) for r in json.loads(p.read_text())["roots"]] if p.exists() else []


def outermost(roots: list[Path]) -> list[Path]:
    """The declared folders without those inside another one, which would list their files twice."""
    real = [Path(os.path.realpath(r)) for r in roots]
    return [r for i, (r, rr) in enumerate(zip(roots, real)) if not any(o != rr and o in rr.parents for o in real)
            and rr not in real[:i]]


def set_roots(paths: list[Path]) -> None:
    home().mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps({"roots": sorted({str(Path(r).resolve()) for r in paths})}, indent=1))


def denied(path: Path) -> bool:
    return any(fnmatch.fnmatch(part.lower(), pat) for part in Path(path).parts for pat in DENY)


def allowed(path: Path, root_list: list[Path] | None = None) -> bool:
    """Inside a declared root after resolving symlinks, and not denied (neither as given nor as resolved)."""
    root_list = roots() if root_list is None else root_list
    real = Path(os.path.realpath(path))
    inside = any(real == r or r in real.parents for r in (Path(os.path.realpath(r)) for r in root_list))
    return inside and not denied(Path(path)) and not denied(real)


# --- secrets ---------------------------------------------------------------------------------------------

def _entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def find_secrets(text: str) -> list[tuple[str, int, int]]:
    """(kind, start, end) of everything that looks like a credential."""
    hits = []
    for kind, rx in _SECRET_RE:          # overlapping matches too: one secret's match may run into the next one
        m = rx.search(text)
        while m:
            hits.append((kind, m.start(), m.end()))
            m = rx.search(text, m.start() + 1)
    for m in _ASSIGN_RE.finditer(text):
        value = m.group(2)
        # random-looking: mixes letters and digits (identifiers such as tokenizer_config do not)
        if _entropy(value) >= 3.5 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value):
            hits.append(("assigned-secret", m.start(2), m.end(2)))
    return sorted(hits, key=lambda h: h[1])


def redact(text: str) -> tuple[str, int]:
    """Every found secret replaced. Matches that overlap (two secrets written back to back: one pattern's match runs
    into the next secret) are redacted as one stretch, so no part of either is left behind."""
    hits = find_secrets(text)
    spans: list[list] = []
    for kind, a, b in hits:                       # sorted by start
        if spans and a < spans[-1][2]:
            spans[-1][2] = max(spans[-1][2], b)
        else:
            spans.append([kind, a, b])
    out, last = [], 0
    for kind, a, b in spans:
        out += [text[last:a], f"[REDACTED:{kind}]"]
        last = b
    return "".join(out) + text[last:], len(hits)


# --- what may leave --------------------------------------------------------------------------------------

def audit(event: str, **fields) -> None:
    """Local audit log; never holds the redacted content itself."""
    home().mkdir(parents=True, exist_ok=True)
    with open(home() / "audit.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")


def outbound(text: str, channel: str) -> str:
    """The only way text leaves the machine: secrets are redacted and the crossing is logged."""
    clean, n = redact(text)
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
