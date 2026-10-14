"""Private saved endpoints and model IDs for the TUI; active settings keep their existing contract."""

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from peresearch import guard, providers, settings

ROLES = {"chat": ("Chat model", "PERESEARCH_LLM"),
         "embedding": ("Embeddings", "PERESEARCH_EMBED"),
         "rerank": ("Reranker", "PERESEARCH_RERANK")}


@dataclass
class Connection:
    id: str
    role: str
    name: str
    url: str
    key: str = field(default="", repr=False)
    models: list[str] = field(default_factory=list)
    provider: str = ""                 # providers.PROVIDERS id; "" for any OpenAI-compatible endpoint


def validate_url(url: str) -> str:
    url = url.strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("The endpoint must be an http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("Enter credentials in the key field, not in the endpoint URL.")
    return url


def host(url: str) -> str:
    parsed = urlsplit(url)
    return guard.sanitize(parsed.hostname or "endpoint")


def load() -> list[Connection]:
    path = guard.home() / "models.json"
    entries = []
    if path.exists():
        if path.is_symlink():
            raise ValueError("The model catalog must be a regular file.")
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict) or raw.get("schema") != 1 or not isinstance(raw.get("connections"), list):
                raise ValueError("unsupported catalog structure")
            for item in raw["connections"]:
                entry = Connection(**item)
                if entry.role not in ROLES or not all(isinstance(getattr(entry, key), str)
                                                     for key in ("id", "name", "url", "key", "provider")):
                    raise ValueError("invalid connection")
                if not isinstance(entry.models, list) or not all(isinstance(model, str) for model in entry.models):
                    raise ValueError("invalid model list")
                validate_url(entry.url)
                entries.append(entry)
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError("The saved model catalog is invalid; correct models.json before changing connections.") from error
    return entries


def adopt_saved_selection() -> None:
    """Save once, as a connection, an endpoint the settings file selects but the catalog does not hold yet (set before
    the catalog existed, or by the command line). The catalog then lists every connection and settings.env says which
    one is in use. A value from the environment is never written here: it stays the environment's."""
    entries, saved, added = load(), settings.saved(), False
    for role, (_, prefix) in ROLES.items():
        url, model, key = (saved.get(prefix + suffix, "") for suffix in ("_URL", "_MODEL", "_KEY"))
        if not url or any(os.environ.get(prefix + suffix) for suffix in ("_URL", "_KEY")):
            continue
        try:
            url = validate_url(url)
        except ValueError:
            continue
        if not any(entry.role == role and entry.url == url and entry.key == key for entry in entries):
            chosen = saved.get("PERESEARCH_LLM_PROVIDER", "") if role == "chat" else ""
            entries.append(Connection(uuid.uuid4().hex, role, host(url), url, key, [model] if model else [],
                                      chosen or providers.guess(url)))
            added = True
    if added:
        save(entries)


def save(entries: list[Connection]) -> None:
    root = guard.private_folder(guard.home())
    fd, name = tempfile.mkstemp(prefix=".models-", suffix=".json", dir=root)
    temp = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema": 1, "connections": [asdict(entry) for entry in entries]}, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, root / "models.json")
    finally:
        temp.unlink(missing_ok=True)


def add(role: str, name: str, url: str, key: str, model: str = "", provider: str = "") -> Connection:
    if role not in ROLES:
        raise ValueError("Unknown model role.")
    url = validate_url(url)
    entry = Connection(uuid.uuid4().hex, role, name.strip() or host(url), url, key, [model.strip()] if model.strip() else [],
                       provider or providers.guess(url))
    entries = load()
    entries.append(entry)
    save(entries)
    return entry


def connect(role: str, name: str, url: str, key: str, provider: str = "") -> Connection:
    """The saved connection for this role, endpoint and key, or a new one: connecting again does not add a copy."""
    url = validate_url(url)
    found = next((entry for entry in load() if entry.role == role and entry.url == url and entry.key == key), None)
    return found or add(role, name, url, key, provider=provider)


def selection(entry: Connection, model: str) -> dict[str, str]:
    prefix = ROLES[entry.role][1]
    chosen = {prefix + "_URL": entry.url, prefix + "_MODEL": model, prefix + "_KEY": entry.key}
    if entry.role == "chat":                   # the gateway's route follows the provider (llm.LLM)
        chosen["PERESEARCH_LLM_PROVIDER"] = entry.provider or providers.guess(entry.url)
    return chosen


def discover(entry: Connection, client=None) -> list[str]:
    """List model IDs only, without submitting prompts, embeddings or reranking requests; through the provider's
    own SDK (Anthropic's Models API for Claude, the OpenAI-compatible /models for every other)."""
    owned = client is None
    if owned and providers.protocol(entry.provider or providers.guess(entry.url)) == "anthropic":
        import anthropic
        client = anthropic.Anthropic(base_url=entry.url.rstrip("/").removesuffix("/v1"), api_key=entry.key or None,
                                     timeout=15, max_retries=0)
    elif owned:
        from openai import OpenAI
        client = OpenAI(base_url=entry.url, api_key=entry.key or "none", timeout=15, max_retries=0)
    try:
        return sorted({item.id for item in client.models.list().data if isinstance(item.id, str) and item.id})
    finally:
        if owned:
            client.close()


def update_models(connection_id: str, models: list[str], expected: Connection) -> None:
    entries = load()
    entry = next((entry for entry in entries if entry.id == connection_id), None)
    if entry is None or (entry.url, entry.key, entry.role) != (expected.url, expected.key, expected.role):
        raise ValueError("The connection changed; discover its models again.")
    entry.models = list(models)
    save(entries)
