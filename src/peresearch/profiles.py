"""What the model behind the gateway can do and how it wants to be asked: its profile.

The gateway is one OpenAI-compatible API, but the models behind it differ, and adapting to whichever model is behind
it is peresearch's job (the owner, 2026-10-06). A profile is built in layers, each over the one before, the way
LiteLLM, models.dev, Continue and Aider do:

1. defaults: native tool calls, nothing forced (the server's own sampling);
2. the capability table: a pinned, trimmed snapshot of models.dev (MIT; data/model_capabilities.json, its license
   beside it) looked up by the model's normalized name: tool calls, reasoning, temperature accepted, the context and
   output limits, the family;
3. the family table, for what models.dev does not say: how a family switches its reasoning, and the sampling its
   authors recommend (forced only where they publish one);
4. the user's settings: PERESEARCH_LLM_TOOLS (native | prompt), PERESEARCH_LLM_TEMPERATURE, PERESEARCH_LLM_TOP_P
   (PERESEARCH_LLM_CONTEXT and PERESEARCH_LLM_THINKING are read by llm.LLM itself).

`source` says where each field came from, for /status. Rebuild the table from a fresh download:
    pixi run python -m peresearch.profiles rebuild [api.json]
"""

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from peresearch import settings

TABLE = Path(__file__).parent / "data" / "model_capabilities.json"
SOURCE = "https://models.dev/api.json"

# Authors' published recommendations and switches, by family (models.dev family names). Qwen3 (and 3.5/3.6):
# temperature 0.6 and top_p 0.95 for thinking mode, reasoning switched by the chat template's enable_thinking.
FAMILIES = {"qwen": {"thinking": "template", "temperature": 0.6, "top_p": 0.95}}
COMPLETION_TOKEN_FAMILIES = frozenset({"o", "o-mini", "o-pro", "gpt", "gpt-mini", "gpt-nano", "gpt-pro",
                                     "gpt-codex", "gpt-codex-spark", "gpt-astra", "gpt-luna", "gpt-sol", "gpt-terra"})

# Endings a served name adds to a model's name: a tag, a quantization, an instruction-tuned variant
_ENDINGS = re.compile(r"-(latest|instruct|it|chat|fp8|fp16|bf16|int8|int4|awq|gptq|gguf|mlx|q\d[a-z0-9-]*)$")


def norm(name: str) -> str:
    """A model's name as the table keys it: lowercase, without its organization (org/name), with every run of other
    characters (":", ".", "_", spaces) as one "-". Ollama's "qwen3:8b" and OpenRouter's "qwen/qwen3-8b" meet."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower().rsplit("/", 1)[-1]).strip("-")


@dataclass(frozen=True)
class Profile:
    model: str = ""
    family: str = ""
    tools: str = "native"                 # native: the API's tool calls; prompt: described in the system prompt
    reasoning: bool = False               # models.dev's word, kept but not relied on: it says no of Qwen3.5-0.8B,
                                          # which reasons (2026-10-06)
    thinking: str = ""                    # how its reasoning is switched: "template" (enable_thinking) or "" (not)
    temperature: float | None = None      # None: the server's own sampling
    top_p: float | None = None
    context: int | None = None
    output: int | None = None
    source: dict = field(default_factory=dict)    # field -> "table" | "family" | "setting"

    @property
    def token_field(self) -> str:
        """Chat Completions capability; local/unknown profiles retain the legacy field.

        OpenAI's max_completion_tokens includes reasoning; max_tokens is incompatible with o-series.
        https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        gpt-oss is a local model family, not the hosted GPT API contract.
        """
        return "max_completion_tokens" if self.family in COMPLETION_TOKEN_FAMILIES else "max_tokens"


def build(raw: dict) -> dict:
    """The trimmed table from models.dev's api.json: per normalized name, [family, tool_call, reasoning,
    temperature accepted, context, output]. One name served by several providers: tool calls and reasoning when any
    provider says so (a provider may lack the parser, not the model), temperature refused only when all refuse it,
    the largest limits (the model's own; a server's smaller one is learned at run time), the most common family."""
    seen: dict[str, list[dict]] = {}
    for provider in raw.values():
        for name, m in (provider.get("models") or {}).items():
            seen.setdefault(norm(name), []).append(m)
    table = {}
    for key, entries in sorted(seen.items()):
        if not key:
            continue
        limits = [m.get("limit") or {} for m in entries]
        families = Counter(m.get("family") or "" for m in entries).most_common()
        table[key] = [families[0][0], any(bool(m.get("tool_call")) for m in entries),       # one entry at least
                      any(bool(m.get("reasoning")) for m in entries),
                      any(m.get("temperature", True) is not False for m in entries),
                      max(x.get("context") or 0 for x in limits) or None,
                      max(x.get("output") or 0 for x in limits) or None]
    return table


_loaded: dict | None = None


def table() -> dict:
    global _loaded
    if _loaded is None:
        _loaded = json.loads(TABLE.read_text())["models"]
    return _loaded


def _bare(key: str) -> str:
    """A key without its endings."""
    while True:
        shorter = _ENDINGS.sub("", key)
        if shorter == key:
            return key
        key = shorter


_compact: dict | None = None


def _compact_index(rows: dict) -> dict:
    """Keys without endings and without dashes → the shortest table key that gives them: Ollama writes "llama3.1"
    and "gemma3" where models.dev has "llama-3-1" and "gemma-3"."""
    index: dict[str, str] = {}
    for key in sorted(rows, key=lambda k: (len(k), k)):
        index.setdefault(_bare(key).replace("-", ""), key)
    return index


CHAT_ENDINGS = ("-instruct", "-it", "-chat")


def _chat(key: str, rows: dict) -> tuple[str, list]:
    """The chat-tuned row beside a base one: a served agent model is chat-tuned, and Ollama names its chat models
    without "-instruct" ("llama3.2:1b", whose base row in models.dev says no tool calls; it makes them, 2026-10-06)."""
    key = next((key + e for e in CHAT_ENDINGS if key + e in rows), key)
    return key, rows[key]


def lookup(name: str, rows: dict | None = None) -> tuple[str, list] | None:
    """(the key found, its row) for a served model name: the name, then without its endings one by one
    ("qwen3-8b-instruct-fp8" → "qwen3-8b-instruct" → "qwen3-8b"), then the same without dashes ("llama3-1-8b" meets
    "llama-3-1-8b"), a chat-tuned row beside the one found first; None when the table does not know it."""
    global _compact
    known = rows is None
    rows = table() if rows is None else rows
    key = norm(name)
    while key:
        if key in rows:
            return _chat(key, rows)
        shorter = _ENDINGS.sub("", key)
        if shorter == key:
            break
        key = shorter
    if not key:
        return None
    if known:
        _compact = _compact if _compact is not None else _compact_index(rows)
        index = _compact
    else:
        index = _compact_index(rows)
    hit = index.get(key.replace("-", ""))
    return _chat(hit, rows) if hit else None


def _number(name: str, low: float, high: float) -> float | None:
    raw = settings.get(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not (low <= value <= high):
        raise ValueError(f"{name} must be a number from {low:g} to {high:g} (got {raw!r})")
    return value


def profile(model: str, rows: dict | None = None) -> Profile:
    """The profile of `model` (the served name, or vLLM's `root` when the served name is an alias), in layers."""
    p, source = Profile(model=model), {}
    found = lookup(model, rows)
    if found:
        _, (family, tool_call, reasoning, temperature, context, output) = found
        p = replace(p, family=family, tools="native" if tool_call else "prompt", reasoning=reasoning,
                    context=context, output=output)
        source.update(dict.fromkeys(("family", "tools", "reasoning", "context", "output"), "table"))
        if not temperature:
            source["temperature"] = "table"          # refused by the model: never sent, whatever the family says
    fam = FAMILIES.get(p.family, {})
    for k, v in fam.items():
        if k in ("temperature", "top_p") and source.get("temperature") == "table":
            continue
        p = replace(p, **{k: v})
        source[k] = "family"
    tools = settings.get("PERESEARCH_LLM_TOOLS")
    if tools:
        if tools not in ("native", "prompt"):
            raise ValueError(f"PERESEARCH_LLM_TOOLS must be native or prompt, or unset to follow the model (got {tools!r})")
        p, source["tools"] = replace(p, tools=tools), "setting"
    for name, attr, high in (("PERESEARCH_LLM_TEMPERATURE", "temperature", 2.0), ("PERESEARCH_LLM_TOP_P", "top_p", 1.0)):
        value = _number(name, 0.0, high)
        if value is not None:
            p, source[attr] = replace(p, **{attr: value}), "setting"
    return replace(p, source=source)


def rebuild(path: str | None = None) -> Path:
    """Fetch (or read) models.dev's api.json, trim it, and write the table with where it came from."""
    if path:
        data = Path(path).read_bytes()
    else:
        import httpx
        data = httpx.get(SOURCE, timeout=60).content
    import datetime
    out = {"source": SOURCE, "license": "MIT (data/MODELS-DEV-LICENSE)",
           "fetched": datetime.date.today().isoformat(), "sha256": hashlib.sha256(data).hexdigest(),
           "fields": ["family", "tool_call", "reasoning", "temperature", "context", "output"],
           "models": build(json.loads(data))}
    TABLE.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
    return TABLE


if __name__ == "__main__":
    if sys.argv[1:2] == ["rebuild"]:
        print(rebuild(sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        sys.exit("usage: python -m peresearch.profiles rebuild [api.json]")
