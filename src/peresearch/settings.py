"""Where the model and the web search live, and their keys.

Each setting is read from the environment, else from `$PERESEARCH_HOME/settings.env` (`NAME=value` lines; keep
the file `chmod 600`). Its name matches the guard's deny list, so it is never indexed, and keys only ever travel
in the request headers of their own service — never inside a prompt or a search query.

    PERESEARCH_LLM_URL    the model's OpenAI-compatible base URL: a server the user runs (Ollama, llama.cpp, LM
                          Studio, vLLM… e.g. http://localhost:11434/v1) or a provider's API
    PERESEARCH_LLM_MODEL  the model's name as that server or provider calls it; unset: the server's only model
    PERESEARCH_LLM_KEY    the bearer token: a provider's API key, or the token the user's own server asks for
    PERESEARCH_LLM_CONTEXT  the model's context in tokens, when neither the server nor the model table says it: the
                          agent keeps each conversation within it
    PERESEARCH_LLM_TOOLS, _TEMPERATURE, _TOP_P, _THINKING  override the model's profile (peresearch/profiles.py)
    PERESEARCH_EMBED_URL, _MODEL, _KEY  the user's embedding endpoint (…/v1 serving /embeddings: Ollama, llama.cpp,
                          vLLM, LM Studio, a provider); unset: the search goes by words alone (zetokrag/embed.py)
    PERESEARCH_RERANK_URL, _MODEL, _KEY  the user's rerank endpoint (…/v1 serving /rerank: vLLM, llama.cpp, TEI, Jina,
                          Cohere); optional
    PERESEARCH_SEARCH_MCP  the web search: an MCP server the user adds, as a command (stdio, e.g. npx -y tavily-mcp)
                          or a URL (Streamable HTTP, e.g. https://mcp.exa.ai/mcp); unset: no web search (web.py)
    PERESEARCH_SEARCH_MCP_KEY  its key as NAME=value (the variable that server reads), put in its environment alone
    PERESEARCH_SEARCH_TOOL  the server's search tool, when the one named for a web search is not it
    JINA_API_KEY          optional: the Jina Reader (a third party) renders the pages the local reader cannot (a page
                          built in JavaScript); unset, no URL goes there
    OPENALEX_API_KEY      optional (free); adds OpenAlex to the scholarly search (Semantic Scholar and arXiv need none)
    SEMANTIC_SCHOLAR_API_KEY  free, given only to a school or work email (not gmail.com); without it Semantic Scholar
                          answers from a pool shared by everyone keyless, which refused (429) 3 of 3 on 2026-10-03
    PERESEARCH_RERANK_DEPTH  how many of the 30 search candidates the rerank endpoint reorders, 0–30 (default 30);
                          fewer is faster and finds less (zetokrag/search.py has the measured trade-off)
"""

import os
from pathlib import Path

from peresearch import guard

DEFAULTS: dict[str, str] = {}   # none for the model: a default name only fits one person's server
# What `peresearch setup` and the interface's /setup ask: (name, label, a key to hide)
FIELDS = [("PERESEARCH_LLM_URL", "model endpoint (an OpenAI-compatible …/v1), e.g. http://localhost:11434/v1 or your "
                                 "provider's", False),
          ("PERESEARCH_LLM_MODEL", "model name, as your server or provider calls it (e.g. qwen3:8b; empty: the server's only model)", False),
          ("PERESEARCH_LLM_KEY", "model key (your provider's API key; empty for a server on this machine)", True),
          ("PERESEARCH_EMBED_URL", "embeddings endpoint (an OpenAI-compatible …/v1 serving /embeddings; empty: the "
                                   "search goes by words alone)", False),
          ("PERESEARCH_EMBED_MODEL", "embedding model, as that server calls it (e.g. bge-m3; empty: the server's only "
                                     "model)", False),
          ("PERESEARCH_EMBED_KEY", "embeddings key (empty for a server on this machine)", True),
          ("PERESEARCH_RERANK_URL", "rerank endpoint (a …/v1 serving /rerank; optional, empty to skip)", False),
          ("PERESEARCH_RERANK_MODEL", "rerank model, as that server calls it (e.g. bge-reranker-v2-m3)", False),
          ("PERESEARCH_RERANK_KEY", "rerank key (empty for a server on this machine)", True),
          ("PERESEARCH_SEARCH_MCP", "web search: an MCP server, as a command (e.g. npx -y tavily-mcp) or a URL (e.g. "
                                    "https://mcp.exa.ai/mcp); empty: no web search", False),
          ("PERESEARCH_SEARCH_MCP_KEY", "its key, as NAME=value (the variable the server reads, e.g. "
                                        "TAVILY_API_KEY=tvly-…; empty if none)", True),
          ("OPENALEX_API_KEY", "OpenAlex key (free; adds a third scholarly source; empty to skip)", True),
          ("SEMANTIC_SCHOLAR_API_KEY", "Semantic Scholar key (free for a school or work email; without one it shares a "
                                       "public pool that often refuses)", True)]


# What /config shows and sets (the model and the keys are /setup's): (key, setting, default shown, what it does,
# its range: (low, high) for a whole number, "on/off", or "context")
# steps, web searches and pages: no limit unless the user sets one (owner, 2026-10-14, as OpenCode's optional `steps`)
CONFIG = [("steps", "PERESEARCH_STEPS", "no limit", "model calls per question", (1, 50)),
          ("web-searches", "PERESEARCH_WEB_SEARCHES", "no limit", "web, paper and Wikipedia searches per question",
           (0, 20)),
          ("pages", "PERESEARCH_PAGES", "no limit", "web pages read per question", (0, 30)),
          ("seconds", "PERESEARCH_SECONDS", "900", "time per question, a cold model start included", (30, 7200)),
          ("thinking", "PERESEARCH_LLM_THINKING", "the server's", "the model's reasoning: on or off", "on/off"),
          ("context", "PERESEARCH_LLM_CONTEXT", "the server's", "the model's context in tokens, when the server does "
                                                               "not say", "context"),
          ("rerank-depth", "PERESEARCH_RERANK_DEPTH", "30", "search candidates reordered by the rerank endpoint (fewer: "
                                                            "faster, finds less)", (0, 30))]


def row(key: str) -> tuple:
    """The CONFIG row of a /config key, or ValueError listing the keys."""
    found = next((c for c in CONFIG if c[0] == key), None)
    if found is None:
        raise ValueError(f"unknown setting {key}: {', '.join(c[0] for c in CONFIG)}")
    return found


def check(key: str, value: str) -> tuple[str, str]:
    """(the setting, the value as kept) for a /config key, or ValueError naming the key and its range."""
    _, name, _, _, kind = row(key)
    value = value.strip()
    if kind == "on/off":
        if value.lower() not in ("on", "off"):
            raise ValueError(f"{key} must be on or off (got {value!r})")
        return name, value.lower()
    if kind == "context":
        if not (value.isdigit() and int(value) >= 1024):
            raise ValueError(f"{key} must be a whole number of tokens, at least 1024 (got {value!r})")
        return name, value
    low, high = kind
    if not (value.isdigit() and low <= int(value) <= high):
        raise ValueError(f"{key} must be a whole number from {low} to {high} (got {value!r})")
    return name, str(int(value))


def saved() -> dict[str, str]:
    path = guard.home() / "settings.env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def get(name: str) -> str:
    return os.environ.get(name) or saved().get(name) or DEFAULTS.get(name, "")


def save(values: dict[str, str]) -> Path:
    """Write every setting to settings.env, readable by the user only: a file found readable by others is closed to
    them before the keys go in."""
    path = guard.home() / "settings.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in values.items()))
    os.chmod(path, 0o600)
    return path
